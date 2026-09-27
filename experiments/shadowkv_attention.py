# Phase 10-11: the full ShadowKV attention path on CPU
# Pre-RoPE keys stored low-rank, landmarks + outliers pick chunks, only the selected keys are
# rebuilt (then RoPE), outlier chunks use exact keys, values are exact.
# Compared with full attention, plus two partial versions to see where error comes from:
#   lowrank_dense: low-rank keys only (all tokens)
#   sparse_exact:  chunk selection only (exact keys)
#   shadowkv:      both together
# Usage: python experiments\shadowkv_attention.py [model_name]

import json
import math
import platform
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn.functional as F
import transformers
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from shadowkv.low_rank import LowRankKeys
from shadowkv.landmarks import build_landmarks, outlier_mask
from shadowkv.selector import select_chunks
from shadowkv.reconstruction import chunk_to_token_idx, reconstruct_selected, sparse_attention
from models.toy_attention import rope_cos_sin, apply_rope

MODEL = sys.argv[1] if len(sys.argv) > 1 else "TinyLlama/TinyLlama_v1.1"
TAG = MODEL.split("/")[-1]
T = 4096
M = 64
N = T - M
CHUNK = 8
OUTLIER_FRAC = 0.01
BUDGETS = [128, 256, 512]
RANK_FRACS = [0.125, 0.25, 0.5]   # rank as a fraction of key dim per layer
N_PROMPTS = 2

# Load model
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).eval()
cfg = model.config
layers = model.model.layers
n_layers = len(layers)
n_heads = cfg.num_attention_heads
n_kv = cfg.num_key_value_heads
head_dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // n_heads
group = n_heads // n_kv
theta = getattr(cfg, "rope_theta", None) or cfg.rope_parameters["rope_theta"]

text = "\n".join(load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")["text"])
ids = tok(text, return_tensors="pt").input_ids[0]
prompts = [ids[p * 60000: p * 60000 + T] for p in range(N_PROMPTS)]

# Capture pre-RoPE queries (last M) and keys
q_pre, k_pre = {}, {}

def hook_q(i):
    def h(m, inp, out):
        q_pre[i] = out[0, -M:].detach()
    return h

def hook_k(i):
    def h(m, inp, out):
        k_pre[i] = out[0].detach()
    return h

for i, layer in enumerate(layers):
    layer.self_attn.q_proj.register_forward_hook(hook_q(i))
    layer.self_attn.k_proj.register_forward_hook(hook_k(i))


def cache_kv(cache, i):
    if hasattr(cache, "layers"):
        return cache.layers[i].keys[0], cache.layers[i].values[0]
    if hasattr(cache, "key_cache"):
        return cache.key_cache[i][0], cache.value_cache[i][0]
    return cache[i][0][0], cache[i][1][0]


def dense_attention(q, k, v):
    k = k.repeat_interleave(group, 0)
    v = v.repeat_interleave(group, 0)
    return torch.softmax(q @ k.transpose(-1, -2) / math.sqrt(head_dim), -1) @ v


def metrics(o, ref):
    return {"out_relerr": ((o - ref).norm() / ref.norm()).item(),
            "out_cos": F.cosine_similarity(o, ref, dim=-1).mean().item()}


rope_cos, rope_sin = rope_cos_sin(torch.arange(T), head_dim, base=theta)
kv_idx = torch.arange(n_kv)[:, None, None]
rows = []

for p_i, prompt in enumerate(prompts):
    with torch.no_grad():
        out = model.model(prompt[None], use_cache=True)

        for i in range(n_layers):
            k_post, v = cache_kv(out.past_key_values, i)
            k_post, v = k_post[:, :N], v[:, :N]                # [kv_heads, N, d]
            K_pre = k_pre[i][:N]                               # [N, D]
            D = K_pre.shape[1]
            q = q_pre[i].reshape(M, n_heads, head_dim).transpose(0, 1)
            q = apply_rope(q, rope_cos[-M:], rope_sin[-M:])   # [heads, M, d]

            ref = dense_attention(q, k_post, v)

            # prefill-time structures
            landmarks, min_cos, _ = build_landmarks(k_post, CHUNK)
            out_mask = outlier_mask(min_cos, OUTLIER_FRAC)
            n_out = int(out_mask[0].sum())
            factors = torch.linalg.svd(K_pre.float(), full_matrices=False)
            ranks = [int(D * f) for f in RANK_FRACS]
            lrs = {r: LowRankKeys.from_keys(K_pre, r, factors) for r in ranks}
            base = {"prompt": p_i, "layer": i, "dim": D}

            # low-rank keys only, all tokens (values stay on GPU)
            for r in ranks:
                k_hat = lrs[r].reconstruct().reshape(N, n_kv, head_dim).transpose(0, 1)
                k_hat = apply_rope(k_hat, rope_cos[:N], rope_sin[:N])
                gpu = (N * r + r * D + N * D) / (2 * N * D)
                rows.append({**base, "method": "lowrank_dense", "rank": r, "budget": N,
                             "tokens": N, "gpu_mem": gpu,
                             **metrics(dense_attention(q, k_hat, v), ref)})

            for B in BUDGETS:
                # pick chunks with landmarks, always add outlier chunks
                sel = select_chunks(q, landmarks, B // CHUNK, group, exclude=out_mask)
                sel = sel | out_mask[:, None, :]
                n_sel = B // CHUNK + n_out
                chunk_idx = sel.nonzero()[:, -1].reshape(n_kv, M, n_sel)
                tok_idx = chunk_to_token_idx(chunk_idx, CHUNK)                  # [kv_heads, M, n]
                is_out = out_mask[kv_idx, chunk_idx].repeat_interleave(CHUNK, -1)

                k_exact = k_post[kv_idx, tok_idx]                               # [kv_heads, M, n, d]
                v_sel = v[kv_idx, tok_idx]
                tokens = n_sel * CHUNK

                # selection only, exact keys (values offloaded)
                gpu = (N * D + (N // CHUNK) * D + n_out * CHUNK * D) / (2 * N * D)
                rows.append({**base, "method": "sparse_exact", "rank": D, "budget": B,
                             "tokens": tokens, "gpu_mem": gpu,
                             **metrics(sparse_attention(q, k_exact, v_sel, group), ref)})

                # full ShadowKV: rebuild only selected keys, outliers exact
                for r in ranks:
                    k_rec = reconstruct_selected(lrs[r], tok_idx, rope_cos, rope_sin, head_dim)
                    k_sel = torch.where(is_out[..., None], k_exact, k_rec)
                    gpu = (N * r + r * D + (N // CHUNK) * D + 2 * n_out * CHUNK * D) / (2 * N * D)
                    rows.append({**base, "method": "shadowkv", "rank": r, "budget": B,
                                 "tokens": tokens, "gpu_mem": gpu,
                                 **metrics(sparse_attention(q, k_sel, v_sel, group), ref)})
    print(f"prompt {p_i} done")

df = pd.DataFrame(rows)

# Save results
df.to_csv(ROOT / f"results/csv/shadowkv_attention_{TAG}.csv", index=False)
with open(ROOT / f"results/json/shadowkv_attention_env_{TAG}.json", "w") as f:
    json.dump({"model": MODEL, "context": N, "queries": M, "chunk": CHUNK,
               "outlier_frac": OUTLIER_FRAC, "budgets": BUDGETS, "rank_fracs": RANK_FRACS,
               "n_prompts": N_PROMPTS, "python": platform.python_version(),
               "torch": torch.__version__, "transformers": transformers.__version__,
               "dtype": "float32"}, f, indent=2)

# Summary
print(f"\nModel: {MODEL}   context {N}   chunk {CHUNK}   outliers {OUTLIER_FRAC:.0%}")
print("gpu_mem = estimated GPU memory vs full KV cache (1.0 = full)")

print("\nLow-rank keys only (all tokens)")
print(df[df.method == "lowrank_dense"].groupby("rank")[["gpu_mem", "out_relerr", "out_cos"]].mean().round(3).to_string())

print("\nSelection only (exact keys)")
print(df[df.method == "sparse_exact"].groupby("budget")[["tokens", "gpu_mem", "out_relerr", "out_cos"]].mean().round(3).to_string())

sk = df[df.method == "shadowkv"]
print("\nShadowKV output relative error (rows = rank, columns = budget)")
print(sk.pivot_table(index="rank", columns="budget", values="out_relerr").round(3).to_string())
print("\nShadowKV estimated GPU memory (rows = rank)")
print(sk.groupby("rank")["gpu_mem"].mean().round(3).to_string())

# Plot
fig, ax = plt.subplots(figsize=(7, 4.5))
se = df[df.method == "sparse_exact"].groupby("budget")["out_relerr"].mean()
ax.plot(se.index, se.values, "k-o", label="selection only (exact keys)")
for j, r in enumerate(sorted(sk["rank"].unique())):
    s = sk[sk["rank"] == r].groupby("budget")["out_relerr"].mean()
    ax.plot(s.index, s.values, "-o", color=f"C{j}", label=f"ShadowKV rank {r}")
    lr_err = df[(df.method == "lowrank_dense") & (df["rank"] == r)]["out_relerr"].mean()
    ax.axhline(lr_err, color=f"C{j}", linestyle="--", alpha=0.6, label=f"low-rank only rank {r}")
ax.set_xscale("log", base=2)
ax.set_xlabel("Sparse budget (tokens)")
ax.set_ylabel("Attention output relative error")
ax.set_title(f"{TAG}: where does ShadowKV's error come from?")
ax.legend(fontsize=7)
fig.tight_layout()
fig.savefig(ROOT / f"results/figures/shadowkv_attention_{TAG}.png", dpi=150)
print("\nSaved results to results/")