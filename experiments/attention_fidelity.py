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
from models.toy_attention import rope_cos_sin, apply_rope

MODEL = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen2.5-0.5B"
TAG = MODEL.split("/")[-1]
T = 4096      # context length
M = 64        # last M positions act as decode queries
TOPK = 64     # about 1.6% of context, similar ratio to ShadowKV's 2048 budget at 128K
RANKS = [8, 16, 32, 64, 96, 128, 160, 192]
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

# Real text
text = "\n".join(load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")["text"])
ids = tok(text, return_tensors="pt").input_ids[0]
prompts = [ids[p * 60000: p * 60000 + T] for p in range(N_PROMPTS)]

# Capture pre-RoPE queries (last M only) and keys
q_pre, k_pre = {}, {}

def hook_q(i):
    def h(m, inp, out):
        q_pre[i] = out[0, -M:].detach()  # [M, heads * head_dim]
    return h

def hook_k(i):
    def h(m, inp, out):
        k_pre[i] = out[0].detach()  # [T, kv_heads * head_dim]
    return h

for i, layer in enumerate(layers):
    layer.self_attn.q_proj.register_forward_hook(hook_q(i))
    layer.self_attn.k_proj.register_forward_hook(hook_k(i))


def cache_kv(cache, i):
    # works across transformers versions, returns [kv_heads, T, head_dim]
    if hasattr(cache, "layers"):
        return cache.layers[i].keys[0], cache.layers[i].values[0]
    if hasattr(cache, "key_cache"):
        return cache.key_cache[i][0], cache.value_cache[i][0]
    return cache[i][0][0], cache[i][1][0]


def heads(x, n):
    # [tokens, n * head_dim] -> [n, tokens, head_dim]
    return x.reshape(x.shape[0], n, head_dim).transpose(0, 1)


def flat(x):
    # [n, tokens, head_dim] -> [tokens, n * head_dim]
    return x.transpose(0, 1).reshape(x.shape[1], -1)


pos = torch.arange(T)
rope_cos, rope_sin = rope_cos_sin(pos, head_dim, base=theta)
mask = pos[None, :] > pos[-M:][:, None]  # [M, T] causal mask


def attention(q, k, v):
    # q: [heads, M, d] after RoPE, k and v: [kv_heads, T, d]
    k = k.repeat_interleave(group, 0)
    v = v.repeat_interleave(group, 0)
    scores = (q @ k.transpose(-1, -2)) / math.sqrt(head_dim)
    scores = scores.masked_fill(mask, float("-inf"))
    return scores, torch.softmax(scores, -1) @ v


def compare(scores, out, scores_hat, out_hat):
    logp = F.log_softmax(scores, -1)
    logq = F.log_softmax(scores_hat, -1)
    p = logp.exp()
    kl = torch.where(p > 0, p * (logp - logq), torch.zeros_like(p)).sum(-1).mean().item()
    rel = ((out - out_hat).norm() / out.norm()).item()
    cos_sim = F.cosine_similarity(out, out_hat, dim=-1).mean().item()

    # top-k recall: how many of the truly most-attended tokens does the compressed version also pick
    true_idx = scores.topk(TOPK, -1).indices
    hat_idx = scores_hat.topk(TOPK, -1).indices
    true_m = torch.zeros_like(scores, dtype=torch.bool).scatter_(-1, true_idx, True)
    hat_m = torch.zeros_like(scores, dtype=torch.bool).scatter_(-1, hat_idx, True)
    recall = ((true_m & hat_m).sum(-1).float() / TOPK).mean().item()

    return {"out_relerr": rel, "out_cos": cos_sim, "kl": kl, "topk_recall": recall}


rows = []
rope_rel_err = 0.0

for p_i, prompt in enumerate(prompts):
    with torch.no_grad():
        out = model.model(prompt[None], use_cache=True)

        for i in range(n_layers):
            k_post, v = cache_kv(out.past_key_values, i)
            q = apply_rope(heads(q_pre[i], n_heads), rope_cos[-M:], rope_sin[-M:])

            # sanity check: our RoPE must reproduce the model's cached keys
            k_check = apply_rope(heads(k_pre[i], n_kv), rope_cos, rope_sin)
            err = ((k_check - k_post).abs().max() / k_post.abs().max()).item()
            rope_rel_err = max(rope_rel_err, err)

            scores, o = attention(q, k_post, v)
            D = k_pre[i].shape[1]
            f_pre = torch.linalg.svd(k_pre[i].float(), full_matrices=False)
            f_post = torch.linalg.svd(flat(k_post).float(), full_matrices=False)

            for r in [r for r in RANKS if r < D]:
                # ShadowKV style: compress pre-RoPE keys, apply RoPE after reconstruction
                lr = LowRankKeys.from_keys(k_pre[i], r, f_pre)
                k_hat = apply_rope(heads(lr.reconstruct(), n_kv), rope_cos, rope_sin)
                s_hat, o_hat = attention(q, k_hat, v)
                rows.append({"prompt": p_i, "layer": i, "variant": "pre_rope", "rank": r,
                             "dim": D, "mem_ratio": lr.nbytes() / lr.full_nbytes(),
                             **compare(scores, o, s_hat, o_hat)})

                # Baseline: compress post-RoPE keys directly
                lr = LowRankKeys.from_keys(flat(k_post), r, f_post)
                k_hat = heads(lr.reconstruct(), n_kv)
                s_hat, o_hat = attention(q, k_hat, v)
                rows.append({"prompt": p_i, "layer": i, "variant": "post_rope", "rank": r,
                             "dim": D, "mem_ratio": lr.nbytes() / lr.full_nbytes(),
                             **compare(scores, o, s_hat, o_hat)})
    print(f"prompt {p_i} done")

print(f"\nRoPE sanity check, max relative error vs model cache: {rope_rel_err:.2e}")
if rope_rel_err > 1e-3:
    print("WARNING: our RoPE does not match the model. Results below are not valid.")

df = pd.DataFrame(rows)

# Save results
df.to_csv(ROOT / f"results/csv/attention_fidelity_{TAG}.csv", index=False)
with open(ROOT / f"results/json/attention_fidelity_env_{TAG}.json", "w") as f:
    json.dump({"model": MODEL, "context": T, "queries": M, "topk": TOPK,
               "ranks": RANKS, "n_prompts": N_PROMPTS, "rope_rel_err": rope_rel_err,
               "python": platform.python_version(), "torch": torch.__version__,
               "transformers": transformers.__version__, "dtype": "float32"}, f, indent=2)

# Summary
D = int(df["dim"].iloc[0])
g = df.groupby(["rank", "variant"])[["topk_recall", "out_relerr", "kl", "mem_ratio"]].mean().unstack()
print(f"\nModel: {MODEL}   key dim per layer: {D}   top-k: {TOPK} of {T}")
print(f"{'rank':>5} | {'mem':>5} | {'recall pre':>10} | {'recall post':>11} | "
      f"{'err pre':>8} | {'err post':>8} | {'KL pre':>8} | {'KL post':>8}")
for r in g.index:
    print(f"{r:>5} | {g.loc[r, ('mem_ratio', 'pre_rope')]:5.2f} | "
          f"{g.loc[r, ('topk_recall', 'pre_rope')]:10.3f} | {g.loc[r, ('topk_recall', 'post_rope')]:11.3f} | "
          f"{g.loc[r, ('out_relerr', 'pre_rope')]:8.4f} | {g.loc[r, ('out_relerr', 'post_rope')]:8.4f} | "
          f"{g.loc[r, ('kl', 'pre_rope')]:8.4f} | {g.loc[r, ('kl', 'post_rope')]:8.4f}")

mid = D // 4
per_layer = df[df["rank"] == mid].groupby(["layer", "variant"])["topk_recall"].mean().unstack()
print(f"\nTop-k recall per layer at rank {mid}")
print(per_layer.round(3).to_string())

# Plot
fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
for variant in ["pre_rope", "post_rope"]:
    sub = df[df.variant == variant].groupby("rank").mean(numeric_only=True)
    axes[0].plot(sub.index, sub["topk_recall"], marker="o", label=variant)
    axes[1].plot(sub.index, sub["out_relerr"], marker="o", label=variant)
    axes[2].plot(per_layer.index, per_layer[variant], marker="o", label=variant)

axes[0].set_xlabel("Rank")
axes[0].set_ylabel(f"Top-{TOPK} recall")
axes[0].set_title(f"{TAG}: top-k token recall")
axes[1].set_xlabel("Rank")
axes[1].set_ylabel("Attention output relative error")
axes[1].set_yscale("log")
axes[1].set_title("Output error")
axes[2].set_xlabel("Layer")
axes[2].set_ylabel(f"Top-{TOPK} recall")
axes[2].set_title(f"Per-layer recall at rank {mid}")
for ax in axes:
    ax.legend()

fig.tight_layout()
fig.savefig(ROOT / f"results/figures/attention_fidelity_{TAG}.png", dpi=150)
print("\nSaved results to results/")