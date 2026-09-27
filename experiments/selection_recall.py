# Phase 8-10: do landmarks and outliers find the tokens attention needs?
# Uses exact post-RoPE keys (no low-rank yet) to isolate selection quality.
# Context = first 4032 tokens, queries = last 64 positions (all see the full context).
# Usage: python experiments\selection_recall.py [model_name]

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
import transformers
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from shadowkv.landmarks import build_landmarks, outlier_mask
from shadowkv.selector import select_chunks
from models.toy_attention import rope_cos_sin, apply_rope

MODEL = sys.argv[1] if len(sys.argv) > 1 else "TinyLlama/TinyLlama_v1.1"
TAG = MODEL.split("/")[-1]
T = 4096
M = 64
N = T - M          # context the queries attend to
TOPK = 64
CHUNKS = [4, 8, 16, 32]
OUTLIER_FRACS = [0.0, 0.005, 0.01, 0.02, 0.05]
BUDGETS = [64, 128, 256, 512]   # tokens fetched by landmark selection
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

# Capture pre-RoPE queries for the last M positions
q_pre = {}

def hook_q(i):
    def h(m, inp, out):
        q_pre[i] = out[0, -M:].detach()
    return h

for i, layer in enumerate(layers):
    layer.self_attn.q_proj.register_forward_hook(hook_q(i))


def cache_keys(cache, i):
    if hasattr(cache, "layers"):
        return cache.layers[i].keys[0]
    if hasattr(cache, "key_cache"):
        return cache.key_cache[i][0]
    return cache[i][0][0]


rope_cos, rope_sin = rope_cos_sin(torch.arange(T), head_dim, base=theta)


def evaluate(tok_mask, p, true_top):
    # tok_mask, p, true_top: [heads, M, N]
    mass = (p * tok_mask).sum(-1).mean().item()
    recall = ((true_top & tok_mask).sum(-1).float() / TOPK).mean().item()
    return mass, recall


rows, locality = [], []

for p_i, prompt in enumerate(prompts):
    with torch.no_grad():
        out = model.model(prompt[None], use_cache=True)

        for i in range(n_layers):
            k = cache_keys(out.past_key_values, i)[:, :N]          # [kv_heads, N, d]
            q = q_pre[i].reshape(M, n_heads, head_dim).transpose(0, 1)
            q = apply_rope(q, rope_cos[-M:], rope_sin[-M:])        # [heads, M, d]

            # exact attention over the context
            scores = q @ k.repeat_interleave(group, 0).transpose(-1, -2) / math.sqrt(head_dim)
            p = torch.softmax(scores, -1)                           # [heads, M, N]
            top_idx = scores.topk(TOPK, -1).indices
            true_top = torch.zeros_like(p, dtype=torch.bool).scatter_(-1, top_idx, True)

            # best possible token selection
            for B in BUDGETS:
                rows.append({"prompt": p_i, "layer": i, "method": "oracle_token", "chunk": 1,
                             "outlier_frac": 0.0, "budget": B, "tokens_used": B,
                             "mass": p.topk(B, -1).values.sum(-1).mean().item(), "recall": 1.0})

            for c in CHUNKS:
                landmarks, min_cos, cos = build_landmarks(k, c)
                C = N // c
                locality.append({"prompt": p_i, "layer": i, "chunk": c,
                                 "mean_cos": cos.mean().item(),
                                 "p5_min_cos": torch.quantile(min_cos.flatten(), 0.05).item()})

                # best possible chunk selection (by true attention mass per chunk)
                chunk_mass = p.reshape(n_heads, M, C, c).sum(-1)
                for B in BUDGETS:
                    idx = chunk_mass.topk(min(B // c, C), -1).indices
                    sel = torch.zeros_like(chunk_mass, dtype=torch.bool).scatter_(-1, idx, True)
                    mass, recall = evaluate(sel.repeat_interleave(c, -1), p, true_top)
                    rows.append({"prompt": p_i, "layer": i, "method": "oracle_chunk", "chunk": c,
                                 "outlier_frac": 0.0, "budget": B, "tokens_used": B,
                                 "mass": mass, "recall": recall})

                # ShadowKV style: landmarks + outlier chunks
                for frac in OUTLIER_FRACS:
                    out_mask = outlier_mask(min_cos, frac)          # [kv_heads, C]
                    n_out_tokens = out_mask.sum(-1).float().mean().item() * c
                    for B in BUDGETS:
                        sel = select_chunks(q, landmarks, B // c, group, exclude=out_mask)
                        sel = sel | out_mask[:, None, :]            # outliers always included
                        tok_mask = sel.repeat_interleave(c, -1).repeat_interleave(group, 0)
                        mass, recall = evaluate(tok_mask, p, true_top)
                        rows.append({"prompt": p_i, "layer": i, "method": "landmark", "chunk": c,
                                     "outlier_frac": frac, "budget": B,
                                     "tokens_used": B + n_out_tokens,
                                     "mass": mass, "recall": recall})
    print(f"prompt {p_i} done")

df = pd.DataFrame(rows)
df_loc = pd.DataFrame(locality)

# Save results
df.to_csv(ROOT / f"results/csv/selection_recall_{TAG}.csv", index=False)
df_loc.to_csv(ROOT / f"results/csv/landmark_locality_{TAG}.csv", index=False)
with open(ROOT / f"results/json/selection_recall_env_{TAG}.json", "w") as f:
    json.dump({"model": MODEL, "context": N, "queries": M, "topk": TOPK,
               "chunks": CHUNKS, "outlier_fracs": OUTLIER_FRACS, "budgets": BUDGETS,
               "n_prompts": N_PROMPTS, "python": platform.python_version(),
               "torch": torch.__version__, "transformers": transformers.__version__,
               "dtype": "float32", "keys": "exact post-RoPE"}, f, indent=2)

# Summary
print(f"\nModel: {MODEL}   context {N}   top-{TOPK}")

print("\n1) Landmark locality: how well chunk means represent their keys")
print(df_loc.groupby("chunk")[["mean_cos", "p5_min_cos"]].mean().round(3).to_string())

print("\n2) Budget sweep (chunk 8, outliers 1%)")
lm = df[(df.method == "landmark") & (df.chunk == 8) & (df.outlier_frac == 0.01)].groupby("budget").mean(numeric_only=True)
oc = df[(df.method == "oracle_chunk") & (df.chunk == 8)].groupby("budget").mean(numeric_only=True)
ot = df[df.method == "oracle_token"].groupby("budget").mean(numeric_only=True)
print(f"{'budget':>6} | {'tokens':>6} | {'mass LM':>7} | {'mass OC':>7} | {'mass OT':>7} | {'recall LM':>9} | {'recall OC':>9}")
for B in BUDGETS:
    print(f"{B:>6} | {lm.loc[B, 'tokens_used']:6.0f} | {lm.loc[B, 'mass']:7.3f} | {oc.loc[B, 'mass']:7.3f} | "
          f"{ot.loc[B, 'mass']:7.3f} | {lm.loc[B, 'recall']:9.3f} | {oc.loc[B, 'recall']:9.3f}")
print("LM = landmark selection, OC = oracle chunk, OT = oracle token")

print("\n3) Outlier effect (chunk 8, budget 256)")
o = df[(df.method == "landmark") & (df.chunk == 8) & (df.budget == 256)].groupby("outlier_frac").mean(numeric_only=True)
print(o[["tokens_used", "mass", "recall"]].round(3).to_string())

print("\n4) Chunk size effect (budget 256, outliers 1%)")
cs = df[(df.method == "landmark") & (df.budget == 256) & (df.outlier_frac == 0.01)].groupby("chunk").mean(numeric_only=True)
print(cs[["tokens_used", "mass", "recall"]].round(3).to_string())

# Plot
fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
axes[0].plot(lm.index, lm["mass"], marker="o", label="landmark (chunk 8, 1% outliers)")
axes[0].plot(oc.index, oc["mass"], marker="o", label="oracle chunk")
axes[0].plot(ot.index, ot["mass"], marker="o", label="oracle token")
axes[0].set_xscale("log", base=2)
axes[0].set_xlabel("Sparse budget (tokens)")
axes[0].set_ylabel("Attention mass captured")
axes[0].set_title(f"{TAG}: selection quality")
axes[0].legend()

axes[1].plot(o.index * 100, o["recall"], marker="o")
axes[1].set_xlabel("Outlier chunks (%)")
axes[1].set_ylabel(f"Top-{TOPK} recall")
axes[1].set_title("Outlier budget (chunk 8, budget 256)")

axes[2].plot(cs.index, cs["recall"], marker="o")
axes[2].set_xscale("log", base=2)
axes[2].set_xlabel("Chunk size")
axes[2].set_ylabel(f"Top-{TOPK} recall")
axes[2].set_title("Chunk size (budget 256, 1% outliers)")

fig.tight_layout()
fig.savefig(ROOT / f"results/figures/selection_recall_{TAG}.png", dpi=150)
print("\nSaved results to results/")