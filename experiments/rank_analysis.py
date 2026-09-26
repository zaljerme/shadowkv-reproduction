# Phase 4 + 5: how low-rank are real key caches, and does RoPE change that?
# Per layer, keys form a matrix [tokens, kv_heads * head_dim], same layout ShadowKV uses for SVD.
# We compare keys BEFORE RoPE (k_proj output) and AFTER RoPE (what the KV cache stores).
# Also reports centered rank (mean removed), since a large k_proj bias can fake low rank.
# Usage: python experiments\rank_analysis.py [model_name]

import json
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
MODEL = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen2.5-0.5B"
TAG = MODEL.split("/")[-1]
LENGTHS = [1024, 2048, 4096]
RANKS = [4, 8, 16, 32, 64, 96, 128, 160, 192, 256]
N_PROMPTS = 2

# Load model
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).eval()
layers = model.model.layers
n_layers = len(layers)

# Real text: different chunks of WikiText as different prompts
text = "\n".join(load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")["text"])
ids = tok(text, return_tensors="pt").input_ids[0]
max_len = max(LENGTHS)
prompts = [ids[p * 60000: p * 60000 + max_len] for p in range(N_PROMPTS)]

# Capture pre-RoPE keys with a hook on each layer's k_proj
pre_rope = {}

def make_hook(i):
    def hook(module, inp, out):
        pre_rope[i] = out[0].detach()  # [tokens, kv_heads * head_dim]
    return hook

for i in range(n_layers):
    layers[i].self_attn.k_proj.register_forward_hook(make_hook(i))


def post_rope_keys(cache, i):
    # works across transformers versions
    if hasattr(cache, "layers"):
        k = cache.layers[i].keys
    elif hasattr(cache, "key_cache"):
        k = cache.key_cache[i]
    else:
        k = cache[i][0]
    k = k[0]  # [kv_heads, tokens, head_dim]
    return k.transpose(0, 1).reshape(k.shape[1], -1)


def spectrum(K):
    s = torch.linalg.svdvals(K.float())
    energy = (s ** 2).cumsum(0) / (s ** 2).sum()
    return s, energy


def rank_for(energy, target):
    return int((energy < target).sum()) + 1


rows, spectra = [], []

for p, prompt in enumerate(prompts):
    with torch.no_grad():
        # model.model skips the output head to save RAM
        out = model.model(prompt[None], use_cache=True)

    for i in range(n_layers):
        keys = {"pre_rope": pre_rope[i],
                "post_rope": post_rope_keys(out.past_key_values, i)}

        for kind, K_all in keys.items():
            for n in LENGTHS:
                K = K_all[:n].float()
                s, energy = spectrum(K)
                _, energy_c = spectrum(K - K.mean(dim=0))

                row = {"prompt": p, "layer": i, "kind": kind, "tokens": n,
                       "dim": K.shape[1],
                       "rank_90": rank_for(energy, 0.90),
                       "rank_99": rank_for(energy, 0.99),
                       "rank_99_centered": rank_for(energy_c, 0.99)}
                for r in RANKS:
                    if r <= len(energy):
                        # best rank-r error (Eckart-Young), raw and centered
                        row[f"relerr_r{r}"] = max(0.0, 1 - energy[r - 1].item()) ** 0.5
                        row[f"relerr_c_r{r}"] = max(0.0, 1 - energy_c[r - 1].item()) ** 0.5
                rows.append(row)

                if p == 0 and n == max_len:
                    for j, v in enumerate((s / s[0]).tolist()):
                        spectra.append({"layer": i, "kind": kind,
                                        "index": j + 1, "sigma_norm": v})
    print(f"prompt {p} done")

df = pd.DataFrame(rows)
df_spec = pd.DataFrame(spectra)

# Save results
df.to_csv(ROOT / f"results/csv/key_rank_analysis_{TAG}.csv", index=False)
df_spec.to_csv(ROOT / f"results/csv/key_spectrum_{TAG}.csv", index=False)
with open(ROOT / f"results/json/rank_analysis_env_{TAG}.json", "w") as f:
    json.dump({"model": MODEL,
               "python": platform.python_version(),
               "torch": torch.__version__,
               "transformers": transformers.__version__,
               "lengths": LENGTHS, "ranks": RANKS,
               "n_prompts": N_PROMPTS, "dtype": "float32"}, f, indent=2)

# Print summary at 4K tokens
d = df[df.tokens == max_len]
dim = int(d["dim"].iloc[0])
print(f"\nModel: {MODEL}   key dim per layer: {dim}")
print("\nMean relative error across layers and prompts (4K tokens)")
print(f"{'rank':>5} | {'pre':>7} | {'post':>7} | {'pre_c':>7} | {'post_c':>7}")
for r in RANKS:
    if f"relerr_r{r}" in d:
        pre = d[d.kind == "pre_rope"]
        post = d[d.kind == "post_rope"]
        print(f"{r:>5} | {pre[f'relerr_r{r}'].mean():7.4f} | {post[f'relerr_r{r}'].mean():7.4f} | "
              f"{pre[f'relerr_c_r{r}'].mean():7.4f} | {post[f'relerr_c_r{r}'].mean():7.4f}")

per_layer = d.groupby(["layer", "kind"])[["rank_99", "rank_99_centered"]].mean().unstack()
per_layer.columns = [f"{a}_{b}" for a, b in per_layer.columns]
print("\nRank for 99% energy per layer (4K tokens)")
print(per_layer.round(1).to_string())

# Plot
fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

ax = axes[0]
for kind in ["pre_rope", "post_rope"]:
    sub = d[d.kind == kind]
    ranks = [r for r in RANKS if f"relerr_r{r}" in sub]
    for prefix, style in [("relerr_r", "-"), ("relerr_c_r", "--")]:
        means = [sub[f"{prefix}{r}"].mean() for r in ranks]
        label = kind + (" centered" if prefix == "relerr_c_r" else "")
        ax.plot(ranks, means, style, marker="o", label=label)
ax.set_xlabel("Rank")
ax.set_ylabel("Relative error")
ax.set_title(f"{TAG}: rank vs error (4K tokens)")
ax.legend()

ax = axes[1]
for layer in [0, n_layers // 2, n_layers - 1]:
    for kind, style in [("pre_rope", "-"), ("post_rope", "--")]:
        sub = df_spec[(df_spec.layer == layer) & (df_spec.kind == kind)]
        ax.plot(sub["index"], sub["sigma_norm"], style, label=f"L{layer} {kind}")
ax.set_yscale("log")
ax.set_xlabel("Singular value index")
ax.set_ylabel("sigma / sigma_1")
ax.set_title("Singular value spectrum")
ax.legend(fontsize=7)

ax = axes[2]
for col in per_layer.columns:
    ax.plot(per_layer.index, per_layer[col], marker="o", label=col)
ax.set_xlabel("Layer")
ax.set_ylabel("Rank for 99% energy")
ax.set_title("Compressibility per layer")
ax.legend(fontsize=7)

fig.tight_layout()
fig.savefig(ROOT / f"results/figures/key_rank_analysis_{TAG}.png", dpi=150)
print("\nSaved results to results/")