# Phase 4 + 5: how low-rank are real key caches, and does RoPE change that?
# Model: Qwen2.5-0.5B (24 layers, 2 KV heads, head_dim 64, RoPE + GQA)
# Per layer, keys form a matrix [tokens, kv_heads * head_dim], same layout ShadowKV uses for SVD.
# We compare keys BEFORE RoPE (k_proj output) and AFTER RoPE (what the KV cache stores).

import json
import platform
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
MODEL = "Qwen/Qwen2.5-0.5B"
LENGTHS = [1024, 2048, 4096]
RANKS = [4, 8, 16, 32, 64, 96, 128]
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


rows, spectra = [], []

for p, prompt in enumerate(prompts):
    with torch.no_grad():
        # model.model skips the output head, saving ~2.5 GB of RAM
        out = model.model(prompt[None], use_cache=True)

    for i in range(n_layers):
        keys = {"pre_rope": pre_rope[i],
                "post_rope": post_rope_keys(out.past_key_values, i)}

        for kind, K_all in keys.items():
            for n in LENGTHS:
                s, energy = spectrum(K_all[:n])
                row = {"prompt": p, "layer": i, "kind": kind, "tokens": n,
                       "rank_90": int((energy < 0.90).sum()) + 1,
                       "rank_99": int((energy < 0.99).sum()) + 1}
                for r in RANKS:
                    if r <= len(energy):
                        e = energy[r - 1].item()
                        row[f"energy_r{r}"] = e
                        # best rank-r approximation error (Eckart-Young)
                        row[f"relerr_r{r}"] = max(0.0, 1 - e) ** 0.5
                rows.append(row)

                if p == 0 and n == max_len:
                    for j, v in enumerate((s / s[0]).tolist()):
                        spectra.append({"layer": i, "kind": kind,
                                        "index": j + 1, "sigma_norm": v})
    print(f"prompt {p} done")

df = pd.DataFrame(rows)
df_spec = pd.DataFrame(spectra)

# Save results
df.to_csv(ROOT / "results/csv/key_rank_analysis.csv", index=False)
df_spec.to_csv(ROOT / "results/csv/key_spectrum.csv", index=False)
with open(ROOT / "results/json/rank_analysis_env.json", "w") as f:
    json.dump({"model": MODEL,
               "python": platform.python_version(),
               "torch": torch.__version__,
               "transformers": transformers.__version__,
               "lengths": LENGTHS, "ranks": RANKS,
               "n_prompts": N_PROMPTS, "dtype": "float32"}, f, indent=2)

# Print summary at 4K tokens
d = df[df.tokens == max_len]
print("\nMean relative error across layers and prompts (4K tokens)")
print(f"{'rank':>5} | {'pre-RoPE':>9} | {'post-RoPE':>9}")
for r in RANKS:
    col = f"relerr_r{r}"
    if col in d:
        pre = d[d.kind == "pre_rope"][col].mean()
        post = d[d.kind == "post_rope"][col].mean()
        print(f"{r:>5} | {pre:9.4f} | {post:9.4f}")

per_layer = d.groupby(["layer", "kind"])["rank_99"].mean().unstack()
print("\nRank needed for 99% energy, per layer (4K tokens)")
print(per_layer.round(1).to_string())

# Plot
fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

ax = axes[0]
for kind in ["pre_rope", "post_rope"]:
    sub = d[d.kind == kind]
    ranks = [r for r in RANKS if f"relerr_r{r}" in sub]
    means = [sub[f"relerr_r{r}"].mean() for r in ranks]
    stds = [sub[f"relerr_r{r}"].std() for r in ranks]
    ax.errorbar(ranks, means, yerr=stds, marker="o", capsize=3, label=kind)
ax.set_xlabel("Rank")
ax.set_ylabel("Relative error ||K - K_r|| / ||K||")
ax.set_title("Rank vs reconstruction error (4K tokens)")
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
ax.plot(per_layer.index, per_layer["pre_rope"], marker="o", label="pre_rope")
ax.plot(per_layer.index, per_layer["post_rope"], marker="o", label="post_rope")
ax.set_xlabel("Layer")
ax.set_ylabel("Rank for 99% energy")
ax.set_title("Compressibility per layer")
ax.legend()

fig.tight_layout()
fig.savefig(ROOT / "results/figures/key_rank_analysis.png", dpi=150)
print("\nSaved results to results/")