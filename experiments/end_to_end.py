# Phase 13-14: end-to-end ShadowKV, baselines, and ablation
# Prefill P tokens exactly, then predict the next G tokens one at a time (teacher forcing)
# with each method, and compare against the full KV cache.
# Usage: python experiments\end_to_end.py [model_name]

import copy
import json
import platform
import sys
import time
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
from models.hf_wrapper import STATE, setup_rope, build_layer_structs

MODEL = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen2.5-0.5B"
TAG = MODEL.split("/")[-1]
P = 3968   # prefill length, divisible by chunk size
G = 128    # decode steps
N_PROMPTS = 4
CONFIGS = [
    {"name": "full", "mode": "full"},
    {"name": "lowrank r=1/4", "mode": "lowrank", "rank_frac": 0.25},
    {"name": "lowrank+outliers r=1/4", "mode": "lowrank_outliers", "rank_frac": 0.25},
    {"name": "sparse B=512", "mode": "sparse", "budget": 512},
    {"name": "streaming 552 tok", "mode": "streaming", "stream_tokens": 552},
    {"name": "streaming 800 tok", "mode": "streaming", "stream_tokens": 800},
    {"name": "shadowkv r=1/4 B=256", "mode": "shadowkv", "rank_frac": 0.25, "budget": 256},
    {"name": "shadowkv r=1/4 B=512", "mode": "shadowkv", "rank_frac": 0.25, "budget": 512},
    {"name": "shadowkv r=1/2 B=512", "mode": "shadowkv", "rank_frac": 0.5, "budget": 512},
]

# Load model with default attention first, for a sanity check
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).eval()
cfg = model.config
n_layers = cfg.num_hidden_layers
n_heads = cfg.num_attention_heads
n_kv = cfg.num_key_value_heads
head_dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // n_heads
theta = getattr(cfg, "rope_theta", None) or cfg.rope_parameters["rope_theta"]
D = n_kv * head_dim
setup_rope(P + G + 1, head_dim, theta, model)

text = "\n".join(load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")["text"])
ids = tok(text, return_tensors="pt").input_ids[0]
prompts = [ids[p * 60000: p * 60000 + P + G + 1] for p in range(N_PROMPTS)]


def set_impl(name):
    if hasattr(model, "set_attn_implementation"):
        model.set_attn_implementation(name)
    else:
        model.config._attn_implementation = name


def cache_keys(cache, i):
    if hasattr(cache, "layers"):
        return cache.layers[i].keys[0]
    if hasattr(cache, "key_cache"):
        return cache.key_cache[i][0]
    return cache[i][0][0]


def gpu_mem_estimate(mode, r):
    """Estimated GPU memory vs a full KV cache (1.0 = full)."""
    c = STATE.chunk
    n_out_tok = round(STATE.outlier_frac * P / c) * c
    if mode == "full":
        return 1.0
    if mode == "lowrank":
        return (P * r + r * D + P * D) / (2 * P * D)
    if mode == "lowrank_outliers":
        return (P * r + r * D + n_out_tok * D + P * D) / (2 * P * D)
    if mode == "sparse":
        return (P * D + (P // c) * D + n_out_tok * D) / (2 * P * D)
    if mode == "streaming":
        return STATE.stream_tokens / P
    return (P * r + r * D + (P // c) * D + 2 * n_out_tok * D) / (2 * P * D)


# Sanity check: our attention in full mode must match the model's own attention
with torch.no_grad():
    probe = prompts[0][:256][None]
    ref = model(probe).logits
    set_impl("shadowkv")
    STATE.mode = "full"
    ours = model(probe).logits
check = (ours - ref).abs().max().item()
print(f"Sanity check, max logit difference vs stock attention: {check:.2e}")
if check > 1e-3:
    print("WARNING: custom attention does not match the model. Stop and check.")

rows = []
for p_i, seq in enumerate(prompts):
    ctx, cont = seq[:P], seq[P:]
    targets = cont[1:G + 1]

    STATE.mode = "full"
    with torch.no_grad():
        base_cache = model.model(ctx[None], use_cache=True).past_key_values
    keys = [cache_keys(base_cache, i) for i in range(n_layers)]

    full_lp = None
    for c in CONFIGS:
        STATE.mode = c["mode"]
        STATE.rank = int(D * c.get("rank_frac", 1.0))
        STATE.budget = c.get("budget", 0)
        STATE.stream_tokens = c.get("stream_tokens", 0)
        if c["mode"] != "full":
            build_layer_structs(keys, P)

        cache = copy.deepcopy(base_cache)
        logits = []
        t0 = time.perf_counter()
        with torch.no_grad():
            for t in range(G):
                out = model(cont[t].view(1, 1), past_key_values=cache, use_cache=True)
                cache = out.past_key_values
                logits.append(out.logits[0, -1].float())
        secs = time.perf_counter() - t0

        lp = torch.log_softmax(torch.stack(logits), -1)          # [G, vocab]
        nll = -lp[torch.arange(G), targets].mean().item()
        if c["mode"] == "full":
            full_lp = lp
        kl = (full_lp.exp() * (full_lp - lp)).sum(-1).mean().item()
        agree = (lp.argmax(-1) == full_lp.argmax(-1)).float().mean().item()

        rows.append({"prompt": p_i, "config": c["name"], "mode": c["mode"],
                     "rank": STATE.rank, "budget": STATE.budget,
                     "stream_tokens": STATE.stream_tokens, "nll": nll,
                     "ppl": float(torch.exp(torch.tensor(nll))), "kl": kl,
                     "top1_agree": agree, "gpu_mem": gpu_mem_estimate(c["mode"], STATE.rank),
                     "decode_s": secs})
        print(f"prompt {p_i} | {c['name']:<24} | ppl {rows[-1]['ppl']:7.3f} | "
              f"agree {agree:.3f} | KL {kl:.4f}")

df = pd.DataFrame(rows)

# Save results
df.to_csv(ROOT / f"results/csv/end_to_end_{TAG}.csv", index=False)
with open(ROOT / f"results/json/end_to_end_env_{TAG}.json", "w") as f:
    json.dump({"model": MODEL, "prefill": P, "decode_steps": G, "n_prompts": N_PROMPTS,
               "chunk": STATE.chunk, "outlier_frac": STATE.outlier_frac, "sink": STATE.sink,
               "sanity_max_logit_diff": check, "python": platform.python_version(),
               "torch": torch.__version__, "transformers": transformers.__version__,
               "dtype": "float32"}, f, indent=2)

# Summary (average over prompts; perplexity from mean NLL)
order = [c["name"] for c in CONFIGS]
s = df.groupby("config")[["nll", "kl", "top1_agree", "gpu_mem"]].mean().reindex(order)
s["agree_std"] = df.groupby("config")["top1_agree"].std().reindex(order)
s["ppl"] = s["nll"].apply(lambda x: float(torch.exp(torch.tensor(x))))
s["ppl_increase_%"] = (s["ppl"] / s.loc["full", "ppl"] - 1) * 100
print(f"\nModel: {MODEL}   prefill {P}   decode {G} tokens x {N_PROMPTS} prompts")
print(s[["gpu_mem", "ppl", "ppl_increase_%", "top1_agree", "agree_std", "kl"]].round(3).to_string())

# Plot
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
axes[0].barh(s.index, s["ppl_increase_%"])
axes[0].set_xlabel("Perplexity increase vs full (%)")
axes[0].invert_yaxis()
axes[0].set_title(f"{TAG}: end-to-end quality")
axes[1].scatter(s["gpu_mem"], s["top1_agree"])
for name, row in s.iterrows():
    axes[1].annotate(name, (row["gpu_mem"], row["top1_agree"]), fontsize=7)
axes[1].set_xlabel("Estimated GPU memory vs full KV")
axes[1].set_ylabel("Top-1 agreement with full")
axes[1].set_title("Memory vs agreement")
fig.tight_layout()
fig.savefig(ROOT / f"results/figures/end_to_end_{TAG}.png", dpi=150)
print("\nSaved results to results/")