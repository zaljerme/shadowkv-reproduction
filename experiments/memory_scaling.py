# Phase 3: how does a normal KV cache grow with context length?
# Part A: theoretical cache size for full Llama-3-8B
# Part B: measured cache size and decode speed for one attention layer on this laptop

import json
import platform
import statistics
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from shadowkv.kv_cache import FullKVCache
from models.toy_attention import ToyAttention

GB = 1024 ** 3
MB = 1024 ** 2
LLAMA3_8B = dict(n_layers=32, n_kv_heads=8, head_dim=128, bytes_per_elem=2)
WARMUP, STEPS = 3, 20


def kv_bytes(n_tokens, batch, n_layers, n_kv_heads, head_dim, bytes_per_elem):
    return 2 * n_layers * n_tokens * n_kv_heads * head_dim * bytes_per_elem * batch


# Part A: Llama-3-8B cache size from 1K to 128K tokens
contexts_a = [1024 * 2 ** i for i in range(8)]
rows_a = []
for batch in [1, 8]:
    for n in contexts_a:
        rows_a.append({"context": n, "batch": batch,
                       "kv_gb": kv_bytes(n, batch, **LLAMA3_8B) / GB})
df_a = pd.DataFrame(rows_a)

# Part B: one Llama-3-8B shaped layer, 1K to 32K tokens
torch.manual_seed(0)
attn = ToyAttention(d_model=4096, n_heads=32, n_kv_heads=8, use_rope=True).eval()
contexts_b = [1024 * 2 ** i for i in range(6)]
rows_b = []

for n in contexts_b:
    # fill the cache with n random tokens
    cache = FullKVCache()
    cache.update(torch.randn(1, 8, n, 128), torch.randn(1, 8, n, 128))
    measured = cache.nbytes()
    theory = kv_bytes(n, 1, n_layers=1, n_kv_heads=8, head_dim=128, bytes_per_elem=4)

    # time one decode step (one new token) several times
    x = torch.randn(1, 1, 4096)
    times = []
    with torch.no_grad():
        for i in range(WARMUP + STEPS):
            t0 = time.perf_counter()
            attn(x, cache)
            ms = (time.perf_counter() - t0) * 1000
            if i >= WARMUP:
                times.append(ms)

    med = statistics.median(times)
    rows_b.append({"context": n,
                   "measured_mb": measured / MB,
                   "theory_mb": theory / MB,
                   "match": measured == theory,
                   "decode_ms_median": med,
                   "decode_ms_std": statistics.stdev(times),
                   "tokens_per_s": 1000 / med})
    print(f"{n:>6} tokens | cache {measured / MB:8.1f} MB | decode {med:7.2f} ms")
    del cache

df_b = pd.DataFrame(rows_b)

# Save results
(ROOT / "results/csv").mkdir(parents=True, exist_ok=True)
(ROOT / "results/json").mkdir(parents=True, exist_ok=True)
(ROOT / "results/figures").mkdir(parents=True, exist_ok=True)

df_a.to_csv(ROOT / "results/csv/kv_theory_llama3_8b.csv", index=False)
df_b.to_csv(ROOT / "results/csv/kv_measured_one_layer_cpu.csv", index=False)

with open(ROOT / "results/json/memory_scaling_env.json", "w") as f:
    json.dump({"python": platform.python_version(),
               "torch": torch.__version__,
               "cpu": platform.processor(),
               "threads": torch.get_num_threads(),
               "dtype": "float32",
               "warmup": WARMUP, "steps": STEPS}, f, indent=2)

# Plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

for batch, g in df_a.groupby("batch"):
    ax1.plot(g["context"], g["kv_gb"], marker="o", label=f"batch {batch}")
ax1.axhline(80, color="red", linestyle="--", label="A100 80 GB")
ax1.set_xscale("log", base=2)
ax1.set_yscale("log")
ax1.set_xlabel("Context length (tokens)")
ax1.set_ylabel("KV cache (GB)")
ax1.set_title("Llama-3-8B KV cache (theoretical, bf16)")
ax1.legend()

ax2.plot(df_b["context"], df_b["decode_ms_median"], marker="o")
ax2.set_xscale("log", base=2)
ax2.set_xlabel("Context length (tokens)")
ax2.set_ylabel("Decode latency (ms/token)")
ax2.set_title("One attention layer, CPU fp32 (measured)")

fig.tight_layout()
fig.savefig(ROOT / "results/figures/kv_memory_vs_context.png", dpi=150)
print("\nSaved results to results/")