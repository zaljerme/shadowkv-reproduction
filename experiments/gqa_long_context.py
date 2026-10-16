import csv
import json
import math
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from models.toy_attention import (
    apply_rope,
    rope_cos_sin_from_model,
)

# ============================================================
# EXPERIMENT SETTINGS
# ============================================================

MODEL = "Qwen/Qwen2.5-0.5B"
LAYERS = [0, 3, 8, 22]

CONTEXT = 4096
N_QUERIES = 16
CHUNK = 8
BUDGET = 256
N_PROMPTS = 10

DEVICE = "cuda"

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results/csv/gqa_long_context_4k.csv"

torch.manual_seed(42)

# ============================================================
# LOAD MODEL
# ============================================================

print("Loading model...")

tokenizer = AutoTokenizer.from_pretrained(MODEL)

model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    dtype=torch.float32,
).to(DEVICE).eval()

cfg = model.config

H = cfg.num_attention_heads
Hkv = cfg.num_key_value_heads
group = H // Hkv
d = getattr(cfg, "head_dim", None) or cfg.hidden_size // H

# ============================================================
# LOAD WIKITEXT VALIDATION
# ============================================================

print("Loading WikiText validation...")

dataset = load_dataset(
    "Salesforce/wikitext",
    "wikitext-2-raw-v1",
    split="validation",
)

text = "\n".join(dataset["text"])

# Tokenize in smaller sections to avoid long-input warnings.
token_ids = []

for start_char in range(0, len(text), 4000):
    section = text[start_char:start_char + 4000]
    token_ids.extend(
        tokenizer(
            section,
            add_special_tokens=False,
        ).input_ids
    )

prompts = []

for i in range(N_PROMPTS):
    start = i * CONTEXT
    window = token_ids[start:start + CONTEXT]

    if len(window) != CONTEXT:
        raise RuntimeError("Not enough WikiText tokens")

    prompts.append(
        torch.tensor([window], device=DEVICE)
    )

# ============================================================
# CAPTURE PRE-ROPE QUERY PROJECTIONS
# ============================================================

captured = {}
handles = []

for layer_idx in LAYERS:

    def make_hook(idx):
        def hook(module, inputs, output):
            captured[idx] = output[:, -N_QUERIES:].detach()
        return hook

    handle = model.model.layers[
        layer_idx
    ].self_attn.q_proj.register_forward_hook(
        make_hook(layer_idx)
    )

    handles.append(handle)

# ============================================================
# ROPE
# ============================================================

cos, sin = rope_cos_sin_from_model(model, CONTEXT)
cos = cos.to(DEVICE)
sin = sin.to(DEVICE)

# ============================================================
# RETRIEVAL SETTINGS
# ============================================================

N = CONTEXT - N_QUERIES

assert N % CHUNK == 0
assert BUDGET % CHUNK == 0

C = N // CHUNK
n_chunks = BUDGET // CHUNK

assert n_chunks <= C

print(f"Context tokens: {CONTEXT}")
print(f"Historical tokens: {N}")
print(f"Chunks available: {C}")
print(f"Chunks retrieved per KV head: {n_chunks}")
print(f"Retrieval budget: {BUDGET} tokens")

rows = []

# ============================================================
# MAIN EXPERIMENT
# ============================================================

for prompt_idx, ids in enumerate(prompts):

    captured.clear()

    print(
        f"Processing prompt {prompt_idx + 1}/{N_PROMPTS}",
        flush=True,
    )

    with torch.inference_mode():

        output = model.model(ids, use_cache=True)
        cache = output.past_key_values

        for layer_idx in LAYERS:

            # ------------------------------------------------
            # QUERY VECTORS
            # ------------------------------------------------

            q_pre = captured[layer_idx].reshape(
                N_QUERIES, H, d
            ).transpose(0, 1)

            q = apply_rope(
                q_pre,
                cos[-N_QUERIES:],
                sin[-N_QUERIES:],
            )

            # ------------------------------------------------
            # CACHED KEYS
            # ------------------------------------------------

            if hasattr(cache, "layers"):
                k = cache.layers[layer_idx].keys[0]
            else:
                k = cache.key_cache[layer_idx][0]

            k = k[:, :N]

            # ------------------------------------------------
            # CHUNK LANDMARKS
            # ------------------------------------------------

            landmarks = k.reshape(
                Hkv, C, CHUNK, d
            ).mean(dim=2)

            # ------------------------------------------------
            # EXACT HISTORICAL ATTENTION
            # ------------------------------------------------

            k_full = k.repeat_interleave(
                group, dim=0
            )

            exact_scores = (
                q @ k_full.transpose(-1, -2)
            ) / math.sqrt(d)

            p = torch.softmax(
                exact_scores,
                dim=-1,
            )

            chunk_mass = p.reshape(
                Hkv, group, N_QUERIES, C, CHUNK
            ).sum(dim=-1)

            # ------------------------------------------------
            # LANDMARK SCORES
            # ------------------------------------------------

            lm = landmarks.repeat_interleave(
                group, dim=0
            )

            lm_scores = (
                q @ lm.transpose(-1, -2)
            ) / math.sqrt(d)

            lm_probs = torch.softmax(
                lm_scores,
                dim=-1,
            ).reshape(
                Hkv, group, N_QUERIES, C
            )

            max_scores = lm_probs.max(dim=1).values
            mean_scores = lm_probs.mean(dim=1)

            # ------------------------------------------------
            # HEAD DISAGREEMENT
            # ------------------------------------------------

            eps = 1e-12

            head_disagreement = (
                lm_probs * (
                    lm_probs.clamp_min(eps).log()
                    - mean_scores.unsqueeze(1)
                    .clamp_min(eps).log()
                )
            ).sum(dim=-1).mean(dim=1)

            disagreement_per_query = (
                head_disagreement.mean(dim=0)
            )

            # ------------------------------------------------
            # AGGREGATION METHODS
            # ------------------------------------------------

            methods = {
                "max": max_scores,
                "mean": mean_scores,
                "hybrid_25": (
                    0.25 * max_scores
                    + 0.75 * mean_scores
                ),
                "hybrid_50": (
                    0.50 * max_scores
                    + 0.50 * mean_scores
                ),
                "hybrid_75": (
                    0.75 * max_scores
                    + 0.25 * mean_scores
                ),
                "oracle": chunk_mass.mean(dim=1),
                "max_sink": max_scores,
                "mean_sink": mean_scores,
                "hybrid_25_sink": (
                    0.25 * max_scores
                    + 0.75 * mean_scores
                ),
                "layer_aware": mean_scores,
            }

            # ------------------------------------------------
            # CHUNK SELECTION
            # ------------------------------------------------

            for method, scores in methods.items():

                protect_sink = (
                    method.endswith("_sink")
                    or (
                        method == "layer_aware"
                        and layer_idx in [3, 8]
                    )
                )

                if protect_sink:

                    # Reserve chunk 0 and choose 31 others.
                    other_scores = scores.clone()
                    other_scores[..., 0] = float("-inf")

                    other_idx = other_scores.topk(
                        n_chunks - 1,
                        dim=-1,
                    ).indices

                    zero_idx = torch.zeros(
                        (*scores.shape[:-1], 1),
                        dtype=torch.long,
                        device=DEVICE,
                    )

                    idx = torch.cat(
                        [zero_idx, other_idx],
                        dim=-1,
                    )

                else:

                    # Standard selection of 32 chunks.
                    idx = scores.topk(
                        n_chunks,
                        dim=-1,
                    ).indices

                # ------------------------------------------------
                # BUILD SELECTION MASK
                # ------------------------------------------------

                selected = torch.zeros(
                    (Hkv, N_QUERIES, C),
                    dtype=torch.bool,
                    device=DEVICE,
                )

                selected.scatter_(
                    -1, idx, True
                )

                # All methods must use the same budget.
                assert torch.all(
                    selected.sum(dim=-1) == n_chunks
                )

                if protect_sink:
                    assert torch.all(
                        selected[..., 0]
                    )

                token_mask = (
                    selected
                    .repeat_interleave(group, dim=0)
                    .repeat_interleave(CHUNK, dim=-1)
                )

                # ------------------------------------------------
                # RETAINED ATTENTION MASS
                # ------------------------------------------------

                mass = (
                    p * token_mask
                ).sum(dim=-1).mean(dim=0)

                for query_idx, value in enumerate(mass):

                    rows.append({
                        "prompt": prompt_idx,
                        "layer": layer_idx,
                        "query": query_idx,
                        "method": method,
                        "budget": BUDGET,
                        "tokens_used": n_chunks * CHUNK,
                        "mass": value.item(),
                        "head_disagreement": (
                            disagreement_per_query[
                                query_idx
                            ].item()
                        ),
                    })

    del output, cache

# ============================================================
# REMOVE HOOKS
# ============================================================

for handle in handles:
    handle.remove()

# ============================================================
# SAVE RESULTS
# ============================================================

OUTPUT.parent.mkdir(
    parents=True,
    exist_ok=True,
)

with OUTPUT.open("w", newline="") as f:

    writer = csv.DictWriter(
        f,
        fieldnames=list(rows[0].keys()),
    )

    writer.writeheader()
    writer.writerows(rows)

# ============================================================
# SAVE METADATA
# ============================================================

metadata = {
    "model": MODEL,
    "layers": LAYERS,
    "context": CONTEXT,
    "historical_tokens": N,
    "queries": N_QUERIES,
    "chunk_size": CHUNK,
    "budget": BUDGET,
    "prompts": N_PROMPTS,
    "prompt_start_window": 0,
    "dataset": "wikitext-2-raw-v1",
    "split": "validation",
    "dtype": "float32",
    "torch": torch.__version__,
    "evaluation": "historical-prefix attention",
    "outlier_chunks": False,
    "sink_protection": (
        "Reserve chunk 0 within the 32-chunk budget"
    ),
    "layer_aware_policy": {
        "protect_sink_layers": [3, 8],
        "unprotected_layers": [0, 22],
    },
    "methods": list(methods.keys()),
}

with OUTPUT.with_suffix(".json").open("w") as f:
    json.dump(
        metadata,
        f,
        indent=2,
    )

# ============================================================
# PRINT SUMMARY
# ============================================================

print("\nAVERAGE ATTENTION MASS")

method_names = [
    "max",
    "mean",
    "hybrid_25",
    "hybrid_50",
    "hybrid_75",
    "max_sink",
    "mean_sink",
    "hybrid_25_sink",
    "layer_aware",
    "oracle",
]

for method in method_names:

    values = [
        row["mass"]
        for row in rows
        if row["method"] == method
    ]

    print(
        f"{method:16s}: "
        f"{sum(values) / len(values):.4f}"
    )

print(f"\nSaved {len(rows)} rows to {OUTPUT}")
