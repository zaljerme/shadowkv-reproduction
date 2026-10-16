import csv
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from models.toy_attention import (
    apply_rope,
    rope_cos_sin_from_model,
)

# ============================================================
# CONFIGURATION
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
OUTPUT = ROOT / "results/csv/gqa_attention_fidelity_4k.csv"

METHOD_NAMES = [
    "mean",
    "max_sink",
    "mean_sink",
    "layer_aware",
    "oracle",
]

torch.manual_seed(42)

# ============================================================
# LOAD MODEL
# ============================================================

print("Loading Qwen...", flush=True)

tokenizer = AutoTokenizer.from_pretrained(MODEL)

model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    dtype=torch.float32,
).to(DEVICE).eval()

cfg = model.config

H = cfg.num_attention_heads
Hkv = cfg.num_key_value_heads

assert H % Hkv == 0

group = H // Hkv
d = getattr(cfg, "head_dim", None) or cfg.hidden_size // H

# ============================================================
# LOAD WIKITEXT
# ============================================================

print("Loading WikiText validation...", flush=True)

dataset = load_dataset(
    "Salesforce/wikitext",
    "wikitext-2-raw-v1",
    split="validation",
)

text = "\n".join(dataset["text"])

# Match the tokenization used in the previous 4K benchmark.
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
# QUERY PROJECTION HOOKS
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

print(f"Context: {CONTEXT} tokens")
print(f"Historical context: {N} tokens")
print(f"Available chunks: {C}")
print(f"Retrieved chunks: {n_chunks}")
print(f"Retrieval budget: {BUDGET} tokens")
print(f"Query heads: {H}")
print(f"KV heads: {Hkv}")

# ============================================================
# HELPER: SELECT CHUNKS
# ============================================================

def select_chunks(scores, protect_sink):
    """
    scores: [KV heads, queries, chunks]

    Returns:
        [KV heads, queries, chunks] boolean mask

    Every method selects exactly n_chunks chunks.
    """

    if protect_sink:

        other_scores = scores.clone()
        other_scores[..., 0] = float("-inf")

        other_idx = other_scores.topk(
            n_chunks - 1,
            dim=-1,
        ).indices

        zero_idx = torch.zeros(
            (*scores.shape[:-1], 1),
            dtype=torch.long,
            device=scores.device,
        )

        idx = torch.cat(
            [zero_idx, other_idx],
            dim=-1,
        )

    else:

        idx = scores.topk(
            n_chunks,
            dim=-1,
        ).indices

    selected = torch.zeros_like(
        scores,
        dtype=torch.bool,
    )

    selected.scatter_(-1, idx, True)

    assert torch.all(
        selected.sum(dim=-1) == n_chunks
    )

    if protect_sink:
        assert torch.all(selected[..., 0])

    return selected


# ============================================================
# MAIN EXPERIMENT
# ============================================================

rows = []

try:

    for prompt_idx, ids in enumerate(prompts):

        captured.clear()

        print(
            f"Processing prompt {prompt_idx + 1}/{N_PROMPTS}",
            flush=True,
        )

        with torch.inference_mode():

            output = model.model(
                ids,
                use_cache=True,
            )

            cache = output.past_key_values

            for layer_idx in LAYERS:

                # --------------------------------------------
                # QUERY VECTORS
                # --------------------------------------------

                q_pre = captured[layer_idx].reshape(
                    N_QUERIES, H, d
                ).transpose(0, 1)

                q = apply_rope(
                    q_pre,
                    cos[-N_QUERIES:],
                    sin[-N_QUERIES:],
                )

                # --------------------------------------------
                # CACHED KEYS AND VALUES
                # --------------------------------------------

                if hasattr(cache, "layers"):

                    k = cache.layers[layer_idx].keys[0]
                    v = cache.layers[layer_idx].values[0]

                else:

                    k = cache.key_cache[layer_idx][0]
                    v = cache.value_cache[layer_idx][0]

                # Use the same historical prefix as before.
                k = k[:, :N]
                v = v[:, :N]

                assert k.shape == v.shape

                # --------------------------------------------
                # EXACT HISTORICAL ATTENTION
                # --------------------------------------------

                k_full = k.repeat_interleave(
                    group,
                    dim=0,
                )

                v_full = v.repeat_interleave(
                    group,
                    dim=0,
                )

                exact_scores = (
                    q @ k_full.transpose(-1, -2)
                ) / math.sqrt(d)

                p = torch.softmax(
                    exact_scores,
                    dim=-1,
                )

                # Shape: [query heads, queries, head_dim]
                full_output = p @ v_full

                # --------------------------------------------
                # CHUNK LANDMARKS
                # --------------------------------------------

                landmarks = k.reshape(
                    Hkv, C, CHUNK, d
                ).mean(dim=2)

                lm = landmarks.repeat_interleave(
                    group,
                    dim=0,
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

                max_scores = lm_probs.max(
                    dim=1
                ).values

                mean_scores = lm_probs.mean(dim=1)

                # --------------------------------------------
                # EXACT ATTENTION-MASS ORACLE
                # --------------------------------------------

                chunk_mass = p.reshape(
                    Hkv,
                    group,
                    N_QUERIES,
                    C,
                    CHUNK,
                ).sum(dim=-1)

                oracle_scores = chunk_mass.mean(dim=1)

                # --------------------------------------------
                # METHODS
                # --------------------------------------------

                methods = {
                    "mean": (
                        mean_scores,
                        False,
                    ),
                    "max_sink": (
                        max_scores,
                        True,
                    ),
                    "mean_sink": (
                        mean_scores,
                        True,
                    ),
                    "layer_aware": (
                        mean_scores,
                        layer_idx in [3, 8],
                    ),
                    "oracle": (
                        oracle_scores,
                        False,
                    ),
                }

                for method, (
                    scores, protect_sink
                ) in methods.items():

                    # ----------------------------------------
                    # CHUNK SELECTION
                    # ----------------------------------------

                    selected = select_chunks(
                        scores,
                        protect_sink,
                    )

                    token_mask = (
                        selected
                        .repeat_interleave(
                            group,
                            dim=0,
                        )
                        .repeat_interleave(
                            CHUNK,
                            dim=-1,
                        )
                    )

                    # ----------------------------------------
                    # ATTENTION MASS
                    # ----------------------------------------

                    retained_mass = (
                        p * token_mask
                    ).sum(dim=-1)

                    # ----------------------------------------
                    # RECONSTRUCT ATTENTION OUTPUT
                    # ----------------------------------------

                    # Renormalize probabilities only over
                    # retrieved historical KV tokens.

                    selected_p = p * token_mask

                    denominators = selected_p.sum(
                        dim=-1,
                        keepdim=True,
                    ).clamp_min(1e-12)

                    selected_p = (
                        selected_p / denominators
                    )

                    retrieved_output = (
                        selected_p @ v_full
                    )

                    # ----------------------------------------
                    # RELATIVE L2 ERROR
                    # ----------------------------------------

                    difference = (
                        retrieved_output - full_output
                    )

                    relative_l2 = (
                        torch.linalg.vector_norm(
                            difference,
                            dim=-1,
                        )
                        /
                        torch.linalg.vector_norm(
                            full_output,
                            dim=-1,
                        ).clamp_min(1e-8)
                    )

                    # ----------------------------------------
                    # COSINE SIMILARITY
                    # ----------------------------------------

                    cosine = F.cosine_similarity(
                        retrieved_output,
                        full_output,
                        dim=-1,
                        eps=1e-8,
                    )

                    # ----------------------------------------
                    # SAVE QUERY-LEVEL METRICS
                    # ----------------------------------------

                    # We average the metrics over the 14
                    # query heads for each query position.

                    mass_by_query = (
                        retained_mass.mean(dim=0)
                    )

                    l2_by_query = (
                        relative_l2.mean(dim=0)
                    )

                    cosine_by_query = (
                        cosine.mean(dim=0)
                    )

                    for query_idx in range(N_QUERIES):

                        rows.append({
                            "prompt": prompt_idx,
                            "layer": layer_idx,
                            "query": query_idx,
                            "method": method,
                            "context": CONTEXT,
                            "budget": BUDGET,
                            "tokens_used": (
                                n_chunks * CHUNK
                            ),
                            "mass": (
                                mass_by_query[
                                    query_idx
                                ].item()
                            ),
                            "relative_l2": (
                                l2_by_query[
                                    query_idx
                                ].item()
                            ),
                            "cosine": (
                                cosine_by_query[
                                    query_idx
                                ].item()
                            ),
                        })

        del output, cache

finally:

    for handle in handles:
        handle.remove()

# ============================================================
# SAVE CSV
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
# SAVE EXPERIMENT METADATA
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
    "dataset": "wikitext-2-raw-v1",
    "split": "validation",
    "prompt_start_window": 0,
    "dtype": "float32",
    "torch": torch.__version__,
    "evaluation": "historical-prefix attention output",
    "reference": "exact historical attention",
    "metrics": [
        "mass",
        "relative_l2",
        "cosine",
    ],
    "layer_aware_sink_layers": [3, 8],
    "methods": METHOD_NAMES,
}

with OUTPUT.with_suffix(".json").open("w") as f:

    json.dump(
        metadata,
        f,
        indent=2,
    )

# ============================================================
# PRINT RESULTS
# ============================================================

print("\nATTENTION OUTPUT FIDELITY RESULTS")

print(
    f"{'Method':16s} "
    f"{'Mass':>8s} "
    f"{'Rel L2':>10s} "
    f"{'Cosine':>10s}"
)

for method in METHOD_NAMES:

    method_rows = [
        row for row in rows
        if row["method"] == method
    ]

    avg_mass = sum(
        r["mass"] for r in method_rows
    ) / len(method_rows)

    avg_l2 = sum(
        r["relative_l2"] for r in method_rows
    ) / len(method_rows)

    avg_cosine = sum(
        r["cosine"] for r in method_rows
    ) / len(method_rows)

    print(
        f"{method:16s} "
        f"{avg_mass:8.4f} "
        f"{avg_l2:10.4f} "
        f"{avg_cosine:10.4f}"
    )

print(
    f"\nSaved {len(rows)} rows to {OUTPUT}"
)
