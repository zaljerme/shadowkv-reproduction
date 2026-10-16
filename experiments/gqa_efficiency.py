import os
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
# SETTINGS
# ============================================================

MODEL = "Qwen/Qwen2.5-1.5B"
LAYERS = [0, 3, 8, 22]

CONTEXT = 4096
N_QUERIES = 16
CHUNK = int(os.environ.get("KV_CHUNK", "8"))
BUDGET = int(os.environ.get("KV_BUDGET", "256"))
N_PROMPTS = 10

DEVICE = "cuda"

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / f"results/csv/gqa_efficiency_{CHUNK}.csv"

METHODS = [
    "mean",
    "max_sink",
    "mean_sink",
    "layer_aware",
    "oracle",
]

torch.manual_seed(42)

# ============================================================
# MODEL AND DATA
# ============================================================

print("Loading Qwen...", flush=True)

tokenizer = AutoTokenizer.from_pretrained(MODEL)

model = AutoModelForCausalLM.from_pretrained(
    MODEL, dtype=torch.float32
).to(DEVICE).eval()

cfg = model.config

H = cfg.num_attention_heads
Hkv = cfg.num_key_value_heads
assert H % Hkv == 0

group = H // Hkv
d = getattr(cfg, "head_dim", None) or cfg.hidden_size // H

print("Loading WikiText validation...", flush=True)

dataset = load_dataset(
    "Salesforce/wikitext",
    "wikitext-2-raw-v1",
    split="validation",
)

text = "\n".join(dataset["text"])

# Same tokenization as our previous 4K experiment.
token_ids = []

for start_char in range(0, len(text), 4000):
    section = text[start_char:start_char + 4000]
    token_ids.extend(
        tokenizer(section, add_special_tokens=False).input_ids
    )

prompts = []

for i in range(N_PROMPTS):
    start = (i + 10) * CONTEXT
    window = token_ids[start:start + CONTEXT]

    if len(window) != CONTEXT:
        raise RuntimeError("Not enough tokens")

    prompts.append(
        torch.tensor([window], device=DEVICE)
    )

# ============================================================
# QUERY CAPTURE
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

cos, sin = rope_cos_sin_from_model(model, CONTEXT)
cos = cos.to(DEVICE)
sin = sin.to(DEVICE)

# ============================================================
# TOKEN RANGES
# ============================================================

N = CONTEXT - N_QUERIES
C = N // CHUNK
n_chunks = BUDGET // CHUNK

assert N % CHUNK == 0
assert BUDGET % CHUNK == 0
assert n_chunks <= C

print(f"Context: {CONTEXT}")
print(f"Historical tokens: {N}")
print(f"Recent query window: {N_QUERIES}")
print(f"Historical retrieval budget: {BUDGET}")
print("Recent tokens are included separately.")
print("All methods have an equal budget.")

# ============================================================
# HELPER: SELECT HISTORICAL CHUNKS
# ============================================================

def select_chunks(scores, protect_sink):
    # scores: [KV heads, queries, historical chunks]

    if protect_sink:
        other = scores.clone()
        other[..., 0] = float("-inf")

        idx_other = other.topk(
            n_chunks - 1, dim=-1
        ).indices

        idx_zero = torch.zeros(
            (*scores.shape[:-1], 1),
            device=scores.device,
            dtype=torch.long,
        )

        idx = torch.cat(
            [idx_zero, idx_other], dim=-1
        )
    else:
        idx = scores.topk(
            n_chunks, dim=-1
        ).indices

    selected = torch.zeros_like(
        scores, dtype=torch.bool
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
                ids, use_cache=True
            )

            cache = output.past_key_values

            for layer_idx in LAYERS:

                q_pre = captured[layer_idx].reshape(
                    N_QUERIES, H, d
                ).transpose(0, 1)

                q = apply_rope(
                    q_pre,
                    cos[-N_QUERIES:],
                    sin[-N_QUERIES:],
                )

                if hasattr(cache, "layers"):
                    k = cache.layers[layer_idx].keys[0]
                    v = cache.layers[layer_idx].values[0]
                else:
                    k = cache.key_cache[layer_idx][0]
                    v = cache.value_cache[layer_idx][0]

                assert k.shape[1] == CONTEXT
                assert v.shape[1] == CONTEXT

                # Expand KV groups to their query heads.
                k_full = k.repeat_interleave(
                    group, dim=0
                )

                v_full = v.repeat_interleave(
                    group, dim=0
                )

                # Exact scores for historical AND recent tokens.
                scores = (
                    q @ k_full.transpose(-1, -2)
                ) / math.sqrt(d)

                # Full causal mask for the final 16 queries.
                positions = torch.arange(
                    CONTEXT, device=DEVICE
                )

                query_positions = torch.arange(
                    N, CONTEXT, device=DEVICE
                )

                causal_mask = (
                    positions.unsqueeze(0)
                    <= query_positions.unsqueeze(1)
                )

                # [heads, queries, context]
                full_scores = scores.masked_fill(
                    ~causal_mask.unsqueeze(0),
                    float("-inf"),
                )

                p_full = torch.softmax(
                    full_scores, dim=-1
                )

                full_output = p_full @ v_full

                # --------------------------------------------
                # HISTORICAL CHUNK SCORES
                # --------------------------------------------

                k_hist = k[:, :N]

                landmarks = k_hist.reshape(
                    Hkv, C, CHUNK, d
                ).mean(dim=2)

                lm = landmarks.repeat_interleave(
                    group, dim=0
                )

                lm_scores = (
                    q @ lm.transpose(-1, -2)
                ) / math.sqrt(d)

                lm_probs = torch.softmax(
                    lm_scores, dim=-1
                ).reshape(
                    Hkv, group, N_QUERIES, C
                )

                max_scores = lm_probs.max(dim=1).values
                mean_scores = lm_probs.mean(dim=1)

                # Oracle chooses historical chunks using
                # exact causal attention mass.
                chunk_mass = p_full[..., :N].reshape(
                    Hkv, group, N_QUERIES, C, CHUNK
                ).sum(dim=-1)

                oracle_scores = chunk_mass.mean(dim=1)

                methods = {
                    "mean": (mean_scores, False),
                    "max_sink": (max_scores, True),
                    "mean_sink": (mean_scores, True),
                    "layer_aware": (
                        mean_scores, layer_idx in [3, 8, 22]
                    ),
                    "oracle": (oracle_scores, False),
                }

                 # --------------------------------------------
                # CHUNK SELECTION DIAGNOSTIC
                # --------------------------------------------

                selected_mean = select_chunks(
                    mean_scores, True
                )

                selected_oracle = select_chunks(
                    oracle_scores, False
                )

                overlap = (
                    selected_mean & selected_oracle
                ).sum(dim=-1).float()

                chunk_recall = overlap / n_chunks

                # Fraction of oracle-selected historical
                # attention mass missed by mean + sink.
                missed_mask = (
                    selected_oracle & ~selected_mean
                )

                missed_mass = (
                    oracle_scores * missed_mask
                ).sum(dim=-1)

                oracle_selected_mass = (
                    oracle_scores * selected_oracle
                ).sum(dim=-1)

                missed_mass_fraction = (
                    missed_mass /
                    oracle_selected_mass.clamp_min(1e-12)
                )

                print(
                    f"Prompt {prompt_idx + 1}, "
                    f"Layer {layer_idx}: "
                    f"Recall={chunk_recall.mean().item():.4f}, "
                    f"Missed mass fraction="
                    f"{missed_mass_fraction.mean().item():.4f}",
                    flush=True,
                )

                # --------------------------------------------
                # RETRIEVE AND RECOMPUTE CAUSAL ATTENTION
                # --------------------------------------------

                for name, (ranking, protect) in methods.items():

                    selected = select_chunks(
                        ranking, protect
                    )

                     # --------------------------------------------
                    # NEW: GATHER SELECTED K/V TOKENS
                    # --------------------------------------------

                    chunk_ids = selected.int().topk(
                        n_chunks, dim=-1
                    ).indices

                    token_offsets = torch.arange(
                        CHUNK, device=DEVICE
                    )

                    token_indices = (
                        chunk_ids.unsqueeze(-1) * CHUNK
                        + token_offsets
                    ).reshape(Hkv, N_QUERIES, BUDGET)

                    head_ids = torch.arange(
                        Hkv, device=DEVICE
                    )[:, None, None]

                    k_selected = k[
                        head_ids, token_indices
                    ]

                    v_selected = v[
                        head_ids, token_indices
                    ]

                    assert k_selected.shape == (
                        Hkv, N_QUERIES, BUDGET, d
                    )

                    assert v_selected.shape == (
                        Hkv, N_QUERIES, BUDGET, d
                    )

                    # --------------------------------------------
                    # BUILD SPARSE K/V SET
                    # --------------------------------------------

                    # Expand KV heads to match GQA query heads.
                    k_sparse = k_selected.repeat_interleave(
                        group, dim=0
                    )

                    v_sparse = v_selected.repeat_interleave(
                        group, dim=0
                    )

                    # Add the recent 16 tokens.
                    k_recent = k_full[:, N:]
                    v_recent = v_full[:, N:]

                    k_recent = k_recent.unsqueeze(1).expand(
                        H, N_QUERIES, N_QUERIES, d
                    )

                    v_recent = v_recent.unsqueeze(1).expand(
                        H, N_QUERIES, N_QUERIES, d
                    )

                    k_sparse = torch.cat(
                        [k_sparse, k_recent], dim=2
                    )

                    v_sparse = torch.cat(
                        [v_sparse, v_recent], dim=2
                    )

                    # Compute attention only on retrieved tokens.
                    sparse_scores = (
                        q.unsqueeze(2) * k_sparse
                    ).sum(dim=-1) / math.sqrt(d)

                    # Apply causal mask to recent tokens.
                    recent_causal = torch.tril(
                        torch.ones(
                            N_QUERIES, N_QUERIES,
                            device=DEVICE, dtype=torch.bool
                        )
                    )

                    sparse_scores[:, :, BUDGET:] = (
                        sparse_scores[:, :, BUDGET:]
                        .masked_fill(
                            ~recent_causal.unsqueeze(0),
                            float("-inf")
                        )
                    )

                    sparse_probs = torch.softmax(
                        sparse_scores, dim=-1
                    )

                    sparse_output = (
                        sparse_probs.unsqueeze(-1) * v_sparse
                    ).sum(dim=2)

                    assert sparse_output.shape == (
                        H, N_QUERIES, d
                    )

                    historical_mask = (
                        selected
                        .repeat_interleave(group, dim=0)
                        .repeat_interleave(CHUNK, dim=-1)
                    )

                    # Retain every causally visible token
                    # in the recent 16-token window.
                    recent_mask = (
                        causal_mask[:, N:]
                        .unsqueeze(0)
                        .expand(H, -1, -1)
                    )

                    retrieval_mask = torch.cat(
                        [historical_mask, recent_mask],
                        dim=-1,
                    )

                    assert retrieval_mask.shape == (
                        H, N_QUERIES, CONTEXT
                    )

                    assert torch.all(
                        retrieval_mask[..., :N].sum(dim=-1)
                        == BUDGET
                    )

                    # Compute coverage of exact causal attention.
                    retained_mass = (
                        p_full * retrieval_mask
                    ).sum(dim=-1)

                    # Recompute softmax on the selected KV set.
                    selected_scores = scores.masked_fill(
                        ~retrieval_mask,
                        float("-inf"),
                    )

                    selected_p = torch.softmax(
                        selected_scores, dim=-1
                    )

                    retrieved_output = selected_p @ v_full

                    # Verify gathered sparse attention against
                    # the full-score, masked-attention reference.
                    sparse_difference = (
                        sparse_output - retrieved_output
                    ).abs().max().item()

                    assert sparse_difference < 1e-4, (
                        f"Sparse attention mismatch: "
                        f"{sparse_difference:.8f} "
                        f"(prompt={prompt_idx}, layer={layer_idx}, "
                        f"method={name})"
                    )

                    relative_l2 = (
                        torch.linalg.vector_norm(
                            retrieved_output - full_output,
                            dim=-1,
                        )
                        /
                        torch.linalg.vector_norm(
                            full_output,
                            dim=-1,
                        ).clamp_min(1e-8)
                    )

                    cosine = F.cosine_similarity(
                        retrieved_output,
                        full_output,
                        dim=-1,
                        eps=1e-8,
                    )

                    # Average metrics over all query heads.
                    mass_by_query = retained_mass.mean(dim=0)
                    l2_by_query = relative_l2.mean(dim=0)
                    cos_by_query = cosine.mean(dim=0)

                    for query_idx in range(N_QUERIES):
                        rows.append({
                            "prompt": prompt_idx,
                            "layer": layer_idx,
                            "query": query_idx,
                            "method": name,
                            "mass": mass_by_query[
                                query_idx
                            ].item(),
                            "relative_l2": l2_by_query[
                                query_idx
                            ].item(),
                            "cosine": cos_by_query[
                                query_idx
                            ].item(),
                            "historical_budget": BUDGET,
                            "recent_tokens": query_idx + 1,
                            "total_tokens_used": (
                                BUDGET + query_idx + 1
                            ),
                        })

            del output, cache

finally:
    for handle in handles:
        handle.remove()

# ============================================================
# SAVE RESULTS
# ============================================================

OUTPUT.parent.mkdir(
    parents=True, exist_ok=True
)

with OUTPUT.open("w", newline="") as f:
    writer = csv.DictWriter(
        f, fieldnames=list(rows[0].keys())
    )
    writer.writeheader()
    writer.writerows(rows)

metadata = {
    "model": MODEL,
    "context": CONTEXT,
    "layers": LAYERS,
    "queries_per_prompt": N_QUERIES,
    "prompts": N_PROMPTS,
    "historical_tokens": N,
    "historical_budget": BUDGET,
    "recent_window": N_QUERIES,
    "chunk_size": CHUNK,
    "dataset": "wikitext-2-raw-v1",
    "split": "validation",
    "reference": "full causal attention",
    "layer_aware_sink_layers": [3, 8, 22],
    "methods": METHODS,
    "torch_version": torch.__version__,
}

with OUTPUT.with_suffix(".json").open("w") as f:
    json.dump(metadata, f, indent=2)

# ============================================================
# SUMMARY
# ============================================================

print("\nFULL CAUSAL ATTENTION FIDELITY")
print(
    f"{'Method':16s} "
    f"{'Mass':>8s} "
    f"{'Rel L2':>10s} "
    f"{'Cosine':>10s}"
)

for method in METHODS:
    result = [
        r for r in rows if r["method"] == method
    ]

    avg_mass = sum(r["mass"] for r in result) / len(result)
    avg_l2 = sum(r["relative_l2"] for r in result) / len(result)
    avg_cos = sum(r["cosine"] for r in result) / len(result)

    print(
        f"{method:16s} "
        f"{avg_mass:8.4f} "
        f"{avg_l2:10.4f} "
        f"{avg_cos:10.4f}"
    )

print(f"\nSaved {len(rows)} rows to {OUTPUT}")
