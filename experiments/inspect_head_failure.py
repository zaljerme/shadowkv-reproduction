import math
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from models.toy_attention import apply_rope, rope_cos_sin_from_model

MODEL = "Qwen/Qwen2.5-0.5B"
LAYER = 8
PROMPT = 14
QUERY = 10
CONTEXT = 1024
N_QUERIES = 16
CHUNK = 8
BUDGET = 256
DEVICE = "cuda"

print("Loading Qwen...")
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(
    MODEL, dtype=torch.float32
).to(DEVICE).eval()

print("Loading target WikiText passage...")
dataset = load_dataset(
    "Salesforce/wikitext",
    "wikitext-2-raw-v1",
    split="test"
)
all_ids = tokenizer(
    "\n".join(dataset["text"]),
    add_special_tokens=False
).input_ids

start = (PROMPT + 100) * CONTEXT
ids = torch.tensor(
    [all_ids[start:start + CONTEXT]],
    device=DEVICE
)
assert ids.shape[1] == CONTEXT

cfg = model.config
H = cfg.num_attention_heads
Hkv = cfg.num_key_value_heads
group = H // Hkv
d = getattr(cfg, "head_dim", None) or cfg.hidden_size // H

captured = {}

def capture_query(module, inputs, output):
    captured["q"] = output[:, -N_QUERIES:].detach()

handle = model.model.layers[
    LAYER
].self_attn.q_proj.register_forward_hook(capture_query)

with torch.inference_mode():
    output = model.model(ids, use_cache=True)

handle.remove()

q_pre = captured["q"][0].reshape(
    N_QUERIES, H, d
).transpose(0, 1)

cos, sin = rope_cos_sin_from_model(model, CONTEXT)
q = apply_rope(
    q_pre,
    cos[-N_QUERIES:].to(DEVICE),
    sin[-N_QUERIES:].to(DEVICE)
)

# Select one query position
q = q[:, QUERY, :]

cache = output.past_key_values
if hasattr(cache, "layers"):
    k = cache.layers[LAYER].keys[0]
else:
    k = cache.key_cache[LAYER][0]

# Same historical prefix as the original benchmark
N = CONTEXT - N_QUERIES
C = N // CHUNK
k = k[:, :N]

with torch.inference_mode():
    landmarks = k.reshape(
        Hkv, C, CHUNK, d
    ).mean(dim=2)

    exact_scores = (
        q[:, None, :] @
        k.repeat_interleave(group, 0).transpose(-1, -2)
    ).squeeze(1) / math.sqrt(d)

    p = torch.softmax(exact_scores, dim=-1)

    lm = landmarks.repeat_interleave(group, 0)
    lm_scores = torch.einsum(
        "hd,hcd->hc", q, lm
    ) / math.sqrt(d)

    lm_probs = torch.softmax(
        lm_scores, dim=-1
    ).reshape(Hkv, group, C)

    max_scores = lm_probs.max(dim=1).values
    mean_scores = lm_probs.mean(dim=1)

    methods = {
        "max": max_scores,
        "mean": mean_scores,
        "hybrid_25": 0.25 * max_scores + 0.75 * mean_scores,
        "hybrid_75": 0.75 * max_scores + 0.25 * mean_scores,
    }

    masks = {}
    for name, scores in methods.items():
        indices = scores.topk(BUDGET // CHUNK, dim=-1).indices
        selected = torch.zeros(
            Hkv, C, dtype=torch.bool, device=DEVICE
        )
        selected.scatter_(1, indices, True)
        masks[name] = selected.repeat_interleave(
            CHUNK, dim=-1
        ).repeat_interleave(group, dim=0)

    print("\nATTENTION MASS BY QUERY HEAD")
    print("Head  Group     Max    Mean    H25    H75")

    for head in range(H):
        values = {
            name: (p[head] * mask[head]).sum().item()
            for name, mask in masks.items()
        }
        print(
            f"{head:4d}  {head // group:5d}  "
            f"{values['max']:.4f}  "
            f"{values['mean']:.4f}  "
            f"{values['hybrid_25']:.4f}  "
            f"{values['hybrid_75']:.4f}"
        )

    print("\nAVERAGE BY KV GROUP")
    for g in range(Hkv):
        head_range = slice(g * group, (g + 1) * group)
        values = {
            name: (
                p[head_range] * mask[head_range]
            ).sum(dim=-1).mean().item()
            for name, mask in masks.items()
        }
        print(f"KV group {g}: {values}")

    print("\nOVERALL ATTENTION MASS")
    for name, mask in masks.items():
        print(
            f"{name:12s}",
            round((p * mask).sum(dim=-1).mean().item(), 4)
        )

    print("\nIMPORTANT CHUNKS MISSED BY MEAN")
    true_chunk_mass = p.reshape(
        Hkv, group, C, CHUNK
    ).sum(dim=-1).mean(dim=1)

    max_selected = masks["max"][::group, ::CHUNK].bool()
    mean_selected = masks["mean"][::group, ::CHUNK].bool()

    for g in range(Hkv):
        missed = max_selected[g] & ~mean_selected[g]
        chunks = torch.where(missed)[0]
        ranked = sorted(
            chunks.tolist(),
            key=lambda c: true_chunk_mass[g, c].item(),
            reverse=True
        )
        print(f"\nKV group {g}:")
        for c in ranked[:5]:
            print(
                f"  Chunk {c:3d}, "
                f"tokens {c*CHUNK}-{c*CHUNK+CHUNK-1}, "
                f"attention mass {true_chunk_mass[g,c].item():.4f}"
            )
