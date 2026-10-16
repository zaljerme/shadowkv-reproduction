
import math
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "Qwen/Qwen2.5-0.5B"
LAYER = 3
CONTEXT = 1024
CHUNK = 8
BUDGET = 256

device = "cuda"

tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(
    MODEL, dtype=torch.float32
).to(device).eval()

cfg = model.config
H = cfg.num_attention_heads
Hkv = cfg.num_key_value_heads
group = H // Hkv
d = cfg.hidden_size // H

# Capture Q and the model's post-RoPE K
q_capture = {}
def q_hook(module, inputs, output):
    q_capture["q"] = output.detach()

model.model.layers[LAYER].self_attn.q_proj.register_forward_hook(q_hook)

text = "The purpose of this experiment is to analyze attention retrieval. " * 100
ids = tokenizer(text, return_tensors="pt").input_ids[:, :CONTEXT].to(device)
print("Actual sequence length:", ids.shape[1])

with torch.inference_mode():
    output = model.model(ids, use_cache=True)

q_pre = q_capture["q"][0, -1].reshape(H, d)

# Use the model's own RoPE
from models.toy_attention import rope_cos_sin_from_model, apply_rope

cos, sin = rope_cos_sin_from_model(model, ids.shape[1])
q = apply_rope(
    q_pre, cos[-1].to(device), sin[-1].to(device)
)

cache = output.past_key_values
if hasattr(cache, "layers"):
    k = cache.layers[LAYER].keys[0]
else:
    k = cache.key_cache[LAYER][0]

# Exclude the current query token
N = ((k.shape[1] - 1) // CHUNK) * CHUNK
k = k[:, :N]

print("Historical context tokens:", N)
C = N // CHUNK

# Landmark construction
landmarks = k.reshape(Hkv, C, CHUNK, d).mean(dim=2)

# Exact attention scores
scores = q[:, None, :] @ k.repeat_interleave(group, 0).transpose(-1, -2)
scores = scores.squeeze(1) / math.sqrt(d)
p = torch.softmax(scores, dim=-1)
chunk_mass = p.reshape(Hkv, group, C, CHUNK).sum(-1)

# Landmark scores
lm = landmarks.repeat_interleave(group, 0)
lm_scores = torch.einsum("hd,hcd->hc", q, lm) / math.sqrt(d)
lm_probs = torch.softmax(lm_scores, dim=-1)
lm_probs = lm_probs.reshape(Hkv, group, C)

methods = {
    "max": lm_probs.max(dim=1).values,
    "mean": lm_probs.mean(dim=1),
    "oracle": chunk_mass.mean(dim=1),
}

n_chunks = BUDGET // CHUNK

for name, values in methods.items():
    idx = values.topk(n_chunks, dim=-1).indices
    selected = torch.zeros((Hkv, C), dtype=torch.bool, device=device)
    selected.scatter_(1, idx, True)

    mask = selected.repeat_interleave(group, 0)
    mask = mask.repeat_interleave(CHUNK, dim=-1)

    mass = (p * mask).sum(-1).mean().item()
    print(f"{name}: attention mass = {mass:.4f}")

print("Actual tokens per KV head:", n_chunks * CHUNK)
