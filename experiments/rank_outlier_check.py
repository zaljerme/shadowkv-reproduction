# Diagnostic: why do some layers need only rank 1 for 99% energy?
# Hypothesis 1: the first token (attention sink) has a huge key
# Hypothesis 2: the k_proj bias adds a large constant to every key

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "Qwen/Qwen2.5-0.5B"
N = 4096
CHECK_LAYERS = [0, 1, 2, 8, 5, 15]  # last two are normal layers for comparison

tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).eval()
text = "\n".join(load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")["text"])
ids = tok(text, return_tensors="pt").input_ids[0][:N]

pre = {}
for i in CHECK_LAYERS:
    model.model.layers[i].self_attn.k_proj.register_forward_hook(
        lambda m, inp, out, i=i: pre.__setitem__(i, out[0].detach()))

with torch.no_grad():
    model.model(ids[None], use_cache=False)


def rank99(K):
    s = torch.linalg.svdvals(K)
    e = (s ** 2).cumsum(0) / (s ** 2).sum()
    return int((e < 0.99).sum()) + 1


print(f"{'layer':>5} | {'tok0 norm':>9} | {'median norm':>11} | {'bias norm':>9} | "
      f"{'raw':>4} | {'no tok0':>7} | {'no tok0 + centered':>18}")

for i in CHECK_LAYERS:
    K = pre[i]
    norms = K.norm(dim=1)
    bias = model.model.layers[i].self_attn.k_proj.bias
    bias_norm = bias.norm().item() if bias is not None else 0.0
    K_rest = K[1:]
    K_centered = K_rest - K_rest.mean(dim=0)
    print(f"{i:>5} | {norms[0].item():9.2f} | {norms[1:].median().item():11.2f} | "
          f"{bias_norm:9.2f} | {rank99(K):>4} | {rank99(K_rest):>7} | {rank99(K_centered):>18}")