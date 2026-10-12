
import math
import torch

torch.manual_seed(42)

# One KV head shared by seven query heads (like Qwen)
n_heads = 7
n_chunks = 100
head_dim = 64
budget = 16

q = torch.randn(n_heads, 1, head_dim)
landmarks = torch.randn(n_chunks, head_dim)

# Score every landmark for every query head
scores = q @ landmarks.T / math.sqrt(head_dim)
probs = torch.softmax(scores, dim=-1)

# Compare aggregation methods
max_scores = probs.max(dim=0).values.squeeze(0)
mean_scores = probs.mean(dim=0).squeeze(0)

max_idx = max_scores.topk(budget).indices
mean_idx = mean_scores.topk(budget).indices

overlap = len(set(max_idx.tolist()) & set(mean_idx.tolist()))

print("Chunks selected:", budget)
print("Overlap:", overlap)
print("Different selections:", budget - overlap)
print("Selection disagreement:", round(100 * (1 - overlap / budget), 2), "%")
