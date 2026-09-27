import torch
import torch.nn.functional as F


def build_landmarks(k, chunk_size):
    """Split post-RoPE keys into chunks and use each chunk's mean as its landmark.

    k: [kv_heads, N, d], N must be divisible by chunk_size
    Returns:
      landmarks [kv_heads, C, d]
      min_cos   [kv_heads, C]      worst cosine similarity between a key and its landmark
      cos       [kv_heads, C, c]   cosine similarity of every key to its landmark
    """
    H, N, d = k.shape
    assert N % chunk_size == 0, "context length must be divisible by chunk size"
    chunks = k.reshape(H, N // chunk_size, chunk_size, d)
    landmarks = chunks.mean(dim=2)
    cos = F.cosine_similarity(chunks, landmarks.unsqueeze(2), dim=-1)
    return landmarks, cos.min(dim=-1).values, cos


def outlier_mask(min_cos, frac):
    """Mark the chunks worst represented by their landmark (lowest min cosine).

    min_cos: [kv_heads, C]. frac: fraction of chunks to keep as outliers.
    Returns bool mask [kv_heads, C].
    """
    H, C = min_cos.shape
    n = int(round(frac * C))
    if frac > 0:
        n = max(n, 1)
    mask = torch.zeros_like(min_cos, dtype=torch.bool)
    if n > 0:
        idx = min_cos.topk(n, dim=-1, largest=False).indices
        mask.scatter_(-1, idx, True)
    return mask