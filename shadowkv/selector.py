import math

import torch


def select_chunks(q, landmarks, n_chunks, group, exclude=None):
    """Pick the chunks most likely to matter for each query, using landmarks.

    q:         [heads, M, d] post-RoPE queries
    landmarks: [kv_heads, C, d]
    group:     query heads per KV head (GQA)
    exclude:   optional bool [kv_heads, C], chunks that are never picked here (e.g. outliers)
    Returns bool mask [kv_heads, M, C].

    Each query head scores landmarks with softmax. Heads sharing a KV head
    are combined with max, since they must share the same fetched chunks.
    """
    Hkv, C, d = landmarks.shape
    L = landmarks.repeat_interleave(group, 0)
    probs = torch.softmax(q @ L.transpose(-1, -2) / math.sqrt(d), dim=-1)  # [heads, M, C]
    probs = probs.reshape(Hkv, group, *probs.shape[1:]).amax(1)            # [kv_heads, M, C]
    if exclude is not None:
        probs = probs.masked_fill(exclude[:, None, :], -1.0)
    idx = probs.topk(min(n_chunks, C), dim=-1).indices
    return torch.zeros_like(probs, dtype=torch.bool).scatter_(-1, idx, True)