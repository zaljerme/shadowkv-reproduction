
import math
import torch


def select_chunks(q, landmarks, n_chunks, group, exclude=None):
    """Select up to n_chunks eligible chunks per KV head and query."""
    Hkv, C, d = landmarks.shape

    L = landmarks.repeat_interleave(group, 0)

    probs = torch.softmax(
        q @ L.transpose(-1, -2) / math.sqrt(d),
        dim=-1
    )

    probs = probs.reshape(
        Hkv, group, *probs.shape[1:]
    ).amax(dim=1)

    if exclude is not None:
        probs = probs.masked_fill(exclude[:, None, :], -torch.inf)

    selected = torch.zeros_like(probs, dtype=torch.bool)

    if n_chunks <= 0:
        return selected

    k = min(n_chunks, C)
    idx = probs.topk(k, dim=-1).indices
    selected.scatter_(-1, idx, True)

    if exclude is not None:
        selected &= ~exclude[:, None, :]

    return selected
