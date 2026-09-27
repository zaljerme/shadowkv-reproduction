import math

import torch

from models.toy_attention import apply_rope


def chunk_to_token_idx(chunk_idx, chunk_size):
    """Chunk ids [..., K] -> token positions [..., K * chunk_size]."""
    offsets = torch.arange(chunk_size, device=chunk_idx.device)
    return (chunk_idx[..., None] * chunk_size + offsets).flatten(-2)


def reconstruct_selected(lr, idx, rope_cos, rope_sin, head_dim):
    """Rebuild post-RoPE keys only at the selected positions.

    lr:  LowRankKeys of pre-RoPE keys, dim = kv_heads * head_dim
    idx: [kv_heads, ..., n] token positions per KV head
    Returns [kv_heads, ..., n, head_dim].
    """
    Hkv = idx.shape[0]
    B = lr.B.reshape(lr.rank, Hkv, head_dim).permute(1, 0, 2)  # [kv_heads, r, d]
    A = lr.A[idx]                                              # [kv_heads, ..., n, r]
    shape = A.shape
    k = (A.reshape(Hkv, -1, lr.rank) @ B).reshape(*shape[:-1], head_dim)
    return apply_rope(k, rope_cos[idx], rope_sin[idx])


def sparse_attention(q, k_sel, v_sel, group):
    """Attention where each query only sees its own selected tokens.

    q:            [heads, M, d]
    k_sel, v_sel: [kv_heads, M, n, d]
    Returns [heads, M, d].
    """
    d = q.shape[-1]
    k = k_sel.repeat_interleave(group, 0)
    v = v_sel.repeat_interleave(group, 0)
    scores = torch.einsum("hmd,hmnd->hmn", q, k) / math.sqrt(d)
    p = torch.softmax(scores, -1)
    return torch.einsum("hmn,hmnd->hmd", p, v)