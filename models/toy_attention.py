import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def rope_cos_sin(positions, head_dim, base=10000.0):
    """Rotation angles for each position. Returns cos, sin of shape [seq, head_dim]."""
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    angles = positions.float()[:, None] * inv_freq[None, :]
    angles = torch.cat([angles, angles], dim=-1)
    return angles.cos(), angles.sin()


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(x, cos, sin):
    """x: [batch, heads, seq, head_dim]"""
    return x * cos + rotate_half(x) * sin


class ToyAttention(nn.Module):
    """Single attention layer with GQA and optional RoPE.
    n_kv_heads == n_heads -> standard multi-head attention
    n_kv_heads <  n_heads -> grouped-query attention
    """

    def __init__(self, d_model, n_heads, n_kv_heads, use_rope=True):
        super().__init__()
        assert d_model % n_heads == 0
        assert n_heads % n_kv_heads == 0
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = d_model // n_heads
        self.group = n_heads // n_kv_heads
        self.use_rope = use_rope
        self.q_proj = nn.Linear(d_model, n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(n_heads * self.head_dim, d_model, bias=False)

    def forward(self, x, cache=None):
        B, T, _ = x.shape
        start = 0 if cache is None else len(cache)

        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)

        if self.use_rope:
            pos = torch.arange(start, start + T, device=x.device)
            cos, sin = rope_cos_sin(pos, self.head_dim)
            cos, sin = cos.to(x.dtype), sin.to(x.dtype)
            q = apply_rope(q, cos, sin)
            k = apply_rope(k, cos, sin)

        if cache is not None:
            k, v = cache.update(k, v)

        # GQA: each KV head is shared by `group` query heads
        k_full = k.repeat_interleave(self.group, dim=1)
        v_full = v.repeat_interleave(self.group, dim=1)

        S = k_full.shape[2]
        scores = q @ k_full.transpose(-2, -1) / math.sqrt(self.head_dim)

        # Causal mask: a query at absolute position p only sees keys at positions <= p
        q_pos = torch.arange(start, start + T, device=x.device)[:, None]
        k_pos = torch.arange(S, device=x.device)[None, :]
        scores = scores.masked_fill(k_pos > q_pos, float("-inf"))

        attn = F.softmax(scores, dim=-1)
        out = (attn @ v_full).transpose(1, 2).reshape(B, T, -1)
        return self.o_proj(out)