import torch

from shadowkv.low_rank import LowRankKeys
from shadowkv.reconstruction import chunk_to_token_idx, reconstruct_selected, sparse_attention
from models.toy_attention import rope_cos_sin, apply_rope


def test_selected_reconstruction_matches_full():
    torch.manual_seed(0)
    T, Hkv, d, r = 64, 2, 8, 6
    K = torch.randn(T, Hkv * d)
    lr = LowRankKeys.from_keys(K, r)
    cos, sin = rope_cos_sin(torch.arange(T), d)
    full = apply_rope(lr.reconstruct().reshape(T, Hkv, d).transpose(0, 1), cos, sin)
    idx = torch.randint(0, T, (Hkv, 3, 5))
    part = reconstruct_selected(lr, idx, cos, sin, d)
    expected = full[torch.arange(Hkv)[:, None, None], idx]
    assert torch.allclose(part, expected, atol=1e-5)


def test_sparse_attention_with_all_tokens_equals_dense():
    torch.manual_seed(0)
    H, Hkv, M, N, d = 4, 2, 3, 10, 8
    group = H // Hkv
    q = torch.randn(H, M, d)
    k = torch.randn(Hkv, N, d)
    v = torch.randn(Hkv, N, d)
    idx = torch.arange(N).expand(Hkv, M, N)
    kv = torch.arange(Hkv)[:, None, None]
    out = sparse_attention(q, k[kv, idx], v[kv, idx], group)
    kr, vr = k.repeat_interleave(group, 0), v.repeat_interleave(group, 0)
    dense = torch.softmax(q @ kr.transpose(-1, -2) / d ** 0.5, -1) @ vr
    assert torch.allclose(out, dense, atol=1e-5)


def test_chunk_to_token_idx():
    idx = chunk_to_token_idx(torch.tensor([[0, 2]]), 4)
    assert idx.tolist() == [[0, 1, 2, 3, 8, 9, 10, 11]]




def test_inverse_rope_roundtrip():
    torch.manual_seed(0)
    x = torch.randn(2, 16, 8)
    cos, sin = rope_cos_sin(torch.arange(16), 8)
    y = apply_rope(x, cos, sin)
    assert torch.allclose(apply_rope(y, cos, -sin), x, atol=1e-5)