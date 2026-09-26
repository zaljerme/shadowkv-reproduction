import pytest
import torch

from shadowkv.kv_cache import FullKVCache
from models.toy_attention import ToyAttention

# (n_heads, n_kv_heads): single head, standard MHA, GQA
CONFIGS = [(1, 1), (4, 4), (8, 2)]


@pytest.mark.parametrize("n_heads,n_kv_heads", CONFIGS)
@pytest.mark.parametrize("seq_len", [8, 64, 256])
@pytest.mark.parametrize("batch", [1, 3])
@pytest.mark.parametrize("use_rope", [False, True])
def test_cached_matches_uncached(n_heads, n_kv_heads, seq_len, batch, use_rope):
    torch.manual_seed(0)
    d_model = 16 * n_heads  # head_dim = 16
    attn = ToyAttention(d_model, n_heads, n_kv_heads, use_rope).eval()
    x = torch.randn(batch, seq_len, d_model)

    with torch.no_grad():
        # Reference: process the whole sequence with no cache
        ref = attn(x)

        # Cached: prefill half the sequence, then decode one token at a time
        cache = FullKVCache()
        prefill = seq_len // 2
        outs = [attn(x[:, :prefill], cache)]
        for t in range(prefill, seq_len):
            outs.append(attn(x[:, t:t + 1], cache))
        out = torch.cat(outs, dim=1)

    max_diff = (out - ref).abs().max().item()
    assert len(cache) == seq_len
    assert max_diff < 1e-5, f"max diff {max_diff}"


def test_cache_size_and_clear():
    cache = FullKVCache()
    k = torch.zeros(2, 4, 10, 16)  # float32 = 4 bytes
    cache.update(k, k.clone())
    assert cache.nbytes() == 2 * (2 * 4 * 10 * 16 * 4)
    cache.clear()
    assert len(cache) == 0