import torch

from shadowkv.low_rank import LowRankKeys


def test_full_rank_is_exact():
    torch.manual_seed(0)
    K = torch.randn(200, 32)
    lr = LowRankKeys.from_keys(K, rank=32)
    assert torch.allclose(lr.reconstruct(), K, atol=1e-4)


def test_true_low_rank_matrix_recovered():
    torch.manual_seed(0)
    K = torch.randn(200, 4) @ torch.randn(4, 32)  # exactly rank 4
    lr = LowRankKeys.from_keys(K, rank=4)
    assert torch.allclose(lr.reconstruct(), K, atol=1e-4)


def test_error_decreases_with_rank():
    torch.manual_seed(0)
    K = torch.randn(200, 32)
    errs = [(LowRankKeys.from_keys(K, r).reconstruct() - K).norm().item()
            for r in [2, 4, 8, 16, 32]]
    assert all(a >= b - 1e-5 for a, b in zip(errs, errs[1:]))


def test_partial_reconstruction_matches_full():
    torch.manual_seed(0)
    K = torch.randn(200, 32)
    lr = LowRankKeys.from_keys(K, rank=8)
    idx = torch.tensor([0, 5, 17, 199])
    assert torch.allclose(lr.reconstruct(idx), lr.reconstruct()[idx], atol=1e-6)


def test_memory():
    lr = LowRankKeys.from_keys(torch.randn(1000, 128), rank=16)
    assert lr.nbytes() == (1000 * 16 + 16 * 128) * 2
    assert lr.full_nbytes() == 1000 * 128 * 2