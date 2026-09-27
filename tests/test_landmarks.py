import torch

from shadowkv.landmarks import build_landmarks, outlier_mask
from shadowkv.selector import select_chunks


def test_constant_chunks_have_perfect_landmarks():
    k = torch.ones(2, 16, 8)
    landmarks, min_cos, _ = build_landmarks(k, chunk_size=4)
    assert landmarks.shape == (2, 4, 8)
    assert torch.allclose(landmarks, torch.ones(2, 4, 8))
    assert torch.allclose(min_cos, torch.ones(2, 4), atol=1e-5)


def test_outlier_chunk_is_detected():
    k = torch.ones(1, 16, 8)
    k[0, 9] = -1.0  # key inside chunk 2 points the opposite way
    _, min_cos, _ = build_landmarks(k, chunk_size=4)
    mask = outlier_mask(min_cos, frac=0.25)  # 1 of 4 chunks
    assert mask[0].tolist() == [False, False, True, False]


def test_selector_picks_matching_chunk():
    d = 8
    k = torch.zeros(1, 16, d)
    for c in range(4):
        k[0, c * 4:(c + 1) * 4, c] = 1.0  # chunk c points along axis c
    landmarks, _, _ = build_landmarks(k, chunk_size=4)
    q = torch.zeros(1, 1, d)
    q[0, 0, 3] = 10.0  # query aligned with chunk 3
    sel = select_chunks(q, landmarks, n_chunks=1, group=1)
    assert sel[0, 0].tolist() == [False, False, False, True]