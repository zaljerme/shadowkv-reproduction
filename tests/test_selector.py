import torch
from shadowkv.selector import select_chunks


def test_excluded_chunks_not_selected():
    q = torch.randn(1, 1, 4)
    landmarks = torch.randn(1, 4, 4)

    exclude = torch.tensor([[True, True, True, False]])

    selected = select_chunks(
        q, landmarks,
        n_chunks=3,
        group=1,
        exclude=exclude
    )

    assert not (selected & exclude[:, None, :]).any()