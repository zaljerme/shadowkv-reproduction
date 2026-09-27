import torch


class LowRankKeys:
    """Low-rank key cache for one layer.

    Stores keys K [tokens, dim] as A @ B:
      A = U_r * S_r   [tokens, r]
      B = V_r^T       [r, dim]
    dim = kv_heads * head_dim (all KV heads of a layer together, like ShadowKV).
    """

    def __init__(self, A, B):
        self.A = A
        self.B = B
        self.rank = A.shape[1]

    @classmethod
    def from_keys(cls, K, rank, factors=None):
        """Compress K to the given rank. Pass precomputed SVD factors to avoid recomputing."""
        if factors is None:
            factors = torch.linalg.svd(K.float(), full_matrices=False)
        U, S, Vh = factors
        rank = min(rank, S.shape[0])
        return cls(U[:, :rank] * S[:rank], Vh[:rank])

    def reconstruct(self, idx=None):
        """Rebuild all keys, or only the rows in idx."""
        A = self.A if idx is None else self.A[idx]
        return A @ self.B

    def nbytes(self, bytes_per_elem=2):
        return (self.A.numel() + self.B.numel()) * bytes_per_elem

    def full_nbytes(self, bytes_per_elem=2):
        return self.A.shape[0] * self.B.shape[1] * bytes_per_elem