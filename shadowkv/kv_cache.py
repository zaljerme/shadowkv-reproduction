import torch


class FullKVCache:
    """Standard KV cache. Stores keys and values for every past token.
    Tensor shape: [batch, kv_heads, seq_len, head_dim]
    """

    def __init__(self):
        self.k = None
        self.v = None

    def update(self, k_new, v_new):
        """Append new keys/values and return the full cache."""
        if self.k is None:
            self.k, self.v = k_new, v_new
        else:
            self.k = torch.cat([self.k, k_new], dim=2)
            self.v = torch.cat([self.v, v_new], dim=2)
        return self.k, self.v

    def get(self):
        return self.k, self.v

    def clear(self):
        self.k = None
        self.v = None

    def __len__(self):
        return 0 if self.k is None else self.k.shape[2]

    def nbytes(self):
        """Actual memory used by the cache in bytes."""
        if self.k is None:
            return 0
        return (self.k.numel() * self.k.element_size()
                + self.v.numel() * self.v.element_size())