import math
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from transformers import AttentionInterface

from shadowkv.low_rank import LowRankKeys
from shadowkv.landmarks import build_landmarks, outlier_mask
from shadowkv.selector import select_chunks
from shadowkv.reconstruction import chunk_to_token_idx, reconstruct_selected
from models.toy_attention import rope_cos_sin, rope_cos_sin_from_model, apply_rope


@dataclass
class ShadowState:
    mode: str = "full"
    rank: int = 64
    budget: int = 256
    chunk: int = 8
    outlier_frac: float = 0.01
    layers: dict = field(default_factory=dict)
    rope_cos: torch.Tensor = None
    rope_sin: torch.Tensor = None


STATE = ShadowState()


def setup_rope(max_pos, head_dim, theta, model=None):
    if model is not None:
        STATE.rope_cos, STATE.rope_sin = rope_cos_sin_from_model(model, max_pos)
    else:
        STATE.rope_cos, STATE.rope_sin = rope_cos_sin(torch.arange(max_pos), head_dim, base=theta)


def build_layer_structs(cache_keys, P):
    """Build ShadowKV prefill structures for every layer.
    cache_keys: list of post-RoPE keys per layer, each [kv_heads, S, d]. Uses the first P tokens.
    """
    STATE.layers = {}
    for i, k in enumerate(cache_keys):
        k = k[:, :P].float()
        Hkv, _, d = k.shape
        # undo RoPE (rotate by the negative angle) to get pre-RoPE keys
        k_pre = apply_rope(k, STATE.rope_cos[:P], -STATE.rope_sin[:P])
        K_pre = k_pre.transpose(0, 1).reshape(P, Hkv * d)
        landmarks, min_cos, _ = build_landmarks(k, STATE.chunk)
        STATE.layers[i] = {
            "P": P,
            "lr": LowRankKeys.from_keys(K_pre, STATE.rank),
            "landmarks": landmarks,
            "out_mask": outlier_mask(min_cos, STATE.outlier_frac),
        }


def shadowkv_attention(module, query, key, value, attention_mask=None,
                       scaling=None, dropout=0.0, **kwargs):
    # query [B, heads, T, d]; key, value [B, kv_heads, S, d] (post-RoPE, cache included)
    B, H, T, d = query.shape
    Hkv = key.shape[1]
    group = H // Hkv
    scale = scaling if scaling is not None else 1 / math.sqrt(d)

    # prefill, full mode, or no structures yet: exact attention
    if T > 1 or STATE.mode == "full" or module.layer_idx not in STATE.layers:
        assert T == 1 or key.shape[2] == T, "prefill must start from an empty cache"
        k = key.repeat_interleave(group, 1)
        v = value.repeat_interleave(group, 1)
        out = F.scaled_dot_product_attention(query, k, v, is_causal=(T > 1), scale=scale)
        return out.transpose(1, 2).contiguous(), None

    assert B == 1, "decode path supports batch 1"
    L = STATE.layers[module.layer_idx]
    P = L["P"]
    q = query[0]                                   # [heads, 1, d]
    k_all, v_all = key[0], value[0]                # [kv_heads, S, d]
    k_ctx, v_ctx = k_all[:, :P], v_all[:, :P]
    k_recent, v_recent = k_all[:, P:], v_all[:, P:]
    kv = torch.arange(Hkv)[:, None, None]

    if STATE.mode == "lowrank":
        if "k_rec_all" not in L:
            k_rec = L["lr"].reconstruct().reshape(P, Hkv, d).transpose(0, 1)
            L["k_rec_all"] = apply_rope(k_rec, STATE.rope_cos[:P], STATE.rope_sin[:P]).to(q.dtype)
        k_sel, v_sel = L["k_rec_all"][:, None], v_ctx[:, None]   # [kv_heads, 1, P, d]
    else:
        c = STATE.chunk
        out_mask = L["out_mask"]
        sel = select_chunks(q, L["landmarks"], STATE.budget // c, group, exclude=out_mask)
        sel = sel | out_mask[:, None, :]
        chunk_idx = sel.nonzero()[:, -1].reshape(Hkv, 1, -1)
        tok_idx = chunk_to_token_idx(chunk_idx, c)                # [kv_heads, 1, n]
        k_sel = k_ctx[kv, tok_idx]                                # [kv_heads, 1, n, d]
        v_sel = v_ctx[kv, tok_idx]
        if STATE.mode == "shadowkv":
            is_out = out_mask[kv, chunk_idx].repeat_interleave(c, -1)
            k_rec = reconstruct_selected(L["lr"], tok_idx, STATE.rope_cos, STATE.rope_sin, d)
            k_sel = torch.where(is_out[..., None], k_sel, k_rec.to(q.dtype))

    # recent tokens are always exact
    k_sel = torch.cat([k_sel, k_recent[:, None]], dim=2)
    v_sel = torch.cat([v_sel, v_recent[:, None]], dim=2)
    k = k_sel.repeat_interleave(group, 0)                         # [heads, 1, n, d]
    v = v_sel.repeat_interleave(group, 0)
    scores = torch.einsum("hmd,hmnd->hmn", q, k) * scale
    out = torch.einsum("hmn,hmnd->hmd", torch.softmax(scores, -1), v)   # [heads, 1, d]
    return out.transpose(0, 1).unsqueeze(0).contiguous(), None           # [1, 1, heads, d]


AttentionInterface.register("shadowkv", shadowkv_attention)