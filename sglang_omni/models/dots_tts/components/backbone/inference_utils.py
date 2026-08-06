from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from sglang_omni.models.dots_tts.components.backbone.layers import rotate_half


def build_fused_qkv_projection(attn: nn.Module) -> nn.Linear:
    hidden_size = int(attn.num_heads) * int(attn.head_dim)
    has_bias = attn.q_proj.bias is not None
    projs = (attn.q_proj, attn.k_proj, attn.v_proj)
    qkv_proj = nn.Linear(hidden_size, 3 * hidden_size, bias=has_bias)
    qkv_proj.to(device=attn.q_proj.weight.device, dtype=attn.q_proj.weight.dtype)
    with torch.no_grad():
        qkv_proj.weight.copy_(torch.cat([proj.weight for proj in projs], dim=0))
        if has_bias:
            qkv_proj.bias.copy_(torch.cat([proj.bias for proj in projs], dim=0))
    return qkv_proj


def fuse_qkv_projection(attn: nn.Module) -> None:
    attn.qkv_proj = build_fused_qkv_projection(attn)


def project_attention(
    attn: nn.Module,
    x: torch.Tensor,
    *,
    num_heads: int,
    head_dim: int,
    rotary_cos: torch.Tensor | None = None,
    rotary_sin: torch.Tensor | None = None,
    key_prefix: torch.Tensor | None = None,
    value_prefix: torch.Tensor | None = None,
    kv_buffer: tuple[torch.Tensor, torch.Tensor] | None = None,
    kv_prefix_len: int = 0,
    attn_mask: torch.Tensor | None = None,
    is_causal: bool = False,
    dropout_p: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch, seq_len, _hidden = x.shape
    num_heads = int(num_heads)
    head_dim = int(head_dim)
    qkv = attn.qkv_proj(x).view(batch, seq_len, 3, num_heads, head_dim)
    q, k, v = qkv.permute(2, 0, 3, 1, 4)

    q = attn.q_norm(q)
    k = attn.k_norm(k)
    if rotary_cos is not None:
        cos, sin = rotary_cos, rotary_sin
        if cos.dim() == 2:
            cos = cos.unsqueeze(0).unsqueeze(0)
            sin = sin.unsqueeze(0).unsqueeze(0)
        elif cos.dim() == 3:
            cos = cos.unsqueeze(1)
            sin = sin.unsqueeze(1)
        q = (q * cos + rotate_half(q) * sin).to(dtype=q.dtype)
        k = (k * cos + rotate_half(k) * sin).to(dtype=k.dtype)

    if kv_buffer is not None:
        # note (chenyang): the caller owns a buffer whose first kv_prefix_len
        # positions already hold the cached keys/values. Writing this step's
        # k/v straight after them keeps the attention inputs allocation-free,
        # which is what keeps engine memory flat across a whole generation.
        key_buffer, value_buffer = kv_buffer
        end = int(kv_prefix_len) + seq_len
        key_buffer[:, :, int(kv_prefix_len) : end, :] = k
        value_buffer[:, :, int(kv_prefix_len) : end, :] = v
        key = key_buffer[:, :, :end, :]
        value = value_buffer[:, :, :end, :]
    elif key_prefix is not None or value_prefix is not None:
        if key_prefix is None or value_prefix is None:
            raise ValueError("key_prefix and value_prefix must be provided together.")
        key = torch.cat([key_prefix, k], dim=2)
        value = torch.cat([value_prefix, v], dim=2)
    else:
        key, value = k, v

    if dropout_p is None:
        dropout_p = attn.attn_drop.p if attn.training else 0.0
    out = F.scaled_dot_product_attention(
        q,
        key,
        value,
        attn_mask=attn_mask,
        dropout_p=float(dropout_p),
        is_causal=bool(is_causal),
    )

    out = out.permute(0, 2, 1, 3).reshape(batch, seq_len, -1)
    return attn.o_dropout(attn.o_proj(out)), k, v
