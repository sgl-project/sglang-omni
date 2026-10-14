# SPDX-License-Identifier: Apache-2.0
"""Native batched port of the MOSS-TTS-Local frame-local transformer."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None
    tl = None

ROTARY_CACHE_BLOCK_SIZE = 256


if triton is not None:

    @triton.jit
    def rotary_cache_kernel(
        qkv,
        cosine,
        sine,
        query,
        key_cache,
        value_cache,
        batch_size,
        position,
        hidden_size: tl.constexpr,
        head_dim: tl.constexpr,
        max_positions: tl.constexpr,
        block_size: tl.constexpr,
    ):
        index = tl.program_id(0) * block_size + tl.arange(0, block_size)
        valid = index < batch_size * hidden_size
        batch = index // hidden_size
        channel = index % hidden_size
        offset = batch * (3 * hidden_size) + channel
        neighbor = batch * (3 * hidden_size) + (channel ^ 1)
        dtype = qkv.dtype.element_ty
        cos = tl.load(cosine + channel % head_dim, valid, other=0).to(dtype)
        sin = tl.load(sine + channel % head_dim, valid, other=0).to(dtype)
        cos = cos.to(tl.float32)
        sin = sin.to(tl.float32)
        sign = tl.where(channel % 2 == 0, -1.0, 1.0)
        q = tl.load(qkv + offset, valid, other=0).to(tl.float32)
        rotated_q = tl.load(qkv + neighbor, valid, other=0).to(tl.float32) * sign
        k = tl.load(qkv + offset + hidden_size, valid, other=0).to(tl.float32)
        rotated_k = (
            tl.load(qkv + neighbor + hidden_size, valid, other=0).to(tl.float32) * sign
        )
        v = tl.load(qkv + offset + 2 * hidden_size, valid, other=0)
        # note (Zhang Yiyang): Preserve eager rounding between the RoPE products and sum.
        q = (q * cos).to(dtype).to(tl.float32) + (rotated_q * sin).to(dtype).to(
            tl.float32
        )
        k = (k * cos).to(dtype).to(tl.float32) + (rotated_k * sin).to(dtype).to(
            tl.float32
        )
        cache_offset = (
            batch * hidden_size * max_positions
            + (channel // head_dim) * max_positions * head_dim
            + position * head_dim
            + channel % head_dim
        )
        tl.store(query + index, q, valid)
        tl.store(key_cache + cache_offset, k, valid)
        tl.store(value_cache + cache_offset, v, valid)

else:
    pass


def rotate_half_interleaved(x: torch.Tensor) -> torch.Tensor:
    """Interleaved-pair rotation: [x0, x1, x2, x3, ...] -> [-x1, x0, -x3, x2, ...]."""
    even = x[..., ::2]
    odd = x[..., 1::2]
    return torch.stack((-odd, even), dim=-1).reshape_as(x)


class MossTTSLocalMLP(nn.Module):
    def __init__(
        self, hidden_size: int, inner_size: int, *, activation: str = "silu"
    ) -> None:
        super().__init__()
        self.fc_in = nn.Linear(hidden_size, inner_size)
        self.fc_out = nn.Linear(inner_size, hidden_size)
        if activation not in {"silu", "gelu_new"}:
            raise ValueError(
                f"unsupported local-transformer activation: {activation!r}"
            )
        else:
            pass
        self.activation = activation

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc_in(hidden_states)
        if self.activation == "gelu_new":
            hidden_states = F.gelu(hidden_states, approximate="tanh")
        else:
            hidden_states = F.silu(hidden_states)
        return self.fc_out(hidden_states)


class MossTTSLocalAttention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int) -> None:
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError(
                f"hidden_size={hidden_size} not divisible by num_heads={num_heads}"
            )
        else:
            pass
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.c_attn = nn.Linear(hidden_size, 3 * hidden_size)
        self.c_proj = nn.Linear(hidden_size, hidden_size)


class MossTTSLocalBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        inner_size: int,
        layer_norm_eps: float,
        activation: str,
    ) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.attn = MossTTSLocalAttention(hidden_size, num_heads)
        self.ln_2 = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.mlp = MossTTSLocalMLP(hidden_size, inner_size, activation=activation)


class MossTTSLocalTransformer(nn.Module):
    """Batched incremental decoder over <= ``max_positions`` local positions.

    Submodule names (``h.{i}.ln_1 / attn.c_attn / attn.c_proj / ln_2 /
    mlp.fc_in / mlp.fc_out`` and ``ln_f``) mirror the checkpoint layout under
    the ``local_transformer.`` prefix so weights load without remapping.
    """

    def __init__(
        self,
        *,
        hidden_size: int,
        num_heads: int,
        inner_size: int,
        num_layers: int,
        max_positions: int,
        rope_base: float,
        layer_norm_eps: float = 1e-6,
        activation: str = "silu",
    ) -> None:
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.num_heads = int(num_heads)
        self.head_dim = self.hidden_size // self.num_heads
        self.max_positions = int(max_positions)
        self.h = nn.ModuleList(
            [
                MossTTSLocalBlock(
                    hidden_size,
                    num_heads,
                    inner_size,
                    layer_norm_eps,
                    activation,
                )
                for _ in range(int(num_layers))
            ]
        )
        self.ln_f = nn.LayerNorm(hidden_size, eps=layer_norm_eps)

        inv_freq = 1.0 / (
            float(rope_base)
            ** (torch.arange(0, self.head_dim, 2, dtype=torch.float32) / self.head_dim)
        )
        positions = torch.arange(self.max_positions, dtype=torch.float32)
        freqs = torch.outer(positions, inv_freq)
        self.register_buffer(
            "rope_cos", freqs.cos().repeat_interleave(2, dim=-1), persistent=False
        )
        self.register_buffer(
            "rope_sin", freqs.sin().repeat_interleave(2, dim=-1), persistent=False
        )

        self.kv_cache: list[tuple[torch.Tensor, torch.Tensor]] = []
        self.kv_capacity = 0
        self.kv_frozen = False

    def freeze_kv_cache(self) -> None:
        """Forbid KV reallocation; captured CUDA graphs hold raw pointers
        into the current buffers, so growing them would leave the graphs
        reading freed memory."""
        self.kv_frozen = True

    def ensure_kv_cache(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> None:
        if (
            self.kv_capacity >= batch_size
            and self.kv_cache
            and self.kv_cache[0][0].device == device
            and self.kv_cache[0][0].dtype == dtype
        ):
            return
        else:
            pass
        if self.kv_frozen:
            raise RuntimeError(
                "local-transformer KV cache is frozen after CUDA graph capture "
                f"(capacity {self.kv_capacity}, requested {batch_size})"
            )
        else:
            pass
        capacity = max(batch_size, self.kv_capacity, 1)
        shape = (capacity, self.num_heads, self.max_positions, self.head_dim)
        self.kv_cache = [
            (
                torch.empty(shape, device=device, dtype=dtype),
                torch.empty(shape, device=device, dtype=dtype),
            )
            for _ in self.h
        ]
        self.kv_capacity = capacity

    def step(self, hidden_states: torch.Tensor, position: int) -> torch.Tensor:
        """One micro-step for the whole batch."""
        if not 0 <= position < self.max_positions:
            raise ValueError(
                f"local position {position} out of range [0, {self.max_positions})"
            )
        else:
            pass
        batch_size = hidden_states.shape[0]
        self.ensure_kv_cache(batch_size, hidden_states.device, hidden_states.dtype)
        cos = self.rope_cos[position]
        sin = self.rope_sin[position]

        x = hidden_states
        for layer_idx, block in enumerate(self.h):
            normed = block.ln_1(x)
            qkv = block.attn.c_attn(normed)
            key_cache, value_cache = self.kv_cache[layer_idx]
            if (
                triton is not None
                and qkv.is_cuda
                and qkv.dtype in (torch.float16, torch.bfloat16, torch.float32)
            ):
                query = torch.empty(
                    (batch_size, self.num_heads, self.head_dim),
                    device=qkv.device,
                    dtype=qkv.dtype,
                )
                grid = (
                    triton.cdiv(batch_size * self.hidden_size, ROTARY_CACHE_BLOCK_SIZE),
                )
                rotary_cache_kernel[grid](
                    qkv,
                    cos,
                    sin,
                    query,
                    key_cache,
                    value_cache,
                    batch_size,
                    position,
                    self.hidden_size,
                    self.head_dim,
                    self.max_positions,
                    ROTARY_CACHE_BLOCK_SIZE,
                    enable_fp_fusion=False,
                )
            else:
                cos = cos.to(dtype=hidden_states.dtype)
                sin = sin.to(dtype=hidden_states.dtype)
                query, key, value = qkv.split(self.hidden_size, dim=-1)
                query = query.view(batch_size, self.num_heads, self.head_dim)
                key = key.view(batch_size, self.num_heads, self.head_dim)
                value = value.view(batch_size, self.num_heads, self.head_dim)
                query = query * cos + rotate_half_interleaved(query) * sin
                key = key * cos + rotate_half_interleaved(key) * sin
                key_cache[:batch_size, :, position] = key
                value_cache[:batch_size, :, position] = value

            attn_out = F.scaled_dot_product_attention(
                query.unsqueeze(2),
                key_cache[:batch_size, :, : position + 1],
                value_cache[:batch_size, :, : position + 1],
            )
            attn_out = attn_out.squeeze(2).reshape(batch_size, self.hidden_size)
            x = x + block.attn.c_proj(attn_out)
            x = x + block.mlp(block.ln_2(x))
        return self.ln_f(x)
