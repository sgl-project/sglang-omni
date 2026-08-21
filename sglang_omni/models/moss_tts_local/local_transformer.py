# SPDX-License-Identifier: Apache-2.0
"""Native batched port of the MOSS-TTS-Local frame-local transformer."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from sglang_omni.cuda_graph import PersistentStateRegistry


def _rotate_half_interleaved(x: torch.Tensor) -> torch.Tensor:
    """Interleaved-pair rotation: [x0, x1, x2, x3, ...] -> [-x1, x0, -x3, x2, ...]."""
    even = x[..., ::2]
    odd = x[..., 1::2]
    return torch.stack((-odd, even), dim=-1).reshape_as(x)


class MossTTSLocalMLP(nn.Module):
    def __init__(self, hidden_size: int, inner_size: int) -> None:
        super().__init__()
        self.fc_in = nn.Linear(hidden_size, inner_size)
        self.fc_out = nn.Linear(inner_size, hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.fc_out(F.silu(self.fc_in(hidden_states)))


class MossTTSLocalAttention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int) -> None:
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError(
                f"hidden_size={hidden_size} not divisible by num_heads={num_heads}"
            )
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
    ) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.attn = MossTTSLocalAttention(hidden_size, num_heads)
        self.ln_2 = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.mlp = MossTTSLocalMLP(hidden_size, inner_size)


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
    ) -> None:
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.num_heads = int(num_heads)
        self.head_dim = self.hidden_size // self.num_heads
        self.max_positions = int(max_positions)
        self.h = nn.ModuleList(
            [
                MossTTSLocalBlock(hidden_size, num_heads, inner_size, layer_norm_eps)
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

        self._kv_cache: list[tuple[torch.Tensor, torch.Tensor]] = []
        self._kv_capacity = 0
        self._kv_frozen = False

    def freeze_kv_cache(self) -> None:
        """Forbid KV reallocation; captured CUDA graphs hold raw pointers
        into the current buffers, so growing them would leave the graphs
        reading freed memory."""
        self._kv_frozen = True

    def _ensure_kv_cache(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> None:
        if (
            self._kv_capacity >= batch_size
            and self._kv_cache
            and self._kv_cache[0][0].device == device
            and self._kv_cache[0][0].dtype == dtype
        ):
            return
        if self._kv_frozen:
            raise RuntimeError(
                "local-transformer KV cache is frozen after CUDA graph capture "
                f"(capacity {self._kv_capacity}, requested {batch_size})"
            )
        capacity = max(batch_size, self._kv_capacity, 1)
        shape = (capacity, self.num_heads, self.max_positions, self.head_dim)
        self._kv_cache = [
            (
                torch.empty(shape, device=device, dtype=dtype),
                torch.empty(shape, device=device, dtype=dtype),
            )
            for _ in self.h
        ]
        self._kv_capacity = capacity

    def reserve_and_freeze_kv_cache(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> None:
        """Size the KV cache for ``batch_size``, then freeze it. Single
        pre-capture entry point so the ensure-then-freeze ordering lives here
        rather than in every caller."""
        self._ensure_kv_cache(batch_size, device, dtype)
        self.freeze_kv_cache()

    def register_persistent_state(self, registry: PersistentStateRegistry) -> None:
        """Declare the frozen KV buffers, whose addresses a frame-decode
        capture bakes in."""
        for layer_idx in range(len(self._kv_cache)):
            for slot, kind in ((0, "key"), (1, "value")):
                registry.declare(
                    f"local_transformer.kv.{layer_idx}.{kind}",
                    lambda layer=layer_idx, index=slot: self._kv_cache[layer][index],
                )

    def step(self, hidden_states: torch.Tensor, position: int) -> torch.Tensor:
        """One micro-step for the whole batch."""
        if not 0 <= position < self.max_positions:
            raise ValueError(
                f"local position {position} out of range [0, {self.max_positions})"
            )
        batch_size = hidden_states.shape[0]
        self._ensure_kv_cache(batch_size, hidden_states.device, hidden_states.dtype)
        cos = self.rope_cos[position].to(dtype=hidden_states.dtype)
        sin = self.rope_sin[position].to(dtype=hidden_states.dtype)

        x = hidden_states
        for layer_idx, block in enumerate(self.h):
            normed = block.ln_1(x)
            qkv = block.attn.c_attn(normed)
            query, key, value = qkv.split(self.hidden_size, dim=-1)
            query = query.view(batch_size, self.num_heads, self.head_dim)
            key = key.view(batch_size, self.num_heads, self.head_dim)
            value = value.view(batch_size, self.num_heads, self.head_dim)
            query = query * cos + _rotate_half_interleaved(query) * sin
            key = key * cos + _rotate_half_interleaved(key) * sin

            key_cache, value_cache = self._kv_cache[layer_idx]
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
