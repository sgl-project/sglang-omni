# SPDX-License-Identifier: Apache-2.0
"""Kernel-backed local RVQ transformer for MOSS-TTS-Realtime."""

from __future__ import annotations

from typing import Any, Literal

import torch
import torch.nn.functional as F
from torch import nn

AttentionBackend = Literal["auto", "sdpa", "fa3"]


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = float(eps)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        dtype = hidden_states.dtype
        values = hidden_states.float()
        values = values * torch.rsqrt(
            values.pow(2).mean(dim=-1, keepdim=True) + self.variance_epsilon
        )
        return self.weight * values.to(dtype)


def _rotate_half(values: torch.Tensor) -> torch.Tensor:
    first, second = values.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


class LocalAttention(nn.Module):
    def __init__(self, config: Any) -> None:
        super().__init__()
        hidden_size = int(config.hidden_size)
        self.num_heads = int(config.num_attention_heads)
        self.num_key_value_heads = int(config.num_key_value_heads)
        self.head_dim = int(getattr(config, "head_dim", hidden_size // self.num_heads))
        self.q_proj = nn.Linear(hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(
            hidden_size, self.num_key_value_heads * self.head_dim, bias=False
        )
        self.v_proj = nn.Linear(
            hidden_size, self.num_key_value_heads * self.head_dim, bias=False
        )
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, hidden_size, bias=False)
        self.register_buffer("_fused_qkv_weight", None, persistent=False)
        eps = float(config.rms_norm_eps)
        self.q_norm = RMSNorm(self.head_dim, eps)
        self.k_norm = RMSNorm(self.head_dim, eps)

    def refresh_fused_qkv(self) -> None:
        with torch.no_grad():
            self._fused_qkv_weight = torch.cat(
                (
                    self.q_proj.weight,
                    self.k_proj.weight,
                    self.v_proj.weight,
                ),
                dim=0,
            ).contiguous()

    def project(
        self, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch = int(hidden_states.shape[0])
        if self._fused_qkv_weight is not None:
            q_size = self.num_heads * self.head_dim
            kv_size = self.num_key_value_heads * self.head_dim
            query_flat, key_flat, value_flat = F.linear(
                hidden_states,
                self._fused_qkv_weight,
            ).split((q_size, kv_size, kv_size), dim=-1)
            query = self.q_norm(query_flat.view(batch, self.num_heads, self.head_dim))
            key = self.k_norm(
                key_flat.view(batch, self.num_key_value_heads, self.head_dim)
            )
            value = value_flat.view(batch, self.num_key_value_heads, self.head_dim)
            return query, key, value
        query = self.q_norm(
            self.q_proj(hidden_states).view(batch, self.num_heads, self.head_dim)
        )
        key = self.k_norm(
            self.k_proj(hidden_states).view(
                batch, self.num_key_value_heads, self.head_dim
            )
        )
        value = self.v_proj(hidden_states).view(
            batch, self.num_key_value_heads, self.head_dim
        )
        return query, key, value


class LocalMLP(nn.Module):
    def __init__(self, config: Any) -> None:
        super().__init__()
        hidden_size = int(config.hidden_size)
        intermediate_size = int(config.intermediate_size)
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.down_proj(
            F.silu(self.gate_proj(hidden_states)) * self.up_proj(hidden_states)
        )


class LocalDecoderLayer(nn.Module):
    def __init__(self, config: Any) -> None:
        super().__init__()
        self.self_attn = LocalAttention(config)
        self.mlp = LocalMLP(config)
        eps = float(config.rms_norm_eps)
        self.input_layernorm = RMSNorm(int(config.hidden_size), eps)
        self.post_attention_layernorm = RMSNorm(int(config.hidden_size), eps)


class LocalBackbone(nn.Module):
    """Incremental Qwen-style decoder over the 16 local codebook positions."""

    def __init__(
        self,
        config: Any,
        *,
        attention_backend: AttentionBackend = "auto",
    ) -> None:
        super().__init__()
        if attention_backend not in ("auto", "sdpa", "fa3"):
            raise ValueError(
                "MOSS-TTS-Realtime attention_backend must be auto, sdpa, or fa3"
            )
        self.config = config
        self.attention_backend = attention_backend
        self.hidden_size = int(config.hidden_size)
        self.num_key_value_heads = int(config.num_key_value_heads)
        self.head_dim = int(
            getattr(
                config,
                "head_dim",
                self.hidden_size // int(config.num_attention_heads),
            )
        )
        self.max_positions = int(getattr(config, "rvq", 16))
        self.embed_tokens = nn.ModuleList(
            [
                nn.Embedding(
                    int(config.audio_vocab_size),
                    self.hidden_size,
                    int(config.audio_pad_token),
                )
                for _ in range(self.max_positions - 1)
            ]
        )
        self.layers = nn.ModuleList(
            [LocalDecoderLayer(config) for _ in range(int(config.num_hidden_layers))]
        )
        self.norm = RMSNorm(self.hidden_size, float(config.rms_norm_eps))

        inv_freq = 1.0 / (
            float(config.rope_theta)
            ** (torch.arange(0, self.head_dim, 2, dtype=torch.float32) / self.head_dim)
        )
        positions = torch.arange(self.max_positions, dtype=torch.float32)
        freqs = torch.outer(positions, inv_freq)
        rope_cos_half = freqs.cos()
        rope_sin_half = freqs.sin()
        self.register_buffer("rope_cos_half", rope_cos_half, persistent=False)
        self.register_buffer("rope_sin_half", rope_sin_half, persistent=False)
        self.register_buffer(
            "rope_cos",
            torch.cat((rope_cos_half, rope_cos_half), dim=-1),
            persistent=False,
        )
        self.register_buffer(
            "rope_sin",
            torch.cat((rope_sin_half, rope_sin_half), dim=-1),
            persistent=False,
        )
        self._kv_cache: list[tuple[torch.Tensor, torch.Tensor]] = []
        self._cache_seqlens: torch.Tensor | None = None
        self._fa3_rope_cos: torch.Tensor | None = None
        self._fa3_rope_sin: torch.Tensor | None = None
        self._kv_capacity = 0
        self._kv_frozen = False
        self._resolved_attention_backend: Literal["sdpa", "fa3"] | None = None

    def _resolve_attention_backend(
        self,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Literal["sdpa", "fa3"]:
        if self.attention_backend == "sdpa":
            return "sdpa"
        supported = False
        if device.type == "cuda" and dtype in (torch.bfloat16, torch.float16):
            try:
                from sgl_kernel.flash_attn import is_fa3_supported

                with torch.cuda.device(device):
                    supported = bool(is_fa3_supported())
            except (ImportError, RuntimeError):
                supported = False
        if supported:
            return "fa3"
        if self.attention_backend == "fa3":
            raise RuntimeError(
                "MOSS-TTS-Realtime FA3 local attention is unavailable on this device"
            )
        return "sdpa"

    def _ensure_kv_cache(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> None:
        if (
            self._kv_cache
            and self._kv_capacity >= batch_size
            and self._kv_cache[0][0].device == device
            and self._kv_cache[0][0].dtype == dtype
        ):
            return
        if self._kv_frozen:
            raise RuntimeError(
                "MOSS-TTS-Realtime local KV cache cannot grow after graph capture"
            )
        backend = self._resolve_attention_backend(device, dtype)
        capacity = max(batch_size, self._kv_capacity, 1)
        if backend == "fa3":
            shape = (
                capacity,
                self.max_positions,
                self.num_key_value_heads,
                self.head_dim,
            )
        else:
            shape = (
                capacity,
                self.num_key_value_heads,
                self.max_positions,
                self.head_dim,
            )
        self._kv_cache = [
            (
                torch.empty(shape, device=device, dtype=dtype),
                torch.empty(shape, device=device, dtype=dtype),
            )
            for _ in self.layers
        ]
        self._cache_seqlens = torch.zeros(
            capacity,
            device=device,
            dtype=torch.int32,
        )
        if backend == "fa3":
            for layer in self.layers:
                layer.self_attn.refresh_fused_qkv()
            self._fa3_rope_cos = self.rope_cos_half.to(
                device=device,
                dtype=dtype,
            )
            self._fa3_rope_sin = self.rope_sin_half.to(
                device=device,
                dtype=dtype,
            )
        else:
            self._fa3_rope_cos = None
            self._fa3_rope_sin = None
        self._kv_capacity = capacity
        self._resolved_attention_backend = backend

    def freeze_kv_cache(self) -> None:
        self._kv_frozen = True

    def step(self, hidden_states: torch.Tensor, position: int) -> torch.Tensor:
        if not 0 <= position < self.max_positions:
            raise ValueError(f"local position {position} is out of range")
        batch = int(hidden_states.shape[0])
        self._ensure_kv_cache(batch, hidden_states.device, hidden_states.dtype)
        cos = None
        sin = None
        if self._resolved_attention_backend == "sdpa":
            cos = self.rope_cos[position].to(hidden_states.dtype)
            sin = self.rope_sin[position].to(hidden_states.dtype)

        values = hidden_states
        for layer_index, layer in enumerate(self.layers):
            residual = values
            query, key, value = layer.self_attn.project(layer.input_layernorm(values))
            key_cache, value_cache = self._kv_cache[layer_index]
            if self._resolved_attention_backend == "fa3":
                from sgl_kernel.flash_attn import flash_attn_with_kvcache

                if self._cache_seqlens is None:
                    raise RuntimeError(
                        "MOSS-TTS-Realtime FA3 cache lengths are not initialized"
                    )
                if self._fa3_rope_cos is None or self._fa3_rope_sin is None:
                    raise RuntimeError(
                        "MOSS-TTS-Realtime FA3 rotary tables are not initialized"
                    )
                cache_seqlens = self._cache_seqlens[:batch]
                cache_seqlens.fill_(position)
                attended = flash_attn_with_kvcache(
                    q=query.unsqueeze(1),
                    k_cache=key_cache[:batch],
                    v_cache=value_cache[:batch],
                    k=key.unsqueeze(1),
                    v=value.unsqueeze(1),
                    rotary_cos=self._fa3_rope_cos,
                    rotary_sin=self._fa3_rope_sin,
                    cache_seqlens=cache_seqlens,
                    causal=False,
                    rotary_interleaved=False,
                    num_splits=0,
                    pack_gqa=True,
                    ver=3,
                ).squeeze(1)
            else:
                if cos is None or sin is None:
                    raise RuntimeError(
                        "MOSS-TTS-Realtime SDPA rotary tables are not initialized"
                    )
                query = query * cos + _rotate_half(query) * sin
                key = key * cos + _rotate_half(key) * sin
                key_cache[:batch, :, position].copy_(key)
                value_cache[:batch, :, position].copy_(value)
                attended = F.scaled_dot_product_attention(
                    query.unsqueeze(2),
                    key_cache[:batch, :, : position + 1],
                    value_cache[:batch, :, : position + 1],
                    enable_gqa=True,
                ).squeeze(2)
            values = residual + layer.self_attn.o_proj(
                attended.reshape(batch, self.hidden_size)
            )
            values = values + layer.mlp(layer.post_attention_layernorm(values))
        return self.norm(values)


class MossTTSRealtimeLocalTransformer(nn.Module):
    """Checkpoint-compatible local decoder and its per-codebook LM heads."""

    def __init__(
        self,
        config: Any,
        *,
        attention_backend: AttentionBackend = "auto",
    ) -> None:
        super().__init__()
        self.config = config
        self.model = LocalBackbone(
            config,
            attention_backend=attention_backend,
        )
        self.local_lm_heads = nn.ModuleList(
            [
                nn.Linear(
                    int(config.hidden_size),
                    int(config.audio_vocab_size),
                    bias=False,
                )
                for _ in range(int(config.rvq))
            ]
        )

    def step(self, hidden_states: torch.Tensor, position: int) -> torch.Tensor:
        return self.model.step(hidden_states, position)

    def freeze_kv_cache(self) -> None:
        self.model.freeze_kv_cache()

    def ensure_kv_cache(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> None:
        self.model._ensure_kv_cache(batch_size, device, dtype)


__all__ = ["AttentionBackend", "MossTTSRealtimeLocalTransformer"]
