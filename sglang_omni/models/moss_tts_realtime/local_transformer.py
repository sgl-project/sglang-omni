# SPDX-License-Identifier: Apache-2.0
"""Eager batched local RVQ transformer for MOSS-TTS-Realtime."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


def _rotate_half(hidden_states: torch.Tensor) -> torch.Tensor:
    first = hidden_states[..., : hidden_states.shape[-1] // 2]
    second = hidden_states[..., hidden_states.shape[-1] // 2 :]
    return torch.cat((-second, first), dim=-1)


def _repeat_kv(hidden_states: torch.Tensor, repeats: int) -> torch.Tensor:
    if repeats == 1:
        return hidden_states
    batch, kv_heads, sequence, head_dim = hidden_states.shape
    expanded = hidden_states[:, :, None, :, :].expand(
        batch,
        kv_heads,
        repeats,
        sequence,
        head_dim,
    )
    return expanded.reshape(batch, kv_heads * repeats, sequence, head_dim)


class MossTTSRealtimeRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, *, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = float(eps)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        work = hidden_states.to(torch.float32)
        variance = work.square().mean(dim=-1, keepdim=True)
        normalized = work * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * normalized.to(input_dtype)


class MossTTSRealtimeLocalMLP(nn.Module):
    def __init__(self, config: Any) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(
            config.hidden_size,
            config.intermediate_size,
            bias=False,
        )
        self.up_proj = nn.Linear(
            config.hidden_size,
            config.intermediate_size,
            bias=False,
        )
        self.down_proj = nn.Linear(
            config.intermediate_size,
            config.hidden_size,
            bias=False,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.down_proj(
            F.silu(self.gate_proj(hidden_states)) * self.up_proj(hidden_states)
        )


class MossTTSRealtimeLocalAttention(nn.Module):
    def __init__(self, config: Any) -> None:
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = (
            config.num_attention_heads // config.num_key_value_heads
        )
        self.head_dim = config.head_dim
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout

        self.q_proj = nn.Linear(
            config.hidden_size,
            config.num_attention_heads * config.head_dim,
            bias=False,
        )
        self.k_proj = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * config.head_dim,
            bias=False,
        )
        self.v_proj = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * config.head_dim,
            bias=False,
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * config.head_dim,
            config.hidden_size,
            bias=False,
        )
        self.q_norm = MossTTSRealtimeRMSNorm(
            config.head_dim,
            eps=config.rms_norm_eps,
        )
        self.k_norm = MossTTSRealtimeRMSNorm(
            config.head_dim,
            eps=config.rms_norm_eps,
        )

    def step(
        self,
        hidden_states: torch.Tensor,
        *,
        position: int,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = hidden_states.shape[0]
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        query = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(
            1, 2
        )
        key = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos = rope_cos[position].to(
            device=hidden_states.device,
            dtype=query.dtype,
        )[None, None, None, :]
        sin = rope_sin[position].to(
            device=hidden_states.device,
            dtype=query.dtype,
        )[None, None, None, :]
        query = query * cos + _rotate_half(query) * sin
        key = key * cos + _rotate_half(key) * sin

        key_cache[:batch_size, :, position].copy_(key[:, :, 0])
        value_cache[:batch_size, :, position].copy_(value[:, :, 0])
        keys = _repeat_kv(
            key_cache[:batch_size],
            self.num_key_value_groups,
        )
        values = _repeat_kv(
            value_cache[:batch_size],
            self.num_key_value_groups,
        )
        attention_mask = (
            torch.arange(key_cache.shape[2], device=hidden_states.device) <= position
        ).view(1, 1, 1, -1)
        attention_mask = attention_mask.expand(batch_size, 1, 1, -1)
        attention = F.scaled_dot_product_attention(
            query,
            keys,
            values,
            attn_mask=attention_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
            scale=self.scaling,
        )
        attention = attention.transpose(1, 2).contiguous()
        attention = attention.reshape(*input_shape, -1)
        return self.o_proj(attention)


class MossTTSRealtimeLocalDecoderLayer(nn.Module):
    def __init__(self, config: Any) -> None:
        super().__init__()
        self.self_attn = MossTTSRealtimeLocalAttention(config)
        self.mlp = MossTTSRealtimeLocalMLP(config)
        self.input_layernorm = MossTTSRealtimeRMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )
        self.post_attention_layernorm = MossTTSRealtimeRMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )

    def step(
        self,
        hidden_states: torch.Tensor,
        *,
        position: int,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn.step(
            hidden_states,
            position=position,
            key_cache=key_cache,
            value_cache=value_cache,
            rope_cos=rope_cos,
            rope_sin=rope_sin,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return residual + hidden_states


class MossTTSRealtimeLocalTransformerModel(nn.Module):
    """Frame-local transformer with a lazily sized batched scratch KV cache."""

    def __init__(self, config: Any) -> None:
        super().__init__()
        config.validate()
        self.config = config
        self.embed_tokens = nn.ModuleList(
            [
                nn.Embedding(
                    config.audio_vocab_size,
                    config.hidden_size,
                    padding_idx=config.audio_pad_token,
                )
                for _ in range(config.rvq - 1)
            ]
        )
        self.layers = nn.ModuleList(
            [
                MossTTSRealtimeLocalDecoderLayer(config)
                for _ in range(config.num_hidden_layers)
            ]
        )
        self.norm = MossTTSRealtimeRMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )

        inverse_frequency = 1.0 / (
            config.rope_theta
            ** (
                torch.arange(0, config.head_dim, 2, dtype=torch.float32)
                / config.head_dim
            )
        )
        positions = torch.arange(
            config.max_position_embeddings,
            dtype=torch.float32,
        )
        frequencies = torch.outer(positions, inverse_frequency)
        rotary = torch.cat((frequencies, frequencies), dim=-1)
        self.register_buffer("rope_cos", rotary.cos(), persistent=False)
        self.register_buffer("rope_sin", rotary.sin(), persistent=False)

        self._kv_cache: list[tuple[torch.Tensor, torch.Tensor]] = []
        self._kv_capacity = 0
        self._kv_frozen = False

    def freeze_kv_cache(self) -> None:
        """Prevent pointer-invalidating growth after local graph capture."""

        self._kv_frozen = True

    def _ensure_kv_cache(
        self,
        batch_size: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
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
        shape = (
            capacity,
            self.config.num_key_value_heads,
            self.config.rvq,
            self.config.head_dim,
        )
        self._kv_cache = [
            (
                torch.zeros(shape, device=device, dtype=dtype),
                torch.zeros(shape, device=device, dtype=dtype),
            )
            for _ in self.layers
        ]
        self._kv_capacity = capacity

    def step(self, hidden_states: torch.Tensor, position: int) -> torch.Tensor:
        if hidden_states.ndim != 2:
            raise ValueError("local hidden_states must have shape [batch, hidden_size]")
        if hidden_states.shape[1] != self.config.hidden_size:
            raise ValueError(
                f"local hidden size must be {self.config.hidden_size}, got "
                f"{hidden_states.shape[1]}"
            )
        if not 0 <= position < self.config.rvq:
            raise ValueError(
                f"local position must be in [0, {self.config.rvq}), got {position}"
            )

        batch_size = int(hidden_states.shape[0])
        self._ensure_kv_cache(
            batch_size,
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )
        if position == 0:
            for key_cache, value_cache in self._kv_cache:
                key_cache[:batch_size].zero_()
                value_cache[:batch_size].zero_()
        current = hidden_states.unsqueeze(1)
        for layer_index, layer in enumerate(self.layers):
            key_cache, value_cache = self._kv_cache[layer_index]
            current = layer.step(
                current,
                position=position,
                key_cache=key_cache,
                value_cache=value_cache,
                rope_cos=self.rope_cos,
                rope_sin=self.rope_sin,
            )
        return self.norm(current).squeeze(1)


class MossTTSRealtimeLocalTransformerForCausalLM(nn.Module):
    """Local transformer plus one projection head per RVQ codebook."""

    def __init__(self, config: Any) -> None:
        super().__init__()
        self.config = config
        self.model = MossTTSRealtimeLocalTransformerModel(config)
        self.local_lm_heads = nn.ModuleList(
            [
                nn.Linear(
                    config.hidden_size,
                    config.audio_vocab_size,
                    bias=False,
                )
                for _ in range(config.rvq)
            ]
        )

    @property
    def dtype(self) -> torch.dtype:
        return self.local_lm_heads[0].weight.dtype

    @property
    def device(self) -> torch.device:
        return self.local_lm_heads[0].weight.device

    @torch.no_grad()
    def teacher_forced_logits(
        self,
        backbone_hidden_states: torch.Tensor,
        prefix_codes: torch.Tensor,
    ) -> torch.Tensor:
        """Return all 16 logits using codebooks 0..14 as the forced prefix."""

        if prefix_codes.ndim != 2:
            raise ValueError("prefix_codes must have shape [batch, 15]")
        if prefix_codes.shape[0] != backbone_hidden_states.shape[0]:
            raise ValueError("prefix_codes batch size must match hidden states")
        if prefix_codes.shape[1] < self.config.rvq - 1:
            raise ValueError(
                f"prefix_codes must provide at least {self.config.rvq - 1} columns"
            )
        if (
            prefix_codes.dtype == torch.bool
            or torch.is_floating_point(prefix_codes)
            or torch.is_complex(prefix_codes)
        ):
            raise TypeError("prefix_codes must be an integer tensor")
        if torch.any(prefix_codes < 0) or torch.any(
            prefix_codes >= self.config.audio_vocab_size
        ):
            raise ValueError("prefix_codes contain an out-of-range audio id")

        prefix_codes = prefix_codes.to(device=self.device, dtype=torch.long)
        current = backbone_hidden_states.to(device=self.device, dtype=self.dtype)
        logits: list[torch.Tensor] = []
        for codebook in range(self.config.rvq):
            local_hidden = self.model.step(current, codebook)
            logits.append(self.local_lm_heads[codebook](local_hidden))
            if codebook + 1 < self.config.rvq:
                current = self.model.embed_tokens[codebook](prefix_codes[:, codebook])
        return torch.stack(logits, dim=1)

    @torch.no_grad()
    def decode_frame(
        self,
        backbone_hidden_states: torch.Tensor,
        *,
        sample_audio: Callable[[torch.Tensor, int], torch.Tensor],
        compute_logits: Callable[[torch.Tensor, int], torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Decode one `[batch, 16]` frame using a caller-owned sampler."""

        current = backbone_hidden_states.to(device=self.device, dtype=self.dtype)
        codes: list[torch.Tensor] = []
        for codebook in range(self.config.rvq):
            if compute_logits is None:
                local_hidden = self.model.step(current, codebook)
                logits = self.local_lm_heads[codebook](local_hidden)
            else:
                logits = compute_logits(current, codebook)
            code = sample_audio(logits, codebook)
            if not isinstance(code, torch.Tensor):
                raise TypeError("sample_audio must return a tensor")
            if code.ndim != 1 or code.shape[0] != current.shape[0]:
                raise ValueError("sample_audio must return shape [batch]")
            if (
                code.dtype == torch.bool
                or torch.is_floating_point(code)
                or torch.is_complex(code)
            ):
                raise TypeError("sample_audio must return integer token ids")
            code = code.to(device=self.device, dtype=torch.long)
            if torch.any(code < 0) or torch.any(code >= self.config.audio_vocab_size):
                raise ValueError("sample_audio returned an out-of-range audio id")
            codes.append(code)
            if codebook + 1 < self.config.rvq:
                current = self.model.embed_tokens[codebook](code)
        return torch.stack(codes, dim=-1)
