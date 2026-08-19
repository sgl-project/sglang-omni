# SPDX-License-Identifier: Apache-2.0
"""Shared MOSS-Audio-Tokenizer runtime."""

from __future__ import annotations

import importlib
import logging
import math
from collections.abc import Iterator, Sequence
from contextlib import nullcontext
from dataclasses import dataclass
from itertools import accumulate
from typing import Any

import torch
import torchaudio
from torch import nn
from transformers import AutoModel

try:
    from sglang.jit_kernel.flash_attention import flash_attn_varlen_func
except ImportError:
    flash_attn_varlen_func = None

from sglang_omni.models.moss_tts.hf_loading import moss_transformers_processor_compat
from sglang_omni.models.moss_tts.vocoder_kernels import (
    apply_exact_interleaved_rope_inplace,
)

logger = logging.getLogger(__name__)

DEFAULT_MOSS_TTS_AUDIO_TOKENIZER = "OpenMOSS-Team/MOSS-Audio-Tokenizer"
_LOUDNESS_TARGET_DBFS = -20.0
_LOUDNESS_GAIN_MIN_DB = -3.0
_LOUDNESS_GAIN_MAX_DB = 3.0

# note (Zhang Yiyang): FA3 local-window attention diverges from SDPA when one
# varlen sequence spans multiple 128-token query tiles. Pack each tile as an
# independent sequence in one kernel launch.
_FA3_LOCAL_WINDOW_QUERY_TILE_SIZE = 128
_HF_FLASH_ATTENTION_2_IMPLEMENTATION = "flash_attention_2"
_PACKED_FLASH_ATTENTION_BACKEND = "packed_flash_attention"


def _single_module(source: nn.Module, singular: str, plural: str) -> nn.Module:
    module = getattr(source, singular, None)
    if module is not None:
        return module
    modules = getattr(source, plural, None)
    if modules is None or len(modules) != 1:
        raise ValueError(
            f"MOSS vocoder expects one {singular!r} module; "
            f"got {plural}={modules!r}"
        )
    return modules[0]


class _V1WeightsFeedForward(nn.Module):
    """Expose the v1-weight Linear-GELU-Linear fields as one callable module."""

    def __init__(self, source: nn.Module) -> None:
        super().__init__()
        self.linear1 = source.linear1
        self.linear2 = source.linear2
        self.activation = source.activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.activation(self.linear1(x)))


def _feed_forward(source: nn.Module) -> nn.Module:
    ffn = getattr(source, "ffn", None)
    if ffn is not None:
        return ffn
    if (
        getattr(source, "linear1", None) is None
        or getattr(source, "linear2", None) is None
    ):
        raise ValueError("MOSS vocoder transformer layer has no supported FFN")
    return _V1WeightsFeedForward(source)


@dataclass(frozen=True)
class _LocalCausalFlashPlan:
    cu_seqlens_q: torch.Tensor
    cu_seqlens_k: torch.Tensor
    kv_indices: torch.Tensor | None
    max_seqlen_q: int
    max_seqlen_k: int
    context: int


def _build_local_causal_flash_plan(
    cu_seqlens: torch.Tensor,
    *,
    context: int,
    query_chunk_size: int = _FA3_LOCAL_WINDOW_QUERY_TILE_SIZE,
    sequence_lengths: Sequence[int] | None = None,
) -> _LocalCausalFlashPlan:
    if context <= 0:
        raise ValueError(f"local causal context must be positive, got {context}")
    if query_chunk_size <= 0:
        raise ValueError(f"query_chunk_size must be positive, got {query_chunk_size}")

    if sequence_lengths is None:
        sequence_lengths = (cu_seqlens[1:] - cu_seqlens[:-1]).to("cpu").tolist()
    else:
        sequence_lengths = [int(length) for length in sequence_lengths]
        if len(sequence_lengths) != int(cu_seqlens.shape[0]) - 1:
            raise ValueError(
                "sequence_lengths must match the number of packed sequences"
            )
    q_lengths: list[int] = []
    k_lengths: list[int] = []
    key_starts: list[int] = []
    packed_offset = 0
    kv_cursor = 0
    needs_kv_gather = False
    for sequence_length in sequence_lengths:
        sequence_length = int(sequence_length)
        for query_start in range(0, sequence_length, query_chunk_size):
            query_end = min(query_start + query_chunk_size, sequence_length)
            key_start = max(0, query_start - context + 1)
            key_end = query_end
            q_lengths.append(query_end - query_start)
            k_lengths.append(key_end - key_start)
            absolute_key_start = packed_offset + key_start
            absolute_key_end = packed_offset + key_end
            key_starts.append(absolute_key_start)
            needs_kv_gather |= absolute_key_start != kv_cursor
            kv_cursor = absolute_key_end
        packed_offset += sequence_length

    if not q_lengths:
        raise ValueError("local causal flash plan requires at least one query token")

    q_lengths_tensor = torch.tensor(
        q_lengths, device=cu_seqlens.device, dtype=torch.int32
    )
    k_lengths_tensor = torch.tensor(
        k_lengths, device=cu_seqlens.device, dtype=torch.int32
    )
    cu_seqlens_q = torch.zeros(
        len(q_lengths) + 1, device=cu_seqlens.device, dtype=torch.int32
    )
    cu_seqlens_k = torch.zeros_like(cu_seqlens_q)
    cu_seqlens_q[1:] = torch.cumsum(q_lengths_tensor, dim=0)
    cu_seqlens_k[1:] = torch.cumsum(k_lengths_tensor, dim=0)
    kv_indices = None
    if needs_kv_gather or kv_cursor != packed_offset:
        packed_kv_length = sum(k_lengths)
        k_lengths_long = k_lengths_tensor.to(torch.long)
        key_starts_tensor = torch.tensor(
            key_starts, device=cu_seqlens.device, dtype=torch.long
        )
        chunk_offsets = key_starts_tensor - cu_seqlens_k[:-1].to(torch.long)
        repeated_offsets = torch.repeat_interleave(
            chunk_offsets,
            k_lengths_long,
            output_size=packed_kv_length,
        )
        kv_indices = (
            torch.arange(packed_kv_length, device=cu_seqlens.device, dtype=torch.long)
            + repeated_offsets
        )
    return _LocalCausalFlashPlan(
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        kv_indices=kv_indices,
        max_seqlen_q=max(q_lengths),
        max_seqlen_k=max(k_lengths),
        context=context,
    )


def _gather_local_flash_kv(
    x: torch.Tensor,
    kv_indices: torch.Tensor | None,
) -> torch.Tensor:
    return x if kv_indices is None else x.index_select(0, kv_indices)


class _PositionIdsCache:
    def __init__(self) -> None:
        self._items: dict[tuple[str, int | None], torch.Tensor] = {}

    def get(
        self,
        *,
        device: torch.device,
        max_seqlen: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if max_seqlen <= 0:
            raise ValueError(f"max_seqlen must be positive, got {max_seqlen}")
        key = (device.type, device.index)
        position_ids = self._items.get(key)
        if position_ids is None or position_ids.shape[0] < max_seqlen:
            position_ids = torch.arange(max_seqlen, device=device, dtype=torch.long)
            self._items[key] = position_ids
        cu_seqlens = torch.tensor([0, max_seqlen], dtype=torch.int32, device=device)
        return cu_seqlens, position_ids[:max_seqlen]


def _pack_padded_sequence(
    x: torch.Tensor,
    input_lengths: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_size, max_seqlen, _ = x.shape
    positions = torch.arange(max_seqlen, device=x.device, dtype=torch.long)
    valid_mask = positions.view(1, max_seqlen) < input_lengths.view(batch_size, 1)
    packed_x = x[valid_mask]
    cu_seqlens = torch.zeros(batch_size + 1, dtype=torch.int32, device=x.device)
    cu_seqlens[1:] = torch.cumsum(input_lengths.to(torch.int32), dim=0)
    position_ids = positions.view(1, max_seqlen).expand(batch_size, -1)[valid_mask]
    return packed_x, valid_mask, cu_seqlens, position_ids


def _pack_padded_sequence_from_host_lengths(
    x: torch.Tensor,
    input_lengths: torch.Tensor,
    input_lengths_cpu: Sequence[int],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pack with matching device/host lengths without synchronizing the device."""

    batch_size, max_seqlen, hidden_size = x.shape
    lengths = list(map(int, input_lengths_cpu))
    if len(lengths) != batch_size:
        raise ValueError("input_lengths_cpu must match the decoder batch size")
    if input_lengths.ndim != 1 or int(input_lengths.numel()) != batch_size:
        raise ValueError("input_lengths must match the decoder batch size")
    if input_lengths.device != x.device:
        raise ValueError("input_lengths and decoder inputs must share one device")
    if any(length < 0 or length > max_seqlen for length in lengths):
        raise ValueError("input_lengths_cpu must be within the padded sequence")

    total_tokens = sum(lengths)
    cu_seqlens = torch.tensor(
        [0, *accumulate(lengths)],
        device=x.device,
        dtype=torch.int32,
    )
    batch_ids = torch.repeat_interleave(
        torch.arange(batch_size, device=x.device, dtype=torch.long),
        input_lengths,
        output_size=total_tokens,
    )
    position_ids = torch.arange(total_tokens, device=x.device, dtype=torch.long)
    position_ids = position_ids - cu_seqlens[:-1].to(torch.long).index_select(
        0, batch_ids
    )
    flat_indices = batch_ids * max_seqlen + position_ids
    packed_x = x.reshape(batch_size * max_seqlen, hidden_size).index_select(
        0, flat_indices
    )
    return packed_x, flat_indices, cu_seqlens, position_ids


def _pack_unpadded_sequence(
    x: torch.Tensor,
    position_ids_cache: "_PositionIdsCache",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    assert x.shape[0] == 1, f"expected a single unpadded sequence, got {x.shape[0]}"
    _, max_seqlen, _ = x.shape
    packed_x = x.reshape(max_seqlen, x.shape[-1])
    cu_seqlens, position_ids = position_ids_cache.get(
        device=x.device,
        max_seqlen=max_seqlen,
    )
    return packed_x, cu_seqlens, position_ids


def _unpack_packed_sequence(
    packed_x: torch.Tensor,
    valid_mask: torch.Tensor,
    batch_size: int,
    max_seqlen: int,
) -> torch.Tensor:
    x = packed_x.new_zeros(batch_size, max_seqlen, packed_x.shape[-1])
    x[valid_mask] = packed_x
    return x


def _unpack_packed_sequence_from_indices(
    packed_x: torch.Tensor,
    flat_indices: torch.Tensor,
    batch_size: int,
    max_seqlen: int,
) -> torch.Tensor:
    x = packed_x.new_zeros(batch_size * max_seqlen, packed_x.shape[-1])
    x.index_copy_(0, flat_indices, packed_x)
    return x.view(batch_size, max_seqlen, packed_x.shape[-1])


def _unpack_unpadded_sequence(
    packed_x: torch.Tensor,
) -> torch.Tensor:
    return packed_x.reshape(1, packed_x.shape[0], packed_x.shape[-1])


class _MossPackedRopeCache:
    def __init__(self, *, max_period: float) -> None:
        self.max_period = float(max_period)
        self._device: torch.device | None = None
        self._head_dim = 0
        self._cos: torch.Tensor | None = None
        self._sin: torch.Tensor | None = None
        self._cos_sin: torch.Tensor | None = None

    def get(
        self,
        *,
        device: torch.device,
        head_dim: int,
        max_positions: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if max_positions <= 0:
            raise ValueError(f"max_positions must be positive, got {max_positions}")
        if head_dim <= 0 or head_dim % 2 != 0:
            raise ValueError(f"RoPE requires an even head_dim, got {head_dim}")
        if (
            self._cos is not None
            and self._sin is not None
            and self._cos_sin is not None
            and self._device == device
            and self._head_dim == head_dim
            and self._cos.shape[0] >= max_positions
        ):
            return self._cos[:max_positions], self._sin[:max_positions]

        half_dim = head_dim // 2
        ds = torch.arange(half_dim, device=device, dtype=torch.float32)
        freqs = torch.exp(ds * (-math.log(self.max_period) * 2 / head_dim))
        positions = torch.arange(
            max_positions, device=device, dtype=torch.float32
        ).view(-1, 1)
        phase = positions * freqs.view(1, -1)
        self._device = device
        self._head_dim = head_dim
        self._cos = torch.cos(phase)
        self._sin = torch.sin(phase)
        self._cos_sin = torch.cat((self._cos, self._sin), dim=-1)
        return self._cos, self._sin

    def get_cos_sin_cache(
        self,
        *,
        device: torch.device,
        head_dim: int,
        max_positions: int,
    ) -> torch.Tensor:
        self.get(
            device=device,
            head_dim=head_dim,
            max_positions=max_positions,
        )
        assert self._cos_sin is not None
        return self._cos_sin[:max_positions]


def _apply_cached_packed_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    position_ids: torch.Tensor,
    *,
    max_positions: int,
    cache: _MossPackedRopeCache,
) -> tuple[torch.Tensor, torch.Tensor]:
    if k.shape != q.shape:
        raise ValueError(
            f"Expected k.shape == q.shape, got k={tuple(k.shape)} q={tuple(q.shape)}"
        )
    if q.dim() != 3:
        raise ValueError(
            f"packed RoPE expects [tokens, heads, dim], got {tuple(q.shape)}"
        )
    _, _, head_dim = q.shape
    if head_dim <= 0 or head_dim % 2 != 0:
        raise ValueError(f"RoPE requires an even head_dim, got {head_dim}")
    if q.device.type == "cuda":
        cos_sin_cache = cache.get_cos_sin_cache(
            device=q.device,
            head_dim=head_dim,
            max_positions=max_positions,
        )
        if apply_exact_interleaved_rope_inplace(
            q,
            k,
            cos_sin_cache,
            position_ids,
        ):
            return q, k

    cos_cache, sin_cache = cache.get(
        device=q.device,
        head_dim=head_dim,
        max_positions=max_positions,
    )
    if position_ids.numel() == max_positions:
        cos = cos_cache.view(max_positions, 1, head_dim // 2)
        sin = sin_cache.view(max_positions, 1, head_dim // 2)
    else:
        cos = cos_cache.index_select(0, position_ids).view(
            position_ids.numel(), 1, head_dim // 2
        )
        sin = sin_cache.index_select(0, position_ids).view(
            position_ids.numel(), 1, head_dim // 2
        )

    dims = q.shape[:-1]
    q_pair = q.view(*dims, head_dim // 2, 2)
    k_pair = k.view(*dims, head_dim // 2, 2)
    qr, qi = q_pair[..., 0].float(), q_pair[..., 1].float()
    kr, ki = k_pair[..., 0].float(), k_pair[..., 1].float()

    qor = qr * cos - qi * sin
    qoi = qr * sin + qi * cos
    kor = kr * cos - ki * sin
    koi = kr * sin + ki * cos

    q_out = torch.stack([qor.to(q.dtype), qoi.to(q.dtype)], dim=-1).view(
        *dims, head_dim
    )
    k_out = torch.stack([kor.to(k.dtype), koi.to(k.dtype)], dim=-1).view(
        *dims, head_dim
    )
    return q_out, k_out


class MossAudioTokenizerAttention(nn.Module):
    """MOSS local-causal self attention over dense or packed decoder frames."""

    def __init__(
        self,
        source: nn.Module,
        *,
        packed_rope_cache: _MossPackedRopeCache | None = None,
    ) -> None:
        super().__init__()
        object.__setattr__(self, "source", source)
        self.in_proj = _single_module(source, "in_proj", "in_projs")
        self.out_proj = _single_module(source, "out_proj", "out_projs")
        self.embed_dim = int(source.embed_dim)
        self.num_heads = int(source.num_heads)
        self.head_dim = int(
            getattr(source, "head_dim", self.embed_dim // self.num_heads)
        )
        if self.embed_dim != self.num_heads * self.head_dim:
            raise ValueError(
                f"invalid attention shape: embed_dim={self.embed_dim}, "
                f"num_heads={self.num_heads}, head_dim={self.head_dim}"
            )
        self.causal = bool(source.causal)
        self.context = source.context
        self.rope = source.rope
        self._uses_v1_weights_attention_api = not hasattr(
            source, "resolve_attention_implementation"
        )
        self._flash_attn_varlen = flash_attn_varlen_func
        max_period = self.rope.max_period if self.rope is not None else 10000.0
        self._packed_rope_cache = packed_rope_cache or _MossPackedRopeCache(
            max_period=max_period
        )

    def resolve_attention_implementation(self, x: torch.Tensor) -> str:
        preferred = getattr(self.source, "attention_implementation", None)
        if preferred is None:
            # note (Zhang Yiyang): The MOSS-Audio-Tokenizer v1-weights attention
            # has no backend selector, but implements the same local-causal
            # operation. Prefer packed FlashAttention when its device and dtype
            # requirements are satisfied.
            preferred = _HF_FLASH_ATTENTION_2_IMPLEMENTATION
        if (
            preferred == _HF_FLASH_ATTENTION_2_IMPLEMENTATION
            and self._can_run_packed_flash(x)
        ):
            return _PACKED_FLASH_ATTENTION_BACKEND
        resolver = getattr(self.source, "resolve_attention_implementation", None)
        if resolver is None:
            return "sdpa"
        resolved = resolver(x, is_streaming=False)
        if resolved == _HF_FLASH_ATTENTION_2_IMPLEMENTATION:
            return _PACKED_FLASH_ATTENTION_BACKEND
        return resolved

    def supports_packed_flash(
        self,
        device: torch.device,
        dtype: torch.dtype | None,
    ) -> bool:
        if dtype is None or self._flash_attn_varlen is None or device.type != "cuda":
            return False
        preferred = getattr(self.source, "attention_implementation", None)
        if preferred not in (None, _HF_FLASH_ATTENTION_2_IMPLEMENTATION):
            return False
        if getattr(self.source, "_get_backend_check_dtype", None) is not None:
            # note (Zhang Yiyang): Preserve the v2/Local contract, whose packed
            # path is BF16-only.
            return dtype == torch.bfloat16
        return dtype in (torch.float16, torch.bfloat16)

    def _can_run_packed_flash(self, x: torch.Tensor) -> bool:
        dtype_resolver = getattr(self.source, "_get_backend_check_dtype", None)
        if dtype_resolver is not None:
            backend_dtype = dtype_resolver(x)
        else:
            backend_dtype = x.dtype
            try:
                autocast_enabled = torch.is_autocast_enabled("cuda")
            except TypeError:
                autocast_enabled = torch.is_autocast_enabled()
            if autocast_enabled:
                try:
                    backend_dtype = torch.get_autocast_dtype("cuda")
                except TypeError:
                    backend_dtype = torch.get_autocast_gpu_dtype()
        return self.supports_packed_flash(x.device, backend_dtype)

    def forward(
        self,
        query: torch.Tensor,
        *,
        cu_seqlens: torch.Tensor | None = None,
        max_seqlen: int | None = None,
        position_ids: torch.Tensor | None = None,
        input_lengths: torch.Tensor | None = None,
        local_flash_plan: _LocalCausalFlashPlan | None = None,
    ) -> torch.Tensor:
        backend = self.resolve_attention_implementation(query)
        if backend == _PACKED_FLASH_ATTENTION_BACKEND:
            if query.dim() != 2:
                raise ValueError(
                    "packed flash attention expects a 2D tensor, "
                    f"got {tuple(query.shape)}"
                )
            if cu_seqlens is None or max_seqlen is None or position_ids is None:
                raise ValueError(
                    "packed flash attention requires cu_seqlens, max_seqlen, "
                    "and position_ids"
                )
            return self._forward_packed_flash(
                query,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
                position_ids=position_ids,
                local_flash_plan=local_flash_plan,
            )
        if query.dim() != 3:
            raise ValueError(
                f"dense attention expects a 3D tensor, got {tuple(query.shape)}"
            )
        if input_lengths is None:
            raise ValueError("dense attention requires input_lengths")
        if self._uses_v1_weights_attention_api:
            return self.source(query, query, query)
        return self.source(query, input_lengths=input_lengths)

    def _project_qkv(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        projected = self.in_proj(x)
        if x.dim() == 3:
            projected = projected.reshape(
                x.shape[0], x.shape[1], 3, self.num_heads, self.head_dim
            ).permute(2, 0, 3, 1, 4)
            return projected[0], projected[1], projected[2]
        if x.dim() == 2:
            projected = projected.view(x.shape[0], 3, self.num_heads, self.head_dim)
            return projected[:, 0], projected[:, 1], projected[:, 2]
        raise ValueError(f"expected a 2D or 3D tensor, got {tuple(x.shape)}")

    def _apply_packed_rope(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        position_ids: torch.Tensor,
        *,
        max_positions: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.rope is None:
            return q, k
        return _apply_cached_packed_rope(
            q,
            k,
            position_ids,
            max_positions=max_positions,
            cache=self._packed_rope_cache,
        )

    def _forward_packed_flash(
        self,
        x: torch.Tensor,
        *,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
        position_ids: torch.Tensor,
        local_flash_plan: _LocalCausalFlashPlan | None,
    ) -> torch.Tensor:
        q, k, v = self._project_qkv(x)
        q, k = self._apply_packed_rope(
            q,
            k,
            position_ids,
            max_positions=max_seqlen,
        )
        assert self._flash_attn_varlen is not None
        if (
            self.causal
            and self.context is not None
            and max_seqlen > _FA3_LOCAL_WINDOW_QUERY_TILE_SIZE
        ):
            context = int(self.context)
            plan = local_flash_plan or _build_local_causal_flash_plan(
                cu_seqlens,
                context=context,
            )
            if plan.context != context:
                raise ValueError(
                    f"local flash plan context {plan.context} does not match "
                    f"attention context {context}"
                )
            packed_k = _gather_local_flash_kv(k, plan.kv_indices)
            packed_v = _gather_local_flash_kv(v, plan.kv_indices)
            out = self._flash_attn_varlen(
                q.contiguous(),
                packed_k.contiguous(),
                packed_v.contiguous(),
                plan.cu_seqlens_q,
                plan.cu_seqlens_k,
                plan.max_seqlen_q,
                plan.max_seqlen_k,
                causal=True,
                window_size=self._flash_window_size(),
            )
            return self.out_proj(out.reshape(x.shape[0], self.embed_dim))

        out = self._flash_attn_varlen(
            q.contiguous(),
            k.contiguous(),
            v.contiguous(),
            cu_seqlens,
            cu_seqlens,
            max_seqlen,
            max_seqlen,
            causal=self.causal,
            window_size=self._flash_window_size(),
        )
        return self.out_proj(out.reshape(x.shape[0], self.embed_dim))

    def _flash_window_size(self) -> tuple[int, int]:
        if self.context is None or not self.causal:
            return (-1, -1)
        # note (Zhang Yiyang): MOSS's SDPA local mask keeps `context` total tokens
        # including the current query token. FlashAttention's left-window argument
        # counts prior keys.
        return (max(int(self.context) - 1, 0), 0)


class MossAudioTokenizerTransformerLayer(nn.Module):
    """One MOSS vocoder transformer layer."""

    def __init__(
        self,
        source: nn.Module,
        *,
        packed_rope_cache: _MossPackedRopeCache | None = None,
    ) -> None:
        super().__init__()
        object.__setattr__(self, "source", source)
        self.norm1 = source.norm1
        self.norm2 = source.norm2
        self.layer_scale_1 = source.layer_scale_1
        self.layer_scale_2 = source.layer_scale_2
        self.ffn = _feed_forward(source)
        self.self_attn = MossAudioTokenizerAttention(
            source.self_attn,
            packed_rope_cache=packed_rope_cache,
        )
        assert callable(self.ffn), "MOSS vocoder transformer layer requires an FFN"

    def forward(self, x: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        residual = x
        x = self.norm1(x)
        x = residual.to(x) + self.layer_scale_1(self.self_attn(x, **kwargs))
        residual = x
        x = self.norm2(x)
        x = residual.to(x) + self.layer_scale_2(self.ffn(x))
        return x


class MossAudioTokenizerTransformer(nn.Module):
    """MOSS vocoder transformer body."""

    def __init__(self, source: nn.Module) -> None:
        super().__init__()
        object.__setattr__(self, "source", source)
        packed_rope_cache = _MossPackedRopeCache(max_period=source.max_period)
        self.layers = nn.ModuleList(
            [
                MossAudioTokenizerTransformerLayer(
                    layer,
                    packed_rope_cache=packed_rope_cache,
                )
                for layer in source.layers
            ]
        )
        self.positional_embedding = source.positional_embedding
        self.positional_scale = float(source.positional_scale)
        self.max_period = source.max_period
        self._remote_module = importlib.import_module(source.__class__.__module__)
        self._create_sin_embedding = self._remote_module.create_sin_embedding

    def resolve_attention_implementation(self, x: torch.Tensor) -> str:
        assert len(self.layers) > 0, "MOSS vocoder transformer must have layers"
        return self.layers[0].self_attn.resolve_attention_implementation(x)

    def supports_packed_flash(
        self,
        device: torch.device,
        dtype: torch.dtype | None,
    ) -> bool:
        return bool(self.layers) and all(
            layer.self_attn.supports_packed_flash(device, dtype)
            for layer in self.layers
        )

    def forward(self, x: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        if self.positional_embedding in {"sin", "sin_rope"}:
            if x.dim() == 3:
                positions = torch.arange(x.shape[1], device=x.device).view(1, -1)
            else:
                positions = kwargs.get("position_ids")
                if positions is None:
                    raise ValueError(
                        "packed transformer inputs require position_ids for "
                        "sinusoidal embeddings"
                    )
            pos_emb = self._create_sin_embedding(
                positions,
                x.shape[-1],
                max_period=self.max_period,
                dtype=x.dtype,
            )
            x = x + self.positional_scale * pos_emb
        for layer in self.layers:
            x = layer(x, **kwargs)
        return x


class MossAudioTokenizerProjectedTransformer(nn.Module):
    """Projected transformer decoder stage with the MOSS input/output layout."""

    def __init__(self, source: nn.Module) -> None:
        super().__init__()
        object.__setattr__(self, "source", source)
        self.input_proj = source.input_proj
        self.output_proj = source.output_proj
        self.transformer = MossAudioTokenizerTransformer(source.transformer)
        self._position_ids_cache = _PositionIdsCache()

    def forward(
        self,
        x: torch.Tensor,
        input_lengths: torch.Tensor,
        *,
        input_lengths_cpu: Sequence[int] | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.input_proj(x.transpose(1, 2))
        backend = self.transformer.resolve_attention_implementation(x)
        if backend == _PACKED_FLASH_ATTENTION_BACKEND:
            batch_size, max_seqlen, _ = x.shape
            if input_lengths_cpu is not None:
                if len(input_lengths_cpu) != batch_size:
                    raise ValueError(
                        "input_lengths_cpu must match the decoder batch size"
                    )
                max_valid_seqlen = max(map(int, input_lengths_cpu), default=0)
            else:
                max_valid_seqlen = int(input_lengths.max().item()) if max_seqlen else 0
            if max_valid_seqlen == 0:
                x = x.new_zeros(x.shape)
            else:
                is_unpadded_single = batch_size == 1 and max_valid_seqlen == max_seqlen
                if is_unpadded_single:
                    packed_x, cu_seqlens, position_ids = _pack_unpadded_sequence(
                        x,
                        self._position_ids_cache,
                    )
                    valid_mask = None
                    flat_indices = None
                elif input_lengths_cpu is not None:
                    packed_x, flat_indices, cu_seqlens, position_ids = (
                        _pack_padded_sequence_from_host_lengths(
                            x,
                            input_lengths,
                            input_lengths_cpu,
                        )
                    )
                    valid_mask = None
                else:
                    packed_x, valid_mask, cu_seqlens, position_ids = (
                        _pack_padded_sequence(x, input_lengths)
                    )
                    flat_indices = None
                first_attention = self.transformer.layers[0].self_attn
                local_flash_plan = None
                if (
                    first_attention.causal
                    and first_attention.context is not None
                    and max_valid_seqlen > _FA3_LOCAL_WINDOW_QUERY_TILE_SIZE
                ):
                    local_flash_plan = _build_local_causal_flash_plan(
                        cu_seqlens,
                        context=int(first_attention.context),
                        sequence_lengths=input_lengths_cpu,
                    )
                packed_x = self.transformer(
                    packed_x,
                    cu_seqlens=cu_seqlens,
                    max_seqlen=max_valid_seqlen,
                    position_ids=position_ids,
                    input_lengths=input_lengths,
                    local_flash_plan=local_flash_plan,
                    **kwargs,
                )
                if is_unpadded_single:
                    x = _unpack_unpadded_sequence(packed_x)
                elif flat_indices is not None:
                    x = _unpack_packed_sequence_from_indices(
                        packed_x,
                        flat_indices,
                        batch_size,
                        max_seqlen,
                    )
                else:
                    assert valid_mask is not None
                    x = _unpack_packed_sequence(
                        packed_x,
                        valid_mask,
                        batch_size,
                        max_seqlen,
                    )
        else:
            x = self.transformer(x, input_lengths=input_lengths, **kwargs)
        return self.output_proj(x).transpose(1, 2), input_lengths

    def supports_packed_flash(
        self,
        device: torch.device,
        dtype: torch.dtype | None,
    ) -> bool:
        return self.transformer.supports_packed_flash(device, dtype)


class MossAudioTokenizerVocoderDecoder(nn.Module):
    """Iterable MOSS vocoder decoder with patched projected transformers."""

    def __init__(self, source: nn.Module) -> None:
        super().__init__()
        source_stages = list(source)
        assert source_stages, "MOSS vocoder decoder must be a non-empty stage list"
        self.stages = nn.ModuleList(
            [self._wrap_stage(stage) for stage in source_stages]
        )

    @staticmethod
    def _wrap_stage(stage: nn.Module) -> nn.Module:
        module_type = stage.module_type
        if module_type == "Transformer":
            return MossAudioTokenizerProjectedTransformer(stage)
        if module_type == "PatchedPretransform":
            return stage
        raise ValueError(
            f"unsupported MOSS vocoder decoder stage {stage.__class__.__name__} "
            f"with module_type={module_type!r}"
        )

    def __iter__(self) -> Iterator[nn.Module]:
        return iter(self.stages)

    def __len__(self) -> int:
        return len(self.stages)

    def __getitem__(self, index: int) -> nn.Module:
        return self.stages[index]

    def supports_packed_flash(
        self,
        device: str | torch.device,
        dtype: torch.dtype | None,
    ) -> bool:
        device = torch.device(device)
        transformer_stages = [
            stage
            for stage in self.stages
            if isinstance(stage, MossAudioTokenizerProjectedTransformer)
        ]
        return bool(transformer_stages) and all(
            stage.supports_packed_flash(device, dtype) for stage in transformer_stages
        )

    @staticmethod
    def _update_cpu_lengths(
        stage: nn.Module,
        input_lengths: Sequence[int],
    ) -> list[int]:
        if isinstance(stage, MossAudioTokenizerProjectedTransformer):
            return list(map(int, input_lengths))
        patch_size = int(getattr(stage, "patch_size", 0))
        if patch_size <= 0:
            raise ValueError("MOSS patched pretransform requires patch_size > 0")
        if bool(getattr(stage, "is_downsample", False)):
            return [int(length) // patch_size for length in input_lengths]
        return [int(length) * patch_size for length in input_lengths]

    def output_lengths(self, input_lengths: Sequence[int]) -> list[int]:
        output_lengths = list(map(int, input_lengths))
        for stage in self.stages:
            output_lengths = self._update_cpu_lengths(stage, output_lengths)
        return output_lengths

    def forward(
        self,
        x: torch.Tensor,
        input_lengths: torch.Tensor,
        *,
        input_lengths_cpu: Sequence[int] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cpu_lengths = (
            None if input_lengths_cpu is None else list(map(int, input_lengths_cpu))
        )
        for stage in self.stages:
            if isinstance(stage, MossAudioTokenizerProjectedTransformer):
                x, input_lengths = stage(
                    x,
                    input_lengths,
                    input_lengths_cpu=cpu_lengths,
                )
            else:
                x, input_lengths = stage(x, input_lengths)
            if cpu_lengths is not None:
                cpu_lengths = self._update_cpu_lengths(stage, cpu_lengths)
        return x, input_lengths


def _torch_dtype(dtype: str | torch.dtype) -> torch.dtype:
    return getattr(torch, dtype) if isinstance(dtype, str) else dtype


def _model_floating_dtype(model: Any) -> torch.dtype:
    parameters = getattr(model, "parameters", None)
    if not callable(parameters):
        return torch.float32
    return next(
        (
            parameter.dtype
            for parameter in parameters()
            if parameter.is_floating_point()
        ),
        torch.float32,
    )


class MossTTSAudioTokenizer:
    """Processor-compatible wrapper around a separately loaded codec model."""

    def __init__(self, model: Any, *, device: str) -> None:
        self.model = model
        self.device = str(device)
        self.dtype = _model_floating_dtype(model)
        self.sample_rate = int(model.config.sampling_rate)

    def _autocast(self) -> Any:
        device_type = torch.device(self.device).type
        if device_type == "cuda" and self.dtype in {torch.float16, torch.bfloat16}:
            return torch.autocast(device_type=device_type, dtype=self.dtype)
        return nullcontext()

    def encode_waveforms(
        self,
        waveforms: list[tuple[torch.Tensor, int]],
        *,
        num_quantizers: int | None = None,
    ) -> list[torch.Tensor]:
        if not waveforms:
            raise ValueError("waveforms must contain at least one waveform")
        prepared = [
            self._prepare_waveform(wav, sample_rate) for wav, sample_rate in waveforms
        ]

        with torch.inference_mode(), self._autocast():
            if hasattr(self.model, "batch_encode"):
                encoded = self.model.batch_encode(
                    prepared,
                    num_quantizers=num_quantizers,
                )
            else:
                max_length = max(int(wav.shape[-1]) for wav in prepared)
                input_values = torch.zeros(
                    len(prepared),
                    1,
                    max_length,
                    device=self.device,
                    dtype=torch.float32,
                )
                padding_mask = torch.zeros(
                    len(prepared),
                    max_length,
                    device=self.device,
                    dtype=torch.bool,
                )
                for index, wav in enumerate(prepared):
                    length = int(wav.shape[-1])
                    input_values[index, 0, :length] = wav
                    padding_mask[index, :length] = True
                encoded = self.model.encode(
                    input_values,
                    padding_mask=padding_mask,
                    num_quantizers=num_quantizers,
                    return_dict=True,
                )

        audio_codes = encoded.audio_codes
        audio_codes_lengths = encoded.audio_codes_lengths
        if audio_codes is None or audio_codes_lengths is None:
            raise RuntimeError(
                "MOSS-TTS audio tokenizer encode returned empty "
                "audio_codes/audio_codes_lengths"
            )
        codes_cpu = audio_codes.detach().to(device="cpu", dtype=torch.long)
        lengths_cpu = audio_codes_lengths.detach().to("cpu")
        return [
            codes_cpu[:, index, : int(lengths_cpu[index])].transpose(0, 1).contiguous()
            for index in range(int(codes_cpu.shape[1]))
        ]

    def _prepare_waveform(self, wav: torch.Tensor, sample_rate: int) -> torch.Tensor:
        if wav.ndim == 1:
            wav = wav.unsqueeze(0)
        if wav.ndim != 2:
            raise ValueError(
                f"expected waveform with shape [channels, samples], got {tuple(wav.shape)}"
            )
        if int(wav.shape[0]) > 1:
            wav = torch.mean(wav, dim=0, keepdim=True)
        if int(sample_rate) != self.sample_rate:
            wav = torchaudio.functional.resample(
                waveform=wav,
                orig_freq=int(sample_rate),
                new_freq=self.sample_rate,
            )
        wav = self._loudness_normalize(wav.squeeze(0))
        return wav.to(device=self.device, dtype=torch.float32)

    def decode_codes(
        self,
        codes: torch.Tensor | list[torch.Tensor],
    ) -> list[torch.Tensor]:
        if isinstance(codes, torch.Tensor):
            codes = [codes]
        if not codes:
            return []

        codes_nq_t = [
            item.transpose(0, 1).contiguous().to(device=self.device, dtype=torch.long)
            for item in codes
        ]
        num_quantizers = int(codes_nq_t[0].shape[0])
        if any(int(item.shape[0]) != num_quantizers for item in codes_nq_t):
            raise ValueError("all audio-code rows must use the same quantizer count")
        max_length = max(int(item.shape[1]) for item in codes_nq_t)
        audio_codes = torch.zeros(
            num_quantizers,
            len(codes_nq_t),
            max_length,
            device=self.device,
            dtype=torch.long,
        )
        padding_mask = torch.zeros(
            len(codes_nq_t),
            max_length,
            device=self.device,
            dtype=torch.bool,
        )
        for index, item in enumerate(codes_nq_t):
            length = int(item.shape[1])
            audio_codes[:, index, :length] = item
            padding_mask[index, :length] = True

        with torch.inference_mode(), self._autocast():
            decoded = self.model.decode(
                audio_codes,
                padding_mask=padding_mask,
                return_dict=True,
                chunk_duration=8,
            )
        audio = decoded.audio
        audio_lengths = decoded.audio_lengths
        if audio is None or audio_lengths is None:
            raise RuntimeError(
                "MOSS-TTS audio tokenizer decode returned empty audio/audio_lengths"
            )
        audio_cpu = audio.detach().to(device="cpu", dtype=torch.float32)
        lengths_cpu = audio_lengths.detach().to("cpu")
        return [
            audio_cpu[index, 0, : int(lengths_cpu[index])].contiguous()
            for index in range(int(audio_cpu.shape[0]))
        ]

    @staticmethod
    def _loudness_normalize(wav: torch.Tensor) -> torch.Tensor:
        wav = wav.to(torch.float32)
        if wav.numel() == 0:
            return wav
        current_dbfs = 10.0 * torch.log10(torch.mean(wav**2) + 1e-9)
        gain = float(_LOUDNESS_TARGET_DBFS - current_dbfs)
        gain = max(_LOUDNESS_GAIN_MIN_DB, min(gain, _LOUDNESS_GAIN_MAX_DB))
        return wav * (10.0 ** (gain / 20.0))


def load_moss_tts_audio_tokenizer(
    model_path: str = DEFAULT_MOSS_TTS_AUDIO_TOKENIZER,
    *,
    device: str = "cpu",
    dtype: str | torch.dtype = "float32",
) -> MossTTSAudioTokenizer:
    logger.info(f"Loading MOSS-TTS audio tokenizer from {model_path} on {device}")
    try:
        with moss_transformers_processor_compat():
            model = AutoModel.from_pretrained(
                model_path,
                trust_remote_code=True,
            )
    except Exception as exc:
        raise RuntimeError(
            "MOSS-TTS support requires OpenMOSS-Team/MOSS-Audio-Tokenizer"
        ) from exc
    model.eval()
    move_kwargs: dict[str, Any] = {"device": device}
    if device != "cpu":
        move_kwargs["dtype"] = _torch_dtype(dtype)
    model.to(**move_kwargs)
    return MossTTSAudioTokenizer(model, device=device)
