# SPDX-License-Identifier: Apache-2.0
"""Attention backends and packed execution for MOSS-Audio-Tokenizer."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import cache
from itertools import accumulate

import torch
import torch.nn.functional as F
from sglang.kernels.ops.attention.flash_attention import flash_attn_varlen_func
from sglang.kernels.ops.attention.flash_attention_v3 import _is_fa3_supported
from torch import nn

from sglang_omni.models.moss_tts.vocoder_kernels import (
    apply_exact_interleaved_rope_inplace,
    can_fuse_streaming_kv,
    commit_streaming_kv_,
    gather_streaming_kv,
)

# note (Zhang Yiyang): Bound SDPA fallback memory with query chunks.
_SDPA_QUERY_CHUNK_SIZE = 512
# note (Zhang Yiyang): Keep 128-token local-causal tiles for A800 audio stability.
_LOCAL_CAUSAL_FLASH_QUERY_CHUNK_SIZE = 128
# note (Zhang Yiyang): Use the direct local-window kernel on Hopper and newer SMs.
_PACKED_FLASH_DIRECT_MIN_SM = 90
PACKED_FLASH_ATTENTION_BACKEND = "packed_flash_attention"
SDPA_ATTENTION_BACKEND = "sdpa"
AUTO_ATTENTION_BACKEND = "auto"
_SUPPORTED_ATTENTION_BACKENDS: frozenset[str] = frozenset(
    {
        AUTO_ATTENTION_BACKEND,
        PACKED_FLASH_ATTENTION_BACKEND,
        SDPA_ATTENTION_BACKEND,
    }
)


def validate_attention_backend(attention_backend: str) -> str:
    if attention_backend not in _SUPPORTED_ATTENTION_BACKENDS:
        raise ValueError(
            "attention_backend must be 'auto', 'packed_flash_attention', "
            "or 'sdpa'; "
            f"got {attention_backend!r}"
        )
    return attention_backend


def _packed_flash_device_unavailable_reason(device: torch.device) -> str | None:
    device = torch.device(device)
    if device.type != "cuda":
        return f"device type {device.type!r} is not CUDA"
    if not torch.cuda.is_available():
        return "the CUDA runtime is unavailable"
    if not _is_fa3_supported():
        return "SGLang packed FlashAttention (FA3) is unavailable"
    return None


@cache
def _packed_flash_requires_query_chunks(device: torch.device) -> bool:
    device = torch.device(device)
    if device.type != "cuda" or not torch.cuda.is_available():
        return True
    major, minor = torch.cuda.get_device_capability(device)
    sm = major * 10 + minor
    return sm < _PACKED_FLASH_DIRECT_MIN_SM


@dataclass(frozen=True)
class AttentionBackendResolution:
    backend: str
    fallback_reason: str | None = None


@dataclass(frozen=True)
class _LocalCausalFlashPlan:
    cu_seqlens_q: torch.Tensor
    cu_seqlens_k: torch.Tensor
    kv_indices: torch.Tensor | None
    max_seqlen_q: int
    max_seqlen_k: int
    context: int


def merge_attention_backend_resolutions(
    resolutions: Sequence[AttentionBackendResolution],
) -> AttentionBackendResolution:
    if not resolutions:
        return AttentionBackendResolution(SDPA_ATTENTION_BACKEND)
    backends = {resolution.backend for resolution in resolutions}
    if len(backends) != 1:
        raise RuntimeError(
            "MOSS-Audio-Tokenizer Transformer layers resolved different "
            f"attention backends: {sorted(backends)}"
        )
    fallback_reasons = list(
        dict.fromkeys(
            resolution.fallback_reason
            for resolution in resolutions
            if resolution.fallback_reason is not None
        )
    )
    return AttentionBackendResolution(
        resolutions[0].backend,
        fallback_reason="; ".join(fallback_reasons) or None,
    )


def _build_local_causal_flash_plan(
    cu_seqlens: torch.Tensor,
    *,
    context: int,
    query_chunk_size: int = _LOCAL_CAUSAL_FLASH_QUERY_CHUNK_SIZE,
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
        q_lengths,
        device=cu_seqlens.device,
        dtype=torch.int32,
    )
    k_lengths_tensor = torch.tensor(
        k_lengths,
        device=cu_seqlens.device,
        dtype=torch.int32,
    )
    cu_seqlens_q = torch.zeros(
        len(q_lengths) + 1,
        device=cu_seqlens.device,
        dtype=torch.int32,
    )
    cu_seqlens_k = torch.zeros_like(cu_seqlens_q)
    cu_seqlens_q[1:] = torch.cumsum(q_lengths_tensor, dim=0)
    cu_seqlens_k[1:] = torch.cumsum(k_lengths_tensor, dim=0)
    kv_indices = None
    if needs_kv_gather or kv_cursor != packed_offset:
        packed_kv_length = sum(k_lengths)
        k_lengths_long = k_lengths_tensor.to(torch.long)
        key_starts_tensor = torch.tensor(
            key_starts,
            device=cu_seqlens.device,
            dtype=torch.long,
        )
        chunk_offsets = key_starts_tensor - cu_seqlens_k[:-1].to(torch.long)
        repeated_offsets = torch.repeat_interleave(
            chunk_offsets,
            k_lengths_long,
            output_size=packed_kv_length,
        )
        kv_indices = (
            torch.arange(
                packed_kv_length,
                device=cu_seqlens.device,
                dtype=torch.long,
            )
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


@dataclass(frozen=True)
class StreamingExecutionContext:
    """Map execution rows to persistent slots; padding rows do not advance state."""

    state_slot_ids: torch.Tensor
    valid_rows: torch.Tensor


@dataclass
class MossAudioTokenizerStreamingState:
    """Persistent per-module state used by the repository-owned codec."""

    batch_size: int
    device: torch.device
    exec_mask: torch.Tensor = field(init=False)

    def __post_init__(self) -> None:
        self.exec_mask = torch.ones(
            self.batch_size,
            device=self.device,
            dtype=torch.bool,
        )

    def set_exec_mask(self, exec_mask: torch.Tensor) -> None:
        if exec_mask.shape != (self.batch_size,):
            raise ValueError(
                "streaming exec_mask must have shape "
                f"({self.batch_size},), got {tuple(exec_mask.shape)}"
            )
        self.exec_mask.copy_(exec_mask.to(device=self.device, dtype=torch.bool))

    def reset_slots(self, state_slot_ids: torch.Tensor) -> None:
        self.exec_mask.index_fill_(0, state_slot_ids, True)


class MossAudioTokenizerStreamingModule(nn.Module):
    """Small lifecycle facade shared by codec modules with causal state."""

    def __init__(self) -> None:
        super().__init__()
        self._streaming_state: MossAudioTokenizerStreamingState | None = None

    @property
    def is_streaming(self) -> bool:
        return self._streaming_state is not None

    def _streaming_device(self) -> torch.device:
        parameter = next(self.parameters(), None)
        if parameter is not None:
            return parameter.device
        buffer = next(self.buffers(), None)
        if buffer is not None:
            return buffer.device
        return torch.device("cpu")

    def _init_streaming_state(
        self, batch_size: int
    ) -> MossAudioTokenizerStreamingState:
        return MossAudioTokenizerStreamingState(
            int(batch_size), self._streaming_device()
        )

    @contextmanager
    def streaming(self, batch_size: int = 1):
        if batch_size <= 0:
            raise ValueError(f"streaming batch_size must be positive, got {batch_size}")
        modules = [
            module
            for module in self.modules()
            if isinstance(module, MossAudioTokenizerStreamingModule)
        ]
        if any(module._streaming_state is not None for module in modules):
            raise RuntimeError("MOSS-Audio-Tokenizer is already streaming")
        initialized: list[MossAudioTokenizerStreamingModule] = []
        try:
            for module in modules:
                module._streaming_state = module._init_streaming_state(batch_size)
                initialized.append(module)
            yield
        finally:
            for module in reversed(initialized):
                module._streaming_state = None

    def _set_streaming_exec_mask(self, exec_mask: torch.Tensor) -> None:
        states = [
            module._streaming_state
            for module in self.modules()
            if isinstance(module, MossAudioTokenizerStreamingModule)
            and module._streaming_state is not None
        ]
        if not states:
            raise RuntimeError("MOSS-Audio-Tokenizer is not streaming")
        for state in states:
            state.set_exec_mask(exec_mask)


@dataclass
class _AttentionStreamingState(MossAudioTokenizerStreamingState):
    cached_keys: torch.Tensor | None
    cached_values: torch.Tensor | None
    cached_positions: torch.Tensor | None
    offset: torch.Tensor

    def reset_slots(self, state_slot_ids: torch.Tensor) -> None:
        """Reset only the persistent rows addressed by ``state_slot_ids``."""
        if state_slot_ids.numel() == 0:
            return
        slots = state_slot_ids.to(device=self.device, dtype=torch.long)
        self.offset.index_fill_(0, slots, 0)
        super().reset_slots(slots)
        if self.cached_positions is not None:
            self.cached_positions.index_fill_(0, slots, -1)


def _single_module(module: nn.Module, singular: str, plural: str) -> nn.Module:
    child = getattr(module, singular, None)
    if child is not None:
        return child
    modules = getattr(module, plural, None)
    if modules is None or len(modules) != 1:
        raise ValueError(
            f"MOSS-Audio-Tokenizer expects one {singular!r} module; "
            f"got {plural}={modules!r}"
        )
    return modules[0]


class PositionIdsCache:
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


def pack_padded_sequence(
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


def pack_padded_sequence_from_host_lengths(
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


def pack_unpadded_sequence(
    x: torch.Tensor,
    position_ids_cache: "PositionIdsCache",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    assert x.shape[0] == 1, f"expected a single unpadded sequence, got {x.shape[0]}"
    _, max_seqlen, _ = x.shape
    packed_x = x.reshape(max_seqlen, x.shape[-1])
    cu_seqlens, position_ids = position_ids_cache.get(
        device=x.device,
        max_seqlen=max_seqlen,
    )
    return packed_x, cu_seqlens, position_ids


def unpack_packed_sequence(
    packed_x: torch.Tensor,
    valid_mask: torch.Tensor,
    batch_size: int,
    max_seqlen: int,
) -> torch.Tensor:
    x = packed_x.new_zeros(batch_size, max_seqlen, packed_x.shape[-1])
    x[valid_mask] = packed_x
    return x


def unpack_packed_sequence_from_indices(
    packed_x: torch.Tensor,
    flat_indices: torch.Tensor,
    batch_size: int,
    max_seqlen: int,
) -> torch.Tensor:
    x = packed_x.new_zeros(batch_size * max_seqlen, packed_x.shape[-1])
    x.index_copy_(0, flat_indices, packed_x)
    return x.view(batch_size, max_seqlen, packed_x.shape[-1])


def unpack_unpadded_sequence(
    packed_x: torch.Tensor,
) -> torch.Tensor:
    return packed_x.reshape(1, packed_x.shape[0], packed_x.shape[-1])


class MossPackedRopeCache:
    def __init__(self, *, max_period: float, streaming_max_positions: int = 0) -> None:
        self.max_period = float(max_period)
        self.streaming_max_positions = streaming_max_positions
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
    cache: MossPackedRopeCache,
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


def _apply_cached_streaming_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    offset: torch.Tensor,
    *,
    cache: MossPackedRopeCache,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply source-equivalent RoPE to dense streaming rows from a fixed table.

    The regular codec implementation creates frequencies, phases, and two
    rotation graphs for every transformer layer and every decode step.  The
    packed cache already contains the exact FP32 table used by the reference
    implementation, so flattening ``[B, H, T, D]`` into token-major rows lets
    us reuse the one-kernel CUDA path.  ``offset`` remains a device tensor, which
    keeps this helper capture-safe.
    """
    if q.shape != k.shape:
        raise ValueError(
            f"Expected k.shape == q.shape, got k={tuple(k.shape)} q={tuple(q.shape)}"
        )
    if q.dim() != 4:
        raise ValueError(
            f"streaming RoPE expects [batch, heads, time, dim], got {tuple(q.shape)}"
        )
    batch_size, num_heads, sequence_length, head_dim = q.shape
    if offset.shape != (batch_size,):
        raise ValueError(
            f"streaming RoPE offset must have shape ({batch_size},), got "
            f"{tuple(offset.shape)}"
        )
    if sequence_length == 0:
        return q, k
    if head_dim <= 0 or head_dim % 2:
        raise ValueError(f"RoPE requires an even head_dim, got {head_dim}")

    # note (Zhang Yiyang): Use the decoder stage's 30-minute position budget
    # so the RoPE table keeps a fixed address throughout graph replay.
    max_positions = cache.streaming_max_positions
    if offset.device.type != "cuda":
        max_offset = int(offset.max().item())
        if max_offset + sequence_length > max_positions:
            raise ValueError(
                "streaming RoPE position exceeds the precomputed table: "
                f"required={max_offset + sequence_length}, "
                f"capacity={max_positions}"
            )
    cos_sin_cache = cache.get_cos_sin_cache(
        device=q.device,
        head_dim=head_dim,
        max_positions=max_positions,
    )
    position_ids = (
        offset.view(batch_size, 1)
        + torch.arange(sequence_length, device=q.device, dtype=torch.long).view(1, -1)
    ).reshape(-1)
    # Note (Zhang Yiyang): Materialize token-major rows for the exact Triton
    # RoPE kernel; the copy is captured once per graph shape.
    q_token_major = q.transpose(1, 2).reshape(-1, num_heads, head_dim).contiguous()
    k_token_major = k.transpose(1, 2).reshape(-1, num_heads, head_dim).contiguous()
    if apply_exact_interleaved_rope_inplace(
        q_token_major,
        k_token_major,
        cos_sin_cache,
        position_ids,
    ):
        return (
            q_token_major.view(
                batch_size, sequence_length, num_heads, head_dim
            ).transpose(1, 2),
            k_token_major.view(
                batch_size, sequence_length, num_heads, head_dim
            ).transpose(1, 2),
        )
    rotated_q, rotated_k = _apply_cached_packed_rope(
        q_token_major,
        k_token_major,
        position_ids,
        max_positions=max_positions,
        cache=cache,
    )
    return (
        rotated_q.view(batch_size, sequence_length, num_heads, head_dim).transpose(
            1, 2
        ),
        rotated_k.view(batch_size, sequence_length, num_heads, head_dim).transpose(
            1, 2
        ),
    )


def _run_query_chunked_sdpa(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    input_lengths: torch.Tensor,
    causal: bool,
    context: int | None,
) -> torch.Tensor:
    """Run dense local attention without materializing the full score matrix."""
    batch_size, _, sequence_length, _ = q.shape
    positions = torch.arange(sequence_length, device=q.device, dtype=torch.long)
    output_chunks = []
    for query_start in range(0, sequence_length, _SDPA_QUERY_CHUNK_SIZE):
        query_end = min(
            query_start + _SDPA_QUERY_CHUNK_SIZE,
            sequence_length,
        )
        key_end = query_end if causal else sequence_length
        key_start = max(0, query_start - context + 1) if context is not None else 0
        query_positions = positions[query_start:query_end]
        key_positions = positions[key_start:key_end]
        valid_keys = key_positions.view(1, 1, -1) < input_lengths.view(-1, 1, 1)
        if not causal and context is None:
            attention_mask = valid_keys[:, None, :, :].expand(
                -1,
                1,
                query_end - query_start,
                -1,
            )
        else:
            delta = query_positions.view(1, -1, 1) - key_positions.view(1, 1, -1)
            attention_mask = torch.ones(
                (1, query_end - query_start, key_end - key_start),
                device=q.device,
                dtype=torch.bool,
            )
            if causal:
                attention_mask &= delta >= 0
            if context is not None:
                attention_mask &= delta < context
            attention_mask = (attention_mask & valid_keys)[:, None, :, :]
        output_chunks.append(
            F.scaled_dot_product_attention(
                q[:, :, query_start:query_end],
                k[:, :, key_start:key_end],
                v[:, :, key_start:key_end],
                attn_mask=attention_mask,
                dropout_p=0.0,
            )
        )
    output = torch.cat(output_chunks, dim=2)
    valid_queries = positions.view(1, -1) < input_lengths.view(-1, 1)
    return torch.where(
        valid_queries.view(batch_size, 1, sequence_length, 1),
        output,
        torch.zeros((), device=q.device, dtype=output.dtype),
    )


def _flash_window_size(causal: bool, context: int | None) -> tuple[int, int]:
    if context is None or not causal:
        return (-1, -1)
    # note (Zhang Yiyang): Map total-token context to FlashAttention's prior-key window.
    return (max(int(context) - 1, 0), 0)


def _run_packed_flash_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    cu_seqlens: torch.Tensor,
    max_seqlen: int,
    causal: bool,
    context: int | None,
    local_flash_plan: _LocalCausalFlashPlan | None,
    flash_attn_varlen: Callable[..., torch.Tensor],
) -> torch.Tensor:
    """Run packed attention through SGLang's packed FlashAttention kernel."""
    if (
        causal
        and context is not None
        and _packed_flash_requires_query_chunks(q.device)
        and max_seqlen > _LOCAL_CAUSAL_FLASH_QUERY_CHUNK_SIZE
    ):
        context = int(context)
        plan = local_flash_plan or _build_local_causal_flash_plan(
            cu_seqlens,
            context=context,
        )
        if plan.context != context:
            raise ValueError(
                f"local flash plan context {plan.context} does not match "
                f"attention context {context}"
            )
        return flash_attn_varlen(
            q,
            _gather_local_flash_kv(k, plan.kv_indices),
            _gather_local_flash_kv(v, plan.kv_indices),
            plan.cu_seqlens_q,
            plan.cu_seqlens_k,
            plan.max_seqlen_q,
            plan.max_seqlen_k,
            causal=True,
            window_size=_flash_window_size(causal, context),
        )
    return flash_attn_varlen(
        q,
        k,
        v,
        cu_seqlens,
        cu_seqlens,
        max_seqlen,
        max_seqlen,
        causal=causal,
        window_size=_flash_window_size(causal, context),
    )


def _merge_attention_heads(output: torch.Tensor, embed_dim: int) -> torch.Tensor:
    if output.dim() == 4:
        output = output.transpose(1, 2)
    elif output.dim() != 3:
        raise ValueError(
            "attention output must have shape [batch, heads, seq, dim] or "
            f"[tokens, heads, dim], got {tuple(output.shape)}"
        )
    return output.reshape(*output.shape[:-2], embed_dim)


class MossAudioTokenizerAttention(MossAudioTokenizerStreamingModule):
    """MOSS-Audio-Tokenizer local-causal attention over codec frames."""

    def __init__(
        self,
        *,
        in_proj: nn.Module,
        out_proj: nn.Module,
        embed_dim: int,
        num_heads: int,
        causal: bool,
        context: int | None,
        rope: nn.Module | None,
        attention_backend: str = AUTO_ATTENTION_BACKEND,
        packed_rope_cache: MossPackedRopeCache | None = None,
    ) -> None:
        super().__init__()
        self.in_proj = in_proj
        self.out_proj = out_proj
        self.embed_dim = int(embed_dim)
        self.num_heads = int(num_heads)
        if self.embed_dim % self.num_heads:
            raise ValueError(
                "embed_dim must be divisible by num_heads: "
                f"{self.embed_dim}, {self.num_heads}"
            )
        self.head_dim = self.embed_dim // self.num_heads
        if self.embed_dim != self.num_heads * self.head_dim:
            raise ValueError(
                f"invalid attention shape: embed_dim={self.embed_dim}, "
                f"num_heads={self.num_heads}, head_dim={self.head_dim}"
            )
        self.causal = bool(causal)
        self.context = None if context is None else int(context)
        self.rope = rope
        self.attention_backend = validate_attention_backend(attention_backend)
        self._flash_attn_varlen = flash_attn_varlen_func
        max_period = self.rope.max_period if self.rope is not None else 10000.0
        self._packed_rope_cache = packed_rope_cache or MossPackedRopeCache(
            max_period=max_period
        )

    def _init_streaming_state(self, batch_size: int) -> _AttentionStreamingState:
        device = self.in_proj.weight.device
        return _AttentionStreamingState(
            int(batch_size),
            device,
            cached_keys=None,
            cached_values=None,
            cached_positions=None,
            offset=torch.zeros(batch_size, device=device, dtype=torch.long),
        )

    @classmethod
    def from_module(
        cls,
        module: nn.Module,
        *,
        attention_backend: str = AUTO_ATTENTION_BACKEND,
        packed_rope_cache: MossPackedRopeCache | None = None,
    ) -> MossAudioTokenizerAttention:
        return cls(
            in_proj=_single_module(module, "in_proj", "in_projs"),
            out_proj=_single_module(module, "out_proj", "out_projs"),
            embed_dim=int(module.embed_dim),
            num_heads=int(module.num_heads),
            causal=bool(module.causal),
            context=module.context,
            rope=module.rope,
            attention_backend=attention_backend,
            packed_rope_cache=packed_rope_cache,
        )

    def resolve_attention_backend(
        self,
        device: torch.device,
        dtype: torch.dtype | None,
    ) -> AttentionBackendResolution:
        if self.attention_backend == SDPA_ATTENTION_BACKEND:
            return AttentionBackendResolution(SDPA_ATTENTION_BACKEND)

        unavailable_reason = self._packed_flash_unavailable_reason(device, dtype)
        if unavailable_reason is None:
            return AttentionBackendResolution(PACKED_FLASH_ATTENTION_BACKEND)
        if self.attention_backend == PACKED_FLASH_ATTENTION_BACKEND:
            raise RuntimeError(
                "MOSS-Audio-Tokenizer "
                "attention_backend='packed_flash_attention' "
                f"is unavailable for device={device}, dtype={dtype}: "
                f"{unavailable_reason}"
            )
        return AttentionBackendResolution(
            SDPA_ATTENTION_BACKEND,
            fallback_reason=unavailable_reason,
        )

    def _packed_flash_unavailable_reason(
        self,
        device: torch.device,
        dtype: torch.dtype | None,
    ) -> str | None:
        if dtype is not torch.bfloat16:
            return f"packed FlashAttention requires torch.bfloat16; got {dtype}"
        return _packed_flash_device_unavailable_reason(device)

    def supports_packed_attention(
        self,
        device: torch.device,
        dtype: torch.dtype | None,
    ) -> bool:
        if self.attention_backend == SDPA_ATTENTION_BACKEND:
            return False
        return self._packed_flash_unavailable_reason(device, dtype) is None

    @staticmethod
    def get_backend_dtype(x: torch.Tensor) -> torch.dtype:
        backend_dtype = x.dtype
        if x.device.type != "cuda":
            return backend_dtype
        try:
            autocast_enabled = torch.is_autocast_enabled("cuda")
        except TypeError:
            autocast_enabled = torch.is_autocast_enabled()
        if autocast_enabled:
            try:
                backend_dtype = torch.get_autocast_dtype("cuda")
            except TypeError:
                backend_dtype = torch.get_autocast_gpu_dtype()
        return backend_dtype

    def build_local_causal_flash_plan(
        self,
        cu_seqlens: torch.Tensor,
        *,
        max_seqlen: int,
        sequence_lengths: Sequence[int] | None = None,
    ) -> _LocalCausalFlashPlan | None:
        if (
            not self.causal
            or self.context is None
            or not _packed_flash_requires_query_chunks(cu_seqlens.device)
            or max_seqlen <= _LOCAL_CAUSAL_FLASH_QUERY_CHUNK_SIZE
        ):
            return None
        return _build_local_causal_flash_plan(
            cu_seqlens,
            context=int(self.context),
            sequence_lengths=sequence_lengths,
        )

    def forward(
        self,
        query: torch.Tensor,
        *,
        cu_seqlens: torch.Tensor | None = None,
        max_seqlen: int | None = None,
        position_ids: torch.Tensor | None = None,
        input_lengths: torch.Tensor | None = None,
        local_flash_plan: _LocalCausalFlashPlan | None = None,
        execution_context: StreamingExecutionContext | None = None,
    ) -> torch.Tensor:
        state = self._streaming_state
        if state is not None:
            if not isinstance(state, _AttentionStreamingState):
                raise RuntimeError("invalid MOSS attention streaming state")
            if query.dim() != 3:
                raise ValueError(
                    "streaming attention expects a 3D tensor, "
                    f"got {tuple(query.shape)}"
                )
            if execution_context is not None:
                return self._forward_streaming_indexed(
                    query,
                    state,
                    execution_context,
                )
            # Note (Zhang Yiyang): Keep streaming on SDPA; packed FlashAttention
            # is reserved for the validated non-streaming path.
            return self._forward_streaming_sdpa(query, state)
        if execution_context is not None:
            raise RuntimeError(
                "streaming execution context requires an active attention state"
            )
        backend = self.resolve_attention_backend(
            query.device,
            self.get_backend_dtype(query),
        ).backend
        is_packed = backend == PACKED_FLASH_ATTENTION_BACKEND
        if is_packed:
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
        else:
            if query.dim() != 3:
                raise ValueError(
                    f"dense attention expects a 3D tensor, got {tuple(query.shape)}"
                )
            if input_lengths is None:
                raise ValueError("dense attention requires input_lengths")
            if query.shape[1] == 0:
                return self.out_proj(query)

        q, k, v = self._project_qkv(query)
        if is_packed:
            q, k = self._apply_packed_rope(
                q,
                k,
                position_ids,
                max_positions=max_seqlen,
            )
            output = _run_packed_flash_attention(
                q,
                k,
                v,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
                causal=self.causal,
                context=self.context,
                local_flash_plan=local_flash_plan,
                flash_attn_varlen=self._flash_attn_varlen,
            )
        else:
            if self.rope is not None:
                offset = torch.zeros(
                    query.shape[0],
                    device=query.device,
                    dtype=torch.long,
                )
                q, k = self.rope(q, k, offset)
            output = _run_query_chunked_sdpa(
                q,
                k,
                v,
                input_lengths=input_lengths,
                causal=self.causal,
                context=self.context,
            )
        return self.out_proj(_merge_attention_heads(output, self.embed_dim))

    def _forward_streaming_indexed(
        self,
        x: torch.Tensor,
        state: _AttentionStreamingState,
        execution_context: StreamingExecutionContext,
    ) -> torch.Tensor:
        """Gather compact rows and write their chronological KV state in place.

        Preserve the source key order: permuting a ring cache changes BF16
        attention reductions even when the position mask selects the same keys.
        """
        if x.shape[1] == 0:
            return self.out_proj(x)
        batch_size = int(x.shape[0])
        slots = execution_context.state_slot_ids
        valid_rows = execution_context.valid_rows
        cached_keys, cached_values, cached_positions = self._ensure_streaming_cache(
            state,
            state.batch_size,
            x.device,
            self.get_backend_dtype(x),
        )
        old_offsets = state.offset.index_select(0, slots)
        if (
            self.context is not None
            and cached_keys.shape[2] == self.context
            and can_fuse_streaming_kv(
                cached_keys,
                cached_values,
                cached_positions,
                state.offset,
                slots,
                valid_rows,
            )
        ):
            q, current_k, current_v, query_positions = self._project_streaming_qkv(
                x, old_offsets
            )
            all_k, all_v, key_positions = gather_streaming_kv(
                cached_keys,
                cached_values,
                cached_positions,
                current_k,
                current_v,
                query_positions,
                slots,
            )
            output = self._streaming_attention(
                q, all_k, all_v, query_positions, key_positions
            )
            output = self.out_proj(output)
            commit_streaming_kv_(
                cached_keys,
                cached_values,
                cached_positions,
                state.offset,
                all_k,
                all_v,
                key_positions,
                slots,
                valid_rows,
            )
            return output.masked_fill(~valid_rows.view(-1, 1, 1), 0)

        old_row_keys = cached_keys.index_select(0, slots)
        old_row_values = cached_values.index_select(0, slots)
        old_row_positions = cached_positions.index_select(0, slots)
        row_state = _AttentionStreamingState(
            batch_size,
            x.device,
            cached_keys=old_row_keys,
            cached_values=old_row_values,
            cached_positions=old_row_positions,
            offset=old_offsets.clone(),
        )
        row_state.exec_mask.copy_(valid_rows)
        if self.context is None:
            row_state.exec_mask.fill_(True)
        output = self._forward_streaming_sdpa(x, row_state)

        # note (Zhang Yiyang): Finite-context SDPA preserves inactive rows and
        # offsets. Unbounded attention advances every execution row while its
        # cache grows, so only that path needs the outer validity guards.
        next_offsets = row_state.offset
        if self.context is None:
            next_offsets = torch.where(valid_rows, next_offsets, old_offsets)
        state.offset.index_copy_(0, slots, next_offsets)
        if self.context is None:
            old_length = int(cached_keys.shape[2])
            new_length = int(row_state.cached_keys.shape[2])
            expanded_keys = torch.zeros(
                state.batch_size,
                self.num_heads,
                new_length,
                self.head_dim,
                device=cached_keys.device,
                dtype=cached_keys.dtype,
            )
            expanded_values = torch.zeros_like(expanded_keys)
            expanded_positions = torch.full(
                (state.batch_size, new_length),
                -1,
                device=cached_positions.device,
                dtype=cached_positions.dtype,
            )
            if old_length:
                expanded_keys[:, :, :old_length] = cached_keys
                expanded_values[:, :, :old_length] = cached_values
                expanded_positions[:, :old_length] = cached_positions
            old_row_keys_expanded = expanded_keys.index_select(0, slots)
            old_row_values_expanded = expanded_values.index_select(0, slots)
            old_row_positions_expanded = expanded_positions.index_select(0, slots)
            state.cached_keys = expanded_keys
            state.cached_values = expanded_values
            state.cached_positions = expanded_positions
            valid_cache_rows = valid_rows.view(-1, 1, 1, 1)
            next_row_keys = torch.where(
                valid_cache_rows, row_state.cached_keys, old_row_keys_expanded
            )
            next_row_values = torch.where(
                valid_cache_rows, row_state.cached_values, old_row_values_expanded
            )
            next_row_positions = torch.where(
                valid_rows.view(-1, 1),
                row_state.cached_positions,
                old_row_positions_expanded,
            )
        else:
            next_row_keys = row_state.cached_keys
            next_row_values = row_state.cached_values
            next_row_positions = row_state.cached_positions

        state.cached_keys.index_copy_(0, slots, next_row_keys)
        state.cached_values.index_copy_(0, slots, next_row_values)
        state.cached_positions.index_copy_(0, slots, next_row_positions)
        return output.masked_fill(~valid_rows.view(-1, 1, 1), 0)

    def _ensure_streaming_cache(
        self,
        state: _AttentionStreamingState,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        cache_length = 0 if self.context is None else self.context
        if (
            state.cached_keys is None
            or state.cached_values is None
            or state.cached_positions is None
        ):
            state.cached_keys = torch.zeros(
                (batch_size, self.num_heads, cache_length, self.head_dim),
                device=device,
                dtype=dtype,
            )
            state.cached_values = torch.zeros_like(state.cached_keys)
            state.cached_positions = torch.full(
                (batch_size, cache_length),
                -1,
                device=device,
                dtype=torch.long,
            )
        else:
            if state.cached_keys.device != device or state.cached_keys.dtype != dtype:
                state.cached_keys = state.cached_keys.to(device=device, dtype=dtype)
            if (
                state.cached_values.device != device
                or state.cached_values.dtype != dtype
            ):
                state.cached_values = state.cached_values.to(
                    device=device,
                    dtype=dtype,
                )
            if state.cached_positions.device != device:
                state.cached_positions = state.cached_positions.to(device=device)
        return state.cached_keys, state.cached_values, state.cached_positions

    @staticmethod
    def _build_streaming_kv(
        cached_k: torch.Tensor,
        cached_v: torch.Tensor,
        cached_pos: torch.Tensor,
        current_k: torch.Tensor,
        current_v: torch.Tensor,
        query_positions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            torch.cat((cached_k, current_k), dim=2),
            torch.cat((cached_v, current_v), dim=2),
            torch.cat((cached_pos, query_positions), dim=1),
        )

    def _update_streaming_cache(
        self,
        state: _AttentionStreamingState,
        cached_k: torch.Tensor,
        cached_v: torch.Tensor,
        cached_pos: torch.Tensor,
        all_k: torch.Tensor,
        all_v: torch.Tensor,
        key_positions: torch.Tensor,
    ) -> None:
        exec_mask = state.exec_mask.view(-1, 1, 1, 1)
        position_exec_mask = state.exec_mask.view(-1, 1)
        if self.context is None:
            if not bool(state.exec_mask.all().item()):
                raise RuntimeError(
                    "streaming exec_mask requires finite attention context"
                )
            state.cached_keys = all_k.contiguous()
            state.cached_values = all_v.contiguous()
            state.cached_positions = key_positions.contiguous()
            return
        new_cached_k = all_k[:, :, -self.context :, :].contiguous()
        new_cached_v = all_v[:, :, -self.context :, :].contiguous()
        new_cached_pos = key_positions[:, -self.context :].contiguous()
        state.cached_keys = torch.where(exec_mask, new_cached_k, cached_k)
        state.cached_values = torch.where(exec_mask, new_cached_v, cached_v)
        state.cached_positions = torch.where(
            position_exec_mask,
            new_cached_pos,
            cached_pos,
        )

    def _project_streaming_qkv(
        self,
        x: torch.Tensor,
        offsets: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        q, current_k, current_v = self._project_qkv(x)
        if self.rope is not None:
            q, current_k = _apply_cached_streaming_rope(
                q, current_k, offsets, cache=self._packed_rope_cache
            )
        query_positions = offsets.view(-1, 1) + torch.arange(
            x.shape[1], device=x.device, dtype=torch.long
        ).view(1, -1)
        return q, current_k, current_v, query_positions

    def _streaming_qkv(
        self,
        x: torch.Tensor,
        state: _AttentionStreamingState,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        q, current_k, current_v, query_positions = self._project_streaming_qkv(
            x, state.offset
        )
        cached_k, cached_v, cached_pos = self._ensure_streaming_cache(
            state,
            x.shape[0],
            current_k.device,
            current_k.dtype,
        )
        all_k, all_v, key_positions = self._build_streaming_kv(
            cached_k,
            cached_v,
            cached_pos,
            current_k,
            current_v,
            query_positions,
        )
        return (
            q,
            all_k,
            all_v,
            query_positions,
            key_positions,
            cached_k,
            cached_v,
            cached_pos,
        )

    def _finish_streaming_attention(
        self,
        output: torch.Tensor,
        *,
        state: _AttentionStreamingState,
        chunk_length: int,
        cached_k: torch.Tensor,
        cached_v: torch.Tensor,
        cached_pos: torch.Tensor,
        all_k: torch.Tensor,
        all_v: torch.Tensor,
        key_positions: torch.Tensor,
    ) -> torch.Tensor:
        self._update_streaming_cache(
            state,
            cached_k,
            cached_v,
            cached_pos,
            all_k,
            all_v,
            key_positions,
        )
        state.offset.copy_(
            torch.where(
                state.exec_mask,
                state.offset + chunk_length,
                state.offset,
            )
        )
        return self.out_proj(output)

    def _forward_streaming_sdpa(
        self,
        x: torch.Tensor,
        state: _AttentionStreamingState,
    ) -> torch.Tensor:
        chunk_length = x.shape[1]
        if chunk_length == 0:
            return self.out_proj(x)
        (
            q,
            all_k,
            all_v,
            query_positions,
            key_positions,
            cached_k,
            cached_v,
            cached_pos,
        ) = self._streaming_qkv(x, state)
        output = self._streaming_attention(
            q, all_k, all_v, query_positions, key_positions
        )
        return self._finish_streaming_attention(
            output,
            state=state,
            chunk_length=chunk_length,
            cached_k=cached_k,
            cached_v=cached_v,
            cached_pos=cached_pos,
            all_k=all_k,
            all_v=all_v,
            key_positions=key_positions,
        )

    def _streaming_attention(
        self,
        q: torch.Tensor,
        all_k: torch.Tensor,
        all_v: torch.Tensor,
        query_positions: torch.Tensor,
        key_positions: torch.Tensor,
    ) -> torch.Tensor:
        delta = query_positions[:, :, None] - key_positions[:, None, :]
        attention_mask = (key_positions[:, None, :] >= 0) & (delta >= 0)
        if self.context is not None:
            attention_mask &= delta < self.context
        output = F.scaled_dot_product_attention(
            q, all_k, all_v, attn_mask=attention_mask[:, None, :, :], dropout_p=0.0
        )
        return output.transpose(1, 2).reshape(
            query_positions.shape[0], query_positions.shape[1], self.embed_dim
        )

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
