# SPDX-License-Identifier: Apache-2.0
"""Streaming window planning for the Higgs vocoder."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Mapping

import torch

from sglang_omni.models.tts_streaming import (
    INITIAL_CODEC_CHUNK_FRAMES_PARAM,
    resolve_initial_codec_chunk_frames,
)
from sglang_omni.utils.audio_payload import audio_waveform_payload


@dataclass
class HiggsStreamState:
    delayed_rows: list[torch.Tensor] = field(default_factory=list)
    emitted_raw_frames: int = 0
    next_decode_rows: int = 0
    has_emitted: bool = False
    num_codebooks: int | None = None
    codebook_size: int | None = None
    initial_codec_chunk_frames: int = 0


@dataclass(frozen=True)
class HiggsStreamConfig:
    stream_stride: int
    stream_followup_stride: int
    stream_overlap_tokens: int
    stream_holdback_tokens: int
    samples_per_frame: int | None
    sample_rate: int


def latch_higgs_stream_metadata(
    request_id: str,
    state: HiggsStreamState,
    metadata: dict[str, Any] | None,
    *,
    config: HiggsStreamConfig,
) -> None:
    if not isinstance(metadata, dict):
        if state.num_codebooks is None or state.codebook_size is None:
            raise RuntimeError(
                f"Higgs stream chunk for {request_id!r} is missing metadata "
                "with num_codebooks and codebook_size"
            )
        return
    if metadata.get("modality") not in (None, "audio_codes"):
        raise ValueError(
            f"Higgs stream chunk modality must be audio_codes, got "
            f"{metadata.get('modality')!r}"
        )
    if metadata.get("stream") is not True:
        raise RuntimeError(
            f"Higgs stream chunk for {request_id!r} must include "
            "metadata['stream'] == True"
        )
    missing = [key for key in ("num_codebooks", "codebook_size") if key not in metadata]
    if missing and (state.num_codebooks is None or state.codebook_size is None):
        raise RuntimeError(
            f"Higgs stream chunk for {request_id!r} is missing metadata fields: "
            f"{', '.join(missing)}"
        )
    if "num_codebooks" in metadata and "codebook_size" in metadata:
        latch_higgs_stream_contract(
            request_id,
            state,
            num_codebooks=metadata["num_codebooks"],
            codebook_size=metadata["codebook_size"],
            source="stream metadata",
        )
    if INITIAL_CODEC_CHUNK_FRAMES_PARAM in metadata:
        latch_initial_codec_chunk_frames_from_mapping(
            request_id,
            state,
            metadata,
            config=config,
        )


def latch_higgs_stream_contract(
    request_id: str,
    state: HiggsStreamState,
    *,
    num_codebooks: Any,
    codebook_size: Any,
    source: str,
) -> None:
    try:
        num_codebooks_i = int(num_codebooks)
        codebook_size_i = int(codebook_size)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            f"Higgs {source} for {request_id!r} must include integer "
            "num_codebooks and codebook_size"
        ) from exc
    if num_codebooks_i <= 0 or codebook_size_i <= 2:
        raise ValueError(
            f"Higgs {source} for {request_id!r} has invalid "
            f"num_codebooks={num_codebooks_i}, codebook_size={codebook_size_i}"
        )
    if state.num_codebooks is not None and state.num_codebooks != num_codebooks_i:
        raise ValueError(
            f"Higgs stream num_codebooks changed for {request_id!r}: "
            f"{state.num_codebooks} -> {num_codebooks_i}"
        )
    if state.codebook_size is not None and state.codebook_size != codebook_size_i:
        raise ValueError(
            f"Higgs stream codebook_size changed for {request_id!r}: "
            f"{state.codebook_size} -> {codebook_size_i}"
        )
    state.num_codebooks = num_codebooks_i
    state.codebook_size = codebook_size_i


def latch_initial_codec_chunk_frames_from_mapping(
    request_id: str,
    state: HiggsStreamState,
    params: Mapping[str, Any] | None,
    *,
    config: HiggsStreamConfig,
) -> None:
    num_codebooks, _ = require_higgs_stream_contract(state, request_id)
    steady_codec_frames = max(1, config.stream_stride - num_codebooks + 1)
    state.initial_codec_chunk_frames = resolve_initial_codec_chunk_frames(
        params,
        steady_chunk_frames=steady_codec_frames,
    )


def require_higgs_stream_contract(
    state: HiggsStreamState,
    request_id: str,
) -> tuple[int, int]:
    if state.num_codebooks is None or state.codebook_size is None:
        raise RuntimeError(
            f"Higgs stream contract for {request_id!r} is missing "
            "num_codebooks or codebook_size"
        )
    return state.num_codebooks, state.codebook_size


def build_higgs_stream_delta(
    state: HiggsStreamState,
    *,
    config: HiggsStreamConfig,
    decode_delayed_rows: Callable[..., torch.Tensor],
    is_final: bool,
) -> dict[str, Any] | None:
    delayed_count = len(state.delayed_rows)
    if delayed_count == 0:
        return None
    num_codebooks, codebook_size = require_higgs_stream_contract(state, "<stream>")
    if delayed_count < num_codebooks:
        return None
    raw_total = delayed_count - num_codebooks + 1

    steady_codec_frames = max(1, config.stream_stride - num_codebooks + 1)
    use_initial_chunk = (
        state.initial_codec_chunk_frames > 0
        and state.initial_codec_chunk_frames < steady_codec_frames
        and not state.has_emitted
    )
    first_decode_rows = max(
        num_codebooks,
        state.initial_codec_chunk_frames + num_codebooks - 1,
    )
    next_decode_rows = state.next_decode_rows or (
        first_decode_rows
        if use_initial_chunk and not is_final
        else max(num_codebooks, config.stream_stride)
    )
    if not is_final and delayed_count < next_decode_rows:
        state.next_decode_rows = next_decode_rows
        return None

    emit_until_raw = raw_total
    if use_initial_chunk and not is_final:
        emit_until_raw = min(raw_total, state.initial_codec_chunk_frames)
    elif not is_final and config.stream_holdback_tokens:
        emit_until_raw = max(0, raw_total - config.stream_holdback_tokens)
    can_flush_codec_tail = is_final and config.samples_per_frame is not None
    if emit_until_raw < state.emitted_raw_frames or (
        emit_until_raw == state.emitted_raw_frames and not can_flush_codec_tail
    ):
        state.next_decode_rows = delayed_count + config.stream_followup_stride
        return None

    window_start_raw = max(0, state.emitted_raw_frames - config.stream_overlap_tokens)
    rows_end = emit_until_raw + num_codebooks - 1
    rows = state.delayed_rows[window_start_raw:rows_end]
    audio = decode_delayed_rows(
        rows,
        num_codebooks=num_codebooks,
        codebook_size=codebook_size,
    )

    decoded_raw_frames = emit_until_raw - window_start_raw
    samples_per_frame = config.samples_per_frame or max(
        int(audio.shape[-1]) // max(decoded_raw_frames, 1), 1
    )
    trim_frames = state.emitted_raw_frames - window_start_raw
    trim_samples = min(int(trim_frames * samples_per_frame), int(audio.shape[-1]))
    if not is_final and config.samples_per_frame is not None:
        new_frames = emit_until_raw - state.emitted_raw_frames
        emit_samples = int(new_frames * samples_per_frame)
        delta = audio[trim_samples : trim_samples + emit_samples].contiguous()
    else:
        delta = audio[trim_samples:].contiguous()
    if delta.numel() == 0:
        state.next_decode_rows = delayed_count + config.stream_followup_stride
        return None

    state.emitted_raw_frames = emit_until_raw
    state.next_decode_rows = _next_decode_rows_after_emit(
        delayed_count,
        num_codebooks=num_codebooks,
        emitted_initial_chunk=use_initial_chunk and not is_final,
        config=config,
    )
    state.has_emitted = True
    return audio_waveform_payload(
        delta,
        sample_rate=config.sample_rate,
        modality="audio",
        source_hint="Higgs TTS streaming",
    )


def _next_decode_rows_after_emit(
    delayed_count: int,
    *,
    num_codebooks: int,
    emitted_initial_chunk: bool,
    config: HiggsStreamConfig,
) -> int:
    if emitted_initial_chunk:
        return max(num_codebooks, config.stream_stride) + config.stream_followup_stride
    return delayed_count + config.stream_followup_stride
