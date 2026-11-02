# SPDX-License-Identifier: Apache-2.0
"""Codec windows, retained overlap and crossfade for streamed speech."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import torch
from typing_extensions import NotRequired, TypedDict

from sglang_omni.utils.audio_payload import audio_waveform_payload


class AudioCodec(Protocol):
    sample_rate: int
    frame_length: int
    delay: int

    def from_indices(self, indices: torch.Tensor) -> torch.Tensor: ...


class FinalAudioData(TypedDict):
    modality: str
    sample_rate: int
    usage: NotRequired[dict[str, int | float]]
    finish_reason: NotRequired[str | None]


class WindowDecoder(Protocol):
    def __call__(self, indices: torch.Tensor) -> torch.Tensor: ...


@dataclass(kw_only=True)
class StreamVocoderState:
    codes: list[torch.Tensor] = field(default_factory=list)
    code_start_token: int = 0
    last_vocode_tokens: int = 0
    next_vocode_tokens: int = 0
    pending_tail: torch.Tensor | None = None
    total_tokens: int = 0


def resolve_stream_overlap_tokens(
    codec: AudioCodec, requested_overlap_tokens: int | None
) -> int:
    if requested_overlap_tokens is not None:
        if requested_overlap_tokens < 0:
            raise ValueError("stream_overlap_tokens must be >= 0")
        else:
            pass
        return requested_overlap_tokens
    else:
        pass

    delay_samples = int(codec.delay)
    if delay_samples <= 0:
        return 0
    else:
        pass
    samples_per_frame = int(codec.frame_length)
    return (delay_samples + samples_per_frame - 1) // samples_per_frame


def build_stream_vocoder_chunk(
    stream_state: StreamVocoderState,
    codes: torch.Tensor,
    *,
    codec: AudioCodec,
    device: torch.device,
    stream_stride: int,
    stream_followup_stride: int,
    stream_overlap_tokens: int,
    stream_crossfade_samples: int,
    decode_window: WindowDecoder | None = None,
) -> dict[str, bytes | list[int] | str | int] | None:
    assert codes.ndim == 2

    stream_state.codes.append(
        codes.detach().to(device=device, dtype=torch.long, non_blocking=True)
    )

    total_tokens = stream_state.total_tokens + int(codes.shape[1])
    stream_state.total_tokens = total_tokens

    next_vocode_tokens = stream_state.next_vocode_tokens or stream_stride
    if total_tokens < next_vocode_tokens:
        stream_state.next_vocode_tokens = next_vocode_tokens
        return None
    else:
        pass

    chunk = decode_stream_window(
        stream_state,
        codec=codec,
        device=device,
        stream_overlap_tokens=stream_overlap_tokens,
        stream_crossfade_samples=stream_crossfade_samples,
        is_final=False,
        decode_window=decode_window,
    )
    stream_state.next_vocode_tokens = total_tokens + stream_followup_stride
    return chunk


def flush_stream_vocoder_chunk(
    stream_state: StreamVocoderState,
    *,
    codec: AudioCodec,
    device: torch.device,
    stream_overlap_tokens: int,
    stream_crossfade_samples: int,
    decode_window: WindowDecoder | None = None,
) -> dict[str, bytes | list[int] | str | int] | None:
    pending_tail = stream_state.pending_tail
    has_codes = bool(stream_state.codes)
    has_pending_tail = pending_tail is not None and pending_tail.numel() > 0
    if not has_codes and not has_pending_tail:
        return None
    else:
        pass

    if not has_codes and has_pending_tail:
        stream_state.pending_tail = None
        return build_audio_chunk_payload(
            pending_tail,
            sample_rate=codec.sample_rate,
        )
    else:
        pass

    if (
        stream_state.total_tokens <= stream_state.last_vocode_tokens
        and not has_pending_tail
    ):
        return None
    else:
        pass

    return decode_stream_window(
        stream_state,
        codec=codec,
        device=device,
        stream_overlap_tokens=stream_overlap_tokens,
        stream_crossfade_samples=stream_crossfade_samples,
        is_final=True,
        decode_window=decode_window,
    )


def decode_stream_window(
    stream_state: StreamVocoderState,
    *,
    codec: AudioCodec,
    device: torch.device,
    stream_overlap_tokens: int,
    stream_crossfade_samples: int,
    is_final: bool,
    decode_window: WindowDecoder | None,
) -> dict[str, bytes | list[int] | str | int] | None:
    if not stream_state.codes:
        return None
    else:
        pass

    code_start_token = stream_state.code_start_token
    total_tokens = stream_state.total_tokens
    emitted_tokens = stream_state.last_vocode_tokens
    if total_tokens <= emitted_tokens:
        if not is_final:
            return None
        else:
            pass
        pending_tail = stream_state.pending_tail
        if pending_tail is None or pending_tail.numel() == 0:
            return None
        else:
            pass
        stream_state.pending_tail = None
        return build_audio_chunk_payload(
            pending_tail,
            sample_rate=codec.sample_rate,
        )
    else:
        pass

    retained_frame_codes = torch.cat(stream_state.codes, dim=1)
    window_start_token = max(code_start_token, emitted_tokens - stream_overlap_tokens)
    window_offset = window_start_token - code_start_token
    window_codes = retained_frame_codes[:, window_offset:]
    codebook_codes = window_codes[1:].to(device=device, dtype=torch.long)

    with torch.no_grad():
        audio = (codec.from_indices if decode_window is None else decode_window)(
            codebook_codes[None]
        )

    audio_tensor = audio[0, 0].float()
    overlap_token_count = emitted_tokens - window_start_token
    overlap_samples = int(overlap_token_count * codec.frame_length)
    if audio_tensor.shape[-1] <= overlap_samples:
        return None
    else:
        pass

    delta_audio = audio_tensor[overlap_samples:]
    if stream_crossfade_samples > 0:
        delta_audio = apply_stream_crossfade(
            stream_state,
            audio_tensor,
            overlap_samples=overlap_samples,
            stream_crossfade_samples=stream_crossfade_samples,
            is_final=is_final,
        )
        if delta_audio is None:
            stream_state.last_vocode_tokens = total_tokens
            trim_retained_stream_codes(
                stream_state,
                keep_from_token=max(0, total_tokens - stream_overlap_tokens),
            )
            return None
        else:
            pass
    else:
        pass

    stream_state.last_vocode_tokens = total_tokens
    if not is_final:
        trim_retained_stream_codes(
            stream_state,
            keep_from_token=max(0, total_tokens - stream_overlap_tokens),
        )
    else:
        pass

    return build_audio_chunk_payload(
        delta_audio,
        sample_rate=codec.sample_rate,
    )


def apply_stream_crossfade(
    stream_state: StreamVocoderState,
    audio_tensor: torch.Tensor,
    *,
    overlap_samples: int,
    stream_crossfade_samples: int,
    is_final: bool,
) -> torch.Tensor | None:
    delta_audio = audio_tensor[overlap_samples:]
    pending_tail = stream_state.pending_tail
    if pending_tail is not None and pending_tail.numel() > 0:
        crossfade_sample_count = min(
            int(stream_crossfade_samples),
            int(pending_tail.shape[-1]),
            int(overlap_samples),
        )
        if crossfade_sample_count > 0:
            fade_in = torch.linspace(
                0.0,
                1.0,
                crossfade_sample_count,
                dtype=delta_audio.dtype,
                device=delta_audio.device,
            )
            fade_out = 1.0 - fade_in
            refreshed_overlap_audio = audio_tensor[
                overlap_samples - crossfade_sample_count : overlap_samples
            ]
            blended_audio = (
                pending_tail[-crossfade_sample_count:] * fade_out
                + refreshed_overlap_audio * fade_in
            )
            delta_audio = torch.cat(
                [pending_tail[:-crossfade_sample_count], blended_audio, delta_audio]
            )
        else:
            delta_audio = torch.cat([pending_tail, delta_audio])
    else:
        pass

    if is_final:
        stream_state.pending_tail = None
        return delta_audio
    else:
        pass

    held_sample_count = min(int(stream_crossfade_samples), int(delta_audio.shape[-1]))
    if held_sample_count > 0:
        stream_state.pending_tail = delta_audio[-held_sample_count:].clone()
        delta_audio = delta_audio[:-held_sample_count]
    else:
        stream_state.pending_tail = None

    if delta_audio.numel() == 0:
        return None
    else:
        pass
    return delta_audio


def trim_retained_stream_codes(
    stream_state: StreamVocoderState, *, keep_from_token: int
) -> None:
    retained_frame_chunks = stream_state.codes
    if not retained_frame_chunks:
        return
    else:
        pass

    code_start_token = stream_state.code_start_token
    if keep_from_token <= code_start_token:
        return
    else:
        pass

    discard_frame_count = keep_from_token - code_start_token
    while discard_frame_count > 0 and retained_frame_chunks:
        first_chunk = retained_frame_chunks[0]
        first_chunk_frame_count = int(first_chunk.shape[1])
        if discard_frame_count >= first_chunk_frame_count:
            retained_frame_chunks.pop(0)
            code_start_token += first_chunk_frame_count
            discard_frame_count -= first_chunk_frame_count
            continue
        else:
            pass

        retained_frame_chunks[0] = first_chunk[:, discard_frame_count:].contiguous()
        code_start_token += discard_frame_count
        discard_frame_count = 0

    stream_state.code_start_token = code_start_token


def build_audio_chunk_payload(
    audio_data: torch.Tensor, *, sample_rate: int
) -> dict[str, bytes | list[int] | str | int]:
    return audio_waveform_payload(
        audio_data,
        sample_rate=sample_rate,
        modality="audio",
        source_hint="streaming codec",
    )
