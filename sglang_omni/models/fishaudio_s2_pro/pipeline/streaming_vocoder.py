# SPDX-License-Identifier: Apache-2.0
"""Streaming vocoder helpers for FishAudio S2-Pro."""

from __future__ import annotations

from typing import Any

import torch

from sglang_omni.proto import StagePayload

STREAM_VOCODER_STATE_KEY = "_stream_vocoder_state"

_STATE_CODES_KEY = "codes"
_STATE_CODE_START_TOKEN_KEY = "code_start_token"
_STATE_LAST_VOCODE_TOKENS_KEY = "last_vocode_tokens"
_STATE_NEXT_VOCODE_TOKENS_KEY = "next_vocode_tokens"
_STATE_PENDING_TAIL_KEY = "pending_tail"
_STATE_TOTAL_TOKENS_KEY = "total_tokens"


def resolve_stream_overlap_tokens(
    codec: Any, requested_overlap_tokens: int | None
) -> int:
    if requested_overlap_tokens is not None:
        if requested_overlap_tokens < 0:
            raise ValueError("stream_overlap_tokens must be >= 0")
        return requested_overlap_tokens

    delay_samples = int(codec.delay)
    if delay_samples <= 0:
        return 0
    frame_length = int(codec.frame_length)
    return (delay_samples + frame_length - 1) // frame_length


def load_stream_vocoder_state(
    payload: StagePayload, *, create: bool = False
) -> dict[str, Any] | None:
    if not isinstance(payload.data, dict):
        return None

    state = payload.data.get(STREAM_VOCODER_STATE_KEY)
    if state is None:
        if not create:
            return None
        state = {
            _STATE_CODES_KEY: [],
            _STATE_CODE_START_TOKEN_KEY: 0,
            _STATE_LAST_VOCODE_TOKENS_KEY: 0,
            _STATE_NEXT_VOCODE_TOKENS_KEY: 0,
            _STATE_PENDING_TAIL_KEY: None,
            _STATE_TOTAL_TOKENS_KEY: 0,
        }
        payload.data[STREAM_VOCODER_STATE_KEY] = state

    if not isinstance(state, dict):
        return None
    return state


def clear_stream_vocoder_state(payload: StagePayload) -> None:
    if isinstance(payload.data, dict):
        payload.data.pop(STREAM_VOCODER_STATE_KEY, None)


def build_stream_vocoder_chunk(
    payload: StagePayload,
    codes: Any,
    *,
    codec: Any,
    device: str,
    stream_stride: int,
    stream_followup_stride: int,
    stream_overlap_tokens: int,
    stream_crossfade_samples: int,
) -> dict[str, Any] | None:
    if not isinstance(codes, torch.Tensor) or codes.ndim != 2:
        return None

    state = load_stream_vocoder_state(payload, create=True)
    if state is None:
        return None

    retained_codes = state[_STATE_CODES_KEY]
    retained_codes.append(codes.detach().cpu())

    total_tokens = int(state[_STATE_TOTAL_TOKENS_KEY]) + int(codes.shape[1])
    state[_STATE_TOTAL_TOKENS_KEY] = total_tokens

    next_vocode_tokens = int(state[_STATE_NEXT_VOCODE_TOKENS_KEY]) or stream_stride
    if total_tokens < next_vocode_tokens:
        state[_STATE_NEXT_VOCODE_TOKENS_KEY] = next_vocode_tokens
        return None

    chunk = _build_stream_vocoder_chunk(
        payload,
        state,
        codec=codec,
        device=device,
        stream_overlap_tokens=stream_overlap_tokens,
        stream_crossfade_samples=stream_crossfade_samples,
        is_final=False,
    )
    state[_STATE_NEXT_VOCODE_TOKENS_KEY] = total_tokens + stream_followup_stride
    return chunk


def flush_stream_vocoder_chunk(
    payload: StagePayload,
    *,
    codec: Any,
    device: str,
    stream_overlap_tokens: int,
    stream_crossfade_samples: int,
) -> dict[str, Any] | None:
    state = load_stream_vocoder_state(payload)
    if state is None:
        return None

    retained_codes = state[_STATE_CODES_KEY]
    pending_tail = state[_STATE_PENDING_TAIL_KEY]
    has_codes = bool(retained_codes)
    has_pending_tail = (
        isinstance(pending_tail, torch.Tensor) and pending_tail.numel() > 0
    )
    if not has_codes and not has_pending_tail:
        clear_stream_vocoder_state(payload)
        return None

    total_tokens = int(state[_STATE_TOTAL_TOKENS_KEY])
    emitted_tokens = int(state[_STATE_LAST_VOCODE_TOKENS_KEY])
    if total_tokens <= emitted_tokens and not has_pending_tail:
        clear_stream_vocoder_state(payload)
        return None

    return _build_stream_vocoder_chunk(
        payload,
        state,
        codec=codec,
        device=device,
        stream_overlap_tokens=stream_overlap_tokens,
        stream_crossfade_samples=stream_crossfade_samples,
        is_final=True,
    )


def _build_stream_vocoder_chunk(
    payload: StagePayload,
    state: dict[str, Any],
    *,
    codec: Any,
    device: str,
    stream_overlap_tokens: int,
    stream_crossfade_samples: int,
    is_final: bool,
) -> dict[str, Any] | None:
    retained_codes = state[_STATE_CODES_KEY]
    if not isinstance(retained_codes, list) or not retained_codes:
        return None

    code_start_token = int(state[_STATE_CODE_START_TOKEN_KEY])
    total_tokens = int(state[_STATE_TOTAL_TOKENS_KEY])
    emitted_tokens = int(state[_STATE_LAST_VOCODE_TOKENS_KEY])
    if total_tokens <= emitted_tokens:
        if not is_final:
            return None
        pending_tail = state[_STATE_PENDING_TAIL_KEY]
        if not isinstance(pending_tail, torch.Tensor) or pending_tail.numel() == 0:
            return None
        clear_stream_vocoder_state(payload)
        return _build_audio_chunk_payload(
            pending_tail.tolist(),
            sample_rate=codec.sample_rate,
        )

    output_codes = torch.cat(retained_codes, dim=1)
    window_start_token = max(code_start_token, emitted_tokens - stream_overlap_tokens)
    window_offset = window_start_token - code_start_token
    window_codes = output_codes[:, window_offset:]
    codebook_codes = window_codes[1:].to(device)

    with torch.no_grad():
        audio = codec.from_indices(codebook_codes[None])

    audio_np = audio[0, 0].float().cpu()
    overlap_token_count = emitted_tokens - window_start_token
    overlap_samples = int(overlap_token_count * codec.frame_length)
    if audio_np.shape[-1] <= overlap_samples:
        return None

    delta_audio = audio_np[overlap_samples:]
    if stream_crossfade_samples > 0:
        delta_audio = _apply_stream_crossfade(
            state,
            delta_audio,
            stream_crossfade_samples=stream_crossfade_samples,
            is_final=is_final,
        )
        if delta_audio is None:
            state[_STATE_LAST_VOCODE_TOKENS_KEY] = total_tokens
            trim_retained_stream_codes(
                state,
                keep_from_token=max(0, total_tokens - stream_overlap_tokens),
            )
            return None

    state[_STATE_LAST_VOCODE_TOKENS_KEY] = total_tokens
    if is_final:
        clear_stream_vocoder_state(payload)
    else:
        trim_retained_stream_codes(
            state,
            keep_from_token=max(0, total_tokens - stream_overlap_tokens),
        )

    return _build_audio_chunk_payload(
        delta_audio.tolist(),
        sample_rate=codec.sample_rate,
    )


def _apply_stream_crossfade(
    state: dict[str, Any],
    delta_audio: torch.Tensor,
    *,
    stream_crossfade_samples: int,
    is_final: bool,
) -> torch.Tensor | None:
    pending_tail = state[_STATE_PENDING_TAIL_KEY]
    if isinstance(pending_tail, torch.Tensor) and pending_tail.numel() > 0:
        crossfade = min(
            int(stream_crossfade_samples),
            int(pending_tail.shape[-1]),
            int(delta_audio.shape[-1]),
        )
        if crossfade > 0:
            fade_in = torch.linspace(
                0.0,
                1.0,
                crossfade,
                dtype=delta_audio.dtype,
            )
            fade_out = 1.0 - fade_in
            blended = (
                pending_tail[-crossfade:] * fade_out + delta_audio[:crossfade] * fade_in
            )
            delta_audio = torch.cat(
                [pending_tail[:-crossfade], blended, delta_audio[crossfade:]]
            )
        else:
            delta_audio = torch.cat([pending_tail, delta_audio])

    if is_final:
        state[_STATE_PENDING_TAIL_KEY] = None
        return delta_audio

    hold = min(int(stream_crossfade_samples), int(delta_audio.shape[-1]))
    if hold > 0:
        state[_STATE_PENDING_TAIL_KEY] = delta_audio[-hold:].clone()
        delta_audio = delta_audio[:-hold]
    else:
        state[_STATE_PENDING_TAIL_KEY] = None

    if delta_audio.numel() == 0:
        return None
    return delta_audio


def trim_retained_stream_codes(state: dict[str, Any], *, keep_from_token: int) -> None:
    retained_codes = state[_STATE_CODES_KEY]
    if not isinstance(retained_codes, list) or not retained_codes:
        return

    code_start_token = int(state[_STATE_CODE_START_TOKEN_KEY])
    if keep_from_token <= code_start_token:
        return

    drop_tokens = keep_from_token - code_start_token
    while drop_tokens > 0 and retained_codes:
        first_chunk = retained_codes[0]
        first_width = int(first_chunk.shape[1])
        if drop_tokens >= first_width:
            retained_codes.pop(0)
            code_start_token += first_width
            drop_tokens -= first_width
            continue

        retained_codes[0] = first_chunk[:, drop_tokens:].contiguous()
        code_start_token += drop_tokens
        drop_tokens = 0

    state[_STATE_CODE_START_TOKEN_KEY] = code_start_token


def _build_audio_chunk_payload(
    audio_data: list[float], *, sample_rate: int
) -> dict[str, Any]:
    return {
        "audio_data": audio_data,
        "sample_rate": sample_rate,
        "modality": "audio",
    }
