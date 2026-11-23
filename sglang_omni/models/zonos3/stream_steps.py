# SPDX-License-Identifier: Apache-2.0
"""Zonos3 stream state and the readiness and urgency rules that choose the streams of one vocoder decode step."""

from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import IntEnum

import torch

from sglang_omni.models.zonos3.batched_decode import length_groups
from sglang_omni.models.zonos3.components.codec import (
    BOUNDARY_HOLDBACK_FRAMES,
    EXTERNAL_SAMPLES_PER_FRAME,
)
from sglang_omni.models.zonos3.components.codec_stream import Zonos3DecoderStream
from sglang_omni.proto.session import SessionOperation

# note (zhaochenyang20): A second batch lets first chunks and finished streams share a step with longer prefixes; with final decodes moved onto the serving loop so queued chunks join them, SeedTTS EN first audio fell from 0.50 s to 0.43 s at the same C50 and throughput on one H100.
MAX_STEP_LENGTH_GROUPS = 2
# note (zhaochenyang20): A playing stream is decoded only once its buffered audio runs out within this long, so each decode of its whole prefix carries more frames; 0.4 s is the old urgency margin of a 6-frame chunk, and 0.8 s was no better on SeedTTS EN.
PLAYBACK_DECODE_HORIZON_SECONDS = 0.4


@dataclass(kw_only=True)
class Zonos3StreamState:
    """Codes [codebooks, frames] received so far and the audio already sent for one request."""

    code_chunks: list[torch.Tensor] = field(default_factory=list)
    frame_count: int = 0
    emitted_sample_count: int = 0
    initial_chunk_frames: int | None = None
    playback_deadline_seconds: float | None = None
    # The finished result arrived; its final decode waits for a serving-loop step.
    is_finished: bool = False
    # Taken at the stream's first decode when a slot is free; the stream then decodes only its new frames.
    decoder_stream: Zonos3DecoderStream | None = None


class DecodeUrgency(IntEnum):
    """Decode order of a stream: running dry within PLAYBACK_DECODE_HORIZON_SECONDS, not played yet, or audio buffered."""

    DUE = 0
    UNPLAYED = 1
    BUFFERED = 2


def step_urgency(
    state: Zonos3StreamState, now_seconds: float
) -> tuple[DecodeUrgency, float]:
    """The stream's urgency and its playback deadline in seconds, which orders streams of one urgency."""
    if state.playback_deadline_seconds is None:
        return (DecodeUrgency.UNPLAYED, 0.0)
    elif (
        state.playback_deadline_seconds <= now_seconds + PLAYBACK_DECODE_HORIZON_SECONDS
    ):
        return (DecodeUrgency.DUE, state.playback_deadline_seconds)
    else:
        return (DecodeUrgency.BUFFERED, state.playback_deadline_seconds)


def stable_sample_count(state: Zonos3StreamState) -> int:
    """Samples that no later frame can change."""
    return (
        max(state.frame_count - BOUNDARY_HOLDBACK_FRAMES, 0)
        * EXTERNAL_SAMPLES_PER_FRAME
    )


def is_chunk_ready(state: Zonos3StreamState, *, steady_chunk_frames: int) -> bool:
    """Whether the stream's unsent stable audio fills its next chunk: the initial chunk first, then steady ones."""
    assert state.initial_chunk_frames is not None
    if state.emitted_sample_count == 0:
        chunk_frames = state.initial_chunk_frames
    else:
        chunk_frames = steady_chunk_frames
    return (
        stable_sample_count(state) - state.emitted_sample_count
        >= chunk_frames * EXTERNAL_SAMPLES_PER_FRAME
    )


def extend_step(
    participants: list[tuple[str, Zonos3StreamState]],
    stream_states: Mapping[str, Zonos3StreamState],
    session_appends: Mapping[str, SessionOperation],
    *,
    include_finished: bool,
    max_step_streams: int,
    steady_chunk_frames: int,
) -> list[tuple[str, Zonos3StreamState]]:
    """Add streams in step_urgency order, up to max_step_streams prefixes in at most MAX_STEP_LENGTH_GROUPS length groups.

    A ready stream joins once it is due or has not played; a finished stream joins only when include_finished and no session append is pending.
    """
    now_seconds = time.monotonic()
    participant_request_ids = {request_id for request_id, _ in participants}
    candidates = sorted(
        (
            (request_id, state)
            for request_id, state in stream_states.items()
            if request_id not in participant_request_ids
            and (
                (
                    not state.is_finished
                    and is_chunk_ready(state, steady_chunk_frames=steady_chunk_frames)
                    and step_urgency(state, now_seconds)[0] != DecodeUrgency.BUFFERED
                )
                or (
                    include_finished
                    and state.is_finished
                    and state.frame_count > 0
                    and request_id not in session_appends
                )
            )
        ),
        key=lambda candidate: step_urgency(candidate[1], now_seconds),
    )
    step = list(participants)
    for request_id, state in candidates:
        if (
            len(step) < max_step_streams
            and len(
                length_groups(
                    [participant_state.frame_count for _, participant_state in step]
                    + [state.frame_count]
                )
            )
            <= MAX_STEP_LENGTH_GROUPS
        ):
            step.append((request_id, state))
        else:
            pass
    return step
