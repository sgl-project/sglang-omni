# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations


def stream_window_lengths(
    *,
    stream_stride: int,
    stream_followup_stride_ramp: tuple[int, ...],
    stream_followup_stride: int,
    stream_overlap_tokens: int,
) -> tuple[int, ...]:
    strides = (stream_stride, *stream_followup_stride_ramp)
    lengths: set[int] = set()
    emitted_frames = 0
    window_index = 0
    while True:
        stride = (
            strides[window_index]
            if window_index < len(strides)
            else stream_followup_stride
        )
        window_end = emitted_frames + stride
        lengths.add(window_end - max(0, emitted_frames - stream_overlap_tokens))
        if window_index >= len(strides) and emitted_frames >= stream_overlap_tokens:
            return tuple(sorted(lengths))
        else:
            pass
        emitted_frames = window_end
        window_index += 1


def stream_window_discards(
    *,
    stream_stride: int,
    stream_followup_stride_ramp: tuple[int, ...],
    stream_followup_stride: int,
    stream_overlap_tokens: int,
) -> dict[int, int]:
    """Per stream window length, the overlap frames at its start whose audio the
    stream discards: the window decodes them for context only."""
    strides = (stream_stride, *stream_followup_stride_ramp)
    discards: dict[int, int] = {}
    emitted_frames = 0
    window_index = 0
    while True:
        stride = (
            strides[window_index]
            if window_index < len(strides)
            else stream_followup_stride
        )
        window_start = max(0, emitted_frames - stream_overlap_tokens)
        discards[emitted_frames + stride - window_start] = emitted_frames - window_start
        if window_index >= len(strides) and emitted_frames >= stream_overlap_tokens:
            return discards
        else:
            pass
        emitted_frames += stride
        window_index += 1


def flush_window_shapes(
    *,
    stream_stride: int,
    stream_followup_stride_ramp: tuple[int, ...],
    stream_followup_stride: int,
    stream_overlap_tokens: int,
) -> set[tuple[int, int]]:
    """(frames, overlap frames) of the final window a stream flushes after each
    complete window: the overlap plus the frames left before the next window would
    complete."""
    strides = (stream_stride, *stream_followup_stride_ramp)
    keys: set[tuple[int, int]] = set()
    emitted_frames = 0
    window_index = 0
    while True:
        stride = (
            strides[window_index]
            if window_index < len(strides)
            else stream_followup_stride
        )
        emitted_frames += stride
        window_index += 1
        discard = min(emitted_frames, stream_overlap_tokens)
        next_stride = (
            strides[window_index]
            if window_index < len(strides)
            else stream_followup_stride
        )
        for remaining in range(1, next_stride + 1):
            keys.add((discard + remaining, discard))
        if window_index >= len(strides) and emitted_frames >= stream_overlap_tokens:
            return keys
        else:
            pass


def utterance_graph_lengths(max_frames: int, *, frame_step: int) -> tuple[int, ...]:
    """Graph lengths for whole-utterance decodes: every frame_step frames below max_frames,
    then max_frames itself; none when max_frames is 0."""
    if max_frames == 0:
        return ()
    else:
        return (*range(frame_step, max_frames, frame_step), max_frames)


def frames_in_complete_windows(
    frame_count: int,
    *,
    stream_stride: int,
    stream_followup_stride_ramp: tuple[int, ...],
    stream_followup_stride: int,
) -> int:
    window_end = 0
    for stride in (stream_stride, *stream_followup_stride_ramp):
        if window_end + stride > frame_count:
            return window_end
        else:
            window_end += stride
    return (
        window_end
        + (frame_count - window_end) // stream_followup_stride * stream_followup_stride
    )
