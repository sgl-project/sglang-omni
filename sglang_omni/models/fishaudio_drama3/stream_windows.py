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
