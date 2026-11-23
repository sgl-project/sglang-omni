# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations


def window_stride(
    strides: tuple[int, ...], stream_followup_stride: int, window_index: int
) -> int:
    """New frames of stream window window_index: the scheduled stride while the ramp
    lasts, then the follow-up stride."""
    if window_index < len(strides):
        return strides[window_index]
    else:
        return stream_followup_stride


def stream_window_lengths(
    *,
    stream_stride: int,
    stream_followup_stride_ramp: tuple[int, ...],
    stream_followup_stride: int,
    stream_overlap_tokens: int,
) -> tuple[int, ...]:
    return tuple(
        sorted(
            stream_window_discards(
                stream_stride=stream_stride,
                stream_followup_stride_ramp=stream_followup_stride_ramp,
                stream_followup_stride=stream_followup_stride,
                stream_overlap_tokens=stream_overlap_tokens,
            )
        )
    )


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
        stride = window_stride(strides, stream_followup_stride, window_index)
        window_start_frame = max(0, emitted_frames - stream_overlap_tokens)
        discards[emitted_frames + stride - window_start_frame] = (
            emitted_frames - window_start_frame
        )
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
    window_shapes: set[tuple[int, int]] = set()
    emitted_frames = 0
    window_index = 0
    while True:
        stride = window_stride(strides, stream_followup_stride, window_index)
        emitted_frames += stride
        window_index += 1
        overlap_frames = min(emitted_frames, stream_overlap_tokens)
        next_stride = window_stride(strides, stream_followup_stride, window_index)
        for remaining_frames in range(1, next_stride + 1):
            window_shapes.add((overlap_frames + remaining_frames, overlap_frames))
        if window_index >= len(strides) and emitted_frames >= stream_overlap_tokens:
            return window_shapes
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


def crossfade_frames(stream_crossfade_samples: int, frame_length: int) -> int:
    """Frames of the overlap a crossfade reads back into."""
    return -(-stream_crossfade_samples // frame_length)


def discarded_frames(overlap_frames: int, crossfade_frames: int) -> int:
    """Leading frames of a window whose audio the stream never reads."""
    return max(0, overlap_frames - crossfade_frames)
