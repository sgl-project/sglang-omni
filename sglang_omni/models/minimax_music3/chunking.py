# SPDX-License-Identifier: Apache-2.0
"""Pure chunk boundary and overlap helpers for MiniMax Music 3."""

from __future__ import annotations

from dataclasses import dataclass

from .constants import AR_CHUNK_FRAMES, AR_CHUNK_HOP_FRAMES


@dataclass(frozen=True)
class ChunkWindow:
    index: int
    start: int
    end: int
    is_first: bool
    is_last: bool

    @property
    def length(self) -> int:
        return self.end - self.start


def chunk_windows(
    frames: int,
) -> list[ChunkWindow]:
    """Return fixed-size windows with a 50% overlap."""

    frames = int(frames)
    if frames == 0:
        return []
    if frames <= AR_CHUNK_FRAMES:
        return [ChunkWindow(0, 0, frames, True, True)]

    windows: list[ChunkWindow] = []
    index = 0
    start = 0
    while start < frames:
        end = min(start + AR_CHUNK_FRAMES, frames)
        windows.append(
            ChunkWindow(
                index=index,
                start=start,
                end=end,
                is_first=index == 0,
                is_last=end >= frames,
            )
        )
        if end >= frames:
            break
        index += 1
        start += AR_CHUNK_HOP_FRAMES
    return windows


_DAV_HOP_SAMPLES = 512
_WINDOW_MEL_FRAMES = 344
_BLEND_MEL_FRAMES = _WINDOW_MEL_FRAMES // 4


def overlap_mel_length() -> int:
    return _WINDOW_MEL_FRAMES // 2


def crop_sample_bounds(
    window: ChunkWindow,
) -> tuple[int, int]:
    """Return (left_samples, right_samples) to trim from a decoded chunk."""

    left = 0 if window.is_first else _BLEND_MEL_FRAMES * _DAV_HOP_SAMPLES
    right = (
        0
        if window.is_last
        else (_WINDOW_MEL_FRAMES - _BLEND_MEL_FRAMES) * _DAV_HOP_SAMPLES
    )
    return left, right


__all__ = [
    "ChunkWindow",
    "chunk_windows",
    "crop_sample_bounds",
    "overlap_mel_length",
]
