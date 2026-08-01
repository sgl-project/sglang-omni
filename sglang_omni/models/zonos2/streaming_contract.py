# SPDX-License-Identifier: Apache-2.0
"""Shared producer/consumer thresholds for ZONOS2 streaming audio."""

from __future__ import annotations

from sglang_omni.models.zonos2.payload_types import N_CODEBOOKS

DEFAULT_ZONOS2_STREAM_STEADY_CHUNK_FRAMES = 40
DEFAULT_ZONOS2_STREAM_INITIAL_CHUNK_FRAMES = 40
DEFAULT_ZONOS2_STREAM_OVERLAP_FRAMES = 2
ZONOS2_STREAM_LOOKAHEAD_FRAMES = 2 * N_CODEBOOKS


def zonos2_producer_first_flush_rows(initial_chunk_frames: int) -> int:
    chunk_frames = int(initial_chunk_frames)
    if chunk_frames <= 0:
        chunk_frames = DEFAULT_ZONOS2_STREAM_STEADY_CHUNK_FRAMES
    chunk_frames = max(
        chunk_frames,
        DEFAULT_ZONOS2_STREAM_OVERLAP_FRAMES + 1,
    )
    return chunk_frames + ZONOS2_STREAM_LOOKAHEAD_FRAMES


DEFAULT_ZONOS2_PRODUCER_FIRST_FLUSH_ROWS = zonos2_producer_first_flush_rows(
    DEFAULT_ZONOS2_STREAM_INITIAL_CHUNK_FRAMES
)


__all__ = [
    "DEFAULT_ZONOS2_PRODUCER_FIRST_FLUSH_ROWS",
    "DEFAULT_ZONOS2_STREAM_INITIAL_CHUNK_FRAMES",
    "DEFAULT_ZONOS2_STREAM_OVERLAP_FRAMES",
    "DEFAULT_ZONOS2_STREAM_STEADY_CHUNK_FRAMES",
    "ZONOS2_STREAM_LOOKAHEAD_FRAMES",
    "zonos2_producer_first_flush_rows",
]
