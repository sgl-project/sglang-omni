# SPDX-License-Identifier: Apache-2.0
"""Shared helpers for text-to-speech streaming models."""

from __future__ import annotations

from typing import Any, Mapping

INITIAL_CODEC_CHUNK_FRAMES_PARAM = "initial_codec_chunk_frames"


def resolve_initial_codec_chunk_frames(
    params: Mapping[str, Any] | None,
    *,
    steady_chunk_frames: int,
) -> int:
    """Return the request-level first codec chunk size, clamped to steady size."""
    if steady_chunk_frames <= 0:
        raise ValueError(
            f"steady_chunk_frames must be positive, got {steady_chunk_frames}"
        )
    if params is None:
        return 0

    value = params.get(INITIAL_CODEC_CHUNK_FRAMES_PARAM)
    if value is None:
        return 0

    try:
        frames = int(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            f"{INITIAL_CODEC_CHUNK_FRAMES_PARAM} must be an integer"
        ) from exc
    if frames < 0:
        raise ValueError(f"{INITIAL_CODEC_CHUNK_FRAMES_PARAM} must be >= 0")

    return min(frames, int(steady_chunk_frames))
