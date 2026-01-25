# SPDX-License-Identifier: Apache-2.0
"""Model-agnostic video frontend utilities."""

from __future__ import annotations

from typing import Any


def ensure_video_list(videos: Any) -> list[Any]:
    """Normalize video inputs into a list."""
    if videos is None:
        return []
    if isinstance(videos, list):
        return videos
    return [videos]


def build_video_mm_inputs(hf_inputs: dict[str, Any]) -> dict[str, Any]:
    """Extract standard video tensors from HF processor outputs.

    This is a placeholder schema; refine it after more models are integrated.
    """
    return {
        "pixel_values_videos": hf_inputs.get("pixel_values_videos"),
        "video_grid_thw": hf_inputs.get("video_grid_thw"),
    }
