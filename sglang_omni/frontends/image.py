# SPDX-License-Identifier: Apache-2.0
"""Model-agnostic image frontend utilities."""

from __future__ import annotations

from typing import Any


def ensure_image_list(images: Any) -> list[Any]:
    """Normalize image inputs into a list."""
    if images is None:
        return []
    if isinstance(images, list):
        return images
    return [images]


def build_image_mm_inputs(hf_inputs: dict[str, Any]) -> dict[str, Any]:
    """Extract standard image tensors from HF processor outputs."""
    return {
        "pixel_values": hf_inputs.get("pixel_values"),
        "image_grid_thw": hf_inputs.get("image_grid_thw"),
    }
