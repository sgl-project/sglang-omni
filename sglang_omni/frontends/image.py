# SPDX-License-Identifier: Apache-2.0
"""Model-agnostic image frontend utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image


def load_image_path(path: str | Path) -> Image.Image:
    """Load an image from disk as RGB."""
    return Image.open(path).convert("RGB")


def ensure_image_list(images: Any) -> list[Any]:
    """Normalize image inputs into a list."""
    if images is None:
        return []
    items = images if isinstance(images, list) else [images]
    normalized: list[Any] = []
    for item in items:
        if isinstance(item, (str, Path)):
            normalized.append(load_image_path(item))
        else:
            normalized.append(item)
    return normalized


def build_image_mm_inputs(hf_inputs: dict[str, Any]) -> dict[str, Any]:
    """Extract standard image tensors from HF processor outputs."""
    return {
        "pixel_values": hf_inputs.get("pixel_values"),
        "image_grid_thw": hf_inputs.get("image_grid_thw"),
    }
