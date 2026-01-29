# SPDX-License-Identifier: Apache-2.0
"""Model-agnostic image frontend utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image

from .cache_key import compute_cache_key, hash_bytes


def load_image_path(path: str | Path) -> Image.Image:
    """Load an image from disk as RGB."""
    return Image.open(path).convert("RGB")


def compute_image_cache_key(images: Any) -> str | None:
    """Compute cache key from raw image inputs (paths, URLs, PIL Images).

    This should be called BEFORE ensure_image_list() to capture original
    paths/URLs which are much cheaper to hash than pixel data.
    """

    def _item_to_part(item: Any) -> str | None:
        if isinstance(item, (str, Path)):
            # Path or URL: use string directly (very cheap)
            return f"path:{item}"
        if isinstance(item, Image.Image):
            # PIL Image: hash mode + size + content
            meta = f"{item.mode}|{item.size}"
            content_hash = hash_bytes(item.tobytes())
            return f"img:{meta}:{content_hash}"
        # Unknown type, skip cache
        return None

    key = compute_cache_key(images, item_to_part=_item_to_part)
    return f"image:{key}" if key else None


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
