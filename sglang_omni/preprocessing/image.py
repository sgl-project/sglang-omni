# SPDX-License-Identifier: Apache-2.0
"""Model-agnostic image preprocessing utilities."""

from __future__ import annotations

import base64
from io import BytesIO
from pathlib import Path
from typing import Any

from PIL import Image, UnidentifiedImageError

from .base import MediaIO, _is_url
from .cache_key import compute_media_cache_key


def load_image_path(path: str | Path) -> Image.Image:
    """Load an image from disk as RGB."""
    return Image.open(path).convert("RGB")


class ImageMediaIO(MediaIO[Image.Image]):
    """MediaIO implementation for image files."""

    def __init__(self, *, image_mode: str = "RGB", **kwargs) -> None:
        """Initialize ImageMediaIO.

        Args:
            image_mode: Target image mode (default: "RGB").
            **kwargs: Additional arguments (for compatibility with MediaConnector).
        """
        super().__init__()
        self.image_mode = image_mode
        self.kwargs = kwargs

    def load_bytes(self, data: bytes) -> Image.Image:
        """Load image from raw bytes."""
        try:
            return Image.open(BytesIO(data)).convert(self.image_mode)
        except UnidentifiedImageError as e:
            raise ValueError(f"Failed to identify image: {e}") from e

    def load_base64(
        self,
        media_type: str,
        data: str,
    ) -> Image.Image:
        """Load image from base64-encoded data."""
        return self.load_bytes(base64.b64decode(data))

    def load_file(self, filepath: Path) -> Image.Image:
        """Load image from a local file path."""
        try:
            return Image.open(filepath).convert(self.image_mode)
        except UnidentifiedImageError as e:
            raise ValueError(f"Failed to identify image: {e}") from e


def compute_image_cache_key(images: Any) -> str | None:
    """Compute cache key from raw image inputs (paths, URLs, PIL Images).

    This should be called BEFORE ensure_image_list() to capture original
    paths/URLs which are much cheaper to hash than pixel data.
    """
    return compute_media_cache_key(images, prefix="image")


def ensure_image_list(
    images: Any,
    *,
    image_mode: str = "RGB",
    media_connector: Any | None = None,
) -> list[Any]:
    """Normalize image inputs into a list.

    Supports local file paths, URLs (HTTP/HTTPS, data URLs, file URLs),
    and PIL Images.

    Args:
        images: Image input(s) - can be a path, URL, PIL Image, or list.
        image_mode: Target image mode (default: "RGB").
        media_connector: Optional MediaConnector instance. If None, uses
                        the global connector.

    Returns:
        List of normalized PIL Images.
    """
    if images is None:
        return []
    items = images if isinstance(images, list) else [images]
    normalized: list[Any] = []

    # Import here to avoid circular dependency
    if media_connector is None:
        from .media_connector import get_global_media_connector

        media_connector = get_global_media_connector()

    for item in items:
        if isinstance(item, (str, Path)):
            if _is_url(item):
                # Load from URL
                image = media_connector.fetch_image(str(item), image_mode=image_mode)
                normalized.append(image)
            else:
                # Load from local path
                # Note: load_image_path always converts to RGB, which matches default image_mode
                normalized.append(load_image_path(item))
        else:
            # Already processed (PIL Image, etc.)
            normalized.append(item)
    return normalized


def build_image_mm_inputs(hf_inputs: dict[str, Any]) -> dict[str, Any]:
    """Extract standard image tensors from HF processor outputs."""
    return {
        "pixel_values": hf_inputs.get("pixel_values"),
        "image_grid_thw": hf_inputs.get("image_grid_thw"),
    }
