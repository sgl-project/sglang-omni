# SPDX-License-Identifier: Apache-2.0
"""Model-agnostic image preprocessing utilities."""

from __future__ import annotations

import asyncio
import base64
from collections.abc import Mapping
from io import BytesIO
from pathlib import Path

import torch
from PIL import Image, UnidentifiedImageError

from sglang_omni.preprocessing.resource_connector import MultiModalResourceConnector

from .base import MediaIO, is_url
from .cache_key import compute_media_cache_key
from .resource_connector import await_media_cleanup


def load_image_path(path: str | Path) -> Image.Image:
    """Load an image from disk as RGB."""
    return Image.open(path).convert("RGB")


class ImageMediaIO(MediaIO[Image.Image]):
    """MediaIO implementation for image files."""

    def __init__(self, *, image_mode: str = "RGB", **kwargs) -> None:
        """Initialize ImageMediaIO.

        Args:
            image_mode: Target image mode (default: "RGB").
            **kwargs: Additional arguments (for compatibility with MultiModalResourceConnector).
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


def compute_image_cache_key(images: object) -> str | None:
    """Compute a cache key from loaded images.

    Pass decoded images, such as the output of ensure_image_list_async. A URL or
    path can name different pixels over time, so it is not a stable key.
    """
    return compute_media_cache_key(images, prefix="image")


async def ensure_image_list_async(
    images: object,
    *,
    image_mode: str = "RGB",
    media_connector: MultiModalResourceConnector | None = None,
) -> list[object]:
    """Asynchronously normalize image inputs into a list.

    Args:
        images: Image input(s) - can be a path, URL, PIL Image, or list.
        image_mode: Target image mode (default: "RGB").
        media_connector: Optional MultiModalResourceConnector instance. If None, uses
                        the global connector.

    Returns:
        List of normalized PIL Images.
    """
    if images is None:
        return []
    else:
        pass
    items = images if isinstance(images, list) else [images]

    # Import here to avoid circular dependency
    if media_connector is None:
        from .resource_connector import get_global_resource_connector

        media_connector = get_global_resource_connector()
    else:
        pass

    # Collect coroutines for URL items
    coroutines: list[asyncio.Task[Image.Image]] = []
    url_indices: list[int] = []
    normalized: list[object] = []

    try:
        # note (Teery): First pass: identify URL items and create coroutines
        for idx, item in enumerate(items):
            if isinstance(item, (str, Path)):
                if is_url(item):
                    # note (Teery): Create coroutine for async URL fetching
                    coro = media_connector.fetch_image_async(
                        str(item), image_mode=image_mode
                    )
                    task = asyncio.create_task(coro)
                    coroutines.append(task)
                    url_indices.append(idx)
                    normalized.append(None)  # note (Teery): Placeholder
                else:
                    path = media_connector.local_media_path(item)
                    normalized.append(load_image_path(path))
            else:
                # note (Teery): Already processed (PIL Image, etc.)
                normalized.append(item)

        # note (Teery): Wait for all URL fetches to complete
        if coroutines:
            results = await asyncio.gather(*coroutines)
            # note (Teery): Fill in the results at the correct indices
            for url_idx, result in zip(url_indices, results):
                normalized[url_idx] = result
        else:
            pass
    finally:
        for task in coroutines:
            if not task.done():
                task.cancel()
            else:
                pass

        async def cleanup_loaders() -> None:
            await asyncio.gather(*coroutines, return_exceptions=True)

        await await_media_cleanup(cleanup_loaders())

    return normalized


def build_image_mm_inputs(
    hf_inputs: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor | None]:
    """Extract standard image tensors from HF processor outputs."""
    return {
        "pixel_values": hf_inputs.get("pixel_values"),
        "image_grid_thw": hf_inputs.get("image_grid_thw"),
    }
