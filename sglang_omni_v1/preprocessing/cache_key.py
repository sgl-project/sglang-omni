from __future__ import annotations

from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlparse

import numpy as np
import torch
import xxhash
from PIL import Image


def _is_url_like(s: str) -> bool:
    """Quick check if a string is a URL (http, https, data, file)."""
    parsed = urlparse(s)
    return bool(parsed.scheme and parsed.scheme in ("http", "https", "data", "file"))


def _hash_joined(parts: list[str]) -> str:
    return xxhash.xxh3_64("|".join(parts).encode("utf-8")).hexdigest()


def hash_bytes(payload: bytes | bytearray | memoryview) -> str:
    return xxhash.xxh3_64(payload).hexdigest()


def hash_file_sampled(
    path: str | Path,
    head_size: int = 8192,
    tail_size: int = 8192,
) -> str:
    """Generate hash from file head + tail + size (fast sampling strategy).

    This avoids reading the entire file while still detecting most changes.
    For compressed formats (JPEG, PNG, WAV, MP4), any content change typically
    affects file size and/or head/tail bytes.
    """
    path = Path(path)
    file_size = path.stat().st_size

    with open(path, "rb") as f:
        head = f.read(head_size)
        if file_size > head_size + tail_size:
            f.seek(-tail_size, 2)  # Seek from end
            tail = f.read(tail_size)
        else:
            tail = b""  # Small file, head already covers it

    payload = head + tail + f"|size:{file_size}".encode()
    return xxhash.xxh3_64(payload).hexdigest()


def hash_media_item(item: Any) -> str | None:
    """Generate hash for a single media item (unified logic for image/audio/video).

    Supported types:
    - str/Path: local file -> sampled hash; URL -> string hash
    - PIL.Image: mode + size + content hash
    - numpy.ndarray: dtype + shape + content hash
    - torch.Tensor: dtype + shape + content hash
    - bytes/bytearray: content hash

    Returns None for unsupported types (caller should skip caching).
    """
    # File path or URL
    if isinstance(item, (str, Path)):
        s = str(item)
        if _is_url_like(s):
            return f"url:{hash_bytes(s.encode())}"
        p = Path(s)
        if p.exists() and p.is_file():
            return f"file:{hash_file_sampled(p)}"
        return f"url:{hash_bytes(s.encode())}"

    # PIL Image
    if isinstance(item, Image.Image):
        meta = f"{item.mode}|{item.size}"
        content_hash = hash_bytes(item.tobytes())
        return f"pil:{meta}:{content_hash}"

    # numpy array
    if isinstance(item, np.ndarray):
        meta = f"{item.dtype}|{item.shape}"
        content_hash = hash_bytes(item.tobytes())
        return f"np:{meta}:{content_hash}"

    # torch Tensor
    if isinstance(item, torch.Tensor):
        cpu = item.detach().cpu()
        meta = f"{cpu.dtype}|{tuple(cpu.shape)}"
        content_hash = hash_bytes(cpu.numpy().tobytes())
        return f"pt:{meta}:{content_hash}"

    # Raw bytes
    if isinstance(item, (bytes, bytearray, memoryview)):
        return f"bytes:{hash_bytes(item)}"

    # Unsupported type
    return None


def compute_media_cache_key(items: Any, *, prefix: str) -> str | None:
    """Compute cache key for media items (image/audio/video).

    Args:
        items: Single item or list of items
        prefix: Type prefix (e.g., "image", "audio", "video")

    Returns:
        Cache key string or None if any item is unsupported.
    """
    if items is None:
        return None
    seq = items if isinstance(items, list) else [items]
    if not seq:
        return None

    parts: list[str] = []
    for item in seq:
        part = hash_media_item(item)
        if part is None:
            return None
        parts.append(part)

    return f"{prefix}:{_hash_joined(parts)}"


def compute_cache_key(
    items: Any, *, item_to_part: Callable[[Any], str | None]
) -> str | None:
    """Compute cache key from a list-like input (legacy API).

    The item_to_part callback must return a string part or None to
    indicate the item type is unsupported (no cache key).

    Note: Prefer compute_media_cache_key() for new code.
    """
    if items is None:
        return None
    seq = items if isinstance(items, list) else [items]
    if not seq:
        return None

    parts: list[str] = []
    for item in seq:
        part = item_to_part(item)
        if part is None:
            return None
        parts.append(part)

    return _hash_joined(parts)
