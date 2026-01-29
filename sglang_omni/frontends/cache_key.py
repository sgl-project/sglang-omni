from __future__ import annotations

from typing import Any, Callable

import xxhash


def _hash_joined(parts: list[str]) -> str:
    return xxhash.xxh3_64("|".join(parts).encode("utf-8")).hexdigest()


def hash_bytes(payload: bytes | bytearray | memoryview) -> str:
    return xxhash.xxh3_64(payload).hexdigest()


def compute_cache_key(
    items: Any, *, item_to_part: Callable[[Any], str | None]
) -> str | None:
    """Compute cache key from a list-like input.

    The item_to_part callback must return a string part or None to
    indicate the item type is unsupported (no cache key).
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
