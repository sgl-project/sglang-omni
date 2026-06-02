# SPDX-License-Identifier: Apache-2.0
"""Shared LRU cache for uploaded-speaker feature artifacts."""

from __future__ import annotations

import sys
from collections import OrderedDict
from dataclasses import dataclass
from threading import RLock
from typing import Any

import numpy as np

DEFAULT_SPEAKER_CACHE_BYTES = 512 * 1024 * 1024


@dataclass(frozen=True)
class SpeakerCacheKey:
    """Stable identity for one model-side uploaded-speaker artifact."""

    model_type: str
    voice_name: str
    voice_version: int
    artifact_kind: str


class SpeakerArtifactCache:
    """Process-wide bounded LRU for voice features shared by all TTS models."""

    def __init__(self, max_bytes: int = DEFAULT_SPEAKER_CACHE_BYTES) -> None:
        if max_bytes <= 0:
            raise ValueError("speaker cache max_bytes must be positive")
        self.max_bytes = int(max_bytes)
        self._entries: OrderedDict[SpeakerCacheKey, tuple[Any, int]] = OrderedDict()
        self._memory_bytes = 0
        self._hit_count = 0
        self._miss_count = 0
        self._eviction_count = 0
        self._delete_invalidation_counter = 0
        self._lock = RLock()

    def get(self, key: SpeakerCacheKey) -> Any | None:
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                self._miss_count += 1
                return None
            value, size = entry
            self._entries.move_to_end(key)
            self._hit_count += 1
            return value

    def put(self, key: SpeakerCacheKey, value: Any) -> None:
        size = estimate_cache_bytes(value)
        with self._lock:
            previous = self._entries.pop(key, None)
            if previous is not None:
                self._memory_bytes -= previous[1]
            if size > self.max_bytes:
                return
            while self._memory_bytes + size > self.max_bytes and self._entries:
                _, (_, removed_size) = self._entries.popitem(last=False)
                self._memory_bytes -= removed_size
                self._eviction_count += 1
            self._entries[key] = (value, size)
            self._memory_bytes += size

    def clear_voice(self, voice_name: str) -> None:
        normalized_voice = voice_name.lower()
        with self._lock:
            for key in list(self._entries):
                if key.voice_name.lower() == normalized_voice:
                    _, size = self._entries.pop(key)
                    self._memory_bytes -= size
            self._delete_invalidation_counter += 1

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()
            self._memory_bytes = 0

    def stats(self) -> dict[str, int]:
        with self._lock:
            return {
                "entries": len(self._entries),
                "memory_bytes": self._memory_bytes,
                "max_bytes": self.max_bytes,
                "hit_count": self._hit_count,
                "miss_count": self._miss_count,
                "eviction_count": self._eviction_count,
                "delete_invalidation_counter": self._delete_invalidation_counter,
            }


def estimate_cache_bytes(value: Any) -> int:
    """Estimate memory held by common artifact containers."""

    if value is None:
        return 0
    if isinstance(value, bytes | bytearray | memoryview):
        return len(value)
    if isinstance(value, str):
        return len(value.encode("utf-8"))
    if isinstance(value, np.ndarray):
        return int(value.nbytes)
    if hasattr(value, "numel") and hasattr(value, "element_size"):
        try:
            return int(value.numel() * value.element_size())
        except Exception:
            return sys.getsizeof(value)
    if isinstance(value, dict):
        return sys.getsizeof(value) + sum(
            estimate_cache_bytes(key) + estimate_cache_bytes(item)
            for key, item in value.items()
        )
    if isinstance(value, list | tuple | set | frozenset):
        return sys.getsizeof(value) + sum(estimate_cache_bytes(item) for item in value)
    return sys.getsizeof(value)


_GLOBAL_SPEAKER_CACHE = SpeakerArtifactCache()


def get_speaker_artifact_cache() -> SpeakerArtifactCache:
    return _GLOBAL_SPEAKER_CACHE
