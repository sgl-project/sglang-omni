# SPDX-License-Identifier: Apache-2.0
"""Shared LRU cache for uploaded-speaker feature artifacts."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from threading import RLock
from typing import Any

import numpy as np

from sglang_omni.scheduling.stage_cache import StageOutputCache

DEFAULT_SPEAKER_CACHE_BYTES = 512 * 1024 * 1024
_KEY_SEPARATOR = "\x1f"


@dataclass(frozen=True)
class SpeakerCacheKey:
    """Stable identity for one model-side uploaded-speaker artifact."""

    model_type: str
    voice_name: str
    voice_version: int
    artifact_kind: str


class SpeakerArtifactCache:
    """Process-wide bounded LRU for voice features shared by TTS stages."""

    def __init__(self, max_bytes: int = DEFAULT_SPEAKER_CACHE_BYTES) -> None:
        if max_bytes <= 0:
            raise ValueError("speaker cache max_bytes must be positive")
        self.max_bytes = int(max_bytes)
        self.cache = StageOutputCache(
            max_bytes=self.max_bytes,
            size_fn=estimate_cache_bytes,
        )
        self.hit_count = 0
        self.miss_count = 0
        self.delete_invalidation_counter = 0
        self.lock = RLock()

    def get(self, key: SpeakerCacheKey) -> Any | None:
        with self.lock:
            value = self.cache.get(encode_key(key))
            if value is None:
                self.miss_count += 1
                return None
            self.hit_count += 1
            return value

    def put(self, key: SpeakerCacheKey, value: Any) -> None:
        with self.lock:
            self.cache.put(encode_key(key), value)

    def clear_voice(self, voice_name: str) -> None:
        normalized_voice = voice_name.lower()
        with self.lock:
            removed_count = self.cache.remove_if(
                lambda key: encoded_key_voice_name(key).lower() == normalized_voice
            )
            self.delete_invalidation_counter += removed_count

    def clear(self) -> None:
        with self.lock:
            self.cache.clear()

    def stats(self) -> dict[str, int]:
        with self.lock:
            return {
                "entries": len(self.cache),
                "memory_bytes": self.cache.current_bytes,
                "max_bytes": self.max_bytes,
                "hit_count": self.hit_count,
                "miss_count": self.miss_count,
                "eviction_count": self.cache.eviction_count,
                "delete_invalidation_counter": self.delete_invalidation_counter,
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


def encode_key(key: SpeakerCacheKey) -> str:
    return _KEY_SEPARATOR.join(
        (
            key.model_type,
            key.voice_name,
            str(int(key.voice_version)),
            key.artifact_kind,
        )
    )


def encoded_key_voice_name(key: str) -> str:
    parts = key.split(_KEY_SEPARATOR, 3)
    return parts[1] if len(parts) == 4 else ""


_GLOBAL_SPEAKER_CACHE = SpeakerArtifactCache()


def get_speaker_artifact_cache() -> SpeakerArtifactCache:
    return _GLOBAL_SPEAKER_CACHE
