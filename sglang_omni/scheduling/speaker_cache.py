# SPDX-License-Identifier: Apache-2.0
"""Shared LRU cache for uploaded-speaker feature artifacts."""

from __future__ import annotations

import concurrent.futures
import sys
from collections.abc import Callable
from dataclasses import dataclass
from threading import RLock, get_ident
from typing import Any, TypeVar

import numpy as np

from sglang_omni.scheduling.stage_cache import StageOutputCache

DEFAULT_SPEAKER_CACHE_BYTES = 512 * 1024 * 1024
_KEY_SEPARATOR = "\x1f"
_ValueT = TypeVar("_ValueT")


@dataclass(frozen=True)
class SpeakerCacheKey:
    """Stable identity for one model-side uploaded-speaker artifact."""

    model_type: str
    voice_name: str
    voice_version: int
    artifact_kind: str


@dataclass
class _SpeakerCacheFlight:
    future: concurrent.futures.Future[Any]
    owner_thread_id: int
    invalidated: bool = False


class SpeakerArtifactCache:
    """Process-wide bounded LRU for voice features shared by TTS stages."""

    def __init__(self, max_bytes: int = DEFAULT_SPEAKER_CACHE_BYTES) -> None:
        if max_bytes <= 0:
            raise ValueError("speaker cache max_bytes must be positive")
        self.max_bytes = int(max_bytes)
        self._cache = StageOutputCache(
            max_bytes=self.max_bytes,
            size_fn=estimate_cache_bytes,
        )
        self._hit_count = 0
        self._miss_count = 0
        self._singleflight_merged_count = 0
        self._delete_invalidation_counter = 0
        self._lock = RLock()
        self._inflight: dict[str, _SpeakerCacheFlight] = {}

    def get(self, key: SpeakerCacheKey) -> Any | None:
        with self._lock:
            value = self._cache.get(_encode_key(key))
            if value is None:
                self._miss_count += 1
                return None
            self._hit_count += 1
            return value

    def put(self, key: SpeakerCacheKey, value: Any) -> None:
        with self._lock:
            encoded_key = _encode_key(key)
            flight = self._inflight.pop(encoded_key, None)
            if flight is not None:
                flight.invalidated = True
            self._cache.put(encoded_key, value)

    def get_or_compute(
        self,
        key: SpeakerCacheKey,
        compute: Callable[[], _ValueT],
        *,
        accept_cached: Callable[[Any], bool] | None = None,
    ) -> _ValueT:
        """Return a cached artifact or compute one leader per exact key.

        Explicit puts and invalidation win over a computation that began earlier:
        the in-flight callers still receive their computed value, but the late
        leader cannot overwrite newer cache state or resurrect a deleted voice.
        """

        encoded_key = _encode_key(key)
        with self._lock:
            cached = self._cache.get(encoded_key)
            if cached is not None and (accept_cached is None or accept_cached(cached)):
                self._hit_count += 1
                return cached
            if cached is not None:
                self._cache.remove_if_same(encoded_key, cached)
            self._miss_count += 1
            flight = self._inflight.get(encoded_key)
            if flight is None:
                flight = _SpeakerCacheFlight(
                    concurrent.futures.Future(),
                    owner_thread_id=get_ident(),
                )
                self._inflight[encoded_key] = flight
                leader = True
            else:
                if flight.owner_thread_id == get_ident():
                    raise RuntimeError(
                        "Speaker cache compute re-entered its own in-flight "
                        f"computation for {key!r}"
                    )
                self._singleflight_merged_count += 1
                leader = False

        if not leader:
            return flight.future.result()

        try:
            value = compute()
            with self._lock:
                if not flight.invalidated and self._inflight.get(encoded_key) is flight:
                    self._cache.put(encoded_key, value)
                if self._inflight.get(encoded_key) is flight:
                    del self._inflight[encoded_key]
        except BaseException as exc:
            with self._lock:
                if self._inflight.get(encoded_key) is flight:
                    del self._inflight[encoded_key]
            flight.future.set_exception(exc)
            raise

        flight.future.set_result(value)
        return value

    def clear_voice(self, voice_name: str) -> None:
        normalized_voice = voice_name.lower()
        with self._lock:
            inflight_keys = [
                encoded_key
                for encoded_key in self._inflight
                if _encoded_key_voice_name(encoded_key).lower() == normalized_voice
            ]
            for encoded_key in inflight_keys:
                self._inflight.pop(encoded_key).invalidated = True
            removed_count = self._cache.remove_if(
                lambda key: _encoded_key_voice_name(key).lower() == normalized_voice
            )
            self._delete_invalidation_counter += removed_count

    def clear(self) -> None:
        with self._lock:
            for flight in self._inflight.values():
                flight.invalidated = True
            self._inflight.clear()
            self._cache.clear()

    def stats(self) -> dict[str, int]:
        with self._lock:
            return {
                "entries": len(self._cache),
                "memory_bytes": self._cache.current_bytes,
                "max_bytes": self.max_bytes,
                "hit_count": self._hit_count,
                "miss_count": self._miss_count,
                "singleflight_merged_count": self._singleflight_merged_count,
                "eviction_count": self._cache.eviction_count,
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


def _encode_key(key: SpeakerCacheKey) -> str:
    return _KEY_SEPARATOR.join(
        (
            key.model_type,
            key.voice_name,
            str(int(key.voice_version)),
            key.artifact_kind,
        )
    )


def _encoded_key_voice_name(key: str) -> str:
    parts = key.split(_KEY_SEPARATOR, 3)
    return parts[1] if len(parts) == 4 else ""


_GLOBAL_SPEAKER_CACHE = SpeakerArtifactCache()


def get_speaker_artifact_cache() -> SpeakerArtifactCache:
    return _GLOBAL_SPEAKER_CACHE
