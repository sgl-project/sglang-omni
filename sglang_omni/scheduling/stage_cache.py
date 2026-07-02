# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch

from sglang_omni.cache import (
    ArtifactHandle,
    CacheKey,
    CacheOwner,
    CacheSelector,
    LocalCachePlane,
    get_global_cache_plane,
)


@dataclass
class _CacheEntry:
    data: Any
    size_bytes: int


def _detach_value(value: Any, *, device: torch.device | None) -> Any:
    if isinstance(value, torch.Tensor):
        value = value.detach()
        if device is not None:
            value = value.to(device=device)
        return value
    if isinstance(value, dict):
        return {key: _detach_value(item, device=device) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_detach_value(item, device=device) for item in value)
    return value


def _value_size_bytes(value: Any) -> int:
    if isinstance(value, torch.Tensor):
        return int(value.numel() * value.element_size())
    if isinstance(value, dict):
        return sum(_value_size_bytes(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return sum(_value_size_bytes(item) for item in value)
    return 0


class StageOutputCache:
    """Small in-memory LRU cache for non-AR stage outputs."""

    def __init__(
        self,
        max_size: int | None = None,
        max_bytes: int | None = None,
        cache_device: torch.device | str | None = None,
        size_fn: Callable[[Any], int] | None = None,
        cache_plane: LocalCachePlane | None = None,
        cache_namespace: str | None = None,
        cache_kind: str = "stage_output",
        cache_owner: CacheOwner | None = None,
        cache_ttl_s: float | None = None,
    ) -> None:
        if max_size is not None and max_size <= 0:
            raise ValueError("max_size must be positive when set")
        if max_bytes is not None and max_bytes <= 0:
            raise ValueError("max_bytes must be positive when set")
        if isinstance(cache_device, str):
            cache_device = torch.device(cache_device)
        self._cache: OrderedDict[str, _CacheEntry] = OrderedDict()
        self.max_size = max_size
        self.max_bytes = max_bytes
        self.cache_device = cache_device
        self.current_bytes = 0
        self.eviction_count = 0
        self._size_fn = size_fn or _value_size_bytes
        self._cache_namespace = cache_namespace
        self._cache_kind = cache_kind
        self._cache_owner = cache_owner or CacheOwner()
        self._cache_ttl_s = cache_ttl_s
        self._cache_plane = (
            cache_plane
            if cache_plane is not None
            else (get_global_cache_plane() if cache_namespace is not None else None)
        )

    def get(self, key: str | None) -> Any | None:
        if key is None:
            return None
        key = str(key)
        entry = self._cache.get(key)
        if entry is None:
            self._record_plane_miss(key)
            return None
        self._cache.move_to_end(key)
        self._touch_plane(key)
        return entry.data

    def put(self, key: str | None, data: Any) -> None:
        if key is None:
            return
        key = str(key)
        size_bytes = self._size_fn(data)
        old_entry = self._cache.pop(key, None)
        if old_entry is not None:
            self.current_bytes -= old_entry.size_bytes
            self._remove_plane_entry(key)
        if self.max_bytes is not None and size_bytes > self.max_bytes:
            return
        self._cache[key] = _CacheEntry(
            data=_detach_value(data, device=self.cache_device),
            size_bytes=size_bytes,
        )
        self.current_bytes += size_bytes
        self._cache.move_to_end(key)
        self._publish_plane_entry(key, size_bytes)
        self._evict_over_budget()

    def clear(self) -> None:
        for key in list(self._cache):
            self._remove_plane_entry(key)
        self._cache.clear()
        self.current_bytes = 0

    def remove_if(self, predicate: Callable[[str], bool]) -> int:
        removed = 0
        for key in list(self._cache):
            if not predicate(key):
                continue
            entry = self._cache.pop(key)
            self.current_bytes -= entry.size_bytes
            self._remove_plane_entry(key)
            removed += 1
        return removed

    def bind_session(self, session_id: str | None) -> None:
        if self._cache_plane is None or not session_id:
            return
        self._cache_plane.bind_session(str(session_id), self._cache_owner)

    def start_build(self, key: str | None) -> bool:
        plane_key = self._plane_key_or_none(key)
        if plane_key is None:
            return True
        return self._cache_plane.start_build(
            plane_key,
            owner=self._cache_owner,
            ttl_s=self._cache_ttl_s,
        )

    def wait_ready(self, key: str | None, timeout_s: float | None = None) -> Any | None:
        plane_key = self._plane_key_or_none(key)
        if plane_key is None:
            return None
        ready = self._cache_plane.wait_ready(plane_key, timeout_s=timeout_s)
        if ready is None:
            return None
        cached = self.get(str(key))
        if cached is None:
            self._cache_plane.remove(plane_key)
        return cached

    def fail_build(self, key: str | None, error: BaseException | str) -> None:
        plane_key = self._plane_key_or_none(key)
        if plane_key is None:
            return
        self._cache_plane.fail_build(plane_key, error)

    def __len__(self) -> int:
        return len(self._cache)

    def _evict_over_budget(self) -> None:
        while self.max_size is not None and len(self._cache) > self.max_size:
            key, entry = self._cache.popitem(last=False)
            self.current_bytes -= entry.size_bytes
            self._remove_plane_entry(key)
            self.eviction_count += 1
        while self.max_bytes is not None and self.current_bytes > self.max_bytes:
            if not self._cache:
                self.current_bytes = 0
                return
            key, entry = self._cache.popitem(last=False)
            self.current_bytes -= entry.size_bytes
            self._remove_plane_entry(key)
            self.eviction_count += 1

    def _plane_key(self, key: str) -> CacheKey | None:
        if self._cache_plane is None or self._cache_namespace is None:
            return None
        return CacheKey(
            namespace=self._cache_namespace,
            kind=self._cache_kind,
            digest=str(key),
            stage_name=self._cache_owner.stage_name,
        )

    def _plane_key_or_none(self, key: str | None) -> CacheKey | None:
        if key is None:
            return None
        return self._plane_key(str(key))

    def _publish_plane_entry(self, key: str, size_bytes: int) -> None:
        plane_key = self._plane_key(key)
        if plane_key is None:
            return
        device = str(self.cache_device) if self.cache_device is not None else None
        self._cache_plane.publish(
            plane_key,
            ArtifactHandle(backend="stage_output", ref={"key": key}),
            owner=self._cache_owner,
            size_bytes=size_bytes,
            device=device,
            ttl_s=self._cache_ttl_s,
        )

    def _remove_plane_entry(self, key: str) -> None:
        plane_key = self._plane_key(key)
        if plane_key is None:
            return
        self._cache_plane.remove(plane_key)

    def _touch_plane(self, key: str) -> None:
        plane_key = self._plane_key(key)
        if plane_key is None:
            return
        self._cache_plane.lookup(plane_key)

    def _record_plane_miss(self, key: str) -> None:
        plane_key = self._plane_key(key)
        if plane_key is None:
            return
        self._cache_plane.record_miss(plane_key)

    def invalidate_plane_entries(self) -> int:
        if self._cache_plane is None or self._cache_namespace is None:
            return 0
        return self._cache_plane.invalidate(
            CacheSelector(
                namespace=self._cache_namespace,
                kind=self._cache_kind,
                stage_name=self._cache_owner.stage_name,
            )
        )
