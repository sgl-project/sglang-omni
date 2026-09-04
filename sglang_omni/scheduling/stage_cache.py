# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import logging
import threading
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch

logger = logging.getLogger(__name__)


@dataclass
class _CacheEntry:
    data: Any
    size_bytes: int


def _to_pinned_host(value: torch.Tensor) -> torch.Tensor:
    if value.device.type == "cpu" and value.is_pinned():
        return value
    host = torch.empty(value.shape, dtype=value.dtype, device="cpu", pin_memory=True)
    host.copy_(value, non_blocking=False)
    return host


def _detach_value(
    value: Any, *, device: torch.device | None, pin_memory: bool = False
) -> Any:
    if isinstance(value, torch.Tensor):
        value = value.detach()
        if device is not None:
            if pin_memory and device.type == "cpu":
                try:
                    return _to_pinned_host(value)
                except RuntimeError as exc:
                    logger.warning(
                        "StageOutputCache pinned host copy failed (%s); "
                        "falling back to pageable memory for this entry",
                        exc,
                    )
            value = value.to(device=device)
        return value
    if isinstance(value, dict):
        return {
            key: _detach_value(item, device=device, pin_memory=pin_memory)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return type(value)(
            _detach_value(item, device=device, pin_memory=pin_memory) for item in value
        )
    return value


def _value_size_bytes(value: Any) -> int:
    if isinstance(value, torch.Tensor):
        return int(value.numel() * value.element_size())
    if isinstance(value, (bytes, bytearray)):
        return len(value)
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
        pin_memory: bool = False,
    ) -> None:
        if max_size is not None and max_size < 0:
            raise ValueError("max_size must be non-negative")
        if max_bytes is not None and max_bytes < 0:
            raise ValueError("max_bytes must be non-negative")
        if isinstance(cache_device, str):
            cache_device = torch.device(cache_device)
        if pin_memory and (cache_device is None or cache_device.type != "cpu"):
            raise ValueError("pin_memory requires cache_device='cpu'")
        self._cache: OrderedDict[str, _CacheEntry] = OrderedDict()
        self.max_size = max_size
        self.max_bytes = max_bytes
        self.cache_device = cache_device
        # note (Jeffro): pinning needs a CUDA context to be meaningful, so on
        # CUDA-less hosts (CI runners, dev boxes) it degrades to pageable storage.
        self.pin_memory = bool(pin_memory) and torch.cuda.is_available()
        self.current_bytes = 0
        self.eviction_count = 0
        self._size_fn = size_fn or _value_size_bytes
        self._lock = threading.Lock()

    def get(self, key: str | None) -> Any | None:
        if key is None:
            return None
        key = str(key)
        with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                return None
            self._cache.move_to_end(key)
            return entry.data

    def put(self, key: str | None, data: Any) -> None:
        if key is None:
            return
        key = str(key)
        size_bytes = self._size_fn(data)
        with self._lock:
            old_entry = self._cache.pop(key, None)
            if old_entry is not None:
                self.current_bytes -= old_entry.size_bytes
            if self.max_bytes is not None and size_bytes > self.max_bytes:
                return
            self._cache[key] = _CacheEntry(
                data=_detach_value(
                    data, device=self.cache_device, pin_memory=self.pin_memory
                ),
                size_bytes=size_bytes,
            )
            self.current_bytes += size_bytes
            self._cache.move_to_end(key)
            self._evict_over_budget()

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()
            self.current_bytes = 0

    def remove_if_same(self, key: str | None, expected_data: Any) -> bool:
        """Remove a key only if it still holds the observed object."""
        if key is None:
            return False
        key = str(key)
        with self._lock:
            entry = self._cache.get(key)
            if entry is None or entry.data is not expected_data:
                return False
            del self._cache[key]
            self.current_bytes -= entry.size_bytes
            return True

    def remove_if(self, predicate: Callable[[str], bool]) -> int:
        # Evaluate the predicate outside the lock: it runs arbitrary caller code
        # and may re-enter this cache, which would deadlock the non-reentrant
        # lock. Snapshot keys, test them lock-free, then re-acquire to delete.
        with self._lock:
            keys = list(self._cache)
        to_remove = [key for key in keys if predicate(key)]
        removed = 0
        with self._lock:
            for key in to_remove:
                entry = self._cache.pop(key, None)
                if entry is None:
                    continue
                self.current_bytes -= entry.size_bytes
                removed += 1
        return removed

    def __len__(self) -> int:
        with self._lock:
            return len(self._cache)

    def _evict_over_budget(self) -> None:
        while self.max_size is not None and len(self._cache) > self.max_size:
            _, entry = self._cache.popitem(last=False)
            self.current_bytes -= entry.size_bytes
            self.eviction_count += 1
        while self.max_bytes is not None and self.current_bytes > self.max_bytes:
            if not self._cache:
                self.current_bytes = 0
                return
            _, entry = self._cache.popitem(last=False)
            self.current_bytes -= entry.size_bytes
            self.eviction_count += 1
