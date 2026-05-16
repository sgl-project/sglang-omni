# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

import torch


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
    ) -> None:
        if isinstance(cache_device, str):
            cache_device = torch.device(cache_device)
        self._cache: OrderedDict[str, _CacheEntry] = OrderedDict()
        self.max_size = max_size
        self.max_bytes = max_bytes
        self.cache_device = cache_device
        self.current_bytes = 0

    def get(self, key: str | None) -> Any | None:
        if key is None:
            return None
        key = str(key)
        entry = self._cache.get(key)
        if entry is None:
            return None
        self._cache.move_to_end(key)
        return entry.data

    def put(self, key: str | None, data: Any) -> None:
        if key is None:
            return
        key = str(key)
        size_bytes = _value_size_bytes(data)
        old_entry = self._cache.pop(key, None)
        if old_entry is not None:
            self.current_bytes -= old_entry.size_bytes
        if self.max_bytes is not None and size_bytes > self.max_bytes:
            return
        self._cache[key] = _CacheEntry(
            data=_detach_value(data, device=self.cache_device),
            size_bytes=size_bytes,
        )
        self.current_bytes += size_bytes
        self._cache.move_to_end(key)
        self._evict_over_budget()

    def clear(self) -> None:
        self._cache.clear()
        self.current_bytes = 0

    def _evict_over_budget(self) -> None:
        while self.max_size is not None and len(self._cache) > self.max_size:
            _, entry = self._cache.popitem(last=False)
            self.current_bytes -= entry.size_bytes
        while self.max_bytes is not None and self.current_bytes > self.max_bytes:
            if not self._cache:
                self.current_bytes = 0
                return
            _, entry = self._cache.popitem(last=False)
            self.current_bytes -= entry.size_bytes
