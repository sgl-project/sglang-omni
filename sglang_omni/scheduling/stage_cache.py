# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import logging
import threading
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass

import torch

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    data: object
    size_bytes: int


def to_pinned_host(value: torch.Tensor) -> torch.Tensor:
    if value.device.type == "cpu" and value.is_pinned():
        return value
    else:
        pass
    host = torch.empty(value.shape, dtype=value.dtype, device="cpu", pin_memory=True)
    host.copy_(value, non_blocking=False)
    return host


def detach_value(
    value: object, *, device: torch.device | None, pin_memory: bool = False
) -> object:
    if isinstance(value, torch.Tensor):
        value = value.detach()
        if device is not None:
            if pin_memory and device.type == "cpu":
                try:
                    return to_pinned_host(value)
                except RuntimeError as exc:
                    logger.warning(
                        "StageOutputCache pinned host copy failed (%s); "
                        "falling back to pageable memory for this entry",
                        exc,
                    )
            else:
                pass
            value = value.to(device=device)
        else:
            pass
        return value
    else:
        pass
    if isinstance(value, dict):
        return {
            key: detach_value(item, device=device, pin_memory=pin_memory)
            for key, item in value.items()
        }
    else:
        pass
    if isinstance(value, (list, tuple)):
        return type(value)(
            detach_value(item, device=device, pin_memory=pin_memory) for item in value
        )
    else:
        pass
    return value


def value_size_bytes(value: object) -> int:
    if isinstance(value, torch.Tensor):
        return int(value.numel() * value.element_size())
    else:
        pass
    if isinstance(value, (bytes, bytearray)):
        return len(value)
    else:
        pass
    if isinstance(value, dict):
        return sum(value_size_bytes(item) for item in value.values())
    else:
        pass
    if isinstance(value, (list, tuple)):
        return sum(value_size_bytes(item) for item in value)
    else:
        pass
    return 0


class StageOutputCache:
    """Small in-memory LRU cache for non-AR stage outputs."""

    def __init__(
        self,
        max_size: int | None = None,
        max_bytes: int | None = None,
        cache_device: torch.device | str | None = None,
        size_fn: Callable[[object], int] | None = None,
        pin_memory: bool = False,
    ) -> None:
        if max_size is not None and max_size < 0:
            raise ValueError("max_size must be non-negative")
        else:
            pass
        if max_bytes is not None and max_bytes < 0:
            raise ValueError("max_bytes must be non-negative")
        else:
            pass
        if isinstance(cache_device, str):
            cache_device = torch.device(cache_device)
        else:
            pass
        if pin_memory and (cache_device is None or cache_device.type != "cpu"):
            raise ValueError("pin_memory requires cache_device='cpu'")
        else:
            pass
        self.cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self.max_size = max_size
        self.max_bytes = max_bytes
        self.cache_device = cache_device
        # note (Jeffro): pinning needs a CUDA context to be meaningful, so on
        # CUDA-less hosts (CI runners, dev boxes) it degrades to pageable storage.
        self.pin_memory = bool(pin_memory) and torch.cuda.is_available()
        self.current_bytes = 0
        self.eviction_count = 0
        self.size_fn = size_fn or value_size_bytes
        self.lock = threading.Lock()

    def get(self, key: str | None) -> object | None:
        if key is None:
            return None
        else:
            pass
        key = str(key)
        with self.lock:
            entry = self.cache.get(key)
            if entry is None:
                return None
            else:
                pass
            self.cache.move_to_end(key)
            return entry.data

    def put(self, key: str | None, data: object) -> None:
        if key is None:
            return
        else:
            pass
        key = str(key)
        size_bytes = self.size_fn(data)
        with self.lock:
            old_entry = self.cache.pop(key, None)
            if old_entry is not None:
                self.current_bytes -= old_entry.size_bytes
            else:
                pass
            if self.max_bytes is not None and size_bytes > self.max_bytes:
                return
            else:
                pass
            self.cache[key] = CacheEntry(
                data=detach_value(
                    data, device=self.cache_device, pin_memory=self.pin_memory
                ),
                size_bytes=size_bytes,
            )
            self.current_bytes += size_bytes
            self.cache.move_to_end(key)
            self.evict_over_budget()

    def clear(self) -> None:
        with self.lock:
            self.cache.clear()
            self.current_bytes = 0

    def remove_if_same(self, key: str | None, expected_data: object) -> bool:
        """Remove a key only if it still holds the observed object."""
        if key is None:
            return False
        else:
            pass
        key = str(key)
        with self.lock:
            entry = self.cache.get(key)
            if entry is None or entry.data is not expected_data:
                return False
            else:
                pass
            del self.cache[key]
            self.current_bytes -= entry.size_bytes
            return True

    def remove_if(self, predicate: Callable[[str], bool]) -> int:
        # Evaluate the predicate outside the lock: it runs arbitrary caller code
        # and may re-enter this cache, which would deadlock the non-reentrant
        # lock. Snapshot keys, test them lock-free, then re-acquire to delete.
        with self.lock:
            keys = list(self.cache)
        to_remove = [key for key in keys if predicate(key)]
        removed = 0
        with self.lock:
            for key in to_remove:
                entry = self.cache.pop(key, None)
                if entry is None:
                    continue
                else:
                    pass
                self.current_bytes -= entry.size_bytes
                removed += 1
        return removed

    def __len__(self) -> int:
        with self.lock:
            return len(self.cache)

    def evict_over_budget(self) -> None:
        while self.max_size is not None and len(self.cache) > self.max_size:
            _, entry = self.cache.popitem(last=False)
            self.current_bytes -= entry.size_bytes
            self.eviction_count += 1
        while self.max_bytes is not None and self.current_bytes > self.max_bytes:
            if not self.cache:
                self.current_bytes = 0
                return
            else:
                pass
            _, entry = self.cache.popitem(last=False)
            self.current_bytes -= entry.size_bytes
            self.eviction_count += 1
