from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

import torch
import xxhash

from ..types import RequestOutput, SchedulerRequest


@dataclass
class _CacheEntry:
    data: Any
    finished: bool
    finish_reason: str | None
    size_bytes: int


def _hash_tensor(value: torch.Tensor) -> str:
    cpu = value.detach().contiguous().cpu()
    payload = cpu.numpy().tobytes()
    meta = f"{cpu.dtype}|{tuple(cpu.shape)}".encode("utf-8")
    return xxhash.xxh3_64(b"tensor|" + meta + b"|" + payload).hexdigest()


def _hash_value(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return _hash_tensor(value)
    if isinstance(value, (list, tuple)):
        parts = [_hash_value(v) for v in value]
        if any(p is None for p in parts):
            return None
        return xxhash.xxh3_64(f"list|{'|'.join(parts)}".encode("utf-8")).hexdigest()
    if isinstance(value, dict):
        items = []
        for key in sorted(value.keys()):
            hashed = _hash_value(value[key])
            if hashed is None:
                return None
            items.append(f"{key}={hashed}")
        return xxhash.xxh3_64(f"dict|{'|'.join(items)}".encode("utf-8")).hexdigest()
    try:
        return xxhash.xxh3_64(f"scalar|{value}".encode("utf-8")).hexdigest()
    except Exception:
        return None


def _detach_value(value: Any, *, device: torch.device | None) -> Any:
    if isinstance(value, torch.Tensor):
        value = value.detach()
        if device is not None:
            value = value.to(device=device)
        return value
    if isinstance(value, dict):
        return {k: _detach_value(v, device=device) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_detach_value(v, device=device) for v in value)
    return value


def _value_size_bytes(value: Any) -> int:
    if isinstance(value, torch.Tensor):
        return int(value.numel() * value.element_size())
    if isinstance(value, dict):
        return sum(_value_size_bytes(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return sum(_value_size_bytes(item) for item in value)
    return 0


def _get_cache_key(request: SchedulerRequest) -> str | None:
    data = getattr(request, "data", None)
    if data is None:
        return None
    # Priority 1: explicit cache_key (set by preprocessing)
    cache_key = getattr(data, "cache_key", None)
    if cache_key is None and isinstance(data, dict):
        cache_key = data.get("cache_key")
    if cache_key is not None:
        return str(cache_key)
    # Priority 2: input_dict
    input_dict = getattr(data, "input_dict", None)
    if input_dict is None and isinstance(data, dict):
        input_dict = data.get("input_dict")
    return _hash_value(input_dict)


class SimpleCacheManager:
    """Simple in-memory LRU cache (temporary, will be replaced by Mooncake)."""

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

    def get(self, request: SchedulerRequest) -> RequestOutput | None:
        key = _get_cache_key(request)
        if key is None:
            return None
        entry = self._cache.get(key)
        if entry is None:
            return None
        self._cache.move_to_end(key)
        return RequestOutput(
            request_id=request.request_id,
            data=entry.data,
            finished=entry.finished,
            finish_reason=entry.finish_reason,
        )

    def put(self, request: SchedulerRequest, output: RequestOutput) -> None:
        key = _get_cache_key(request)
        if key is None:
            return
        size_bytes = _value_size_bytes(output.data)
        old_entry = self._cache.pop(key, None)
        if old_entry is not None:
            self.current_bytes -= old_entry.size_bytes
        if self.max_bytes is not None and size_bytes > self.max_bytes:
            return
        data = _detach_value(output.data, device=self.cache_device)
        entry = _CacheEntry(
            data=data,
            finished=bool(output.finished),
            finish_reason=output.finish_reason,
            size_bytes=size_bytes,
        )
        self._cache[key] = entry
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
