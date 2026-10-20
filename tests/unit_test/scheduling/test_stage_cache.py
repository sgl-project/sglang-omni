# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import sys
import threading
from types import FrameType

import pytest
import torch

from sglang_omni.scheduling.stage_cache import StageOutputCache, value_size_bytes


def fixed_size(_value: object) -> int:
    return 8


def test_value_size_bytes_counts_byte_buffers() -> None:
    assert value_size_bytes(b"x" * 1024) == 1024
    assert value_size_bytes(bytearray(16)) == 16
    assert value_size_bytes({"a": b"xx", "b": [b"yyy"]}) == 5
    assert value_size_bytes(torch.zeros(4, dtype=torch.float32)) == 16


def test_stage_output_cache_evicts_on_byte_buffer_size() -> None:
    cache = StageOutputCache(max_bytes=1024)
    cache.put("a", {"payload": b"x" * 800})
    cache.put("b", {"payload": b"y" * 800})
    assert cache.get("a") is None
    assert cache.get("b") is not None


@pytest.mark.parametrize("capacity", ["max_size", "max_bytes"])
def test_stage_output_cache_rejects_negative_capacity(capacity: str) -> None:
    with pytest.raises(ValueError, match=rf"{capacity} must be non-negative"):
        StageOutputCache(**{capacity: -1})


@pytest.mark.parametrize("capacity", ["max_size", "max_bytes"])
def test_stage_output_cache_allows_zero_capacity(capacity: str) -> None:
    cache = StageOutputCache(**{capacity: 0})

    cache.put("key", b"value")

    assert cache.get("key") is None


def test_remove_if_same_preserves_a_replacement() -> None:
    cache = StageOutputCache()
    cache.put("key", torch.zeros(1))
    observed = cache.get("key")
    cache.put("key", torch.ones(1))

    assert cache.remove_if_same("key", observed) is False
    assert torch.equal(cache.get("key"), torch.ones(1))

    replacement = cache.get("key")
    assert cache.remove_if_same("key", replacement) is True
    assert cache.get("key") is None


def test_concurrent_get_put_keeps_byte_accounting_consistent() -> None:
    cache = StageOutputCache(max_size=1, size_fn=fixed_size)
    cache.put("read", 0)
    reader_paused = threading.Event()
    release_reader = threading.Event()
    errors: list[BaseException] = []

    def pause_after_lookup(frame: FrameType, event: str, argument: object) -> None:
        if event == "c_return" and argument == cache.cache.get:
            sys.setprofile(None)
            reader_paused.set()
            assert release_reader.wait(5)
        else:
            pass

    def reader() -> None:
        sys.setprofile(pause_after_lookup)
        try:
            cache.get("read")
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)
        finally:
            sys.setprofile(None)

    def writer() -> None:
        try:
            cache.put("write", 1)
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    reader_thread = threading.Thread(target=reader)
    reader_thread.start()
    assert reader_paused.wait(5)
    writer_thread = threading.Thread(target=writer)
    writer_thread.start()
    writer_thread.join(timeout=0.5)
    release_reader.set()
    reader_thread.join(timeout=5)
    writer_thread.join(timeout=5)

    assert not reader_thread.is_alive() and not writer_thread.is_alive()
    assert not errors, errors
    assert cache.current_bytes == len(cache) * 8


def test_remove_if_predicate_may_reenter_cache_without_deadlock() -> None:
    # Regression: a predicate that re-acquires the (non-reentrant) lock used to
    # deadlock remove_if. It must now evaluate lock-free.
    cache = StageOutputCache(size_fn=fixed_size)
    for i in range(10):
        cache.put(str(i), i)

    def reentrant_predicate(key: str) -> bool:
        # Touches the cache lock while remove_if is "in flight".
        cache.get(key)
        return int(key) % 2 == 0

    result: list[int] = []
    worker = threading.Thread(
        target=lambda: result.append(cache.remove_if(reentrant_predicate))
    )
    worker.start()
    worker.join(timeout=5)

    assert not worker.is_alive(), "remove_if deadlocked on a re-entrant predicate"
    assert result == [5]
    assert len(cache) == 5
    assert cache.current_bytes == 5 * 8


def test_concurrent_remove_if_and_put_do_not_corrupt_state() -> None:
    cache = StageOutputCache(max_size=1, size_fn=fixed_size)
    cache.put("old", 0)
    putter_paused = threading.Event()
    release_putter = threading.Event()
    errors: list[BaseException] = []

    def pause_before_eviction(frame: FrameType, event: str, argument: object) -> None:
        if (
            event == "c_return"
            and argument is len
            and frame.f_code.co_name == "evict_over_budget"
        ):
            sys.setprofile(None)
            putter_paused.set()
            assert release_putter.wait(5)
        else:
            pass

    def putter() -> None:
        sys.setprofile(pause_before_eviction)
        try:
            cache.put("new", 1)
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)
        finally:
            sys.setprofile(None)

    def remover() -> None:
        try:
            cache.remove_if(lambda key: True)
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    putter_thread = threading.Thread(target=putter)
    putter_thread.start()
    assert putter_paused.wait(5)
    remover_thread = threading.Thread(target=remover)
    remover_thread.start()
    remover_thread.join(timeout=0.5)
    release_putter.set()
    putter_thread.join(timeout=5)
    remover_thread.join(timeout=5)

    assert not putter_thread.is_alive() and not remover_thread.is_alive()
    assert not errors, errors
    assert cache.current_bytes == len(cache) * 8


requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="pinned host memory needs a CUDA context"
)


def test_pin_memory_requires_cpu_cache_device() -> None:
    with pytest.raises(ValueError, match="pin_memory requires cache_device='cpu'"):
        StageOutputCache(pin_memory=True)
    with pytest.raises(ValueError, match="pin_memory requires cache_device='cpu'"):
        StageOutputCache(cache_device="cuda", pin_memory=True)


def test_pin_memory_is_inert_without_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    cache = StageOutputCache(cache_device="cpu", pin_memory=True)
    assert cache.pin_memory is False
    cache.put("k", torch.ones(4))
    cached = cache.get("k")
    assert cached is not None and cached.device.type == "cpu"


@pytest.mark.accelerator
@requires_cuda
def test_pinned_cache_stores_device_tensors_in_page_locked_memory() -> None:
    cache = StageOutputCache(cache_device="cpu", pin_memory=True)
    src = torch.arange(16, dtype=torch.float16, device="cuda").reshape(4, 4)
    cache.put("dev", src)
    cached = cache.get("dev")
    assert cached is not None
    assert cached.device.type == "cpu" and cached.is_pinned()
    assert torch.equal(cached, src.cpu())
    # note (Jeffro): nested containers are pinned too.
    cache.put("nested", {"a": [src, src + 1]})
    nested = cache.get("nested")
    assert nested is not None
    assert all(t.is_pinned() for t in nested["a"])


@pytest.mark.accelerator
@requires_cuda
def test_pinned_cache_reuses_already_pinned_host_tensor() -> None:
    cache = StageOutputCache(cache_device="cpu", pin_memory=True)
    host = torch.zeros(8, pin_memory=True)
    cache.put("host", host)
    cached_host = cache.get("host")
    assert cached_host is not None
    # No second copy for a pre-pinned tensor: same storage, still pinned.
    assert cached_host.data_ptr() == host.data_ptr() and cached_host.is_pinned()


@pytest.mark.accelerator
@requires_cuda
def test_pinned_cache_falls_back_to_pageable_on_alloc_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from sglang_omni.scheduling import stage_cache

    def boom(_value: torch.Tensor) -> torch.Tensor:
        raise RuntimeError("cudaHostAlloc failed")

    monkeypatch.setattr(stage_cache, "to_pinned_host", boom)
    cache = StageOutputCache(cache_device="cpu", pin_memory=True)
    src = torch.ones(4, device="cuda")
    cache.put("k", src)
    cached = cache.get("k")
    assert cached is not None
    assert cached.device.type == "cpu" and not cached.is_pinned()
    assert torch.equal(cached, src.cpu())
