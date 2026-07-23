# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import threading

import pytest
import torch

from sglang_omni.scheduling.stage_cache import StageOutputCache, _value_size_bytes


def _fixed_size(_value: object) -> int:
    return 8


def test_value_size_bytes_counts_byte_buffers() -> None:
    assert _value_size_bytes(b"x" * 1024) == 1024
    assert _value_size_bytes(bytearray(16)) == 16
    assert _value_size_bytes({"a": b"xx", "b": [b"yyy"]}) == 5
    assert _value_size_bytes(torch.zeros(4, dtype=torch.float32)) == 16


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
    # Mirrors the moss-td encoder: many reader threads racing one writer.
    cache = StageOutputCache(max_size=64, size_fn=_fixed_size)
    stop = threading.Event()
    errors: list[BaseException] = []

    def writer() -> None:
        try:
            for i in range(5000):
                cache.put(str(i % 128), i)
        except BaseException as exc:  # noqa: BLE001 - surface to main thread
            errors.append(exc)
        finally:
            stop.set()

    def reader() -> None:
        try:
            while not stop.is_set():
                cache.get(str(threading.get_ident() % 128))
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=writer)]
    threads += [threading.Thread(target=reader) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)

    assert not any(thread.is_alive() for thread in threads)
    assert not errors, errors
    # current_bytes must exactly track the surviving entries.
    assert cache.current_bytes == len(cache) * 8


def test_remove_if_predicate_may_reenter_cache_without_deadlock() -> None:
    # Regression: a predicate that re-acquires the (non-reentrant) lock used to
    # deadlock remove_if. It must now evaluate lock-free.
    cache = StageOutputCache(size_fn=_fixed_size)
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
    cache = StageOutputCache(size_fn=_fixed_size)
    errors: list[BaseException] = []

    def putter() -> None:
        try:
            for i in range(2000):
                cache.put(str(i % 50), i)
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    def remover() -> None:
        try:
            for _ in range(2000):
                cache.remove_if(lambda key: int(key) % 3 == 0)
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=putter), threading.Thread(target=remover)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)

    assert not any(thread.is_alive() for thread in threads)
    assert not errors, errors
    assert cache.current_bytes == len(cache) * 8
