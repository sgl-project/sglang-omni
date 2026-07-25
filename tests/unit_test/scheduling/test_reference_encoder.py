# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import concurrent.futures
import threading
import time
from collections import Counter
from typing import Any

import pytest
import torch

from sglang_omni.scheduling.reference_encoder import (
    ReferenceEncodeKey,
    ReferenceEncodeService,
    TensorReferenceEncodeHook,
)


class _FirstWaveGate:
    def __init__(self, count: int) -> None:
        self._barrier = threading.Barrier(count)
        self._done = threading.Event()

    def wait(self) -> None:
        if self._done.is_set():
            return
        self._barrier.wait(timeout=5)
        self._done.set()


def _wait_for_merged(
    service: ReferenceEncodeService[Any, Any, Any],
    expected: int,
    timeout_s: float = 5.0,
) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if service.stats()["merged"] >= expected:
            return
        time.sleep(0.001)
    assert service.stats()["merged"] >= expected


class _TensorHook(TensorReferenceEncodeHook[str]):
    model_id = "test"
    model_revision = "rev"
    encoder_id = "encoder"
    encoder_config_hash = "cfg"
    artifact_kind = "codes"
    storage_dtype = torch.long
    output_dtype = torch.long

    def __init__(self) -> None:
        self.calls: Counter[str] = Counter()
        self.lock = threading.Lock()

    def input_key(self, item: str) -> str | None:
        if item.startswith("uncacheable"):
            return None
        return item

    def encode_one(self, item: str) -> torch.Tensor:
        with self.lock:
            self.calls[item] += 1
        value = sum(ord(ch) for ch in item) % 127
        return torch.full((2,), value, dtype=torch.long)


def test_tensor_hook_builds_key_and_owns_stored_and_loaded_tensors() -> None:
    class _CompressedHook(_TensorHook):
        storage_dtype = torch.int32
        option = "v1"

        def options_key(self, item: str) -> str:
            return self.option

    hook = _CompressedHook()
    key = hook.cache_key("stable")
    assert key is not None
    assert key == ReferenceEncodeKey(
        model_id="test",
        model_revision="rev",
        encoder_id="encoder",
        encoder_config_hash="cfg",
        artifact_kind="codes",
        input_key="stable",
        options_key="v1",
    )
    assert hook.revalidate("stable", key)
    assert not hook.revalidate("changed", key)
    hook.option = "v2"
    assert not hook.revalidate("stable", key)

    artifact = torch.tensor([1, 2], dtype=torch.long)
    stored = hook.store_artifact(artifact)
    loaded = hook.load_artifact(stored)
    artifact.fill_(9)
    stored.fill_(8)

    assert stored.dtype == torch.int32
    assert loaded.dtype == torch.long
    assert torch.equal(loaded, torch.tensor([1, 2], dtype=torch.long))

    class _PreservingHook(_TensorHook):
        storage_dtype = None
        output_dtype = None

    preserving_hook = _PreservingHook()
    source = torch.tensor([1.5], dtype=torch.float32)
    preserved = preserving_hook.load_artifact(preserving_hook.store_artifact(source))
    assert preserved.dtype == torch.float32
    assert preserved.data_ptr() != source.data_ptr()


def test_same_key_concurrent_single_flight() -> None:
    release = threading.Event()
    entered = threading.Event()
    worker_count = 8
    gate = _FirstWaveGate(worker_count)

    class _GatedHook(_TensorHook):
        def normalize_input(self, raw_input: Any) -> str:
            item = super().normalize_input(raw_input)
            gate.wait()
            return item

        def encode_one(self, item: str) -> torch.Tensor:
            entered.set()
            assert release.wait(timeout=5)
            return super().encode_one(item)

    hook = _GatedHook()
    service = ReferenceEncodeService(hook, max_items=16, max_bytes=1024)
    results: list[torch.Tensor | None] = [None] * worker_count
    errors: list[Exception] = []

    def worker(index: int) -> None:
        try:
            results[index] = service.get_or_encode("same")
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(len(results))]
    for thread in threads:
        thread.start()
    assert entered.wait(timeout=5)
    _wait_for_merged(service, worker_count - 1)
    release.set()
    for thread in threads:
        thread.join(timeout=5)

    assert not errors
    assert hook.calls["same"] == 1
    assert all(result is not None for result in results)
    first = results[0]
    assert first is not None
    assert all(torch.equal(first, result) for result in results if result is not None)
    assert len({result.data_ptr() for result in results if result is not None}) == 8
    stats = service.stats()
    assert stats["misses"] == 1
    assert stats["merged"] == 7
    assert stats["hits"] == 0
    assert stats["entries"] == 1


def test_cache_hit_returns_loaded_artifact() -> None:
    hook = _TensorHook()
    service = ReferenceEncodeService(hook, max_items=16, max_bytes=1024)

    first = service.get_or_encode("hit")
    first.fill_(-1)
    second = service.get_or_encode("hit")

    assert hook.calls["hit"] == 1
    assert torch.all(second >= 0)
    assert first.data_ptr() != second.data_ptr()
    assert service.stats()["hits"] == 1


def test_key_none_bypasses_cache() -> None:
    hook = _TensorHook()
    service = ReferenceEncodeService(hook, max_items=16, max_bytes=1024)

    service.get_or_encode("uncacheable-a")
    service.get_or_encode("uncacheable-a")

    assert hook.calls["uncacheable-a"] == 2
    stats = service.stats()
    assert stats["uncacheable"] == 2
    assert stats["misses"] == 0
    assert stats["entries"] == 0


def test_exception_propagates_to_all_waiters_and_does_not_poison() -> None:
    release = threading.Event()
    entered = threading.Event()
    worker_count = 4
    gate = _FirstWaveGate(worker_count)

    class _FlakyHook(_TensorHook):
        def normalize_input(self, raw_input: Any) -> str:
            item = super().normalize_input(raw_input)
            gate.wait()
            return item

        def encode_one(self, item: str) -> torch.Tensor:
            with self.lock:
                self.calls[item] += 1
                call = self.calls[item]
            entered.set()
            assert release.wait(timeout=5)
            if call == 1:
                raise ValueError("boom")
            return torch.tensor([9], dtype=torch.long)

    hook = _FlakyHook()
    service = ReferenceEncodeService(hook, max_items=16, max_bytes=1024)
    errors: list[Exception] = []

    def worker() -> None:
        try:
            service.get_or_encode("flaky", desc="flaky")
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(worker_count)]
    for thread in threads:
        thread.start()
    assert entered.wait(timeout=5)
    _wait_for_merged(service, worker_count - 1)
    release.set()
    for thread in threads:
        thread.join(timeout=5)

    assert len(errors) == 4
    assert all(isinstance(error, ValueError) for error in errors)
    assert all(str(error) == "boom" for error in errors)
    if hasattr(errors[0], "__notes__"):
        assert "Reference encode context: flaky" in errors[0].__notes__
    assert len({id(error) for error in errors}) == len(errors)
    assert service.stats()["entries"] == 0
    result = service.get_or_encode("flaky")
    assert torch.equal(result, torch.tensor([9], dtype=torch.long))
    assert hook.calls["flaky"] == 2


def test_artifact_larger_than_budget_is_returned_but_not_cached() -> None:
    hook = _TensorHook()
    service = ReferenceEncodeService(hook, max_items=16, max_bytes=1)

    first = service.get_or_encode("large")
    second = service.get_or_encode("large")

    assert torch.equal(first, second)
    assert hook.calls["large"] == 2
    assert service.stats()["entries"] == 0


def test_lru_eviction_respects_max_bytes() -> None:
    hook = _TensorHook()
    service = ReferenceEncodeService(hook, max_items=16, max_bytes=16)

    service.get_or_encode("a")
    service.get_or_encode("b")
    service.get_or_encode("b")
    service.get_or_encode("a")

    assert hook.calls["b"] == 1
    assert hook.calls["a"] == 2
    assert service.stats()["evictions"] >= 1


def test_constructor_rejects_nonpositive_capacity() -> None:
    with pytest.raises(ValueError, match="max_items"):
        ReferenceEncodeService(_TensorHook(), max_items=0)
    with pytest.raises(ValueError, match="max_items"):
        ReferenceEncodeService(_TensorHook(), max_items=-1)
    with pytest.raises(ValueError, match="max_bytes"):
        ReferenceEncodeService(_TensorHook(), max_items=16, max_bytes=0)


def test_revalidate_false_returns_but_does_not_cache() -> None:
    class _NoCacheHook(_TensorHook):
        def revalidate(self, item: str, key: ReferenceEncodeKey) -> bool:
            return False

    hook = _NoCacheHook()
    service = ReferenceEncodeService(hook, max_items=16, max_bytes=1024)

    service.get_or_encode("changed")
    service.get_or_encode("changed")

    assert hook.calls["changed"] == 2
    assert service.stats()["entries"] == 0


def test_follower_timeout_does_not_remove_leader_inflight() -> None:
    release = threading.Event()
    entered = threading.Event()

    class _SlowHook(_TensorHook):
        def encode_one(self, item: str) -> torch.Tensor:
            entered.set()
            assert release.wait(timeout=5)
            return super().encode_one(item)

    hook = _SlowHook()
    service = ReferenceEncodeService(hook, max_items=16, max_bytes=1024, timeout_s=0.01)
    leader_result: list[torch.Tensor] = []

    leader = threading.Thread(
        target=lambda: leader_result.append(service.get_or_encode("slow"))
    )
    leader.start()
    assert entered.wait(timeout=5)
    with pytest.raises(concurrent.futures.TimeoutError):
        service.get_or_encode("slow")
    release.set()
    leader.join(timeout=5)

    assert len(leader_result) == 1
    assert hook.calls["slow"] == 1
    assert torch.equal(service.get_or_encode("slow"), leader_result[0])


def test_stats_hits_misses_merged_entries_bytes() -> None:
    hook = _TensorHook()
    service = ReferenceEncodeService(hook, max_items=16, max_bytes=1024)

    service.get_or_encode("stats")
    service.get_or_encode("stats")
    stats = service.stats()

    assert stats["hits"] == 1
    assert stats["misses"] == 1
    assert stats["merged"] == 0
    assert stats["entries"] == 1
    assert stats["bytes"] > 0


def test_revalidate_exception_clears_inflight_and_does_not_poison() -> None:
    # A hook whose revalidate() raises (e.g. reference file mutated mid-encode)
    # must not strand the in-flight entry: the leader has to fail the future and
    # drop the key so the next same-key request is a fresh leader, never a
    # follower blocked on a dead future for the full timeout.
    class _RaisingRevalidateHook(_TensorHook):
        def revalidate(self, item: str, key: ReferenceEncodeKey) -> bool:
            raise RuntimeError("revalidate boom")

    hook = _RaisingRevalidateHook()
    service = ReferenceEncodeService(hook, max_items=16, max_bytes=1024)

    with pytest.raises(RuntimeError, match="revalidate boom"):
        service.get_or_encode("k")
    assert service.stats()["entries"] == 0
    assert service.stats()["failed"] == 1
    assert len(service._inflight) == 0

    # New leader (not a stuck follower) -> encode runs again instead of hanging.
    with pytest.raises(RuntimeError, match="revalidate boom"):
        service.get_or_encode("k")
    assert hook.calls["k"] == 2


def test_revalidate_exception_propagates_to_followers_without_timeout() -> None:
    # Concurrent same-key followers must receive the leader's failure promptly,
    # not sit on follower_fut.result() until timeout_s.
    release = threading.Event()
    entered = threading.Event()
    worker_count = 4
    gate = _FirstWaveGate(worker_count)

    class _GatedRaisingHook(_TensorHook):
        def normalize_input(self, raw_input: Any) -> str:
            item = super().normalize_input(raw_input)
            gate.wait()
            return item

        def encode_one(self, item: str) -> torch.Tensor:
            entered.set()
            assert release.wait(timeout=5)
            return super().encode_one(item)

        def revalidate(self, item: str, key: ReferenceEncodeKey) -> bool:
            raise RuntimeError("revalidate boom")

    hook = _GatedRaisingHook()
    # Large timeout: if the key were poisoned, followers would block far past the
    # join() below, leaving their threads alive and errors incomplete.
    service = ReferenceEncodeService(hook, max_items=16, max_bytes=1024, timeout_s=30)
    errors: list[Exception] = []

    def worker() -> None:
        try:
            service.get_or_encode("k", desc="k")
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(worker_count)]
    for thread in threads:
        thread.start()
    assert entered.wait(timeout=5)
    _wait_for_merged(service, worker_count - 1)
    release.set()
    for thread in threads:
        thread.join(timeout=5)

    assert all(not thread.is_alive() for thread in threads)
    assert len(errors) == 4
    assert all(isinstance(error, RuntimeError) for error in errors)
    assert all(str(error) == "revalidate boom" for error in errors)
    assert len(service._inflight) == 0
