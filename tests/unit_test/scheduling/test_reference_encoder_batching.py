# SPDX-License-Identifier: Apache-2.0
"""M4b different-key batch coalescing in ReferenceEncodeService."""

from __future__ import annotations

import concurrent.futures
import threading
import time
from typing import Any

import pytest
import torch

from sglang_omni.scheduling.reference_encoder import (
    ReferenceEncodeService,
    TensorReferenceEncodeHook,
)


class _BatchHook(TensorReferenceEncodeHook[str]):
    model_id = "test"
    model_revision = "rev"
    encoder_id = "encoder"
    encoder_config_hash = "cfg"
    artifact_kind = "codes"
    storage_dtype = torch.long
    output_dtype = torch.long

    def __init__(self, *, batch_enabled: bool = True) -> None:
        self.batch_enabled = batch_enabled
        self.batch_sizes: list[int] = []
        self.single_calls: list[str] = []
        self.lock = threading.Lock()
        self.fail_batch = False
        self.fail_items: set[str] = set()
        self.gate: threading.Event | None = None

    def input_key(self, item: str) -> str | None:
        if item.startswith("uncacheable"):
            return None
        return item

    def can_encode_batch(self) -> bool:
        return self.batch_enabled

    @staticmethod
    def _encode(item: str) -> torch.Tensor:
        return torch.tensor([len(item), ord(item[-1])], dtype=torch.long)

    def encode_one(self, item: str) -> torch.Tensor:
        with self.lock:
            self.single_calls.append(item)
        if item in self.fail_items:
            raise ValueError(f"bad reference: {item}")
        return self._encode(item)

    def encode_batch(self, items: list[str]) -> list[torch.Tensor]:
        if self.gate is not None:
            self.gate.wait(timeout=5)
        with self.lock:
            self.batch_sizes.append(len(items))
        if self.fail_batch:
            raise RuntimeError("batch encode exploded")
        return [self._encode(item) for item in items]


def _service(hook: _BatchHook, **kwargs: Any) -> ReferenceEncodeService[Any, Any, Any]:
    params: dict[str, Any] = {"max_batch_size": 8, "max_batch_wait_ms": 50.0}
    params.update(kwargs)
    return ReferenceEncodeService(hook, **params)


def _parallel(
    service: ReferenceEncodeService[Any, Any, Any], items: list[str]
) -> list[Any]:
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(items)) as pool:
        futures = [pool.submit(service.get_or_encode, item) for item in items]
        return [future.result(timeout=10) for future in futures]


def test_batching_is_opt_in_per_hook() -> None:
    hook = _BatchHook(batch_enabled=False)
    service = _service(hook)
    try:
        assert service.batching_enabled is False
        service.get_or_encode("a")
        assert hook.batch_sizes == []
        assert hook.single_calls == ["a"]
    finally:
        service.close()


def test_batching_requires_max_batch_size_above_one() -> None:
    hook = _BatchHook()
    service = _service(hook, max_batch_size=1)
    try:
        assert service.batching_enabled is False
        service.get_or_encode("a")
        assert hook.batch_sizes == []
    finally:
        service.close()


def test_distinct_keys_coalesce_into_one_batch() -> None:
    hook = _BatchHook()
    hook.gate = threading.Event()
    service = _service(hook)
    try:
        items = ["a", "bb", "ccc", "dddd"]
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            futures = [pool.submit(service.get_or_encode, item) for item in items]
            # Let all four threads reach the queue before the worker proceeds.
            time.sleep(0.2)
            hook.gate.set()
            results = [future.result(timeout=10) for future in futures]

        assert [tuple(r.tolist()) for r in results] == [
            (1, ord("a")),
            (2, ord("b")),
            (3, ord("c")),
            (4, ord("d")),
        ]
        assert sum(hook.batch_sizes) == 4
        assert max(hook.batch_sizes) > 1, "expected real coalescing"
        assert hook.single_calls == []
        stats = service.stats()
        assert stats["batched_items"] == 4
        assert stats["misses"] == 4
    finally:
        service.close()


def test_batch_results_map_to_the_right_items() -> None:
    hook = _BatchHook()
    service = _service(hook)
    try:
        items = [f"ref-{index}{chr(ord('a') + index)}" for index in range(6)]
        results = _parallel(service, items)
        for item, result in zip(items, results):
            assert tuple(result.tolist()) == (len(item), ord(item[-1]))
    finally:
        service.close()


def test_cache_hits_never_reach_the_batch_queue() -> None:
    hook = _BatchHook()
    service = _service(hook)
    try:
        service.get_or_encode("a")
        before = list(hook.batch_sizes)
        for _ in range(5):
            service.get_or_encode("a")
        assert hook.batch_sizes == before
        assert service.stats()["hits"] == 5
    finally:
        service.close()


def test_same_key_followers_merge_before_batching() -> None:
    hook = _BatchHook()
    hook.gate = threading.Event()
    service = _service(hook)
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            futures = [pool.submit(service.get_or_encode, "same") for _ in range(4)]
            time.sleep(0.2)
            hook.gate.set()
            results = [future.result(timeout=10) for future in futures]

        assert all(tuple(r.tolist()) == (4, ord("e")) for r in results)
        # Single-flight collapses the four callers to one encoded item.
        assert sum(hook.batch_sizes) == 1
        assert service.stats()["merged"] == 3
    finally:
        service.close()


def test_batch_failure_retries_per_item() -> None:
    hook = _BatchHook()
    hook.fail_batch = True
    service = _service(hook)
    try:
        results = _parallel(service, ["a", "bb", "ccc"])
        assert [tuple(r.tolist()) for r in results] == [
            (1, ord("a")),
            (2, ord("b")),
            (3, ord("c")),
        ]
        assert sorted(hook.single_calls) == ["a", "bb", "ccc"]
    finally:
        service.close()


def test_one_bad_reference_does_not_fail_its_batch_peers() -> None:
    hook = _BatchHook()
    hook.fail_batch = True
    hook.fail_items = {"bad"}
    service = _service(hook)
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
            good = pool.submit(service.get_or_encode, "good")
            bad = pool.submit(service.get_or_encode, "bad")
            other = pool.submit(service.get_or_encode, "other")
            assert tuple(good.result(timeout=10).tolist()) == (4, ord("d"))
            assert tuple(other.result(timeout=10).tolist()) == (5, ord("r"))
            with pytest.raises(ValueError, match="bad reference"):
                bad.result(timeout=10)
        # A failed reference must not poison the key for later requests.
        hook.fail_items = set()
        hook.fail_batch = False
        assert tuple(service.get_or_encode("bad").tolist()) == (3, ord("d"))
    finally:
        service.close()


def test_uncacheable_items_still_batch() -> None:
    hook = _BatchHook()
    service = _service(hook)
    try:
        results = _parallel(service, ["uncacheable-a", "uncacheable-b"])
        assert len(results) == 2
        assert sum(hook.batch_sizes) == 2
        assert service.stats()["uncacheable"] == 2
    finally:
        service.close()


def test_max_batch_size_caps_the_forward() -> None:
    hook = _BatchHook()
    hook.gate = threading.Event()
    service = _service(hook, max_batch_size=2)
    try:
        items = [f"item-{index}" for index in range(6)]
        with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
            futures = [pool.submit(service.get_or_encode, item) for item in items]
            time.sleep(0.2)
            hook.gate.set()
            for future in futures:
                future.result(timeout=10)
        assert max(hook.batch_sizes) <= 2
        assert sum(hook.batch_sizes) == 6
    finally:
        service.close()


def test_close_stops_the_batch_worker() -> None:
    hook = _BatchHook()
    service = _service(hook)
    service.get_or_encode("a")
    service.close()
    assert service._batch_thread is not None
    assert not service._batch_thread.is_alive()


def test_shutdown_fails_queued_waiters_instead_of_hanging() -> None:
    hook = _BatchHook()
    hook.gate = threading.Event()
    service = _service(hook, max_batch_size=2, timeout_s=30.0)
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            futures = [
                pool.submit(service.get_or_encode, f"item-{index}")
                for index in range(4)
            ]
            time.sleep(0.2)
            service.close()
            hook.gate.set()
            outcomes = []
            for future in futures:
                try:
                    future.result(timeout=10)
                    outcomes.append("ok")
                except Exception:
                    outcomes.append("failed")
        # Nothing may block on the 30s encode timeout after close().
        assert len(outcomes) == 4
    finally:
        service.close()


def test_invalid_batch_settings_are_rejected() -> None:
    with pytest.raises(ValueError, match="max_batch_size"):
        ReferenceEncodeService(_BatchHook(), max_batch_size=0)
    with pytest.raises(ValueError, match="max_batch_wait_ms"):
        ReferenceEncodeService(_BatchHook(), max_batch_wait_ms=-1)
