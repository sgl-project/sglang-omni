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


class BatchHook(TensorReferenceEncodeHook[str]):
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
        self.batch_entered = threading.Event()

    def input_key(self, item: str) -> str | None:
        if item.startswith("uncacheable"):
            return None
        return item

    def can_encode_batch(self) -> bool:
        return self.batch_enabled

    @staticmethod
    def encode(item: str) -> torch.Tensor:
        return torch.tensor([len(item), ord(item[-1])], dtype=torch.long)

    def encode_one(self, item: str) -> torch.Tensor:
        with self.lock:
            self.single_calls.append(item)
        if item in self.fail_items:
            raise ValueError(f"bad reference: {item}")
        return self.encode(item)

    def encode_batch(self, items: list[str]) -> list[torch.Tensor]:
        self.batch_entered.set()
        if self.gate is not None:
            self.gate.wait(timeout=5)
        with self.lock:
            self.batch_sizes.append(len(items))
        if self.fail_batch:
            raise RuntimeError("batch encode exploded")
        return [self.encode(item) for item in items]


class PreparingHook(BatchHook):
    """Records which thread prepares each item and prefixes it, so the batch shows what it was handed."""

    def __init__(self) -> None:
        super().__init__()
        self.prepared: list[tuple[str, str]] = []

    def prepare_batch_input(self, item: str) -> str:
        self.prepared.append((item, threading.current_thread().name))
        return f"prepared-{item}"


class SlotFailureHook(BatchHook):
    """Fails single items inside an otherwise successful batch."""

    def encode_batch(self, items: list[str]) -> list[torch.Tensor | Exception]:
        with self.lock:
            self.batch_sizes.append(len(items))
        return [
            (
                ValueError(f"bad reference: {item}")
                if item in self.fail_items
                else self.encode(item)
            )
            for item in items
        ]


class SoloHook(BatchHook):
    """Opts in to the solo path, records the thread of every encode, and can hold a solo encode open."""

    def __init__(self, *, solo_opt_in: bool = True) -> None:
        super().__init__()
        self.solo_opt_in = solo_opt_in
        self.single_threads: list[str] = []
        self.single_done = threading.Condition(self.lock)
        self.single_gate: threading.Event | None = None
        self.single_entered = threading.Event()
        self.fail_prepare: set[str] = set()
        self.prepared: list[str] = []

    def can_encode_alone(self) -> bool:
        return self.solo_opt_in

    def prepare_batch_input(self, item: str) -> str:
        with self.lock:
            self.prepared.append(item)
        if item in self.fail_prepare:
            raise ValueError(f"cannot prepare: {item}")
        else:
            pass
        return item

    def encode_one(self, item: str) -> torch.Tensor:
        self.single_entered.set()
        if self.single_gate is not None:
            self.single_gate.wait(timeout=5)
        else:
            pass
        with self.single_done:
            self.single_threads.append(threading.current_thread().name)
            self.single_done.notify_all()
        return super().encode_one(item)


def make_service(
    hook: BatchHook, **kwargs: Any
) -> ReferenceEncodeService[Any, Any, Any]:
    params: dict[str, Any] = {"max_batch_size": 8, "max_batch_wait_ms": 50.0}
    params.update(kwargs)
    return ReferenceEncodeService(hook, **params)


def parallel(
    service: ReferenceEncodeService[Any, Any, Any], items: list[str]
) -> list[Any]:
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(items)) as pool:
        futures = [pool.submit(service.get_or_encode, item) for item in items]
        return [future.result(timeout=10) for future in futures]


def test_distinct_keys_coalesce_into_one_batch() -> None:
    hook = BatchHook()
    hook.gate = threading.Event()
    service = make_service(hook)
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


def test_same_key_followers_merge_before_batching() -> None:
    hook = BatchHook()
    hook.gate = threading.Event()
    service = make_service(hook)
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
    hook = BatchHook()
    hook.fail_batch = True
    service = make_service(hook)
    try:
        results = parallel(service, ["a", "bb", "ccc"])
        assert [tuple(r.tolist()) for r in results] == [
            (1, ord("a")),
            (2, ord("b")),
            (3, ord("c")),
        ]
        assert sorted(hook.single_calls) == ["a", "bb", "ccc"]
    finally:
        service.close()


def test_one_bad_reference_does_not_fail_its_batch_peers() -> None:
    hook = BatchHook()
    hook.fail_batch = True
    hook.fail_items = {"bad"}
    service = make_service(hook)
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


def test_max_batch_size_caps_the_forward() -> None:
    hook = BatchHook()
    hook.gate = threading.Event()
    service = make_service(hook, max_batch_size=2)
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
    hook = BatchHook()
    service = make_service(hook)
    service.get_or_encode("a")
    service.close()
    assert service.batch_thread is not None
    assert not service.batch_thread.is_alive()


def test_shutdown_fails_queued_waiters_instead_of_hanging() -> None:
    hook = BatchHook()
    hook.gate = threading.Event()
    service = make_service(hook, max_batch_size=2, timeout_s=30.0)
    queued_entries: list[object] = []
    queue_grew = threading.Condition()
    original_put = service.batch_queue.put

    def recording_put(entry: object) -> None:
        original_put(entry)
        with queue_grew:
            queued_entries.append(entry)
            queue_grew.notify_all()

    service.batch_queue.put = recording_put
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
            early = [
                pool.submit(service.get_or_encode, f"item-{index}")
                for index in range(2)
            ]
            assert hook.batch_entered.wait(timeout=5)
            close_thread = threading.Thread(target=service.close)
            close_thread.start()
            with queue_grew:
                assert queue_grew.wait_for(lambda: None in queued_entries, timeout=5)
            late = pool.submit(service.get_or_encode, "item-late")
            with queue_grew:
                assert queue_grew.wait_for(
                    lambda: isinstance(queued_entries[-1], tuple)
                    and queued_entries[-1][0] == "item-late",
                    timeout=5,
                )
            hook.gate.set()

            with pytest.raises(RuntimeError, match="shutting down"):
                late.result(timeout=5)
            assert [tuple(future.result(timeout=5).tolist()) for future in early] == [
                (6, ord("0")),
                (6, ord("1")),
            ]
        close_thread.join(timeout=5)
        assert not close_thread.is_alive()
    finally:
        service.close()


def test_invalid_batch_settings_are_rejected() -> None:
    with pytest.raises(ValueError, match="max_batch_size"):
        ReferenceEncodeService(BatchHook(), max_batch_size=0)
    with pytest.raises(ValueError, match="max_batch_wait_ms"):
        ReferenceEncodeService(BatchHook(), max_batch_wait_ms=-1)


def test_prepare_runs_in_the_caller_thread_for_misses_only() -> None:
    hook = PreparingHook()
    service = make_service(hook, max_batch_wait_ms=0.0)
    try:
        caller_thread = threading.current_thread().name
        first = service.get_or_encode("a")
        service.get_or_encode("a")
        service.get_or_encode("uncacheable-b")
        # Hits skip the prepare step; misses and uncacheable inputs run it before the batch worker sees them.
        assert hook.prepared == [("a", caller_thread), ("uncacheable-b", caller_thread)]
        assert tuple(first.tolist()) == (len("prepared-a"), ord("a"))
    finally:
        service.close()


def test_prepare_is_skipped_when_batching_is_disabled() -> None:
    hook = PreparingHook()
    service = make_service(hook, max_batch_size=1)
    try:
        result = service.get_or_encode("a")
        assert hook.prepared == []
        assert hook.single_calls == ["a"]
        assert tuple(result.tolist()) == (1, ord("a"))
    finally:
        service.close()


def test_exception_in_a_batch_slot_fails_only_that_item() -> None:
    hook = SlotFailureHook()
    hook.fail_items = {"bad"}
    service = make_service(hook, max_batch_size=3, max_batch_wait_ms=5000.0)
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
            good = pool.submit(service.get_or_encode, "good")
            bad = pool.submit(service.get_or_encode, "bad")
            other = pool.submit(service.get_or_encode, "other")
            assert tuple(good.result(timeout=10).tolist()) == (4, ord("d"))
            assert tuple(other.result(timeout=10).tolist()) == (5, ord("r"))
            with pytest.raises(ValueError, match="bad reference"):
                bad.result(timeout=10)
        # The batch-mates were not re-encoded one by one.
        assert hook.batch_sizes == [3]
        assert hook.single_calls == []
    finally:
        service.close()


def test_lone_requests_from_different_threads_encode_on_one_solo_thread() -> None:
    hook = SoloHook()
    # A request that reached the batch worker would sit out this whole linger.
    service = make_service(hook, max_batch_wait_ms=5000.0)
    try:
        items = ["a", "uncacheable-b", "cc", "ddd"]
        results: dict[str, torch.Tensor] = {}
        caller_threads = []
        # One request at a time, each from a new request thread, as a stage pool rotates its threads after load.
        for index, item in enumerate(items):
            caller = threading.Thread(
                target=lambda item=item: results.update(
                    {item: service.get_or_encode(item)}
                ),
                name=f"request-{index}",
            )
            caller.start()
            caller.join(timeout=10)
            caller_threads.append(caller.name)
        assert tuple(results["a"].tolist()) == (1, ord("a"))
        assert hook.single_calls == items
        assert len(set(hook.single_threads)) == 1
        assert hook.single_threads[0] not in caller_threads
        assert service.batch_thread is not None
        assert hook.single_threads[0] != service.batch_thread.name
        assert hook.batch_sizes == []
        # The solo path hands encode_one the raw item, so the batch-only prepare step must not run.
        assert hook.prepared == []
        stats = service.stats()
        assert (stats["batches"], stats["solo_encodes"]) == (0, len(items))
    finally:
        service.close()


def test_close_stops_the_solo_thread() -> None:
    hook = SoloHook()
    service = make_service(hook)
    service.get_or_encode("a")
    solo_threads = [
        thread
        for thread in threading.enumerate()
        if thread.name == hook.single_threads[0]
    ]
    service.close()
    assert len(solo_threads) == 1
    assert not solo_threads[0].is_alive()


def test_a_timed_out_lone_request_keeps_the_next_one_on_the_solo_thread() -> None:
    """A caller that gives up on a slow lone encode must not let the next lone request start on a second thread."""
    hook = SoloHook()
    hook.single_gate = threading.Event()
    service = make_service(hook, timeout_s=0.1)
    try:
        with pytest.raises(concurrent.futures.TimeoutError):
            service.get_or_encode("a")
        # The caller of "a" has left, so "b" is a lone request while "a" still holds the solo thread.
        with pytest.raises(concurrent.futures.TimeoutError):
            service.get_or_encode("b")
        hook.single_gate.set()
        with hook.single_done:
            assert hook.single_done.wait_for(
                lambda: len(hook.single_threads) == 2, timeout=5
            )
        assert len(set(hook.single_threads)) == 1
        assert service.stats()["solo_encodes"] == 2
    finally:
        service.close()


def test_hook_without_solo_opt_in_still_batches_a_lone_request() -> None:
    hook = SoloHook(solo_opt_in=False)
    service = make_service(hook, max_batch_wait_ms=0.0)
    try:
        result = service.get_or_encode("a")
        assert tuple(result.tolist()) == (1, ord("a"))
        assert hook.batch_sizes == [1]
        assert hook.single_calls == []
    finally:
        service.close()


def test_requests_arriving_during_a_solo_encode_join_the_batch_queue() -> None:
    hook = SoloHook()
    hook.single_gate = threading.Event()
    service = make_service(hook, max_batch_size=2, max_batch_wait_ms=5000.0)
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
            first = pool.submit(service.get_or_encode, "a")
            assert hook.single_entered.wait(timeout=5)
            # "a" holds its solo encode open, so these two fill a batch instead of going solo.
            followers = [
                pool.submit(service.get_or_encode, name) for name in ("bb", "ccc")
            ]
            results = [future.result(timeout=10) for future in followers]
            assert [tuple(r.tolist()) for r in results] == [
                (2, ord("b")),
                (3, ord("c")),
            ]
            assert hook.batch_sizes == [2]
            assert hook.single_calls == []
            hook.single_gate.set()
            assert tuple(first.result(timeout=10).tolist()) == (1, ord("a"))
        # Once everything has finished, the next request is alone again.
        service.get_or_encode("dddd")
        assert hook.single_calls == ["a", "dddd"]
        assert hook.batch_sizes == [2]
        assert service.stats()["solo_encodes"] == 2
    finally:
        service.close()


def test_failed_encodes_leave_the_service_idle_for_the_next_lone_request() -> None:
    hook = SoloHook()
    hook.fail_items = {"bad"}
    hook.fail_prepare = {"bad-prepare"}
    hook.single_gate = threading.Event()
    service = make_service(hook, max_batch_wait_ms=0.0)
    try:
        hook.single_gate.set()
        with pytest.raises(ValueError, match="bad reference"):
            service.get_or_encode("bad")
        hook.single_gate.clear()
        hook.single_entered.clear()
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            held = pool.submit(service.get_or_encode, "a")
            assert hook.single_entered.wait(timeout=5)
            # A request that fails while preparing for the batch queue must still release its place.
            with pytest.raises(ValueError, match="cannot prepare"):
                service.get_or_encode("bad-prepare")
            hook.single_gate.set()
            held.result(timeout=10)
        service.get_or_encode("ok")
        assert hook.single_calls == ["bad", "a", "ok"]
        assert hook.batch_sizes == []
        stats = service.stats()
        assert (stats["failed"], stats["solo_encodes"]) == (2, 3)
    finally:
        service.close()
