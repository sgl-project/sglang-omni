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
    BatchQueueEntry,
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


class GroupingHook(BatchHook):
    """One group per item, "slow" last and held until its gate opens."""

    def __init__(self) -> None:
        super().__init__()
        self.slow_gate = threading.Event()

    def batch_groups(self, items: list[str]) -> list[list[int]]:
        return sorted(
            ([index] for index in range(len(items))),
            key=lambda group: items[group[0]] == "slow",
        )

    def encode_batch(self, items: list[str]) -> list[torch.Tensor]:
        if items == ["slow"]:
            self.slow_gate.wait(timeout=10)
        else:
            pass
        return super().encode_batch(items)


class SlowFirstGroupingHook(GroupingHook):
    """Like GroupingHook, but the held "slow" group runs first."""

    def __init__(self) -> None:
        super().__init__()
        self.slow_entered = threading.Event()

    def batch_groups(self, items: list[str]) -> list[list[int]]:
        return sorted(
            ([index] for index in range(len(items))),
            key=lambda group: items[group[0]] != "slow",
        )

    def encode_batch(self, items: list[str]) -> list[torch.Tensor]:
        if items == ["slow"]:
            self.slow_entered.set()
        else:
            pass
        return super().encode_batch(items)


class BrokenGroupingHook(BatchHook):
    """A hook whose batch split raises, as a buggy hook would."""

    def batch_groups(self, items: list[str]) -> list[list[int]]:
        raise RuntimeError("cannot split")


class DroppingGroupingHook(BatchHook):
    """A hook whose batch split loses an item, as a buggy hook would."""

    def batch_groups(self, items: list[str]) -> list[list[int]]:
        return [[0]]


class IsolatedGroupsHook(BatchHook):
    """One group per item, "bad" first; the "bad" group forward fails."""

    def batch_groups(self, items: list[str]) -> list[list[int]]:
        return sorted(
            ([index] for index in range(len(items))),
            key=lambda group: items[group[0]] != "bad",
        )

    def encode_batch(self, items: list[str]) -> list[torch.Tensor]:
        if items == ["bad"]:
            raise RuntimeError("group forward exploded")
        else:
            return super().encode_batch(items)


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
                    lambda: isinstance(queued_entries[-1], BatchQueueEntry)
                    and queued_entries[-1].queued_input == "item-late",
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


def test_each_batch_group_is_delivered_as_soon_as_it_finishes() -> None:
    """A finished group does not wait for a slower group of its batch."""
    hook = GroupingHook()
    service = make_service(hook, max_batch_size=2, max_batch_wait_ms=5000.0)
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            slow = pool.submit(service.get_or_encode, "slow")
            fast = pool.submit(service.get_or_encode, "fast")
            assert tuple(fast.result(timeout=2).tolist()) == (4, ord("t"))
            assert not slow.done()
            hook.slow_gate.set()
            assert tuple(slow.result(timeout=5).tolist()) == (4, ord("w"))
        assert hook.batch_sizes == [1, 1]
    finally:
        service.close()


@pytest.mark.parametrize(
    "hook_type",
    [BrokenGroupingHook, DroppingGroupingHook],
    ids=["raises", "loses_an_item"],
)
def test_a_bad_batch_split_falls_back_to_one_group(hook_type: type[BatchHook]) -> None:
    """A split that raises or loses an item falls back to one group."""
    hook = hook_type()
    service = make_service(
        hook, max_batch_size=2, max_batch_wait_ms=1000.0, timeout_s=5.0
    )
    try:
        results = parallel(service, ["a", "bb"])
        assert [tuple(r.tolist()) for r in results] == [(1, ord("a")), (2, ord("b"))]
        assert tuple(service.get_or_encode("ccc").tolist()) == (3, ord("c"))
        assert hook.batch_sizes == [2, 1]
    finally:
        service.close()


def test_a_cancelled_entry_inside_a_drained_batch_is_skipped() -> None:
    """Given-up entries between live ones are skipped and take no slot."""
    hook = BatchHook()
    hook.gate = threading.Event()
    service = make_service(hook, max_batch_size=2, max_batch_wait_ms=0.0)
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            running = pool.submit(service.get_or_encode, "a")
            assert hook.batch_entered.wait(timeout=5)
            # Timeouts jitter, so given-up entries can sit between live ones.
            entries: list[BatchQueueEntry[str, torch.Tensor]] = [
                BatchQueueEntry(queued_input=name, future=concurrent.futures.Future())
                for name in ("bb", "ccc", "dddd", "eeeee")
            ]
            assert entries[1].future.cancel()
            assert entries[2].abandon()
            assert entries[2].queued_input is None
            for entry in entries:
                service.batch_queue.put(entry)
            hook.gate.set()
            assert tuple(running.result(timeout=5).tolist()) == (1, ord("a"))
        assert tuple(entries[0].future.result(timeout=5).tolist()) == (2, ord("b"))
        assert tuple(entries[3].future.result(timeout=5).tolist()) == (5, ord("e"))
        assert hook.batch_sizes == [1, 2]
    finally:
        service.close()


def test_a_request_cancelled_while_an_earlier_group_runs_is_not_encoded() -> None:
    """A drained request whose caller gives up before its group is skipped."""
    hook = SlowFirstGroupingHook()
    service = make_service(hook, max_batch_size=2, max_batch_wait_ms=5000.0)
    try:
        slow: BatchQueueEntry[str, torch.Tensor] = BatchQueueEntry(
            queued_input="slow", future=concurrent.futures.Future()
        )
        late: BatchQueueEntry[str, torch.Tensor] = BatchQueueEntry(
            queued_input="late", future=concurrent.futures.Future()
        )
        service.batch_queue.put(slow)
        service.batch_queue.put(late)
        assert hook.slow_entered.wait(timeout=5)
        # The caller of "late" times out here, while "slow" is still encoding.
        assert late.abandon()
        hook.slow_gate.set()
        assert tuple(slow.future.result(timeout=5).tolist()) == (4, ord("w"))
    finally:
        service.close()
    assert hook.batch_sizes == [1]


def test_a_failed_group_does_not_fail_the_other_groups() -> None:
    """A failing group retries per item; later groups are untouched."""
    hook = IsolatedGroupsHook()
    hook.fail_items = {"bad"}
    service = make_service(hook, max_batch_size=2, max_batch_wait_ms=5000.0)
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            good = pool.submit(service.get_or_encode, "good")
            bad = pool.submit(service.get_or_encode, "bad")
            assert tuple(good.result(timeout=10).tolist()) == (4, ord("d"))
            with pytest.raises(ValueError, match="bad reference"):
                bad.result(timeout=10)
        assert hook.single_calls == ["bad"]
        assert hook.batch_sizes == [1]
    finally:
        service.close()
