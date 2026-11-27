# SPDX-License-Identifier: Apache-2.0
"""FireRedTTS3 reference encode service: solo thread, admission, timeouts."""

from __future__ import annotations

import concurrent.futures
import contextlib
import gc
import threading
import weakref
from unittest.mock import DEFAULT, patch

import pytest
import torch

from sglang_omni.models.firered_tts3.ref_encode import (
    FireRedReferenceHook,
    ReferenceAudioInput,
    ReferenceConditioning,
)
from sglang_omni.models.firered_tts3.ref_service import FireRedReferenceService
from sglang_omni.scheduling.reference_encoder import BatchQueueEntry
from tests.unit_test.fixtures.firered_tts3_preprocessing import without_solo_thread


class ConfigOnlyBundle:
    """The bundle fields the FireRedTTS3 hook reads when it is built."""

    sample_rate: int = 24000
    pad_multiple: int = 3840
    redae_scale: float = 1.0
    spk_extractor: torch.nn.Module | None = None


def artifact_for(name: str) -> ReferenceConditioning:
    return {"prompt_latents": torch.tensor([[float(len(name)), float(ord(name[-1]))]])}


def codes(artifact: ReferenceConditioning) -> tuple[int, ...]:
    return tuple(int(value) for value in artifact["prompt_latents"].flatten().tolist())


def request(name: str) -> ReferenceAudioInput:
    return ReferenceAudioInput(name, None)


class ScriptedReferenceHook(FireRedReferenceHook):
    """A FireRedTTS3 hook whose encodes can be held, failed and traced."""

    def __init__(self) -> None:
        super().__init__(ConfigOnlyBundle(), model_id="stub", cache_enabled=True)
        self.record_lock = threading.Lock()
        self.prepared_changed = threading.Condition(self.record_lock)
        self.single_calls: list[str] = []
        self.single_threads: list[str] = []
        self.single_entered = threading.Event()
        self.single_gate: threading.Event | None = None
        self.batch_sizes: list[int] = []
        self.batch_entered = threading.Event()
        self.batch_gate: threading.Event | None = None
        self.prepared: list[tuple[str, str]] = []
        self.payloads: list[weakref.ref[torch.Tensor]] = []
        self.fail_items: set[str] = set()
        self.fail_prepare: set[str] = set()

    def input_key(self, item: ReferenceAudioInput) -> str | None:
        if item.audio_path.startswith("uncacheable"):
            return None
        else:
            return item.audio_path

    def encode_one(self, item: ReferenceAudioInput) -> ReferenceConditioning:
        self.single_entered.set()
        if self.single_gate is not None:
            self.single_gate.wait(timeout=5)
        else:
            pass
        with self.record_lock:
            self.single_calls.append(item.audio_path)
            self.single_threads.append(threading.current_thread().name)
        if item.audio_path in self.fail_items:
            raise ValueError(f"bad reference: {item.audio_path}")
        else:
            return artifact_for(item.audio_path)

    def prepare_batch_input(self, item: ReferenceAudioInput) -> ReferenceAudioInput:
        with self.prepared_changed:
            self.prepared.append((item.audio_path, threading.current_thread().name))
            self.prepared_changed.notify_all()
        if item.audio_path in self.fail_prepare:
            raise ValueError(f"cannot prepare: {item.audio_path}")
        else:
            pass
        decoded_audio = torch.zeros(1, 4)
        with self.record_lock:
            self.payloads.append(weakref.ref(decoded_audio))
        return ReferenceAudioInput(
            item.audio_path, item.max_audio_seconds, decoded_audio
        )

    def batch_groups(self, items: list[ReferenceAudioInput]) -> list[list[int]]:
        return [list(range(len(items)))]

    def encode_batch(
        self, items: list[ReferenceAudioInput]
    ) -> list[ReferenceConditioning | Exception]:
        assert all(item.normalized_audio is not None for item in items)
        self.batch_entered.set()
        if self.batch_gate is not None:
            self.batch_gate.wait(timeout=5)
        else:
            pass
        with self.record_lock:
            self.batch_sizes.append(len(items))
        return [artifact_for(item.audio_path) for item in items]


class FrameHoldingReferenceHook(ScriptedReferenceHook):
    """A solo encode that fails holding a tensor, like a CUDA OOM."""

    def __init__(self) -> None:
        super().__init__()
        self.frame_tensors: list[weakref.ref[torch.Tensor]] = []

    def encode_one(self, item: ReferenceAudioInput) -> ReferenceConditioning:
        activation = torch.zeros(4)
        self.frame_tensors.append(weakref.ref(activation))
        raise RuntimeError("out of memory")


def make_service(
    hook: ScriptedReferenceHook,
    *,
    max_batch_size: int = 8,
    max_batch_wait_ms: float = 50.0,
    timeout_s: float | None = None,
) -> FireRedReferenceService:
    service = FireRedReferenceService(
        hook, max_batch_size=max_batch_size, max_batch_wait_ms=max_batch_wait_ms
    )
    if timeout_s is not None:
        service.timeout_s = timeout_s
    else:
        pass
    return service


def test_prepare_runs_in_the_caller_thread_for_misses_only() -> None:
    hook = ScriptedReferenceHook()
    service = without_solo_thread(make_service(hook, max_batch_wait_ms=0.0))
    try:
        caller_thread = threading.current_thread().name
        first = service.get_or_encode(request("a"))
        service.get_or_encode(request("a"))
        service.get_or_encode(request("uncacheable-b"))
        # Hits skip prepare; misses and uncacheable inputs run it first.
        assert hook.prepared == [("a", caller_thread), ("uncacheable-b", caller_thread)]
        assert hook.batch_sizes == [1, 1]
        assert codes(first) == (1, ord("a"))
    finally:
        service.close()


def test_prepare_is_skipped_when_batching_is_disabled() -> None:
    hook = ScriptedReferenceHook()
    service = make_service(hook, max_batch_size=1)
    try:
        result = service.get_or_encode(request("a"))
        assert hook.prepared == []
        assert hook.single_threads == [threading.current_thread().name]
        assert codes(result) == (1, ord("a"))
    finally:
        service.close()


def test_lone_requests_from_different_threads_encode_on_one_solo_thread() -> None:
    hook = ScriptedReferenceHook()
    # A request that reached the batch worker would sit out this whole linger.
    service = make_service(hook, max_batch_wait_ms=5000.0)
    try:
        items = ["a", "uncacheable-b", "cc", "ddd"]
        results: dict[str, ReferenceConditioning] = {}
        caller_threads = []
        # One request at a time, each from a new thread, like a rotating pool.
        for index, item in enumerate(items):
            caller = threading.Thread(
                target=lambda item=item: results.update(
                    {item: service.get_or_encode(request(item))}
                ),
                name=f"request-{index}",
            )
            caller.start()
            caller.join(timeout=10)
            caller_threads.append(caller.name)
        assert codes(results["a"]) == (1, ord("a"))
        assert hook.single_calls == items
        assert len(set(hook.single_threads)) == 1
        assert hook.single_threads[0] not in caller_threads
        assert service.batch_thread is not None
        assert hook.single_threads[0] != service.batch_thread.name
        assert hook.batch_sizes == []
        # The solo path gets the raw item, so prepare must not run.
        assert hook.prepared == []
        stats = service.stats()
        assert (stats["batches"], stats["solo_encodes"]) == (0, len(items))
    finally:
        service.close()


def test_close_stops_the_solo_thread() -> None:
    hook = ScriptedReferenceHook()
    service = make_service(hook)
    service.get_or_encode(request("a"))
    solo_threads = [
        thread
        for thread in threading.enumerate()
        if thread.name == hook.single_threads[0]
    ]
    service.close()
    assert len(solo_threads) == 1
    solo_threads[0].join(timeout=5)
    assert not solo_threads[0].is_alive()


def test_close_lets_a_running_solo_encode_finish_before_the_hook_closes() -> None:
    """A close during a solo encode waits for it, then closes the hook."""
    hook = ScriptedReferenceHook()
    hook.single_gate = threading.Event()
    encodes_done_at_hook_close: list[list[str]] = []
    hook.close = lambda: encodes_done_at_hook_close.append(list(hook.single_calls))
    service = make_service(hook, timeout_s=30.0)
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        lone = pool.submit(service.get_or_encode, request("a"))
        assert hook.single_entered.wait(timeout=5)
        close_thread = threading.Thread(target=service.close)
        close_thread.start()
        # A close that skipped the wait would finish here and close the hook.
        close_thread.join(timeout=0.2)
        assert close_thread.is_alive()
        hook.single_gate.set()
        assert codes(lone.result(timeout=5)) == (1, ord("a"))
    close_thread.join(timeout=5)
    assert not close_thread.is_alive()
    assert encodes_done_at_hook_close == [["a"]]


def test_a_lone_request_skips_a_solo_thread_held_by_a_timed_out_encode() -> None:
    """An orphaned solo encode must not make the next lone request wait."""
    hook = ScriptedReferenceHook()
    hook.single_gate = threading.Event()
    service = make_service(hook, timeout_s=0.5, max_batch_wait_ms=0.0)
    try:
        with pytest.raises(concurrent.futures.TimeoutError):
            service.get_or_encode(request("a"))
        # "a" still holds the solo thread, so lone "b" goes to the batch worker.
        assert codes(service.get_or_encode(request("b"))) == (1, ord("b"))
        assert hook.batch_sizes == [1]
        # Read it first: the done callback clears solo_future once "a" ends.
        orphan = service.solo_future
        assert orphan is not None
        hook.single_gate.set()
        orphan.result(timeout=5)
        # Once the solo thread is idle again, the next lone request takes it.
        service.get_or_encode(request("cc"))
        assert hook.single_calls == ["a", "cc"]
        assert len(set(hook.single_threads)) == 1
        assert service.stats()["solo_encodes"] == 2
    finally:
        service.close()


def test_a_solo_submit_that_raises_leaves_the_solo_path_on() -> None:
    """After a solo submit raises, the next lone request still runs solo."""
    hook = ScriptedReferenceHook()
    service = make_service(hook, max_batch_wait_ms=0.0)
    try:
        assert service.solo_executor is not None
        with patch.object(
            service.solo_executor,
            "submit",
            wraps=service.solo_executor.submit,
            side_effect=[RuntimeError("can't start new thread"), DEFAULT],
        ):
            with pytest.raises(RuntimeError, match="start new thread"):
                service.get_or_encode(request("a"))
            assert codes(service.get_or_encode(request("bb"))) == (2, ord("b"))
        assert hook.single_calls == ["bb"]
        assert hook.batch_sizes == []
        assert service.stats()["solo_encodes"] == 1
    finally:
        service.close()


def test_requests_arriving_during_a_solo_encode_join_the_batch_queue() -> None:
    hook = ScriptedReferenceHook()
    hook.single_gate = threading.Event()
    service = make_service(hook, max_batch_size=2, max_batch_wait_ms=5000.0)
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
            first = pool.submit(service.get_or_encode, request("a"))
            assert hook.single_entered.wait(timeout=5)
            # "a" holds the solo thread, so these two form a batch.
            followers = [
                pool.submit(service.get_or_encode, request(name))
                for name in ("bb", "ccc")
            ]
            results = [future.result(timeout=10) for future in followers]
            assert [codes(result) for result in results] == [
                (2, ord("b")),
                (3, ord("c")),
            ]
            assert hook.batch_sizes == [2]
            assert hook.single_calls == []
            hook.single_gate.set()
            assert codes(first.result(timeout=10)) == (1, ord("a"))
        # Once everything has finished, the next request is alone again.
        service.get_or_encode(request("dddd"))
        assert hook.single_calls == ["a", "dddd"]
        assert hook.batch_sizes == [2]
        assert service.stats()["solo_encodes"] == 2
    finally:
        service.close()


def test_a_request_arriving_during_a_batched_encode_joins_the_queue() -> None:
    """During a queued encode, a new request batches even if solo is idle."""
    hook = ScriptedReferenceHook()
    hook.single_gate = threading.Event()
    hook.batch_gate = threading.Event()
    service = make_service(hook, max_batch_wait_ms=0.0)
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
            lone = pool.submit(service.get_or_encode, request("a"))
            assert hook.single_entered.wait(timeout=5)
            # "bb" arrives while "a" runs solo, so it goes to the batch worker.
            queued = pool.submit(service.get_or_encode, request("bb"))
            assert hook.batch_entered.wait(timeout=5)
            hook.single_gate.set()
            assert codes(lone.result(timeout=5)) == (1, ord("a"))
            # The solo thread is idle now, but "bb" is still in flight.
            late = pool.submit(service.get_or_encode, request("ccc"))
            with hook.prepared_changed:
                assert hook.prepared_changed.wait_for(
                    lambda: "ccc" in [name for name, _ in hook.prepared], timeout=5
                )
            hook.batch_gate.set()
            assert codes(queued.result(timeout=5)) == (2, ord("b"))
            assert codes(late.result(timeout=5)) == (3, ord("c"))
        assert hook.single_calls == ["a"]
        assert hook.batch_sizes == [1, 1]
    finally:
        service.close()


def test_failed_encodes_leave_the_service_idle_for_the_next_lone_request() -> None:
    hook = ScriptedReferenceHook()
    hook.fail_items = {"bad"}
    hook.fail_prepare = {"bad-prepare"}
    hook.single_gate = threading.Event()
    service = make_service(hook, max_batch_wait_ms=0.0)
    try:
        hook.single_gate.set()
        with pytest.raises(ValueError, match="bad reference"):
            service.get_or_encode(request("bad"))
        hook.single_gate.clear()
        hook.single_entered.clear()
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            held = pool.submit(service.get_or_encode, request("a"))
            assert hook.single_entered.wait(timeout=5)
            # A request that fails in prepare must still release its place.
            with pytest.raises(ValueError, match="cannot prepare"):
                service.get_or_encode(request("bad-prepare"))
            hook.single_gate.set()
            held.result(timeout=10)
        service.get_or_encode(request("ok"))
        assert hook.single_calls == ["bad", "a", "ok"]
        assert hook.batch_sizes == []
        stats = service.stats()
        assert (stats["failed"], stats["solo_encodes"]) == (2, 3)
    finally:
        service.close()


def test_a_request_that_times_out_in_the_queue_is_not_encoded() -> None:
    """Queued work whose caller gave up is freed at once and never encoded."""
    hook = ScriptedReferenceHook()
    hook.batch_gate = threading.Event()
    service = without_solo_thread(
        make_service(hook, max_batch_wait_ms=0.0, timeout_s=0.3)
    )
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
            running = pool.submit(service.get_or_encode, request("a"))
            assert hook.batch_entered.wait(timeout=5)
            # "bb" and "ccc" wait behind the held forward until they time out.
            expired = [
                pool.submit(service.get_or_encode, request(name))
                for name in ("bb", "ccc")
            ]
            for future in [running, *expired]:
                with pytest.raises(concurrent.futures.TimeoutError):
                    future.result(timeout=5)
            gc.collect()
            # The held forward still has "a"; nothing keeps "bb" or "ccc".
            assert [payload() is not None for payload in hook.payloads] == [
                True,
                False,
                False,
            ]
        hook.batch_gate.set()
        assert codes(service.get_or_encode(request("dddd"))) == (4, ord("d"))
        assert hook.batch_sizes == [1, 1]
        assert service.stats()["batched_items"] == 2
    finally:
        service.close()


def test_a_failed_solo_encode_releases_its_frames() -> None:
    """After the caller gets the error, its traceback is not kept alive."""
    hook = FrameHoldingReferenceHook()
    service = make_service(hook)
    try:
        with contextlib.suppress(RuntimeError):
            service.get_or_encode(request("a"))
        gc.collect()
        assert len(hook.frame_tensors) == 1
        assert hook.frame_tensors[0]() is None
    finally:
        service.close()


def test_a_request_after_close_fails_before_it_prepares() -> None:
    """After close, a request fails before any encode work."""
    hook = ScriptedReferenceHook()
    service = make_service(hook)
    service.close()
    with pytest.raises(RuntimeError, match="shutting down"):
        service.get_or_encode(request("a"))
    assert (hook.prepared, hook.single_calls) == ([], [])
    assert service.encodes_in_flight == 0


def test_a_close_during_prepare_keeps_the_request_out_of_the_queue() -> None:
    """A close that lands while a request prepares still rejects it."""
    hook = ScriptedReferenceHook()
    entered, release = threading.Event(), threading.Event()
    prepare = hook.prepare_batch_input

    def held_prepare(item: ReferenceAudioInput) -> ReferenceAudioInput:
        entered.set()
        assert release.wait(timeout=5)
        return prepare(item)

    hook.prepare_batch_input = held_prepare
    service = without_solo_thread(make_service(hook, timeout_s=30.0))
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        pending = pool.submit(service.get_or_encode, request("a"))
        assert entered.wait(timeout=5)
        service.close()
        release.set()
        with pytest.raises(RuntimeError, match="shutting down"):
            pending.result(timeout=5)
    assert [name for name, _ in hook.prepared] == ["a"]
    assert service.encodes_in_flight == 0


def test_shutdown_refuses_new_requests_and_finishes_the_running_batch() -> None:
    hook = ScriptedReferenceHook()
    hook.batch_gate = threading.Event()
    service = without_solo_thread(make_service(hook, max_batch_size=2, timeout_s=30.0))
    queued_entries: list[
        BatchQueueEntry[ReferenceAudioInput, ReferenceConditioning] | None
    ] = []
    queue_grew = threading.Condition()
    assert service.batch_queue is not None
    original_put = service.batch_queue.put

    def recording_put(
        entry: BatchQueueEntry[ReferenceAudioInput, ReferenceConditioning] | None,
    ) -> None:
        original_put(entry)
        with queue_grew:
            queued_entries.append(entry)
            queue_grew.notify_all()

    service.batch_queue.put = recording_put
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
            early = [
                pool.submit(service.get_or_encode, request(f"item-{index}"))
                for index in range(2)
            ]
            assert hook.batch_entered.wait(timeout=5)
            close_thread = threading.Thread(target=service.close)
            close_thread.start()
            with queue_grew:
                assert queue_grew.wait_for(lambda: None in queued_entries, timeout=5)
            # The stop marker is queued, so admission now fails at once.
            late = pool.submit(service.get_or_encode, request("item-late"))
            with pytest.raises(RuntimeError, match="shutting down"):
                late.result(timeout=5)
            assert sorted(
                entry.queued_input.audio_path
                for entry in queued_entries
                if entry is not None and entry.queued_input is not None
            ) == ["item-0", "item-1"]
            hook.batch_gate.set()
            assert [codes(future.result(timeout=5)) for future in early] == [
                (6, ord("0")),
                (6, ord("1")),
            ]
        close_thread.join(timeout=5)
        assert not close_thread.is_alive()
    finally:
        service.close()
