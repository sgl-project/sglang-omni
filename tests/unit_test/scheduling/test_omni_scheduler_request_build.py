# SPDX-License-Identifier: Apache-2.0
"""Request build, readiness, and scheduler-thread finalization tests."""

from __future__ import annotations

import threading
from array import array
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
from types import SimpleNamespace
from typing import Any

import pytest

from sglang_omni.scheduling.omni_scheduler import OmniScheduler


def _scheduler(
    *,
    executor: ThreadPoolExecutor | None = None,
    max_pending: int = 1,
) -> OmniScheduler:
    scheduler = object.__new__(OmniScheduler)
    scheduler.outbox = Queue()
    scheduler.inbox = Queue()
    scheduler.is_entry_rank = True
    scheduler.waiting_queue = []
    scheduler.running_batch = SimpleNamespace(reqs=[], batch_is_full=False)
    scheduler.cur_batch = None
    scheduler.last_batch = None
    scheduler._async_pending = None
    scheduler.chunked_req = None
    scheduler._pending_stream_ingress = {}
    scheduler._deferred_request_payloads = {}
    scheduler._dirty_deferred_request_ids = set()
    scheduler._first_emit_done = set()
    scheduler._prefill_start_done = set()
    scheduler._prefill_end_done = set()
    scheduler._aborted_request_ids = set()
    scheduler._aborted_request_id_order = deque()
    scheduler._abort_callback = None
    scheduler.tree_cache = None
    scheduler._request_admission_lock = threading.RLock()
    scheduler._request_build_executor = executor
    scheduler.request_build_max_pending = max_pending if executor is not None else 0
    scheduler._request_build_backlog_limit = max_pending
    scheduler._pending_request_builds = {}
    scheduler._pending_request_admissions = {}
    scheduler._backlogged_request_build_payloads = deque()
    scheduler._request_build_max_pending_observed = 0
    scheduler._scheduler_thread_id = threading.get_ident()
    # These tests focus on builder/finalizer ordering. Match the production
    # defaults explicitly so the admission policy does not become an
    # accidental part of the fixture contract.
    scheduler.max_queued_requests = None
    scheduler.enable_priority_scheduling = False
    scheduler.schedule_low_priority_values_first = False
    scheduler.abort_on_priority_when_disabled = False
    scheduler._request_kv_capacity_error = lambda req: None
    scheduler._initialize_request_stream_state = lambda data, payload: None
    scheduler._mark_running_request_aborted = lambda request_id: False
    scheduler._release_immediate_request_resources = lambda request_id: None
    return scheduler


def _payload(request_id: str) -> SimpleNamespace:
    return SimpleNamespace(
        request_id=request_id,
        prefetched_chunks=[],
        prefetched_stream_done=False,
    )


def _request_data(request_id: str) -> SimpleNamespace:
    return SimpleNamespace(
        req=SimpleNamespace(
            rid=request_id,
            origin_input_ids=array("q"),
            origin_input_ids_unpadded=array("q"),
            priority=None,
        ),
        enforce_request_limits=False,
    )


def test_sync_builder_finalizes_before_req_access_on_scheduler_thread() -> None:
    scheduler = _scheduler()
    scheduler._pending_stream_ingress["request-1"] = SimpleNamespace(
        chunks=[],
        done=True,
    )
    builder_threads: list[int] = []
    finalizer_calls: list[tuple[str, bool, int, Any]] = []

    def _build(payload: Any) -> SimpleNamespace:
        builder_threads.append(threading.get_ident())
        return SimpleNamespace(prepared_for=payload.request_id)

    def _finalize(payload: Any, pending_done: bool, prepared: Any) -> Any:
        assert threading.get_ident() == scheduler._scheduler_thread_id
        finalizer_calls.append(
            (payload.request_id, pending_done, threading.get_ident(), prepared)
        )
        assert not hasattr(prepared, "req")
        return _request_data(payload.request_id)

    scheduler._request_builder = _build
    scheduler._finalize_built_request = _finalize

    scheduler.process_input_requests([_payload("request-1")])

    scheduler_thread = threading.get_ident()
    assert builder_threads == [scheduler_thread]
    assert [
        (request_id, done, thread_id)
        for request_id, done, thread_id, _ in finalizer_calls
    ] == [("request-1", True, scheduler_thread)]
    assert [req.rid for req in scheduler.waiting_queue] == ["request-1"]


def test_async_builder_runs_off_thread_but_finalizer_runs_on_scheduler_thread() -> None:
    executor = ThreadPoolExecutor(max_workers=1)
    scheduler = _scheduler(executor=executor)
    build_started = threading.Event()
    release_build = threading.Event()
    build_finished = threading.Event()
    builder_threads: list[int] = []
    finalizer_threads: list[int] = []

    def _build(payload: Any) -> SimpleNamespace:
        builder_threads.append(threading.get_ident())
        build_started.set()
        assert release_build.wait(timeout=2.0)
        build_finished.set()
        return SimpleNamespace(prepared_for=payload.request_id)

    def _finalize(payload: Any, pending_done: bool, prepared: Any) -> Any:
        del pending_done
        assert threading.get_ident() == scheduler._scheduler_thread_id
        finalizer_threads.append(threading.get_ident())
        assert prepared.prepared_for == payload.request_id
        return _request_data(payload.request_id)

    scheduler._request_builder = _build
    scheduler._finalize_built_request = _finalize

    try:
        scheduler.process_input_requests([_payload("request-1")])
        assert build_started.wait(timeout=2.0)
        assert scheduler.waiting_queue == []

        release_build.set()
        assert build_finished.wait(timeout=2.0)
        scheduler._pending_request_builds["request-1"][2].result(timeout=2.0)
        scheduler.process_input_requests([])
        scheduler.process_input_requests([])
    finally:
        executor.shutdown(wait=True, cancel_futures=True)

    scheduler_thread = threading.get_ident()
    assert len(builder_threads) == 1
    assert builder_threads[0] != scheduler_thread
    assert finalizer_threads == [scheduler_thread]
    assert [req.rid for req in scheduler.waiting_queue] == ["request-1"]


def test_async_finalization_preserves_request_admission_order() -> None:
    executor = ThreadPoolExecutor(max_workers=2)
    scheduler = _scheduler(executor=executor, max_pending=2)
    first_started = threading.Event()
    release_first = threading.Event()
    second_finished = threading.Event()
    first_finished = threading.Event()
    finalized: list[str] = []

    def _build(payload: Any) -> SimpleNamespace:
        if payload.request_id == "request-1":
            first_started.set()
            assert release_first.wait(timeout=2.0)
            first_finished.set()
        else:
            second_finished.set()
        return SimpleNamespace(prepared_for=payload.request_id)

    def _finalize(payload: Any, pending_done: bool, prepared: Any) -> Any:
        del pending_done
        assert threading.get_ident() == scheduler._scheduler_thread_id
        assert prepared.prepared_for == payload.request_id
        finalized.append(payload.request_id)
        return _request_data(payload.request_id)

    scheduler._request_builder = _build
    scheduler._finalize_built_request = _finalize

    try:
        scheduler.process_input_requests([_payload("request-1"), _payload("request-2")])
        assert first_started.wait(timeout=2.0)
        assert second_finished.wait(timeout=2.0)
        second_future = scheduler._pending_request_builds["request-2"][2]
        second_future.result(timeout=2.0)

        scheduler.process_input_requests([])
        assert finalized == []

        release_first.set()
        assert first_finished.wait(timeout=2.0)
        first_future = scheduler._pending_request_builds["request-1"][2]
        first_future.result(timeout=2.0)
        scheduler.process_input_requests([])
        scheduler.process_input_requests([])
    finally:
        executor.shutdown(wait=True, cancel_futures=True)

    assert finalized == ["request-1", "request-2"]
    assert [req.rid for req in scheduler.waiting_queue] == [
        "request-1",
        "request-2",
    ]


@pytest.mark.parametrize("failure", ["raise", "none"])
def test_finalizer_failure_is_request_local(failure: str) -> None:
    scheduler = _scheduler()
    scheduler._request_builder = lambda payload: SimpleNamespace(
        prepared_for=payload.request_id
    )

    def _finalize(payload: Any, pending_done: bool, prepared: Any) -> Any:
        del pending_done, prepared
        if payload.request_id == "bad":
            if failure == "raise":
                raise RuntimeError("finalizer failed")
            return None
        return _request_data(payload.request_id)

    scheduler._finalize_built_request = _finalize

    scheduler.process_input_requests([_payload("bad"), _payload("good")])

    output = scheduler.outbox.get_nowait()
    assert output.request_id == "bad"
    assert output.type == "error"
    if failure == "raise":
        assert "finalizer failed" in str(output.data)
    else:
        assert "must return request data" in str(output.data)
    assert "bad" in scheduler._aborted_request_ids
    assert [req.rid for req in scheduler.waiting_queue] == ["good"]


def test_follower_finalizer_failure_does_not_emit_user_error() -> None:
    scheduler = _scheduler()
    scheduler.is_entry_rank = False
    scheduler._request_builder = lambda payload: SimpleNamespace()
    scheduler._finalize_built_request = lambda payload, done, data: None

    scheduler.process_input_requests([_payload("bad")])

    assert scheduler.outbox.empty()
    assert "bad" in scheduler._aborted_request_ids
    assert scheduler.waiting_queue == []


def test_request_readiness_failure_is_request_local() -> None:
    scheduler = _scheduler()
    scheduler._request_builder = lambda payload: _request_data(payload.request_id)

    def _ready(payload: Any, *, pending_stream_done: bool) -> bool:
        del pending_stream_done
        if payload.request_id == "bad":
            raise ValueError("invalid readiness metadata")
        return True

    scheduler._is_request_build_ready = _ready

    scheduler.process_input_requests([_payload("bad"), _payload("good")])

    output = scheduler.outbox.get_nowait()
    assert output.request_id == "bad"
    assert "invalid readiness metadata" in str(output.data)
    assert "bad" in scheduler._aborted_request_ids
    assert [req.rid for req in scheduler.waiting_queue] == ["good"]
