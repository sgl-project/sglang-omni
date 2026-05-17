# SPDX-License-Identifier: Apache-2.0
"""Tests for SimpleScheduler max_concurrency dispatch path.

Kept in a separate file from test_scheduler.py because that module imports
torch at top level (for OmniScheduler / StageOutputCache tests). These
tests only exercise the SimpleScheduler scheduling layer and have no torch
dependency.
"""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Callable
from typing import Any

import pytest

from sglang_omni.scheduling.messages import IncomingMessage
from sglang_omni.scheduling.simple_scheduler import SimpleScheduler


def run_scheduler(
    scheduler: SimpleScheduler,
    messages: list[IncomingMessage],
    *,
    output_count: int,
    before_collect: Callable[[], None] | None = None,
) -> list[Any]:
    """Inlined copy of tests.unit_test.pipeline.helpers.run_scheduler to avoid
    the torch transitive import in helpers.py.

    Note (Chenchen, Chenyang):
    This is a copy of tests.unit_test.pipeline.helpers.run_scheduler to avoid
    the torch transitive import in helpers.py.
    """
    thread = threading.Thread(target=scheduler.start, daemon=True)
    thread.start()
    try:
        for message in messages:
            scheduler.inbox.put(message)
        if before_collect is not None:
            before_collect()
        return [scheduler.outbox.get(timeout=2.0) for _ in range(output_count)]
    finally:
        scheduler.stop()
        thread.join(timeout=2.0)


def test_max_concurrency_runs_sync_fn_in_parallel() -> None:
    """Two sync ``compute_fn`` invocations must be in flight simultaneously
    when ``max_concurrency=2``, not serialized."""
    started: list[str] = []
    lock = threading.Lock()
    both_started = threading.Event()
    release = threading.Event()

    def compute(payload: str) -> str:
        with lock:
            started.append(payload)
            if len(started) == 2:
                both_started.set()
        assert release.wait(timeout=2.0)
        return payload.upper()

    def wait_for_both_started() -> None:
        try:
            assert both_started.wait(timeout=2.0)
        finally:
            release.set()

    outputs = run_scheduler(
        SimpleScheduler(compute, max_concurrency=2),
        [
            IncomingMessage("req-1", "new_request", "a"),
            IncomingMessage("req-2", "new_request", "b"),
        ],
        output_count=2,
        before_collect=wait_for_both_started,
    )

    assert {out.request_id for out in outputs} == {"req-1", "req-2"}
    assert {out.data for out in outputs} == {"A", "B"}


def test_max_concurrency_runs_async_fn_in_parallel() -> None:
    """async ``compute_fn`` must also run concurrently (worker thread spins
    its own event loop so sync chunks inside the coroutine do not pin the
    scheduler's loop)."""
    started: list[str] = []
    lock = threading.Lock()
    both_started = threading.Event()
    release = threading.Event()

    async def compute(payload: str) -> str:
        with lock:
            started.append(payload)
            if len(started) == 2:
                both_started.set()
        assert release.wait(timeout=2.0)
        return payload.upper()

    def wait_for_both_started() -> None:
        try:
            assert both_started.wait(timeout=2.0)
        finally:
            release.set()

    outputs = run_scheduler(
        SimpleScheduler(compute, max_concurrency=2),
        [
            IncomingMessage("req-1", "new_request", "a"),
            IncomingMessage("req-2", "new_request", "b"),
        ],
        output_count=2,
        before_collect=wait_for_both_started,
    )

    assert {out.request_id for out in outputs} == {"req-1", "req-2"}
    assert {out.data for out in outputs} == {"A", "B"}


def test_max_concurrency_reports_worker_errors() -> None:
    """Per-request exceptions in the concurrent path land as outbox errors
    tagged with the originating request_id."""

    def compute(payload: str) -> str:
        raise RuntimeError(payload)

    outputs = run_scheduler(
        SimpleScheduler(compute, max_concurrency=2),
        [
            IncomingMessage("req-1", "new_request", "boom-1"),
            IncomingMessage("req-2", "new_request", "boom-2"),
        ],
        output_count=2,
    )

    assert {out.request_id for out in outputs} == {"req-1", "req-2"}
    assert all(
        out.type == "error" and isinstance(out.data, RuntimeError) for out in outputs
    )


def test_max_concurrency_awaits_coroutine_returned_by_sync_callable() -> None:
    """Sync wrappers that return coroutines must emit the awaited result."""

    async def compute_async(payload: str) -> str:
        await asyncio.sleep(0)
        return payload.upper()

    def compute(payload: str):
        return compute_async(payload)

    outputs = run_scheduler(
        SimpleScheduler(compute, max_concurrency=2),
        [IncomingMessage("req-await", "new_request", "payload")],
        output_count=1,
    )

    assert outputs[0].request_id == "req-await"
    assert outputs[0].type == "result"
    assert outputs[0].data == "PAYLOAD"


def test_max_concurrency_awaits_async_call_object() -> None:
    """Callable objects with async __call__ must not emit coroutine objects."""

    class Compute:
        async def __call__(self, payload: str) -> str:
            await asyncio.sleep(0)
            return payload.upper()

    outputs = run_scheduler(
        SimpleScheduler(Compute(), max_concurrency=2),
        [IncomingMessage("req-call", "new_request", "payload")],
        output_count=1,
    )

    assert outputs[0].request_id == "req-call"
    assert outputs[0].type == "result"
    assert outputs[0].data == "PAYLOAD"


def test_max_concurrency_reports_errors_from_returned_coroutines() -> None:
    """Returned coroutine exceptions must become per-request error rows."""

    async def compute_async(payload: str) -> str:
        await asyncio.sleep(0)
        raise RuntimeError(payload)

    def compute(payload: str):
        return compute_async(payload)

    outputs = run_scheduler(
        SimpleScheduler(compute, max_concurrency=2),
        [IncomingMessage("req-await-error", "new_request", "boom")],
        output_count=1,
    )

    assert outputs[0].request_id == "req-await-error"
    assert outputs[0].type == "error"
    assert isinstance(outputs[0].data, RuntimeError)


def test_max_concurrency_with_batch_fn_is_rejected() -> None:
    """``max_concurrency > 1`` and ``batch_compute_fn`` are mutually exclusive
    construction options."""
    with pytest.raises(ValueError, match="mutually exclusive"):
        SimpleScheduler(
            lambda payload: payload,
            batch_compute_fn=lambda payloads: payloads,
            max_batch_size=2,
            max_concurrency=4,
        )


def test_default_max_concurrency_is_one_for_backcompat() -> None:
    """Without ``max_concurrency``, behavior matches the historical serial path."""
    started: list[str] = []
    lock = threading.Lock()
    release = threading.Event()

    def compute(payload: str) -> str:
        with lock:
            started.append(payload)
        assert release.wait(timeout=2.0)
        return payload

    def release_after_first_started() -> None:
        # Give the scheduler thread a moment to start the first request.
        deadline = threading.Event()
        deadline.wait(0.1)
        # Only one should have started; the second is still queued.
        with lock:
            assert len(started) == 1, f"expected serial dispatch, got {started}"
        release.set()

    outputs = run_scheduler(
        SimpleScheduler(compute),  # default max_concurrency=1
        [
            IncomingMessage("req-1", "new_request", "first"),
            IncomingMessage("req-2", "new_request", "second"),
        ],
        output_count=2,
        before_collect=release_after_first_started,
    )

    assert [out.request_id for out in outputs] == ["req-1", "req-2"]
