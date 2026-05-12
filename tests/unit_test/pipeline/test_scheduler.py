# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import threading
from types import SimpleNamespace

from sglang_omni_v1.scheduling.messages import IncomingMessage
from sglang_omni_v1.scheduling.omni_scheduler import OmniScheduler
from sglang_omni_v1.scheduling.simple_scheduler import SimpleScheduler
from sglang_omni_v1.scheduling.threaded_simple_scheduler import ThreadedSimpleScheduler
from tests.unit_test.pipeline.helpers import run_scheduler


def test_simple_scheduler_batch_and_error_contracts() -> None:
    """Preserves batched success output and per-request batch failure emission."""
    good = SimpleScheduler(
        lambda payload: payload,
        batch_compute_fn=lambda payloads: [payload.upper() for payload in payloads],
        max_batch_size=2,
        max_batch_wait_ms=10,
    )
    outputs = run_scheduler(
        good,
        [
            IncomingMessage("req-1", "new_request", "a"),
            IncomingMessage("req-2", "new_request", "b"),
        ],
        output_count=2,
    )
    assert {out.data for out in outputs} == {"A", "B"}

    bad = SimpleScheduler(
        lambda payload: payload,
        batch_compute_fn=lambda payloads: ["only-one"],
        max_batch_size=2,
        max_batch_wait_ms=10,
    )
    outputs = run_scheduler(
        bad,
        [
            IncomingMessage("req-1", "new_request", "a"),
            IncomingMessage("req-2", "new_request", "b"),
        ],
        output_count=2,
    )
    assert {out.request_id for out in outputs} == {"req-1", "req-2"}
    assert all(
        out.type == "error" and isinstance(out.data, ValueError) for out in outputs
    )


def test_threaded_simple_scheduler_runs_requests_concurrently() -> None:
    """Covers concurrent worker execution before result emission."""
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
        return payload

    def wait_for_both_started() -> None:
        try:
            assert both_started.wait(timeout=2.0)
        finally:
            release.set()

    outputs = run_scheduler(
        ThreadedSimpleScheduler(compute, max_concurrency=2),
        [
            IncomingMessage("req-1", "new_request", "one"),
            IncomingMessage("req-2", "new_request", "two"),
        ],
        output_count=2,
        before_collect=wait_for_both_started,
    )

    assert {output.request_id for output in outputs} == {"req-1", "req-2"}
    assert {output.data for output in outputs} == {"one", "two"}


def test_threaded_simple_scheduler_reports_worker_errors() -> None:
    """Covers worker exception emission as scheduler errors."""

    def compute(payload: str) -> str:
        raise RuntimeError(payload)

    outputs = run_scheduler(
        ThreadedSimpleScheduler(compute, max_concurrency=1),
        [IncomingMessage("req-err", "new_request", "boom")],
        output_count=1,
    )

    assert outputs[0].request_id == "req-err"
    assert outputs[0].type == "error"
    assert isinstance(outputs[0].data, RuntimeError)


def test_omni_scheduler_default_stream_chunk_buffers_raw_chunks() -> None:
    """Preserves generic stream chunk buffering when no custom handler exists."""
    req_data = SimpleNamespace()
    chunk = SimpleNamespace(data="chunk-data", metadata={"token_id": 1})

    OmniScheduler._append_stream_chunk_default(req_data, chunk)

    assert list(req_data.stream_chunks) == [chunk]


def test_omni_scheduler_default_stream_done_sets_generic_flag() -> None:
    """Preserves generic stream completion state when no custom handler exists."""
    scheduler = object.__new__(OmniScheduler)
    scheduler._stream_done_handler = None
    req_data = SimpleNamespace()

    scheduler._mark_stream_done(req_data)

    assert req_data.stream_done is True
