# SPDX-License-Identifier: Apache-2.0
"""End-to-end loop tests for :class:`EncoderScheduler`.

Drives the scheduler loop from its public ``start()`` entry point with
mocked TP collectives, covering the per-iteration error boundary and
the request-level error emission against drained messages.
"""
from __future__ import annotations

import queue as _queue_mod
import threading
import time
import logging
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from sglang_omni.proto import StagePayload
from sglang_omni.scheduling.encoder_scheduler import (
    EncoderScheduler,
    _GpuGuardDecision,
    _GpuReservation,
)
from sglang_omni.scheduling.messages import IncomingMessage


def _mk_msg(rid: str, data: dict | None = None) -> IncomingMessage:
    return IncomingMessage(
        request_id=rid,
        type="new_request",
        data=StagePayload(request_id=rid, request=None, data=data or {}),
    )


class _PassthroughAdapter:
    stage_name = "image_encoder"

    def build_batch(self, messages):
        return SimpleNamespace(
            adapter=self, image_items=[], video_items=[], audio_items=[],
            spans=[], is_empty=True,
        )

    def run_feature(self, model, plan):
        return {"image": None, "video": None, "audio": None}

    def slice_results(self, raw, plan, messages):
        return [
            StagePayload(request_id=m.request_id, request=None, data={"out": True})
            for m in messages
        ]


class _CrashingForwardAdapter(_PassthroughAdapter):
    """Adapter whose forward raises — used to test error path."""

    def run_feature(self, model, plan):
        raise RuntimeError("forward boom")


class _FakeWorker:
    def __init__(self):
        self.tp_size = 1
        self.tp_rank = 0
        self.is_entry_rank = True
        self.device = torch.device("cpu")
        self.tp_group = None
        self.encode_calls = 0

    def encode_batch(self, plan):
        self.encode_calls += 1
        return plan.adapter.run_feature(self, plan)


class _DictCache:
    def __init__(self, data: dict | None = None):
        self.data = dict(data or {})
        self.stored: list[tuple[str, object]] = []

    def get(self, key):
        return self.data.get(str(key))

    def put(self, key, value):
        self.data[str(key)] = value
        self.stored.append((str(key), value))


class _CachingPassthroughAdapter(_PassthroughAdapter):
    def lookup_cached_output(self, msg, cache):
        return cache.get(msg.request_id)

    def store_cached_output(self, msg, output, cache):
        cache.put(msg.request_id, output)


class _LimitGpuGuard:
    def __init__(self, limit: int):
        self.limit = limit
        self.checked: list[int] = []
        self.reserved: list[int] = []
        self.released: list[int] = []

    def _decision(self, cost: int) -> _GpuGuardDecision:
        self.checked.append(cost)
        return _GpuGuardDecision(
            allowed=cost <= self.limit,
            reason="fits" if cost <= self.limit else "insufficient_free_memory",
            batch_cost_bytes=cost,
            free_bytes=self.limit,
            total_bytes=self.limit * 2,
            reserved_bytes=0,
            transient_margin_bytes=0,
            allocator_margin_bytes=0,
        )

    def check(self, batch_cost: int) -> _GpuGuardDecision:
        return self._decision(batch_cost)

    def reserve(
        self,
        batch_cost: int,
    ) -> tuple[_GpuGuardDecision, _GpuReservation | None]:
        decision = self._decision(batch_cost)
        if not decision.allowed:
            return decision, None
        self.reserved.append(batch_cost)
        return decision, _GpuReservation(key=f"test-{len(self.reserved)}", bytes=batch_cost)

    def release(self, reservation: _GpuReservation | None) -> None:
        if reservation is not None:
            self.released.append(reservation.bytes)


class _TransientReservationGuard(_LimitGpuGuard):
    def __init__(self, limit: int, transient_reserved: int):
        super().__init__(limit)
        self.transient_reserved = transient_reserved
        self.failed_once = False

    def reserve(
        self,
        batch_cost: int,
    ) -> tuple[_GpuGuardDecision, _GpuReservation | None]:
        if not self.failed_once:
            self.failed_once = True
            self.checked.append(batch_cost)
            return (
                _GpuGuardDecision(
                    allowed=False,
                    reason="insufficient_free_memory",
                    batch_cost_bytes=batch_cost,
                    free_bytes=self.limit,
                    total_bytes=self.limit * 2,
                    reserved_bytes=self.transient_reserved,
                    transient_margin_bytes=0,
                    allocator_margin_bytes=0,
                ),
                None,
            )
        return super().reserve(batch_cost)


def _drain_outbox(sch: EncoderScheduler) -> list:
    out = []
    while True:
        try:
            out.append(sch.outbox.get_nowait())
        except _queue_mod.Empty:
            return out


def _run_until_outbox(sch: EncoderScheduler, expected: int, timeout: float = 5.0):
    """Run scheduler in a thread; stop once expected outbox events arrived."""
    t = threading.Thread(target=sch.start, daemon=True)
    t.start()
    deadline = time.monotonic() + timeout
    while sch.outbox.qsize() < expected:
        if time.monotonic() > deadline:
            sch.stop()
            t.join(timeout=1.0)
            raise AssertionError(
                f"timed out waiting for {expected} outbox events; got {sch.outbox.qsize()}"
            )
        time.sleep(0.01)
    sch.stop()
    t.join(timeout=2.0)


def test_loop_emits_one_result_per_request_at_tp_size_1():
    sch = EncoderScheduler(_FakeWorker(), _PassthroughAdapter())
    sch.inbox.put(_mk_msg("r0"))
    sch.inbox.put(_mk_msg("r1"))
    _run_until_outbox(sch, expected=2)
    out = _drain_outbox(sch)
    assert {o.type for o in out} == {"result"}
    assert sorted(o.request_id for o in out) == ["r0", "r1"]


def test_loop_forward_failure_is_fatal_even_at_tp_size_1():
    """Locks the [Fatal forward] domain: forward errors are fatal, not recoverable.

    Under RFC v2, ``encode_batch`` failures cannot recover with a CPU
    gather because peers may be in NCCL. The contract holds at
    ``tp_size=1`` too — uniformity of the rule matters more than the
    short-circuit, and a tp_size=1 runner is still inside upstream
    parallel-layer code paths.

    We monkey-patch ``_fatal_tp_forward_error`` so the test process
    survives — in production the rank exits non-zero and the runner
    fails outstanding Coordinator futures.
    """
    sch = EncoderScheduler(_FakeWorker(), _CrashingForwardAdapter())
    fatal_calls = []

    def _capture_fatal(exc):
        fatal_calls.append(exc)
        sch._running = False  # exit the scheduler loop

    sch._fatal_tp_forward_error = _capture_fatal
    sch.inbox.put(_mk_msg("r0"))

    t = threading.Thread(target=sch.start, daemon=True)
    t.start()
    deadline = time.monotonic() + 5.0
    while not fatal_calls and time.monotonic() < deadline:
        time.sleep(0.01)
    sch.stop()
    t.join(timeout=2.0)

    assert len(fatal_calls) == 1
    assert "forward boom" in str(fatal_calls[0])
    # Crucially: NO per-request error emission for a forward fault.
    assert sch.outbox.qsize() == 0


def test_loop_drops_aborted_request_results():
    sch = EncoderScheduler(_FakeWorker(), _PassthroughAdapter())
    sch.abort("r1")
    sch.inbox.put(_mk_msg("r0"))
    sch.inbox.put(_mk_msg("r1"))
    sch.inbox.put(_mk_msg("r2"))
    _run_until_outbox(sch, expected=2)
    out = _drain_outbox(sch)
    rids = sorted(o.request_id for o in out)
    assert rids == ["r0", "r2"]
    assert all(o.type == "result" for o in out)


def test_loop_recovers_from_recv_error_and_continues():
    """A recv-time failure produces an error per request and the scheduler
    survives — next iteration reads the next request fine."""

    cost_calls = {"n": 0}

    def cost_fn(payload):
        cost_calls["n"] += 1
        if cost_calls["n"] == 1:
            raise RuntimeError("first cost boom")
        return 0

    sch = EncoderScheduler(
        _FakeWorker(),
        _PassthroughAdapter(),
        max_batch_size=1,
        max_batch_wait_ms=0,
        request_cost_fn=cost_fn,
        max_batch_cost=10**9,
    )
    sch.inbox.put(_mk_msg("r0"))
    sch.inbox.put(_mk_msg("r1"))
    _run_until_outbox(sch, expected=2)
    out = _drain_outbox(sch)
    by_id = {o.request_id: o for o in out}
    assert by_id["r0"].type == "error"
    assert by_id["r1"].type == "result"


def test_loop_uses_batch_cost_fn_for_admission():
    seen_batch_sizes: list[int] = []

    def batch_cost_fn(payloads):
        seen_batch_sizes.append(len(payloads))
        return 100 * len(payloads)

    sch = EncoderScheduler(
        _FakeWorker(),
        _PassthroughAdapter(),
        max_batch_size=4,
        max_batch_wait_ms=0,
        batch_cost_fn=batch_cost_fn,
        max_batch_cost=150,
    )
    sch.inbox.put(_mk_msg("r0"))
    sch.inbox.put(_mk_msg("r1"))
    batch = sch._collect_batch_from_inbox()
    assert [msg.request_id for msg in batch] == ["r0"]
    assert [msg.request_id for msg in sch._collect_batch_from_inbox()] == ["r1"]
    assert seen_batch_sizes == [1, 2, 1]


def test_loop_cached_hit_bypasses_forward_and_admission():
    worker = _FakeWorker()
    cached = StagePayload(request_id="r0", request=None, data={"cached": True})
    cache = _DictCache({"r0": cached})

    def cost_fn(payload):
        raise AssertionError("cached hit should not be costed")

    guard = _LimitGpuGuard(limit=0)
    sch = EncoderScheduler(
        worker,
        _CachingPassthroughAdapter(),
        max_batch_size=1,
        max_batch_wait_ms=0,
        request_cost_fn=cost_fn,
        max_batch_cost=1,
        gpu_memory_guard=guard,
        output_cache=cache,
    )
    sch.inbox.put(_mk_msg("r0"))

    _run_until_outbox(sch, expected=1)
    out = _drain_outbox(sch)

    assert out[0].request_id == "r0"
    assert out[0].type == "result"
    assert out[0].data is cached
    assert worker.encode_calls == 0
    assert guard.checked == []
    assert guard.reserved == []


def test_loop_stores_forward_outputs_in_cache():
    worker = _FakeWorker()
    cache = _DictCache()
    sch = EncoderScheduler(
        worker,
        _CachingPassthroughAdapter(),
        max_batch_size=1,
        max_batch_wait_ms=0,
        output_cache=cache,
    )
    sch.inbox.put(_mk_msg("r0"))

    _run_until_outbox(sch, expected=1)
    out = _drain_outbox(sch)

    assert out[0].type == "result"
    assert worker.encode_calls == 1
    assert cache.stored == [("r0", out[0].data)]


def test_loop_logs_batch_admission_decisions(caplog):
    def batch_cost_fn(payloads):
        return 100 * len(payloads)

    sch = EncoderScheduler(
        _FakeWorker(),
        _PassthroughAdapter(),
        max_batch_size=4,
        max_batch_wait_ms=0,
        batch_cost_fn=batch_cost_fn,
        max_batch_cost=150,
    )
    sch.inbox.put(_mk_msg("r0"))
    sch.inbox.put(_mk_msg("r1"))

    with caplog.at_level(logging.INFO, logger="sglang_omni.scheduling.encoder_scheduler"):
        batch = sch._collect_batch_from_inbox()

    assert [msg.request_id for msg in batch] == ["r0"]
    text = caplog.text
    assert "encoder_admission stage=image_encoder decision=defer" in text
    assert "batch_cost=100" in text
    assert "candidate_cost=200" in text
    assert "max_batch_cost=150" in text
    assert "encoder_admission stage=image_encoder decision=admit" in text


def test_loop_uses_gpu_guard_for_admission_defer():
    def batch_cost_fn(payloads):
        return 100 * len(payloads)

    guard = _LimitGpuGuard(limit=150)
    sch = EncoderScheduler(
        _FakeWorker(),
        _PassthroughAdapter(),
        max_batch_size=4,
        max_batch_wait_ms=0,
        batch_cost_fn=batch_cost_fn,
        max_batch_cost=10**9,
        gpu_memory_guard=guard,
    )
    sch.inbox.put(_mk_msg("r0"))
    sch.inbox.put(_mk_msg("r1"))

    batch = sch._collect_batch_from_inbox()

    assert [msg.request_id for msg in batch] == ["r0"]
    sch._release_active_reservation()
    assert [msg.request_id for msg in sch._collect_batch_from_inbox()] == ["r1"]
    sch._release_active_reservation()
    assert 200 in guard.checked
    assert guard.reserved == [100, 100]
    assert guard.released == [100, 100]


def test_loop_gpu_guard_single_request_rejects_with_request_error():
    def cost_fn(payload):
        return 200

    guard = _LimitGpuGuard(limit=100)
    sch = EncoderScheduler(
        _FakeWorker(),
        _PassthroughAdapter(),
        max_batch_size=1,
        max_batch_wait_ms=0,
        request_cost_fn=cost_fn,
        max_batch_cost=10**9,
        gpu_memory_guard=guard,
    )
    sch.inbox.put(_mk_msg("r0"))

    _run_until_outbox(sch, expected=1)
    out = _drain_outbox(sch)
    assert out[0].request_id == "r0"
    assert out[0].type == "error"
    assert "whole-GPU guard" in str(out[0].data)
    assert guard.reserved == []


def test_loop_gpu_guard_single_request_defers_while_reservation_active():
    def cost_fn(payload):
        return 100

    guard = _TransientReservationGuard(limit=100, transient_reserved=100)
    sch = EncoderScheduler(
        _FakeWorker(),
        _PassthroughAdapter(),
        max_batch_size=1,
        max_batch_wait_ms=0,
        request_cost_fn=cost_fn,
        max_batch_cost=10**9,
        gpu_memory_guard=guard,
    )
    sch.inbox.put(_mk_msg("r0"))

    assert sch._collect_batch_from_inbox() == []
    batch = sch._collect_batch_from_inbox()

    assert [msg.request_id for msg in batch] == ["r0"]
    sch._release_active_reservation()
    assert guard.reserved == [100]
    assert guard.released == [100]


def test_loop_releases_gpu_guard_reservation_after_success():
    def cost_fn(payload):
        return 100

    guard = _LimitGpuGuard(limit=1000)
    sch = EncoderScheduler(
        _FakeWorker(),
        _PassthroughAdapter(),
        max_batch_size=1,
        max_batch_wait_ms=0,
        request_cost_fn=cost_fn,
        max_batch_cost=10**9,
        gpu_memory_guard=guard,
    )
    sch.inbox.put(_mk_msg("r0"))

    _run_until_outbox(sch, expected=1)
    assert guard.reserved == [100]
    assert guard.released == [100]


def test_loop_logs_recv_path_timing_fields(caplog):
    sch = EncoderScheduler(
        _FakeWorker(),
        _PassthroughAdapter(),
        max_batch_size=1,
        max_batch_wait_ms=0,
    )
    sch.inbox.put(_mk_msg("r0"))

    with caplog.at_level(logging.INFO, logger="sglang_omni.scheduling.encoder_scheduler"):
        _run_until_outbox(sch, expected=1)

    text = caplog.text
    assert "encoder_batch_timing stage=image_encoder" in text
    assert "inbox_admission_ms=" in text
    assert "strip_h2d_ms=" in text
    assert "metadata_broadcast_ms=" in text
    assert "follower_allocation_ms=" in text
    assert "allocation_handshake_ms=" in text
    assert "tensor_broadcast_ms=" in text
    assert "rank_wait_skew_ms=" in text


def test_loop_single_request_cost_guard_emits_request_error():
    def cost_fn(payload):
        return 200

    sch = EncoderScheduler(
        _FakeWorker(),
        _PassthroughAdapter(),
        max_batch_size=1,
        max_batch_wait_ms=0,
        request_cost_fn=cost_fn,
        max_single_request_cost=100,
    )
    sch.inbox.put(_mk_msg("r0"))
    _run_until_outbox(sch, expected=1)
    out = _drain_outbox(sch)
    assert out[0].request_id == "r0"
    assert out[0].type == "error"
    assert "max_single_request_cost" in str(out[0].data)


# ---------------------------------------------------------------------------
# RFC v2: three-domain error boundary contracts
# ---------------------------------------------------------------------------


class _BuildBatchOnceFailingAdapter(_PassthroughAdapter):
    """Adapter whose build_batch raises on the first call only.

    Used to exercise the **recoverable pre-forward** error domain
    (build_batch failure → CPU-gather fan-out → per-request error,
    scheduler proceeds to the next iteration).
    """

    def __init__(self):
        self.calls = 0
        self.encode_calls = 0

    def build_batch(self, messages):
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError("first build_batch boom")
        return super().build_batch(messages)

    def run_feature(self, model, plan):
        self.encode_calls += 1
        return super().run_feature(model, plan)


def test_loop_pre_forward_build_error_recovery():
    """Locks the [Recoverable pre-forward] domain.

    A ``build_batch`` failure must:
    - never let ``encode_batch`` run for that batch;
    - emit one ``OutgoingMessage(type="error")`` per drained request;
    - leave the scheduler alive for the next iteration.
    """
    adapter = _BuildBatchOnceFailingAdapter()
    # max_batch_size=1 + zero wait so each request lands in its own
    # iteration — otherwise the failed iteration would consume both.
    sch = EncoderScheduler(
        _FakeWorker(),
        adapter,
        max_batch_size=1,
        max_batch_wait_ms=0,
    )
    sch.inbox.put(_mk_msg("r0"))
    sch.inbox.put(_mk_msg("r1"))
    _run_until_outbox(sch, expected=2)
    out = _drain_outbox(sch)
    by_id = {o.request_id: o for o in out}
    assert by_id["r0"].type == "error"
    assert by_id["r1"].type == "result"
    # encode_batch was NOT called for the failed batch (build_batch
    # never returned a plan), but it WAS called for the recovered one.
    assert adapter.encode_calls == 1


class _SliceOnceFailingAdapter(_PassthroughAdapter):
    """Adapter whose slice_results raises on the first call only."""

    def __init__(self):
        self.slice_calls = 0

    def slice_results(self, raw, plan, messages):
        self.slice_calls += 1
        if self.slice_calls == 1:
            raise RuntimeError("first slice boom")
        return super().slice_results(raw, plan, messages)


def test_loop_post_forward_slice_error_recovery():
    """Locks the [Recoverable post-forward] domain.

    A ``slice_results`` failure happens after ``encode_batch`` completed
    on all ranks and runs only on the entry rank, so it should emit
    request-level errors and continue to the next batch.
    """
    adapter = _SliceOnceFailingAdapter()
    sch = EncoderScheduler(
        _FakeWorker(),
        adapter,
        max_batch_size=1,
        max_batch_wait_ms=0,
    )
    sch.inbox.put(_mk_msg("r0"))
    sch.inbox.put(_mk_msg("r1"))
    _run_until_outbox(sch, expected=2)
    out = _drain_outbox(sch)
    by_id = {o.request_id: o for o in out}
    assert by_id["r0"].type == "error"
    assert by_id["r1"].type == "result"
    assert adapter.slice_calls == 2


class _CrashingForwardWorker:
    """Runner whose encode_batch raises — must trigger the fatal path."""
    def __init__(self):
        self.tp_size = 1
        self.tp_rank = 0
        self.is_entry_rank = True
        self.device = torch.device("cpu")
        self.tp_group = None

    def encode_batch(self, plan):
        raise RuntimeError("fatal forward boom inside NCCL")


def test_loop_fatal_tp_forward_fault_calls_fatal_handler():
    """Locks the [Fatal forward] domain.

    A rank-local exception inside ``encode_batch`` must invoke
    :meth:`_fatal_tp_forward_error` rather than emit per-request errors,
    because peer ranks may still be blocked in NCCL.

    We monkey-patch ``_fatal_tp_forward_error`` so the test process
    survives — in production it calls ``os._exit(1)`` so
    ``MultiProcessPipelineRunner`` can tear down the TP group.
    """
    sch = EncoderScheduler(_CrashingForwardWorker(), _PassthroughAdapter())
    fatal_calls = []

    def _capture_fatal(exc):
        fatal_calls.append(exc)
        sch._running = False  # exit the scheduler loop instead of os._exit

    sch._fatal_tp_forward_error = _capture_fatal
    sch.inbox.put(_mk_msg("r0"))

    t = threading.Thread(target=sch.start, daemon=True)
    t.start()
    deadline = time.monotonic() + 5.0
    while not fatal_calls and time.monotonic() < deadline:
        time.sleep(0.01)
    sch.stop()
    t.join(timeout=2.0)

    assert len(fatal_calls) == 1
    assert isinstance(fatal_calls[0], RuntimeError)
    assert "fatal forward boom" in str(fatal_calls[0])

    # The fatal path must NOT have emitted a request-level error: that
    # would falsely promise recoverability for a TP-collective fault.
    assert sch.outbox.qsize() == 0
