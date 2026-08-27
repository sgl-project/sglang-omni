# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import collections
import gc
import threading
import weakref
from array import array
from collections import deque
from queue import Queue
from types import SimpleNamespace

import pytest
import torch

from sglang_omni.admission import QueueFullError
from sglang_omni.proto import OmniRequest, StagePayload
from sglang_omni.scheduling import omni_scheduler as omni_scheduler_module
from sglang_omni.scheduling.messages import IncomingMessage
from sglang_omni.scheduling.omni_scheduler import OmniScheduler
from sglang_omni.scheduling.simple_scheduler import SimpleScheduler
from sglang_omni.scheduling.stage_cache import StageOutputCache
from sglang_omni.scheduling.threaded_simple_scheduler import ThreadedSimpleScheduler
from sglang_omni.scheduling.types import ModelRunnerOutput
from tests.unit_test.pipeline.helpers import run_scheduler


@pytest.fixture(autouse=True)
def _serving_bag(monkeypatch):
    monkeypatch.setattr(
        omni_scheduler_module,
        "get_serving",
        lambda: SimpleNamespace(weight_version=None),
    )


def _ingress(
    *chunks, done: bool = False
) -> omni_scheduler_module._PendingStreamIngress:
    entry = omni_scheduler_module._PendingStreamIngress()
    entry.chunks.extend(chunks)
    entry.done = done
    return entry


def _init_sync_request_build_state(scheduler: OmniScheduler) -> None:
    scheduler._request_admission_lock = threading.RLock()
    scheduler._request_build_executor = None
    scheduler.request_build_max_pending = 0
    scheduler._pending_request_builds = {}
    scheduler._pending_request_admissions = {}
    scheduler._backlogged_request_build_payloads = deque()
    scheduler._request_build_max_pending_observed = 0
    scheduler._async_pending = None
    scheduler.enable_priority_scheduling = False
    scheduler.abort_on_priority_when_disabled = False
    if not hasattr(scheduler, "max_queued_requests"):
        scheduler.max_queued_requests = None
    if not hasattr(scheduler, "_deferred_request_payloads"):
        scheduler._deferred_request_payloads = {}


def _init_terminal_output_state(scheduler: OmniScheduler) -> None:
    scheduler._request_admission_lock = threading.RLock()
    scheduler.is_entry_rank = True
    scheduler._model_runner = None
    scheduler._stream_output_builder = None
    scheduler._request_finished_callback = None
    scheduler._completed_request_ids = {}
    scheduler._pending_stream_ingress = {}


def _new_stage_payload(request_id: str) -> StagePayload:
    return StagePayload(
        request_id=request_id,
        request=OmniRequest(inputs={}),
        data=None,
    )


def test_scheduler_idle_sleep_yields_to_pending_request_builds(monkeypatch) -> None:
    scheduler = object.__new__(OmniScheduler)
    scheduler._request_admission_lock = threading.RLock()
    scheduler._pending_request_builds = {}
    scheduler._pending_request_admissions = {}
    sleep_calls: list[float] = []
    monkeypatch.setattr(omni_scheduler_module.time, "sleep", sleep_calls.append)

    scheduler._sleep_during_idle()
    scheduler._pending_request_builds["req"] = object()
    scheduler._sleep_during_idle()

    assert sleep_calls == [0.001, 0.0001]


def test_normal_event_loop_uses_request_build_aware_idle_sleep(monkeypatch) -> None:
    scheduler = object.__new__(OmniScheduler)
    scheduler._running = True
    scheduler._engine_paused = False
    scheduler._request_admission_lock = threading.RLock()
    scheduler._pending_request_builds = {"req": object()}
    scheduler._pending_request_admissions = {}
    scheduler._process_admin_requests = lambda: None
    scheduler.recv_requests = lambda: []
    scheduler._take_deferred_request_payloads = lambda: []
    scheduler.process_input_requests = lambda _requests: None
    scheduler.self_check_during_idle = lambda: None
    scheduler.self_check_during_busy = lambda: None

    def get_next_batch_to_run():
        scheduler._running = False
        return None

    scheduler.get_next_batch_to_run = get_next_batch_to_run
    sleep_calls: list[float] = []
    monkeypatch.setattr(omni_scheduler_module.time, "sleep", sleep_calls.append)

    scheduler._event_loop_normal()

    assert sleep_calls == [0.0001]


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


def test_take_deferred_request_payloads_is_event_driven() -> None:
    scheduler = object.__new__(OmniScheduler)
    scheduler.running_batch = None
    scheduler.cur_batch = None
    scheduler.last_batch = None
    scheduler._async_pending = None
    scheduler.waiting_queue = []
    scheduler._completed_request_ids = {}
    scheduler._pending_stream_ingress = {}
    payload = object()
    scheduler._deferred_request_payloads = {"req-deferred": payload}
    scheduler._dirty_deferred_request_ids = set()

    assert scheduler._take_deferred_request_payloads() == []
    assert scheduler._deferred_request_payloads == {"req-deferred": payload}

    OmniScheduler._on_stream_chunk(scheduler, "req-deferred", "chunk-1")
    assert scheduler._dirty_deferred_request_ids == {"req-deferred"}
    assert scheduler._take_deferred_request_payloads() == [payload]
    assert scheduler._deferred_request_payloads == {}
    assert scheduler._dirty_deferred_request_ids == set()

    scheduler._deferred_request_payloads["req-deferred"] = payload

    OmniScheduler._on_stream_chunk(scheduler, "req-unknown", "chunk-x")
    assert scheduler._dirty_deferred_request_ids == set()
    assert scheduler._pending_stream_ingress["req-unknown"].chunks == ["chunk-x"]
    assert scheduler._take_deferred_request_payloads() == []

    OmniScheduler._on_stream_done(scheduler, "req-deferred")
    assert scheduler._dirty_deferred_request_ids == {"req-deferred"}
    assert scheduler._take_deferred_request_payloads() == [payload]
    assert scheduler._dirty_deferred_request_ids == set()


def test_omni_scheduler_run_batch_failure_emits_error_and_aborts(monkeypatch) -> None:
    """Forward failures are owned by the scheduler, not model executors."""
    release_calls: list[tuple[str, object]] = []
    tree_cache = object()
    model_path_events: list[tuple[str, str, str | None]] = []
    monkeypatch.setattr(
        omni_scheduler_module,
        "_emit_model_path_start",
        lambda rid: model_path_events.append(("start", rid, None)),
    )
    monkeypatch.setattr(
        omni_scheduler_module,
        "_emit_model_path_end",
        lambda rid, *, status: model_path_events.append(("end", rid, status)),
    )
    monkeypatch.setattr(
        omni_scheduler_module,
        "release_kv_cache",
        lambda req, cache: release_calls.append((req.rid, cache)),
    )

    class BoomModelRunner:
        def execute(self, sched_output):
            assert [req.request_id for req in sched_output.requests] == [
                "req-1",
                "req-2",
            ]
            raise RuntimeError("cuda out of memory")

    scheduler = object.__new__(OmniScheduler)
    scheduler._model_runner = BoomModelRunner()
    scheduler._stream_output_builder = None
    scheduler.outbox = Queue()
    scheduler.inbox = Queue()
    scheduler.is_entry_rank = True
    scheduler._aborted_request_ids = set()
    scheduler._aborted_request_id_order = deque()
    scheduler._pending_stream_ingress = {
        "req-1": _ingress("stale"),
        "req-2": _ingress(done=True),
    }
    scheduler._deferred_request_payloads = {"req-1": object()}
    scheduler._dirty_deferred_request_ids = {"req-1"}
    scheduler._abort_callback = None
    scheduler.tree_cache = tree_cache
    scheduler.waiting_queue = []
    scheduler.last_batch = None
    scheduler.forward_ct = 0
    scheduler._sched_idled = False
    scheduler._first_emit_done = set()
    scheduler._prefill_start_done = set()
    scheduler._prefill_end_done = set()

    batch = SimpleNamespace(
        reqs=[
            SimpleNamespace(
                rid="req-1",
                _omni_data=SimpleNamespace(),
                req_pool_idx=1,
                mamba_pool_idx=None,
                inflight_middle_chunks=0,
            ),
            SimpleNamespace(
                rid="req-2",
                _omni_data=SimpleNamespace(),
                req_pool_idx=2,
                mamba_pool_idx=None,
                inflight_middle_chunks=0,
            ),
        ],
        batch_is_full=True,
        is_prefill_only=True,
        is_extend_in_batch=False,
    )
    failed_reqs = list(batch.reqs)
    for req in failed_reqs:
        req._omni_data.req = req
    scheduler.running_batch = batch
    scheduler.cur_batch = batch
    _init_sync_request_build_state(scheduler)

    result = scheduler.run_batch(batch)

    assert result is omni_scheduler_module._FAILED_BATCH_RESULT
    outputs = [scheduler.outbox.get_nowait(), scheduler.outbox.get_nowait()]
    assert {output.request_id for output in outputs} == {"req-1", "req-2"}
    assert all(output.type == "error" for output in outputs)
    assert all(isinstance(output.data, RuntimeError) for output in outputs)
    assert all("cuda out of memory" in str(output.data) for output in outputs)
    assert scheduler._aborted_request_ids == {"req-1", "req-2"}
    assert batch.reqs == []
    assert all(req._omni_data is None for req in failed_reqs)
    assert release_calls == [("req-1", tree_cache), ("req-2", tree_cache)]
    assert scheduler._pending_stream_ingress == {}
    assert scheduler._deferred_request_payloads == {}
    assert scheduler._dirty_deferred_request_ids == set()
    assert model_path_events == [
        ("start", "req-1", None),
        ("start", "req-2", None),
        ("end", "req-1", "error"),
        ("end", "req-2", "error"),
    ]


def test_upstream_queue_limit_abort_is_translated_to_omni_output() -> None:
    from sglang.srt.disaggregation.utils import DisaggregationMode

    scheduler = object.__new__(OmniScheduler)
    scheduler.outbox = Queue()
    scheduler.is_entry_rank = True
    scheduler.disaggregation_mode = DisaggregationMode.NULL
    scheduler.enable_priority_scheduling = False
    scheduler.abort_on_priority_when_disabled = False
    scheduler.max_queued_requests = 0
    scheduler.waiting_queue = []
    scheduler.enable_hicache_storage = False
    scheduler.enable_hierarchical_cache = False
    aborts: list[tuple[str, bool]] = []
    scheduler.abort = lambda rid, *, defer_running_cleanup=True: aborts.append(
        (rid, defer_running_cleanup)
    )
    scheduler.send_to_detokenizer = omni_scheduler_module._NoOpSender()
    scheduler.ipc_channels = omni_scheduler_module._OmniIpcChannels(scheduler)
    trace_aborts: list[dict] = []
    req = SimpleNamespace(
        rid="req-over-limit",
        priority=None,
        time_stats=SimpleNamespace(
            trace_ctx=SimpleNamespace(
                abort=lambda *, abort_info: trace_aborts.append(abort_info)
            ),
        ),
    )

    omni_scheduler_module._Upstream._add_request_to_queue(scheduler, req)

    output = scheduler.outbox.get_nowait()
    assert output.request_id == req.rid
    assert output.type == "error"
    assert "queue is full" in str(output.data)
    assert aborts == [(req.rid, False)]
    assert trace_aborts == [{"reason": "The request queue is full."}]
    assert scheduler.waiting_queue == []


def _enqueue_limit_scheduler(monkeypatch):
    from sglang.srt.disaggregation.utils import DisaggregationMode

    events: list[str] = []
    monkeypatch.setattr(
        omni_scheduler_module,
        "_emit_event",
        lambda **kwargs: events.append(kwargs["event_name"]),
    )
    scheduler = object.__new__(OmniScheduler)
    scheduler.outbox = Queue()
    scheduler.is_entry_rank = True
    scheduler.disaggregation_mode = DisaggregationMode.NULL
    scheduler.enable_priority_scheduling = True
    scheduler.schedule_low_priority_values_first = False
    scheduler.abort_on_priority_when_disabled = False
    scheduler.max_queued_requests = 1
    scheduler.waiting_queue = []
    scheduler.enable_hicache_storage = False
    scheduler.enable_hierarchical_cache = False
    scheduler._aborted_request_ids = set()
    scheduler._aborted_request_id_order = deque()
    scheduler._deferred_request_payloads = {}
    scheduler._pending_stream_ingress = {}
    scheduler._request_admission_lock = threading.RLock()
    scheduler._abort_callback = None
    aborts: list[str] = []
    scheduler.abort = lambda rid, *, defer_running_cleanup=True: aborts.append(rid)
    scheduler.send_to_detokenizer = omni_scheduler_module._NoOpSender()
    scheduler.ipc_channels = omni_scheduler_module._OmniIpcChannels(scheduler)
    scheduler._request_kv_capacity_error = lambda req: None
    scheduler._initialize_request_stream_state = lambda req_data, payload: None
    scheduler._append_stream_chunk = lambda *args, **kwargs: None
    scheduler._mark_stream_done = lambda *args, **kwargs: None
    return scheduler, events, aborts


def test_enqueue_built_request_honors_max_queued_requests(monkeypatch) -> None:
    scheduler, events, aborts = _enqueue_limit_scheduler(monkeypatch)

    def _req(rid: str):
        return SimpleNamespace(
            rid=rid,
            priority=None,
            origin_input_ids=array("q", [1]),
            origin_input_ids_unpadded=array("q", [1]),
            time_stats=SimpleNamespace(
                wait_queue_entry_time=0.0,
                trace_ctx=SimpleNamespace(abort=lambda *, abort_info: None),
            ),
        )

    first, second = _req("req-ok"), _req("req-reject")
    for req in (first, second):
        OmniScheduler._enqueue_built_request(
            scheduler,
            SimpleNamespace(request_id=req.rid),
            False,
            SimpleNamespace(req=req, enforce_request_limits=False),
        )

    assert [req.rid for req in scheduler.waiting_queue] == ["req-ok"]
    assert first.priority is not None
    assert second.priority is not None
    assert events.count("scheduler_queue_enter") == 1
    reject = scheduler.outbox.get_nowait()
    assert reject.request_id == "req-reject"
    assert reject.type == "error"
    assert "queue is full" in str(reject.data)
    assert aborts == ["req-reject"]


def test_process_input_requests_rejects_before_build_when_waiting_queue_is_full() -> (
    None
):
    built: list[str] = []
    scheduler = object.__new__(OmniScheduler)
    scheduler.outbox = Queue()
    scheduler.is_entry_rank = True
    scheduler.max_queued_requests = 1
    scheduler.waiting_queue = [SimpleNamespace(rid="req-occupant")]
    scheduler._pending_stream_ingress = {}
    scheduler._deferred_request_payloads = {}
    scheduler._dirty_deferred_request_ids = set()
    scheduler._aborted_request_ids = set()
    scheduler._aborted_request_id_order = deque()
    _init_sync_request_build_state(scheduler)
    aborts: list[str] = []
    scheduler.abort = lambda rid, *, defer_running_cleanup=True: aborts.append(rid)
    scheduler._request_builder = lambda payload: built.append(payload.request_id)

    scheduler.process_input_requests([_new_stage_payload("req-new")])

    assert built == []
    assert [req.rid for req in scheduler.waiting_queue] == ["req-occupant"]
    output = scheduler.outbox.get_nowait()
    assert output.request_id == "req-new"
    assert output.type == "error"
    assert str(output.data) == QueueFullError.MESSAGE
    assert isinstance(output.data, QueueFullError)
    assert aborts == ["req-new"]


def _staging_scheduler(
    *,
    max_queued_requests: int,
    waiting: bool = False,
    pending: tuple[str, ...] = (),
    admission_pending: tuple[str, ...] = (),
    backlog: tuple[str, ...] = (),
    request_build_max_pending: int = 4,
    backlog_limit: int = 4,
) -> OmniScheduler:
    scheduler = object.__new__(OmniScheduler)
    scheduler._request_admission_lock = threading.RLock()
    scheduler._request_build_executor = object()
    scheduler.request_build_max_pending = request_build_max_pending
    scheduler._request_build_backlog_limit = backlog_limit
    scheduler._pending_request_builds = {
        rid: (object(), False, object()) for rid in pending
    }
    scheduler._pending_request_admissions = {
        rid: (object(), False, object()) for rid in admission_pending
    }
    scheduler._backlogged_request_build_payloads = deque(
        [_new_stage_payload(rid) for rid in backlog]
    )
    scheduler._deferred_request_payloads = {}
    scheduler._aborted_request_ids = set()
    scheduler.max_queued_requests = max_queued_requests
    scheduler.waiting_queue = [SimpleNamespace(rid="req-occupant")] if waiting else []
    return scheduler


def _stage_ids(payloads) -> list[str]:
    return [payload.request_id for payload in payloads]


@pytest.mark.parametrize(
    "setup, recv, selected, rejected, leftover_backlog",
    [
        pytest.param(
            {
                "max_queued_requests": 1,
                "waiting": True,
                "pending": ("req-busy",),
                "backlog": ("req-backlog",),
                "request_build_max_pending": 1,
                "backlog_limit": 1,
            },
            ["req-new"],
            [],
            ["req-backlog", "req-new"],
            [],
            id="waiting-full-dumps-backlog",
        ),
        pytest.param(
            {"max_queued_requests": 1, "pending": ("req-busy",)},
            ["req-new"],
            [],
            ["req-new"],
            [],
            id="pending-counts-toward-limit",
        ),
        pytest.param(
            {"max_queued_requests": 1, "admission_pending": ("req-busy",)},
            ["req-new"],
            [],
            ["req-new"],
            [],
            id="pending-admission-counts-toward-limit",
        ),
        pytest.param(
            {"max_queued_requests": 1},
            ["req-a", "req-b"],
            ["req-a"],
            ["req-b"],
            [],
            id="does-not-over-select",
        ),
    ],
)
def test_stage_request_build_payloads(
    setup: dict,
    recv: list[str],
    selected: list[str],
    rejected: list[str],
    leftover_backlog: list[str],
) -> None:
    scheduler = _staging_scheduler(**setup)
    got_selected, got_rejected = OmniScheduler._stage_request_build_payloads(
        scheduler, [_new_stage_payload(rid) for rid in recv]
    )
    assert _stage_ids(got_selected) == selected
    assert _stage_ids(got_rejected) == rejected
    assert _stage_ids(scheduler._backlogged_request_build_payloads) == leftover_backlog


def test_upstream_kv_exhaustion_abort_is_translated_to_omni_output() -> None:
    scheduler = object.__new__(OmniScheduler)
    scheduler.outbox = Queue()
    scheduler.is_entry_rank = True
    scheduler.enable_hierarchical_cache = False
    scheduler.forward_ct = 1
    scheduler.server_args = SimpleNamespace()
    scheduler.token_to_kv_pool_allocator = SimpleNamespace(available_size=lambda: 0)
    scheduler.tree_cache = SimpleNamespace(
        req_to_token_pool=SimpleNamespace(mamba_allocator=None)
    )
    scheduler.new_token_ratio_tracker = SimpleNamespace(current=0.5)
    scheduler.metrics_reporter = SimpleNamespace(
        num_retracted_reqs=0,
        enable_metrics=False,
    )
    aborts: list[tuple[str, bool]] = []
    scheduler.abort = lambda rid, *, defer_running_cleanup=True: aborts.append(
        (rid, defer_running_cleanup)
    )
    scheduler.send_to_detokenizer = omni_scheduler_module._NoOpSender()
    scheduler.ipc_channels = omni_scheduler_module._OmniIpcChannels(scheduler)

    req = SimpleNamespace(
        rid="req-kv-exhausted",
        to_finish=omni_scheduler_module.FINISH_ABORT("decode KV exhausted"),
    )

    class ExhaustedBatch:
        def __init__(self) -> None:
            self.reqs = [req]
            self.batch_is_full = True

        def batch_size(self) -> int:
            return len(self.reqs)

        def filter_batch(self) -> None:
            pass

        def is_empty(self) -> bool:
            return not self.reqs

        def check_decode_mem(self) -> bool:
            return False

        def retract_decode(self, _server_args):
            self.reqs = []
            return [], 0.5, [req]

    batch = ExhaustedBatch()
    result = omni_scheduler_module._Upstream.update_running_batch(scheduler, batch)

    output = scheduler.outbox.get_nowait()
    assert result is batch
    assert output.request_id == req.rid
    assert output.type == "error"
    assert "decode KV exhausted" in str(output.data)
    assert aborts == [(req.rid, False)]
    assert batch.reqs == []


def test_upstream_abort_translation_emits_only_on_entry_rank() -> None:
    from sglang.srt.managers.io_struct import AbortReq

    scheduler = object.__new__(OmniScheduler)
    scheduler.outbox = Queue()
    scheduler.is_entry_rank = False
    aborts: list[tuple[str, bool]] = []
    scheduler.abort = lambda rid, *, defer_running_cleanup=True: aborts.append(
        (rid, defer_running_cleanup)
    )
    sender = omni_scheduler_module._UpstreamAbortSender(scheduler)

    sender.send_output(
        AbortReq(
            rid="req-follower",
            finished_reason={"type": "abort", "message": "out of KV"},
        )
    )

    assert scheduler.outbox.empty()
    assert aborts == [("req-follower", False)]


def test_omni_scheduler_custom_runner_stamps_upstream_launch_metadata() -> None:
    """OmniScheduler overrides upstream run_batch, so it must count forwards
    itself; otherwise forward_ct stays 0 and the SGLANG_TEST_RETRACT_INTERVAL
    gate (``forward_ct % INTERVAL == 0``) fires every step. One forward per
    sync run_batch and per async launch; resolve does no forward.
    """

    class FakeModelRunner:
        def execute(self, sched_output):
            return ModelRunnerOutput(
                outputs={},
                can_run_cuda_graph=False,
                next_token_ids=torch.tensor([1], dtype=torch.int32),
            )

        def execute_launch(self, sched_output):
            return SimpleNamespace()

    scheduler = object.__new__(OmniScheduler)
    scheduler._model_runner = FakeModelRunner()
    scheduler._stream_output_builder = None
    scheduler._prefill_start_done = set()
    scheduler._prefill_end_done = set()
    scheduler.forward_ct = 0
    scheduler._sched_idled = True

    def _batch():
        return SimpleNamespace(
            reqs=[
                SimpleNamespace(
                    rid="r", _omni_data=SimpleNamespace(), inflight_middle_chunks=0
                )
            ],
            is_prefill_only=False,
            is_extend_in_batch=False,
        )

    sync_batch = _batch()
    scheduler._run_batch(sync_batch)
    assert scheduler.forward_ct == 1, "sync run_batch must advance forward_ct"
    assert sync_batch.forward_iter == 1
    assert isinstance(sync_batch.launch_ts, float)
    assert sync_batch.after_idle_gap is True

    async_batch = _batch()
    scheduler._run_batch_launch(async_batch)
    assert scheduler.forward_ct == 2, "async launch must advance forward_ct"
    assert async_batch.forward_iter == 2
    assert async_batch.launch_ts >= sync_batch.launch_ts
    assert async_batch.after_idle_gap is False


def test_omni_scheduler_resolve_drops_retracted_req() -> None:
    """A request retracted (KV freed, back to waiting) while its lagged async
    step was in flight must be dropped from the resolve batch — skip_rids plus
    excluded from process_batch_result and next_token_ids — so upstream never
    re-frees its already-freed KV (the double-free assertion). Shared crash-fix
    for the async resolve path used by Higgs and MOSS-TTS-Local.
    """
    captured: dict = {}

    def fake_resolve(batch, sched_output, pending_step, skip_rids=None):
        captured["skip_rids"] = skip_rids
        return SimpleNamespace(next_token_ids=torch.tensor([10, 20], dtype=torch.long))

    def fake_process(batch, result):
        captured["reqs"] = [r.rid for r in batch.reqs]
        captured["ntids"] = result.next_token_ids.tolist()

    scheduler = object.__new__(OmniScheduler)
    scheduler._run_batch_resolve = fake_resolve
    scheduler.process_batch_result = fake_process

    keep = SimpleNamespace(rid="keep", finished=lambda: False, is_retracted=False)
    retr = SimpleNamespace(rid="retr", finished=lambda: False, is_retracted=True)
    batch = SimpleNamespace(reqs=[keep, retr])

    scheduler._resolve_and_process(batch, object(), object())

    assert captured["skip_rids"] == {"retr"}
    assert captured["reqs"] == ["keep"]
    assert captured["ntids"] == [10]  # retracted row trimmed from next_token_ids


def test_omni_scheduler_fast_path_drops_retracted_req() -> None:
    """The synchronous fast path runs after _resolve_pending_async, whose drain can
    retract a req still present in the stale batch. The fast path must drop finished
    AND retracted reqs (not only finished) before run_batch, or a retracted req is
    forwarded/finalized again. The dropped rows' step slots are not freed here:
    the drain's release_kv_cache already covered them (see the real-pool test in
    test_async_decode.py).
    """
    captured: dict = {}

    class FakeBatch:
        def __init__(self, reqs):
            self.reqs = reqs
            self.out_cache_loc = torch.arange(100, 100 + len(reqs))
            self.decoding_reqs = None
            self.forward_mode = None

        def filter_batch(self, keep_indices=None):
            captured["keep_indices"] = keep_indices
            self.reqs = [self.reqs[i] for i in keep_indices]
            self.out_cache_loc = None

    scheduler = object.__new__(OmniScheduler)
    keep = SimpleNamespace(rid="keep", finished=lambda: False, is_retracted=False)
    retr = SimpleNamespace(rid="retr", finished=lambda: False, is_retracted=True)

    # retracted (not finished) must be dropped from the stale batch
    out = scheduler._drop_stale_overrun(FakeBatch([keep, retr]))
    assert captured["keep_indices"] == [0]
    assert [r.rid for r in out.reqs] == ["keep"]
    assert out.out_cache_loc.tolist() == [100]

    # all dropped -> None so run_batch is skipped
    fin = SimpleNamespace(rid="fin", finished=lambda: True, is_retracted=False)
    assert scheduler._drop_stale_overrun(FakeBatch([retr, fin])) is None

    # nothing stale -> batch returned unchanged, filter_batch never called
    captured.clear()
    clean = FakeBatch([keep])
    assert scheduler._drop_stale_overrun(clean) is clean
    assert "keep_indices" not in captured


def test_omni_scheduler_abort_propagates_immediate_kv_cleanup_failure(
    monkeypatch,
) -> None:
    """Immediate abort cleanup must not hide allocator failures."""

    def fail_release(_req, _cache) -> None:
        raise RuntimeError("kv cleanup failed")

    monkeypatch.setattr(omni_scheduler_module, "release_kv_cache", fail_release)
    scheduler = object.__new__(OmniScheduler)
    scheduler._abort_callback = None
    scheduler._aborted_request_ids = set()
    scheduler._aborted_request_id_order = deque()
    scheduler._pending_stream_ingress = {}
    scheduler._deferred_request_payloads = {}
    scheduler._dirty_deferred_request_ids = set()
    scheduler._first_emit_done = set()
    scheduler._prefill_start_done = set()
    scheduler._prefill_end_done = set()
    scheduler.inbox = Queue()
    scheduler.waiting_queue = []
    scheduler.tree_cache = object()

    req = SimpleNamespace(
        rid="req-fail",
        _omni_data=SimpleNamespace(),
        req_pool_idx=1,
        mamba_pool_idx=None,
    )
    batch = SimpleNamespace(reqs=[req], batch_is_full=True)
    scheduler.running_batch = batch
    scheduler.cur_batch = batch
    scheduler.last_batch = None
    _init_sync_request_build_state(scheduler)

    with pytest.raises(RuntimeError, match="kv cleanup failed"):
        scheduler.abort("req-fail", defer_running_cleanup=False)

    assert batch.reqs == [req]


def test_omni_scheduler_abort_marks_running_request_for_finish(monkeypatch) -> None:
    """Running aborts follow upstream SGLang's deferred KV cleanup path."""
    cleaned: list[str] = []
    release_calls: list[str] = []
    monkeypatch.setattr(
        omni_scheduler_module,
        "release_kv_cache",
        lambda req, _cache: release_calls.append(req.rid),
    )
    model_path_ends: list[tuple[str, str]] = []
    monkeypatch.setattr(
        omni_scheduler_module,
        "_emit_model_path_end",
        lambda rid, *, status: model_path_ends.append((rid, status)),
    )
    scheduler = object.__new__(OmniScheduler)
    scheduler._abort_callback = cleaned.append
    scheduler._request_finished_callback = None
    scheduler._aborted_request_ids = set()
    scheduler._aborted_request_id_order = deque()
    scheduler._completed_request_ids = {}
    scheduler._pending_stream_ingress = {"req-run": _ingress("stale", done=True)}
    scheduler._deferred_request_payloads = {"req-run": object()}
    scheduler._dirty_deferred_request_ids = {"req-run"}
    scheduler._first_emit_done = {"req-run"}
    scheduler._prefill_start_done = {"req-run"}
    scheduler._prefill_end_done = set()
    scheduler.inbox = Queue()
    scheduler.waiting_queue = []

    req = SimpleNamespace(
        rid="req-run",
        to_finish=None,
        finished_reason=None,
        req_pool_idx=1,
        is_retracted=False,
        finished=lambda: False,
        _omni_terminal_claimed=False,
    )
    batch = SimpleNamespace(reqs=[req], batch_is_full=True)
    scheduler.running_batch = batch
    scheduler.cur_batch = batch
    scheduler.last_batch = None
    _init_sync_request_build_state(scheduler)

    scheduler.abort("req-run")

    assert req in batch.reqs
    assert req.to_finish.to_json()["type"] == "abort"
    assert cleaned == []
    assert release_calls == []
    assert scheduler._aborted_request_ids == {"req-run"}
    assert scheduler._pending_stream_ingress == {}
    assert scheduler._deferred_request_payloads == {}
    assert scheduler._dirty_deferred_request_ids == set()
    assert scheduler._first_emit_done == set()
    # The model-path interval closes at abort time rather than waiting for
    # stream_output, which a running abort is not guaranteed to reach.
    assert scheduler._prefill_start_done == set()
    assert model_path_ends == [("req-run", "aborted")]
    req.finished = lambda: True
    scheduler.stream_output([req])
    assert cleaned == ["req-run"]
    assert scheduler._prefill_start_done == set()
    assert model_path_ends == [("req-run", "aborted")]


def test_omni_scheduler_abort_cleans_queued_request_immediately(monkeypatch) -> None:
    """Queued aborts have no KV allocation, so callback cleanup can run now."""
    cleaned: list[str] = []
    model_path_ends: list[tuple[str, str]] = []
    monkeypatch.setattr(
        omni_scheduler_module,
        "_emit_model_path_end",
        lambda rid, *, status: model_path_ends.append((rid, status)),
    )
    scheduler = object.__new__(OmniScheduler)
    scheduler._abort_callback = cleaned.append
    scheduler._aborted_request_ids = set()
    scheduler._aborted_request_id_order = deque()
    scheduler._pending_stream_ingress = {}
    scheduler._deferred_request_payloads = {}
    scheduler._dirty_deferred_request_ids = set()
    scheduler._first_emit_done = set()
    scheduler._prefill_start_done = set()
    scheduler._prefill_end_done = set()
    scheduler.inbox = Queue()

    req = SimpleNamespace(rid="req-wait")
    request_data = SimpleNamespace(req=req)
    req._omni_data = request_data
    scheduler.waiting_queue = [req]
    scheduler.running_batch = SimpleNamespace(reqs=[], batch_is_full=False)
    scheduler.cur_batch = None
    scheduler.last_batch = None
    _init_sync_request_build_state(scheduler)

    scheduler.abort("req-wait")

    assert scheduler.waiting_queue == []
    assert cleaned == ["req-wait"]
    assert req._omni_data is None
    assert request_data.req is req


def test_omni_scheduler_abort_treats_retracted_alias_as_waiting_owned() -> None:
    cleaned: list[str] = []
    scheduler = object.__new__(OmniScheduler)
    scheduler._abort_callback = cleaned.append
    scheduler._aborted_request_ids = set()
    scheduler._aborted_request_id_order = deque()
    scheduler._pending_stream_ingress = {}
    scheduler._deferred_request_payloads = {}
    scheduler._dirty_deferred_request_ids = set()
    scheduler._first_emit_done = set()
    scheduler._prefill_start_done = set()
    scheduler._prefill_end_done = set()
    scheduler.inbox = Queue()
    scheduler.tree_cache = None

    req = SimpleNamespace(
        rid="req-retracted",
        is_retracted=True,
        finished=lambda: False,
        to_finish=None,
        finished_reason=None,
        req_pool_idx=None,
        mamba_pool_idx=None,
        _omni_terminal_claimed=False,
    )
    request_data = SimpleNamespace(req=req)
    req._omni_data = request_data
    stale_batch = SimpleNamespace(reqs=[req], batch_is_full=True)
    scheduler.waiting_queue = [req]
    scheduler.running_batch = SimpleNamespace(reqs=[], batch_is_full=False)
    scheduler.cur_batch = None
    scheduler.last_batch = stale_batch
    _init_sync_request_build_state(scheduler)

    scheduler.abort("req-retracted")

    assert scheduler.waiting_queue == []
    assert stale_batch.reqs == []
    assert req.to_finish is None
    assert req._omni_data is None
    assert request_data.req is req
    assert cleaned == ["req-retracted"]


def test_omni_scheduler_emit_stream_output_skips_aborted_requests() -> None:
    """A mid-step abort must not ship one more chunk to the vocoder."""
    scheduler = object.__new__(OmniScheduler)
    scheduler.outbox = Queue()
    scheduler._aborted_request_ids = {"req-aborted"}
    scheduler._first_emit_done = set()
    scheduler._stream_output_builder = lambda rid, data, output: [
        SimpleNamespace(request_id=rid, type="stream")
    ]

    sched_output = SimpleNamespace(
        requests=[
            SimpleNamespace(request_id="req-live", data=None),
            SimpleNamespace(request_id="req-aborted", data=None),
        ]
    )
    mr_output = SimpleNamespace(outputs={"req-live": object(), "req-aborted": object()})

    scheduler._emit_stream_output(sched_output, mr_output)

    assert scheduler.outbox.get_nowait().request_id == "req-live"
    assert scheduler.outbox.empty()


def test_omni_scheduler_flushes_stream_before_terminal_result(monkeypatch) -> None:
    scheduler = object.__new__(OmniScheduler)
    _init_terminal_output_state(scheduler)
    scheduler.outbox = Queue()
    scheduler._aborted_request_ids = set()
    scheduler._first_emit_done = {"req-finished"}
    scheduler._prefill_start_done = {"req-finished"}
    scheduler._prefill_end_done = set()
    calls: list[str] = []
    model_path_ends: list[tuple[str, str]] = []
    monkeypatch.setattr(
        omni_scheduler_module,
        "_emit_model_path_end",
        lambda rid, *, status: model_path_ends.append((rid, status)),
    )

    request_data = SimpleNamespace(
        prefill_input_embeds=None,
        decode_input_embeds=None,
    )
    req = SimpleNamespace(
        rid="req-finished",
        _omni_data=request_data,
        output_ids=[1, 2],
        finished=lambda: True,
        finished_reason=None,
        _omni_terminal_claimed=False,
    )
    request_data.req = req

    def stream_output_builder(rid, data, output):
        raise AssertionError("terminal flush must use the explicit flush hook")

    def flush_stream_output(rid, data):
        assert rid == "req-finished"
        assert data is request_data
        calls.append("flush")
        return [SimpleNamespace(request_id=rid, type="stream")]

    stream_output_builder.flush = flush_stream_output

    def result_adapter(data):
        assert data is request_data
        calls.append("result")
        return {"text": "AB"}

    scheduler._stream_output_builder = stream_output_builder
    scheduler._result_adapter = result_adapter

    scheduler.stream_output([req])

    assert calls == ["flush", "result"]
    assert scheduler.outbox.get_nowait().type == "stream"
    assert scheduler.outbox.get_nowait().type == "result"
    assert req._omni_data is None
    assert request_data.req is req
    assert model_path_ends == [("req-finished", "success")]


def test_omni_scheduler_fish_abort_during_step_suppresses_chunk_and_result() -> None:
    """A Fish abort landing mid-step defers per-request cleanup to the
    upstream FINISH_ABORT path, leaves the buffered codes unconsumed, and
    ships neither the pending stream chunk nor the terminal result."""
    from sglang_omni.models.fishaudio_s2_pro.request_builders import (
        make_tts_scheduler_adapters,
    )
    from tests.unit_test.fixtures.fish_fakes import (
        FakeFishTokenizer,
        make_s2pro_payload,
    )

    _, result_adapter, stream_output_builder = make_tts_scheduler_adapters(
        tokenizer=FakeFishTokenizer()
    )
    adapted: list = []
    cleaned: list = []
    scheduler = object.__new__(OmniScheduler)
    scheduler.outbox = Queue()
    scheduler.inbox = Queue()
    scheduler._abort_callback = cleaned.append
    scheduler._request_finished_callback = None
    scheduler._aborted_request_ids = set()
    scheduler._aborted_request_id_order = deque()
    scheduler._completed_request_ids = {}
    scheduler._pending_stream_ingress = {}
    scheduler._deferred_request_payloads = {}
    scheduler._dirty_deferred_request_ids = set()
    scheduler._first_emit_done = set()
    scheduler._prefill_start_done = set()
    scheduler._prefill_end_done = set()
    scheduler._stream_output_builder = stream_output_builder

    def tracking_result_adapter(data):
        adapted.append(data)
        return result_adapter(data)

    scheduler._result_adapter = tracking_result_adapter
    scheduler.waiting_queue = []
    _init_sync_request_build_state(scheduler)

    codes = torch.full((11, 1), 7, dtype=torch.long)
    data = SimpleNamespace(
        stage_payload=make_s2pro_payload(
            request_id="req-fish", params={"stream": True}
        ),
        latest_stream_code_chunk=codes,
        output_codes=[codes],
    )
    req = SimpleNamespace(
        rid="req-fish",
        to_finish=None,
        finished=lambda: False,
        finished_reason=None,
        req_pool_idx=1,
        is_retracted=False,
        _omni_data=data,
        _omni_terminal_claimed=False,
    )
    data.req = req
    batch = SimpleNamespace(reqs=[req], batch_is_full=True)
    scheduler.running_batch = batch
    scheduler.cur_batch = batch
    scheduler.last_batch = None

    scheduler.abort("req-fish")

    assert req in batch.reqs
    assert req.to_finish.to_json()["type"] == "abort"
    assert cleaned == []

    sched_output = SimpleNamespace(
        requests=[SimpleNamespace(request_id="req-fish", data=data)]
    )
    mr_output = SimpleNamespace(outputs={"req-fish": object()})
    scheduler._emit_stream_output(sched_output, mr_output)

    assert scheduler.outbox.empty()
    assert data.latest_stream_code_chunk is codes
    assert len(data.output_codes) == 1 and data.output_codes[0] is codes

    req.finished = lambda: True
    scheduler.stream_output([req])

    assert adapted == []
    assert cleaned == ["req-fish"]
    assert scheduler.outbox.empty()
    assert req._omni_data is None
    assert data.req is req


def test_stream_output_drains_runner_before_terminal_payload() -> None:
    """The runner hook must fire on a non-abort finish, and strictly before the
    terminal payload lands on the shared outbox."""
    calls: list[tuple[str, object, int]] = []
    scheduler = object.__new__(OmniScheduler)
    _init_terminal_output_state(scheduler)
    scheduler.outbox = Queue()
    scheduler._aborted_request_ids = set()
    scheduler._first_emit_done = set()
    scheduler._prefill_start_done = set()
    scheduler._prefill_end_done = set()
    scheduler._result_adapter = lambda data: {"ok": True}

    data = SimpleNamespace(prefill_input_embeds=None, decode_input_embeds=None)
    scheduler._model_runner = SimpleNamespace(
        on_request_finished=lambda rid, req_data: calls.append(
            (rid, req_data, scheduler.outbox.qsize())
        )
    )
    req = SimpleNamespace(
        rid="req-1",
        finished=lambda: True,
        finished_reason=None,
        output_ids=[7],
        _omni_data=data,
        _omni_terminal_claimed=False,
    )
    data.req = req

    scheduler.stream_output([req])

    # qsize 0 at call time proves the flush is ordered ahead of the result.
    assert calls == [("req-1", data, 0)]
    assert scheduler.outbox.qsize() == 1
    assert scheduler.outbox.get().type == "result"
    assert req._omni_data is None
    assert data.req is req


def test_stream_output_cleans_request_when_runner_finish_hook_fails() -> None:
    scheduler = object.__new__(OmniScheduler)
    _init_terminal_output_state(scheduler)
    scheduler.outbox = Queue()
    scheduler._aborted_request_ids = set()
    scheduler._first_emit_done = set()
    scheduler._prefill_start_done = set()
    scheduler._prefill_end_done = set()
    cleanup_calls: list[str] = []
    scheduler._request_finished_callback = cleanup_calls.append

    def fail_finish_hook(_rid, _data):
        raise RuntimeError("finish hook failed")

    scheduler._model_runner = SimpleNamespace(on_request_finished=fail_finish_hook)
    data = SimpleNamespace(prefill_input_embeds=None, decode_input_embeds=None)
    req = SimpleNamespace(
        rid="req-hook-error",
        finished=lambda: True,
        finished_reason=None,
        output_ids=[],
        _omni_data=data,
        _omni_terminal_claimed=False,
    )
    data.req = req

    scheduler.stream_output([req])

    assert req._omni_data is None
    assert data.req is req
    assert cleanup_calls == ["req-hook-error"]
    error = scheduler.outbox.get_nowait()
    assert error.type == "error"
    assert "finish hook failed" in str(error.data)


def test_stream_output_releases_request_when_terminal_flush_fails() -> None:
    scheduler = object.__new__(OmniScheduler)
    _init_terminal_output_state(scheduler)
    scheduler.outbox = Queue()
    scheduler._aborted_request_ids = set()
    scheduler._first_emit_done = set()
    scheduler._prefill_start_done = set()
    scheduler._prefill_end_done = set()
    cleanup_calls: list[str] = []
    scheduler._request_finished_callback = cleanup_calls.append

    def fail_flush(_rid, _data):
        raise RuntimeError("flush failed")

    scheduler._stream_output_builder = SimpleNamespace(flush=fail_flush)
    scheduler._result_adapter = lambda _data: pytest.fail(
        "the result adapter must not run after a failed terminal flush"
    )
    data = SimpleNamespace(prefill_input_embeds=None, decode_input_embeds=None)
    req = SimpleNamespace(
        rid="req-flush-error",
        finished=lambda: True,
        finished_reason=None,
        output_ids=[],
        _omni_data=data,
        _omni_terminal_claimed=False,
    )
    data.req = req

    scheduler.stream_output([req])

    assert cleanup_calls == ["req-flush-error"]
    assert req._omni_data is None
    error = scheduler.outbox.get_nowait()
    assert error.type == "error"
    assert "flush failed" in str(error.data)


def test_stream_output_atomically_claims_request_data_against_abort() -> None:
    data_read_started = threading.Event()
    abort_started = threading.Event()
    abort_done = threading.Event()

    class InstrumentedRLock:
        def __init__(self):
            self._lock = threading.RLock()
            self._owner: int | None = None
            self.contender_waiting = threading.Event()

        def __enter__(self):
            thread_id = threading.get_ident()
            if self._owner is not None and self._owner != thread_id:
                self.contender_waiting.set()
            self._lock.acquire()
            self._owner = thread_id
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            self._owner = None
            self._lock.release()

        def is_owned_by_current_thread(self) -> bool:
            return self._owner == threading.get_ident()

    terminal_lock = InstrumentedRLock()

    class Request:
        def __init__(self, data):
            self.rid = "req-terminal-abort-race"
            self._data = data
            self.output_ids = []
            self.finished_reason = None
            self.is_retracted = False
            self.to_finish = None
            self.req_pool_idx = None
            self.mamba_pool_idx = None
            self._omni_terminal_claimed = False

        @property
        def _omni_data(self):
            data_read_started.set()
            assert abort_started.wait(timeout=1)
            if terminal_lock.is_owned_by_current_thread():
                # New code: prove abort has reached and is blocked on the lock
                # before allowing the terminal data read to complete.
                assert terminal_lock.contender_waiting.wait(timeout=1)
            else:
                # Negative control for the old unlocked code: wait until abort
                # has detached the request, so this read deterministically
                # returns None and exposes the race.
                assert abort_done.wait(timeout=1)
            return self._data

        @_omni_data.setter
        def _omni_data(self, value):
            self._data = value

        def finished(self):
            return True

    scheduler = object.__new__(OmniScheduler)
    scheduler.outbox = Queue()
    scheduler.inbox = Queue()
    scheduler._aborted_request_ids = set()
    scheduler._aborted_request_id_order = deque()
    scheduler._completed_request_ids = {}
    scheduler._pending_stream_ingress = {}
    scheduler._deferred_request_payloads = {}
    scheduler._dirty_deferred_request_ids = set()
    scheduler._first_emit_done = set()
    scheduler._prefill_start_done = set()
    scheduler._prefill_end_done = set()
    abort_cleanup: list[str] = []
    finished_cleanup: list[str] = []
    scheduler._abort_callback = abort_cleanup.append
    scheduler._request_finished_callback = finished_cleanup.append
    scheduler._result_adapter = lambda _data: {"ok": True}
    scheduler._model_runner = None
    scheduler._stream_output_builder = None
    scheduler.tree_cache = None
    scheduler.waiting_queue = []
    _init_sync_request_build_state(scheduler)
    scheduler._request_admission_lock = terminal_lock

    data = SimpleNamespace(
        prefill_input_embeds=None,
        decode_input_embeds=None,
    )
    req = Request(data)
    data.req = req
    batch = SimpleNamespace(reqs=[req], batch_is_full=True)
    scheduler.running_batch = batch
    scheduler.cur_batch = batch
    scheduler.last_batch = None

    def abort_request() -> None:
        assert data_read_started.wait(timeout=1)
        abort_started.set()
        scheduler.abort(req.rid)
        abort_done.set()

    abort_thread = threading.Thread(target=abort_request)
    abort_thread.start()
    scheduler.stream_output([req])
    abort_thread.join(timeout=1)

    assert not abort_thread.is_alive()
    assert abort_done.is_set()
    assert req._data is None
    assert data.req is req
    assert finished_cleanup == [req.rid]
    assert abort_cleanup == [req.rid]
    assert scheduler.outbox.get_nowait().type == "result"


def test_abort_after_terminal_close_runs_its_own_cleanup() -> None:
    scheduler = object.__new__(OmniScheduler)
    scheduler.inbox = Queue()
    scheduler._aborted_request_ids = set()
    scheduler._aborted_request_id_order = deque()
    scheduler._completed_request_ids = {}
    scheduler._pending_stream_ingress = {}
    scheduler._deferred_request_payloads = {}
    scheduler._dirty_deferred_request_ids = set()
    scheduler._first_emit_done = set()
    scheduler._prefill_start_done = set()
    scheduler._prefill_end_done = set()
    scheduler.waiting_queue = []
    scheduler.tree_cache = None
    cleaned: list[str] = []
    scheduler._abort_callback = cleaned.append
    _init_sync_request_build_state(scheduler)

    data = SimpleNamespace()
    req = SimpleNamespace(
        rid="req-abort-after-close",
        _omni_data=data,
        _omni_terminal_claimed=True,
        req_pool_idx=None,
        mamba_pool_idx=None,
    )
    data.req = req
    batch = SimpleNamespace(reqs=[req], batch_is_full=True)
    scheduler.running_batch = batch
    scheduler.cur_batch = batch
    scheduler.last_batch = None

    assert scheduler._close_completed_request(req) is False
    assert req._omni_data is None
    assert batch.reqs == [req]

    scheduler.abort(req.rid)

    assert cleaned == [req.rid]
    assert batch.reqs == []


def test_abort_publishes_request_id_before_marking_terminal_finish() -> None:
    class ObservedRLock:
        def __init__(self):
            self._lock = threading.RLock()
            self._owner: int | None = None
            self.contender_waiting = threading.Event()

        def __enter__(self):
            thread_id = threading.get_ident()
            if self._owner is not None and self._owner != thread_id:
                self.contender_waiting.set()
            self._lock.acquire()
            self._owner = thread_id
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            self._owner = None
            self._lock.release()

    scheduler = object.__new__(OmniScheduler)
    _init_sync_request_build_state(scheduler)
    scheduler._request_admission_lock = ObservedRLock()
    scheduler.outbox = Queue()
    scheduler.inbox = Queue()
    scheduler._aborted_request_ids = set()
    scheduler._aborted_request_id_order = deque()
    scheduler._completed_request_ids = {}
    scheduler._pending_stream_ingress = {}
    scheduler._deferred_request_payloads = {}
    scheduler._dirty_deferred_request_ids = set()
    scheduler._first_emit_done = set()
    scheduler._prefill_start_done = set()
    scheduler._prefill_end_done = set()
    cleaned: list[str] = []
    scheduler._abort_callback = cleaned.append
    scheduler._request_finished_callback = None
    scheduler._result_adapter = lambda _data: pytest.fail(
        "an abort-winning terminal request must not be adapted"
    )
    scheduler.tree_cache = None
    scheduler.waiting_queue = []

    data = SimpleNamespace(prefill_input_embeds=None, decode_input_embeds=None)
    req = SimpleNamespace(
        rid="req-abort-wins",
        output_ids=[],
        finished_reason=None,
        is_retracted=False,
        to_finish=None,
        req_pool_idx=None,
        mamba_pool_idx=None,
        _omni_data=data,
        _omni_terminal_claimed=False,
    )
    req.finished = lambda: req.to_finish is not None
    data.req = req
    batch = SimpleNamespace(reqs=[req], batch_is_full=True)
    scheduler.running_batch = batch
    scheduler.cur_batch = batch
    scheduler.last_batch = None

    mark_started = threading.Event()

    def controlled_mark(request_id: str) -> bool:
        assert request_id in scheduler._aborted_request_ids
        req.to_finish = object()
        mark_started.set()
        assert scheduler._request_admission_lock.contender_waiting.wait(timeout=1)
        return True

    scheduler._mark_running_request_aborted = controlled_mark
    thread_errors: list[BaseException] = []

    def run_in_thread(fn) -> None:
        try:
            fn()
        except BaseException as exc:
            thread_errors.append(exc)

    abort_thread = threading.Thread(
        target=run_in_thread,
        args=(lambda: scheduler.abort(req.rid),),
    )
    abort_thread.start()
    assert mark_started.wait(timeout=1)

    terminal_thread = threading.Thread(
        target=run_in_thread,
        args=(lambda: scheduler.stream_output([req]),),
    )
    terminal_thread.start()
    abort_thread.join(timeout=1)
    terminal_thread.join(timeout=1)

    assert not abort_thread.is_alive()
    assert not terminal_thread.is_alive()
    assert thread_errors == []
    assert scheduler._aborted_request_ids == {req.rid}
    assert req._omni_data is None
    assert cleaned == [req.rid]
    assert scheduler.outbox.empty()


def test_terminal_request_data_is_collectable_without_cyclic_gc() -> None:
    class Request:
        pass

    class RequestData:
        pass

    scheduler = object.__new__(OmniScheduler)
    _init_terminal_output_state(scheduler)
    scheduler.outbox = Queue()
    scheduler._aborted_request_ids = set()
    scheduler._first_emit_done = set()
    scheduler._prefill_start_done = set()
    scheduler._prefill_end_done = set()
    scheduler._result_adapter = lambda _data: None

    gc_was_enabled = gc.isenabled()
    gc.disable()
    try:
        req = Request()
        req.rid = "req-collectable"
        req.finished = lambda: True
        req.finished_reason = None
        req.output_ids = []
        req._omni_terminal_claimed = False
        data = RequestData()
        data.prefill_input_embeds = None
        data.decode_input_embeds = None
        req._omni_data = data
        data.req = req
        req_ref = weakref.ref(req)
        data_ref = weakref.ref(data)

        scheduler.stream_output([req])

        del req
        del data
        assert req_ref() is None
        assert data_ref() is None
    finally:
        if gc_was_enabled:
            gc.enable()


def test_stream_output_skips_runner_hook_for_aborted_requests() -> None:
    calls: list[str] = []
    scheduler = object.__new__(OmniScheduler)
    _init_terminal_output_state(scheduler)
    scheduler.outbox = Queue()
    scheduler._aborted_request_ids = {"req-1"}
    scheduler._abort_callback = None
    scheduler._first_emit_done = set()
    scheduler._prefill_start_done = set()
    scheduler._prefill_end_done = set()
    scheduler._model_runner = SimpleNamespace(
        on_request_finished=lambda rid, _data: calls.append(rid)
    )
    data = SimpleNamespace()
    req = SimpleNamespace(
        rid="req-1",
        finished=lambda: True,
        finished_reason=None,
        _omni_data=data,
        _omni_terminal_claimed=False,
    )
    data.req = req

    scheduler.stream_output([req])

    assert calls == []
    assert scheduler.outbox.empty()
    assert req._omni_data is None
    assert data.req is req


def test_stream_output_closes_late_stream_ingress() -> None:
    scheduler = object.__new__(OmniScheduler)
    _init_terminal_output_state(scheduler)
    scheduler.outbox = Queue()
    scheduler._aborted_request_ids = set()
    scheduler._first_emit_done = set()
    scheduler._prefill_start_done = set()
    scheduler._prefill_end_done = set()
    scheduler._result_adapter = lambda _data: {"ok": True}

    data = SimpleNamespace(prefill_input_embeds=None, decode_input_embeds=None)
    req = SimpleNamespace(
        rid="req-late-stream",
        finished=lambda: True,
        finished_reason=None,
        output_ids=[],
        _omni_data=data,
        _omni_terminal_claimed=False,
    )
    data.req = req

    scheduler.stream_output([req])
    scheduler._on_stream_chunk(req.rid, "late")
    scheduler._on_stream_done(req.rid)

    assert req.rid in scheduler._completed_request_ids
    assert req.rid not in scheduler._pending_stream_ingress


def test_completed_request_id_is_cleared_on_explicit_readmission() -> None:
    scheduler = object.__new__(OmniScheduler)
    scheduler.tp_size = 1
    scheduler.is_entry_rank = True
    scheduler.outbox = Queue()
    scheduler._aborted_request_ids = set()
    scheduler._completed_request_ids = {"req-complete": None}
    scheduler._pending_stream_ingress = {}
    scheduler.inbox = Queue()
    scheduler.inbox.put(
        IncomingMessage(
            request_id="req-complete",
            type="new_request",
            data=object(),
        )
    )

    new_reqs = scheduler.recv_requests()

    assert len(new_reqs) == 1
    assert "req-complete" not in scheduler._completed_request_ids
    assert scheduler.outbox.empty()


def test_pending_stream_requests_are_bounded(monkeypatch, caplog) -> None:
    monkeypatch.setattr(omni_scheduler_module, "_PENDING_STREAM_REQUEST_LIMIT", 3)
    monkeypatch.setattr(omni_scheduler_module, "_PENDING_STREAM_REQUEST_RETAINED", 2)
    scheduler = object.__new__(OmniScheduler)
    scheduler.running_batch = None
    scheduler.cur_batch = None
    scheduler.last_batch = None
    scheduler._async_pending = None
    scheduler.waiting_queue = []
    scheduler._completed_request_ids = {}
    scheduler._pending_stream_ingress = {}
    scheduler._deferred_request_payloads = {}
    scheduler._dirty_deferred_request_ids = set()

    for index in range(4):
        scheduler._on_stream_chunk(f"req-{index}", index)

    # Eviction is oldest-first: the still-fresh req-2 survives alongside the
    # arrival that triggered the eviction.
    assert list(scheduler._pending_stream_ingress) == ["req-2", "req-3"]
    assert scheduler._pending_stream_ingress["req-3"].chunks == [3]
    assert "evicted 2 pending stream request(s)" in caplog.text


def test_completed_request_tombstones_evict_oldest(monkeypatch) -> None:
    monkeypatch.setattr(omni_scheduler_module, "_COMPLETED_REQUEST_ID_LIMIT", 3)
    scheduler = object.__new__(OmniScheduler)
    scheduler._completed_request_ids = {}
    scheduler._pending_stream_ingress = {}

    for request_id in ("r0", "r1", "r2", "r3"):
        scheduler._remember_completed_request(request_id)

    assert list(scheduler._completed_request_ids) == ["r1", "r2", "r3"]


def test_stream_output_drops_stale_terminal_alias_without_raising() -> None:
    scheduler = object.__new__(OmniScheduler)
    _init_terminal_output_state(scheduler)
    scheduler.outbox = Queue()
    scheduler._aborted_request_ids = set()
    scheduler._first_emit_done = set()
    scheduler._prefill_start_done = set()
    scheduler._prefill_end_done = set()

    req = SimpleNamespace(
        rid="req-stale-terminal",
        finished=lambda: True,
        finished_reason=None,
        _omni_data=None,
        _omni_terminal_claimed=False,
    )

    scheduler.stream_output([req])

    assert req.rid in scheduler._completed_request_ids
    assert scheduler.outbox.empty()


def test_omni_scheduler_abort_caps_aborted_id_set() -> None:
    """The aborted-id set is trimmed instead of growing without bound."""
    scheduler = object.__new__(OmniScheduler)
    scheduler._abort_callback = None
    scheduler._aborted_request_ids = set()
    scheduler._aborted_request_id_order = deque()
    for i in range(omni_scheduler_module._ABORTED_REQUEST_ID_LIMIT):
        scheduler._aborted_request_ids.add(f"req-{i}")
        scheduler._aborted_request_id_order.append(f"req-{i}")
    scheduler._pending_stream_ingress = {}
    scheduler._deferred_request_payloads = {}
    scheduler._dirty_deferred_request_ids = set()
    scheduler._first_emit_done = set()
    scheduler._prefill_start_done = set()
    scheduler._prefill_end_done = set()
    scheduler.inbox = Queue()
    scheduler.waiting_queue = []
    scheduler.running_batch = SimpleNamespace(reqs=[], batch_is_full=False)
    scheduler.cur_batch = None
    scheduler.last_batch = None
    _init_sync_request_build_state(scheduler)

    scheduler.abort("req-overflow")

    assert "req-overflow" in scheduler._aborted_request_ids
    assert "req-0" not in scheduler._aborted_request_ids
    newest = f"req-{omni_scheduler_module._ABORTED_REQUEST_ID_LIMIT - 1}"
    assert newest in scheduler._aborted_request_ids
    assert (
        len(scheduler._aborted_request_ids)
        == omni_scheduler_module._ABORTED_REQUEST_ID_RETAINED
    )


def test_omni_scheduler_distinguishes_queue_enter_from_prefill_start(
    monkeypatch,
) -> None:
    """Queueing a built request must not report actual prefill execution."""
    events: list[dict] = []
    monkeypatch.setattr(
        "sglang_omni.scheduling.omni_scheduler._emit_event",
        lambda **kwargs: events.append(kwargs),
    )
    model_path_starts: list[str] = []
    monkeypatch.setattr(
        "sglang_omni.scheduling.omni_scheduler._emit_model_path_start",
        model_path_starts.append,
    )
    scheduler = object.__new__(OmniScheduler)
    scheduler.outbox = Queue()
    scheduler.waiting_queue = []
    scheduler._pending_stream_ingress = {}
    scheduler._deferred_request_payloads = {}
    scheduler._dirty_deferred_request_ids = set()
    scheduler._aborted_request_ids = set()
    scheduler._aborted_request_id_order = deque()
    scheduler._prefill_start_done = set()
    scheduler._prefill_end_done = set()
    scheduler.max_req_len = 16
    scheduler.max_req_input_len = 16
    _init_sync_request_build_state(scheduler)

    req = SimpleNamespace(
        rid="req-delayed",
        origin_input_ids=[1, 2, 3],
        origin_input_ids_unpadded=[1, 2, 3],
        sampling_params=SimpleNamespace(max_new_tokens=1, min_new_tokens=0),
        output_ids=[],
        priority=None,
    )
    scheduler._request_builder = lambda payload: SimpleNamespace(
        req=req,
        enforce_request_limits=False,
        max_new_tokens=1,
    )

    scheduler.process_input_requests([_new_stage_payload("req-delayed")])

    names = [event["event_name"] for event in events]
    assert "scheduler_queue_enter" in names
    assert "scheduler_prefill_start" not in names
    assert scheduler.waiting_queue == [req]

    batch = SimpleNamespace(reqs=[req], is_prefill_only=True, is_extend_in_batch=False)
    scheduler._emit_prefill_start_for_batch(batch)
    scheduler._emit_prefill_start_for_batch(batch)

    names = [event["event_name"] for event in events]
    assert names.count("scheduler_prefill_start") == 1
    assert names.index("scheduler_queue_enter") < names.index("scheduler_prefill_start")
    assert model_path_starts == ["req-delayed"]


def test_omni_scheduler_normalizes_req_token_arrays() -> None:
    origin = [1, 2, 3]
    req = SimpleNamespace(
        origin_input_ids=origin,
        origin_input_ids_unpadded=origin,
    )

    OmniScheduler._normalize_req_token_arrays(req)

    assert isinstance(req.origin_input_ids, array)
    assert req.origin_input_ids.tolist() == origin
    assert req.origin_input_ids_unpadded is req.origin_input_ids

    OmniScheduler._normalize_req_token_arrays(req)
    assert req.origin_input_ids.tolist() == origin


def _construct_omni_scheduler(
    monkeypatch,
    *,
    return_runtime_context: bool = False,
    server_max_queued_requests: int | None = 7,
    **kwargs,
) -> OmniScheduler | tuple[OmniScheduler, object]:
    """Build an OmniScheduler over the minimum stub surface __init__ touches."""
    monkeypatch.setattr(
        OmniScheduler,
        "_init_parallel_state",
        lambda self, _tp_worker: setattr(self, "ps", SimpleNamespace(pp_size=1)),
    )
    monkeypatch.setattr(
        OmniScheduler,
        "init_metrics_collector",
        lambda self, *_args, **_kwargs: None,
        raising=False,
    )
    monkeypatch.setattr(
        OmniScheduler,
        "init_metrics_reporter",
        lambda self, *_args, **_kwargs: setattr(
            self,
            "metrics_reporter",
            SimpleNamespace(
                reset_metrics=lambda: None,
                is_stats_logging_rank=False,
            ),
        ),
        raising=False,
    )

    class StrictParallelContext:
        def __init__(self) -> None:
            object.__setattr__(self, "pp_max_micro_batch_size", None)
            object.__setattr__(self, "attn_dcp_size", 1)

        def __setattr__(self, name, value) -> None:
            raise AttributeError(f"bare mutation of {name}")

    class StrictRuntimeContext:
        def __init__(self, parallel) -> None:
            self.parallel = parallel
            self.override_calls = []

        def override(self, source, **fields) -> None:
            self.override_calls.append((source, dict(fields)))
            for name, value in fields.items():
                object.__setattr__(self.parallel, name, value)

    parallel_context = StrictParallelContext()
    runtime_context = StrictRuntimeContext(parallel_context)
    monkeypatch.setattr(
        "sglang.srt.runtime_context.get_parallel",
        lambda: parallel_context,
    )
    monkeypatch.setattr(
        "sglang.srt.runtime_context.get_context",
        lambda: runtime_context,
    )
    tp_worker = SimpleNamespace(
        gpu_id=0,
        tp_rank=0,
        model_runner=SimpleNamespace(
            max_total_num_tokens=128,
            effective_max_total_num_tokens=64,
            max_running_requests=1,
        ),
        random_seed=0,
        device=torch.device("cpu"),
    )
    server_args = SimpleNamespace(
        tp_size=1,
        pp_size=1,
        dp_size=1,
        moe_dp_size=1,
        attn_cp_size=1,
        dcp_size=1,
        page_size=1,
        max_prefill_tokens=32,
        max_running_requests=2,
        max_queued_requests=server_max_queued_requests,
        context_length=128,
        chunked_prefill_size=0,
        enable_mixed_chunk=False,
        schedule_policy="fcfs",
        enable_hierarchical_cache=False,
        enable_hisparse=False,
        enable_dp_attention=False,
        enable_priority_scheduling=False,
        disable_priority_preemption=False,
        schedule_low_priority_values_first=False,
        priority_scheduling_preemption_threshold=0,
        schedule_conservativeness=1.0,
        enable_metrics=False,
        enable_metrics_for_all_schedulers=False,
    )
    monkeypatch.setattr(
        "sglang.srt.managers.scheduler_components.new_token_ratio_tracker.get_schedule",
        lambda: SimpleNamespace(
            schedule_conservativeness=server_args.schedule_conservativeness
        ),
    )

    scheduler = OmniScheduler(
        tp_worker=tp_worker,
        tree_cache=None,
        req_to_token_pool=None,
        token_to_kv_pool_allocator=None,
        server_args=server_args,
        model_config=SimpleNamespace(),
        **kwargs,
    )

    if return_runtime_context:
        return scheduler, runtime_context
    return scheduler


def test_omni_scheduler_initializes_upstream_queue_limit(monkeypatch) -> None:
    """Upstream requeue helpers read max_queued_requests on OmniScheduler."""
    scheduler, runtime_context = _construct_omni_scheduler(
        monkeypatch, return_runtime_context=True
    )

    assert scheduler._pending_chunked_abort_req is None
    assert scheduler.new_token_ratio_tracker is not None
    assert scheduler.dp_attn_adapter is not None
    assert scheduler.pool_stats_observer is not None
    assert scheduler.load_inquirer is not None
    assert scheduler.min_free_slots_delayer is None
    assert scheduler.max_queued_requests == 7
    assert scheduler.max_running_requests == 1
    assert scheduler.max_req_len == 63
    assert runtime_context.parallel.pp_max_micro_batch_size == 1
    assert runtime_context.override_calls == [
        (
            "sglang_omni.scheduler.pp_max_micro_batch_size_default",
            {"pp_max_micro_batch_size": 1},
        )
    ]
    assert scheduler._abort_on_queued_limit(object()) is False


def test_request_build_pending_limit_does_not_cap_unconfigured_backlog(
    monkeypatch,
) -> None:
    scheduler = _construct_omni_scheduler(
        monkeypatch,
        server_max_queued_requests=None,
        request_build_max_workers=2,
        request_build_max_pending=16,
    )
    payloads = [_new_stage_payload(f"req-{index}") for index in range(40)]

    try:
        selected, rejected = scheduler._stage_request_build_payloads(payloads)
    finally:
        scheduler._request_build_executor.shutdown()

    assert scheduler._request_build_backlog_limit is None
    assert len(selected) == 16
    assert len(scheduler._backlogged_request_build_payloads) == 24
    assert rejected == []


def test_request_build_backlog_honors_configured_queue_limit(monkeypatch) -> None:
    """Queued occupancy includes pending builds, so a full limit rejects extras."""
    scheduler = _construct_omni_scheduler(
        monkeypatch,
        server_max_queued_requests=16,
        request_build_max_workers=2,
        request_build_max_pending=16,
    )
    payloads = [_new_stage_payload(f"req-{index}") for index in range(40)]

    try:
        selected, rejected = scheduler._stage_request_build_payloads(payloads)
    finally:
        scheduler._request_build_executor.shutdown()

    assert scheduler._request_build_backlog_limit == 16
    assert len(selected) == 16
    assert len(scheduler._backlogged_request_build_payloads) == 0
    assert len(rejected) == 24


@pytest.mark.parametrize(
    ("enable_overlap", "enable_async_decode", "bind_late"),
    [
        (False, True, False),
        (True, False, False),
        (False, True, True),
    ],
)
def test_omni_scheduler_binds_one_execution_bridge_to_any_runner(
    monkeypatch,
    enable_overlap,
    enable_async_decode,
    bind_late,
) -> None:
    """Initial and late runners receive the same execution bridge contract."""
    monkeypatch.setattr(
        OmniScheduler,
        "_init_parallel_state",
        lambda self, _tp_worker: setattr(self, "ps", SimpleNamespace(pp_size=1)),
    )
    monkeypatch.setattr(
        OmniScheduler,
        "init_metrics_collector",
        lambda self, *_args, **_kwargs: None,
        raising=False,
    )
    monkeypatch.setattr(
        OmniScheduler,
        "init_metrics_reporter",
        lambda self, *_args, **_kwargs: setattr(
            self,
            "metrics_reporter",
            SimpleNamespace(
                reset_metrics=lambda: None,
                is_stats_logging_rank=False,
            ),
        ),
        raising=False,
    )
    bridge_parallel = SimpleNamespace(pp_max_micro_batch_size=None, attn_dcp_size=1)

    def _override(_source, **fields) -> None:
        for name, value in fields.items():
            setattr(bridge_parallel, name, value)

    monkeypatch.setattr(
        "sglang.srt.runtime_context.get_parallel",
        lambda: bridge_parallel,
    )
    monkeypatch.setattr(
        "sglang.srt.runtime_context.get_context",
        lambda: SimpleNamespace(override=_override),
    )

    observed = []

    class _ExecutionBridge:
        def __init__(self, **kwargs):
            del kwargs
            self.future_map = object()

    monkeypatch.setattr(
        "sglang_omni.model_runner.sglang_execution.SGLangExecutionBridge",
        _ExecutionBridge,
    )
    model_runner = SimpleNamespace(
        bind_execution_bridge=lambda bridge: observed.append(bridge)
    )
    tp_worker = SimpleNamespace(
        gpu_id=0,
        tp_rank=0,
        model_runner=SimpleNamespace(
            max_total_num_tokens=128,
            effective_max_total_num_tokens=64,
            max_running_requests=1,
        ),
        random_seed=0,
        device=torch.device("cpu"),
    )
    server_args = SimpleNamespace(
        tp_size=1,
        pp_size=1,
        dp_size=1,
        moe_dp_size=1,
        attn_cp_size=1,
        dcp_size=1,
        page_size=1,
        max_prefill_tokens=32,
        max_running_requests=2,
        max_queued_requests=7,
        context_length=128,
        chunked_prefill_size=0,
        enable_mixed_chunk=False,
        schedule_policy="fcfs",
        enable_hierarchical_cache=False,
        enable_hisparse=False,
        enable_dp_attention=False,
        enable_priority_scheduling=False,
        disable_priority_preemption=False,
        schedule_low_priority_values_first=False,
        priority_scheduling_preemption_threshold=0,
        schedule_conservativeness=1.0,
        enable_metrics=False,
        enable_metrics_for_all_schedulers=False,
    )
    monkeypatch.setattr(
        "sglang.srt.managers.scheduler_components.new_token_ratio_tracker.get_schedule",
        lambda: SimpleNamespace(
            schedule_conservativeness=server_args.schedule_conservativeness
        ),
    )

    scheduler = OmniScheduler(
        tp_worker=tp_worker,
        tree_cache=None,
        req_to_token_pool=None,
        token_to_kv_pool_allocator=None,
        server_args=server_args,
        model_config=SimpleNamespace(),
        model_runner=None if bind_late else model_runner,
        enable_overlap=enable_overlap,
        enable_async_decode=enable_async_decode,
    )
    if bind_late:
        scheduler.bind_model_runner(model_runner)

    assert observed == [scheduler._execution_bridge]
    assert model_runner._async_enabled is enable_async_decode


def test_omni_scheduler_refuses_overlap_with_async_decode(monkeypatch) -> None:
    """The async loop reuses the overlap batch-result contract; enabling both
    would leak KV for finished requests, so construction must refuse."""
    monkeypatch.setattr(
        OmniScheduler,
        "_init_parallel_state",
        lambda self, _tp_worker: setattr(self, "ps", SimpleNamespace(pp_size=1)),
    )
    tp_worker = SimpleNamespace(
        gpu_id=0,
        tp_rank=0,
        model_runner=SimpleNamespace(
            max_total_num_tokens=128,
            effective_max_total_num_tokens=64,
            max_running_requests=1,
        ),
        random_seed=0,
        device=torch.device("cpu"),
    )
    server_args = SimpleNamespace(
        tp_size=1,
        pp_size=1,
        dp_size=1,
        moe_dp_size=1,
        attn_cp_size=1,
        dcp_size=1,
        page_size=1,
        max_prefill_tokens=32,
        max_running_requests=2,
        max_queued_requests=7,
        context_length=128,
        chunked_prefill_size=0,
        enable_mixed_chunk=False,
        schedule_policy="fcfs",
        enable_hierarchical_cache=False,
        enable_hisparse=False,
        enable_dp_attention=False,
        enable_priority_scheduling=False,
        disable_priority_preemption=False,
        schedule_low_priority_values_first=False,
        priority_scheduling_preemption_threshold=0,
        schedule_conservativeness=1.0,
        enable_metrics=False,
        enable_metrics_for_all_schedulers=False,
    )

    with pytest.raises(ValueError, match="mutually exclusive"):
        OmniScheduler(
            tp_worker=tp_worker,
            tree_cache=None,
            req_to_token_pool=None,
            token_to_kv_pool_allocator=None,
            server_args=server_args,
            model_config=SimpleNamespace(),
            enable_overlap=True,
            enable_async_decode=True,
        )


def test_omni_scheduler_normalizes_prefill_coalesce_args(monkeypatch) -> None:
    """Defaults keep the gate off; the wait is stored in seconds."""
    scheduler = _construct_omni_scheduler(monkeypatch)
    assert scheduler.prefill_coalesce_requests == 0
    assert scheduler.prefill_coalesce_wait_s == pytest.approx(0.06)

    enabled = _construct_omni_scheduler(
        monkeypatch, prefill_coalesce_requests=32.0, prefill_coalesce_wait_ms=300
    )
    assert enabled.prefill_coalesce_requests == 32
    assert enabled.prefill_coalesce_wait_s == pytest.approx(0.3)


def test_omni_scheduler_trusts_validated_coalesce_values(monkeypatch) -> None:
    """Range and type are configuration rules (FactoryArgs and the lossless
    conversion in ConfigPath.coerce); the scheduler trusts its callers and
    keeps only its own TP-interaction rule."""
    scheduler = _construct_omni_scheduler(
        monkeypatch, prefill_coalesce_requests=0, prefill_coalesce_wait_ms=1.0
    )
    assert scheduler.prefill_coalesce_requests == 0


def test_stage_output_cache_eviction_uses_lru_order() -> None:
    cache = StageOutputCache(max_size=2)

    cache.put("a", torch.tensor([1]))
    cache.put("b", torch.tensor([2]))
    assert torch.equal(cache.get("a"), torch.tensor([1]))

    cache.put("c", torch.tensor([3]))

    assert cache.get("b") is None
    assert torch.equal(cache.get("a"), torch.tensor([1]))
    assert torch.equal(cache.get("c"), torch.tensor([3]))


def test_stage_output_cache_tracks_bytes_and_detaches() -> None:
    cache = StageOutputCache(max_bytes=8, cache_device="cpu")

    cache.put("fit", {"x": torch.ones(2, dtype=torch.float32, requires_grad=True)})
    cached = cache.get("fit")

    assert cache.current_bytes == 8
    assert cached["x"].device.type == "cpu"
    assert cached["x"].requires_grad is False

    cache.put("too-large", torch.ones(3, dtype=torch.float32))

    assert cache.get("too-large") is None
    assert cache.current_bytes == 8


def test_omni_scheduler_stop_runs_shutdown_callback_once() -> None:
    scheduler = object.__new__(OmniScheduler)
    shutdowns: list[None] = []
    scheduler._running = True
    scheduler._request_admission_lock = threading.RLock()
    scheduler._pending_request_admissions = {}
    scheduler._shutdown_lock = threading.Lock()
    scheduler._shutdown_callback = lambda: shutdowns.append(None)

    scheduler.stop()
    scheduler.stop()

    assert scheduler._running is False
    assert shutdowns == [None]


@pytest.mark.parametrize(
    ("loop_error", "expected_status"),
    [
        (None, "aborted"),
        (RuntimeError("scheduler loop failed"), "error"),
    ],
)
def test_omni_scheduler_start_closes_active_model_paths(
    monkeypatch,
    loop_error,
    expected_status,
) -> None:
    model_path_ends: list[tuple[str, str]] = []
    monkeypatch.setattr(
        omni_scheduler_module,
        "_emit_model_path_end",
        lambda rid, *, status: model_path_ends.append((rid, status)),
    )
    scheduler = object.__new__(OmniScheduler)
    scheduler.enable_async_decode = False
    scheduler.enable_overlap = False
    scheduler._prefill_start_done = {"req-1", "req-2"}
    scheduler._prefill_end_done = set()
    scheduler._request_build_executor = None
    scheduler._request_admission_lock = threading.RLock()
    scheduler._pending_request_admissions = {}
    scheduler._shutdown_lock = threading.Lock()
    scheduler._shutdown_callback = None

    def run_loop() -> None:
        if loop_error is not None:
            raise loop_error
        scheduler._running = False

    scheduler._event_loop_normal = run_loop

    if loop_error is None:
        scheduler.start()
    else:
        with pytest.raises(RuntimeError, match="scheduler loop failed"):
            scheduler.start()

    assert set(model_path_ends) == {
        ("req-1", expected_status),
        ("req-2", expected_status),
    }
    assert scheduler._prefill_start_done == set()


def test_omni_scheduler_request_builder_errors_do_not_stop_loop() -> None:
    """Covers per-request build errors before an SGLang Req exists."""
    scheduler = object.__new__(OmniScheduler)
    scheduler.outbox = Queue()
    scheduler.waiting_queue = []
    scheduler._pending_stream_ingress = {}
    scheduler._deferred_request_payloads = {}
    scheduler._dirty_deferred_request_ids = set()
    scheduler._aborted_request_ids = set()
    scheduler._aborted_request_id_order = deque()
    scheduler.running_batch = SimpleNamespace(reqs=[], batch_is_full=False)
    scheduler.cur_batch = None
    scheduler.last_batch = None
    scheduler._abort_callback = None
    scheduler._first_emit_done = set()
    scheduler._prefill_start_done = set()
    scheduler._prefill_end_done = set()
    scheduler.inbox = Queue()
    scheduler.tree_cache = None
    _init_sync_request_build_state(scheduler)

    def request_builder(payload: SimpleNamespace) -> None:
        raise ValueError(payload.request_id)

    scheduler._request_builder = request_builder

    scheduler.is_entry_rank = True
    scheduler.process_input_requests([_new_stage_payload("req-err")])

    output = scheduler.outbox.get_nowait()
    assert output.request_id == "req-err"
    assert output.type == "error"
    assert isinstance(output.data, ValueError)
    assert scheduler.waiting_queue == []


def test_omni_scheduler_follower_request_builder_errors_do_not_emit() -> None:
    """TP followers clean local state but do not emit user-visible errors."""
    scheduler = object.__new__(OmniScheduler)
    scheduler.outbox = Queue()
    scheduler.waiting_queue = []
    scheduler._pending_stream_ingress = {"req-err": _ingress(done=True)}
    scheduler._deferred_request_payloads = {"req-err": object()}
    scheduler._dirty_deferred_request_ids = set()
    scheduler._aborted_request_ids = set()
    scheduler._aborted_request_id_order = deque()
    scheduler.is_entry_rank = False
    scheduler.running_batch = SimpleNamespace(reqs=[], batch_is_full=False)
    scheduler.cur_batch = None
    scheduler.last_batch = None
    scheduler._abort_callback = None
    scheduler._first_emit_done = set()
    scheduler._prefill_start_done = set()
    scheduler._prefill_end_done = set()
    scheduler.inbox = Queue()
    scheduler.tree_cache = None
    _init_sync_request_build_state(scheduler)

    def request_builder(payload: SimpleNamespace) -> None:
        raise ValueError(payload.request_id)

    scheduler._request_builder = request_builder

    scheduler.process_input_requests([_new_stage_payload("req-err")])

    assert scheduler.outbox.empty()
    assert scheduler.waiting_queue == []
    assert scheduler._pending_stream_ingress == {}
    assert scheduler._deferred_request_payloads == {}


def test_omni_scheduler_prepares_custom_request_token_budget() -> None:
    """Preserves upstream max_new_tokens clamping for custom request builders."""
    scheduler = object.__new__(OmniScheduler)
    scheduler.outbox = Queue()
    scheduler.waiting_queue = []
    scheduler._pending_stream_ingress = {}
    scheduler._deferred_request_payloads = {}
    scheduler._dirty_deferred_request_ids = set()
    scheduler._aborted_request_ids = set()
    scheduler._aborted_request_id_order = deque()
    scheduler.max_req_len = 6
    scheduler.max_req_input_len = 5
    scheduler.max_new_tokens_limit = None
    scheduler.page_size = 1
    scheduler.max_total_num_tokens = 128
    _init_sync_request_build_state(scheduler)

    sampling_params = SimpleNamespace(max_new_tokens=10, min_new_tokens=0)
    req = SimpleNamespace(
        rid="req-ok",
        origin_input_ids=[1, 2, 3],
        origin_input_ids_unpadded=[1, 2, 3],
        sampling_params=sampling_params,
        output_ids=[],
        priority=None,
    )
    req_data = SimpleNamespace(req=req, max_new_tokens=10, enforce_request_limits=True)
    scheduler._request_builder = lambda payload: req_data

    scheduler.process_input_requests([_new_stage_payload("req-ok")])

    assert scheduler.waiting_queue == [req]
    assert req._omni_data is req_data
    assert req.sampling_params.max_new_tokens == 2
    assert req_data.max_new_tokens == 2
    assert scheduler.outbox.empty()


def test_omni_scheduler_rejects_custom_request_over_context() -> None:
    """Covers context-length validation for custom request builders."""
    scheduler = object.__new__(OmniScheduler)
    scheduler.outbox = Queue()
    scheduler.waiting_queue = []
    scheduler._pending_stream_ingress = {}
    scheduler._deferred_request_payloads = {}
    scheduler._dirty_deferred_request_ids = set()
    scheduler._aborted_request_ids = set()
    scheduler._aborted_request_id_order = deque()
    scheduler.max_req_len = 6
    scheduler.max_req_input_len = 5
    scheduler.max_new_tokens_limit = None
    scheduler.page_size = 1
    scheduler.max_total_num_tokens = 128
    scheduler.running_batch = SimpleNamespace(reqs=[], batch_is_full=False)
    scheduler.cur_batch = None
    scheduler.last_batch = None
    scheduler._abort_callback = None
    scheduler._first_emit_done = set()
    scheduler._prefill_start_done = set()
    scheduler._prefill_end_done = set()
    scheduler.inbox = Queue()
    scheduler.tree_cache = None
    _init_sync_request_build_state(scheduler)

    req = SimpleNamespace(
        rid="req-long",
        origin_input_ids=[1, 2, 3, 4, 5],
        origin_input_ids_unpadded=[1, 2, 3, 4, 5],
        sampling_params=SimpleNamespace(max_new_tokens=10, min_new_tokens=0),
        output_ids=[],
    )
    request_data = SimpleNamespace(
        req=req,
        enforce_request_limits=True,
        max_new_tokens=10,
    )
    scheduler._request_builder = lambda payload: request_data

    scheduler.is_entry_rank = True
    scheduler.process_input_requests([_new_stage_payload("req-long")])

    output = scheduler.outbox.get_nowait()
    assert output.request_id == "req-long"
    assert output.type == "error"
    assert isinstance(output.data, ValueError)
    assert "Input length (5 tokens) exceeds" in str(output.data)
    assert scheduler.waiting_queue == []
    assert not hasattr(req, "_omni_data")
    assert request_data.req is req


def test_omni_scheduler_follower_rejections_do_not_emit_errors() -> None:
    """Request-limit and KV-capacity rejections are entry-rank emissions only."""
    scheduler = object.__new__(OmniScheduler)
    scheduler.outbox = Queue()
    scheduler.waiting_queue = []
    scheduler._pending_stream_ingress = {}
    scheduler._deferred_request_payloads = {}
    scheduler._dirty_deferred_request_ids = set()
    scheduler._aborted_request_ids = set()
    scheduler._aborted_request_id_order = deque()
    scheduler.is_entry_rank = False
    scheduler.running_batch = SimpleNamespace(reqs=[], batch_is_full=False)
    scheduler.cur_batch = None
    scheduler.last_batch = None
    scheduler._abort_callback = None
    scheduler._first_emit_done = set()
    scheduler._prefill_start_done = set()
    scheduler._prefill_end_done = set()
    scheduler.inbox = Queue()
    scheduler.tree_cache = None
    scheduler.max_req_len = 6
    scheduler.max_req_input_len = 5
    scheduler.max_new_tokens_limit = None
    scheduler.page_size = 1
    scheduler.max_total_num_tokens = 128
    scheduler.server_args = SimpleNamespace(mem_fraction_static=0.85)
    _init_sync_request_build_state(scheduler)

    over_context_req = SimpleNamespace(
        rid="req-long",
        origin_input_ids=[1, 2, 3, 4, 5],
        origin_input_ids_unpadded=[1, 2, 3, 4, 5],
        sampling_params=SimpleNamespace(max_new_tokens=10, min_new_tokens=0),
        output_ids=[],
    )
    scheduler._request_builder = lambda payload: SimpleNamespace(
        req=over_context_req,
        enforce_request_limits=True,
        max_new_tokens=10,
    )

    scheduler.process_input_requests([_new_stage_payload("req-long")])

    assert scheduler.outbox.empty()
    assert scheduler.waiting_queue == []

    over_kv_req = SimpleNamespace(
        rid="req-kv",
        origin_input_ids=[1, 2, 3],
        origin_input_ids_unpadded=[1, 2, 3],
        sampling_params=SimpleNamespace(max_new_tokens=4, min_new_tokens=0),
        output_ids=[],
    )
    scheduler._request_builder = lambda payload: SimpleNamespace(
        req=over_kv_req,
        enforce_request_limits=False,
        max_new_tokens=4,
    )

    scheduler.process_input_requests([_new_stage_payload("req-kv")])

    assert scheduler.outbox.empty()
    assert scheduler.waiting_queue == []


def test_omni_scheduler_leaves_request_budget_unchanged_without_opt_in() -> None:
    """Keeps existing OmniScheduler users on their original request semantics."""
    scheduler = object.__new__(OmniScheduler)
    scheduler.outbox = Queue()
    scheduler.waiting_queue = []
    scheduler._pending_stream_ingress = {}
    scheduler._deferred_request_payloads = {}
    scheduler._dirty_deferred_request_ids = set()
    scheduler._aborted_request_ids = set()
    scheduler._aborted_request_id_order = deque()
    scheduler.max_req_len = 6
    scheduler.max_req_input_len = 5
    scheduler.max_new_tokens_limit = None
    scheduler.page_size = 1
    scheduler.max_total_num_tokens = 128
    _init_sync_request_build_state(scheduler)

    sampling_params = SimpleNamespace(max_new_tokens=3, min_new_tokens=0)
    req = SimpleNamespace(
        rid="req-original",
        origin_input_ids=[1, 2, 3],
        origin_input_ids_unpadded=[1, 2, 3],
        sampling_params=sampling_params,
        output_ids=[],
        priority=None,
    )
    req_data = SimpleNamespace(
        req=req,
        max_new_tokens=3,
        enforce_request_limits=False,
    )
    scheduler._request_builder = lambda payload: req_data

    scheduler.process_input_requests([_new_stage_payload("req-original")])

    assert scheduler.waiting_queue == [req]
    assert req.sampling_params.max_new_tokens == 3
    assert req_data.max_new_tokens == 3
    assert scheduler.outbox.empty()


def test_omni_scheduler_result_adapter_failure_emits_error_without_raise(
    monkeypatch,
) -> None:
    """Finished-request adapter failures remain request-local."""
    scheduler = object.__new__(OmniScheduler)
    _init_terminal_output_state(scheduler)
    scheduler.outbox = Queue()
    scheduler.is_entry_rank = True
    scheduler._aborted_request_ids = set()
    scheduler._first_emit_done = {"req-adapter"}
    scheduler._prefill_start_done = {"req-adapter"}
    scheduler._prefill_end_done = set()
    model_path_ends: list[tuple[str, str]] = []
    monkeypatch.setattr(
        omni_scheduler_module,
        "_emit_model_path_end",
        lambda rid, *, status: model_path_ends.append((rid, status)),
    )

    def fail_adapter(_data):
        raise RuntimeError("adapter failed")

    scheduler._result_adapter = fail_adapter
    request_data = SimpleNamespace(
        prefill_input_embeds=torch.ones(1),
        decode_input_embeds=[torch.ones(1)],
    )
    req = SimpleNamespace(
        rid="req-adapter",
        _omni_data=request_data,
        _omni_terminal_claimed=False,
        output_ids=[1, 2],
        finished=lambda: True,
        finished_reason=None,
    )
    request_data.req = req

    scheduler.stream_output([req])

    output = scheduler.outbox.get_nowait()
    assert output.request_id == "req-adapter"
    assert output.type == "error"
    assert isinstance(output.data, RuntimeError)
    assert scheduler._first_emit_done == set()
    assert scheduler._prefill_start_done == set()
    assert model_path_ends == [("req-adapter", "error")]
    assert request_data.prefill_input_embeds is None
    assert request_data.decode_input_embeds is None
    assert req._omni_data is None
    assert request_data.req is req


def test_omni_scheduler_running_abort_does_not_leak_prefill_dedup_state(
    monkeypatch,
) -> None:
    """A running abort that never reaches stream_output must not leak.

    The rid used to stay in the set that also dedups prefill_start, so the set
    grew without bound and a later prefill_start for the same id was silently
    swallowed.
    """
    ends: list[tuple[str, str]] = []
    monkeypatch.setattr(
        omni_scheduler_module,
        "_emit_model_path_end",
        lambda rid, *, status: ends.append((rid, status)),
    )
    scheduler = object.__new__(OmniScheduler)
    scheduler._mark_running_request_aborted = lambda _rid: True
    scheduler._request_admission_lock = threading.Lock()
    scheduler._aborted_request_ids = set()
    scheduler._aborted_request_id_order = collections.deque()
    scheduler._pending_request_builds = {}
    scheduler._pending_request_admissions = {}
    scheduler._backlogged_request_build_payloads = []
    scheduler.waiting_queue = []
    scheduler._abort_callback = None
    scheduler._pending_stream_chunks = {}
    scheduler._pending_stream_done = set()
    scheduler._pending_stream_ingress = {}
    scheduler._deferred_request_payloads = {}
    scheduler._dirty_deferred_request_ids = set()
    scheduler._first_emit_done = {"req-1"}
    scheduler._prefill_start_done = {"req-1"}
    scheduler._prefill_end_done = set()
    scheduler._drain_inbox_for_request = lambda _rid: None

    scheduler.abort("req-1")

    assert ends == [("req-1", "aborted")]
    assert scheduler._prefill_start_done == set()
    assert scheduler._prefill_start_done == set()
