# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import threading
from queue import Queue
from types import SimpleNamespace

import pytest
import torch

from sglang_omni.scheduling import omni_scheduler as omni_scheduler_module
from sglang_omni.scheduling.messages import IncomingMessage
from sglang_omni.scheduling.omni_scheduler import OmniScheduler
from sglang_omni.scheduling.simple_scheduler import SimpleScheduler
from sglang_omni.scheduling.stage_cache import StageOutputCache
from sglang_omni.scheduling.threaded_simple_scheduler import ThreadedSimpleScheduler
from tests.unit_test.pipeline.helpers import run_scheduler


def _init_sync_request_build_state(scheduler: OmniScheduler) -> None:
    scheduler._request_admission_lock = threading.RLock()
    scheduler._request_build_executor = None
    scheduler.request_build_max_pending = 0
    scheduler._pending_request_builds = {}
    scheduler._backlogged_request_build_payloads = []
    scheduler._request_build_max_pending_observed = 0


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
    scheduler.waiting_queue = []
    scheduler._pending_stream_chunks = {}
    scheduler._pending_stream_done = set()
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
    assert scheduler._pending_stream_chunks["req-unknown"] == ["chunk-x"]
    assert scheduler._take_deferred_request_payloads() == []

    OmniScheduler._on_stream_done(scheduler, "req-deferred")
    assert scheduler._dirty_deferred_request_ids == {"req-deferred"}
    assert scheduler._take_deferred_request_payloads() == [payload]
    assert scheduler._dirty_deferred_request_ids == set()


def test_omni_scheduler_run_batch_failure_emits_error_and_aborts(monkeypatch) -> None:
    """Forward failures are owned by the scheduler, not model executors."""
    release_calls: list[tuple[str, object]] = []
    tree_cache = object()
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
    scheduler._pending_stream_chunks = {"req-1": ["stale"]}
    scheduler._pending_stream_done = {"req-2"}
    scheduler._deferred_request_payloads = {"req-1": object()}
    scheduler._dirty_deferred_request_ids = {"req-1"}
    scheduler._abort_callback = None
    scheduler.tree_cache = tree_cache
    scheduler.waiting_queue = []
    scheduler.last_batch = None
    scheduler._first_emit_done = set()
    scheduler._prefill_start_done = set()

    batch = SimpleNamespace(
        reqs=[
            SimpleNamespace(
                rid="req-1",
                _omni_data=SimpleNamespace(),
                req_pool_idx=1,
                mamba_pool_idx=None,
            ),
            SimpleNamespace(
                rid="req-2",
                _omni_data=SimpleNamespace(),
                req_pool_idx=2,
                mamba_pool_idx=None,
            ),
        ],
        batch_is_full=True,
        is_prefill_only=True,
        is_extend_in_batch=False,
    )
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
    assert release_calls == [("req-1", tree_cache), ("req-2", tree_cache)]
    assert scheduler._pending_stream_chunks == {}
    assert scheduler._pending_stream_done == set()
    assert scheduler._deferred_request_payloads == {}
    assert scheduler._dirty_deferred_request_ids == set()


def test_omni_scheduler_custom_runner_updates_next_input_ids() -> None:
    """Custom AR runners must preserve SGLang's decode handoff contract."""

    next_token_ids = torch.tensor([11, 12], dtype=torch.int32)

    class FakeModelRunner:
        def execute(self, sched_output):
            sched_output.batch_data.output_ids = next_token_ids
            return SimpleNamespace(outputs={}, can_run_cuda_graph=False)

    scheduler = object.__new__(OmniScheduler)
    scheduler._model_runner = FakeModelRunner()
    scheduler._stream_output_builder = None
    scheduler._prefill_start_done = set()

    batch = SimpleNamespace(
        reqs=[
            SimpleNamespace(rid="req-1", _omni_data=SimpleNamespace()),
            SimpleNamespace(rid="req-2", _omni_data=SimpleNamespace()),
        ],
        output_ids=None,
        is_prefill_only=False,
        is_extend_in_batch=False,
    )

    result = scheduler._run_batch(batch)

    assert result.next_token_ids is next_token_ids
    assert batch.input_ids.dtype == torch.int64
    assert batch.input_ids.tolist() == [11, 12]


def test_omni_scheduler_custom_runner_advances_forward_ct() -> None:
    """OmniScheduler overrides upstream run_batch, so it must count forwards
    itself; otherwise forward_ct stays 0 and the SGLANG_TEST_RETRACT_INTERVAL
    gate (``forward_ct % INTERVAL == 0``) fires every step. One forward per
    sync run_batch and per async launch; resolve does no forward.
    """

    class FakeModelRunner:
        def execute(self, sched_output):
            sched_output.batch_data.output_ids = torch.tensor([1], dtype=torch.int32)
            return SimpleNamespace(outputs={}, can_run_cuda_graph=False)

        def execute_launch(self, sched_output):
            return SimpleNamespace()

    scheduler = object.__new__(OmniScheduler)
    scheduler._model_runner = FakeModelRunner()
    scheduler._stream_output_builder = None
    scheduler._prefill_start_done = set()
    scheduler.forward_ct = 0

    def _batch():
        return SimpleNamespace(
            reqs=[SimpleNamespace(rid="r", _omni_data=SimpleNamespace())],
            output_ids=None,
            is_prefill_only=False,
            is_extend_in_batch=False,
        )

    scheduler._run_batch(_batch())
    assert scheduler.forward_ct == 1, "sync run_batch must advance forward_ct"

    scheduler._run_batch_launch(_batch())
    assert scheduler.forward_ct == 2, "async launch must advance forward_ct"


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
    forwarded/finalized again and re-frees its already-freed KV.
    """
    captured: dict = {}
    freed: list[int] = []

    class FakeBatch:
        def __init__(self, reqs):
            self.reqs = reqs
            self.out_cache_loc = torch.arange(100, 100 + len(reqs))

        def filter_batch(self, keep_indices=None):
            captured["keep_indices"] = keep_indices
            self.reqs = [self.reqs[i] for i in keep_indices]
            self.out_cache_loc = None

    scheduler = object.__new__(OmniScheduler)
    scheduler.page_size = 1
    scheduler.server_args = SimpleNamespace(disable_radix_cache=False)
    scheduler.token_to_kv_pool_allocator = SimpleNamespace(
        free=lambda t: freed.extend(t.tolist())
    )
    keep = SimpleNamespace(rid="keep", finished=lambda: False, is_retracted=False)
    retr = SimpleNamespace(rid="retr", finished=lambda: False, is_retracted=True)

    # retracted (not finished) must be dropped from the stale batch
    out = scheduler._drop_stale_overrun(FakeBatch([keep, retr]))
    assert captured["keep_indices"] == [0]
    assert [r.rid for r in out.reqs] == ["keep"]
    assert out.out_cache_loc.tolist() == [100]
    assert freed == [101]

    # all dropped -> None so run_batch is skipped
    freed.clear()
    fin = SimpleNamespace(rid="fin", finished=lambda: True, is_retracted=False)
    assert scheduler._drop_stale_overrun(FakeBatch([retr, fin])) is None
    assert freed == [100, 101]

    # nothing stale -> batch returned unchanged, filter_batch never called
    captured.clear()
    freed.clear()
    clean = FakeBatch([keep])
    assert scheduler._drop_stale_overrun(clean) is clean
    assert "keep_indices" not in captured
    assert freed == []


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
    scheduler._pending_stream_chunks = {}
    scheduler._pending_stream_done = set()
    scheduler._deferred_request_payloads = {}
    scheduler._dirty_deferred_request_ids = set()
    scheduler._first_emit_done = set()
    scheduler._prefill_start_done = set()
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
    scheduler = object.__new__(OmniScheduler)
    scheduler._abort_callback = cleaned.append
    scheduler._aborted_request_ids = set()
    scheduler._pending_stream_chunks = {"req-run": ["stale"]}
    scheduler._pending_stream_done = {"req-run"}
    scheduler._deferred_request_payloads = {"req-run": object()}
    scheduler._dirty_deferred_request_ids = {"req-run"}
    scheduler._first_emit_done = {"req-run"}
    scheduler._prefill_start_done = {"req-run"}
    scheduler.inbox = Queue()
    scheduler.waiting_queue = []

    req = SimpleNamespace(
        rid="req-run",
        to_finish=None,
        req_pool_idx=1,
        finished=lambda: False,
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
    assert scheduler._pending_stream_chunks == {}
    assert scheduler._pending_stream_done == set()
    assert scheduler._deferred_request_payloads == {}
    assert scheduler._dirty_deferred_request_ids == set()
    assert scheduler._first_emit_done == set()
    assert scheduler._prefill_start_done == set()


def test_omni_scheduler_abort_cleans_queued_request_immediately() -> None:
    """Queued aborts have no KV allocation, so callback cleanup can run now."""
    cleaned: list[str] = []
    scheduler = object.__new__(OmniScheduler)
    scheduler._abort_callback = cleaned.append
    scheduler._aborted_request_ids = set()
    scheduler._pending_stream_chunks = {}
    scheduler._pending_stream_done = set()
    scheduler._deferred_request_payloads = {}
    scheduler._dirty_deferred_request_ids = set()
    scheduler._first_emit_done = set()
    scheduler._prefill_start_done = set()
    scheduler.inbox = Queue()

    req = SimpleNamespace(rid="req-wait")
    scheduler.waiting_queue = [req]
    scheduler.running_batch = SimpleNamespace(reqs=[], batch_is_full=False)
    scheduler.cur_batch = None
    scheduler.last_batch = None
    _init_sync_request_build_state(scheduler)

    scheduler.abort("req-wait")

    assert scheduler.waiting_queue == []
    assert cleaned == ["req-wait"]


def test_omni_scheduler_distinguishes_queue_enter_from_prefill_start(
    monkeypatch,
) -> None:
    """Queueing a built request must not report actual prefill execution."""
    events: list[dict] = []
    monkeypatch.setattr(
        "sglang_omni.scheduling.omni_scheduler._emit_event",
        lambda **kwargs: events.append(kwargs),
    )
    scheduler = object.__new__(OmniScheduler)
    scheduler.outbox = Queue()
    scheduler.waiting_queue = []
    scheduler._pending_stream_chunks = {}
    scheduler._pending_stream_done = set()
    scheduler._deferred_request_payloads = {}
    scheduler._dirty_deferred_request_ids = set()
    scheduler._aborted_request_ids = set()
    scheduler._prefill_start_done = set()
    scheduler.max_req_len = 16
    scheduler.max_req_input_len = 16
    _init_sync_request_build_state(scheduler)

    req = SimpleNamespace(
        rid="req-delayed",
        origin_input_ids=[1, 2, 3],
        sampling_params=SimpleNamespace(max_new_tokens=1),
        output_ids=[],
    )
    scheduler._request_builder = lambda payload: SimpleNamespace(req=req)

    scheduler.process_input_requests([SimpleNamespace(request_id="req-delayed")])

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


def test_omni_scheduler_initializes_upstream_queue_limit(monkeypatch) -> None:
    """Upstream requeue helpers read max_queued_requests on OmniScheduler."""
    monkeypatch.setattr(
        OmniScheduler, "_init_parallel_state", lambda self, _tp_worker: None
    )
    monkeypatch.setattr(
        OmniScheduler,
        "init_metrics",
        lambda self, *_args, **_kwargs: None,
        raising=False,
    )
    monkeypatch.setattr(
        "sglang.srt.server_args.get_global_server_args",
        lambda: SimpleNamespace(pp_max_micro_batch_size=None),
    )
    tp_worker = SimpleNamespace(
        gpu_id=0,
        tp_rank=0,
        model_runner=SimpleNamespace(max_total_num_tokens=128),
        random_seed=0,
        device=torch.device("cpu"),
    )
    server_args = SimpleNamespace(
        tp_size=1,
        pp_size=1,
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
        enable_priority_scheduling=False,
        disable_priority_preemption=False,
        schedule_low_priority_values_first=False,
        priority_scheduling_preemption_threshold=0,
        schedule_conservativeness=1.0,
        enable_metrics=False,
        enable_metrics_for_all_schedulers=False,
    )

    scheduler = OmniScheduler(
        tp_worker=tp_worker,
        tree_cache=None,
        req_to_token_pool=None,
        token_to_kv_pool_allocator=None,
        server_args=server_args,
        model_config=SimpleNamespace(),
    )

    assert scheduler.max_queued_requests == 7
    assert scheduler._abort_on_queued_limit(object()) is False


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


def test_omni_scheduler_request_builder_errors_do_not_stop_loop() -> None:
    """Covers per-request build errors before an SGLang Req exists."""
    scheduler = object.__new__(OmniScheduler)
    scheduler.outbox = Queue()
    scheduler.waiting_queue = []
    scheduler._pending_stream_chunks = {}
    scheduler._pending_stream_done = set()
    scheduler._deferred_request_payloads = {}
    scheduler._dirty_deferred_request_ids = set()
    scheduler._aborted_request_ids = set()
    scheduler.running_batch = SimpleNamespace(reqs=[], batch_is_full=False)
    scheduler.cur_batch = None
    scheduler.last_batch = None
    scheduler._abort_callback = None
    scheduler._first_emit_done = set()
    scheduler._prefill_start_done = set()
    scheduler.inbox = Queue()
    scheduler.tree_cache = None
    _init_sync_request_build_state(scheduler)

    def request_builder(payload: SimpleNamespace) -> None:
        raise ValueError(payload.request_id)

    scheduler._request_builder = request_builder

    scheduler.process_input_requests([SimpleNamespace(request_id="req-err")])

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
    scheduler._pending_stream_chunks = {}
    scheduler._pending_stream_done = {"req-err"}
    scheduler._deferred_request_payloads = {"req-err": object()}
    scheduler._dirty_deferred_request_ids = set()
    scheduler._aborted_request_ids = set()
    scheduler.is_entry_rank = False
    scheduler.running_batch = SimpleNamespace(reqs=[], batch_is_full=False)
    scheduler.cur_batch = None
    scheduler.last_batch = None
    scheduler._abort_callback = None
    scheduler._first_emit_done = set()
    scheduler._prefill_start_done = set()
    scheduler.inbox = Queue()
    scheduler.tree_cache = None
    _init_sync_request_build_state(scheduler)

    def request_builder(payload: SimpleNamespace) -> None:
        raise ValueError(payload.request_id)

    scheduler._request_builder = request_builder

    scheduler.process_input_requests([SimpleNamespace(request_id="req-err")])

    assert scheduler.outbox.empty()
    assert scheduler.waiting_queue == []
    assert scheduler._pending_stream_done == set()
    assert scheduler._deferred_request_payloads == {}


def test_omni_scheduler_prepares_custom_request_token_budget() -> None:
    """Preserves upstream max_new_tokens clamping for custom request builders."""
    scheduler = object.__new__(OmniScheduler)
    scheduler.outbox = Queue()
    scheduler.waiting_queue = []
    scheduler._pending_stream_chunks = {}
    scheduler._pending_stream_done = set()
    scheduler._deferred_request_payloads = {}
    scheduler._dirty_deferred_request_ids = set()
    scheduler._aborted_request_ids = set()
    scheduler.max_req_len = 6
    scheduler.max_req_input_len = 5
    scheduler.page_size = 1
    scheduler.max_total_num_tokens = 128
    _init_sync_request_build_state(scheduler)

    sampling_params = SimpleNamespace(max_new_tokens=10)
    req = SimpleNamespace(
        rid="req-ok",
        origin_input_ids=[1, 2, 3],
        sampling_params=sampling_params,
        output_ids=[],
    )
    req_data = SimpleNamespace(req=req, max_new_tokens=10, enforce_request_limits=True)
    scheduler._request_builder = lambda payload: req_data

    scheduler.process_input_requests([SimpleNamespace(request_id="req-ok")])

    assert scheduler.waiting_queue == [req]
    assert req.sampling_params.max_new_tokens == 2
    assert req_data.max_new_tokens == 2
    assert scheduler.outbox.empty()


def test_omni_scheduler_rejects_custom_request_over_context() -> None:
    """Covers context-length validation for custom request builders."""
    scheduler = object.__new__(OmniScheduler)
    scheduler.outbox = Queue()
    scheduler.waiting_queue = []
    scheduler._pending_stream_chunks = {}
    scheduler._pending_stream_done = set()
    scheduler._deferred_request_payloads = {}
    scheduler._dirty_deferred_request_ids = set()
    scheduler._aborted_request_ids = set()
    scheduler.max_req_len = 6
    scheduler.max_req_input_len = 5
    scheduler.page_size = 1
    scheduler.max_total_num_tokens = 128
    scheduler.running_batch = SimpleNamespace(reqs=[], batch_is_full=False)
    scheduler.cur_batch = None
    scheduler.last_batch = None
    scheduler._abort_callback = None
    scheduler._first_emit_done = set()
    scheduler._prefill_start_done = set()
    scheduler.inbox = Queue()
    scheduler.tree_cache = None
    _init_sync_request_build_state(scheduler)

    req = SimpleNamespace(
        rid="req-long",
        origin_input_ids=[1, 2, 3, 4, 5],
        sampling_params=SimpleNamespace(max_new_tokens=10),
        output_ids=[],
    )
    scheduler._request_builder = lambda payload: SimpleNamespace(
        req=req,
        enforce_request_limits=True,
    )

    scheduler.process_input_requests([SimpleNamespace(request_id="req-long")])

    output = scheduler.outbox.get_nowait()
    assert output.request_id == "req-long"
    assert output.type == "error"
    assert isinstance(output.data, ValueError)
    assert "Input length (5 tokens) exceeds" in str(output.data)
    assert scheduler.waiting_queue == []


def test_omni_scheduler_follower_rejections_do_not_emit_errors() -> None:
    """Request-limit and KV-capacity rejections are entry-rank emissions only."""
    scheduler = object.__new__(OmniScheduler)
    scheduler.outbox = Queue()
    scheduler.waiting_queue = []
    scheduler._pending_stream_chunks = {}
    scheduler._pending_stream_done = set()
    scheduler._deferred_request_payloads = {}
    scheduler._dirty_deferred_request_ids = set()
    scheduler._aborted_request_ids = set()
    scheduler.is_entry_rank = False
    scheduler.running_batch = SimpleNamespace(reqs=[], batch_is_full=False)
    scheduler.cur_batch = None
    scheduler.last_batch = None
    scheduler._abort_callback = None
    scheduler._first_emit_done = set()
    scheduler._prefill_start_done = set()
    scheduler.inbox = Queue()
    scheduler.tree_cache = None
    scheduler.max_req_len = 6
    scheduler.max_req_input_len = 5
    scheduler.page_size = 1
    scheduler.max_total_num_tokens = 128
    scheduler.server_args = SimpleNamespace(mem_fraction_static=0.85)
    _init_sync_request_build_state(scheduler)

    over_context_req = SimpleNamespace(
        rid="req-long",
        origin_input_ids=[1, 2, 3, 4, 5],
        sampling_params=SimpleNamespace(max_new_tokens=10),
        output_ids=[],
    )
    scheduler._request_builder = lambda payload: SimpleNamespace(
        req=over_context_req,
        enforce_request_limits=True,
    )

    scheduler.process_input_requests([SimpleNamespace(request_id="req-long")])

    assert scheduler.outbox.empty()
    assert scheduler.waiting_queue == []

    over_kv_req = SimpleNamespace(
        rid="req-kv",
        origin_input_ids=[1, 2, 3],
        sampling_params=SimpleNamespace(max_new_tokens=4),
        output_ids=[],
    )
    scheduler._request_builder = lambda payload: SimpleNamespace(
        req=over_kv_req,
        enforce_request_limits=False,
    )

    scheduler.process_input_requests([SimpleNamespace(request_id="req-kv")])

    assert scheduler.outbox.empty()
    assert scheduler.waiting_queue == []


def test_omni_scheduler_leaves_request_budget_unchanged_without_opt_in() -> None:
    """Keeps existing OmniScheduler users on their original request semantics."""
    scheduler = object.__new__(OmniScheduler)
    scheduler.outbox = Queue()
    scheduler.waiting_queue = []
    scheduler._pending_stream_chunks = {}
    scheduler._pending_stream_done = set()
    scheduler._deferred_request_payloads = {}
    scheduler._dirty_deferred_request_ids = set()
    scheduler._aborted_request_ids = set()
    scheduler.max_req_len = 6
    scheduler.max_req_input_len = 5
    scheduler.page_size = 1
    scheduler.max_total_num_tokens = 128
    _init_sync_request_build_state(scheduler)

    sampling_params = SimpleNamespace(max_new_tokens=3)
    req = SimpleNamespace(
        rid="req-original",
        origin_input_ids=[1, 2, 3],
        sampling_params=sampling_params,
        output_ids=[],
    )
    req_data = SimpleNamespace(req=req, max_new_tokens=3)
    scheduler._request_builder = lambda payload: req_data

    scheduler.process_input_requests([SimpleNamespace(request_id="req-original")])

    assert scheduler.waiting_queue == [req]
    assert req.sampling_params.max_new_tokens == 3
    assert req_data.max_new_tokens == 3
    assert scheduler.outbox.empty()


def test_omni_scheduler_result_adapter_failure_emits_error_without_raise() -> None:
    """Finished-request adapter failures remain request-local."""
    scheduler = object.__new__(OmniScheduler)
    scheduler.outbox = Queue()
    scheduler.is_entry_rank = True
    scheduler.server_args = SimpleNamespace(weight_version=None)
    scheduler._first_emit_done = {"req-adapter"}
    scheduler._prefill_start_done = {"req-adapter"}

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
        output_ids=[1, 2],
        finished=lambda: True,
        finished_reason=None,
    )

    scheduler.stream_output([req])

    output = scheduler.outbox.get_nowait()
    assert output.request_id == "req-adapter"
    assert output.type == "error"
    assert isinstance(output.data, RuntimeError)
    assert scheduler._first_emit_done == set()
    assert scheduler._prefill_start_done == set()
    assert request_data.prefill_input_embeds is None
    assert request_data.decode_input_embeds is None
