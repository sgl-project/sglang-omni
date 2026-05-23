# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import threading
from queue import Queue
from types import SimpleNamespace

import torch

from sglang_omni.scheduling.messages import IncomingMessage
from sglang_omni.scheduling.omni_scheduler import OmniScheduler
from sglang_omni.scheduling.simple_scheduler import SimpleScheduler
from sglang_omni.scheduling.stage_cache import StageOutputCache
from sglang_omni.scheduling.threaded_simple_scheduler import ThreadedSimpleScheduler
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
        enable_priority_scheduling=False,
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
    scheduler._aborted_request_ids = set()

    def request_builder(payload: SimpleNamespace) -> None:
        raise ValueError(payload.request_id)

    scheduler._request_builder = request_builder

    scheduler.process_input_requests([SimpleNamespace(request_id="req-err")])

    output = scheduler.outbox.get_nowait()
    assert output.request_id == "req-err"
    assert output.type == "error"
    assert isinstance(output.data, ValueError)
    assert scheduler.waiting_queue == []


def test_omni_scheduler_prepares_custom_request_token_budget() -> None:
    """Preserves upstream max_new_tokens clamping for custom request builders."""
    scheduler = object.__new__(OmniScheduler)
    scheduler.outbox = Queue()
    scheduler.waiting_queue = []
    scheduler._pending_stream_chunks = {}
    scheduler._pending_stream_done = set()
    scheduler._deferred_request_payloads = {}
    scheduler._aborted_request_ids = set()
    scheduler.max_req_len = 6
    scheduler.max_req_input_len = 5

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
    scheduler._aborted_request_ids = set()
    scheduler.max_req_len = 6
    scheduler.max_req_input_len = 5

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


def test_omni_scheduler_leaves_request_budget_unchanged_without_opt_in() -> None:
    """Keeps existing OmniScheduler users on their original request semantics."""
    scheduler = object.__new__(OmniScheduler)
    scheduler.outbox = Queue()
    scheduler.waiting_queue = []
    scheduler._pending_stream_chunks = {}
    scheduler._pending_stream_done = set()
    scheduler._deferred_request_payloads = {}
    scheduler._aborted_request_ids = set()
    scheduler.max_req_len = 6
    scheduler.max_req_input_len = 5

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
