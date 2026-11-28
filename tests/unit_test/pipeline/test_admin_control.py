# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import os
import queue
import threading
import time
from collections import deque
from types import SimpleNamespace
from unittest import mock

import pytest

from sglang_omni.pipeline.control_plane import deserialize_message, serialize_message
from sglang_omni.pipeline.coordinator import Coordinator
from sglang_omni.pipeline.stage import runtime as runtime_mod
from sglang_omni.pipeline.stage.runtime import Stage
from sglang_omni.proto import (
    AdminMessage,
    AdminOperation,
    AdminResult,
    AdminResultMessage,
    DataAckMessage,
    ProfilerStartMessage,
    ProfilerStopMessage,
    parse_message,
)
from sglang_omni.scheduling import omni_scheduler as scheduler_module
from tests.unit_test.fixtures.pipeline_fakes import (
    FakeRelay,
    FakeScheduler,
    RecordingCoordinatorControlPlane,
    RecordingStageControlPlane,
)


class AdminScheduler(FakeScheduler):
    def __init__(self) -> None:
        super().__init__()
        self.calls: list[tuple[str, dict]] = []
        self.tp_rank = 0

    def admin(self, action: str, payload: dict):
        self.calls.append((action, payload))
        return {"success": True, "message": "ok", "data": {"action": action}}


class ProfilerControlScheduler(FakeScheduler):
    def __init__(self, *, thread_ready: bool = True, succeed: bool = True) -> None:
        super().__init__()
        self.profiler_calls: list[tuple] = []
        self.thread_ready = thread_ready
        self.succeed = succeed

    def response(self) -> dict:
        if self.succeed:
            return {"success": True}
        return {"success": False, "error": "kineto unavailable"}

    def wait_until_scheduler_thread_ready(self, timeout_s: float) -> bool:
        return self.thread_ready

    def start_torch_profiler(self, trace_path_template: str, run_id: str | None):
        self.profiler_calls.append(("start", trace_path_template, run_id))
        return self.response()

    def stop_torch_profiler(self, run_id: str | None):
        self.profiler_calls.append(("stop", run_id))
        return self.response()


def patch_direct_torch_profiler(
    monkeypatch: pytest.MonkeyPatch, calls: list[tuple]
) -> None:
    """Record direct (non-delegated) TorchProfiler singleton access."""
    monkeypatch.setattr(
        runtime_mod.TorchProfiler,
        "start",
        classmethod(
            lambda cls, template, run_id=None: calls.append(("start", template, run_id))
        ),
    )
    monkeypatch.setattr(
        runtime_mod.TorchProfiler,
        "stop",
        classmethod(lambda cls, *, run_id=None: calls.append(("stop", run_id))),
    )
    monkeypatch.setattr(
        runtime_mod.TorchProfiler, "is_active", classmethod(lambda cls: bool(calls))
    )
    monkeypatch.setattr(
        runtime_mod.TorchProfiler, "get_active_run_id", classmethod(lambda cls: "run-1")
    )


async def release_profiler_lane(stage: Stage) -> None:
    """Drain and tear down the profiler lane, as Stage.stop does.

    Each helper below gets its own asyncio.run loop, and the queue and
    worker are bound to the loop that created them. Production keeps one loop
    for the stage's whole life; tests have to reset between calls.
    """
    await stage.shutdown_profiler_operations()
    stage.profiler_operation_queue = None
    stage.profiler_shutdown.clear()
    stage.profiler_lane_abandoned.clear()


def profile_start(stage: Stage, msg: ProfilerStartMessage) -> None:
    """Handle a start, then wait for the deferred profiler operation."""

    async def run() -> None:
        stage.on_profiler_start(msg)
        await release_profiler_lane(stage)

    asyncio.run(run())


def profile_stop(stage: Stage, msg: ProfilerStopMessage) -> None:
    """Handle a stop, then wait for the deferred profiler operation."""

    async def run() -> None:
        stage.on_profiler_stop(msg)
        await release_profiler_lane(stage)

    asyncio.run(run())


def profiler_stage(scheduler, *, name: str, owner: bool, process_has_owner: bool):
    return Stage(
        name=name,
        role="single",
        get_next=lambda request_id, output: None,
        gpu_id=None,
        endpoints={},
        control_plane=RecordingStageControlPlane(),
        relay=FakeRelay(),
        scheduler=scheduler,
        torch_profiler_owner=owner,
        torch_profiler_process_has_owner=process_has_owner,
    )


def test_admin_messages_round_trip() -> None:
    op = AdminOperation(
        op_id="op-1",
        action="model_info",
        payload={"x": 1},
        target_stages=["decode"],
        timeout_s=12.5,
    )
    msg = AdminMessage(op)

    decoded = deserialize_message(serialize_message(msg))

    assert isinstance(decoded, AdminMessage)
    assert decoded.operation.op_id == "op-1"
    assert decoded.operation.payload == {"x": 1}

    result = AdminResultMessage(
        AdminResult(
            op_id="op-1",
            stage="decode",
            action="model_info",
            success=True,
            data={"model_path": "m"},
        )
    )
    parsed = parse_message(result.to_dict())
    assert isinstance(parsed, AdminResultMessage)
    assert parsed.result.data["model_path"] == "m"


def test_omni_scheduler_admin_enqueues_to_scheduler_thread() -> None:
    from sglang_omni.scheduling.omni_scheduler import OmniScheduler

    scheduler = object.__new__(OmniScheduler)
    scheduler.running = True
    scheduler.tp_size = 1
    scheduler.admin_queue = queue.Queue()
    scheduler.scheduler_thread_id = None

    ready = threading.Event()
    done = threading.Event()
    calls: list[tuple[int, str, dict]] = []

    def run_admin_action(action: str, payload: dict) -> dict:
        calls.append((threading.get_ident(), action, payload))
        done.set()
        return {"success": True, "data": {"thread": "scheduler"}}

    scheduler.run_admin_action = run_admin_action

    def scheduler_thread() -> None:
        scheduler.scheduler_thread_id = threading.get_ident()
        ready.set()
        while not done.is_set():
            OmniScheduler.process_admin_requests(scheduler)
            time.sleep(0.001)

    thread = threading.Thread(target=scheduler_thread)
    thread.start()
    assert ready.wait(timeout=1.0)
    caller_thread_id = threading.get_ident()

    result = OmniScheduler.admin(
        scheduler,
        "model_info",
        {"detail": True, "_admin_timeout_s": 1.0},
    )

    done.set()
    thread.join(timeout=1.0)
    assert result == {"success": True, "data": {"thread": "scheduler"}}
    assert len(calls) == 1
    scheduler_thread_id, action, payload = calls[0]
    assert action == "model_info"
    assert payload == {"detail": True}
    assert scheduler_thread_id != caller_thread_id


def test_omni_scheduler_update_weights_rejects_active_requests_by_default() -> None:
    from sglang_omni.scheduling.omni_scheduler import OmniScheduler

    update_calls: list[dict] = []
    scheduler = object.__new__(OmniScheduler)
    scheduler.model_worker = SimpleNamespace(
        update_weights_from_disk=lambda payload: update_calls.append(payload)
        or (True, "ok")
    )
    scheduler.admin_lock = threading.Lock()
    scheduler._engine_paused = False  # noqa: leading-underscore  # production name
    scheduler.last_pause_mode = None
    scheduler.async_pending = None
    scheduler.resolve_pending_async = lambda: None
    scheduler.active_request_ids = lambda: ["req-1"]

    result = OmniScheduler.admin_update_weights_from_disk(
        scheduler,
        {
            "model_path": "/tmp/new-model",
            "flush_cache": False,
            "abort_all_requests": False,
        },
    )

    assert result["success"] is False
    assert "active requests are present" in result["message"]
    assert result["data"]["active_request_count"] == 1
    assert (
        scheduler._engine_paused is False
    )  # noqa: leading-underscore  # production name
    assert result["data"]["engine_paused"] is False
    assert update_calls == []


def test_omni_scheduler_weights_checker_compare_change_is_success() -> None:
    from sglang_omni.scheduling.omni_scheduler import OmniScheduler

    scheduler = object.__new__(OmniScheduler)
    scheduler.admin_lock = threading.Lock()
    scheduler.model_worker = SimpleNamespace(
        weights_checker=lambda action: {
            "action": action,
            "matched": False,
            "changed": ["weight"],
        }
    )
    result = OmniScheduler.admin_weights_checker(scheduler, {"action": "compare"})
    assert result["success"] is True
    assert result["data"]["matched"] is False
    assert result["data"]["changed"] == ["weight"]


@pytest.mark.parametrize(
    ("admin_method", "worker_method", "payload"),
    [
        pytest.param(
            "admin_update_weights_from_disk",
            "update_weights_from_disk",
            {"model_path": "/tmp/new-model", "torch_empty_cache": True},
            id="disk",
        ),
    ],
)
def test_omni_scheduler_weight_updates_flush_and_advance_epoch(
    admin_method: str, worker_method: str, payload: dict
) -> None:
    from sglang_omni.scheduling.omni_scheduler import OmniScheduler

    update_calls: list[dict] = []
    flush_calls = 0
    empty_cache_calls = 0

    def update_weights(payload: dict) -> tuple[bool, str]:
        update_calls.append(dict(payload))
        return True, "ok"

    def flush_cache() -> bool:
        nonlocal flush_calls
        flush_calls += 1
        return True

    def empty_torch_cache() -> None:
        nonlocal empty_cache_calls
        empty_cache_calls += 1

    scheduler = object.__new__(OmniScheduler)
    scheduler.model_worker = SimpleNamespace(**{worker_method: update_weights})
    scheduler.admin_lock = threading.Lock()
    scheduler._engine_paused = False  # noqa: leading-underscore  # production name
    scheduler.last_pause_mode = None
    scheduler.async_pending = None
    scheduler.request_admission_lock = threading.RLock()
    scheduler.prompt_cache_epoch = 0
    scheduler.waiting_queue = []
    scheduler.resolve_pending_async = lambda: None
    scheduler.active_request_ids = lambda: []
    scheduler.flush_cache = flush_cache
    scheduler.empty_torch_cache = empty_torch_cache

    result = getattr(OmniScheduler, admin_method)(scheduler, payload)

    assert result["success"] is True
    assert result["data"]["flush_cache"] is True
    assert result["data"]["flush_success"] is True
    assert update_calls == [payload]
    assert flush_calls == 1
    assert empty_cache_calls == 1
    assert scheduler.prompt_cache_epoch == 1


def test_weight_swap_isolates_prompt_cache_when_flush_fails() -> None:
    from sglang_omni.scheduling.omni_scheduler import OmniScheduler

    cache_key = "qwen3_tts:prompt:v1"
    retracted = SimpleNamespace(
        extra_key=f"{cache_key}:weights:0",
        _omni_prompt_cache_key=cache_key,
    )
    scheduler = object.__new__(OmniScheduler)
    scheduler.model_worker = SimpleNamespace(
        update_weights_from_disk=lambda _: (True, "swapped")
    )
    scheduler.admin_lock = threading.Lock()
    scheduler.request_admission_lock = threading.RLock()
    scheduler._engine_paused = True  # noqa: leading-underscore  # production name
    scheduler.last_pause_mode = "retract"
    scheduler.prompt_cache_epoch = 0
    scheduler.waiting_queue = [retracted]
    scheduler.resolve_pending_async = lambda: None
    scheduler.active_request_ids = lambda: ["req-1"]
    scheduler.flush_cache = lambda: False
    scheduler.empty_torch_cache = lambda: None

    result = OmniScheduler.admin_update_weights_from_disk(
        scheduler, {"model_path": "/tmp/new-model"}
    )

    fresh = SimpleNamespace(extra_key=cache_key, _omni_prompt_cache_key=cache_key)
    plain = SimpleNamespace(extra_key="plain")
    scheduler.apply_prompt_cache_epoch(fresh)
    scheduler.apply_prompt_cache_epoch(plain)
    assert result["success"] is False
    assert result["data"]["flush_success"] is False
    assert result["data"]["engine_paused"] is True
    assert retracted.extra_key == fresh.extra_key == f"{cache_key}:weights:1"
    assert plain.extra_key == "plain"


@pytest.mark.parametrize("failure_mode", ["return", "raise"])
def test_tensor_update_failure_keeps_engine_paused(failure_mode: str) -> None:
    from sglang_omni.scheduling.omni_scheduler import OmniScheduler

    def update_weights_from_tensor(payload: dict) -> tuple[bool, str]:
        if failure_mode == "raise":
            raise RuntimeError("partially updated")
        return False, "partially updated"

    scheduler = object.__new__(OmniScheduler)
    scheduler.model_worker = SimpleNamespace(
        update_weights_from_tensor=update_weights_from_tensor
    )
    scheduler.admin_lock = threading.Lock()
    scheduler._engine_paused = False  # noqa: leading-underscore  # production name
    scheduler.last_pause_mode = None
    scheduler.prompt_cache_epoch = 0
    scheduler.resolve_pending_async = lambda: None
    scheduler.active_request_ids = lambda: []

    if failure_mode == "raise":
        with pytest.raises(RuntimeError, match="partially updated"):
            scheduler.admin_update_weights_from_tensor({})
    else:
        result = scheduler.admin_update_weights_from_tensor({})
        assert result["success"] is False
        assert result["data"]["engine_paused"] is True

    assert (
        scheduler._engine_paused is True
    )  # noqa: leading-underscore  # production name
    assert scheduler.prompt_cache_epoch == 0


def test_omni_scheduler_flush_cache_has_upstream_idle_compat_fields() -> None:
    from sglang.srt.runtime_context import get_context

    from sglang_omni.scheduling.omni_scheduler import OmniScheduler

    class EmptyBatch:
        reqs: list = []

        def is_empty(self) -> bool:
            return True

    reset_calls: list[str] = []
    scheduler = object.__new__(OmniScheduler)
    scheduler.device = "cuda"
    scheduler.session_adapter = None
    scheduler.tree_cache = SimpleNamespace(reset=lambda: reset_calls.append("tree"))
    OmniScheduler.init_upstream_compat_flags(
        scheduler,
        SimpleNamespace(
            enable_hisparse=False,
            enable_priority_scheduling=False,
            disable_priority_preemption=False,
        ),
    )
    scheduler.running_batch = EmptyBatch()
    scheduler.chunked_req = None
    scheduler.last_batch = None
    scheduler.cur_batch = None
    scheduler.enable_overlap = False
    scheduler.pp_size = 1
    scheduler.waiting_queue = []
    scheduler.grammar_manager = SimpleNamespace(
        grammar_queue=[], clear=lambda: reset_calls.append("grammar")
    )
    scheduler.disaggregation_mode = None
    scheduler.enable_hierarchical_cache = False
    scheduler.enable_lmcache = False
    scheduler._engine_paused = False  # noqa: leading-underscore  # production name
    scheduler.req_to_token_pool = SimpleNamespace(
        clear=lambda: reset_calls.append("req_pool"),
        reset_aux_cache_allocator=lambda: reset_calls.append("aux_cache"),
    )
    scheduler.token_to_kv_pool_allocator = SimpleNamespace(
        clear=lambda: reset_calls.append("kv_pool")
    )
    scheduler.metrics_reporter = SimpleNamespace(
        reset_metrics=lambda: reset_calls.append("metrics"),
        is_stats_logging_rank=False,
    )
    scheduler.draft_worker = None

    with get_context().override_server_args():
        assert OmniScheduler.flush_cache_after_update(scheduler) is True
    assert scheduler.device_module is not None
    assert reset_calls == [
        "tree",
        "req_pool",
        "kv_pool",
        "aux_cache",
        "grammar",
        "metrics",
    ]


def test_omni_scheduler_abort_pause_ends_each_dropped_request() -> None:
    from sglang_omni.admission import AdminDroppedRequestError
    from sglang_omni.scheduling.omni_scheduler import OmniScheduler

    aborted: list[str] = []
    scheduler = object.__new__(OmniScheduler)
    scheduler.admin_lock = threading.Lock()
    scheduler._engine_paused = False  # noqa: leading-underscore  # production name
    scheduler.resolve_pending_async = lambda: None
    scheduler.active_request_ids = lambda: ["req-1", "req-2"]
    scheduler.abort = lambda request_id, **_kwargs: aborted.append(request_id)
    scheduler.running_batch = scheduler.cur_batch = scheduler.last_batch = None
    scheduler.async_pending = None
    scheduler.result_queue = deque()
    scheduler.is_entry_rank = True
    scheduler.outbox = queue.Queue()

    result = OmniScheduler.admin_pause_generation(scheduler, {"mode": "abort"})

    assert aborted == ["req-1", "req-2"]
    assert result["data"]["num_paused_requests"] == 2
    assert "aborted_request_ids" not in result["data"]
    ended = [scheduler.outbox.get_nowait() for _ in range(2)]
    assert [(msg.request_id, msg.type) for msg in ended] == [
        ("req-1", "error"),
        ("req-2", "error"),
    ]
    assert all(AdminDroppedRequestError.matches(msg.data) for msg in ended)


def test_omni_scheduler_distributed_update_aborts_and_flushes_cache() -> None:
    from sglang_omni.scheduling.omni_scheduler import OmniScheduler

    update_calls: list[dict] = []
    flush_calls = 0
    empty_cache_calls = 0
    abort_calls = 0

    def update_weights_from_distributed(payload: dict) -> tuple[bool, str]:
        update_calls.append(dict(payload))
        return True, "ok"

    def flush_cache() -> bool:
        nonlocal flush_calls
        flush_calls += 1
        return True

    def empty_torch_cache() -> None:
        nonlocal empty_cache_calls
        empty_cache_calls += 1

    def abort_all_requests() -> list[str]:
        nonlocal abort_calls
        abort_calls += 1
        return ["req-1"]

    scheduler = object.__new__(OmniScheduler)
    scheduler.model_worker = SimpleNamespace(
        update_weights_from_distributed=update_weights_from_distributed
    )
    scheduler.admin_lock = threading.Lock()
    scheduler._engine_paused = False  # noqa: leading-underscore  # production name
    scheduler.last_pause_mode = None
    scheduler.async_pending = None
    scheduler.request_admission_lock = threading.RLock()
    scheduler.prompt_cache_epoch = 0
    scheduler.waiting_queue = []
    scheduler.resolve_pending_async = lambda: None
    scheduler.active_request_ids = lambda: ["req-1"]
    scheduler.abort_all_requests = abort_all_requests
    scheduler.flush_cache = flush_cache
    scheduler.empty_torch_cache = empty_torch_cache

    payload = {
        "names": ["w.0"],
        "dtypes": ["bfloat16"],
        "shapes": [[2, 2]],
        "group_name": "talker_group",
        "abort_all_requests": True,
        "torch_empty_cache": True,
    }
    result = OmniScheduler.admin_update_weights_from_distributed(scheduler, payload)

    assert result["success"] is True
    assert result["data"]["num_paused_requests"] == 1
    assert result["data"]["flush_cache"] is True
    assert result["data"]["flush_success"] is True
    assert result["data"]["group_name"] == "talker_group"
    assert result["data"]["names"] == ["w.0"]
    assert update_calls == [payload]
    assert abort_calls == 1
    assert flush_calls == 1
    assert empty_cache_calls == 1
    assert scheduler.prompt_cache_epoch == 1


def test_omni_scheduler_distributed_update_failure_keeps_engine_paused() -> None:
    from sglang_omni.scheduling.omni_scheduler import OmniScheduler

    def update_weights_from_distributed(payload: dict) -> tuple[bool, str]:
        return (
            False,
            "Failed to update parameter online: partially updated; discard weights",
        )

    scheduler = object.__new__(OmniScheduler)
    scheduler.model_worker = SimpleNamespace(
        update_weights_from_distributed=update_weights_from_distributed
    )
    scheduler.admin_lock = threading.Lock()
    scheduler._engine_paused = False  # noqa: leading-underscore  # production name
    scheduler.last_pause_mode = None
    scheduler.async_pending = None
    scheduler.resolve_pending_async = lambda: None
    scheduler.active_request_ids = lambda: []

    result = OmniScheduler.admin_update_weights_from_distributed(
        scheduler,
        {
            "names": ["w.0"],
            "dtypes": ["bfloat16"],
            "shapes": [[2, 2]],
            "flush_cache": False,
        },
    )

    assert result["success"] is False
    assert "partially updated" in result["message"]
    assert result["data"]["engine_paused"] is True
    assert (
        scheduler._engine_paused is True
    )  # noqa: leading-underscore  # production name


def test_coordinator_admin_waits_for_all_stage_results() -> None:
    async def run() -> None:
        coordinator = Coordinator(
            "inproc://complete",
            "inproc://abort",
            entry_stage="preprocess",
        )
        control_plane = RecordingCoordinatorControlPlane()
        coordinator.control_plane = control_plane
        coordinator.running = True
        coordinator.register_stage("decode", "inproc://decode")
        coordinator.register_stage("vocoder", "inproc://vocoder")

        task = asyncio.create_task(
            coordinator.admin("model_info", {"detail": True}, timeout_s=1)
        )
        while len(control_plane.submitted) < 2:
            await asyncio.sleep(0)

        for stage, _, msg in control_plane.submitted:
            assert isinstance(msg, AdminMessage)
            coordinator.handle_admin_result(
                AdminResult(
                    op_id=msg.operation.op_id,
                    stage=stage,
                    action=msg.operation.action,
                    success=True,
                    data={"stage": stage},
                )
            )

        result = await task
        assert result["success"] is True
        assert {item["stage"] for item in result["results"]} == {"decode", "vocoder"}

    asyncio.run(run())


def test_stage_admin_dispatches_to_scheduler() -> None:
    async def run() -> None:
        scheduler = AdminScheduler()
        control_plane = RecordingStageControlPlane()
        stage = Stage(
            name="decode",
            role="single",
            get_next=lambda request_id, output: None,
            gpu_id=None,
            endpoints={},
            control_plane=control_plane,
            relay=FakeRelay(),
            scheduler=scheduler,
        )

        await stage.on_admin(
            AdminMessage(
                AdminOperation(
                    op_id="op-1",
                    action="pause_generation",
                    payload={"mode": "in_place"},
                )
            )
        )

        assert scheduler.calls == [("pause_generation", {"mode": "in_place"})]
        result_msg = control_plane.completions[0]
        assert isinstance(result_msg, AdminResultMessage)
        assert result_msg.result.success is True
        assert result_msg.result.data["action"] == "pause_generation"

    asyncio.run(run())


def test_omni_scheduler_model_info_adds_the_model_runner_entries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from sglang_omni.scheduling import omni_scheduler
    from sglang_omni.scheduling.omni_scheduler import OmniScheduler

    monkeypatch.setattr(
        omni_scheduler,
        "get_model",
        lambda: SimpleNamespace(model_path="/ckpt", load_format="auto"),
    )
    monkeypatch.setattr(
        omni_scheduler, "get_serving", lambda: SimpleNamespace(weight_version="v1")
    )
    scheduler = object.__new__(OmniScheduler)
    scheduler.model_worker = SimpleNamespace(
        model_info=lambda: {"prefill_cuda_graph": {"replay_count": 3}}
    )
    scheduler.model_runner = SimpleNamespace(
        model_info=lambda: {"predictor_cuda_graph": {"replays": 5}}
    )
    scheduler.request_admission_lock = threading.Lock()
    scheduler.pending_request_builds = {}
    scheduler.pending_request_admissions = {}
    scheduler.backlogged_request_build_payloads = []
    scheduler.waiting_queue = []
    scheduler.tp_rank = 0
    scheduler.tp_size = 1
    scheduler._engine_paused = False  # noqa: leading-underscore  # production name
    scheduler.request_build_max_workers = 1
    scheduler.request_build_max_pending = 16
    scheduler.request_build_max_pending_observed = 0
    scheduler.running_batch = SimpleNamespace(reqs=[])

    result = OmniScheduler.admin_model_info(scheduler)

    assert result["data"]["prefill_cuda_graph"] == {"replay_count": 3}
    assert result["data"]["predictor_cuda_graph"] == {"replays": 5}
    assert result["data"]["weight_version"] == "v1"


def test_stage_routes_torch_profiler_to_explicit_owner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SGLANG_TORCH_PROFILER_SCHEDULER_THREAD", "1")
    scheduler = ProfilerControlScheduler()
    stage = Stage(
        name="ar-with-an-arbitrary-name",
        role="single",
        get_next=lambda request_id, output: None,
        gpu_id=None,
        endpoints={},
        control_plane=RecordingStageControlPlane(),
        relay=FakeRelay(),
        scheduler=scheduler,
        torch_profiler_owner=True,
    )

    profile_start(
        stage,
        ProfilerStartMessage(
            run_id="run-1",
            trace_path_template="/tmp/{run_id}/{stage}/trace",
        ),
    )
    profile_stop(stage, ProfilerStopMessage(run_id="run-1"))

    assert scheduler.profiler_calls == [
        (
            "start",
            f"/tmp/run-1/ar-with-an-arbitrary-name/trace_pid{os.getpid()}",
            "run-1",
        ),
        ("stop", "run-1"),
    ]


def test_stage_rejects_profiler_owner_without_control_capability() -> None:
    with pytest.raises(TypeError, match="does not implement"):
        Stage(
            name="decode",
            role="single",
            get_next=lambda request_id, output: None,
            gpu_id=None,
            endpoints={},
            control_plane=RecordingStageControlPlane(),
            relay=FakeRelay(),
            scheduler=FakeScheduler(),
            torch_profiler_owner=True,
        )


def test_ownerless_process_still_profiles_directly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A process with no owner keeps the pre-existing direct path, so a partial
    rollout never silently drops that process's Torch trace."""
    monkeypatch.setenv("SGLANG_TORCH_PROFILER_SCHEDULER_THREAD", "1")
    direct_calls: list[tuple] = []
    patch_direct_torch_profiler(monkeypatch, direct_calls)

    scheduler = ProfilerControlScheduler()
    stage = profiler_stage(
        scheduler, name="vocoder", owner=False, process_has_owner=False
    )

    profile_start(
        stage,
        ProfilerStartMessage(run_id="run-1", trace_path_template="/tmp/{stage}/trace"),
    )
    profile_stop(stage, ProfilerStopMessage(run_id="run-1"))

    assert direct_calls == [
        ("start", f"/tmp/vocoder/trace_pid{os.getpid()}", "run-1"),
        ("stop", "run-1"),
    ]
    assert scheduler.profiler_calls == []


def test_ownerless_process_forwards_a_new_run_to_the_profiler(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The process-wide singleton decides whether a start is idempotent or
    replaces an active run. The stage must not hide a different run from it."""
    monkeypatch.setenv("SGLANG_TORCH_PROFILER_SCHEDULER_THREAD", "1")
    direct_calls: list[tuple] = []
    patch_direct_torch_profiler(monkeypatch, direct_calls)

    scheduler = ProfilerControlScheduler()
    stage = profiler_stage(
        scheduler, name="vocoder", owner=False, process_has_owner=False
    )

    profile_start(
        stage,
        ProfilerStartMessage(run_id="run-1", trace_path_template="/tmp/{stage}/trace"),
    )
    profile_start(
        stage,
        ProfilerStartMessage(run_id="run-2", trace_path_template="/tmp/{stage}/trace"),
    )

    assert direct_calls == [
        ("start", f"/tmp/vocoder/trace_pid{os.getpid()}", "run-1"),
        ("start", f"/tmp/vocoder/trace_pid{os.getpid()}", "run-2"),
    ]
    assert scheduler.profiler_calls == []


def test_colocated_sibling_never_touches_the_profiler_singleton(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """MOSS-TTS-Local runs preprocessing and the owning tts_engine in
    one process. If the sibling handled the broadcast directly it would create
    the singleton on the control thread and the owner's start would then no-op
    on it -- a trace with CUDA activity but no CPU operators, the exact defect
    this mode exists to fix."""
    monkeypatch.setenv("SGLANG_TORCH_PROFILER_SCHEDULER_THREAD", "1")
    direct_calls: list[tuple] = []
    patch_direct_torch_profiler(monkeypatch, direct_calls)

    scheduler = ProfilerControlScheduler()
    sibling = profiler_stage(
        scheduler, name="preprocessing", owner=False, process_has_owner=True
    )
    owner = profiler_stage(
        scheduler, name="tts_engine", owner=True, process_has_owner=True
    )

    # The sibling is handed the broadcast first: worst-case ordering.
    msg = ProfilerStartMessage(run_id="run-1", trace_path_template="/tmp/{stage}/trace")
    profile_start(sibling, msg)
    profile_start(owner, msg)
    profile_stop(sibling, ProfilerStopMessage(run_id="run-1"))
    profile_stop(owner, ProfilerStopMessage(run_id="run-1"))

    assert direct_calls == []
    assert scheduler.profiler_calls == [
        ("start", f"/tmp/tts_engine/trace_pid{os.getpid()}", "run-1"),
        ("stop", "run-1"),
    ]


def test_owner_refuses_to_profile_before_the_scheduler_thread_is_ready(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Stage.start does not wait for the scheduler thread to publish its id,
    so a start can arrive early. Falling back to a direct start would silently
    profile on the wrong thread; refusing is the honest outcome."""
    monkeypatch.setenv("SGLANG_TORCH_PROFILER_SCHEDULER_THREAD", "1")
    direct_calls: list[tuple] = []
    patch_direct_torch_profiler(monkeypatch, direct_calls)

    scheduler = ProfilerControlScheduler(thread_ready=False)
    owner = profiler_stage(
        scheduler, name="tts_engine", owner=True, process_has_owner=True
    )

    profile_start(
        owner,
        ProfilerStartMessage(run_id="run-1", trace_path_template="/tmp/{stage}/trace"),
    )

    assert direct_calls == []
    assert scheduler.profiler_calls == []


def test_profiler_control_failure_does_not_kill_the_stage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Profiler broadcasts are fire-and-forget PUSH with no response channel,
    and an exception escaping a message handler tears down every stage in the
    process. A failed profiler start must not cost the serving process."""
    monkeypatch.setenv("SGLANG_TORCH_PROFILER_SCHEDULER_THREAD", "1")
    direct_calls: list[tuple] = []
    patch_direct_torch_profiler(monkeypatch, direct_calls)

    scheduler = ProfilerControlScheduler(succeed=False)
    owner = profiler_stage(
        scheduler, name="tts_engine", owner=True, process_has_owner=True
    )

    profile_start(
        owner,
        ProfilerStartMessage(run_id="run-1", trace_path_template="/tmp/{stage}/trace"),
    )
    profile_stop(owner, ProfilerStopMessage(run_id="run-1"))

    assert [name for name, *_ in scheduler.profiler_calls] == ["start", "stop"]
    assert direct_calls == []


def test_scheduler_thread_mode_off_keeps_legacy_behavior(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without the env flag an owner stage behaves exactly as before the PR."""
    monkeypatch.delenv("SGLANG_TORCH_PROFILER_SCHEDULER_THREAD", raising=False)
    direct_calls: list[tuple] = []
    patch_direct_torch_profiler(monkeypatch, direct_calls)

    scheduler = ProfilerControlScheduler()
    owner = profiler_stage(
        scheduler, name="tts_engine", owner=True, process_has_owner=True
    )

    profile_start(
        owner,
        ProfilerStartMessage(run_id="run-1", trace_path_template="/tmp/{stage}/trace"),
    )
    profile_stop(owner, ProfilerStopMessage(run_id="run-1"))

    assert direct_calls == [
        ("start", f"/tmp/tts_engine/trace_pid{os.getpid()}", "run-1"),
        ("stop", "run-1"),
    ]
    assert scheduler.profiler_calls == []


def test_enable_torch_false_never_starts_the_profiler(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The event recorder still runs, but no Torch profiler is touched."""
    monkeypatch.setenv("SGLANG_TORCH_PROFILER_SCHEDULER_THREAD", "1")
    direct_calls: list[tuple] = []
    patch_direct_torch_profiler(monkeypatch, direct_calls)

    scheduler = ProfilerControlScheduler()
    owner = profiler_stage(
        scheduler, name="tts_engine", owner=True, process_has_owner=True
    )

    profile_start(
        owner,
        ProfilerStartMessage(
            run_id="run-1",
            trace_path_template="/tmp/{stage}/trace",
            enable_torch=False,
        ),
    )

    assert direct_calls == []
    assert scheduler.profiler_calls == []


def test_scheduler_admin_runs_torch_profiler_on_the_scheduler_thread() -> None:
    """The point of the fix: Kineto CPU callbacks are thread-local, so
    TorchProfiler.start/stop must execute on the model-execution thread, not
    on the caller's control-plane thread."""
    from sglang_omni.scheduling.omni_scheduler import OmniScheduler

    scheduler = OmniScheduler.__new__(OmniScheduler)
    scheduler.tp_size = 1
    scheduler.admin_queue = queue.Queue()
    scheduler.scheduler_thread_id = None
    scheduler.running = True

    executed_on: list[tuple[str, int]] = []

    class RecordingTorchProfiler:
        @staticmethod
        def start(template: str, run_id: str | None = None) -> str:
            executed_on.append(("start", threading.get_ident()))
            return f"{template}_rank0.trace.json.gz"

        @staticmethod
        def stop(*, run_id: str | None = None) -> dict:
            executed_on.append(("stop", threading.get_ident()))
            return {"trace": "t", "table": None}

    stop_event = threading.Event()
    ready = threading.Event()

    def scheduler_loop() -> None:
        scheduler.scheduler_thread_id = threading.get_ident()
        ready.set()
        while not stop_event.is_set():
            scheduler.process_admin_requests()
            time.sleep(0.001)

    thread = threading.Thread(target=scheduler_loop, daemon=True)
    thread.start()
    ready.wait(timeout=5.0)

    with mock.patch.object(scheduler_module, "TorchProfiler", RecordingTorchProfiler):
        try:
            start_response = scheduler.start_torch_profiler("/tmp/trace", "run-1")
            stop_response = scheduler.stop_torch_profiler("run-1")
        finally:
            stop_event.set()
            thread.join(timeout=5.0)

    assert start_response["success"] is True
    assert stop_response["success"] is True

    caller_thread_id = threading.get_ident()
    assert [name for name, _ in executed_on] == ["start", "stop"]
    for name, thread_id in executed_on:
        assert thread_id == scheduler.scheduler_thread_id, name
        assert thread_id != caller_thread_id, name


def test_scheduler_admin_profiler_failure_surfaces_to_the_caller() -> None:
    """Exceptions raised on the scheduler thread come back as admin errors so
    Stage can turn them into a RuntimeError instead of hanging."""
    from sglang_omni.scheduling.omni_scheduler import OmniScheduler

    scheduler = OmniScheduler.__new__(OmniScheduler)
    scheduler.tp_size = 1
    scheduler.enable_overlap = False
    scheduler.admin_queue = queue.Queue()
    scheduler.scheduler_thread_id = None
    scheduler.running = True

    class ExplodingTorchProfiler:
        @staticmethod
        def start(template: str, run_id: str | None = None) -> str:
            raise RuntimeError("kineto unavailable")

    stop_event = threading.Event()
    ready = threading.Event()

    def scheduler_loop() -> None:
        scheduler.scheduler_thread_id = threading.get_ident()
        ready.set()
        while not stop_event.is_set():
            scheduler.process_admin_requests()
            time.sleep(0.001)

    thread = threading.Thread(target=scheduler_loop, daemon=True)
    thread.start()
    ready.wait(timeout=5.0)

    with mock.patch.object(scheduler_module, "TorchProfiler", ExplodingTorchProfiler):
        try:
            response = scheduler.start_torch_profiler("/tmp/trace", "run-1")
        finally:
            stop_event.set()
            thread.join(timeout=5.0)

    assert response["success"] is False
    assert "kineto unavailable" in response["error"]


def test_scheduler_thread_readiness_reports_when_admin_would_run_inline() -> None:
    """should_enqueue_admin silently runs inline until the scheduler thread
    publishes its id. The profiler owner needs to distinguish those states."""
    from sglang_omni.scheduling.omni_scheduler import OmniScheduler

    scheduler = OmniScheduler.__new__(OmniScheduler)
    scheduler.tp_size = 1
    scheduler.admin_queue = queue.Queue()
    scheduler.scheduler_thread_id = None
    scheduler.running = False

    assert scheduler.wait_until_scheduler_thread_ready(0.05) is False

    scheduler.running = True
    assert scheduler.wait_until_scheduler_thread_ready(0.05) is False

    def publish() -> None:
        time.sleep(0.02)
        scheduler.scheduler_thread_id = threading.get_ident()

    thread = threading.Thread(target=publish, daemon=True)
    thread.start()
    try:
        assert scheduler.wait_until_scheduler_thread_ready(5.0) is True
    finally:
        thread.join(timeout=5.0)


def test_owner_refuses_to_stop_after_the_scheduler_thread_is_gone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Stopping inline on the control thread is the same wrong-thread defect as
    starting there, so a shutting-down scheduler must not be worked around."""
    monkeypatch.setenv("SGLANG_TORCH_PROFILER_SCHEDULER_THREAD", "1")
    direct_calls: list[tuple] = []
    patch_direct_torch_profiler(monkeypatch, direct_calls)

    scheduler = ProfilerControlScheduler(thread_ready=False)
    owner = profiler_stage(
        scheduler, name="tts_engine", owner=True, process_has_owner=True
    )

    profile_stop(owner, ProfilerStopMessage(run_id="run-1"))

    assert direct_calls == []
    assert scheduler.profiler_calls == []


def test_profiler_admin_never_runs_inline_after_the_loop_exits() -> None:
    """OmniScheduler.start clears scheduler_thread_id in its finally
    but leaves _running set until stop runs, so an exited scheduler loop
    leaves a durable window where should_enqueue_admin reports inline. A
    readiness check on the Stage side cannot close it -- the loop can exit
    between that check and the handoff -- so the profiler path must re-check and
    fail rather than run TorchProfiler on the caller's thread."""
    from sglang_omni.scheduling.omni_scheduler import OmniScheduler

    scheduler = OmniScheduler.__new__(OmniScheduler)
    scheduler.tp_size = 1
    scheduler.admin_queue = queue.Queue()
    # Exactly the post-loop state: id cleared, _running not yet cleared.
    scheduler.running = True
    scheduler.scheduler_thread_id = None

    executed_on: list[int] = []

    class RecordingTorchProfiler:
        @staticmethod
        def start(template: str, run_id: str | None = None) -> str:
            executed_on.append(threading.get_ident())
            return "trace"

        @staticmethod
        def stop(*, run_id: str | None = None) -> dict:
            executed_on.append(threading.get_ident())
            return {}

    with mock.patch.object(scheduler_module, "TorchProfiler", RecordingTorchProfiler):
        start_response = scheduler.start_torch_profiler("/tmp/trace", "run-1")
        stop_response = scheduler.stop_torch_profiler("run-1")

    assert start_response["success"] is False
    assert stop_response["success"] is False
    # The whole point: nothing ran on this thread.
    assert executed_on == []
    assert scheduler.admin_queue.empty()


def test_profiler_admin_runs_inline_when_already_on_the_scheduler_thread() -> None:
    """Being the scheduler thread is the goal state, not a fallback."""
    from sglang_omni.scheduling.omni_scheduler import OmniScheduler

    scheduler = OmniScheduler.__new__(OmniScheduler)
    scheduler.tp_size = 1
    scheduler.admin_queue = queue.Queue()
    scheduler.running = True
    scheduler.scheduler_thread_id = threading.get_ident()

    executed_on: list[int] = []

    class RecordingTorchProfiler:
        @staticmethod
        def start(template: str, run_id: str | None = None) -> str:
            executed_on.append(threading.get_ident())
            return "trace"

    with mock.patch.object(scheduler_module, "TorchProfiler", RecordingTorchProfiler):
        response = scheduler.start_torch_profiler("/tmp/trace", "run-1")

    assert response["success"] is True
    assert executed_on == [threading.get_ident()]


def test_expired_profiler_start_is_dropped_when_drained_late() -> None:
    """enqueue_admin reports a timeout but leaves the request queued. A
    scheduler that was busy past the deadline must drop it: the caller was told
    the start failed and will never issue a stop, so running it late leaves an
    orphan profiler taxing the serving path."""
    from sglang_omni.scheduling.omni_scheduler import OmniScheduler

    scheduler = OmniScheduler.__new__(OmniScheduler)
    scheduler.tp_size = 1
    scheduler.admin_queue = queue.Queue()
    scheduler.running = True
    # A live scheduler thread that is too busy to drain: forces the enqueue path.
    scheduler.scheduler_thread_id = threading.get_ident() + 1

    started: list[str] = []

    class RecordingTorchProfiler:
        @staticmethod
        def start(template: str, run_id: str | None = None) -> str:
            started.append(template)
            return "trace"

    with mock.patch.object(
        OmniScheduler, "enqueue_admin", autospec=True
    ) as enqueue_admin:

        def queue_and_time_out(self, action, payload):
            # Mirror the real body, minus the blocking wait the caller gave up on.
            queued = dict(payload)
            queued.pop("_admin_timeout_s", None)
            self.admin_queue.put((action, queued, queue.Queue(maxsize=1)))
            return {"success": False, "error": "admin operation timed out"}

        enqueue_admin.side_effect = queue_and_time_out
        response = scheduler.start_torch_profiler("/tmp/trace", "run-1")

    assert response["success"] is False
    assert scheduler.admin_queue.qsize() == 1

    # The scheduler frees up after the deadline and drains what it missed.
    action, payload, _ = scheduler.admin_queue.get_nowait()
    payload["_deadline"] = time.monotonic() - 1.0
    with mock.patch.object(scheduler_module, "TorchProfiler", RecordingTorchProfiler):
        late = scheduler.run_admin_action(action, payload)

    assert late["success"] is False
    assert "expired" in late["error"]
    assert started == []


def test_live_profiler_request_carries_a_deadline_but_still_runs() -> None:
    """The deadline must not reject requests that arrive in time."""
    from sglang_omni.scheduling.omni_scheduler import OmniScheduler

    scheduler = OmniScheduler.__new__(OmniScheduler)
    scheduler.tp_size = 1
    started: list[str] = []

    class RecordingTorchProfiler:
        @staticmethod
        def start(template: str, run_id: str | None = None) -> str:
            started.append(template)
            return "trace"

    with mock.patch.object(scheduler_module, "TorchProfiler", RecordingTorchProfiler):
        response = scheduler.run_admin_action(
            "torch_profiler_start",
            {
                "trace_path_template": "/tmp/trace",
                "run_id": "run-1",
                "_deadline": time.monotonic() + 30.0,
            },
        )

    assert response["success"] is True
    assert started == ["/tmp/trace"]


def test_profiler_delegation_does_not_stall_control_message_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The readiness wait and the admin handoff each block for up to 30s. If a
    handler awaited them, run would not reach its next control_plane.recv()
    and this stage's submits and acks would queue behind a profiler request."""
    monkeypatch.setenv("SGLANG_TORCH_PROFILER_SCHEDULER_THREAD", "1")
    direct_calls: list[tuple] = []
    patch_direct_torch_profiler(monkeypatch, direct_calls)

    release = threading.Event()

    class BlockingScheduler(ProfilerControlScheduler):
        def wait_until_scheduler_thread_ready(self, timeout_s: float) -> bool:
            release.wait(timeout=5.0)
            return True

    owner = profiler_stage(
        BlockingScheduler(), name="tts_engine", owner=True, process_has_owner=True
    )
    handled: list[str] = []

    async def run() -> None:
        # Exactly what Stage.run does: dispatch, then take the next message.
        await owner.handle_message(
            ProfilerStartMessage(
                run_id="run-1", trace_path_template="/tmp/{stage}/trace"
            )
        )
        handled.append("profiler_start")

        for index in range(3):
            await owner.handle_message(
                DataAckMessage(
                    request_id=f"req-{index}",
                    from_stage="upstream",
                    to_stage="tts_engine",
                    object_id=f"obj-{index}",
                )
            )
            handled.append(f"ack-{index}")

        # Dispatch kept flowing while the delegation was still blocked.
        assert owner.scheduler.profiler_calls == []
        release.set()
        await asyncio.wait_for(owner.profiler_operation_queue.join(), timeout=5.0)
        await owner.stop()

    asyncio.run(run())

    assert handled == ["profiler_start", "ack-0", "ack-1", "ack-2"]
    assert owner.scheduler.profiler_calls == [
        ("start", f"/tmp/tts_engine/trace_pid{os.getpid()}", "run-1")
    ]


def test_profiler_operations_run_in_arrival_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Deferring the work must not let a stop overtake its own start."""
    monkeypatch.setenv("SGLANG_TORCH_PROFILER_SCHEDULER_THREAD", "1")
    direct_calls: list[tuple] = []
    patch_direct_torch_profiler(monkeypatch, direct_calls)

    class SlowStartScheduler(ProfilerControlScheduler):
        def start_torch_profiler(self, trace_path_template: str, run_id: str | None):
            time.sleep(0.05)
            return super().start_torch_profiler(trace_path_template, run_id)

    owner = profiler_stage(
        SlowStartScheduler(), name="tts_engine", owner=True, process_has_owner=True
    )

    async def run() -> None:
        owner.on_profiler_start(
            ProfilerStartMessage(
                run_id="run-1", trace_path_template="/tmp/{stage}/trace"
            )
        )
        owner.on_profiler_stop(ProfilerStopMessage(run_id="run-1"))
        await asyncio.wait_for(owner.profiler_operation_queue.join(), timeout=5.0)
        await owner.stop()

    asyncio.run(run())

    assert [name for name, *_ in owner.scheduler.profiler_calls] == ["start", "stop"]


def test_shutdown_drains_a_queued_profiler_stop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """/stop_profile immediately followed by shutdown is an ordinary operator
    sequence. The queued stop is what exports the trace, so shutdown has to let
    it run instead of cancelling it away."""
    monkeypatch.setenv("SGLANG_TORCH_PROFILER_SCHEDULER_THREAD", "1")
    direct_calls: list[tuple] = []
    patch_direct_torch_profiler(monkeypatch, direct_calls)

    scheduler = ProfilerControlScheduler()
    owner = profiler_stage(
        scheduler, name="tts_engine", owner=True, process_has_owner=True
    )

    async def run() -> None:
        owner.on_profiler_stop(ProfilerStopMessage(run_id="run-1"))
        # No drain here: shutdown lands while the stop is still queued.
        await owner.stop()

    asyncio.run(run())

    assert scheduler.profiler_calls == [("stop", "run-1")]


def test_shutdown_is_bounded_when_a_profiler_operation_is_stuck(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Shutdown must not hang on a scheduler that is already gone, and must not
    accept new profiler work once it has begun."""
    monkeypatch.setenv("SGLANG_TORCH_PROFILER_SCHEDULER_THREAD", "1")
    monkeypatch.setattr(runtime_mod, "PROFILER_SHUTDOWN_DRAIN_TIMEOUT_S", 0.05)
    direct_calls: list[tuple] = []
    patch_direct_torch_profiler(monkeypatch, direct_calls)

    release = threading.Event()

    class StuckScheduler(ProfilerControlScheduler):
        def wait_until_scheduler_thread_ready(self, timeout_s: float) -> bool:
            release.wait(timeout=5.0)
            return True

    scheduler = StuckScheduler()
    owner = profiler_stage(
        scheduler, name="tts_engine", owner=True, process_has_owner=True
    )

    async def run() -> float:
        owner.on_profiler_start(
            ProfilerStartMessage(
                run_id="run-1", trace_path_template="/tmp/{stage}/trace"
            )
        )
        await asyncio.sleep(0)  # let the worker reach the stuck call
        started = time.monotonic()
        await owner.stop()
        elapsed = time.monotonic() - started

        # Shutdown has begun: further profiler work is refused outright.
        owner.on_profiler_stop(ProfilerStopMessage(run_id="run-1"))
        assert owner.profiler_operation_queue is not None
        assert owner.profiler_operation_queue.empty()
        return elapsed

    try:
        elapsed = asyncio.run(run())
    finally:
        release.set()

    assert elapsed < 2.0
    assert direct_calls == []
    # The executor call outlived shutdown -- Task.cancel cannot reach it -- but
    # it found the lane abandoned and never reached the scheduler.
    assert [name for name, *_ in scheduler.profiler_calls] == []
