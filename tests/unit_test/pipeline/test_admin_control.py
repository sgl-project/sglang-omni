# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import os
import queue
import sys
import threading
import time
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
        self._thread_ready = thread_ready
        self._succeed = succeed

    def _response(self) -> dict:
        if self._succeed:
            return {"success": True}
        return {"success": False, "error": "kineto unavailable"}

    def wait_until_scheduler_thread_ready(self, timeout_s: float) -> bool:
        return self._thread_ready

    def start_torch_profiler(self, trace_path_template: str, run_id: str | None):
        self.profiler_calls.append(("start", trace_path_template, run_id))
        return self._response()

    def stop_torch_profiler(self, run_id: str | None):
        self.profiler_calls.append(("stop", run_id))
        return self._response()


def _patch_direct_torch_profiler(
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


async def _release_profiler_lane(stage: Stage) -> None:
    """Drain and tear down the profiler lane, as ``Stage.stop`` does.

    Each helper below gets its own ``asyncio.run`` loop, and the queue and
    worker are bound to the loop that created them. Production keeps one loop
    for the stage's whole life; tests have to reset between calls.
    """
    await stage._shutdown_profiler_ops()
    stage._profiler_op_queue = None
    stage._profiler_shutdown.clear()
    stage._profiler_lane_abandoned.clear()


def _profile_start(stage: Stage, msg: ProfilerStartMessage) -> None:
    """Handle a start, then wait for the deferred profiler operation."""

    async def _run() -> None:
        stage._on_profiler_start(msg)
        await _release_profiler_lane(stage)

    asyncio.run(_run())


def _profile_stop(stage: Stage, msg: ProfilerStopMessage) -> None:
    """Handle a stop, then wait for the deferred profiler operation."""

    async def _run() -> None:
        stage._on_profiler_stop(msg)
        await _release_profiler_lane(stage)

    asyncio.run(_run())


def _profiler_stage(scheduler, *, name: str, owner: bool, process_has_owner: bool):
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
    scheduler._running = True
    scheduler._admin_queue = queue.Queue()
    scheduler._scheduler_thread_id = None

    ready = threading.Event()
    done = threading.Event()
    calls: list[tuple[int, str, dict]] = []

    def run_admin_action(action: str, payload: dict) -> dict:
        calls.append((threading.get_ident(), action, payload))
        done.set()
        return {"success": True, "data": {"thread": "scheduler"}}

    scheduler._run_admin_action = run_admin_action

    def scheduler_thread() -> None:
        scheduler._scheduler_thread_id = threading.get_ident()
        ready.set()
        while not done.is_set():
            OmniScheduler._process_admin_requests(scheduler)
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
    scheduler._admin_lock = threading.Lock()
    scheduler._engine_paused = False
    scheduler._last_pause_mode = None
    scheduler._async_pending = None
    scheduler._resolve_pending_async = lambda: None
    scheduler._active_request_ids = lambda: ["req-1"]

    result = OmniScheduler._admin_update_weights_from_disk(
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
    assert scheduler._engine_paused is False
    assert result["data"]["engine_paused"] is False
    assert update_calls == []


def test_omni_scheduler_weights_checker_compare_change_is_success() -> None:
    from sglang_omni.scheduling.omni_scheduler import OmniScheduler

    scheduler = object.__new__(OmniScheduler)
    scheduler._admin_lock = threading.Lock()
    scheduler.model_worker = SimpleNamespace(
        weights_checker=lambda action: {
            "action": action,
            "matched": False,
            "changed": ["weight"],
        }
    )
    result = OmniScheduler._admin_weights_checker(scheduler, {"action": "compare"})
    assert result["success"] is True
    assert result["data"]["matched"] is False
    assert result["data"]["changed"] == ["weight"]


def test_omni_scheduler_update_weights_flushes_cache_without_kwargs() -> None:
    from sglang_omni.scheduling.omni_scheduler import OmniScheduler

    update_calls: list[dict] = []
    flush_calls = 0
    empty_cache_calls = 0

    def update_weights_from_disk(payload: dict) -> tuple[bool, str]:
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
    scheduler.model_worker = SimpleNamespace(
        update_weights_from_disk=update_weights_from_disk
    )
    scheduler._admin_lock = threading.Lock()
    scheduler._engine_paused = False
    scheduler._last_pause_mode = None
    scheduler._async_pending = None
    scheduler._resolve_pending_async = lambda: None
    scheduler._active_request_ids = lambda: []
    scheduler.flush_cache = flush_cache
    scheduler._empty_torch_cache = empty_torch_cache

    result = OmniScheduler._admin_update_weights_from_disk(
        scheduler,
        {
            "model_path": "/tmp/new-model",
            "torch_empty_cache": True,
        },
    )

    assert result["success"] is True
    assert result["data"]["flush_cache"] is True
    assert result["data"]["flush_success"] is True
    assert update_calls == [{"model_path": "/tmp/new-model", "torch_empty_cache": True}]
    assert flush_calls == 1
    assert empty_cache_calls == 1


def test_omni_scheduler_flush_cache_has_upstream_idle_compat_fields() -> None:
    from sglang_omni.scheduling.omni_scheduler import OmniScheduler

    class EmptyBatch:
        reqs: list = []

        def is_empty(self) -> bool:
            return True

    reset_calls: list[str] = []
    scheduler = object.__new__(OmniScheduler)
    scheduler.device = "cuda"
    OmniScheduler._init_upstream_compat_flags(
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
    scheduler.tree_cache = SimpleNamespace(reset=lambda: reset_calls.append("tree"))
    scheduler.req_to_token_pool = SimpleNamespace(
        clear=lambda: reset_calls.append("req_pool")
    )
    scheduler.token_to_kv_pool_allocator = SimpleNamespace(
        clear=lambda: reset_calls.append("kv_pool")
    )
    scheduler.ps = SimpleNamespace(pp_size=1)
    scheduler.metrics_reporter = SimpleNamespace(
        reset_metrics=lambda: reset_calls.append("metrics"),
        is_stats_logging_rank=False,
    )
    scheduler.draft_worker = None

    assert OmniScheduler._flush_cache_after_update(scheduler) is True
    assert scheduler.device_module is not None
    assert reset_calls == [
        "tree",
        "req_pool",
        "kv_pool",
        "grammar",
        "metrics",
    ]


def test_omni_scheduler_distributed_update_rejects_active_requests_by_default() -> None:
    from sglang_omni.scheduling.omni_scheduler import OmniScheduler

    update_calls: list[dict] = []
    scheduler = object.__new__(OmniScheduler)
    scheduler.model_worker = SimpleNamespace(
        update_weights_from_distributed=lambda payload: update_calls.append(payload)
        or (True, "ok")
    )
    scheduler._admin_lock = threading.Lock()
    scheduler._engine_paused = False
    scheduler._last_pause_mode = None
    scheduler._async_pending = None
    scheduler._resolve_pending_async = lambda: None
    scheduler._active_request_ids = lambda: ["req-1"]

    result = OmniScheduler._admin_update_weights_from_distributed(
        scheduler,
        {
            "names": ["w.0"],
            "dtypes": ["bfloat16"],
            "shapes": [[2, 2]],
            "flush_cache": False,
            "abort_all_requests": False,
        },
    )

    assert result["success"] is False
    assert "active requests are present" in result["message"]
    assert result["data"]["active_request_count"] == 1
    assert scheduler._engine_paused is False
    assert result["data"]["engine_paused"] is False
    assert update_calls == []


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

    def abort_all_requests() -> int:
        nonlocal abort_calls
        abort_calls += 1
        return 1

    scheduler = object.__new__(OmniScheduler)
    scheduler.model_worker = SimpleNamespace(
        update_weights_from_distributed=update_weights_from_distributed
    )
    scheduler._admin_lock = threading.Lock()
    scheduler._engine_paused = False
    scheduler._last_pause_mode = None
    scheduler._async_pending = None
    scheduler._resolve_pending_async = lambda: None
    scheduler._active_request_ids = lambda: ["req-1"]
    scheduler._abort_all_requests = abort_all_requests
    scheduler.flush_cache = flush_cache
    scheduler._empty_torch_cache = empty_torch_cache

    payload = {
        "names": ["w.0"],
        "dtypes": ["bfloat16"],
        "shapes": [[2, 2]],
        "group_name": "talker_group",
        "abort_all_requests": True,
        "torch_empty_cache": True,
    }
    result = OmniScheduler._admin_update_weights_from_distributed(scheduler, payload)

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
    scheduler._admin_lock = threading.Lock()
    scheduler._engine_paused = False
    scheduler._last_pause_mode = None
    scheduler._async_pending = None
    scheduler._resolve_pending_async = lambda: None
    scheduler._active_request_ids = lambda: []

    result = OmniScheduler._admin_update_weights_from_distributed(
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
    assert scheduler._engine_paused is True


def test_coordinator_admin_waits_for_all_stage_results() -> None:
    async def _run() -> None:
        coordinator = Coordinator(
            "inproc://complete",
            "inproc://abort",
            entry_stage="preprocess",
        )
        control_plane = RecordingCoordinatorControlPlane()
        coordinator.control_plane = control_plane
        coordinator._running = True
        coordinator.register_stage("decode", "inproc://decode")
        coordinator.register_stage("vocoder", "inproc://vocoder")

        task = asyncio.create_task(
            coordinator.admin("model_info", {"detail": True}, timeout_s=1)
        )
        while len(control_plane.submitted) < 2:
            await asyncio.sleep(0)

        for stage, _, msg in control_plane.submitted:
            assert isinstance(msg, AdminMessage)
            coordinator._handle_admin_result(
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

    asyncio.run(_run())


def test_stage_admin_dispatches_to_scheduler() -> None:
    async def _run() -> None:
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

        await stage._on_admin(
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

    asyncio.run(_run())


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

    _profile_start(
        stage,
        ProfilerStartMessage(
            run_id="run-1",
            trace_path_template="/tmp/{run_id}/{stage}/trace",
        ),
    )
    _profile_stop(stage, ProfilerStopMessage(run_id="run-1"))

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
    _patch_direct_torch_profiler(monkeypatch, direct_calls)

    scheduler = ProfilerControlScheduler()
    stage = _profiler_stage(
        scheduler, name="vocoder", owner=False, process_has_owner=False
    )

    _profile_start(
        stage,
        ProfilerStartMessage(run_id="run-1", trace_path_template="/tmp/{stage}/trace"),
    )
    _profile_stop(stage, ProfilerStopMessage(run_id="run-1"))

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
    _patch_direct_torch_profiler(monkeypatch, direct_calls)

    scheduler = ProfilerControlScheduler()
    stage = _profiler_stage(
        scheduler, name="vocoder", owner=False, process_has_owner=False
    )

    _profile_start(
        stage,
        ProfilerStartMessage(run_id="run-1", trace_path_template="/tmp/{stage}/trace"),
    )
    _profile_start(
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
    """MOSS-TTS-Local runs ``preprocessing`` and the owning ``tts_engine`` in
    one process. If the sibling handled the broadcast directly it would create
    the singleton on the control thread and the owner's start would then no-op
    on it -- a trace with CUDA activity but no CPU operators, the exact defect
    this mode exists to fix."""
    monkeypatch.setenv("SGLANG_TORCH_PROFILER_SCHEDULER_THREAD", "1")
    direct_calls: list[tuple] = []
    _patch_direct_torch_profiler(monkeypatch, direct_calls)

    scheduler = ProfilerControlScheduler()
    sibling = _profiler_stage(
        scheduler, name="preprocessing", owner=False, process_has_owner=True
    )
    owner = _profiler_stage(
        scheduler, name="tts_engine", owner=True, process_has_owner=True
    )

    # The sibling is handed the broadcast first: worst-case ordering.
    msg = ProfilerStartMessage(run_id="run-1", trace_path_template="/tmp/{stage}/trace")
    _profile_start(sibling, msg)
    _profile_start(owner, msg)
    _profile_stop(sibling, ProfilerStopMessage(run_id="run-1"))
    _profile_stop(owner, ProfilerStopMessage(run_id="run-1"))

    assert direct_calls == []
    assert scheduler.profiler_calls == [
        ("start", f"/tmp/tts_engine/trace_pid{os.getpid()}", "run-1"),
        ("stop", "run-1"),
    ]


def test_owner_refuses_to_profile_before_the_scheduler_thread_is_ready(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``Stage.start`` does not wait for the scheduler thread to publish its id,
    so a start can arrive early. Falling back to a direct start would silently
    profile on the wrong thread; refusing is the honest outcome."""
    monkeypatch.setenv("SGLANG_TORCH_PROFILER_SCHEDULER_THREAD", "1")
    direct_calls: list[tuple] = []
    _patch_direct_torch_profiler(monkeypatch, direct_calls)

    scheduler = ProfilerControlScheduler(thread_ready=False)
    owner = _profiler_stage(
        scheduler, name="tts_engine", owner=True, process_has_owner=True
    )

    _profile_start(
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
    _patch_direct_torch_profiler(monkeypatch, direct_calls)

    scheduler = ProfilerControlScheduler(succeed=False)
    owner = _profiler_stage(
        scheduler, name="tts_engine", owner=True, process_has_owner=True
    )

    _profile_start(
        owner,
        ProfilerStartMessage(run_id="run-1", trace_path_template="/tmp/{stage}/trace"),
    )
    _profile_stop(owner, ProfilerStopMessage(run_id="run-1"))

    assert [name for name, *_ in scheduler.profiler_calls] == ["start", "stop"]
    assert direct_calls == []


def test_scheduler_thread_mode_off_keeps_legacy_behavior(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without the env flag an owner stage behaves exactly as before the PR."""
    monkeypatch.delenv("SGLANG_TORCH_PROFILER_SCHEDULER_THREAD", raising=False)
    direct_calls: list[tuple] = []
    _patch_direct_torch_profiler(monkeypatch, direct_calls)

    scheduler = ProfilerControlScheduler()
    owner = _profiler_stage(
        scheduler, name="tts_engine", owner=True, process_has_owner=True
    )

    _profile_start(
        owner,
        ProfilerStartMessage(run_id="run-1", trace_path_template="/tmp/{stage}/trace"),
    )
    _profile_stop(owner, ProfilerStopMessage(run_id="run-1"))

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
    _patch_direct_torch_profiler(monkeypatch, direct_calls)

    scheduler = ProfilerControlScheduler()
    owner = _profiler_stage(
        scheduler, name="tts_engine", owner=True, process_has_owner=True
    )

    _profile_start(
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
    scheduler._admin_queue = queue.Queue()
    scheduler._scheduler_thread_id = None
    scheduler._running = True

    executed_on: list[tuple[str, int]] = []

    class _RecordingTorchProfiler:
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

    def _scheduler_loop() -> None:
        scheduler._scheduler_thread_id = threading.get_ident()
        ready.set()
        while not stop_event.is_set():
            scheduler._process_admin_requests()
            time.sleep(0.001)

    thread = threading.Thread(target=_scheduler_loop, daemon=True)
    thread.start()
    ready.wait(timeout=5.0)

    with mock.patch.dict(
        sys.modules,
        {
            "sglang_omni.profiler.torch_profiler": SimpleNamespace(
                TorchProfiler=_RecordingTorchProfiler
            )
        },
    ):
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
        assert thread_id == scheduler._scheduler_thread_id, name
        assert thread_id != caller_thread_id, name


def test_scheduler_admin_profiler_failure_surfaces_to_the_caller() -> None:
    """Exceptions raised on the scheduler thread come back as admin errors so
    Stage can turn them into a RuntimeError instead of hanging."""
    from sglang_omni.scheduling.omni_scheduler import OmniScheduler

    scheduler = OmniScheduler.__new__(OmniScheduler)
    scheduler._admin_queue = queue.Queue()
    scheduler._scheduler_thread_id = None
    scheduler._running = True

    class _ExplodingTorchProfiler:
        @staticmethod
        def start(template: str, run_id: str | None = None) -> str:
            raise RuntimeError("kineto unavailable")

    stop_event = threading.Event()
    ready = threading.Event()

    def _scheduler_loop() -> None:
        scheduler._scheduler_thread_id = threading.get_ident()
        ready.set()
        while not stop_event.is_set():
            scheduler._process_admin_requests()
            time.sleep(0.001)

    thread = threading.Thread(target=_scheduler_loop, daemon=True)
    thread.start()
    ready.wait(timeout=5.0)

    with mock.patch.dict(
        sys.modules,
        {
            "sglang_omni.profiler.torch_profiler": SimpleNamespace(
                TorchProfiler=_ExplodingTorchProfiler
            )
        },
    ):
        try:
            response = scheduler.start_torch_profiler("/tmp/trace", "run-1")
        finally:
            stop_event.set()
            thread.join(timeout=5.0)

    assert response["success"] is False
    assert "kineto unavailable" in response["error"]


def test_scheduler_thread_readiness_reports_when_admin_would_run_inline() -> None:
    """``_should_enqueue_admin`` silently runs inline until the scheduler thread
    publishes its id. The profiler owner needs to distinguish those states."""
    from sglang_omni.scheduling.omni_scheduler import OmniScheduler

    scheduler = OmniScheduler.__new__(OmniScheduler)
    scheduler._admin_queue = queue.Queue()
    scheduler._scheduler_thread_id = None
    scheduler._running = False

    assert scheduler.wait_until_scheduler_thread_ready(0.05) is False

    scheduler._running = True
    assert scheduler.wait_until_scheduler_thread_ready(0.05) is False

    def _publish() -> None:
        time.sleep(0.02)
        scheduler._scheduler_thread_id = threading.get_ident()

    thread = threading.Thread(target=_publish, daemon=True)
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
    _patch_direct_torch_profiler(monkeypatch, direct_calls)

    scheduler = ProfilerControlScheduler(thread_ready=False)
    owner = _profiler_stage(
        scheduler, name="tts_engine", owner=True, process_has_owner=True
    )

    _profile_stop(owner, ProfilerStopMessage(run_id="run-1"))

    assert direct_calls == []
    assert scheduler.profiler_calls == []


def test_profiler_admin_never_runs_inline_after_the_loop_exits() -> None:
    """``OmniScheduler.start`` clears ``_scheduler_thread_id`` in its ``finally``
    but leaves ``_running`` set until ``stop`` runs, so an exited scheduler loop
    leaves a durable window where ``_should_enqueue_admin`` reports inline. A
    readiness check on the Stage side cannot close it -- the loop can exit
    between that check and the handoff -- so the profiler path must re-check and
    fail rather than run TorchProfiler on the caller's thread."""
    from sglang_omni.scheduling.omni_scheduler import OmniScheduler

    scheduler = OmniScheduler.__new__(OmniScheduler)
    scheduler._admin_queue = queue.Queue()
    # Exactly the post-loop state: id cleared, _running not yet cleared.
    scheduler._running = True
    scheduler._scheduler_thread_id = None

    executed_on: list[int] = []

    class _RecordingTorchProfiler:
        @staticmethod
        def start(template: str, run_id: str | None = None) -> str:
            executed_on.append(threading.get_ident())
            return "trace"

        @staticmethod
        def stop(*, run_id: str | None = None) -> dict:
            executed_on.append(threading.get_ident())
            return {}

    with mock.patch.dict(
        sys.modules,
        {
            "sglang_omni.profiler.torch_profiler": SimpleNamespace(
                TorchProfiler=_RecordingTorchProfiler
            )
        },
    ):
        start_response = scheduler.start_torch_profiler("/tmp/trace", "run-1")
        stop_response = scheduler.stop_torch_profiler("run-1")

    assert start_response["success"] is False
    assert stop_response["success"] is False
    # The whole point: nothing ran on this thread.
    assert executed_on == []
    assert scheduler._admin_queue.empty()


def test_profiler_admin_runs_inline_when_already_on_the_scheduler_thread() -> None:
    """Being the scheduler thread is the goal state, not a fallback."""
    from sglang_omni.scheduling.omni_scheduler import OmniScheduler

    scheduler = OmniScheduler.__new__(OmniScheduler)
    scheduler._admin_queue = queue.Queue()
    scheduler._running = True
    scheduler._scheduler_thread_id = threading.get_ident()

    executed_on: list[int] = []

    class _RecordingTorchProfiler:
        @staticmethod
        def start(template: str, run_id: str | None = None) -> str:
            executed_on.append(threading.get_ident())
            return "trace"

    with mock.patch.dict(
        sys.modules,
        {
            "sglang_omni.profiler.torch_profiler": SimpleNamespace(
                TorchProfiler=_RecordingTorchProfiler
            )
        },
    ):
        response = scheduler.start_torch_profiler("/tmp/trace", "run-1")

    assert response["success"] is True
    assert executed_on == [threading.get_ident()]


def test_expired_profiler_start_is_dropped_when_drained_late() -> None:
    """``_enqueue_admin`` reports a timeout but leaves the request queued. A
    scheduler that was busy past the deadline must drop it: the caller was told
    the start failed and will never issue a stop, so running it late leaves an
    orphan profiler taxing the serving path."""
    from sglang_omni.scheduling.omni_scheduler import OmniScheduler

    scheduler = OmniScheduler.__new__(OmniScheduler)
    scheduler._admin_queue = queue.Queue()
    scheduler._running = True
    # A live scheduler thread that is too busy to drain: forces the enqueue path.
    scheduler._scheduler_thread_id = threading.get_ident() + 1

    started: list[str] = []

    class _RecordingTorchProfiler:
        @staticmethod
        def start(template: str, run_id: str | None = None) -> str:
            started.append(template)
            return "trace"

    with mock.patch.object(
        OmniScheduler, "_enqueue_admin", autospec=True
    ) as enqueue_admin:

        def _queue_and_time_out(self, action, payload):
            # Mirror the real body, minus the blocking wait the caller gave up on.
            queued = dict(payload)
            queued.pop("_admin_timeout_s", None)
            self._admin_queue.put((action, queued, queue.Queue(maxsize=1)))
            return {"success": False, "error": "admin operation timed out"}

        enqueue_admin.side_effect = _queue_and_time_out
        response = scheduler.start_torch_profiler("/tmp/trace", "run-1")

    assert response["success"] is False
    assert scheduler._admin_queue.qsize() == 1

    # The scheduler frees up after the deadline and drains what it missed.
    action, payload, _ = scheduler._admin_queue.get_nowait()
    payload["_deadline"] = time.monotonic() - 1.0
    with mock.patch.dict(
        sys.modules,
        {
            "sglang_omni.profiler.torch_profiler": SimpleNamespace(
                TorchProfiler=_RecordingTorchProfiler
            )
        },
    ):
        late = scheduler._run_admin_action(action, payload)

    assert late["success"] is False
    assert "expired" in late["error"]
    assert started == []


def test_live_profiler_request_carries_a_deadline_but_still_runs() -> None:
    """The deadline must not reject requests that arrive in time."""
    from sglang_omni.scheduling.omni_scheduler import OmniScheduler

    scheduler = OmniScheduler.__new__(OmniScheduler)
    started: list[str] = []

    class _RecordingTorchProfiler:
        @staticmethod
        def start(template: str, run_id: str | None = None) -> str:
            started.append(template)
            return "trace"

    with mock.patch.dict(
        sys.modules,
        {
            "sglang_omni.profiler.torch_profiler": SimpleNamespace(
                TorchProfiler=_RecordingTorchProfiler
            )
        },
    ):
        response = scheduler._run_admin_action(
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
    handler awaited them, `run` would not reach its next `control_plane.recv()`
    and this stage's submits and acks would queue behind a profiler request."""
    monkeypatch.setenv("SGLANG_TORCH_PROFILER_SCHEDULER_THREAD", "1")
    direct_calls: list[tuple] = []
    _patch_direct_torch_profiler(monkeypatch, direct_calls)

    release = threading.Event()

    class _BlockingScheduler(ProfilerControlScheduler):
        def wait_until_scheduler_thread_ready(self, timeout_s: float) -> bool:
            release.wait(timeout=5.0)
            return True

    owner = _profiler_stage(
        _BlockingScheduler(), name="tts_engine", owner=True, process_has_owner=True
    )
    handled: list[str] = []

    async def _run() -> None:
        # Exactly what Stage.run does: dispatch, then take the next message.
        await owner._handle_message(
            ProfilerStartMessage(
                run_id="run-1", trace_path_template="/tmp/{stage}/trace"
            )
        )
        handled.append("profiler_start")

        for index in range(3):
            await owner._handle_message(
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
        await asyncio.wait_for(owner._wait_for_profiler_ops(), timeout=5.0)
        await owner.stop()

    asyncio.run(_run())

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
    _patch_direct_torch_profiler(monkeypatch, direct_calls)

    class _SlowStartScheduler(ProfilerControlScheduler):
        def start_torch_profiler(self, trace_path_template: str, run_id: str | None):
            time.sleep(0.05)
            return super().start_torch_profiler(trace_path_template, run_id)

    owner = _profiler_stage(
        _SlowStartScheduler(), name="tts_engine", owner=True, process_has_owner=True
    )

    async def _run() -> None:
        owner._on_profiler_start(
            ProfilerStartMessage(
                run_id="run-1", trace_path_template="/tmp/{stage}/trace"
            )
        )
        owner._on_profiler_stop(ProfilerStopMessage(run_id="run-1"))
        await asyncio.wait_for(owner._wait_for_profiler_ops(), timeout=5.0)
        await owner.stop()

    asyncio.run(_run())

    assert [name for name, *_ in owner.scheduler.profiler_calls] == ["start", "stop"]


def test_shutdown_drains_a_queued_profiler_stop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`/stop_profile` immediately followed by shutdown is an ordinary operator
    sequence. The queued stop is what exports the trace, so shutdown has to let
    it run instead of cancelling it away."""
    monkeypatch.setenv("SGLANG_TORCH_PROFILER_SCHEDULER_THREAD", "1")
    direct_calls: list[tuple] = []
    _patch_direct_torch_profiler(monkeypatch, direct_calls)

    scheduler = ProfilerControlScheduler()
    owner = _profiler_stage(
        scheduler, name="tts_engine", owner=True, process_has_owner=True
    )

    async def _run() -> None:
        owner._on_profiler_stop(ProfilerStopMessage(run_id="run-1"))
        # No drain here: shutdown lands while the stop is still queued.
        await owner.stop()

    asyncio.run(_run())

    assert scheduler.profiler_calls == [("stop", "run-1")]


def test_shutdown_is_bounded_when_a_profiler_operation_is_stuck(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Shutdown must not hang on a scheduler that is already gone, and must not
    accept new profiler work once it has begun."""
    monkeypatch.setenv("SGLANG_TORCH_PROFILER_SCHEDULER_THREAD", "1")
    monkeypatch.setattr(runtime_mod, "_PROFILER_SHUTDOWN_DRAIN_TIMEOUT_S", 0.05)
    direct_calls: list[tuple] = []
    _patch_direct_torch_profiler(monkeypatch, direct_calls)

    release = threading.Event()

    class _StuckScheduler(ProfilerControlScheduler):
        def wait_until_scheduler_thread_ready(self, timeout_s: float) -> bool:
            release.wait(timeout=5.0)
            return True

    scheduler = _StuckScheduler()
    owner = _profiler_stage(
        scheduler, name="tts_engine", owner=True, process_has_owner=True
    )

    async def _run() -> float:
        owner._on_profiler_start(
            ProfilerStartMessage(
                run_id="run-1", trace_path_template="/tmp/{stage}/trace"
            )
        )
        await asyncio.sleep(0)  # let the worker reach the stuck call
        started = time.monotonic()
        await owner.stop()
        elapsed = time.monotonic() - started

        # Shutdown has begun: further profiler work is refused outright.
        owner._on_profiler_stop(ProfilerStopMessage(run_id="run-1"))
        assert owner._profiler_op_queue is not None
        assert owner._profiler_op_queue.empty()
        return elapsed

    try:
        elapsed = asyncio.run(_run())
    finally:
        release.set()

    assert elapsed < 2.0
    assert direct_calls == []
    # The executor call outlived shutdown -- Task.cancel cannot reach it -- but
    # it found the lane abandoned and never reached the scheduler.
    assert [name for name, *_ in scheduler.profiler_calls] == []
