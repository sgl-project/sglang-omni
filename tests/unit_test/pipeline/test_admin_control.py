# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import queue
import threading
import time
from collections import deque
from concurrent.futures import Future
from functools import partial
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from sglang_omni.pipeline.control_plane import deserialize_message, serialize_message
from sglang_omni.pipeline.coordinator import Coordinator
from sglang_omni.pipeline.stage.runtime import Stage
from sglang_omni.proto import (
    AdminMessage,
    AdminOperation,
    AdminResult,
    AdminResultMessage,
    parse_message,
)
from sglang_omni.proto.admin import ADMIN_MEMORY_CONTROL
from sglang_omni.scheduling.memory_control import WorkerMemoryControl
from sglang_omni.scheduling.types import DeferredAdmission
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


@pytest.fixture
def stage() -> Stage:
    return Stage(
        name="decoder",
        role="single",
        get_next=lambda request_id, output: None,
        gpu_id=None,
        endpoints={},
        control_plane=RecordingStageControlPlane(),
        relay=FakeRelay(),
        scheduler=AdminScheduler(),
    )


def make_memory_scheduler():
    from sglang_omni.scheduling.omni_scheduler import OmniScheduler

    scheduler = object.__new__(OmniScheduler)
    scheduler.tp_size = 1
    scheduler.device = "cuda"
    scheduler.tp_worker = SimpleNamespace(
        model_runner=SimpleNamespace(
            server_args=SimpleNamespace(
                enable_memory_saver=True, weight_cache_mode="off"
            )
        )
    )
    scheduler._admin_lock = threading.Lock()
    scheduler._engine_paused = False
    scheduler._last_pause_mode = None
    scheduler.memory_manager = Mock()
    scheduler.memory_allocator = Mock()
    scheduler.released_memory_tags = set()
    scheduler.memory_transition_active = False
    scheduler.memory_transition_failed = False
    scheduler.paused_before_memory_release = False
    scheduler.request_build_idle_callback = None
    scheduler._deferred_request_payloads = {}
    scheduler._pending_stream_ingress = {}
    scheduler.resolve_pending_async = Mock()
    scheduler.active_request_ids = lambda: []
    scheduler.is_fully_idle = lambda: True
    return scheduler


@pytest.fixture
def memory_scheduler(monkeypatch):
    monkeypatch.setattr("torch.get_device_module", lambda *args: Mock())
    return make_memory_scheduler()


@pytest.fixture
def memory_control(memory_scheduler):
    return WorkerMemoryControl(
        {"asr": memory_scheduler.admin_memory_phase}, worker="audio"
    )


@pytest.fixture
def worker_memory(memory_scheduler):
    schedulers = {"asr": memory_scheduler, "decoder": make_memory_scheduler()}
    controller = WorkerMemoryControl(
        {name: scheduler.admin_memory_phase for name, scheduler in schedulers.items()},
        worker="audio",
    )
    return controller, schedulers


def test_worker_memory_freezes_all_stages_before_one_allocator_release(
    worker_memory,
) -> None:
    controller, schedulers = worker_memory
    allocator = schedulers["asr"].memory_allocator
    stashed = []

    for name, scheduler in schedulers.items():

        def stash(req, stage=name):
            assert all(peer.memory_transition_active for peer in schedulers.values())
            assert allocator.pause.call_count == 0
            stashed.append(stage)

        scheduler.memory_manager.release_memory_occupation.side_effect = stash

    result = controller.run("release_memory_occupation", {})
    assert stashed == ["asr", "decoder"]
    assert result["data"]["worker"] == "audio"
    assert result["data"]["affected_stages"] == ["asr", "decoder"]
    assert allocator.pause.call_count == 3
    schedulers["decoder"].memory_allocator.pause.assert_not_called()
    for scheduler in schedulers.values():
        assert scheduler.released_memory_tags == {"weights", "kv_cache", "cuda_graph"}
        assert scheduler._engine_paused
    controller.run("release_memory_occupation", {})
    assert allocator.pause.call_count == 3


@pytest.mark.parametrize("worker_scoped", [False, True])
def test_memory_partial_resume_and_idempotency(
    memory_scheduler, memory_control, worker_memory, worker_scoped
) -> None:
    controller, schedulers = (
        worker_memory if worker_scoped else (memory_control, {"asr": memory_scheduler})
    )
    if worker_scoped:
        schedulers["decoder"]._engine_paused = True
    controller.run("resume_memory_occupation", {})
    for name, scheduler in schedulers.items():
        assert scheduler._engine_paused == (name == "decoder")
    controller.run("release_memory_occupation", {})
    controller.run("release_memory_occupation", {})
    controller.run("resume_memory_occupation", {"tags": ["weights"]})
    for scheduler in schedulers.values():
        scheduler.memory_manager.release_memory_occupation.assert_called_once()
        assert scheduler._engine_paused
        assert scheduler.released_memory_tags == {"kv_cache", "cuda_graph"}
        with pytest.raises(RuntimeError, match="resume memory first"):
            scheduler.admin_continue_generation({})
    controller.run("resume_memory_occupation", {})
    result = controller.run("resume_memory_occupation", {})
    assert result["data"]["engine_paused"] == worker_scoped
    for name, scheduler in schedulers.items():
        assert scheduler._engine_paused == (name == "decoder")
        assert not scheduler.released_memory_tags
        assert scheduler.memory_manager.resume_memory_occupation.call_count == 2
    assert schedulers["asr"].memory_allocator.resume.call_count == 3
    if worker_scoped:
        schedulers["decoder"].memory_allocator.resume.assert_not_called()


def test_worker_busy_peer_cancels_preparation_without_releasing(worker_memory) -> None:
    controller, schedulers = worker_memory
    schedulers["decoder"].active_request_ids = lambda: ["busy"]
    schedulers["decoder"]._engine_paused = True
    with pytest.raises(RuntimeError, match="active requests"):
        controller.run("release_memory_occupation", {})
    assert all(
        not scheduler.memory_transition_active for scheduler in schedulers.values()
    )
    assert not schedulers["asr"]._engine_paused
    assert schedulers["decoder"]._engine_paused
    schedulers["asr"].memory_allocator.pause.assert_not_called()


@pytest.mark.parametrize("paused", [False, True])
def test_worker_cancels_prepare_that_runs_after_timeout(worker_memory, paused) -> None:
    controller, schedulers = worker_memory
    delayed = schedulers["decoder"]
    delayed._engine_paused = paused
    delayed._admin_queue = queue.Queue()
    controller.handlers["decoder"] = lambda payload: delayed.enqueue_admin(
        ADMIN_MEMORY_CONTROL, {**payload, "_admin_timeout_s": 0}
    )
    with pytest.raises(RuntimeError, match="timed out"):
        controller.run("release_memory_occupation", {})

    delayed.process_admin_requests()
    assert not schedulers["asr"]._engine_paused
    assert delayed._engine_paused == paused
    for scheduler in schedulers.values():
        assert not scheduler.memory_transition_active
        assert not scheduler.memory_transition_failed
        scheduler.memory_allocator.pause.assert_not_called()


def test_stage_memory_request_controls_colocated_stages(stage, worker_memory) -> None:
    controller, schedulers = worker_memory
    stage.memory_control = controller
    result = asyncio.run(
        stage.run_admin_operation(
            AdminOperation(
                op_id="release",
                action="release_memory_occupation",
                payload={"tags": ["weights"]},
            )
        )
    )
    assert result.success
    assert result.data["affected_stages"] == ["asr", "decoder"]
    assert all(
        scheduler.released_memory_tags == {"weights"}
        for scheduler in schedulers.values()
    )


def test_memory_preparation_blocks_other_admin_actions(memory_scheduler) -> None:
    payload = {"action": "release_memory_occupation", "tags": ["weights"]}
    memory_scheduler.admin_memory_phase({**payload, "phase": "prepare"})
    for action in (
        "continue_generation",
        "update_weights_from_disk",
        "pause_generation",
    ):
        with pytest.raises(RuntimeError, match="resume memory first"):
            memory_scheduler.run_admin_action(action, {})
    memory_scheduler.admin_memory_phase({**payload, "phase": "cancel"})
    assert not memory_scheduler._engine_paused


@pytest.mark.parametrize("failure", ["stash", "allocator", "reload"])
def test_worker_failure_keeps_every_stage_paused(worker_memory, failure) -> None:
    controller, schedulers = worker_memory
    if failure == "reload":
        controller.run("release_memory_occupation", {})
        schedulers["decoder"].memory_manager.resume_memory_occupation.side_effect = (
            RuntimeError("failed")
        )
        action = "resume_memory_occupation"
    elif failure == "stash":
        schedulers["decoder"].memory_manager.release_memory_occupation.side_effect = (
            RuntimeError("failed")
        )
        action = "release_memory_occupation"
    else:
        schedulers["asr"].memory_allocator.pause.side_effect = RuntimeError("failed")
        action = "release_memory_occupation"
    with pytest.raises(RuntimeError, match="failed"):
        controller.run(action, {})
    assert all(scheduler._engine_paused for scheduler in schedulers.values())
    assert all(scheduler.memory_transition_failed for scheduler in schedulers.values())
    for scheduler in schedulers.values():
        with pytest.raises(RuntimeError, match="resume memory first"):
            scheduler.run_admin_action("continue_generation", {})


def test_worker_memory_runs_phases_on_each_scheduler_thread(worker_memory) -> None:
    _, schedulers = worker_memory
    stop = threading.Event()
    threads = []
    ready_events = []
    phase_threads = []
    for name, scheduler in schedulers.items():
        scheduler._running = True
        scheduler._admin_queue = queue.Queue()
        scheduler._scheduler_thread_id = None
        ready = threading.Event()

        def record_phase(payload, stage=name, handler=scheduler.admin_memory_phase):
            phase_threads.append((stage, threading.get_ident()))
            return handler(payload)

        scheduler.admin_memory_phase = record_phase

        def run(peer=scheduler, event=ready):
            peer._scheduler_thread_id = threading.get_ident()
            event.set()
            while not stop.is_set():
                peer.process_admin_requests()
                time.sleep(0.001)

        thread = threading.Thread(target=run)
        threads.append(thread)
        ready_events.append(ready)
        thread.start()
    try:
        assert all(event.wait(1) for event in ready_events)
        controller = WorkerMemoryControl(
            {
                name: partial(scheduler.admin, ADMIN_MEMORY_CONTROL)
                for name, scheduler in schedulers.items()
            },
            worker="audio",
        )
        assert controller.run("release_memory_occupation", {})["success"]
        assert controller.run("resume_memory_occupation", {})["success"]
    finally:
        stop.set()
        for thread in threads:
            thread.join(1)
    assert all(not thread.is_alive() for thread in threads)
    assert {stage for stage, _ in phase_threads} == set(schedulers)
    assert all(
        thread_id == schedulers[stage]._scheduler_thread_id
        for stage, thread_id in phase_threads
    )


@pytest.mark.parametrize("pending_kind", ["builder", "admission", "builder_admission"])
@pytest.mark.parametrize("evict_aborts", [False, True])
def test_memory_waits_for_aborted_request_work(
    memory_scheduler, memory_control, pending_kind, evict_aborts, monkeypatch
) -> None:
    scheduler = memory_scheduler
    del scheduler.active_request_ids
    scheduler._request_admission_lock = threading.RLock()
    scheduler._aborted_request_ids = set()
    scheduler._aborted_request_id_order = deque()
    scheduler._backlogged_request_build_payloads = deque()
    scheduler._pending_request_builds = {}
    scheduler._pending_request_admissions = {}
    scheduler._dirty_deferred_request_ids = set()
    scheduler._first_emit_done = set()
    scheduler._prefill_start_done = set()
    scheduler._prefill_end_done = set()
    scheduler._abort_callback = None
    scheduler.waiting_queue = []
    scheduler.running_batch = scheduler.cur_batch = scheduler.last_batch = None
    scheduler._async_pending = None
    scheduler._idle_wait_message = None
    scheduler.inbox = queue.Queue()
    scheduler.enqueue_built_request = Mock()
    payload = SimpleNamespace(request_id="cancelled")
    builder = Future()
    builder.set_running_or_notify_cancel()
    ready = Future()
    deferred = DeferredAdmission(value=object(), ready=ready)
    if pending_kind == "admission":
        scheduler._pending_request_admissions[payload.request_id] = (
            payload,
            False,
            deferred,
        )
    else:
        scheduler._pending_request_builds[payload.request_id] = (
            payload,
            False,
            builder,
        )

    scheduler.abort(payload.request_id)
    if evict_aborts:
        from sglang_omni.scheduling import omni_scheduler

        monkeypatch.setattr(omni_scheduler, "_ABORTED_REQUEST_ID_LIMIT", 2)
        monkeypatch.setattr(omni_scheduler, "_ABORTED_REQUEST_ID_RETAINED", 1)
        for index in range(3):
            scheduler.abort(f"other-{index}")
    with pytest.raises(RuntimeError, match="active requests"):
        memory_control.run("release_memory_occupation", {})
    scheduler.memory_allocator.pause.assert_not_called()

    if pending_kind == "builder_admission":
        builder.set_result(deferred)
        scheduler.drain_request_build_results()
        with pytest.raises(RuntimeError, match="active requests"):
            memory_control.run("release_memory_occupation", {})
    elif pending_kind == "builder":
        builder.set_result(object())
    ready.set_result(None)
    scheduler.drain_request_build_results()
    scheduler.drain_request_admission_results()
    assert memory_control.run("release_memory_occupation", {})["success"]
    scheduler.enqueue_built_request.assert_not_called()


@pytest.mark.parametrize("tags", [["unknown"], "weights", [1]])
def test_memory_rejects_invalid_tags(memory_scheduler, memory_control, tags) -> None:
    with pytest.raises(ValueError, match="tags"):
        memory_control.run("release_memory_occupation", {"tags": tags})
    memory_scheduler.memory_manager.release_memory_occupation.assert_not_called()


@pytest.mark.parametrize(
    "field, value",
    [
        ("active_request_ids", lambda: ["request"]),
        ("is_fully_idle", lambda: False),
        ("_deferred_request_payloads", {"stream": object()}),
        ("_pending_stream_ingress", {"stream": object()}),
    ],
)
def test_memory_requires_idle_stage(
    memory_scheduler, memory_control, field, value
) -> None:
    setattr(memory_scheduler, field, value)
    with pytest.raises(RuntimeError, match="active requests"):
        memory_control.run("release_memory_occupation", {})
    assert not memory_scheduler._engine_paused
    memory_scheduler.memory_manager.release_memory_occupation.assert_not_called()


def test_memory_requires_enabled_allocator(memory_scheduler, memory_control) -> None:
    memory_scheduler.tp_worker.model_runner.server_args.enable_memory_saver = False
    with pytest.raises(RuntimeError, match="enable_memory_saver"):
        memory_control.run("release_memory_occupation", {})


def test_memory_waits_for_timed_out_encoder_work(
    memory_scheduler, memory_control
) -> None:
    from sglang_omni.models.moss_transcribe_diarize.encoder_service import (
        BatchedAudioEncoderService,
    )

    service = object.__new__(BatchedAudioEncoderService)
    service._queue = queue.Queue()
    service._worker_state_lock = threading.Lock()
    service._worker_error = None
    service.pending_futures = set()
    service.ENCODE_TIMEOUT_S = 0
    memory_scheduler.request_build_idle_callback = service.is_idle
    with pytest.raises(TimeoutError):
        service.encode_item(SimpleNamespace())
    entry = service._queue.get_nowait()
    entry.future.cancel()
    with pytest.raises(RuntimeError, match="active requests"):
        memory_control.run("release_memory_occupation", {})
    memory_scheduler.memory_allocator.pause.assert_not_called()

    service.set_result(entry, object())
    assert memory_control.run("release_memory_occupation", {})["success"]


def test_memory_rejects_unsynchronized_tp_control(
    memory_scheduler, memory_control
) -> None:
    memory_scheduler.tp_size = 2
    with pytest.raises(RuntimeError, match="tp_size=1"):
        memory_control.run("release_memory_occupation", {})
    memory_scheduler.memory_manager.release_memory_occupation.assert_not_called()


@pytest.mark.parametrize("stage_count", [1, 2])
def test_memory_bridges_upstream_allocator_and_restores_buffers(
    memory_scheduler, monkeypatch, stage_count
) -> None:
    import torch

    schedulers = [memory_scheduler] + [
        make_memory_scheduler() for _ in range(stage_count - 1)
    ]
    for index, scheduler in enumerate(schedulers):
        model = torch.nn.Module()
        model.register_buffer("scale", torch.tensor([3.0 + index]))
        scheduler.tp_worker.model_runner.model = model
        scheduler.tp_cpu_group = None
        scheduler.memory_manager = None
        scheduler.flush_cache = Mock(return_value=True)
    adapter = schedulers[0].memory_allocator

    def lose_buffers(tag):
        if tag == "weights":
            for scheduler in schedulers:
                scheduler.tp_worker.model_runner.model.scale.zero_()

    adapter.pause.side_effect = lose_buffers
    controller = WorkerMemoryControl(
        {
            str(index): scheduler.admin_memory_phase
            for index, scheduler in enumerate(schedulers)
        },
        worker="audio",
    )
    monkeypatch.setattr(torch.distributed, "barrier", Mock())
    controller.run("release_memory_occupation", {})
    assert [call.args[0] for call in adapter.pause.call_args_list] == [
        "kv_cache",
        "weights",
        "cuda_graph",
    ]
    for scheduler in schedulers:
        scheduler.flush_cache.assert_called_once()
    controller.run("resume_memory_occupation", {})
    assert [call.args[0] for call in adapter.resume.call_args_list] == [
        "cuda_graph",
        "weights",
        "kv_cache",
    ]
    for index, scheduler in enumerate(schedulers):
        assert scheduler.tp_worker.model_runner.model.scale.item() == 3.0 + index
        assert not scheduler._engine_paused


@pytest.mark.parametrize("message_type", ["new_request", "stream_chunk", "stream_done"])
def test_sleeping_stage_rejects_requests_without_building(
    memory_scheduler, message_type
) -> None:
    from sglang_omni.scheduling.messages import IncomingMessage

    memory_scheduler.released_memory_tags = {"weights"}
    memory_scheduler._aborted_request_ids = set()
    memory_scheduler.recv_scheduler_messages = lambda: [
        IncomingMessage(request_id="new", type=message_type, data=object())
    ]
    memory_scheduler.emit_request_error = Mock()
    assert memory_scheduler.recv_requests() == []
    memory_scheduler.emit_request_error.assert_called_once()


@pytest.mark.parametrize(
    "action", ["release_memory_occupation", "resume_memory_occupation"]
)
def test_memory_rejects_pd_stages(action) -> None:
    from sglang_omni.scheduling.pd_scheduler import (
        OmniDecodeScheduler,
        OmniPrefillScheduler,
    )

    for scheduler_type in (OmniDecodeScheduler, OmniPrefillScheduler):
        scheduler = object.__new__(scheduler_type)
        with pytest.raises(RuntimeError, match="does not support PD stages"):
            scheduler.admin_memory_phase(
                {"phase": "prepare", "action": action, "tags": []}
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
        pytest.param(
            "admin_update_weights_from_tensor",
            "update_weights_from_tensor",
            {"serialized_named_tensors": None, "torch_empty_cache": True},
            id="tensor",
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
    scheduler.req_to_token_pool = SimpleNamespace(
        clear=lambda: reset_calls.append("req_pool"),
        reset_aux_cache_allocator=lambda: reset_calls.append("aux_cache"),
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


def test_omni_scheduler_distributed_update_rejects_active_requests_by_default() -> None:
    from sglang_omni.scheduling.omni_scheduler import OmniScheduler

    update_calls: list[dict] = []
    scheduler = object.__new__(OmniScheduler)
    scheduler.model_worker = SimpleNamespace(
        update_weights_from_distributed=lambda payload: update_calls.append(payload)
        or (True, "ok")
    )
    scheduler.admin_lock = threading.Lock()
    scheduler._engine_paused = False  # noqa: leading-underscore  # production name
    scheduler.last_pause_mode = None
    scheduler.async_pending = None
    scheduler.resolve_pending_async = lambda: None
    scheduler.active_request_ids = lambda: ["req-1"]

    result = OmniScheduler.admin_update_weights_from_distributed(
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
    assert (
        scheduler._engine_paused is False
    )  # noqa: leading-underscore  # production name
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


@pytest.mark.parametrize(
    "action, has_handler, error",
    [
        ("release_memory_occupation", False, "requires a supported worker"),
        ("release_memory_occupation", True, "requires a supported worker"),
        ("resume_memory_occupation", True, "requires a supported worker"),
        (ADMIN_MEMORY_CONTROL, False, "internal worker operations"),
        (ADMIN_MEMORY_CONTROL, True, "internal worker operations"),
    ],
)
def test_stage_rejects_unsupported_memory_admin(
    stage, action, has_handler, error
) -> None:
    scheduler = FakeScheduler()
    handler = Mock()
    if has_handler:
        scheduler.admin = handler
    stage.scheduler = scheduler
    result = asyncio.run(
        stage.run_admin_operation(AdminOperation(op_id="memory", action=action))
    )
    assert not result.success
    assert error in result.error
    handler.assert_not_called()
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

        assert stage.scheduler.calls == [("pause_generation", {"mode": "in_place"})]
        result_msg = stage.control_plane.completions[0]
        assert isinstance(result_msg, AdminResultMessage)
        assert result_msg.result.success is True
        assert result_msg.result.data["action"] == "pause_generation"

    asyncio.run(run())
