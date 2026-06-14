# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import queue
import threading
import time
from types import SimpleNamespace

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
    scheduler.result_queue = None
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
    scheduler.result_queue = None
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
