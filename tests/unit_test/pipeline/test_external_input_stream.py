# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import asyncio
import queue
from types import SimpleNamespace

import pytest
import torch

from sglang_omni.comm import stage_io
from sglang_omni.config import PipelineConfig, ProcessConfig
from sglang_omni.config.topology import compile_logical_processes
from sglang_omni.pipeline.coordinator import Coordinator
from sglang_omni.pipeline.replicas import expand_replica_stages
from sglang_omni.pipeline.stage import runtime as stage_runtime
from sglang_omni.pipeline.stage.runtime import Stage
from sglang_omni.proto import (
    CompleteMessage,
    DataReadyMessage,
    OmniRequest,
    StagePayload,
    SubmitMessage,
)
from tests.unit_test.fixtures.pipeline_fakes import RecordingCoordinatorControlPlane
from tests.unit_test.pipeline.helpers import stage


def _coordinator(
    *,
    max_chunks: int = 8,
    max_bytes: int = 4096,
) -> tuple[Coordinator, RecordingCoordinatorControlPlane]:
    coordinator = Coordinator(
        "inproc://complete",
        "inproc://abort",
        entry_stage="asr",
        terminal_stages=["asr"],
        max_external_input_chunks=max_chunks,
        max_external_input_bytes=max_bytes,
    )
    control_plane = RecordingCoordinatorControlPlane()
    coordinator.control_plane = control_plane
    coordinator.register_stage("asr", "inproc://asr")
    return coordinator, control_plane


def _external_scheduler(*, maxsize: int = 8):
    scheduler = SimpleNamespace(
        inbox=queue.Queue(maxsize=maxsize),
        outbox=queue.Queue(),
        aborted=[],
        supports_external_input_stream=True,
    )
    scheduler.abort = lambda request_id: scheduler.aborted.append(request_id)
    return scheduler


class _StageControlPlane:
    def __init__(self) -> None:
        self.completions = []

    async def send_complete(self, msg) -> None:
        self.completions.append(msg)


def _stage_with_control_plane(
    scheduler, *, role: str = "single", tp_size: int = 1
) -> tuple[Stage, _StageControlPlane]:
    control_plane = _StageControlPlane()
    stage_obj = Stage(
        name="asr",
        role=role,
        get_next=lambda request_id, output: None,
        gpu_id=None,
        endpoints={},
        control_plane=control_plane,
        relay=SimpleNamespace(device="cpu", cleanup=lambda request_id: None),
        scheduler=scheduler,
        tp_size=tp_size,
        is_terminal=True,
    )
    return stage_obj, control_plane


def _payload(request_id: str = "req") -> StagePayload:
    return StagePayload(
        request_id=request_id,
        request=OmniRequest(inputs=None, params={"stream": True}),
        data={"raw_inputs": None},
    )


def _chunk(request_id: str, chunk_id: int, value: int = 1) -> DataReadyMessage:
    tensor = torch.tensor([value], dtype=torch.int16)
    return DataReadyMessage(
        request_id=request_id,
        from_stage="coordinator",
        to_stage="asr",
        data_ref=stage_io.serialize_inline_stream_chunk(tensor, {"modality": "audio"}),
        chunk_id=chunk_id,
    )


def test_coordinator_stream_start_chunk_done_and_cleanup() -> None:
    async def run() -> None:
        coordinator, control_plane = _coordinator()
        events = await coordinator.start_input_stream(
            "req", OmniRequest(inputs=None, params={"stream": True})
        )

        assert control_plane.submitted[0][2].external_input_stream is True
        assert (
            await coordinator.send_input_chunk(
                "req",
                torch.tensor([1, 2], dtype=torch.int16),
                metadata={"modality": "audio"},
            )
            == 0
        )
        assert (
            await coordinator.send_input_chunk(
                "req", torch.tensor([3], dtype=torch.int16)
            )
            == 1
        )
        await coordinator.finish_input_stream("req")

        sent = control_plane.input_stream_events
        assert [message.chunk_id for _, _, message in sent] == [0, 1, None]
        assert sent[-1][2].is_done is True
        restored, metadata = stage_io.deserialize_inline_stream_chunk(
            sent[0][2].data_ref
        )
        assert restored.tolist() == [1, 2]
        assert metadata == {"modality": "audio"}

        await coordinator._handle_completion(
            CompleteMessage("req", "asr", True, result={"text": "ok"})
        )
        assert [message async for message in events][-1].result == {"text": "ok"}
        assert "req" not in coordinator._external_input_streams
        assert "req" not in coordinator._stream_queues
        assert "req" not in coordinator._completion_futures

    asyncio.run(run())


def test_coordinator_start_failure_rolls_back_every_owner() -> None:
    class FailingControlPlane(RecordingCoordinatorControlPlane):
        async def submit_to_stage(self, stage, endpoint, msg) -> None:
            del stage, endpoint, msg
            raise RuntimeError("start failed")

    async def run() -> None:
        coordinator, _ = _coordinator()
        coordinator.control_plane = FailingControlPlane()

        with pytest.raises(RuntimeError, match="start failed"):
            await coordinator.start_input_stream("req", OmniRequest(inputs=None))

        assert "req" not in coordinator._requests
        assert "req" not in coordinator._completion_futures
        assert "req" not in coordinator._stream_queues
        assert "req" not in coordinator._external_input_streams

    asyncio.run(run())


def test_external_stream_is_pinned_to_entry_replica() -> None:
    async def run() -> None:
        config = PipelineConfig(
            model_path="dummy",
            stages=[stage("asr", process="front", terminal=True)],
            processes={"front": ProcessConfig(num_replicas=2)},
        )
        logical_plan, stages = compile_logical_processes(config)
        _, replicas = expand_replica_stages(stages, logical_plan)
        coordinator = Coordinator(
            "inproc://complete",
            "inproc://abort",
            entry_stage="asr",
            terminal_stages=["asr"],
            replica_topology=replicas,
            logical_process_plan=logical_plan,
        )
        control_plane = RecordingCoordinatorControlPlane()
        coordinator.control_plane = control_plane
        coordinator.register_stage("asr@r0", "inproc://asr0")
        coordinator.register_stage("asr@r1", "inproc://asr1")

        handle = await coordinator.start_input_stream("req", OmniRequest(inputs=None))
        await coordinator.send_input_chunk("req", torch.tensor([1], dtype=torch.int16))
        await coordinator.finish_input_stream("req")

        submit_target, submit_endpoint, submit = control_plane.submitted[0]
        assert submit_target in {"asr@r0", "asr@r1"}
        assert all(
            (target, endpoint) == (submit_target, submit_endpoint)
            for target, endpoint, _ in control_plane.input_stream_events
        )
        assert all(
            message.replica_bindings == submit.replica_bindings
            for _, _, message in control_plane.input_stream_events
        )
        await coordinator.close_input_stream("req")
        await handle.aclose()

    asyncio.run(run())


def test_coordinator_enforces_chunk_and_total_byte_boundaries() -> None:
    async def run() -> None:
        coordinator, _ = _coordinator(max_chunks=1, max_bytes=4)
        handle = await coordinator.start_input_stream("req", OmniRequest(inputs=None))
        await coordinator.send_input_chunk(
            "req", torch.tensor([1, 2], dtype=torch.int16)
        )
        with pytest.raises(ValueError, match="max_external_input_chunks=1"):
            await coordinator.send_input_chunk(
                "req", torch.tensor([3], dtype=torch.int16)
            )
        await coordinator.close_input_stream("req")
        await handle.aclose()

        coordinator, _ = _coordinator(max_chunks=4, max_bytes=3)
        handle = await coordinator.start_input_stream("req", OmniRequest(inputs=None))
        with pytest.raises(ValueError, match="max_external_input_bytes=3"):
            await coordinator.send_input_chunk(
                "req", torch.tensor([1, 2], dtype=torch.int16)
            )
        await coordinator.close_input_stream("req")
        await handle.aclose()

    asyncio.run(run())


def test_coordinator_rejects_invalid_chunks_and_writes_after_done() -> None:
    async def run() -> None:
        coordinator, _ = _coordinator()
        handle = await coordinator.start_input_stream("req", OmniRequest(inputs=None))
        with pytest.raises(TypeError, match="torch.Tensor"):
            await coordinator.send_input_chunk("req", b"pcm")
        oversized = torch.zeros(
            stage_io.INLINE_STREAM_CHUNK_BYTES_LIMIT + 1, dtype=torch.uint8
        )
        with pytest.raises(ValueError, match="serialized payload"):
            await coordinator.send_input_chunk("req", oversized)
        await coordinator.finish_input_stream("req")
        with pytest.raises(RuntimeError, match="already done"):
            await coordinator.send_input_chunk(
                "req", torch.tensor([1], dtype=torch.int16)
            )
        with pytest.raises(RuntimeError, match="already done"):
            await coordinator.finish_input_stream("req")
        await coordinator.close_input_stream("req")
        await handle.aclose()

    asyncio.run(run())


def test_abort_waits_for_inflight_send_and_cleans_all_owners() -> None:
    class BlockingControlPlane(RecordingCoordinatorControlPlane):
        def __init__(self) -> None:
            super().__init__()
            self.entered = asyncio.Event()
            self.release = asyncio.Event()

        async def send_input_stream_event(self, stage, endpoint, msg) -> None:
            self.entered.set()
            await self.release.wait()
            await super().send_input_stream_event(stage, endpoint, msg)

    async def run() -> None:
        coordinator, _ = _coordinator()
        control_plane = BlockingControlPlane()
        coordinator.control_plane = control_plane
        handle = await coordinator.start_input_stream("req", OmniRequest(inputs=None))
        send_task = asyncio.create_task(
            coordinator.send_input_chunk("req", torch.tensor([1], dtype=torch.int16))
        )
        await control_plane.entered.wait()
        abort_task = asyncio.create_task(coordinator.close_input_stream("req"))
        await asyncio.sleep(0)
        assert not abort_task.done()
        control_plane.release.set()
        assert await send_task == 0
        assert await abort_task is True
        assert len(control_plane.input_stream_events) == 1
        assert len(control_plane.aborts) == 1
        assert "req" not in coordinator._requests
        assert "req" not in coordinator._external_input_streams
        assert "req" not in coordinator._stream_queues
        assert "req" not in coordinator._completion_futures
        await handle.aclose()

    asyncio.run(run())


def test_completion_waits_for_inflight_send_and_prevents_late_writes() -> None:
    class BlockingControlPlane(RecordingCoordinatorControlPlane):
        def __init__(self) -> None:
            super().__init__()
            self.entered = asyncio.Event()
            self.release = asyncio.Event()

        async def send_input_stream_event(self, stage, endpoint, msg) -> None:
            self.entered.set()
            await self.release.wait()
            await super().send_input_stream_event(stage, endpoint, msg)

    async def run() -> None:
        coordinator, _ = _coordinator()
        control_plane = BlockingControlPlane()
        coordinator.control_plane = control_plane
        events = await coordinator.start_input_stream("req", OmniRequest(inputs=None))

        send_task = asyncio.create_task(
            coordinator.send_input_chunk("req", torch.tensor([1], dtype=torch.int16))
        )
        await control_plane.entered.wait()
        completion_task = asyncio.create_task(
            coordinator._handle_completion(
                CompleteMessage("req", "asr", True, result={"text": "ok"})
            )
        )
        await asyncio.sleep(0)
        assert not completion_task.done()

        control_plane.release.set()
        assert await send_task == 0
        await completion_task
        assert len(control_plane.input_stream_events) == 1
        with pytest.raises(ValueError, match="No active input stream"):
            await coordinator.send_input_chunk(
                "req", torch.tensor([2], dtype=torch.int16)
            )
        assert [message async for message in events][-1].result == {"text": "ok"}

    asyncio.run(run())


@pytest.mark.parametrize("terminalizer", ["failure", "stop"])
def test_terminal_cleanup_precedes_stale_external_input_write(
    terminalizer: str,
) -> None:
    class BlockingControlPlane(RecordingCoordinatorControlPlane):
        def __init__(self) -> None:
            super().__init__()
            self.entered = asyncio.Event()
            self.release = asyncio.Event()
            self.block_next_send = True

        async def send_input_stream_event(self, stage, endpoint, msg) -> None:
            if self.block_next_send:
                self.block_next_send = False
                await super().send_input_stream_event(stage, endpoint, msg)
                self.entered.set()
                await self.release.wait()
                return
            await super().send_input_stream_event(stage, endpoint, msg)

    async def run() -> None:
        coordinator, _ = _coordinator()
        control_plane = BlockingControlPlane()
        coordinator.control_plane = control_plane
        events = await coordinator.start_input_stream("req", OmniRequest(inputs=None))
        other_events = await coordinator.start_input_stream(
            "other", OmniRequest(inputs=None)
        )

        in_flight = asyncio.create_task(
            coordinator.send_input_chunk("req", torch.tensor([1], dtype=torch.int16))
        )
        await control_plane.entered.wait()
        stale = asyncio.create_task(
            coordinator.send_input_chunk("req", torch.tensor([2], dtype=torch.int16))
        )
        await asyncio.sleep(0)
        assert not stale.done()

        if terminalizer == "failure":
            terminal = asyncio.create_task(
                coordinator.fail_pending_requests(RuntimeError("stage died"))
            )
        else:
            terminal = asyncio.create_task(coordinator.stop())
        await asyncio.sleep(0)
        assert not terminal.done()
        error = (
            "stage died"
            if terminalizer == "failure"
            else "not accepting external input writes"
        )
        with pytest.raises(RuntimeError, match=error):
            await coordinator.send_input_chunk(
                "other", torch.tensor([3], dtype=torch.int16)
            )

        control_plane.release.set()
        with pytest.raises(RuntimeError, match=error):
            await in_flight
        with pytest.raises(RuntimeError, match=error):
            await stale
        await terminal
        assert len(control_plane.input_stream_events) == 1

        if terminalizer == "stop":
            await coordinator.close_input_stream("req")
            await coordinator.close_input_stream("other")
        await events.aclose()
        await other_events.aclose()

    asyncio.run(run())


def test_concurrent_chunk_and_done_calls_keep_wire_order() -> None:
    class OrderedControlPlane(RecordingCoordinatorControlPlane):
        def __init__(self) -> None:
            super().__init__()
            self.first_send_entered = asyncio.Event()
            self.release_first_send = asyncio.Event()

        async def send_input_stream_event(self, stage, endpoint, msg) -> None:
            if not self.input_stream_events:
                self.first_send_entered.set()
                await self.release_first_send.wait()
            await super().send_input_stream_event(stage, endpoint, msg)

    async def run() -> None:
        coordinator, _ = _coordinator()
        control_plane = OrderedControlPlane()
        coordinator.control_plane = control_plane
        events = await coordinator.start_input_stream("req", OmniRequest(inputs=None))

        first = asyncio.create_task(
            coordinator.send_input_chunk("req", torch.tensor([1], dtype=torch.int16))
        )
        await control_plane.first_send_entered.wait()
        second = asyncio.create_task(
            coordinator.send_input_chunk("req", torch.tensor([2], dtype=torch.int16))
        )
        done = asyncio.create_task(coordinator.finish_input_stream("req"))
        await asyncio.sleep(0)
        control_plane.release_first_send.set()

        assert await first == 0
        assert await second == 1
        await done
        assert [
            (message.chunk_id, message.is_done)
            for _, _, message in control_plane.input_stream_events
        ] == [(0, False), (1, False), (None, True)]

        await coordinator.close_input_stream("req")
        await events.aclose()

    asyncio.run(run())


def test_stage_accepts_ordered_external_stream_and_marks_payload() -> None:
    async def run() -> None:
        scheduler = _external_scheduler()
        stage_obj, control_plane = _stage_with_control_plane(scheduler)
        payload = _payload()
        await stage_obj._on_submit(
            SubmitMessage("req", payload, external_input_stream=True)
        )
        request_message = scheduler.inbox.get_nowait()
        assert request_message.type == "new_request"
        assert request_message.data.external_input_stream is True

        await stage_obj._on_stream_chunk(_chunk("req", 0, 1))
        await stage_obj._on_stream_chunk(_chunk("req", 1, 2))
        await stage_obj._on_stream_signal(
            DataReadyMessage("req", "coordinator", "asr", data_ref=None, is_done=True)
        )
        assert [scheduler.inbox.get_nowait().type for _ in range(3)] == [
            "stream_chunk",
            "stream_chunk",
            "stream_done",
        ]
        assert control_plane.completions == []

        stage_obj._on_abort("req")
        assert "req" not in stage_obj._external_input_next_chunk_ids
        assert "req" not in stage_obj._external_input_done
        assert not stage_obj._stream_queue.has("req")

    asyncio.run(run())


def test_stage_rejects_unsupported_or_unbounded_scheduler() -> None:
    async def run() -> None:
        unsupported = SimpleNamespace(
            inbox=queue.Queue(maxsize=4),
            outbox=queue.Queue(),
            abort=lambda request_id: None,
        )
        stage_obj, control_plane = _stage_with_control_plane(unsupported)
        await stage_obj._on_submit(
            SubmitMessage(
                "unsupported", _payload("unsupported"), external_input_stream=True
            )
        )
        assert "does not support" in control_plane.completions[0].error
        assert unsupported.inbox.empty()
        assert "unsupported" not in stage_obj._active_requests
        assert "unsupported" not in stage_obj._external_input_next_chunk_ids

        unbounded = _external_scheduler(maxsize=0)
        stage_obj, control_plane = _stage_with_control_plane(unbounded)
        await stage_obj._on_submit(
            SubmitMessage(
                "unbounded", _payload("unbounded"), external_input_stream=True
            )
        )
        assert "must be bounded" in control_plane.completions[0].error
        assert unbounded.inbox.empty()
        assert "unbounded" not in stage_obj._active_requests
        assert "unbounded" not in stage_obj._external_input_next_chunk_ids

    asyncio.run(run())


def test_stage_rejects_external_stream_for_tensor_parallel_stage_only() -> None:
    async def run() -> None:
        scheduler = _external_scheduler()
        stage_obj, control_plane = _stage_with_control_plane(
            scheduler, role="leader", tp_size=2
        )

        await stage_obj._on_submit(
            SubmitMessage("external", _payload("external"), external_input_stream=True)
        )

        assert "require tp_size=1; got tp_size=2" in control_plane.completions[0].error
        assert scheduler.inbox.empty()
        assert "external" not in stage_obj._active_requests
        assert "external" not in stage_obj._external_input_next_chunk_ids

        await stage_obj._on_submit(SubmitMessage("regular", _payload("regular")))
        request_message = scheduler.inbox.get_nowait()
        assert request_message.type == "new_request"
        assert request_message.request_id == "regular"

    asyncio.run(run())


def test_stage_rejects_out_of_order_chunk() -> None:
    async def run() -> None:
        scheduler = _external_scheduler()
        stage_obj, control_plane = _stage_with_control_plane(scheduler)
        await stage_obj._on_submit(
            SubmitMessage("order", _payload("order"), external_input_stream=True)
        )
        scheduler.inbox.get_nowait()
        await stage_obj._on_stream_chunk(_chunk("order", 1))
        assert "expected chunk_id=0" in control_plane.completions[0].error
        assert scheduler.aborted == ["order"]

    asyncio.run(run())


def test_stage_waits_for_chunk_and_done_queue_capacity() -> None:
    async def run() -> None:
        scheduler = _external_scheduler(maxsize=1)
        stage_obj, control_plane = _stage_with_control_plane(scheduler)
        await stage_obj._on_submit(
            SubmitMessage("req", _payload(), external_input_stream=True)
        )
        assert scheduler.inbox.qsize() == 1

        chunk_task = asyncio.create_task(stage_obj._on_stream_chunk(_chunk("req", 0)))
        await asyncio.sleep(stage_runtime._EXTERNAL_INPUT_ENQUEUE_RETRY_S * 2)
        assert not chunk_task.done()
        assert scheduler.inbox.get_nowait().type == "new_request"
        await chunk_task
        assert scheduler.inbox.qsize() == 1

        done_task = asyncio.create_task(
            stage_obj._on_stream_signal(
                DataReadyMessage(
                    "req", "coordinator", "asr", data_ref=None, is_done=True
                )
            )
        )
        await asyncio.sleep(stage_runtime._EXTERNAL_INPUT_ENQUEUE_RETRY_S * 2)
        assert not done_task.done()
        assert scheduler.inbox.get_nowait().type == "stream_chunk"
        await done_task
        assert scheduler.inbox.get_nowait().type == "stream_done"
        assert control_plane.completions == []

    asyncio.run(run())


def test_stage_queue_timeout_fails_and_cleans_stream(monkeypatch) -> None:
    monkeypatch.setattr(stage_runtime, "_EXTERNAL_INPUT_ENQUEUE_TIMEOUT_S", 0.02)

    async def run() -> None:
        scheduler = _external_scheduler(maxsize=1)
        stage_obj, control_plane = _stage_with_control_plane(scheduler)
        await stage_obj._on_submit(
            SubmitMessage("req", _payload(), external_input_stream=True)
        )

        await stage_obj._on_stream_chunk(_chunk("req", 0))

        assert "timed out waiting" in control_plane.completions[0].error
        assert scheduler.aborted == ["req"]
        assert "req" not in stage_obj._active_requests
        assert "req" not in stage_obj._external_input_next_chunk_ids
        assert "req" not in stage_obj._external_input_done
        assert not stage_obj._stream_queue.has("req")

    asyncio.run(run())


def test_stage_abort_interrupts_queue_wait_without_failure() -> None:
    async def run() -> None:
        scheduler = _external_scheduler(maxsize=1)
        stage_obj, control_plane = _stage_with_control_plane(scheduler)
        await stage_obj._on_submit(
            SubmitMessage("req", _payload(), external_input_stream=True)
        )

        chunk_task = asyncio.create_task(stage_obj._on_stream_chunk(_chunk("req", 0)))
        await asyncio.sleep(stage_runtime._EXTERNAL_INPUT_ENQUEUE_RETRY_S * 2)
        assert not chunk_task.done()
        stage_obj._on_abort("req")
        await asyncio.wait_for(chunk_task, timeout=0.25)

        assert control_plane.completions == []
        assert scheduler.aborted == ["req"]
        assert "req" not in stage_obj._active_requests
        assert "req" not in stage_obj._external_input_next_chunk_ids
        assert "req" not in stage_obj._external_input_done

    asyncio.run(run())


def test_submit_message_external_stream_flag_round_trips_strictly() -> None:
    encoded = SubmitMessage("req", _payload(), external_input_stream=True).to_dict()
    assert SubmitMessage.from_dict(encoded).external_input_stream is True
    encoded["external_input_stream"] = 1
    with pytest.raises(TypeError, match="external_input_stream must be bool"):
        SubmitMessage.from_dict(encoded)
