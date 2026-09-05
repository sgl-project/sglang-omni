# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import asyncio
import queue
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
import torch
from pydantic import ValidationError

import sglang_omni.platforms as platforms
from sglang_omni.comm import stage_io
from sglang_omni.comm.data_ref import DataKind, DataRef, TransportKind
from sglang_omni.comm.engine import CommEngine
from sglang_omni.config.schema import StageConfig
from sglang_omni.models.fishaudio_s2_pro.config import S2ProPipelineConfig
from sglang_omni.pipeline.stage.runtime import Stage
from sglang_omni.pipeline.stage.stream_queue import StreamItem, StreamQueue
from sglang_omni.proto import (
    DataReadyMessage,
    InputUpdateMessage,
    OmniRequest,
    StagePayload,
)
from sglang_omni.relay.shm import ShmRelay
from sglang_omni.scheduling.messages import OutgoingMessage
from tests.unit_test.fixtures.pipeline_fakes import (
    FakeScheduler,
    RecordingStageControlPlane,
)
from tests.unit_test.fixtures.trace_capture import capture_comm_trace, events_named
from tests.unit_test.pipeline.helpers import make_stage


def _input_update_message(request_id: str = "request-1") -> InputUpdateMessage:
    return InputUpdateMessage(
        request_id=request_id,
        session_id="session-1",
        turn_id="turn-1",
        seq_no=0,
        token_ids=(7,),
    )


def test_stage_routes_input_update_to_scheduler_inbox() -> None:
    async def _run() -> None:
        scheduler = FakeScheduler()
        stage = make_stage(
            name="tts_engine",
            scheduler=scheduler,
            can_accept_stream_before_payload=True,
        )
        message = _input_update_message()

        await stage._handle_message(message)

        queued = scheduler.inbox.get_nowait()
        assert queued.request_id == "request-1"
        assert queued.type == "input_update"
        assert queued.data is message
        assert "request-1" in stage._active_requests

    asyncio.run(_run())


def test_stage_drops_input_update_for_scheduler_terminal_tombstone() -> None:
    async def _run() -> None:
        scheduler = FakeScheduler()
        scheduler.is_input_update_terminal = lambda request_id: request_id == "done"
        stage = make_stage(
            name="tts_engine",
            scheduler=scheduler,
            can_accept_stream_before_payload=True,
        )

        await stage._handle_message(_input_update_message("done"))

        assert scheduler.inbox.empty()
        assert "done" not in stage._active_requests

    asyncio.run(_run())


def test_stage_rejects_unconfigured_pre_payload_input_update() -> None:
    async def _run() -> None:
        scheduler = FakeScheduler()
        control_plane = RecordingStageControlPlane()
        stage = make_stage(
            name="ordinary_stage",
            scheduler=scheduler,
            control_plane=control_plane,
        )

        await stage._handle_message(_input_update_message())

        assert scheduler.inbox.empty()
        assert control_plane.completions[0].success is False
        assert "before the request payload" in control_plane.completions[0].error

    asyncio.run(_run())


class _FakeControlPlane:
    recv_endpoint = "inproc://stage"

    def __init__(self) -> None:
        self.streams = []
        self.stage_messages = []
        self.completions = []

    async def start(self) -> None:
        pass

    def close(self) -> None:
        pass

    async def send_stream(self, msg) -> None:
        self.streams.append(msg)

    async def send_to_stage(self, target, endpoint, msg) -> None:
        self.stage_messages.append((target, endpoint, msg))

    async def send_complete(self, msg) -> None:
        self.completions.append(msg)


class _FakeRelay:
    def __init__(self) -> None:
        self.device = "cpu"
        self.puts = []

    async def put_async(
        self,
        tensor,
        request_id=None,
        dst_rank=None,
        receiver_id=None,
    ):
        del dst_rank, receiver_id
        self.puts.append((request_id, tensor))
        return _DoneOp(tensor.numel())

    def close(self) -> None:
        pass

    def cleanup(self, request_id: str) -> None:
        pass


class _DoneOp:
    def __init__(self, size: int = 1) -> None:
        self.metadata = {"transfer_info": {"size": size}}

    async def wait_for_completion(self) -> None:
        pass

    def mark_receiver_done(self) -> None:
        pass

    def mark_receiver_failed(self, exc: BaseException) -> None:
        raise exc


class _AbortOnReadRelay(_FakeRelay):
    def __init__(self, on_wait) -> None:
        super().__init__()
        self._on_wait = on_wait
        self.gets = 0

    async def get_async(self, metadata, dest_tensor, request_id):
        del metadata, dest_tensor, request_id
        self.gets += 1
        return _CallbackOp(self._on_wait)


class _CallbackOp:
    def __init__(self, on_wait) -> None:
        self._on_wait = on_wait

    async def wait_for_completion(self) -> None:
        self._on_wait()

    def mark_receiver_done(self) -> None:
        pass

    def mark_receiver_failed(self, exc: BaseException) -> None:
        raise exc


# Stream chunks now always move through the transport router's relay (CUDA-IPC on
# GPU, shm on CPU). These helpers build real relay-backed messages over an
# in-process ShmRelay so the receive path is exercised end to end, matching what
# ``write_stream_chunk`` / ``write_payload`` produce on the wire.


async def _make_relay_chunk(
    relay,
    *,
    request_id: str,
    from_stage: str,
    to_stage: str,
    chunk_id: int,
    data,
    metadata: dict | None = None,
) -> DataReadyMessage:
    object_id = f"{request_id}:stream:{from_stage}:{to_stage}:{chunk_id}"
    data_ref, op = await stage_io.write_tensor(
        relay,
        object_id,
        data,
        transport=TransportKind.SHM,
        kind=DataKind.STREAM_CHUNK,
        request_id=request_id,
        from_stage=from_stage,
        to_stage=to_stage,
    )
    pending_ops = [op]
    data_ref = await stage_io._with_stream_metadata(
        relay,
        data_ref,
        metadata,
        TransportKind.SHM,
        pending_ops,
    )
    return DataReadyMessage(
        request_id=request_id,
        from_stage=from_stage,
        to_stage=to_stage,
        data_ref=data_ref.to_dict(),
        chunk_id=chunk_id,
    )


async def _make_relay_payload(
    relay,
    payload: StagePayload,
    *,
    from_stage: str = "tts_engine",
    to_stage: str = "vocoder",
) -> DataReadyMessage:
    data_ref, op = await stage_io.write_payload(
        relay,
        payload.request_id,
        payload,
        transport=TransportKind.SHM,
    )
    return DataReadyMessage(
        request_id=payload.request_id,
        from_stage=from_stage,
        to_stage=to_stage,
        data_ref=data_ref.to_dict(),
    )


def test_terminal_scheduler_stream_routes_to_coordinator() -> None:
    async def _run() -> None:
        control_plane = _FakeControlPlane()
        scheduler = SimpleNamespace(outbox=queue.Queue())
        stage = Stage(
            name="vocoder",
            role="single",
            get_next=lambda request_id, output: None,
            gpu_id=None,
            endpoints={"tts_engine": "inproc://tts_engine"},
            control_plane=control_plane,
            relay=_FakeRelay(),
            scheduler=scheduler,
            is_terminal=True,
        )
        stage._active_requests.add("req")
        scheduler.outbox.put(
            OutgoingMessage(
                request_id="req",
                type="stream",
                data={"audio_data": [0.1], "modality": "audio"},
            )
        )
        scheduler.outbox.put(
            OutgoingMessage(
                request_id="req",
                type="stream",
                data={"audio_data": [0.2], "modality": "audio"},
            )
        )

        await stage._drain_outbox_external()

        assert len(control_plane.streams) == 2
        msg = control_plane.streams[0]
        assert msg.request_id == "req"
        assert msg.from_stage == "vocoder"
        assert msg.chunk == {"audio_data": [0.1], "modality": "audio"}
        assert msg.modality == "audio"
        assert [msg.chunk_id for msg in control_plane.streams] == [0, 1]

    asyncio.run(_run())


def test_outbox_drain_reuses_one_executor_wakeup_for_ready_messages(
    monkeypatch,
) -> None:
    async def _run() -> None:
        loop = asyncio.get_running_loop()
        run_in_executor = AsyncMock(side_effect=lambda _, get: get())
        monkeypatch.setattr(loop, "run_in_executor", run_in_executor)

        control_plane = _FakeControlPlane()
        scheduler = SimpleNamespace(outbox=queue.Queue())
        stage = Stage(
            name="vocoder",
            role="single",
            get_next=lambda request_id, output: None,
            gpu_id=None,
            endpoints={"tts_engine": "inproc://tts_engine"},
            control_plane=control_plane,
            relay=_FakeRelay(),
            scheduler=scheduler,
            is_terminal=True,
        )
        stage._active_requests.add("req-live")

        messages = [
            OutgoingMessage("req-live", "stream", {"sequence": 0}),
            OutgoingMessage("req-stale", "stream", {"sequence": 999}),
            OutgoingMessage("req-live", "stream", {"sequence": 1}),
            OutgoingMessage("req-live", "result", {"answer": "done"}),
        ]
        for message in messages:
            scheduler.outbox.put(message)

        await stage._drain_outbox_external()

        assert [
            (msg.chunk_id, msg.chunk["sequence"]) for msg in control_plane.streams
        ] == [(0, 0), (1, 1)]
        assert len(control_plane.completions) == 1
        completion = control_plane.completions[0]
        assert (completion.success, completion.result) == (True, {"answer": "done"})
        assert scheduler.outbox.empty()
        assert not stage._active_requests

        # Before this optimization each message required an executor round trip.
        run_in_executor.assert_awaited_once()

    asyncio.run(_run())


def test_outbox_drain_yields_after_ready_message_batch(monkeypatch) -> None:
    async def _run() -> None:
        loop = asyncio.get_running_loop()
        run_in_executor = AsyncMock(side_effect=lambda _, get: get())
        monkeypatch.setattr(loop, "run_in_executor", run_in_executor)

        execution_order: list[int | str] = []

        class _RecordingControlPlane(_FakeControlPlane):
            async def send_stream(self, msg) -> None:
                execution_order.append(msg.chunk["sequence"])
                await super().send_stream(msg)

        control_plane = _RecordingControlPlane()
        scheduler = SimpleNamespace(outbox=queue.Queue())
        stage = Stage(
            name="vocoder",
            role="single",
            get_next=lambda request_id, output: None,
            gpu_id=None,
            endpoints={"tts_engine": "inproc://tts_engine"},
            control_plane=control_plane,
            relay=_FakeRelay(),
            scheduler=scheduler,
            is_terminal=True,
        )
        stage._active_requests.add("req-live")

        for sequence in range(1, 66):
            scheduler.outbox.put(
                OutgoingMessage("req-live", "stream", {"sequence": sequence})
            )

        async def _competing_ready_coroutine() -> None:
            execution_order.append("competing-coroutine")

        competing_task = asyncio.create_task(_competing_ready_coroutine())
        await stage._drain_outbox_external()
        await competing_task

        assert execution_order == [
            *range(1, 65),
            "competing-coroutine",
            65,
        ]
        assert [msg.chunk["sequence"] for msg in control_plane.streams] == list(
            range(1, 66)
        )
        assert scheduler.outbox.empty()
        assert run_in_executor.await_count == 2

    asyncio.run(_run())


def test_explicit_scheduler_stream_target_keeps_stage_to_stage_routing(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        platforms.current_platform, "device_type", "cuda", raising=False
    )

    async def _run() -> None:
        control_plane = _FakeControlPlane()
        relay = _FakeRelay()
        scheduler = SimpleNamespace(outbox=queue.Queue())
        codes = torch.empty(4096, 1, dtype=torch.long)
        stage = Stage(
            name="tts_engine",
            role="single",
            get_next=lambda request_id, output: None,
            gpu_id=None,
            endpoints={"vocoder": "inproc://vocoder"},
            control_plane=control_plane,
            relay=relay,
            scheduler=scheduler,
        )
        stage._active_requests.add("req")
        scheduler.outbox.put(
            OutgoingMessage(
                request_id="req",
                type="stream",
                data=codes,
                target="vocoder",
                metadata={"modality": "audio_codes"},
            )
        )

        await stage._drain_outbox_external()

        assert control_plane.streams == []
        assert len(relay.puts) == 1
        assert len(control_plane.stage_messages) == 1
        target, endpoint, msg = control_plane.stage_messages[0]
        assert target == "vocoder"
        assert endpoint == "inproc://vocoder"
        assert msg.request_id == "req"
        assert msg.from_stage == "tts_engine"
        assert msg.to_stage == "vocoder"
        assert msg.chunk_id == 0
        assert DataRef.from_dict(msg.data_ref).metadata == {"modality": "audio_codes"}

    asyncio.run(_run())


def test_small_cpu_scheduler_stream_chunk_rides_inline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _run() -> None:
        control_plane = _FakeControlPlane()
        relay = _FakeRelay()
        scheduler = SimpleNamespace(outbox=queue.Queue())
        token = torch.tensor([42], dtype=torch.long)
        stage = Stage(
            name="thinker",
            role="single",
            get_next=lambda request_id, output: None,
            gpu_id=None,
            endpoints={"decode": "inproc://decode"},
            control_plane=control_plane,
            relay=relay,
            scheduler=scheduler,
        )
        stage._active_requests.add("req")
        stage._replica_bindings["req"] = {"decode": 1}
        scheduler.outbox.put(
            OutgoingMessage(
                request_id="req",
                type="stream",
                data=token,
                target="decode",
                metadata={"token_id": 42},
            )
        )

        await stage._drain_outbox_external()

        assert relay.puts == []
        assert len(control_plane.stage_messages) == 1
        target, endpoint, msg = control_plane.stage_messages[0]
        assert target == "decode"
        assert endpoint == "inproc://decode"
        assert msg.chunk_id == 0
        assert msg.replica_bindings == {"decode": 1}
        assert stage_io.is_inline_stream_chunk_ref(msg.data_ref)
        data, metadata = stage_io.deserialize_inline_stream_chunk(msg.data_ref)
        assert torch.equal(data, token)
        assert metadata == {"token_id": 42}

    with capture_comm_trace(monkeypatch) as events:
        asyncio.run(_run())

    selected = events_named(events, "comm_transport_selected")
    assert len(selected) == 1
    assert selected[0]["direction"] == "stream"
    assert selected[0]["peer_stage"] == "decode"
    assert selected[0]["transport"] == "inline"

    sent = events_named(events, "comm_stream_send")
    assert len(sent) == 1
    assert sent[0]["request_id"] == "req"
    assert sent[0]["from_stage"] == "thinker"
    assert sent[0]["to_stage"] == "decode"
    assert sent[0]["transport"] == "inline"
    assert sent[0]["bytes"] == 8


def test_inline_stream_chunk_gate() -> None:
    small = torch.zeros(8, dtype=torch.long)
    assert stage_io.serialize_inline_stream_chunk(small, {"token_id": 1}) is not None
    assert stage_io.serialize_inline_stream_chunk(small, None) is not None
    big = torch.zeros(stage_io._INLINE_STREAM_CHUNK_BYTES_LIMIT + 1, dtype=torch.uint8)
    assert stage_io.serialize_inline_stream_chunk(big, None) is None
    assert (
        stage_io.serialize_inline_stream_chunk(small, {"extra": torch.zeros(2)}) is None
    )
    assert stage_io.serialize_inline_stream_chunk("not-a-tensor", None) is None


def test_inline_stream_chunk_rejects_oversized_serialized_payload() -> None:
    token = torch.tensor([7], dtype=torch.long)
    metadata = {"text": "x" * stage_io._INLINE_STREAM_CHUNK_BYTES_LIMIT}

    data_ref = stage_io.serialize_inline_stream_chunk(token, metadata)

    assert data_ref is None


def test_inline_stream_chunk_rejects_non_cpu_tensor() -> None:
    tensor = torch.empty(1, device="meta")

    data_ref = stage_io.serialize_inline_stream_chunk(tensor, None)

    assert data_ref is None


def test_inline_stream_chunk_rejects_invalid_metadata_type() -> None:
    data_ref = {
        "_type": stage_io._INLINE_STREAM_CHUNK_TYPE,
        "version": 1,
        "payload": stage_io.pickle.dumps((torch.tensor([7]), ["invalid"])),
    }

    with pytest.raises(TypeError, match="metadata must be dict or None"):
        stage_io.deserialize_inline_stream_chunk(data_ref)


def test_inline_stream_chunk_rejects_oversized_received_payload() -> None:
    data_ref = {
        "_type": stage_io._INLINE_STREAM_CHUNK_TYPE,
        "version": 1,
        "payload": b"x" * (stage_io._INLINE_STREAM_CHUNK_BYTES_LIMIT + 1),
    }

    with pytest.raises(ValueError, match="payload exceeds"):
        stage_io.deserialize_inline_stream_chunk(data_ref)


def test_inline_stream_chunk_rejects_non_cpu_received_tensor() -> None:
    data_ref = {
        "_type": stage_io._INLINE_STREAM_CHUNK_TYPE,
        "version": 1,
        "payload": stage_io.pickle.dumps((torch.empty(1, device="meta"), None)),
    }

    with pytest.raises(ValueError, match="data must be a CPU tensor"):
        stage_io.deserialize_inline_stream_chunk(data_ref)


def test_inline_stream_chunk_materializes_small_view_storage() -> None:
    view = torch.zeros(1024 * 1024, dtype=torch.uint8)[:8]

    data_ref = stage_io.serialize_inline_stream_chunk(view, None)
    restored, _ = stage_io.deserialize_inline_stream_chunk(data_ref)

    assert restored.untyped_storage().nbytes() == restored.numel()


def test_inline_stream_chunk_does_not_clone_small_owning_tensor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tensor = torch.tensor([7], dtype=torch.long)
    serialized_tensor: torch.Tensor | None = None

    def capture_payload(payload) -> bytes:
        nonlocal serialized_tensor
        serialized_tensor = payload[0]
        return b"payload"

    monkeypatch.setattr(stage_io.pickle, "dumps", capture_payload)

    stage_io.serialize_inline_stream_chunk(tensor, None)

    assert serialized_tensor is not None
    assert serialized_tensor.data_ptr() == tensor.data_ptr()


def test_stage_routes_inline_stream_chunk_to_scheduler(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _run() -> None:
        control_plane = _FakeControlPlane()
        relay = ShmRelay(engine_id="t-inline", device="cpu")
        scheduler = SimpleNamespace(
            outbox=queue.Queue(),
            inbox=queue.Queue(),
            abort=lambda request_id: None,
        )
        stage = Stage(
            name="decode",
            role="single",
            get_next=lambda request_id, output: None,
            gpu_id=None,
            endpoints={
                "thinker": "inproc://thinker",
                "tts_engine": "inproc://tts_engine",
            },
            control_plane=control_plane,
            relay=relay,
            scheduler=scheduler,
        )
        stage._stream_queue = StreamQueue(max_pending=4096)
        payload = StagePayload(
            request_id="req",
            request=OmniRequest(inputs="hello"),
            data={"ready": True},
        )
        await stage._on_data_ready(await _make_relay_payload(relay, payload))
        token = torch.tensor([7], dtype=torch.long)
        msg = DataReadyMessage(
            request_id="req",
            from_stage="thinker",
            to_stage="decode",
            data_ref=stage_io.serialize_inline_stream_chunk(token, {"token_id": 7}),
            chunk_id=0,
        )

        await stage._on_stream_chunk(msg)

        payload_msg = scheduler.inbox.get_nowait()
        chunk_msg = scheduler.inbox.get_nowait()
        assert payload_msg.type == "new_request"
        assert chunk_msg.type == "stream_chunk"
        assert torch.equal(chunk_msg.data.data, token)
        assert chunk_msg.data.metadata == {"token_id": 7}

    with capture_comm_trace(monkeypatch) as events:
        asyncio.run(_run())

    received = events_named(events, "comm_stream_read")
    assert len(received) == 1
    assert received[0]["request_id"] == "req"
    assert received[0]["from_stage"] == "thinker"
    assert received[0]["to_stage"] == "decode"
    assert received[0]["transport"] == "inline"
    assert received[0]["bytes"] == 8


def test_stage_config_rejects_unknown_model_transport_field() -> None:
    field_name = "stream_" + "transport"
    with pytest.raises(ValidationError):
        StageConfig(
            name="tts_engine",
            factory_path="pkg.create",
            next="vocoder",
            stream_to=["vocoder"],
            **{field_name: {"vocoder": "relay"}},
        )


def test_s2pro_config_declares_topology_without_transport_policy() -> None:
    config = S2ProPipelineConfig(model_path="dummy")
    tts_stage = next(stage for stage in config.stages if stage.name == "tts_engine")
    vocoder_stage = next(stage for stage in config.stages if stage.name == "vocoder")
    assert tts_stage.stream_to == ["vocoder"]
    assert vocoder_stage.can_accept_stream_before_payload
    assert "stream_transport" not in StageConfig.model_fields


def test_stream_queue_drops_chunk_after_request_cleanup() -> None:
    stream_queue = StreamQueue()
    stream_queue.open("req")
    stream_queue.close("req")

    stream_queue.put(
        "req",
        StreamItem(
            chunk_id=0,
            data=torch.tensor([1]),
            from_stage="thinker",
        ),
    )

    assert not stream_queue.has("req")


def test_stage_fails_pre_payload_stream_chunk_by_default() -> None:
    async def _run() -> None:
        control_plane = _FakeControlPlane()
        scheduler = SimpleNamespace(
            outbox=queue.Queue(),
            inbox=queue.Queue(),
            abort=lambda request_id: None,
        )
        relay = ShmRelay(engine_id="t", device="cpu")
        stage = Stage(
            name="vocoder",
            role="single",
            get_next=lambda request_id, output: None,
            gpu_id=None,
            endpoints={"tts_engine": "inproc://tts_engine"},
            control_plane=control_plane,
            relay=relay,
            scheduler=scheduler,
        )
        stage._stream_queue = StreamQueue(max_pending=4096)
        codes = torch.arange(11, dtype=torch.float32)

        await stage._on_stream_chunk(
            await _make_relay_chunk(
                relay,
                request_id="req",
                from_stage="tts_engine",
                to_stage="vocoder",
                chunk_id=0,
                data=codes,
            )
        )

        assert scheduler.inbox.empty()
        assert len(control_plane.completions) == 1
        assert control_plane.completions[0].success is False
        assert "pre-payload stream data" in control_plane.completions[0].error

    asyncio.run(_run())


def test_stage_routes_stream_chunk_after_payload_by_default() -> None:
    async def _run() -> None:
        control_plane = _FakeControlPlane()
        scheduler = SimpleNamespace(
            outbox=queue.Queue(),
            inbox=queue.Queue(),
            abort=lambda request_id: None,
        )
        relay = ShmRelay(engine_id="t", device="cpu")
        stage = Stage(
            name="vocoder",
            role="single",
            get_next=lambda request_id, output: None,
            gpu_id=None,
            endpoints={"tts_engine": "inproc://tts_engine"},
            control_plane=control_plane,
            relay=relay,
            scheduler=scheduler,
        )
        stage._stream_queue = StreamQueue(max_pending=4096)
        payload = StagePayload(
            request_id="req",
            request=OmniRequest(inputs="hello"),
            data={"ready": True},
        )
        await stage._on_data_ready(await _make_relay_payload(relay, payload))
        codes = torch.arange(11, dtype=torch.float32)
        await stage._on_stream_chunk(
            await _make_relay_chunk(
                relay,
                request_id="req",
                from_stage="tts_engine",
                to_stage="vocoder",
                chunk_id=0,
                data=codes,
            )
        )
        payload_msg = scheduler.inbox.get_nowait()
        chunk_msg = scheduler.inbox.get_nowait()
        assert payload_msg.type == "new_request"
        assert chunk_msg.type == "stream_chunk"
        assert torch.equal(chunk_msg.data.data, codes)

    asyncio.run(_run())


def test_stage_routes_pre_payload_stream_events_for_capable_receiver() -> None:
    async def _run() -> None:
        control_plane = _FakeControlPlane()
        scheduler = SimpleNamespace(
            outbox=queue.Queue(),
            inbox=queue.Queue(),
            abort=lambda request_id: None,
        )
        relay = ShmRelay(engine_id="t", device="cpu")
        stage = Stage(
            name="vocoder",
            role="single",
            get_next=lambda request_id, output: None,
            gpu_id=None,
            endpoints={"tts_engine": "inproc://tts_engine"},
            control_plane=control_plane,
            relay=relay,
            scheduler=scheduler,
            can_accept_stream_before_payload=True,
        )
        stage._stream_queue = StreamQueue(max_pending=4096)
        codes = torch.arange(11, dtype=torch.float32)

        await stage._on_stream_chunk(
            await _make_relay_chunk(
                relay,
                request_id="req",
                from_stage="tts_engine",
                to_stage="vocoder",
                chunk_id=0,
                data=codes,
                metadata={"modality": "audio_codes"},
            )
        )

        chunk_msg = scheduler.inbox.get_nowait()
        assert chunk_msg.request_id == "req"
        assert chunk_msg.type == "stream_chunk"
        assert torch.equal(chunk_msg.data.data, codes)
        assert chunk_msg.data.metadata == {"modality": "audio_codes"}

        await stage._on_stream_signal(
            DataReadyMessage(
                request_id="req",
                from_stage="tts_engine",
                to_stage="vocoder",
                data_ref=None,
                is_done=True,
            )
        )
        early_done_msg = scheduler.inbox.get_nowait()
        assert early_done_msg.request_id == "req"
        assert early_done_msg.type == "stream_done"

        payload = StagePayload(
            request_id="req",
            request=OmniRequest(inputs="hello"),
            data={"ready": True},
        )
        await stage._on_data_ready(await _make_relay_payload(relay, payload))
        payload_msg = scheduler.inbox.get_nowait()
        assert payload_msg.request_id == "req"
        assert payload_msg.type == "new_request"
        assert payload_msg.data.data == {"ready": True}

    asyncio.run(_run())


def test_stage_stream_chunk_received_after_relay_materialization(monkeypatch) -> None:
    """Cross-GPU chunks emit receive only after relay data and metadata are restored."""
    order: list[str] = []

    async def fake_read_stream_chunk(self, *, relay, data_ref):
        del self, relay, data_ref
        order.append("stream_chunk_read")
        return "chunk", {"modality": "audio_codes"}

    async def fake_route(self, request_id, item):
        del self, request_id, item
        order.append("routed")

    monkeypatch.setattr(CommEngine, "read_stream_chunk", fake_read_stream_chunk)
    monkeypatch.setattr(Stage, "_route_stream_item_or_fail", fake_route)
    monkeypatch.setattr(
        "sglang_omni.pipeline.stage.runtime._emit_event",
        lambda **kwargs: order.append(kwargs["event_name"]),
    )

    async def _run() -> None:
        stage = Stage(
            name="vocoder",
            role="single",
            get_next=lambda request_id, output: None,
            gpu_id=None,
            endpoints={"tts_engine": "inproc://tts_engine"},
            control_plane=_FakeControlPlane(),
            relay=_FakeRelay(),
            scheduler=SimpleNamespace(outbox=queue.Queue()),
            can_accept_stream_before_payload=True,
        )
        await stage._on_stream_chunk(
            DataReadyMessage(
                request_id="req",
                from_stage="tts_engine",
                to_stage="vocoder",
                data_ref=(
                    await _make_relay_chunk(
                        _FakeRelay(),
                        request_id="req",
                        from_stage="tts_engine",
                        to_stage="vocoder",
                        chunk_id=0,
                        data=torch.empty(1),
                    )
                ).data_ref,
                chunk_id=0,
            )
        )

    asyncio.run(_run())

    assert order == ["stream_chunk_read", "stage_stream_chunk_received", "routed"]


def test_stage_stream_error_fails_request_even_with_stream_queue() -> None:
    async def _run() -> None:
        control_plane = _FakeControlPlane()
        scheduler = SimpleNamespace(
            outbox=queue.Queue(),
            inbox=queue.Queue(),
            aborted=[],
            abort=lambda request_id: scheduler.aborted.append(request_id),
        )
        stage = Stage(
            name="decode",
            role="single",
            get_next=lambda request_id, output: None,
            gpu_id=None,
            endpoints={"tts_engine": "inproc://tts_engine"},
            control_plane=control_plane,
            relay=_FakeRelay(),
            scheduler=scheduler,
            is_terminal=True,
        )
        stage._stream_queue = StreamQueue(max_pending=4096)
        stage._stream_queue.open("req")

        await stage._queue_stream_error(
            "req",
            from_stage="thinker",
            error=RuntimeError("stream failed"),
        )

        assert scheduler.aborted == ["req"]
        assert len(control_plane.completions) == 1
        assert control_plane.completions[0].success is False
        assert control_plane.completions[0].error == "stream failed"
        assert not stage._stream_queue.has("req")
        assert "req" in stage._aborted

    asyncio.run(_run())


def test_terminal_request_can_be_readmitted_after_cleanup() -> None:
    async def _run() -> None:
        scheduler = SimpleNamespace(
            outbox=queue.Queue(),
            inbox=queue.Queue(),
            abort=lambda request_id: None,
        )
        stage = Stage(
            name="vocoder",
            role="single",
            get_next=lambda request_id, output: None,
            gpu_id=None,
            endpoints={},
            control_plane=_FakeControlPlane(),
            relay=_FakeRelay(),
            scheduler=scheduler,
            can_accept_stream_before_payload=True,
        )
        stage._stream_queue = StreamQueue(max_pending=4096)
        stage._stream_queue.open("req")
        stage._active_requests.add("req")

        stage._clear_request_state("req")
        await stage.receive_local_payload(
            "req",
            "tts_engine",
            SimpleNamespace(request_id="req"),
        )
        incoming = scheduler.inbox.get_nowait()
        assert incoming.request_id == "req"
        assert incoming.type == "new_request"
        assert "req" in stage._active_requests
        assert stage._stream_queue.has("req")
        assert stage.control_plane.completions == []

    asyncio.run(_run())


def test_write_stream_chunk_uses_relay() -> None:
    async def _run() -> None:
        control_plane = _FakeControlPlane()
        relay = _FakeRelay()
        codes = torch.empty(11, 1, dtype=torch.long)

        data_ref, ops = await stage_io.write_stream_chunk(
            relay,
            request_id="req",
            data=codes,
            target_stage="vocoder",
            from_stage="tts_engine",
            chunk_id=0,
            transport=TransportKind.SHM,
        )
        await control_plane.send_to_stage(
            "vocoder",
            "inproc://vocoder",
            DataReadyMessage(
                request_id="req",
                from_stage="tts_engine",
                to_stage="vocoder",
                data_ref=data_ref.to_dict(),
                chunk_id=0,
            ),
        )
        for op in ops:
            op.mark_receiver_done()
            await op.wait_for_completion()

        assert len(relay.puts) == 1
        assert relay.puts[0][0] == "req:stream:tts_engine:vocoder:0"
        assert len(control_plane.stage_messages) == 1
        _, _, msg = control_plane.stage_messages[0]
        expected_size = codes.contiguous().view(torch.uint8).numel()
        data_ref = DataRef.from_dict(msg.data_ref)
        assert data_ref.buffer.info == {"transfer_info": {"size": expected_size}}

    asyncio.run(_run())


def test_stage_drops_stream_chunk_after_abort_during_relay_read() -> None:
    async def _run() -> None:
        control_plane = _FakeControlPlane()
        codes = torch.empty(11, 1, dtype=torch.long)
        scheduler = SimpleNamespace(
            outbox=queue.Queue(),
            inbox=queue.Queue(),
            abort=lambda request_id: None,
        )
        relay = _AbortOnReadRelay(lambda: stage._on_abort("req"))
        stage = Stage(
            name="vocoder",
            role="single",
            get_next=lambda request_id, output: None,
            gpu_id=None,
            endpoints={"tts_engine": "inproc://tts_engine"},
            control_plane=control_plane,
            relay=relay,
            scheduler=scheduler,
        )
        stage._stream_queue = None

        await stage._on_stream_chunk(
            await _make_relay_chunk(
                relay,
                request_id="req",
                from_stage="tts_engine",
                to_stage="vocoder",
                chunk_id=0,
                data=codes,
            )
        )

        assert scheduler.inbox.empty()
        assert relay.gets == 1

    asyncio.run(_run())


def test_stage_drains_relay_stream_chunk_for_already_aborted_request() -> None:
    async def _run() -> None:
        control_plane = _FakeControlPlane()
        codes = torch.empty(11, 1, dtype=torch.long)
        scheduler = SimpleNamespace(
            outbox=queue.Queue(),
            inbox=queue.Queue(),
            abort=lambda request_id: None,
        )
        relay = _AbortOnReadRelay(lambda: None)
        stage = Stage(
            name="vocoder",
            role="single",
            get_next=lambda request_id, output: None,
            gpu_id=None,
            endpoints={"tts_engine": "inproc://tts_engine"},
            control_plane=control_plane,
            relay=relay,
            scheduler=scheduler,
        )
        stage._aborted.add("req")
        await stage._on_stream_chunk(
            await _make_relay_chunk(
                relay,
                request_id="req",
                from_stage="tts_engine",
                to_stage="vocoder",
                chunk_id=0,
                data=codes,
                metadata={"latency": torch.zeros(1)},
            )
        )

        assert scheduler.inbox.empty()
        assert relay.gets == 2

    asyncio.run(_run())


def test_stage_drains_relay_payload_for_already_aborted_request() -> None:
    async def _run() -> None:
        control_plane = _FakeControlPlane()
        scheduler = SimpleNamespace(
            outbox=queue.Queue(),
            inbox=queue.Queue(),
            abort=lambda request_id: None,
        )
        relay = _AbortOnReadRelay(lambda: None)
        stage = Stage(
            name="vocoder",
            role="single",
            get_next=lambda request_id, output: None,
            gpu_id=None,
            endpoints={"tts_engine": "inproc://tts_engine"},
            control_plane=control_plane,
            relay=relay,
            scheduler=scheduler,
        )
        stage._aborted.add("req")
        payload = StagePayload(
            request_id="req",
            request=OmniRequest(inputs="hello"),
            data={},
        )

        await stage._on_data_ready(await _make_relay_payload(relay, payload))

        assert scheduler.inbox.empty()
        assert relay.gets == 1

    asyncio.run(_run())


def test_stage_routes_relay_stream_chunk_to_scheduler() -> None:
    async def _run() -> None:
        control_plane = _FakeControlPlane()
        codes = torch.arange(2048, dtype=torch.float32)
        scheduler = SimpleNamespace(
            outbox=queue.Queue(),
            inbox=queue.Queue(),
            abort=lambda request_id: None,
        )
        relay = ShmRelay(engine_id="t", device="cpu")
        stage = Stage(
            name="vocoder",
            role="single",
            get_next=lambda request_id, output: None,
            gpu_id=None,
            endpoints={"tts_engine": "inproc://tts_engine"},
            control_plane=control_plane,
            relay=relay,
            scheduler=scheduler,
        )
        stage._stream_queue = StreamQueue(max_pending=4096)
        stage._stream_queue.open("req")

        await stage._on_stream_chunk(
            await _make_relay_chunk(
                relay,
                request_id="req",
                from_stage="tts_engine",
                to_stage="vocoder",
                chunk_id=0,
                data=codes,
                metadata={"modality": "audio_codes"},
            )
        )

        queued = scheduler.inbox.get_nowait()
        assert queued.request_id == "req"
        assert queued.type == "stream_chunk"
        assert torch.equal(queued.data.data, codes)
        assert queued.data.metadata == {"modality": "audio_codes"}

    asyncio.run(_run())


def test_stage_drops_payload_after_abort_during_relay_read() -> None:
    async def _run() -> None:
        control_plane = _FakeControlPlane()
        scheduler = SimpleNamespace(
            outbox=queue.Queue(),
            inbox=queue.Queue(),
            abort=lambda request_id: None,
        )
        relay = _AbortOnReadRelay(lambda: stage._on_abort("req"))
        stage = Stage(
            name="vocoder",
            role="single",
            get_next=lambda request_id, output: None,
            gpu_id=None,
            endpoints={"tts_engine": "inproc://tts_engine"},
            control_plane=control_plane,
            relay=relay,
            scheduler=scheduler,
        )
        payload = StagePayload(
            request_id="req",
            request=OmniRequest(inputs="hello"),
            data={},
        )

        await stage._on_data_ready(await _make_relay_payload(relay, payload))

        assert scheduler.inbox.empty()

    asyncio.run(_run())
