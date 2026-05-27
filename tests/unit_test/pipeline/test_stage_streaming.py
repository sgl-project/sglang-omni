# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import asyncio
import base64
import multiprocessing
import pickle
import queue
import traceback
from types import SimpleNamespace

import pytest
import torch
from pydantic import ValidationError

from sglang_omni.config.schema import StageConfig
from sglang_omni.models.fishaudio_s2_pro.config import S2ProPipelineConfig
from sglang_omni.pipeline import relay_io
from sglang_omni.pipeline.stage.runtime import Stage
from sglang_omni.pipeline.stage.stream_queue import StreamItem, StreamQueue
from sglang_omni.proto import DataReadyMessage, OmniRequest, StagePayload
from sglang_omni.scheduling.messages import OutgoingMessage


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
        self.puts = []

    async def put_async(self, tensor, request_id):
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


def _payload_metadata(payload: StagePayload) -> dict:
    return {
        "payload_pickle": base64.b64encode(pickle.dumps(payload)).decode("ascii"),
        "relay_info": {"transfer_info": {"size": 1}},
        "tensor_info": [],
    }


def _cuda_ipc_stage_receiver(msg_queue, result_queue) -> None:
    try:
        torch.cuda.set_device(0)
        msg = msg_queue.get(timeout=30)

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
                gpu_id=0,
                endpoints={},
                control_plane=_FakeControlPlane(),
                relay=_FakeRelay(),
                scheduler=scheduler,
            )
            stage._stream_queue = StreamQueue(max_pending=4096)
            stage._stream_queue.open("req")

            await stage._on_stream_chunk(msg)

            queued = scheduler.inbox.get_nowait()
            item = queued.data
            result_queue.put(
                {
                    "message_type": queued.type,
                    "chunk_id": item.chunk_id,
                    "data": item.data.detach().cpu().tolist(),
                    "modality": item.metadata["modality"],
                    "nested_hidden": item.metadata["nested"]["layer_hidden"]
                    .detach()
                    .cpu()
                    .tolist(),
                    "pair_tensor": item.metadata["pair"][0].detach().cpu().tolist(),
                    "pair_value": item.metadata["pair"][1],
                }
            )

            del item, queued, stage, scheduler
            torch.cuda.synchronize()

        asyncio.run(_run())
    except BaseException:
        result_queue.put({"error": traceback.format_exc()})


def test_terminal_scheduler_stream_routes_to_coordinator() -> None:
    async def _run() -> None:
        control_plane = _FakeControlPlane()
        scheduler = SimpleNamespace(outbox=queue.Queue())
        stage = Stage(
            name="vocoder",
            role="single",
            get_next=lambda request_id, output: None,
            gpu_id=None,
            endpoints={},
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


def test_explicit_scheduler_stream_target_keeps_stage_to_stage_routing() -> None:
    async def _run() -> None:
        control_plane = _FakeControlPlane()
        relay = _FakeRelay()
        scheduler = SimpleNamespace(outbox=queue.Queue())
        codes = torch.empty(11, 1, dtype=torch.long)
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
        assert msg.shm_metadata["chunk_metadata"] == {"modality": "audio_codes"}

    asyncio.run(_run())


def test_stage_config_rejects_unknown_model_transport_field() -> None:
    field_name = "stream_" + "transport"
    with pytest.raises(ValidationError):
        StageConfig(
            name="tts_engine",
            factory="pkg.create",
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


def test_stage_fails_pre_payload_stream_chunk_by_default() -> None:
    async def _run() -> None:
        control_plane = _FakeControlPlane()
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
            control_plane=control_plane,
            relay=_AbortOnReadRelay(lambda: None),
            scheduler=scheduler,
        )
        stage._stream_queue = StreamQueue(max_pending=4096)
        codes = torch.arange(11, dtype=torch.float32)

        await stage._on_stream_chunk(
            DataReadyMessage(
                request_id="req",
                from_stage="tts_engine",
                to_stage="vocoder",
                shm_metadata=relay_io.serialize_ipc_chunk(codes, None),
                chunk_id=0,
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
        stage = Stage(
            name="vocoder",
            role="single",
            get_next=lambda request_id, output: None,
            gpu_id=None,
            endpoints={},
            control_plane=control_plane,
            relay=_AbortOnReadRelay(lambda: None),
            scheduler=scheduler,
        )
        stage._stream_queue = StreamQueue(max_pending=4096)
        payload = StagePayload(
            request_id="req",
            request=OmniRequest(inputs="hello"),
            data={"ready": True},
        )
        await stage._on_data_ready(
            DataReadyMessage(
                request_id="req",
                from_stage="tts_engine",
                to_stage="vocoder",
                shm_metadata=_payload_metadata(payload),
            )
        )
        codes = torch.arange(11, dtype=torch.float32)
        await stage._on_stream_chunk(
            DataReadyMessage(
                request_id="req",
                from_stage="tts_engine",
                to_stage="vocoder",
                shm_metadata=relay_io.serialize_ipc_chunk(codes, None),
                chunk_id=0,
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
        stage = Stage(
            name="vocoder",
            role="single",
            get_next=lambda request_id, output: None,
            gpu_id=None,
            endpoints={},
            control_plane=control_plane,
            relay=_AbortOnReadRelay(lambda: None),
            scheduler=scheduler,
            can_accept_stream_before_payload=True,
        )
        stage._stream_queue = StreamQueue(max_pending=4096)
        codes = torch.arange(11, dtype=torch.float32)

        await stage._on_stream_chunk(
            DataReadyMessage(
                request_id="req",
                from_stage="tts_engine",
                to_stage="vocoder",
                shm_metadata=relay_io.serialize_ipc_chunk(
                    codes, {"modality": "audio_codes"}
                ),
                chunk_id=0,
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
                shm_metadata={},
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
        await stage._on_data_ready(
            DataReadyMessage(
                request_id="req",
                from_stage="tts_engine",
                to_stage="vocoder",
                shm_metadata=_payload_metadata(payload),
            )
        )
        payload_msg = scheduler.inbox.get_nowait()
        assert payload_msg.request_id == "req"
        assert payload_msg.type == "new_request"
        assert payload_msg.data.data == {"ready": True}

    asyncio.run(_run())


def test_stage_stream_chunk_received_after_ipc_materialization(monkeypatch) -> None:
    """The paired receive event marks scheduler-ready chunks, not control arrival."""
    order: list[str] = []

    def fake_deserialize(msg):
        order.append("deserialized")
        return StreamItem(
            chunk_id=msg.chunk_id,
            data="chunk",
            from_stage=msg.from_stage,
            metadata=None,
        )

    async def fake_route(self, request_id, item):
        del self, request_id, item
        order.append("routed")

    monkeypatch.setattr(Stage, "_deserialize_ipc_chunk", staticmethod(fake_deserialize))
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
            endpoints={},
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
                shm_metadata={"_ipc": True, "tensor_bytes": b"unused"},
                chunk_id=0,
            )
        )

    asyncio.run(_run())

    assert order == ["deserialized", "stage_stream_chunk_received", "routed"]


def test_stage_stream_chunk_received_after_relay_materialization(monkeypatch) -> None:
    """Cross-GPU chunks emit receive only after relay data and metadata are restored."""
    order: list[str] = []

    async def fake_read_blob(relay, key, metadata):
        del relay, key, metadata
        order.append("blob")
        return "chunk"

    async def fake_read_metadata(self, shm_metadata, blob_key):
        del self, shm_metadata, blob_key
        order.append("metadata")
        return {"modality": "audio_codes"}

    async def fake_route(self, request_id, item):
        del self, request_id, item
        order.append("routed")

    monkeypatch.setattr(relay_io, "read_blob", fake_read_blob)
    monkeypatch.setattr(Stage, "_read_chunk_metadata", fake_read_metadata)
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
            endpoints={},
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
                shm_metadata={"relay_info": {}},
                chunk_id=0,
            )
        )

    asyncio.run(_run())

    assert order == ["blob", "metadata", "stage_stream_chunk_received", "routed"]


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
            endpoints={},
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


def test_send_stream_chunk_uses_relay() -> None:
    async def _run() -> None:
        control_plane = _FakeControlPlane()
        relay = _FakeRelay()
        codes = torch.empty(11, 1, dtype=torch.long)

        await relay_io.send_stream_chunk(
            relay,
            control_plane,
            request_id="req",
            data=codes,
            target_stage="vocoder",
            target_endpoint="inproc://vocoder",
            from_stage="tts_engine",
            chunk_id=0,
        )

        assert len(relay.puts) == 1
        assert relay.puts[0][0] == "req:stream:tts_engine:vocoder:0"
        assert len(control_plane.stage_messages) == 1
        _, _, msg = control_plane.stage_messages[0]
        expected_size = codes.contiguous().view(torch.uint8).numel()
        assert msg.shm_metadata["relay_info"] == {
            "transfer_info": {"size": expected_size}
        }

    asyncio.run(_run())


def test_send_stream_chunk_uses_relay_for_cpu_same_gpu_chunk() -> None:
    async def _run() -> None:
        control_plane = _FakeControlPlane()
        relay = _FakeRelay()
        codes = torch.arange(11, dtype=torch.float32)

        await relay_io.send_stream_chunk(
            relay,
            control_plane,
            request_id="req",
            data=codes,
            target_stage="vocoder",
            target_endpoint="inproc://vocoder",
            from_stage="tts_engine",
            chunk_id=0,
            same_gpu_targets={"vocoder"},
        )

        assert len(relay.puts) == 1
        assert len(control_plane.stage_messages) == 1
        _, _, msg = control_plane.stage_messages[0]
        assert "_ipc" not in msg.shm_metadata

    asyncio.run(_run())


def test_send_stream_chunk_uses_ipc_for_detected_cuda_same_gpu_chunk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _run() -> None:
        control_plane = _FakeControlPlane()
        relay = _FakeRelay()
        codes = torch.arange(11, dtype=torch.float32)

        monkeypatch.setattr(relay_io, "_is_cuda_tensor", lambda data: data is codes)
        monkeypatch.setattr(
            relay_io,
            "serialize_ipc_chunk",
            lambda data, metadata: {
                "_ipc": True,
                "data": data.tolist(),
                "metadata": metadata,
            },
        )

        await relay_io.send_stream_chunk(
            relay,
            control_plane,
            request_id="req",
            data=codes,
            target_stage="vocoder",
            target_endpoint="inproc://vocoder",
            from_stage="tts_engine",
            chunk_id=0,
            metadata={"modality": "audio_codes"},
            same_gpu_targets={"vocoder"},
        )

        assert relay.puts == []
        assert len(control_plane.stage_messages) == 1
        _, _, msg = control_plane.stage_messages[0]
        assert msg.shm_metadata == {
            "_ipc": True,
            "data": codes.tolist(),
            "metadata": {"modality": "audio_codes"},
        }

    asyncio.run(_run())


def test_send_stream_chunk_falls_back_to_relay_for_cpu_tensor_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _run() -> None:
        control_plane = _FakeControlPlane()
        relay = _FakeRelay()
        codes = torch.arange(11, dtype=torch.float32)
        metadata = {"modality": "audio_codes", "stats": torch.arange(3)}

        monkeypatch.setattr(relay_io, "_is_cuda_tensor", lambda data: data is codes)
        monkeypatch.setattr(
            relay_io,
            "serialize_ipc_chunk",
            lambda data, metadata: pytest.fail("CPU tensor metadata must use relay"),
        )

        await relay_io.send_stream_chunk(
            relay,
            control_plane,
            request_id="req",
            data=codes,
            target_stage="vocoder",
            target_endpoint="inproc://vocoder",
            from_stage="tts_engine",
            chunk_id=0,
            metadata=metadata,
            same_gpu_targets={"vocoder"},
        )

        assert len(relay.puts) == 2
        assert len(control_plane.stage_messages) == 1
        _, _, msg = control_plane.stage_messages[0]
        assert "_ipc" not in msg.shm_metadata
        assert msg.shm_metadata["chunk_metadata"] == {
            "modality": "audio_codes",
            "stats": {
                "_tensor_placeholder": "stats",
                "shape": [3],
                "dtype": "torch.int64",
                "device": "cpu",
            },
        }
        assert set(msg.shm_metadata["chunk_metadata_tensors"]) == {"stats"}

    asyncio.run(_run())


def test_send_stream_chunk_falls_back_to_relay_for_large_inline_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _run() -> None:
        control_plane = _FakeControlPlane()
        relay = _FakeRelay()
        codes = torch.arange(11, dtype=torch.float32)
        transcript = "x" * (128 * 1024)

        monkeypatch.setattr(relay_io, "_is_cuda_tensor", lambda data: data is codes)
        monkeypatch.setattr(
            relay_io,
            "serialize_ipc_chunk",
            lambda data, metadata: pytest.fail("large metadata must use relay"),
        )

        await relay_io.send_stream_chunk(
            relay,
            control_plane,
            request_id="req",
            data=codes,
            target_stage="vocoder",
            target_endpoint="inproc://vocoder",
            from_stage="tts_engine",
            chunk_id=0,
            metadata={"modality": "audio_codes", "transcript": transcript},
            same_gpu_targets={"vocoder"},
        )

        assert len(relay.puts) == 1
        assert len(control_plane.stage_messages) == 1
        _, _, msg = control_plane.stage_messages[0]
        assert "_ipc" not in msg.shm_metadata
        assert msg.shm_metadata["chunk_metadata"] == {
            "modality": "audio_codes",
            "transcript": transcript,
        }

    asyncio.run(_run())


def test_send_stream_chunk_falls_back_to_relay_for_large_python_container_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _run() -> None:
        control_plane = _FakeControlPlane()
        relay = _FakeRelay()
        codes = torch.arange(11, dtype=torch.float32)
        token_ids = list(range(128 * 1024))

        monkeypatch.setattr(relay_io, "_is_cuda_tensor", lambda data: data is codes)
        monkeypatch.setattr(
            relay_io,
            "serialize_ipc_chunk",
            lambda data, metadata: pytest.fail("large metadata must use relay"),
        )

        await relay_io.send_stream_chunk(
            relay,
            control_plane,
            request_id="req",
            data=codes,
            target_stage="vocoder",
            target_endpoint="inproc://vocoder",
            from_stage="tts_engine",
            chunk_id=0,
            metadata={"modality": "audio_codes", "token_ids": token_ids},
            same_gpu_targets={"vocoder"},
        )

        assert len(relay.puts) == 1
        assert len(control_plane.stage_messages) == 1
        _, _, msg = control_plane.stage_messages[0]
        assert "_ipc" not in msg.shm_metadata
        assert msg.shm_metadata["chunk_metadata"] == {
            "modality": "audio_codes",
            "token_ids": token_ids,
        }

    asyncio.run(_run())


def test_send_stream_chunk_rejects_mixed_cuda_cpu_data_object_graph(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _run() -> None:
        control_plane = _FakeControlPlane()
        relay = _FakeRelay()
        cuda_codes = torch.arange(11, dtype=torch.float32)
        cpu_stats = torch.arange(3)

        monkeypatch.setattr(
            relay_io,
            "_is_cuda_tensor",
            lambda data: data is cuda_codes,
        )
        monkeypatch.setattr(
            relay_io,
            "serialize_ipc_chunk",
            lambda data, metadata: pytest.fail("mixed data must not use IPC"),
        )

        with pytest.raises(ValueError, match="mixed object graphs"):
            await relay_io.send_stream_chunk(
                relay,
                control_plane,
                request_id="req",
                data={"codes": cuda_codes, "stats": cpu_stats},
                target_stage="vocoder",
                target_endpoint="inproc://vocoder",
                from_stage="tts_engine",
                chunk_id=0,
                metadata={"modality": "audio_codes"},
                same_gpu_targets={"vocoder"},
            )

        assert relay.puts == []
        assert control_plane.stage_messages == []

    asyncio.run(_run())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_send_stream_chunk_uses_cuda_ipc_for_cuda_same_gpu_chunk() -> None:
    async def _run() -> None:
        ctx = multiprocessing.get_context("spawn")
        msg_queue = ctx.Queue()
        result_queue = ctx.Queue()
        receiver = ctx.Process(
            target=_cuda_ipc_stage_receiver,
            args=(msg_queue, result_queue),
        )
        receiver.start()

        control_plane = _FakeControlPlane()
        relay = _FakeRelay()
        codes = torch.arange(11, dtype=torch.float32, device="cuda")
        hidden = torch.ones(2, dtype=torch.float32, device="cuda")
        pair_hidden = hidden + 1
        metadata = {
            "modality": "audio_codes",
            "nested": {"layer_hidden": hidden},
            "pair": (pair_hidden, "kept"),
        }

        try:
            await relay_io.send_stream_chunk(
                relay,
                control_plane,
                request_id="req",
                data=codes,
                target_stage="vocoder",
                target_endpoint="inproc://vocoder",
                from_stage="tts_engine",
                chunk_id=0,
                metadata=metadata,
                same_gpu_targets={"vocoder"},
            )

            assert relay.puts == []
            assert len(control_plane.stage_messages) == 1
            _, _, msg = control_plane.stage_messages[0]
            assert msg.shm_metadata["_ipc"] is True
            assert "relay_info" not in msg.shm_metadata

            msg_queue.put(msg)
            result = result_queue.get(timeout=30)
        finally:
            receiver.join(timeout=30)
            if receiver.is_alive():
                receiver.terminate()
                receiver.join(timeout=10)
            msg_queue.close()
            result_queue.close()

        assert receiver.exitcode == 0
        assert "error" not in result, result.get("error")
        assert result == {
            "message_type": "stream_chunk",
            "chunk_id": 0,
            "data": codes.detach().cpu().tolist(),
            "modality": "audio_codes",
            "nested_hidden": hidden.detach().cpu().tolist(),
            "pair_tensor": pair_hidden.detach().cpu().tolist(),
            "pair_value": "kept",
        }

    asyncio.run(_run())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_send_stream_chunk_uses_relay_for_cuda_non_same_gpu_chunk() -> None:
    async def _run() -> None:
        control_plane = _FakeControlPlane()
        relay = _FakeRelay()
        codes = torch.arange(11, dtype=torch.float32, device="cuda")

        await relay_io.send_stream_chunk(
            relay,
            control_plane,
            request_id="req",
            data=codes,
            target_stage="vocoder",
            target_endpoint="inproc://vocoder",
            from_stage="tts_engine",
            chunk_id=0,
            metadata={"modality": "audio_codes"},
            same_gpu_targets={"other_stage"},
        )

        assert len(relay.puts) == 1
        assert len(control_plane.stage_messages) == 1
        _, _, msg = control_plane.stage_messages[0]
        assert "_ipc" not in msg.shm_metadata
        assert msg.shm_metadata["chunk_metadata"] == {"modality": "audio_codes"}

    asyncio.run(_run())


def test_ipc_stream_chunk_survives_control_plane_serialization() -> None:
    pytest.importorskip("msgpack")
    from sglang_omni.pipeline.control_plane import (
        deserialize_message,
        serialize_message,
    )

    codes = torch.arange(3, dtype=torch.float32)
    hidden = torch.tensor([4.0, 5.0])
    msg = DataReadyMessage(
        request_id="req",
        from_stage="tts_engine",
        to_stage="vocoder",
        shm_metadata=relay_io.serialize_ipc_chunk(
            codes,
            {
                "modality": "audio_codes",
                "nested": {"hidden": hidden},
                "items": [hidden + 1],
                "pair": (hidden + 2, "kept"),
            },
        ),
        chunk_id=0,
    )

    round_tripped = deserialize_message(serialize_message(msg))
    item = Stage._deserialize_ipc_chunk(round_tripped)

    assert torch.equal(item.data, codes)
    assert item.metadata["modality"] == "audio_codes"
    assert torch.equal(item.metadata["nested"]["hidden"], hidden)
    assert torch.equal(item.metadata["items"][0], hidden + 1)
    assert torch.equal(item.metadata["pair"][0], hidden + 2)
    assert item.metadata["pair"][1] == "kept"


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
            endpoints={},
            control_plane=control_plane,
            relay=relay,
            scheduler=scheduler,
        )
        stage._stream_queue = None

        await stage._on_stream_chunk(
            DataReadyMessage(
                request_id="req",
                from_stage="tts_engine",
                to_stage="vocoder",
                shm_metadata={
                    "relay_info": {
                        "transfer_info": {
                            "size": codes.contiguous().view(torch.uint8).numel()
                        }
                    },
                    "tensor_shape": list(codes.shape),
                    "tensor_dtype": str(codes.dtype),
                },
                chunk_id=0,
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
            endpoints={},
            control_plane=control_plane,
            relay=relay,
            scheduler=scheduler,
        )
        stage._aborted.add("req")
        size = codes.contiguous().view(torch.uint8).numel()
        metadata = {
            "relay_info": {"transfer_info": {"size": size}},
            "tensor_shape": list(codes.shape),
            "tensor_dtype": str(codes.dtype),
            "chunk_metadata": {"latency": {"_tensor_placeholder": "latency"}},
            "chunk_metadata_tensors": {
                "latency": {
                    "blob_key": "req:stream:tts_engine:vocoder:0:meta:0",
                    "relay_metadata": {
                        "relay_info": {"transfer_info": {"size": 4}},
                        "tensor_shape": [1],
                        "tensor_dtype": "torch.float32",
                    },
                }
            },
        }

        await stage._on_stream_chunk(
            DataReadyMessage(
                request_id="req",
                from_stage="tts_engine",
                to_stage="vocoder",
                shm_metadata=metadata,
                chunk_id=0,
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
            endpoints={},
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

        await stage._on_data_ready(
            DataReadyMessage(
                request_id="req",
                from_stage="tts_engine",
                to_stage="vocoder",
                shm_metadata=_payload_metadata(payload),
            )
        )

        assert scheduler.inbox.empty()
        assert relay.gets == 1

    asyncio.run(_run())


def test_stage_routes_ipc_stream_chunk_to_scheduler() -> None:
    async def _run() -> None:
        control_plane = _FakeControlPlane()
        codes = torch.arange(2048, dtype=torch.float32)
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
            control_plane=control_plane,
            relay=_FakeRelay(),
            scheduler=scheduler,
        )
        stage._stream_queue = StreamQueue(max_pending=4096)
        stage._stream_queue.open("req")

        await stage._on_stream_chunk(
            DataReadyMessage(
                request_id="req",
                from_stage="tts_engine",
                to_stage="vocoder",
                shm_metadata=relay_io.serialize_ipc_chunk(
                    codes, {"modality": "audio_codes"}
                ),
                chunk_id=0,
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
            endpoints={},
            control_plane=control_plane,
            relay=relay,
            scheduler=scheduler,
        )
        payload = StagePayload(
            request_id="req",
            request=OmniRequest(inputs="hello"),
            data={},
        )

        await stage._on_data_ready(
            DataReadyMessage(
                request_id="req",
                from_stage="tts_engine",
                to_stage="vocoder",
                shm_metadata=_payload_metadata(payload),
            )
        )

        assert scheduler.inbox.empty()

    asyncio.run(_run())
