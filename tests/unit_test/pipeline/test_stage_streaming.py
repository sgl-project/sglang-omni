# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import asyncio
import base64
import pickle
import queue
from types import SimpleNamespace

import pytest
import torch
from pydantic import ValidationError

from sglang_omni_v1.config.schema import StageConfig
from sglang_omni_v1.models.fishaudio_s2_pro.config import S2ProPipelineConfig
from sglang_omni_v1.pipeline import relay_io
from sglang_omni_v1.pipeline.stage.runtime import Stage
from sglang_omni_v1.pipeline.stage.stream_queue import StreamQueue
from sglang_omni_v1.proto import DataReadyMessage, OmniRequest, StagePayload
from sglang_omni_v1.scheduling.messages import OutgoingMessage


class _FakeControlPlane:
    recv_endpoint = "inproc://stage"

    def __init__(self) -> None:
        self.streams = []
        self.stage_messages = []

    async def start(self) -> None:
        pass

    def close(self) -> None:
        pass

    async def send_stream(self, msg) -> None:
        self.streams.append(msg)

    async def send_to_stage(self, target, endpoint, msg) -> None:
        self.stage_messages.append((target, endpoint, msg))


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

    async def get_async(self, metadata, dest_tensor, request_id):
        del metadata, dest_tensor, request_id
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
        )
        stage._active_requests.add("req")
        scheduler.outbox.put(
            OutgoingMessage(
                request_id="req",
                type="stream",
                data={"audio_data": [0.1], "modality": "audio"},
            )
        )

        await stage._drain_outbox_external()

        assert len(control_plane.streams) == 1
        msg = control_plane.streams[0]
        assert msg.request_id == "req"
        assert msg.from_stage == "vocoder"
        assert msg.chunk == {"audio_data": [0.1], "modality": "audio"}
        assert msg.modality == "audio"

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
    assert tts_stage.stream_to == ["vocoder"]
    assert "stream_transport" not in StageConfig.model_fields


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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_send_stream_chunk_uses_relay_for_cuda_same_gpu_chunk() -> None:
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
            same_gpu_targets={"vocoder"},
        )

        assert len(relay.puts) == 1
        assert len(control_plane.stage_messages) == 1
        _, _, msg = control_plane.stage_messages[0]
        assert "_ipc" not in msg.shm_metadata
        assert msg.shm_metadata["chunk_metadata"] == {"modality": "audio_codes"}

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
