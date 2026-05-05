# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import asyncio
import queue
from types import SimpleNamespace

import pytest
import torch
from pydantic import ValidationError

from sglang_omni_v1.config.schema import StageConfig
from sglang_omni_v1.models.fishaudio_s2_pro.config import S2ProPipelineConfig
from sglang_omni_v1.pipeline import relay_io
from sglang_omni_v1.pipeline.stage.runtime import Stage
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
    def close(self) -> None:
        pass

    def cleanup(self, request_id: str) -> None:
        pass


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


def test_explicit_scheduler_stream_target_keeps_stage_to_stage_routing(
    monkeypatch,
) -> None:
    async def _run() -> None:
        sent = []

        async def _fake_send_stream_chunk(*args, **kwargs):
            sent.append(kwargs)

        monkeypatch.setattr(
            "sglang_omni_v1.pipeline.stage.runtime.relay_io.send_stream_chunk",
            _fake_send_stream_chunk,
        )
        control_plane = _FakeControlPlane()
        scheduler = SimpleNamespace(outbox=queue.Queue())
        stage = Stage(
            name="tts_engine",
            role="single",
            get_next=lambda request_id, output: None,
            gpu_id=None,
            endpoints={"vocoder": "inproc://vocoder"},
            control_plane=control_plane,
            relay=_FakeRelay(),
            scheduler=scheduler,
        )
        stage._active_requests.add("req")
        scheduler.outbox.put(
            OutgoingMessage(
                request_id="req",
                type="stream",
                data="codes",
                target="vocoder",
                metadata={"modality": "audio_codes"},
            )
        )

        await stage._drain_outbox_external()

        assert control_plane.streams == []
        assert len(sent) == 1
        assert sent[0]["target_stage"] == "vocoder"
        assert sent[0]["data"] == "codes"
        assert sent[0]["metadata"] == {"modality": "audio_codes"}
        assert sent[0]["same_gpu_targets"] == set()

    asyncio.run(_run())


def test_stage_passes_same_gpu_targets_to_relay_io(monkeypatch) -> None:
    async def _run() -> None:
        sent = []

        async def _fake_send_stream_chunk(*args, **kwargs):
            sent.append(kwargs)

        monkeypatch.setattr(
            "sglang_omni_v1.pipeline.stage.runtime.relay_io.send_stream_chunk",
            _fake_send_stream_chunk,
        )
        control_plane = _FakeControlPlane()
        scheduler = SimpleNamespace(outbox=queue.Queue())
        stage = Stage(
            name="tts_engine",
            role="single",
            get_next=lambda request_id, output: None,
            gpu_id=None,
            endpoints={"vocoder": "inproc://vocoder"},
            control_plane=control_plane,
            relay=_FakeRelay(),
            scheduler=scheduler,
            same_gpu_targets={"vocoder"},
        )
        stage._active_requests.add("req")
        scheduler.outbox.put(
            OutgoingMessage(
                request_id="req",
                type="stream",
                data="codes",
                target="vocoder",
            )
        )

        await stage._drain_outbox_external()

        assert len(sent) == 1
        assert sent[0]["same_gpu_targets"] == {"vocoder"}

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
    assert not hasattr(tts_stage, "stream_" + "transport")


def test_stream_chunk_selector_uses_relay_for_spawned_process(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        relay_io.multiprocessing,
        "current_process",
        lambda: SimpleNamespace(name="SpawnProcess-1"),
    )

    assert not relay_io._should_use_stream_ipc(
        target_stage="vocoder",
        same_gpu_targets={"vocoder"},
    )


def test_stream_chunk_selector_allows_ipc_for_main_process_same_gpu(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        relay_io.multiprocessing,
        "current_process",
        lambda: SimpleNamespace(name="MainProcess"),
    )

    assert relay_io._should_use_stream_ipc(
        target_stage="vocoder",
        same_gpu_targets={"vocoder"},
    )


def test_stream_chunk_selector_uses_relay_for_non_same_gpu(monkeypatch) -> None:
    monkeypatch.setattr(
        relay_io.multiprocessing,
        "current_process",
        lambda: SimpleNamespace(name="MainProcess"),
    )

    assert not relay_io._should_use_stream_ipc(
        target_stage="vocoder",
        same_gpu_targets={"other"},
    )


def test_stage_drops_stream_chunk_after_abort_during_relay_read(monkeypatch) -> None:
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
            relay=_FakeRelay(),
            scheduler=scheduler,
        )
        stage._stream_queue = None

        async def _abort_then_return(*args, **kwargs):
            del args, kwargs
            stage._on_abort("req")
            return torch.empty(11, 1, dtype=torch.long)

        monkeypatch.setattr(
            "sglang_omni_v1.pipeline.stage.runtime.relay_io.read_blob",
            _abort_then_return,
        )

        await stage._on_stream_chunk(
            DataReadyMessage(
                request_id="req",
                from_stage="tts_engine",
                to_stage="vocoder",
                shm_metadata={
                    "relay_info": {"transfer_info": {"size": 1}},
                    "tensor_shape": [11, 1],
                    "tensor_dtype": "torch.int64",
                },
                chunk_id=0,
            )
        )

        assert scheduler.inbox.empty()

    asyncio.run(_run())


def test_stage_drops_payload_after_abort_during_relay_read(monkeypatch) -> None:
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
            relay=_FakeRelay(),
            scheduler=scheduler,
        )

        async def _abort_then_return(*args, **kwargs):
            del args, kwargs
            stage._on_abort("req")
            return StagePayload(
                request_id="req",
                request=OmniRequest(inputs="hello"),
                data={},
            )

        monkeypatch.setattr(
            "sglang_omni_v1.pipeline.stage.runtime.relay_io.read_payload",
            _abort_then_return,
        )

        await stage._on_data_ready(
            DataReadyMessage(
                request_id="req",
                from_stage="tts_engine",
                to_stage="vocoder",
                shm_metadata={},
            )
        )

        assert scheduler.inbox.empty()

    asyncio.run(_run())
