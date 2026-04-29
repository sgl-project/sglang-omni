# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import asyncio
import queue
from types import SimpleNamespace

from sglang_omni_v1.pipeline.stage.runtime import Stage
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

    asyncio.run(_run())
