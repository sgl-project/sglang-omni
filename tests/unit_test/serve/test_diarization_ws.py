# SPDX-License-Identifier: Apache-2.0
"""Live diarization protocol and worker-session cleanup through the real client."""

import asyncio

import pytest
from fastapi.testclient import TestClient

from sglang_omni.client import Client
from sglang_omni.serve import create_app


class SessionCoordinator:
    def __init__(self):
        self.operations = []
        self.sessions = {}

    def health(self):
        return {"stages": ["diarization"], "entry_stage": "diarization"}

    async def submit(self, request_id, request):
        assert request.metadata["task"] == "diarization_stream"
        inputs = request.inputs
        session = inputs["session_id"]
        operation = inputs["operation"]
        self.operations.append((session, operation))
        if operation == "open":
            self.sessions[session] = 0
            duration = 0
        elif operation == "close":
            self.sessions.pop(session, None)
            duration = 0
        else:
            self.sessions[session] += len(inputs["pcm"]) / 32000
            duration = self.sessions[session]
        return {"diarization": {"duration": duration, "segments": []}}

    async def abort(self, request_id):
        pass


def connection(coordinator, architecture="SortformerEncLabelModel"):
    app = create_app(
        Client(coordinator), model_name="diarizer", architectures=[architecture]
    )
    return TestClient(app)


def test_live_updates_precede_eof_and_reset_owns_a_new_session():
    coordinator = SessionCoordinator()
    with connection(coordinator) as client:
        with client.websocket_connect("/v1/audio/diarizations/stream") as ws:
            ready = ws.receive_json()
            assert ready["type"] == "session.ready"
            first_id = ready["session_id"]
            ws.send_bytes(b"\0\0" * 8000)
            update = ws.receive_json()
            assert update == {
                "type": "diarization.update",
                "start": 0,
                "end": 0.5,
                "segments": [],
            }
            assert ws.receive_json()["type"] == "audio.ack"
            ws.send_json({"type": "session.reset"})
            reset = ws.receive_json()
            assert reset["type"] == "session.ready"
            assert reset["session_id"] != first_id
            ws.send_bytes(b"\0\0" * 160)
            assert ws.receive_json()["start"] == 0
            assert ws.receive_json()["type"] == "audio.ack"
            ws.send_json({"type": "audio.end"})
            assert ws.receive_json() == {"type": "diarization.done", "duration": 0.01}
            assert ws.receive()["type"] == "websocket.close"
    assert coordinator.sessions == {}
    assert [op for _, op in coordinator.operations] == [
        "open",
        "append",
        "close",
        "open",
        "append",
        "finish",
        "close",
    ]


@pytest.mark.parametrize(
    "invalid",
    [
        b"x",
        b"",
        b"\0" * 32002,
        {"type": "unknown"},
        {"type": "audio.end", "extra": True},
    ],
)
def test_invalid_messages_close_the_session_without_dispatching_audio(invalid):
    coordinator = SessionCoordinator()
    with connection(coordinator) as client:
        with client.websocket_connect("/v1/audio/diarizations/stream") as ws:
            assert ws.receive_json()["type"] == "session.ready"
            if isinstance(invalid, bytes):
                ws.send_bytes(invalid)
            else:
                ws.send_json(invalid)
            assert ws.receive_json()["type"] == "error"
            assert ws.receive()["type"] == "websocket.close"
    assert coordinator.sessions == {}
    assert [op for _, op in coordinator.operations] == ["open", "close"]


def test_other_architectures_reject_live_diarization_without_dispatch():
    coordinator = SessionCoordinator()
    with connection(coordinator, "Qwen3ASRForConditionalGeneration") as client:
        with client.websocket_connect("/v1/audio/diarizations/stream") as ws:
            assert ws.receive_json()["type"] == "error"
            assert ws.receive()["code"] == 1008
    assert coordinator.operations == []


def test_canceled_admission_settles_before_session_can_be_closed():
    from sglang_omni.serve.diarization_ws import DiarizationSession

    async def scenario():
        started = asyncio.Event()
        release = asyncio.Event()

        class SlowOpen(SessionCoordinator):
            async def submit(self, request_id, request):
                if request.inputs["operation"] == "open":
                    started.set()
                    await release.wait()
                return await super().submit(request_id, request)

        coordinator = SlowOpen()
        session = DiarizationSession(
            None, client=Client(coordinator), model_name="test"
        )
        pending = asyncio.create_task(session.request("open"))
        await started.wait()
        pending.cancel()
        await asyncio.sleep(0)
        assert not pending.done()  # Admission must settle before teardown continues.
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await pending
        await session.request("close")
        assert [op for _, op in coordinator.operations] == ["open", "close"]
        assert coordinator.sessions == {}

    asyncio.run(scenario())


def test_queue_overflow_aborts_pending_work_and_releases_session():
    from sglang_omni.serve.diarization_ws import DiarizationSession

    async def scenario():
        ready = asyncio.Event()
        started = asyncio.Event()
        aborted = []
        outputs = []

        class SlowAppend(SessionCoordinator):
            async def submit(self, request_id, request):
                if request.inputs["operation"] == "append":
                    started.set()
                    await asyncio.Event().wait()
                return await super().submit(request_id, request)

            async def abort(self, request_id):
                aborted.append(request_id)
                return True

        class Socket:
            count = 0

            async def receive(self):
                await ready.wait()
                if self.count:
                    await started.wait()
                self.count += 1
                return {"type": "websocket.receive", "bytes": b"\0\0" * 160}

            async def send_json(self, event):
                outputs.append(event)
                if event["type"] == "session.ready":
                    ready.set()

            async def close(self):
                pass

        coordinator = SlowAppend()
        session = DiarizationSession(
            Socket(), client=Client(coordinator), model_name="test"
        )
        await asyncio.wait_for(session.run(), 2)
        assert any(
            event["type"] == "error" and "queue is full" in event["message"]
            for event in outputs
        )
        assert len(aborted) == 1
        assert coordinator.sessions == {}
        assert session.messages.qsize() <= 8

    asyncio.run(scenario())


def test_replicated_live_pipeline_is_rejected_before_session_admission():
    class ReplicatedCoordinator(SessionCoordinator):
        def health(self):
            return {
                "stages": ["diarization_0", "diarization_1"],
                "entry_stage": "diarization",
            }

    coordinator = ReplicatedCoordinator()
    with connection(coordinator) as client:
        with client.websocket_connect("/v1/audio/diarizations/stream") as ws:
            error = ws.receive_json()
            assert error["type"] == "error" and "replicas" in error["message"]
            assert ws.receive()["code"] == 1008
    assert coordinator.operations == []
