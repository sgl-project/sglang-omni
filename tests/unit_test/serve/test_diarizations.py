# SPDX-License-Identifier: Apache-2.0
"""Upload and live diarization endpoints through the real client."""

import asyncio
from typing import TypedDict

import pytest
from fastapi.testclient import TestClient
from httpx import Response

from sglang_omni.client.client import Client
from sglang_omni.proto.request import OmniRequest
from sglang_omni.serve.diarization_ws import DiarizationSession
from sglang_omni.serve.openai_api import create_app

ARCHITECTURE = "Nemotron3DiarizationForAudioFrameClassification"
ServerEvent = dict[str, str | float | list[dict[str, float | str]]]


class DiarizationPayload(TypedDict):
    duration: float
    segments: list[dict[str, float | str]]


class BackendResult(TypedDict, total=False):
    diarization: DiarizationPayload
    text: str


class Coordinator:
    def __init__(
        self,
        result: BackendResult | None = None,
        stages: tuple[str, ...] | list[str] = ("diarization",),
    ) -> None:
        self.result = result
        self.stages = list(stages)
        self.requests: list[tuple[str, OmniRequest]] = []
        self.sessions: dict[str, float] = {}
        self.aborted: list[str] = []

    def health(self) -> dict[str, list[str] | str]:
        return {"stages": self.stages, "entry_stage": "diarization"}

    async def submit(self, request_id: str, request: OmniRequest) -> BackendResult:
        self.requests.append((request_id, request))
        if self.result is not None:
            return self.result
        else:
            pass
        session_id = request.inputs["session_id"]
        operation = request.inputs["operation"]
        duration_s = 0.0
        if operation == "open":
            self.sessions[session_id] = 0.0
        elif operation == "reset":
            self.sessions.pop(session_id, None)
            self.sessions[request.inputs["next_session_id"]] = 0.0
        elif operation == "append":
            self.sessions[session_id] += len(request.inputs["pcm"]) / 32000
            duration_s = self.sessions[session_id]
        elif operation == "finish":
            duration_s = self.sessions.pop(session_id)
        else:
            self.sessions.pop(session_id, None)
        return {"diarization": {"duration": duration_s, "segments": []}}

    async def abort(self, request_id: str) -> None:
        self.aborted.append(request_id)

    def operations(self) -> list[str]:
        return [request.inputs["operation"] for _, request in self.requests]


def connect(coordinator: Coordinator, architecture: str = ARCHITECTURE) -> TestClient:
    app = create_app(
        Client(coordinator), model_name="diarizer", architectures=[architecture]
    )
    return TestClient(app)


def upload(
    coordinator: Coordinator,
    architecture: str = ARCHITECTURE,
    form_fields: dict[str, str] | None = None,
) -> Response:
    return connect(coordinator, architecture).post(
        "/v1/audio/diarizations",
        data=form_fields or {},
        files={"file": ("test.wav", b"RIFF")},
    )


def test_upload_returns_overlapping_intervals_with_request_identity() -> None:
    result: DiarizationPayload = {
        "duration": 2.0,
        "segments": [
            {"start": 0.0, "end": 1.5, "speaker": "speaker_0"},
            {"start": 1.0, "end": 2.0, "speaker": "speaker_7"},
        ],
    }
    coordinator = Coordinator({"diarization": result})
    response = upload(coordinator)
    assert response.status_code == 200
    assert response.json() == result
    [(request_id, request)] = coordinator.requests
    assert response.headers["x-request-id"] == request_id
    assert request.inputs["audio_bytes"] == b"RIFF"
    assert request.metadata["task"] == "diarization"


@pytest.mark.parametrize(
    "form_fields, status",
    [
        ({"stream": "true"}, 400),
        ({"response_format": "text"}, 400),
        ({"language": "en"}, 400),
        ({"model": "unknown"}, 404),
    ],
)
def test_unsupported_controls_or_models_are_rejected_before_dispatch(
    form_fields: dict[str, str], status: int
) -> None:
    coordinator = Coordinator({})
    assert upload(coordinator, form_fields=form_fields).status_code == status
    assert coordinator.requests == []


def test_missing_backend_result_is_an_error() -> None:
    assert upload(Coordinator({"text": ""})).status_code == 500


@pytest.mark.parametrize(
    "architecture, stages",
    [
        ("Qwen3ASRForConditionalGeneration", ["diarization"]),
        (ARCHITECTURE, ["diarization_0", "diarization_1"]),
    ],
    ids=["other-architecture", "process-replicas"],
)
def test_unsupported_servers_reject_diarization_without_dispatch(
    architecture: str, stages: list[str]
) -> None:
    coordinator = Coordinator(stages=stages)
    with connect(coordinator, architecture) as client:
        with client.websocket_connect("/v1/audio/diarizations/stream") as websocket:
            assert websocket.receive_json()["type"] == "error"
            assert websocket.receive()["code"] == 1008
    if architecture != ARCHITECTURE:
        assert upload(coordinator, architecture).status_code == 400
    else:
        pass
    assert coordinator.requests == []


def test_live_updates_precede_eof_and_reset_replaces_the_session() -> None:
    coordinator = Coordinator()
    with connect(coordinator) as client:
        with client.websocket_connect("/v1/audio/diarizations/stream") as websocket:
            ready = websocket.receive_json()
            assert ready["type"] == "session.ready"
            websocket.send_bytes(b"\0\0" * 8000)
            assert websocket.receive_json() == {
                "type": "diarization.update",
                "start": 0,
                "end": 0.5,
                "segments": [],
            }
            assert websocket.receive_json()["type"] == "audio.ack"
            websocket.send_json({"type": "session.reset"})
            reset = websocket.receive_json()
            assert reset["type"] == "session.ready"
            assert reset["session_id"] != ready["session_id"]
            websocket.send_bytes(b"\0\0" * 160)
            assert websocket.receive_json()["start"] == 0
            assert websocket.receive_json()["type"] == "audio.ack"
            websocket.send_json({"type": "audio.end"})
            assert websocket.receive_json() == {
                "type": "diarization.done",
                "duration": 0.01,
            }
            assert websocket.receive()["type"] == "websocket.close"
    assert coordinator.sessions == {}
    assert coordinator.operations() == [
        "open",
        "append",
        "reset",
        "append",
        "finish",
        "close",
    ]


@pytest.mark.parametrize("invalid", [b"x", b"\0" * 32002, {"type": "unknown"}])
def test_invalid_messages_close_the_session_without_dispatching_audio(
    invalid: bytes | dict[str, str],
) -> None:
    coordinator = Coordinator()
    with connect(coordinator) as client:
        with client.websocket_connect("/v1/audio/diarizations/stream") as websocket:
            assert websocket.receive_json()["type"] == "session.ready"
            if isinstance(invalid, bytes):
                websocket.send_bytes(invalid)
            else:
                websocket.send_json(invalid)
            assert websocket.receive_json()["type"] == "error"
            assert websocket.receive()["type"] == "websocket.close"
    assert coordinator.sessions == {}
    assert coordinator.operations() == ["open", "close"]


def test_canceled_admission_settles_before_session_can_be_closed() -> None:
    async def scenario() -> None:
        started, release = asyncio.Event(), asyncio.Event()

        class SlowOpen(Coordinator):
            async def submit(
                self, request_id: str, request: OmniRequest
            ) -> BackendResult:
                if request.inputs["operation"] == "open":
                    started.set()
                    await release.wait()
                else:
                    pass
                return await super().submit(request_id, request)

        coordinator = SlowOpen()
        session = DiarizationSession(None, client=Client(coordinator), model_name="m")
        pending = asyncio.create_task(session.request("open"))
        await started.wait()
        pending.cancel()
        await asyncio.sleep(0)
        # note (Ryan Stewart): Admission must settle before teardown continues.
        assert not pending.done()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await pending
        await session.request("close")
        assert coordinator.operations() == ["open", "close"]
        assert coordinator.sessions == {}

    asyncio.run(scenario())


def test_queue_overflow_aborts_pending_work_and_releases_session() -> None:
    async def scenario() -> None:
        ready, started = asyncio.Event(), asyncio.Event()
        outputs: list[ServerEvent] = []

        class SlowAppend(Coordinator):
            async def submit(
                self, request_id: str, request: OmniRequest
            ) -> BackendResult:
                if request.inputs["operation"] == "append":
                    started.set()
                    await asyncio.Event().wait()
                else:
                    pass
                return await super().submit(request_id, request)

        class Socket:
            receive_count = 0

            async def receive(self) -> dict[str, str | bytes]:
                await ready.wait()
                if self.receive_count:
                    await started.wait()
                else:
                    pass
                self.receive_count += 1
                return {"type": "websocket.receive", "bytes": b"\0\0" * 160}

            async def send_json(self, event: ServerEvent) -> None:
                outputs.append(event)
                if event["type"] == "session.ready":
                    ready.set()
                else:
                    pass

            async def close(self) -> None:
                pass

        coordinator = SlowAppend()
        session = DiarizationSession(
            Socket(), client=Client(coordinator), model_name="m"
        )
        await asyncio.wait_for(session.run(), 2)
        assert any(
            event["type"] == "error" and "queue is full" in event["message"]
            for event in outputs
        )
        assert len(coordinator.aborted) == 1
        assert coordinator.sessions == {}
        assert session.messages.qsize() <= 8

    asyncio.run(scenario())
