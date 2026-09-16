# SPDX-License-Identifier: Apache-2.0
"""Diarization result transport, endpoint admission, and disconnect cleanup."""

import asyncio

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from sglang_omni.client import Client, GenerateChunk, GenerateRequest
from sglang_omni.serve import create_app
from sglang_omni.serve.diarizations import complete_diarization


class RecordingCoordinator:
    def __init__(self, result):
        self.result = result
        self.requests = []

    async def submit(self, request_id, request):
        self.requests.append((request_id, request))
        return self.result


def make_client(result, architecture="SortformerEncLabelModel"):
    coordinator = RecordingCoordinator(result)
    app = create_app(
        Client(coordinator), model_name="diarizer", architectures=[architecture]
    )
    return TestClient(app), coordinator


@pytest.mark.parametrize(
    "segments",
    [
        [],
        [
            {"start": 0.0, "end": 1.5, "speaker": "speaker_0"},
            {"start": 1.0, "end": 2.0, "speaker": "speaker_7"},
        ],
    ],
)
def test_structured_results_survive_client_completion_and_http(segments):
    result = {"duration": 2.0, "segments": segments}
    client, coordinator = make_client({"diarization": result})
    response = client.post(
        "/v1/audio/diarizations", files={"file": ("test.wav", b"RIFF")}
    )
    assert response.status_code == 200
    assert response.json() == result
    assert len(coordinator.requests) == 1
    request_id, request = coordinator.requests[0]
    assert response.headers["x-request-id"] == request_id
    assert request.inputs["audio_bytes"] == b"RIFF"
    assert request.metadata["task"] == "diarization"


@pytest.mark.parametrize(
    "data",
    [
        {"stream": "true"},
        {"response_format": "text"},
        {"language": "en"},
        {"temperature": "0"},
    ],
)
def test_unsupported_diarization_controls_are_rejected_before_dispatch(data):
    client, coordinator = make_client({})
    response = client.post(
        "/v1/audio/diarizations", data=data, files={"file": ("test.wav", b"RIFF")}
    )
    assert response.status_code == 400
    assert coordinator.requests == []


def test_transcription_architecture_cannot_silently_return_empty_diarization():
    client, coordinator = make_client(
        {"text": "hello"}, architecture="WhisperForConditionalGeneration"
    )
    response = client.post(
        "/v1/audio/diarizations", files={"file": ("test.wav", b"RIFF")}
    )
    assert response.status_code == 400
    assert coordinator.requests == []


def test_missing_backend_result_is_an_error():
    client, _ = make_client({"text": ""})
    response = client.post(
        "/v1/audio/diarizations", files={"file": ("test.wav", b"RIFF")}
    )
    assert response.status_code == 500


@pytest.mark.parametrize(
    "audio, data, status", [(b"", {}, 400), (b"RIFF", {"model": "unknown"}, 404)]
)
def test_invalid_upload_or_model_is_rejected_before_dispatch(audio, data, status):
    client, coordinator = make_client({})
    response = client.post(
        "/v1/audio/diarizations", files={"file": ("test.wav", audio)}, data=data
    )
    assert response.status_code == status
    assert not coordinator.requests


def test_audio_decode_error_crosses_client_boundary_as_bad_request():
    from sglang_omni.client import ClientError

    class FailingCoordinator:
        async def submit(self, request_id, request):
            raise ClientError("could not decode the uploaded audio")

    app = create_app(
        Client(FailingCoordinator()), architectures=["SortformerEncLabelModel"]
    )
    response = TestClient(app).post(
        "/v1/audio/diarizations", files={"file": ("bad.wav", b"bad")}
    )
    assert response.status_code == 400


def test_existing_text_chunk_serialization_does_not_gain_a_null_field():
    assert "diarization" not in GenerateChunk(request_id="text", text="hello").to_dict()


def test_disconnect_cancels_completion_and_aborts_engine_request():
    class DisconnectedRequest:
        async def is_disconnected(self):
            await client.started.wait()
            return True

    class PendingClient:
        aborted = []
        started = asyncio.Event()
        closed = False

        async def completion(self, request, *, request_id):
            self.started.set()
            try:
                await asyncio.Event().wait()
            finally:
                self.closed = True

        async def abort(self, request_id):
            self.aborted.append(request_id)

    client = PendingClient()
    with pytest.raises(HTTPException) as error:
        asyncio.run(
            complete_diarization(
                DisconnectedRequest(),
                client,
                GenerateRequest(stream=False),
                "disconnected",
            )
        )
    assert error.value.status_code == 499
    assert client.aborted == ["disconnected"]
    assert client.closed
