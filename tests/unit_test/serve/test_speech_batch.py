# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import logging
from typing import Any

import pytest
from fastapi.testclient import TestClient

from sglang_omni.client import ClientError
from sglang_omni.client.types import SpeechResult
from sglang_omni.serve import create_app
from sglang_omni.serve.openai_api import _create_speech_batch_with_disconnect_watch
from sglang_omni.serve.speech_service import SpeechRequestValidator


class RecordingBatchSpeechClient:
    def __init__(self) -> None:
        self.requests: list[Any] = []

    def health(self) -> dict[str, Any]:
        return {"running": True}

    async def speech(
        self,
        request: Any,
        *,
        request_id: str,
        response_format: str = "wav",
        speed: float = 1.0,
        allow_format_fallback: bool = True,
    ) -> SpeechResult:
        del request_id, speed, allow_format_fallback
        self.requests.append(request)
        return SpeechResult(
            audio_bytes=f"audio:{request.prompt}".encode(),
            mime_type=f"audio/{response_format}",
            format=response_format,
        )


class BlockingBatchSpeechClient:
    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.aborted: list[str] = []

    async def speech(
        self,
        request: Any,
        *,
        request_id: str,
        response_format: str = "wav",
        speed: float = 1.0,
        allow_format_fallback: bool = True,
    ) -> SpeechResult:
        del request, request_id, response_format, speed, allow_format_fallback
        self.started.set()
        await asyncio.Future()
        raise AssertionError("unreachable")

    async def abort(self, request_id: str) -> None:
        self.aborted.append(request_id)


class FailingAbortBatchSpeechClient(BlockingBatchSpeechClient):
    async def abort(self, request_id: str) -> None:
        self.aborted.append(request_id)
        raise RuntimeError("abort failed")


class DisconnectingBatchRequest:
    def __init__(self, client_impl: BlockingBatchSpeechClient) -> None:
        self.client_impl = client_impl

    async def is_disconnected(self) -> bool:
        return self.client_impl.started.is_set()


class MixedBatchSpeechClient:
    def __init__(self) -> None:
        self.requests: list[str] = []

    def health(self) -> dict[str, Any]:
        return {"running": True}

    async def speech(
        self,
        request: Any,
        *,
        request_id: str,
        response_format: str = "wav",
        speed: float = 1.0,
        allow_format_fallback: bool = True,
    ) -> SpeechResult:
        del request_id, speed, allow_format_fallback
        self.requests.append(request.prompt)
        if request.prompt == "slow":
            await asyncio.sleep(0.01)
        if request.prompt == "fail":
            raise ClientError("model failed")
        return SpeechResult(
            audio_bytes=f"audio:{request.prompt}".encode(),
            mime_type=f"audio/{response_format}",
            format=response_format,
        )


class CountingReferenceSpeechRequestValidator(SpeechRequestValidator):
    def __init__(self) -> None:
        super().__init__(default_model="tts")
        self.reference_loads: list[str] = []

    def _load_media_reference_descriptor(
        self, value: str, *, param: str
    ) -> dict[str, str]:
        self.reference_loads.append(value)
        return {"data": "UklGRg==", "media_type": "audio/wav"}


def test_batch_speech_preserves_order_and_item_errors() -> None:
    client_impl = RecordingBatchSpeechClient()
    client = TestClient(
        create_app(client_impl, model_name="tts", tts_batch_max_items=5)
    )

    response = client.post(
        "/v1/audio/speech/batch",
        json={
            "response_format": "wav",
            "items": [
                {"input": "first"},
                {"input": "   "},
                {"input": "third", "response_format": "pcm"},
                {"input": 123},
                {"response_format": "wav"},
            ],
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["total"] == 5
    assert body["succeeded"] == 2
    assert body["failed"] == 3
    assert [item["index"] for item in body["results"]] == [0, 1, 2, 3, 4]
    assert body["results"][0]["status"] == "success"
    assert "success" not in body["results"][0]
    assert body["results"][1]["status"] == "error"
    assert "success" not in body["results"][1]
    assert body["results"][1]["error"]["param"] == "items.1.input"
    assert body["results"][2]["media_type"] == "audio/pcm"
    assert body["results"][3]["error"]["param"] == "items.3.input"
    assert body["results"][4]["error"]["param"] == "items.4.input"
    assert [request.prompt for request in client_impl.requests] == ["first", "third"]


def test_batch_speech_rejects_invalid_envelope_before_item_work() -> None:
    client_impl = RecordingBatchSpeechClient()
    client = TestClient(
        create_app(client_impl, model_name="tts", tts_batch_max_items=1)
    )

    response = client.post(
        "/v1/audio/speech/batch",
        json={"items": [{"input": "one"}, {"input": "two"}]},
    )

    assert response.status_code == 400
    assert response.json()["error"]["param"] == "items"
    assert client_impl.requests == []


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("speed", "1.2"),
        ("max_new_tokens", "5"),
        ("token_count", "5"),
        ("duration_tokens", "5"),
        ("x_vector_only_mode", "true"),
    ],
)
def test_batch_speech_rejects_stringified_default_types(
    field_name: str, value: str
) -> None:
    client_impl = RecordingBatchSpeechClient()
    client = TestClient(create_app(client_impl, model_name="tts"))

    response = client.post(
        "/v1/audio/speech/batch",
        json={
            field_name: value,
            "items": [{"input": "one"}],
        },
    )

    assert response.status_code == 400
    assert response.json()["error"]["param"] == field_name
    assert client_impl.requests == []


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("token_count", 0),
        ("token_count", -1),
        ("duration_tokens", 0),
        ("duration_tokens", -1),
    ],
)
def test_batch_speech_rejects_non_positive_default_duration_fields(
    field_name: str, value: int
) -> None:
    client_impl = RecordingBatchSpeechClient()
    client = TestClient(create_app(client_impl, model_name="tts"))

    response = client.post(
        "/v1/audio/speech/batch",
        json={field_name: value, "items": [{"input": "one"}]},
    )

    assert response.status_code == 400
    assert response.json()["error"]["param"] == field_name
    assert client_impl.requests == []


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("token_count", "5"),
        ("duration_tokens", "5"),
    ],
)
def test_batch_speech_rejects_stringified_item_integer_overrides(
    field_name: str, value: str
) -> None:
    client_impl = RecordingBatchSpeechClient()
    client = TestClient(create_app(client_impl, model_name="tts"))

    response = client.post(
        "/v1/audio/speech/batch",
        json={"items": [{"input": "one", field_name: value}]},
    )

    assert response.status_code == 200
    item = response.json()["results"][0]
    assert item["status"] == "error"
    assert item["error"]["param"] == f"items.0.{field_name}"
    assert client_impl.requests == []


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("token_count", 0),
        ("token_count", -1),
        ("duration_tokens", 0),
        ("duration_tokens", -1),
    ],
)
def test_batch_speech_rejects_non_positive_item_duration_fields(
    field_name: str, value: int
) -> None:
    client_impl = RecordingBatchSpeechClient()
    client = TestClient(create_app(client_impl, model_name="tts"))

    response = client.post(
        "/v1/audio/speech/batch",
        json={"items": [{"input": "one", field_name: value}]},
    )

    assert response.status_code == 200
    item = response.json()["results"][0]
    assert item["status"] == "error"
    assert item["error"]["param"] == f"items.0.{field_name}"
    assert client_impl.requests == []


def test_batch_speech_rejects_streaming_items() -> None:
    client_impl = RecordingBatchSpeechClient()
    client = TestClient(create_app(client_impl, model_name="tts"))

    response = client.post(
        "/v1/audio/speech/batch",
        json={"items": [{"input": "one", "stream": True}]},
    )

    assert response.status_code == 200
    item = response.json()["results"][0]
    assert item["status"] == "error"
    assert item["error"]["param"] == "items.0.stream"
    assert client_impl.requests == []


def test_batch_speech_accepts_item_model_override() -> None:
    client_impl = RecordingBatchSpeechClient()
    client = TestClient(create_app(client_impl, model_name="tts"))

    response = client.post(
        "/v1/audio/speech/batch",
        json={
            "items": [
                {"input": "first"},
                {
                    "input": "wrong model",
                    "model": "other-tts",
                    "unknown_field": "ignored",
                },
                {"input": "third"},
            ]
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["succeeded"] == 3
    assert body["failed"] == 0
    assert [request.prompt for request in client_impl.requests] == [
        "first",
        "wrong model",
        "third",
    ]
    assert client_impl.requests[1].model == "other-tts"


def test_batch_speech_isolates_runtime_failures_and_preserves_order() -> None:
    client_impl = MixedBatchSpeechClient()
    client = TestClient(create_app(client_impl, model_name="tts"))

    response = client.post(
        "/v1/audio/speech/batch",
        json={"items": [{"input": "slow"}, {"input": "fail"}, {"input": "fast"}]},
    )

    assert response.status_code == 200
    results = response.json()["results"]
    assert [item["index"] for item in results] == [0, 1, 2]
    assert [item["status"] for item in results] == ["success", "error", "success"]
    assert all("success" not in item for item in results)
    assert results[1]["error"]["type"] == "server_error"
    assert set(client_impl.requests) == {"slow", "fail", "fast"}


def test_batch_speech_cancellation_aborts_started_items() -> None:
    async def run() -> None:
        service = SpeechRequestValidator(default_model="tts")
        batch = service.parse_batch_request({"items": [{"input": "one"}]})
        client_impl = BlockingBatchSpeechClient()

        task = asyncio.create_task(
            service.create_speech_batch(client_impl, batch, request_id="batch")
        )
        await client_impl.started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        assert client_impl.aborted == ["batch-0"]

    asyncio.run(run())


def test_batch_speech_logs_abort_failures(caplog: pytest.LogCaptureFixture) -> None:
    async def run() -> None:
        service = SpeechRequestValidator(default_model="tts")
        batch = service.parse_batch_request({"items": [{"input": "one"}]})
        client_impl = FailingAbortBatchSpeechClient()

        with caplog.at_level(logging.WARNING):
            task = asyncio.create_task(
                service.create_speech_batch(client_impl, batch, request_id="batch")
            )
            await client_impl.started.wait()
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

        assert client_impl.aborted == ["batch-0"]
        assert "Failed to abort speech batch item request batch-0" in caplog.text

    asyncio.run(run())


def test_batch_speech_request_disconnect_aborts_started_items() -> None:
    async def run() -> None:
        service = SpeechRequestValidator(default_model="tts")
        batch = service.parse_batch_request({"items": [{"input": "one"}]})
        client_impl = BlockingBatchSpeechClient()
        request = DisconnectingBatchRequest(client_impl)

        with pytest.raises(asyncio.CancelledError):
            await _create_speech_batch_with_disconnect_watch(
                request,
                client=client_impl,
                speech_service=service,
                batch=batch,
                request_id="batch",
            )

        assert client_impl.aborted == ["batch-0"]

    asyncio.run(run())


def test_batch_speech_reuses_shared_default_reference_loads() -> None:
    async def run() -> None:
        service = CountingReferenceSpeechRequestValidator()
        client_impl = RecordingBatchSpeechClient()
        batch = service.parse_batch_request(
            {
                "ref_audio": "data:audio/wav;base64,AAAA",
                "items": [
                    {"input": "first"},
                    {"input": "second", "speed": 1.1, "temperature": 0.7},
                    {"input": "override", "ref_audio": "data:audio/wav;base64,BBBB"},
                ],
            }
        )

        response = await service.create_speech_batch(
            client_impl,
            batch,
            request_id="batch",
        )

        assert response.succeeded == 3
        assert service.reference_loads == [
            "data:audio/wav;base64,AAAA",
            "data:audio/wav;base64,BBBB",
        ]
        assert client_impl.requests[1].metadata["tts_params"]["speed"] == 1.1
        assert client_impl.requests[1].sampling.temperature == 0.7

    asyncio.run(run())
