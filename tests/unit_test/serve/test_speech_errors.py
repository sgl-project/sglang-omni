# SPDX-License-Identifier: Apache-2.0
"""Speech API error mapping helpers."""

from __future__ import annotations

import pytest

from sglang_omni.admission import QueueFullError
from sglang_omni.serve.speech_errors import speech_generation_error


@pytest.mark.parametrize(
    "exc",
    [QueueFullError(), RuntimeError(QueueFullError.MESSAGE)],
)
def test_speech_generation_error_maps_queue_full_to_503(exc: BaseException) -> None:
    err = speech_generation_error(exc)
    assert err.status_code == 503
    assert QueueFullError.MESSAGE in err.message


def test_speech_generation_error_keeps_other_failures_as_500() -> None:
    err = speech_generation_error(RuntimeError("cuda out of memory"))
    assert err.status_code == 500
    assert "cuda out of memory" in err.message


@pytest.mark.parametrize(
    "message",
    [
        "The request is longer than the model's context length",
        "Requested token count exceeds the model's maximum context length",
        "Request requires more tokens than the thinker KV cache can hold",
        "Request req-1 exceeds the maximum number of tokens: 8193 > 8192",
        "Request req-1 requires too many SWA KV tokens for decode preallocation",
    ],
)
def test_speech_generation_error_maps_context_rejection_to_400(
    message: str,
) -> None:
    err = speech_generation_error(RuntimeError(message))

    assert err.status_code == 400
    assert err.error_type == "BadRequestError"
    assert err.code == 400
    assert err.message == message


def test_speech_generation_error_does_not_match_unrelated_token_message() -> None:
    err = speech_generation_error(
        RuntimeError(
            "kernel assertion: Request req-1 exceeds the maximum number of "
            "tokens: temporary buffer"
        )
    )

    assert err.status_code == 500
    assert err.error_type == "server_error"


@pytest.mark.parametrize("params", [{}, {"nfe": 1}, {"gen_seconds": -1}])
def test_auk_validation_reaches_http_as_bad_request(params, caplog):
    from fastapi.testclient import TestClient

    from sglang_omni.client.client import Client
    from sglang_omni.client.types import ClientError
    from sglang_omni.models.auk.hf_config import AuKRuntimeConfig
    from sglang_omni.models.auk.request_builders import build_auk_state
    from sglang_omni.proto import StagePayload
    from sglang_omni.serve import create_app

    class PreprocessingClient:
        async def speech(self, request, *, request_id, **kwargs):
            payload = StagePayload(
                request_id=request_id,
                request=Client._build_omni_request(request),
                data={},
            )
            try:
                build_auk_state(payload, AuKRuntimeConfig(model_path="unused"))
            except ValueError as error:
                raise ClientError(str(error)) from error
            raise AssertionError("invalid request reached generation")

    client = TestClient(create_app(PreprocessingClient(), model_name="tencent/AuK"))
    response = client.post(
        "/v1/audio/speech",
        json={
            "input": "Hello.",
            "stage_params": {"auk_engine": params},
        },
    )
    assert response.status_code == 400
    assert response.json()["error"]["type"] == "BadRequestError"
    assert "AuK" in response.json()["error"]["message"]
    assert not any(record.exc_info for record in caplog.records)


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize(
    "reference_fields, message",
    [
        ({"ref_text": "Reference transcript."}, "ref_text was given without"),
        (
            {"references": [{"text": "Reference transcript."}]},
            "reference audio is missing",
        ),
    ],
)
def test_voxcpm2_missing_reference_reaches_http_as_bad_request(
    stream, reference_fields, message, caplog
):
    from fastapi.testclient import TestClient

    from sglang_omni.client.client import Client
    from sglang_omni.client.types import ClientError
    from sglang_omni.models.voxcpm2.hf_config import VoxCPM2RuntimeConfig
    from sglang_omni.models.voxcpm2.request_builders import (
        VoxCPM2PreprocessingContext,
        build_voxcpm2_state,
    )
    from sglang_omni.proto import StagePayload
    from sglang_omni.serve import create_app

    class PreprocessingClient:
        def validate(self, request, request_id):
            payload = StagePayload(
                request_id=request_id,
                request=Client._build_omni_request(request),
                data={},
            )
            context = VoxCPM2PreprocessingContext(
                config=VoxCPM2RuntimeConfig(model_path="unused"), tokenizer=None
            )
            try:
                build_voxcpm2_state(payload, context)
            except ValueError as error:
                raise ClientError(str(error)) from error
            raise AssertionError("invalid request reached generation")

        async def speech(self, request, *, request_id, **kwargs):
            self.validate(request, request_id)

        async def generate(self, request, request_id=None):
            self.validate(request, request_id)
            yield  # Make this the same async iterator interface as Client.generate.

        async def abort(self, request_id):
            pass

    client = TestClient(create_app(PreprocessingClient(), model_name="openbmb/VoxCPM2"))
    response = client.post(
        "/v1/audio/speech",
        json={
            "input": "Hello.",
            "stream": stream,
            "response_format": "pcm" if stream else "wav",
            **reference_fields,
        },
    )
    assert response.status_code == 400
    assert response.json()["error"]["type"] == "BadRequestError"
    assert message in response.json()["error"]["message"]
    assert not any(record.exc_info for record in caplog.records)
