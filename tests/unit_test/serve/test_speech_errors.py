# SPDX-License-Identifier: Apache-2.0
"""Speech API error mapping helpers."""

from __future__ import annotations

import pytest

from sglang_omni.admission import QueueFullError
from sglang_omni.serve.speech_errors import (
    SpeechAPIError,
    resolve_served_model,
    speech_generation_error,
)


@pytest.mark.parametrize("requested_model", [None, "served-model"])
def test_resolve_served_model_accepts_default(
    requested_model: str | None,
) -> None:
    assert resolve_served_model(requested_model, "served-model") == "served-model"


def test_resolve_served_model_rejects_unknown_name() -> None:
    with pytest.raises(SpeechAPIError) as exc_info:
        resolve_served_model("unknown/model", "served-model")

    error = exc_info.value
    assert error.status_code == 404
    assert error.error_type == "invalid_request_error"
    assert error.param == "model"
    assert error.code == "model_not_found"


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
