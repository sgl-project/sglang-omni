# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import io
import wave

import numpy as np
import pytest

from sglang_omni.models.parakeet.request_builders import (
    build_parakeet_result,
    make_parakeet_request_builder,
    validate_parakeet_params,
)
from sglang_omni.proto import OmniRequest, StagePayload
from sglang_omni.serve.openai_errors import is_bad_request_error


def wav_bytes(seconds: float, sample_rate: int = 16000) -> bytes:
    samples = (np.sin(np.arange(int(seconds * sample_rate)) / 8.0) * 8000).astype(
        np.int16
    )
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(sample_rate)
        writer.writeframes(samples.tobytes())
    return buffer.getvalue()


def make_payload(audio: bytes, **params: object) -> StagePayload:
    return StagePayload(
        request_id="req-1",
        request=OmniRequest(
            inputs={"audio_bytes": audio},
            params={"task": "transcribe", "temperature": 0.0, **params},
        ),
        data=None,
    )


def test_validate_params_accepts_whisper_style_defaults() -> None:
    assert validate_parakeet_params({"temperature": 0.0, "task": "transcribe"}) is None
    assert validate_parakeet_params({"language": " fr "}) == "fr"
    assert validate_parakeet_params({"language": "  "}) is None
    assert validate_parakeet_params({"prompt": ""}) is None


@pytest.mark.parametrize(
    "params, message",
    [
        ({"temperature": 0.4}, "greedy decoding only"),
        ({"prompt": "names: Ada"}, "does not support a text prompt"),
        ({"task": "translate"}, "transcription only"),
        ({"max_new_tokens": 8}, "does not support max_new_tokens"),
        ({"language": 3}, "string language only"),
    ],
)
def test_validate_params_rejects_unsupported_requests_as_bad_requests(
    params: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message) as info:
        validate_parakeet_params(params)
    assert is_bad_request_error(info.value)


def test_request_builder_decodes_and_resamples_audio() -> None:
    builder = make_parakeet_request_builder(sample_rate=16000)
    request = builder(make_payload(wav_bytes(0.5, sample_rate=8000), language="en"))

    assert request.waveform.dtype == np.float32
    assert request.waveform.shape == (8000,)
    assert request.duration_s == pytest.approx(0.5)
    assert request.language == "en"


def test_request_builder_reports_undecodable_audio_as_bad_request() -> None:
    builder = make_parakeet_request_builder(sample_rate=16000)
    with pytest.raises(ValueError, match="could not decode") as info:
        builder(make_payload(b"definitely not audio" * 64))
    assert is_bad_request_error(info.value)


def test_result_carries_text_and_timing() -> None:
    builder = make_parakeet_request_builder(sample_rate=16000)
    request = builder(make_payload(wav_bytes(0.25), language="de"))
    result = build_parakeet_result(request, text="Hallo", model_latency_s=0.125)

    assert result.request_id == "req-1"
    assert result.data["text"] == "Hallo"
    assert result.data["language"] == "de"
    assert result.data["duration_s"] == pytest.approx(0.25)
    assert result.data["usage"] == {"engine_time_s": 0.125}
    assert result.data["modality"] == "text"
    assert result.data["asr_latency_s"] >= 0.0
