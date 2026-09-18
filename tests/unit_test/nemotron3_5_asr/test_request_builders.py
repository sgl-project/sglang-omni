# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from sglang_omni.models.nemotron3_5_asr import request_builders
from sglang_omni.proto import OmniRequest, StagePayload


def make_payload(**params) -> StagePayload:
    return StagePayload(
        request_id="request-1",
        request=OmniRequest(inputs=b"audio", params=params),
        data=None,
    )


def make_builder(monkeypatch):
    calls: dict[str, object] = {}

    def fake_prepare(payload, **kwargs):
        calls["payload"] = payload
        calls["kwargs"] = kwargs
        return SimpleNamespace(
            waveform=np.arange(8000, dtype=np.float32),
            duration_s=0.5,
        )

    monkeypatch.setattr(request_builders, "prepare_audio", fake_prepare)
    builder = request_builders.make_nemotron3_5_asr_request_builder(
        prompt_dictionary={"auto": 101, "en-US": 0, "en": 0, "zh-CN": 4}
    )
    return builder, calls


def test_builder_reuses_16khz_preparation_and_preserves_duration(monkeypatch) -> None:
    builder, calls = make_builder(monkeypatch)

    request = builder(make_payload(language="EN-us", temperature=0))

    assert request.language == "en-US"
    assert request.duration_s == 0.5
    assert request.waveform.shape == (8000,)
    assert calls["kwargs"] == {
        "source_name": "Nemotron 3.5 ASR",
        "target_sample_rate": 16000,
        "max_duration_s": None,
    }


def test_builder_defaults_missing_or_empty_language_to_auto(monkeypatch) -> None:
    builder, _ = make_builder(monkeypatch)

    assert builder(make_payload()).language == "auto"
    assert builder(make_payload(language="  ")).language == "auto"


def test_builder_rejects_unknown_language_before_model_inference(monkeypatch) -> None:
    builder, _ = make_builder(monkeypatch)

    with pytest.raises(ValueError, match="Unsupported language"):
        builder(make_payload(language="xx-XX"))


@pytest.mark.parametrize(
    ("params", "message"),
    [
        ({"temperature": 0.1}, "greedy"),
        ({"prompt": "context"}, "text prompt"),
        ({"task": "translate"}, "transcription only"),
    ],
)
def test_builder_rejects_unsupported_generation_modes(
    monkeypatch, params, message
) -> None:
    builder, _ = make_builder(monkeypatch)

    with pytest.raises(ValueError, match=message):
        builder(make_payload(**params))


@pytest.mark.parametrize("waveform", [np.array([]), np.array([float("nan")])])
def test_builder_rejects_empty_or_nonfinite_audio(monkeypatch, waveform):
    monkeypatch.setattr(
        request_builders,
        "prepare_audio",
        lambda *args, **kwargs: SimpleNamespace(waveform=waveform, duration_s=0),
    )
    builder = request_builders.make_nemotron3_5_asr_request_builder(
        prompt_dictionary={"auto": 101}
    )
    with pytest.raises(ValueError, match="requires"):
        builder(make_payload())


@pytest.mark.parametrize(
    "params",
    [
        {"language": "xx-XX"},
        {"temperature": 0.5},
        {"prompt": "context"},
        {"task": "translate"},
    ],
)
def test_validation_errors_survive_string_only_worker_transport(monkeypatch, params):
    from sglang_omni.serve.openai_errors import is_bad_request_error

    builder, _ = make_builder(monkeypatch)
    with pytest.raises(ValueError) as caught:
        builder(make_payload(**params))
    assert is_bad_request_error(RuntimeError(str(caught.value)))
    assert not is_bad_request_error(RuntimeError("Metal allocation failed"))


@pytest.mark.parametrize("decode_error", [True, False])
def test_audio_error_mapping_preserves_operational_failures(monkeypatch, decode_error):
    from sglang_omni.serve.openai_errors import is_bad_request_error
    from sglang_omni.utils.audio import AudioDecodeError

    error = (
        AudioDecodeError("invalid container")
        if decode_error
        else RuntimeError("decoder allocation failed")
    )

    def fail(*args, **kwargs):
        raise error

    monkeypatch.setattr(request_builders, "prepare_audio", fail)
    builder = request_builders.make_nemotron3_5_asr_request_builder(
        prompt_dictionary={"auto": 101}
    )
    with pytest.raises(Exception) as caught:
        builder(make_payload())
    assert is_bad_request_error(RuntimeError(str(caught.value))) == decode_error
    if not decode_error:
        assert caught.value is error
    else:
        assert caught.value.__cause__ is error
