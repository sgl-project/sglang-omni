# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from sglang_omni.models.nemotron3_5_asr import request_builders
from sglang_omni.proto import OmniRequest, StagePayload


def _payload(**params) -> StagePayload:
    return StagePayload(
        request_id="request-1",
        request=OmniRequest(inputs=b"audio", params=params),
        data=None,
    )


def _builder(monkeypatch):
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
    builder, calls = _builder(monkeypatch)

    request = builder(_payload(language="EN-us", temperature=0))

    assert request.language == "en-US"
    assert request.duration_s == 0.5
    assert request.waveform.shape == (8000,)
    assert calls["kwargs"] == {
        "source_name": "Nemotron 3.5 ASR",
        "target_sample_rate": 16000,
    }


def test_builder_defaults_missing_or_empty_language_to_auto(monkeypatch) -> None:
    builder, _ = _builder(monkeypatch)

    assert builder(_payload()).language == "auto"
    assert builder(_payload(language="  ")).language == "auto"


def test_builder_rejects_unknown_language_before_model_inference(monkeypatch) -> None:
    builder, _ = _builder(monkeypatch)

    with pytest.raises(ValueError, match="Unknown language"):
        builder(_payload(language="xx-XX"))


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
    builder, _ = _builder(monkeypatch)

    with pytest.raises(ValueError, match=message):
        builder(_payload(**params))
