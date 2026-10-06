# SPDX-License-Identifier: Apache-2.0
"""Audio preparation and validation before model inference."""

from __future__ import annotations

from unittest.mock import Mock

import numpy as np
import pytest

from sglang_omni.models.nemotron3_5_asr import request_builders
from sglang_omni.preprocessing.transcription import PreparedAudio
from sglang_omni.proto.request import OmniRequest, StagePayload


def make_payload(**params: str | float) -> StagePayload:
    return StagePayload(
        request_id="request-1",
        request=OmniRequest(inputs=b"audio", params=params),
        data=None,
    )


def make_builder(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[request_builders.NemotronRequestBuilder, Mock]:
    prepare = Mock(
        spec=request_builders.prepare_audio,
        return_value=PreparedAudio(
            waveform=np.arange(8000, dtype=np.float32),
            sample_rate=16000,
            duration_s=0.5,
            fingerprint="test-audio",
        ),
    )
    monkeypatch.setattr(request_builders, "prepare_audio", prepare)
    builder = request_builders.make_nemotron3_5_asr_request_builder(
        prompt_dictionary={"auto": 101, "en-US": 0, "en": 0, "zh-CN": 4}
    )
    return builder, prepare


def test_builder_reuses_16khz_preparation_and_preserves_duration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    builder, prepare = make_builder(monkeypatch)

    request = builder(make_payload(language="EN-us", temperature=0))

    assert request.language == "en-US"
    assert request.duration_s == 0.5
    assert request.waveform.shape == (8000,)
    assert prepare.call_args.kwargs == {
        "source_name": "Nemotron 3.5 ASR",
        "target_sample_rate": 16000,
    }


def test_builder_defaults_missing_or_empty_language_to_auto(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    builder, _ = make_builder(monkeypatch)

    assert builder(make_payload()).language == "auto"
    assert builder(make_payload(language="  ")).language == "auto"


def test_builder_rejects_unknown_language_before_model_inference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    builder, _ = make_builder(monkeypatch)

    with pytest.raises(ValueError, match="Unknown language"):
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
    monkeypatch: pytest.MonkeyPatch, params: dict[str, str | float], message: str
) -> None:
    builder, _ = make_builder(monkeypatch)

    with pytest.raises(ValueError, match=message):
        builder(make_payload(**params))
