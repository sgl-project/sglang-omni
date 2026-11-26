# SPDX-License-Identifier: Apache-2.0
"""Audio8's request validation before preprocessing or generation."""

from collections.abc import Mapping

import pytest

from sglang_omni.client.client import build_params
from sglang_omni.models.audio8_tts.config import Audio8TTSPipelineConfig
from sglang_omni.models.audio8_tts.payload_types import Audio8SamplingOptions
from sglang_omni.models.audio8_tts.speech_request import Audio8SpeechRequestValidator
from sglang_omni.serve.speech_errors import SpeechAPIError

MODEL_PATH = "Edge0/Audio8-TTS-Realistic-Air"


@pytest.fixture
def validator() -> Audio8SpeechRequestValidator:
    config = Audio8TTSPipelineConfig(model_path=MODEL_PATH)
    return Audio8SpeechRequestValidator(
        default_model=MODEL_PATH,
        supports_uploaded_voice_references=config.supports_uploaded_voice_references(),
        speech_reference_text_required=config.speech_reference_text_required,
        supported_speech_fields=config.supported_speech_fields,
    )


def test_sampling_overrides_reach_the_engine(
    validator: Audio8SpeechRequestValidator,
) -> None:
    """Explicit overrides survive lowering, while omitted options use model defaults."""
    overrides = {"temperature": 0, "top_k": -1, "top_p": 1, "seed": 42}
    for payload in ({"input": "Hello."}, {"input": "Hello.", **overrides}):
        request = validator.parse_raw_request(payload)
        engine_params = build_params(validator.build_generate_request(request))
        expected_overrides = overrides if "seed" in payload else {}
        assert {
            name: engine_params[name] for name in overrides if name in engine_params
        } == expected_overrides
        sampling = Audio8SamplingOptions.model_validate(engine_params)
        assert sampling.temperature == (0 if expected_overrides else 0.7)


@pytest.mark.parametrize(
    ("fields", "parameter"),
    [
        ({"stream": True}, "stream"),
        ({"stream_format": "sse"}, "stream"),
        ({"voice": "unavailable-voice"}, "voice"),
        ({"ref_text": "Hello."}, "ref_audio"),
        ({"references": [{"text": "Hello."}]}, "references.0"),
        (
            {"references": [{"text": "Hello.", "vq_codes": [[0]]}]},
            "references.0.vq_codes",
        ),
        ({"temperature": -1}, "temperature"),
        ({"top_p": 0}, "top_p"),
        ({"top_k": 0}, "top_k"),
        ({"seed": 2**64}, "seed"),
        (
            {
                "ref_audio": "/missing.wav",
                "references": [{"audio_path": "/also-missing.wav", "text": "Hello."}],
            },
            "references",
        ),
    ],
)
def test_unsupported_options_fail_before_preprocessing(
    validator: Audio8SpeechRequestValidator,
    fields: Mapping[str, bool | int | str | list[dict[str, str | list[list[int]]]]],
    parameter: str,
) -> None:
    with pytest.raises(SpeechAPIError) as error:
        validator.parse_generation_request({"input": "Hello.", **fields})
    assert error.value.status_code == 400
    assert error.value.param == parameter
