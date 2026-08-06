# SPDX-License-Identifier: Apache-2.0

from typing import Any

import pytest

from sglang_omni.config.runtime import resolve_stage_static_factory_args
from sglang_omni.models.registry import PIPELINE_CONFIG_REGISTRY
from sglang_omni.serve.speech_errors import SpeechAPIError
from sglang_omni.serve.speech_service import SpeechRequestValidator


def test_dots_tts_uses_framework_stage_boundaries() -> None:
    from sglang_omni.models.dots_tts.config import DotsTTSPipelineConfig
    from sglang_omni.models.dots_tts.payload_types import DotsTTSState

    config = DotsTTSPipelineConfig(model_path="model")

    assert [stage.name for stage in config.stages] == [
        "preprocessing",
        "reference_encode",
        "latent_engine",
        "vocoder",
    ]
    assert {stage.process for stage in config.stages} == {"pipeline"}
    assert config.terminal_stages == ["vocoder"]
    assert config.generation_sglang_role_to_stage() == {"generation": "latent_engine"}
    assert config.required_speech_reference_count == 1
    assert config.speech_reference_text_required is True
    assert config.additional_speech_languages == frozenset({"auto_detect"})
    assert (
        PIPELINE_CONFIG_REGISTRY.get_config("DotsTTSForConditionalGeneration")
        is DotsTTSPipelineConfig
    )
    assert DotsTTSState().num_steps == 4


def test_dots_tts_syncs_acoustic_limits_from_latent_engine() -> None:
    from sglang_omni.models.dots_tts.config import DotsTTSPipelineConfig

    config = DotsTTSPipelineConfig(
        model_path="model",
        runtime_overrides={
            "latent_engine": {"num_steps": 8, "max_generate_length": 256}
        },
    )
    stages = {stage.name: stage for stage in config.stages}
    preprocessing = resolve_stage_static_factory_args(stages["preprocessing"], config)

    assert preprocessing["default_num_steps"] == 8
    assert preprocessing["max_generate_length"] == 256


def test_dots_tts_rejects_preprocessing_acoustic_limit_overrides() -> None:
    from sglang_omni.models.dots_tts.config import DotsTTSPipelineConfig

    with pytest.raises(ValueError, match="configure it on the latent_engine stage"):
        DotsTTSPipelineConfig(
            model_path="model",
            runtime_overrides={"preprocessing": {"default_num_steps": 8}},
        )


def test_public_speech_boundary_accepts_dots_auto_detect_alias() -> None:
    from sglang_omni.models.dots_tts.config import DotsTTSPipelineConfig

    config = DotsTTSPipelineConfig(model_path="dots-studio/dots.tts-mf")
    validator = SpeechRequestValidator(
        default_model=config.model_path,
        required_speech_reference_count=config.required_speech_reference_count,
        speech_reference_text_required=config.speech_reference_text_required,
        additional_speech_languages=config.additional_speech_languages,
    )

    request = validator.parse_generation_request(
        {
            "input": "hello",
            "language": "AUTO_DETECT",
            "ref_audio": "data:audio/wav;base64,UklGRg==",
            "ref_text": "reference",
        }
    )

    assert request.request.language == "auto_detect"


@pytest.mark.parametrize(
    ("payload", "param"),
    [
        ({"input": "target"}, "ref_audio"),
        (
            {
                "input": "target",
                "ref_audio": "data:audio/wav;base64,UklGRg==",
            },
            "ref_text",
        ),
        (
            {
                "input": "target",
                "ref_audio": "data:audio/wav;base64,UklGRg==",
                "ref_text": "reference",
                "references": [
                    {
                        "data": "UklGRg==",
                        "media_type": "audio/wav",
                        "text": "reference",
                    }
                ],
            },
            "references",
        ),
    ],
)
def test_public_speech_boundary_enforces_dots_conditioning_contract(
    payload: dict[str, Any], param: str
) -> None:
    from sglang_omni.models.dots_tts.config import DotsTTSPipelineConfig

    config = DotsTTSPipelineConfig(model_path="dots-studio/dots.tts-mf")
    validator = SpeechRequestValidator(
        default_model=config.model_path,
        required_speech_reference_count=config.required_speech_reference_count,
        speech_reference_text_required=config.speech_reference_text_required,
    )

    with pytest.raises(SpeechAPIError) as exc_info:
        validator.parse_generation_request(payload)

    assert exc_info.value.status_code == 400
    assert exc_info.value.param == param


def test_dots_tts_rejects_tp() -> None:
    from sglang_omni.models.dots_tts.config import DotsTTSPipelineConfig

    raw = DotsTTSPipelineConfig(model_path="model").model_dump()
    stage = next(item for item in raw["stages"] if item["name"] == "latent_engine")
    stage["tp_size"] = 2
    stage["parallelism"] = {"tp": 2}
    stage["gpu"] = [0, 1]
    with pytest.raises(ValueError, match="tp_size=1"):
        DotsTTSPipelineConfig(**raw)
