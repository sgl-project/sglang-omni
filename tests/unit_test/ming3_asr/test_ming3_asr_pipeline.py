# SPDX-License-Identifier: Apache-2.0
"""Pipeline registration and backbone configuration contracts."""

import pytest

from sglang_omni.models.ming3_asr.config import (
    Ming3ASRFactoryArgs,
    Ming3ASRPipelineConfig,
)
from sglang_omni.models.ming3_asr.stages import Ming3ASRConfig, create_thinker_executor
from sglang_omni.models.model_capabilities import get_model_capabilities
from sglang_omni.models.registry import PIPELINE_CONFIG_REGISTRY

CHECKPOINT_ARCHITECTURE = "MingV3ASRForConditionalGeneration"
HIDDEN_SIZE = 64
NATIVE_CLIP_SECONDS = 30.0


def test_checkpoint_architecture_resolves_to_ming3_asr_pipeline() -> None:
    assert (
        PIPELINE_CONFIG_REGISTRY.get_config(CHECKPOINT_ARCHITECTURE)
        is Ming3ASRPipelineConfig
    )
    capabilities = get_model_capabilities(CHECKPOINT_ARCHITECTURE)
    assert capabilities is not None
    assert capabilities.supports_reference_audio is False


def test_default_pipeline_ends_at_thinker() -> None:
    config = Ming3ASRPipelineConfig(model_path="inclusionAI/Ming-3.0-asr-tiny")
    assert [stage.name for stage in config.stages] == [
        "preprocessing",
        "thinker",
    ]
    assert config.terminal_stages == ["thinker"]
    thinker = config.stages[1]
    assert isinstance(thinker.factory, Ming3ASRFactoryArgs)
    assert thinker.factory.pre_lm_max_batch_size == 8
    assert thinker.factory.pre_lm_max_batch_wait_ms == 2


def two_stage_document(
    *,
    pre_lm_max_batch_size: int = 4,
    pre_lm_max_batch_wait_ms: int = 0,
) -> dict[str, object]:
    return {
        "model_path": "inclusionAI/Ming-3.0-asr-tiny",
        "stages": [
            {
                "name": "preprocessing",
                "process": "preprocessing",
                "factory_path": (
                    "sglang_omni.models.ming3_asr.stages.create_preprocessing_executor"
                ),
                "next": "thinker",
            },
            {
                "name": "thinker",
                "process": "thinker",
                "gpu": 0,
                "terminal": True,
                "factory_path": (
                    "sglang_omni.models.ming3_asr.stages.create_thinker_executor"
                ),
                "factory": {
                    "pre_lm_max_batch_size": pre_lm_max_batch_size,
                    "pre_lm_max_batch_wait_ms": pre_lm_max_batch_wait_ms,
                },
            },
        ],
    }


def test_two_stage_configuration_loads_pre_lm_knobs() -> None:
    config = Ming3ASRPipelineConfig.model_validate(two_stage_document())
    assert [stage.name for stage in config.stages] == ["preprocessing", "thinker"]
    thinker = config.stages[1]
    assert thinker.factory.pre_lm_max_batch_size == 4
    assert thinker.factory.pre_lm_max_batch_wait_ms == 0


@pytest.mark.parametrize(
    ("batch_size", "wait_ms", "match"),
    [
        (0, 2, "pre_lm_max_batch_size"),
        (8, -1, "pre_lm_max_batch_wait_ms"),
    ],
)
def test_configuration_rejects_invalid_pre_lm_knobs(
    batch_size: int,
    wait_ms: int,
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        Ming3ASRPipelineConfig.model_validate(
            two_stage_document(
                pre_lm_max_batch_size=batch_size,
                pre_lm_max_batch_wait_ms=wait_ms,
            )
        )


@pytest.mark.parametrize(
    ("batch_size", "wait_ms", "match"),
    [
        (0, 2, "pre_lm_max_batch_size"),
        (8, -1, "pre_lm_max_batch_wait_ms"),
    ],
)
def test_thinker_factory_rejects_invalid_pre_lm_knobs_before_load(
    batch_size: int,
    wait_ms: int,
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        create_thinker_executor(
            "missing-checkpoint",
            max_seq_len=8,
            max_new_tokens=4,
            pre_lm_max_batch_size=batch_size,
            pre_lm_max_batch_wait_ms=wait_ms,
            request_build_max_workers=1,
            request_build_max_pending=1,
        )


def test_thinker_factory_rejects_tensor_parallelism_before_load() -> None:
    with pytest.raises(ValueError, match="tp_size"):
        create_thinker_executor(
            "missing-checkpoint",
            max_seq_len=8,
            max_new_tokens=4,
            pre_lm_max_batch_size=8,
            pre_lm_max_batch_wait_ms=2,
            request_build_max_workers=1,
            request_build_max_pending=1,
            tp_size=2,
        )


def test_long_uploads_split_into_native_clips() -> None:
    config = Ming3ASRPipelineConfig(model_path="inclusionAI/Ming-3.0-asr-tiny")
    chunking = config.resolved_audio_chunking
    assert chunking.allow_audio_chunking is True
    assert chunking.max_native_clip_s == NATIVE_CLIP_SECONDS


def test_backbone_config_flattens_text_config() -> None:
    config = Ming3ASRConfig(
        text_config={
            "hidden_size": HIDDEN_SIZE,
            "model_type": "bailing_moe_v3",
        },
        audio_token_id=7,
    )
    assert config.model_type == "ming_v3_asr"
    assert config.hidden_size == HIDDEN_SIZE
