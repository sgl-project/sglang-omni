# SPDX-License-Identifier: Apache-2.0
"""Pipeline registration and backbone configuration contracts."""

from sglang_omni.models.ming3_asr.config import Ming3ASRPipelineConfig
from sglang_omni.models.ming3_asr.stages import Ming3ASRConfig
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
        "audio_encoder",
        "thinker",
    ]
    assert config.terminal_stages == ["thinker"]


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
