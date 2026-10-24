# SPDX-License-Identifier: Apache-2.0
"""Pipeline registration and backbone configuration contracts."""

from sglang_omni.models.ming3_tts.config import Ming3TTSPipelineConfig
from sglang_omni.models.ming3_tts.sglang_model import Ming3TTSConfig
from sglang_omni.models.model_capabilities import get_model_capabilities
from sglang_omni.models.registry import PIPELINE_CONFIG_REGISTRY

CHECKPOINT_ARCHITECTURE = "LingV3TTSForConditionalGeneration"
HIDDEN_SIZE = 64


def test_checkpoint_architecture_resolves_to_ming3_tts_pipeline() -> None:
    assert (
        PIPELINE_CONFIG_REGISTRY.get_config(CHECKPOINT_ARCHITECTURE)
        is Ming3TTSPipelineConfig
    )
    capabilities = get_model_capabilities(CHECKPOINT_ARCHITECTURE)
    assert capabilities is not None
    assert capabilities.supports_reference_audio is True


def test_default_pipeline_ends_at_vocoder() -> None:
    config = Ming3TTSPipelineConfig(model_path="inclusionAI/Ming-3.0-tts-tiny")
    assert [stage.name for stage in config.stages] == [
        "preprocessing",
        "tts_engine",
        "vocoder",
    ]
    assert config.terminal_stages == ["vocoder"]


def test_backbone_config_flattens_llm_config_and_keeps_speech_sections() -> None:
    config = Ming3TTSConfig(
        llm_config={
            "hidden_size": HIDDEN_SIZE,
            "model_type": "bailing_moe_v3",
        },
        tts_config={"latent_dim": 4},
        flow_config={"depth": 1},
        aggregator_config={"depth": 1},
    )
    assert config.model_type == "lingv3_tts"
    assert config.hidden_size == HIDDEN_SIZE
    assert config.tts_config == {"latent_dim": 4}
    assert config.flow_config == {"depth": 1}
    assert config.aggregator_config == {"depth": 1}
