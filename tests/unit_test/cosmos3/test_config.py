# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from sglang_omni.models.cosmos3.config import (
    Cosmos3PipelineConfig,
    Cosmos3TextPipelineConfig,
)
from sglang_omni.models.registry import PIPELINE_CONFIG_REGISTRY


def test_pipeline_has_separate_vision_encoder_topology() -> None:
    config = Cosmos3PipelineConfig(model_path="nvidia/Cosmos3-Nano")

    assert [stage.name for stage in config.stages] == [
        "preprocessing",
        "vision_encoder",
        "thinker",
        "decode",
    ]
    assert config.stages[0].next == ["vision_encoder", "thinker"]
    assert config.stages[1].next == "thinker"
    assert config.stages[2].wait_for == ["preprocessing", "vision_encoder"]
    assert config.stages[2].wait_for_fn.endswith(".resolve_thinker_wait_sources")
    assert config.stages[2].merge_fn.endswith(".merge_for_thinker")
    assert config.stages[2].tp_size == 1
    assert config.stages[2].next == "decode"
    assert config.stages[2].stream_to == ["decode"]
    assert config.stages[3].terminal is True
    assert config.stages[3].can_accept_stream_before_payload is True


def test_cosmos_architecture_is_discoverable() -> None:
    assert (
        PIPELINE_CONFIG_REGISTRY.get_config("Cosmos3ForConditionalGeneration")
        is Cosmos3PipelineConfig
    )


def test_text_variant_does_not_load_vision_stage() -> None:
    config = Cosmos3TextPipelineConfig(model_path="nvidia/Cosmos3-Nano")

    assert [stage.name for stage in config.stages] == [
        "preprocessing",
        "thinker",
        "decode",
    ]
    assert config.stages[1].wait_for is None
    assert config.stages[1].merge_fn is None
    assert config.generation_sglang_role_to_stage() == {"generation": "thinker"}
