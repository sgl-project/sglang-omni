# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from sglang_omni.models.cosmos3.config import Cosmos3TextPipelineConfig
from sglang_omni.models.registry import PIPELINE_CONFIG_REGISTRY


def test_text_pipeline_has_preprocess_ar_decode_topology() -> None:
    config = Cosmos3TextPipelineConfig(model_path="nvidia/Cosmos3-Nano")

    assert [stage.name for stage in config.stages] == [
        "preprocessing",
        "thinker",
        "decode",
    ]
    assert config.stages[0].next == "thinker"
    assert config.stages[1].tp_size == 1
    assert config.stages[1].next == "decode"
    assert config.stages[1].stream_to == ["decode"]
    assert config.stages[2].terminal is True
    assert config.stages[2].can_accept_stream_before_payload is True


def test_cosmos_architecture_is_discoverable() -> None:
    assert (
        PIPELINE_CONFIG_REGISTRY.get_config("Cosmos3ForConditionalGeneration")
        is Cosmos3TextPipelineConfig
    )
