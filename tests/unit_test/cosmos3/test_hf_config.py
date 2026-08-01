# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json

from transformers import AutoConfig
from transformers.models.qwen3_vl.configuration_qwen3_vl import (
    Qwen3VLTextConfig,
    Qwen3VLVisionConfig,
)

from sglang_omni.models.cosmos3.hf_config import Cosmos3OmniConfig


def _nano_config() -> dict:
    return {
        "model_type": "cosmos3_omni",
        "architectures": ["Cosmos3ForConditionalGeneration"],
        "image_token_id": 151655,
        "video_token_id": 151656,
        "vision_start_token_id": 151652,
        "vision_end_token_id": 151653,
        "tie_word_embeddings": False,
        "text_config": {
            "model_type": "qwen3_vl_text",
            "vocab_size": 151936,
            "hidden_size": 4096,
            "intermediate_size": 12288,
            "num_hidden_layers": 36,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "head_dim": 128,
            "max_position_embeddings": 262144,
            "bos_token_id": 151643,
            "eos_token_id": 151645,
            "rope_theta": 5000000,
            "rope_scaling": {
                "mrope_interleaved": True,
                "mrope_section": [24, 20, 20],
                "rope_type": "default",
            },
        },
        "vision_config": {
            "model_type": "qwen3_vl",
            "depth": 27,
            "hidden_size": 1152,
            "intermediate_size": 4304,
            "num_heads": 16,
            "out_hidden_size": 4096,
        },
        "model": {
            "_target": "cosmos_predict2._src.predict2.cosmos3.models.Cosmos3Model",
            "config": {"vlm_config": {"model_name": "Cosmos3-Nano-Reasoner"}},
        },
    }


def test_auto_config_loads_cosmos3_nano_shape(tmp_path) -> None:
    (tmp_path / "config.json").write_text(json.dumps(_nano_config()))

    config = AutoConfig.from_pretrained(tmp_path)

    assert isinstance(config, Cosmos3OmniConfig)
    assert isinstance(config.text_config, Qwen3VLTextConfig)
    assert isinstance(config.vision_config, Qwen3VLVisionConfig)
    assert config.architectures == ["Cosmos3ForConditionalGeneration"]
    assert config.text_config.hidden_size == 4096
    assert config.text_config.intermediate_size == 12288
    assert config.text_config.num_hidden_layers == 36
    assert config.text_config.num_key_value_heads == 8
    assert config.text_config.bos_token_id == 151643
    assert config.text_config.eos_token_id == 151645
    assert config.text_config.rope_scaling["mrope_section"] == [24, 20, 20]
    assert config.model["config"]["vlm_config"]["model_name"] == (
        "Cosmos3-Nano-Reasoner"
    )


def test_cosmos3_config_preserves_top_level_model_type() -> None:
    config = Cosmos3OmniConfig(**_nano_config())

    assert config.model_type == "cosmos3_omni"
    assert config.to_dict()["model_type"] == "cosmos3_omni"
