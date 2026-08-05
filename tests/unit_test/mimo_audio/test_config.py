# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json

from sglang_omni.config.manager import resolve_config_cls_for_model_path
from sglang_omni.config.placement import StagePlacementPlanner
from sglang_omni.config.topology import build_process_topology_plan
from sglang_omni.models.mimo_audio.config import (
    MIMO_AUDIO_TOKENIZER_PATH,
    MiMoAudioPipelineConfig,
)
from sglang_omni.models.registry import PIPELINE_CONFIG_REGISTRY


def test_mimo_audio_pipeline_is_registered() -> None:
    assert (
        PIPELINE_CONFIG_REGISTRY.get_config("MiMoAudioModel") is MiMoAudioPipelineConfig
    )


def test_mimo_audio_pipeline_is_text_output_only() -> None:
    config = MiMoAudioPipelineConfig(model_path="XiaomiMiMo/MiMo-Audio-7B-Instruct")

    assert config.entry_stage == "audio_tokenizer"
    assert [stage.name for stage in config.stages] == [
        "audio_tokenizer",
        "thinker",
    ]
    assert config.terminal_stages == ["thinker"]
    assert config.gpu_placement == {"audio_tokenizer": 0, "thinker": 0}
    assert config.stages[0].next == "thinker"
    assert config.stages[0].factory_args["tokenizer_path"] == (
        MIMO_AUDIO_TOKENIZER_PATH
    )
    fractions = {
        stage.name: stage.runtime.resources.total_gpu_memory_fraction
        for stage in config.stages
    }
    assert fractions == {"audio_tokenizer": 0.08, "thinker": 0.92}
    placement = StagePlacementPlanner(config).build()
    topology = build_process_topology_plan(config, placement)
    assert topology.stage_to_process == {
        "audio_tokenizer": "audio_tokenizer",
        "thinker": "thinker",
    }
    assert all(
        forbidden not in stage.name
        for stage in config.stages
        for forbidden in ("decoder", "vocoder", "code2wav", "patch_decoder")
    )


def test_checkpoint_architecture_resolves_to_mimo_pipeline(tmp_path) -> None:
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["MiMoAudioModel"],
                "model_type": "qwen2",
                "hidden_size": 4096,
                "num_hidden_layers": 36,
                "num_attention_heads": 32,
                "num_key_value_heads": 8,
                "vocab_size": 151680,
            }
        )
    )

    assert resolve_config_cls_for_model_path(str(tmp_path)) is MiMoAudioPipelineConfig
