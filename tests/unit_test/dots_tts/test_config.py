# SPDX-License-Identifier: Apache-2.0
"""Pipeline-config contract for dots.tts."""

from __future__ import annotations

from sglang_omni.config.manager import ConfigManager
from sglang_omni.config.placement import build_stage_placement_plan
from sglang_omni.config.topology import build_process_topology_plan
from sglang_omni.models.dots_tts.config import DotsTtsPipelineConfig
from sglang_omni.models.registry import PIPELINE_CONFIG_REGISTRY

MODEL_PATH = "dots-studio/dots.tts-mf"


def test_architecture_is_registered() -> None:
    assert (
        PIPELINE_CONFIG_REGISTRY.get_config("DotsTTSForConditionalGeneration")
        is DotsTtsPipelineConfig
    )


def test_stage_graph_and_speech_contract() -> None:
    config = DotsTtsPipelineConfig(model_path=MODEL_PATH)

    assert [stage.name for stage in config.stages] == [
        "preprocessing",
        "reference_encode",
        "tts_engine",
        "audio_decode",
    ]
    assert config.resolved_entry_stage == "preprocessing"
    assert config.terminal_stages == ["audio_decode"]
    assert config.required_speech_reference_count == 1
    assert config.speech_reference_text_required is True
    assert config.supports_uploaded_voice_references() is True


def test_every_edge_is_process_safe() -> None:
    config = DotsTtsPipelineConfig(model_path=MODEL_PATH)
    edges = {
        (stage.name, stage.next)
        for stage in config.stages
        if isinstance(stage.next, str)
    }

    assert DotsTtsPipelineConfig.process_safe_edges() == edges


def test_example_config_matches_defaults_and_places_on_one_gpu() -> None:
    config = ConfigManager.from_file("examples/configs/dots_tts.yaml").config

    assert isinstance(config, DotsTtsPipelineConfig)
    assert [stage.name for stage in config.stages] == [
        stage.name for stage in DotsTtsPipelineConfig(model_path=MODEL_PATH).stages
    ]
    build_process_topology_plan(config, build_stage_placement_plan(config))
    assert config.gpu_placement == {
        "reference_encode": 0,
        "tts_engine": 0,
        "audio_decode": 0,
    }
