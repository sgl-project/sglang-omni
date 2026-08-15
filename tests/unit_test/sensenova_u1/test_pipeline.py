# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json

from sglang_omni.config.manager import ConfigManager
from sglang_omni.models.registry import PIPELINE_CONFIG_REGISTRY
from sglang_omni.models.sensenova_u1.config import (
    SenseNovaU1FlowPipelineConfig,
    SenseNovaU1InterleavePipelineConfig,
    SenseNovaU1NativeServingPipelineConfig,
    SenseNovaU1PipelineConfig,
    Variants,
)


def _stage_names(config) -> list[str]:
    return [stage.name for stage in config.stages]


def test_sensenova_u1_pipeline_configs_register_architecture_and_alias() -> None:
    assert PIPELINE_CONFIG_REGISTRY.get_config("NEOChatModel") is SenseNovaU1PipelineConfig
    assert PIPELINE_CONFIG_REGISTRY.get_config("SenseNovaU1") is SenseNovaU1PipelineConfig

    default = SenseNovaU1PipelineConfig(model_path="model")
    flow = SenseNovaU1FlowPipelineConfig(model_path="model")
    interleave = SenseNovaU1InterleavePipelineConfig(model_path="model")
    native_serving = SenseNovaU1NativeServingPipelineConfig(model_path="model")

    assert _stage_names(default) == ["u1_interleave"]
    assert _stage_names(flow) == ["u1_flow"]
    assert _stage_names(interleave) == ["u1_interleave"]
    assert default.mem_fraction_role_to_stage() == {"interleave": "u1_interleave"}
    assert flow.mem_fraction_role_to_stage() == {"generation": "u1_flow"}
    assert interleave.mem_fraction_role_to_stage() == {"interleave": "u1_interleave"}
    assert default.stages[0].terminal is True
    assert flow.stages[0].terminal is True
    assert interleave.stages[0].terminal is True
    assert default.stages[0].runtime.resources.total_gpu_memory_fraction == 0.75
    assert default.stages[0].factory.endswith(
        ".stages.create_sensenova_u1_interleave_executor"
    )
    assert interleave.stages[0].factory.endswith(
        ".stages.create_sensenova_u1_interleave_executor"
    )
    serving_args = native_serving.stages[0].factory_args
    assert serving_args["max_running_requests"] == 16
    assert serving_args["max_concurrency"] == 16
    assert serving_args["enable_cuda_graph"] is True
    assert serving_args["cuda_graph_bs"] == [1, 8, 16]
    assert serving_args["eager_prefix_cache_max_entries"] == 4
    assert serving_args["eager_decode_graph_cache_max_entries"] == 2
    assert serving_args["eager_decode_graph_max_captures"] == 4
    assert serving_args["eager_prefix_cache_max_tokens"] == 2048
    assert serving_args["eager_decode_graph_max_total_tokens"] == 1024
    assert default.stages[0].factory_args["max_total_tokens"] == 4096
    assert flow.stages[0].factory_args["max_total_tokens"] == 4096


def test_sensenova_u1_config_manager_resolves_raw_hf_config_and_variant(tmp_path) -> None:
    (tmp_path / "config.json").write_text(
        json.dumps({"architectures": ["NEOChatModel"], "model_type": "neo_chat"})
    )

    manager = ConfigManager.from_model_path(str(tmp_path), variant="interleave")
    default_manager = ConfigManager.from_model_path(str(tmp_path))

    assert isinstance(manager.config, SenseNovaU1InterleavePipelineConfig)
    assert isinstance(default_manager.config, SenseNovaU1PipelineConfig)
    assert manager.config.model_path == str(tmp_path)
    assert default_manager.config.entry_stage == "u1_interleave"
    assert manager.config.entry_stage == "u1_interleave"
    assert Variants["default"] is SenseNovaU1PipelineConfig
    assert Variants["flow"] is SenseNovaU1FlowPipelineConfig
    assert Variants["interleave"] is SenseNovaU1InterleavePipelineConfig
    assert all(
        "create_sensenova_u1_vqa_executor" not in stage.factory
        for variant in Variants.values()
        for stage in variant(model_path="model").stages
    )
