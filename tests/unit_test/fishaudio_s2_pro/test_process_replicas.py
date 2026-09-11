# SPDX-License-Identifier: Apache-2.0
"""Fish Audio process-replica factory contracts."""

from pathlib import Path

from sglang_omni.config import ProcessConfig, resolve_stage_factory_args
from sglang_omni.config.manager import ConfigManager
from sglang_omni.config.topology import compile_logical_processes
from sglang_omni.models.fishaudio_s2_pro.config import S2ProPipelineConfig
from sglang_omni.pipeline.replicas import expand_replica_stages
from sglang_omni.pipeline.runtime_config import prepare_pipeline_runtime


def _expanded_replica_stages():
    config = S2ProPipelineConfig(
        model_path="model",
        processes={"pipeline": ProcessConfig(num_replicas=2, replica_devices=[1, 2])},
    )
    process_plan, stages = compile_logical_processes(config)
    expanded, _ = expand_replica_stages(stages, process_plan)
    return config, {stage.name: stage for stage in expanded}


def test_engine_factory_forwards_each_process_replica_gpu_id() -> None:
    config, by_name = _expanded_replica_stages()

    gpu_ids = [
        resolve_stage_factory_args(
            by_name[f"tts_engine@r{replica_id}"],
            config,
            gpu_id=gpu_id,
        )["gpu_id"]
        for replica_id, gpu_id in enumerate((1, 2))
    ]

    assert gpu_ids == [1, 2]


def test_vocoder_factory_accepts_each_process_replica_gpu_id() -> None:
    config, by_name = _expanded_replica_stages()

    gpu_ids = [
        resolve_stage_factory_args(
            by_name[f"vocoder@r{replica_id}"],
            config,
            gpu_id=gpu_id,
        )["gpu_id"]
        for replica_id, gpu_id in enumerate((1, 2))
    ]

    assert gpu_ids == [1, 2]


def test_split_example_preserves_tts_capacity_and_separate_vocoders() -> None:
    path = (
        Path(__file__).resolve().parents[3]
        / "examples/configs/s2pro_tts_replica2_split_h100.yaml"
    )
    config = ConfigManager.from_file(str(path)).config
    prep = prepare_pipeline_runtime(config)
    try:
        assert config.weight_share == "on"
        assert {
            group.name: group.stage_names for group in prep.process_plan.groups
        } == {
            "preprocessing": ("preprocessing",),
            "tts_engine@r0": ("tts_engine@r0",),
            "tts_engine@r1": ("tts_engine@r1",),
            "vocoder@r0": ("vocoder@r0",),
            "vocoder@r1": ("vocoder@r1",),
        }
        assert all(
            group.gpu_id == 0
            for group in prep.process_plan.groups
            if group.name != "preprocessing"
        )
        engines = [stage for stage in prep.stages_cfg if stage.engine_stage]
        assert len(engines) == 2
        assert all(stage.engine.max_total_tokens == 32000 for stage in engines)
        assert all(stage.engine.mem_fraction_static == 0.35 for stage in engines)
    finally:
        prep.runtime_dir.close()
