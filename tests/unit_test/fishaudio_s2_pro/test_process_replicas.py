# SPDX-License-Identifier: Apache-2.0
"""Fish Audio process-replica factory contracts."""

import pytest

from sglang_omni.config import ProcessConfig, resolve_stage_factory_args
from sglang_omni.config.manager import ConfigManager
from sglang_omni.config.topology import compile_logical_processes
from sglang_omni.models.fishaudio_s2_pro.config import S2ProPipelineConfig
from sglang_omni.pipeline.replicas import expand_replica_stages
from tests.unit_test.pipeline.helpers import build_compiled_process_topology


def expanded_replica_stages():
    config = S2ProPipelineConfig(
        model_path="model",
        processes={"pipeline": ProcessConfig(num_replicas=2, replica_devices=[1, 2])},
    )
    process_plan, stages = compile_logical_processes(config)
    expanded, _ = expand_replica_stages(stages, process_plan)
    return config, {stage.name: stage for stage in expanded}


def test_engine_factory_forwards_each_process_replica_gpu_id() -> None:
    config, by_name = expanded_replica_stages()

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
    config, by_name = expanded_replica_stages()

    gpu_ids = [
        resolve_stage_factory_args(
            by_name[f"vocoder@r{replica_id}"],
            config,
            gpu_id=gpu_id,
        )["gpu_id"]
        for replica_id, gpu_id in enumerate((1, 2))
    ]

    assert gpu_ids == [1, 2]


@pytest.mark.parametrize("gpu_id", [None, 0, 2])
def test_preprocessing_factory_receives_explicit_gpu_placement(
    gpu_id: int | None,
) -> None:
    """GPU reference encoding keeps an explicit, independently budgeted process."""
    manager = ConfigManager(S2ProPipelineConfig(model_path="model"))
    overrides = {
        "tts_engine.gpu_memory_fraction": 0.75,
        "vocoder.gpu_memory_fraction": 0.1,
    }
    if gpu_id is not None:
        overrides.update(
            {
                "preprocessing.gpu": gpu_id,
                "preprocessing.gpu_memory_fraction": 0.1,
            }
        )
    config = manager.merge_config(overrides)
    build_compiled_process_topology(config)
    preprocessing = config.stages[0]
    arguments = resolve_stage_factory_args(preprocessing, config)
    assert preprocessing.process == "preprocessing"
    assert arguments["gpu_id"] == gpu_id


@pytest.mark.parametrize("gpu_id", [None, 0])
@pytest.mark.parametrize(
    "process_overrides",
    [
        {"preprocessing.process": "pipeline"},
        {"vocoder.process": "preprocessing"},
        {"preprocessing.process": "reference", "vocoder.process": "reference"},
    ],
)
def test_cuda_preprocessing_requires_dedicated_process(
    gpu_id: int | None, process_overrides: dict[str, str]
) -> None:
    """CUDA reference precision settings cannot leak into another stage's process."""
    manager = ConfigManager(S2ProPipelineConfig(model_path="model"))
    overrides: dict[str, str | int | None] = {
        **process_overrides,
        "preprocessing.gpu": gpu_id,
    }
    if gpu_id is None:
        config = manager.merge_config(overrides)
        assert config.stages[0].gpu is None
    else:
        with pytest.raises(ValueError, match="preprocessing in a dedicated process"):
            manager.merge_config(overrides)
