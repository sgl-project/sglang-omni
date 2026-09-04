# SPDX-License-Identifier: Apache-2.0
"""Fish Audio process-replica factory contracts."""

import pytest

from sglang_omni.config import ProcessConfig, resolve_stage_factory_args
from sglang_omni.config.topology import compile_logical_processes
from sglang_omni.models.fishaudio_s2_pro.config import S2ProPipelineConfig
from sglang_omni.pipeline.replicas import expand_replica_stages


def _expanded_replica_stages():
    config = S2ProPipelineConfig(
        model_path="model",
        processes={"pipeline": ProcessConfig(num_replicas=2, replica_devices=[1, 2])},
    )
    process_plan, stages = compile_logical_processes(config)
    expanded, _ = expand_replica_stages(stages, process_plan)
    return config, {stage.name: stage for stage in expanded}


def test_engine_factory_rejects_process_replica_device_injection() -> None:
    config, by_name = _expanded_replica_stages()

    with pytest.raises(
        ValueError,
        match="tts_engine@r0.*replica_devices.*does not declare a gpu_id parameter",
    ):
        resolve_stage_factory_args(by_name["tts_engine@r0"], config, gpu_id=1)


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
