# SPDX-License-Identifier: Apache-2.0
import pytest

from sglang_omni.config import StageConfig
from sglang_omni.config.placement import build_stage_placement_plan
from sglang_omni.models.cosmos3.config import Cosmos3PipelineConfig
from sglang_omni.models.cosmos3.reasoner import native_reasoner_kwargs
from sglang_omni.models.cosmos3.stages import native_server_kwargs
from sglang_omni.pipeline import runtime_config
from sglang_omni.pipeline.mp_runner import _build_stage_groups
from sglang_omni.pipeline.replicas import validate_device_assignment
from sglang_omni.pipeline.stage_workers import _stage_gpu_ids


def config(devices=(1, 3)):
    cfg = Cosmos3PipelineConfig(model_path="checkpoint")
    cfg.stages[0].gpu = devices[0]
    cfg.stages[0].runtime_gpu_ids = list(devices)
    return cfg


def test_native_workers_reserve_all_gpus_but_use_one_omni_process(monkeypatch):
    monkeypatch.setattr(runtime_config, "_visible_device_count", lambda: 4)
    cfg = config()
    prep = runtime_config.prepare_pipeline_runtime(cfg)
    try:
        groups = _build_stage_groups(
            cfg,
            stages_cfg=prep.stages_cfg,
            endpoints=prep.endpoints,
            placement_plan=prep.placement_plan,
            process_plan=prep.process_plan,
        )
        assert set(prep.placement_plan.gpus) == {1, 3}
        assert len(groups) == 1
        assert len(groups[0].specs) == 1
        spec = groups[0].specs[0]
        assert spec.tp_size == 1
        assert spec.factory_kwargs["runtime_gpu_ids"] == [1, 3]
        assert _stage_gpu_ids(groups[0].specs) == [1, 3]
    finally:
        prep.runtime_dir.close()


def test_native_gpu_budget_includes_child_devices():
    cfg = config()
    cfg.stages[0].gpu_memory_fraction = 0.6
    cfg.stages.append(
        StageConfig(
            name="other",
            factory_path=cfg.stages[0].factory_path,
            gpu=3,
            gpu_memory_fraction=0.6,
            terminal=True,
        )
    )
    with pytest.raises(ValueError, match="exceeds placement limit"):
        build_stage_placement_plan(cfg)


def test_native_gpu_range_is_validated_before_spawn():
    with pytest.raises(ValueError, match="GPU id 3 out of range"):
        validate_device_assignment(config().stages, device_count=2)


def test_native_parallel_options_use_the_declared_gpu_group():
    gen = native_server_kwargs("checkpoint", 1, {"tp_size": 2}, [1, 3])
    assert gen["gpu_ids"] == [1, 3]
    assert gen["num_gpus"] == gen["tp_size"] == 2
    reasoner = native_reasoner_kwargs("checkpoint", 1, None, [1, 3])
    assert reasoner["base_gpu_id"] == 1
    assert reasoner["gpu_id_step"] == 2
    assert reasoner["tp_size"] == 2


@pytest.mark.parametrize("devices", [[], [1, 1], [2, 3], [1, -1]])
def test_invalid_native_gpu_lists_are_rejected(devices):
    cfg = config()
    cfg.stages[0].runtime_gpu_ids = devices
    with pytest.raises(ValueError):
        build_stage_placement_plan(cfg)


def test_native_runtime_cannot_share_a_process_with_another_stage(monkeypatch):
    monkeypatch.setattr(runtime_config, "_visible_device_count", lambda: 4)
    cfg = config()
    cfg.stages.append(
        StageConfig(
            name="other",
            process="generation",
            factory_path=cfg.stages[0].factory_path,
            terminal=True,
        )
    )
    with pytest.raises(ValueError, match="own its OS process"):
        runtime_config.prepare_pipeline_runtime(cfg)
