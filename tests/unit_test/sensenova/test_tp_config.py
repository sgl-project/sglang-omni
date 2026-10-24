# SPDX-License-Identifier: Apache-2.0
from pathlib import Path

import pytest

from sglang_omni.config import ProcessConfig
from sglang_omni.config.manager import ConfigManager
from sglang_omni.config.runtime import (
    resolve_stage_factory_kwargs,
    resolve_stage_typed_kwargs,
)
from sglang_omni.config.topology import compile_logical_processes
from sglang_omni.models.sensenova_u1.config import (
    SenseNovaGenerationFactoryArgs,
    SenseNovaU1PipelineConfig,
)
from sglang_omni.pipeline.replicas import expand_replica_stages


def make_config(size=2, batch=1):
    data = SenseNovaU1PipelineConfig(model_path="/model/sensenova").model_dump()
    generate = next(stage for stage in data["stages"] if stage["name"] == "generate")
    generate["tp_size"] = size
    generate["gpu"] = list(range(size))
    generate["factory"]["max_batch_size"] = batch
    return data


@pytest.mark.parametrize("size", [1, 2, 4, 8])
def test_supported_tp_sizes(size):
    config = SenseNovaU1PipelineConfig.model_validate(make_config(size))
    assert config.stage_named("generate").tp_size == size


def test_tp_rejects_unsupported_size_and_dynamic_batching():
    with pytest.raises(ValueError, match="tp_size=1, 2, 4, or 8"):
        SenseNovaU1PipelineConfig.model_validate(make_config(3))
    with pytest.raises(ValueError, match="max_batch_size=1"):
        SenseNovaU1PipelineConfig.model_validate(make_config(batch=2))
    assert SenseNovaU1PipelineConfig.model_validate(make_config(1, 2))


def test_dp_tp_expands_into_independent_gpu_groups():
    data = make_config()
    data["processes"] = {
        "sensenova_generate": ProcessConfig(
            num_replicas=2, replica_devices=[0, 1, 2, 3]
        )
    }
    config = SenseNovaU1PipelineConfig.model_validate(data)
    plan, stages = compile_logical_processes(config)
    expanded, _ = expand_replica_stages(stages, plan)
    assert [
        (stage.name, stage.tp_size, stage.gpu)
        for stage in expanded
        if stage.name.startswith("generate")
    ] == [
        ("generate@r0", 2, [0, 1]),
        ("generate@r1", 2, [2, 3]),
    ]


def test_dp_tp_rejects_overlapping_replica_devices():
    data = make_config()
    data["processes"] = {
        "sensenova_generate": ProcessConfig(
            num_replicas=2, replica_devices=[0, 1, 1, 2]
        )
    }
    with pytest.raises(ValueError, match="distinct GPUs"):
        SenseNovaU1PipelineConfig.model_validate(data)


@pytest.mark.parametrize("name", ["sensenova_u1_tp2.yaml", "sensenova_u1_dp2_tp2.yaml"])
def test_tp_example_make_config(name):
    path = Path(__file__).resolve().parents[3] / "examples" / "configs" / name
    config = ConfigManager.from_file(str(path)).config
    plan, stages = compile_logical_processes(config)
    expanded, _ = expand_replica_stages(stages, plan)
    assert all(
        stage.tp_size == 2 for stage in expanded if stage.name.startswith("generate")
    )


def test_dp2_example_makes_independent_replicas():
    path = (
        Path(__file__).resolve().parents[3]
        / "examples"
        / "configs"
        / "sensenova_u1_dp2.yaml"
    )
    config = ConfigManager.from_file(str(path)).config
    plan, stages = compile_logical_processes(config)
    expanded, _ = expand_replica_stages(stages, plan)
    assert [
        (stage.name, stage.tp_size, stage.gpu)
        for stage in expanded
        if stage.name.startswith("generate")
    ] == [
        ("generate@r0", 1, 0),
        ("generate@r1", 1, 1),
    ]


def test_dotted_cli_enables_tp_and_rejects_batching():
    manager = ConfigManager(SenseNovaU1PipelineConfig(model_path="/model/sensenova"))
    config = manager.merge_config(
        [("generate.tp_size", "2"), ("generate.gpu", "[0, 1]")]
    )
    assert config.stage_named("generate").tp_size == 2
    with pytest.raises(ValueError, match="max_batch_size=1"):
        ConfigManager(config).merge_config([("generate.factory.max_batch_size", "2")])


@pytest.mark.parametrize(
    "dtype,offload,expected_dtype,expected_offload",
    [(None, None, "bfloat16", False), ("float16", True, "float16", True)],
)
def test_generation_defaults_reach_factory(
    dtype: str | None,
    offload: bool | None,
    expected_dtype: str,
    expected_offload: bool,
) -> None:
    config = SenseNovaU1PipelineConfig(model_path="/model/sensenova")
    stage = config.stage_named("generate")
    default_kwargs = {
        **resolve_stage_factory_kwargs(stage, config),
        **resolve_stage_typed_kwargs(stage),
    }
    assert default_kwargs["dtype"] == "bfloat16"
    assert default_kwargs["dit_layerwise_offload"] is False
    stage.factory = SenseNovaGenerationFactoryArgs(
        dtype=dtype, dit_layerwise_offload=offload
    )
    factory_kwargs = {
        **resolve_stage_factory_kwargs(stage, config),
        **resolve_stage_typed_kwargs(stage),
    }
    assert factory_kwargs["dtype"] == expected_dtype
    assert factory_kwargs["dit_layerwise_offload"] is expected_offload


@pytest.mark.parametrize("use_none", [False, True])
def test_offload_defaults_reach_factory(use_none: bool) -> None:
    expected = {
        "dit_offload_prefetch_size": 1,
        "dit_layerwise_resident_layers": 0,
        "dit_layerwise_residency_policy": "leading",
        "pin_cpu_memory": False,
    }
    config = SenseNovaU1PipelineConfig(model_path="/model/sensenova")
    stage = config.stage_named("generate")
    if use_none:
        stage.factory = SenseNovaGenerationFactoryArgs.model_validate(
            dict.fromkeys(expected)
        )
    else:
        stage.factory = SenseNovaGenerationFactoryArgs()
    resolved = {
        **resolve_stage_factory_kwargs(stage, config),
        **resolve_stage_typed_kwargs(stage),
    }
    assert {name: resolved[name] for name in expected} == expected


def test_offload_overrides_reach_factory() -> None:
    overrides = {
        "dit_offload_prefetch_size": 2,
        "dit_layerwise_resident_layers": 3,
        "dit_layerwise_residency_policy": "strided",
        "pin_cpu_memory": True,
    }
    config = SenseNovaU1PipelineConfig(model_path="/model/sensenova")
    stage = config.stage_named("generate")
    stage.factory = SenseNovaGenerationFactoryArgs.model_validate(overrides)
    resolved = {
        **resolve_stage_factory_kwargs(stage, config),
        **resolve_stage_typed_kwargs(stage),
    }
    assert {name: resolved[name] for name in overrides} == overrides
