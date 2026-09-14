# SPDX-License-Identifier: Apache-2.0
"""Leader/follower assignment for runtime-native CUDA IPC weight sharing."""

from __future__ import annotations

import os
from typing import ClassVar

import pytest

from sglang_omni.config import (
    EndpointsConfig,
    EngineArgs,
    PipelineConfig,
    ProcessConfig,
    StageConfig,
)
from sglang_omni.config.schema import EngineStageConfig
from sglang_omni.config.topology import compile_logical_processes
from sglang_omni.pipeline.stage_workers import StageLaunchConfig, StageWorkerProcessSpec
from sglang_omni.pipeline.weight_share import WeightShareError, plan_weight_share
from sglang_omni.utils.ipc_weights import (
    ENV_WEIGHT_SHARE,
    ENV_WEIGHT_SHARE_COMPAT,
    ENV_WEIGHT_SHARE_RUN_ID,
)

pytestmark = pytest.mark.skipif(
    os.name != "posix",
    reason="weight sharing plans only on POSIX hosts",
)


def noop_factory():  # pragma: no cover - never constructed here
    raise AssertionError("factory must not run")


class _SharingPipelineConfig(PipelineConfig):
    stage_config_types: ClassVar[dict[str, type[StageConfig]]] = {
        "engine": EngineStageConfig,
        "second_engine": EngineStageConfig,
    }


def _engine_stage(
    name: str = "engine",
    *,
    process: str = "gen",
    max_total_tokens: int | None = 30000,
    gpu: int | None = 0,
    tp_size: int = 1,
) -> EngineStageConfig:
    engine = EngineArgs()
    if max_total_tokens is not None:
        engine.max_total_tokens = max_total_tokens
    return EngineStageConfig(
        name=name,
        process=process,
        factory_path=f"{__name__}.noop_factory",
        gpu=gpu,
        gpu_memory_fraction=0.45 if gpu is not None else None,
        tp_size=tp_size,
        terminal=True,
        engine=engine,
    )


def _make_config(
    tmp_path,
    *,
    stages,
    processes,
    weight_share: str = "on",
) -> PipelineConfig:
    return _SharingPipelineConfig(
        model_path="model",
        entry_stage=stages[0].name,
        stages=list(stages),
        processes=processes,
        weight_share=weight_share,
        endpoints=EndpointsConfig(base_path=str(tmp_path)),
    )


def _spec(
    process_name: str,
    *,
    stage_name: str,
    gpu_id: int | None,
    tp_size: int = 1,
    env_defaults: dict[str, str] | None = None,
) -> StageWorkerProcessSpec:
    return StageWorkerProcessSpec(
        process_name=process_name,
        stage_specs=[
            StageLaunchConfig(
                stage_name=stage_name,
                factory=f"{__name__}.noop_factory",
                placement_gpu_id=gpu_id,
                gpu_id=gpu_id,
                tp_size=tp_size,
                env_defaults=dict(env_defaults or {}),
            )
        ],
    )


def _plan(config: PipelineConfig, specs, tmp_path):
    logical_plan, _ = compile_logical_processes(config)
    return plan_weight_share(
        config,
        logical_process_plan=logical_plan,
        process_specs=specs,
        runtime_dir=tmp_path,
    )


def test_off_plans_nothing(tmp_path) -> None:
    config = _make_config(
        tmp_path,
        stages=[_engine_stage()],
        processes={"gen": ProcessConfig(num_replicas=2, replica_devices=[0, 0])},
        weight_share="off",
    )
    specs = [
        _spec("gen@r0", stage_name="engine@r0", gpu_id=0),
        _spec("gen@r1", stage_name="engine@r1", gpu_id=0),
    ]

    assert _plan(config, specs, tmp_path) is None


def test_lowest_replica_leads_and_the_rest_follow(tmp_path) -> None:
    config = _make_config(
        tmp_path,
        stages=[_engine_stage()],
        processes={"gen": ProcessConfig(num_replicas=3, replica_devices=[0, 0, 0])},
    )
    specs = [
        _spec(f"gen@r{index}", stage_name=f"engine@r{index}", gpu_id=0)
        for index in range(3)
    ]

    plan = _plan(config, specs, tmp_path)

    assert plan is not None
    (group,) = plan.groups
    assert group.leader == "gen@r0"
    assert group.followers == ("gen@r1", "gen@r2")
    assert plan.follower_process_names == {"gen@r1", "gen@r2"}
    assert plan.env_by_process["gen@r0"][ENV_WEIGHT_SHARE] == (
        f"leader:{group.store_dir}"
    )
    assert plan.env_by_process["gen@r1"][ENV_WEIGHT_SHARE] == (
        f"follower:{group.store_dir}"
    )
    run_ids = {env[ENV_WEIGHT_SHARE_RUN_ID] for env in plan.env_by_process.values()}
    assert run_ids == {plan.run_id}
    assert group.store_dir.is_dir()
    assert oct(group.store_dir.stat().st_mode)[-3:] == "700"


def test_a_replica_alone_on_its_gpu_keeps_loading_normally(tmp_path) -> None:
    """replica_devices=[0, 0, 1]: the GPU-1 singleton is not a follower."""
    config = _make_config(
        tmp_path,
        stages=[_engine_stage()],
        processes={"gen": ProcessConfig(num_replicas=3, replica_devices=[0, 0, 1])},
    )
    specs = [
        _spec("gen@r0", stage_name="engine@r0", gpu_id=0),
        _spec("gen@r1", stage_name="engine@r1", gpu_id=0),
        _spec("gen@r2", stage_name="engine@r2", gpu_id=1),
    ]

    plan = _plan(config, specs, tmp_path)

    assert plan is not None
    (group,) = plan.groups
    assert group.gpu_id == 0
    assert group.leader == "gen@r0"
    assert group.followers == ("gen@r1",)
    assert "gen@r2" not in plan.follower_process_names
    assert ENV_WEIGHT_SHARE not in plan.env_by_process["gen@r2"]


def test_two_gpus_each_get_their_own_leader(tmp_path) -> None:
    config = _make_config(
        tmp_path,
        stages=[_engine_stage()],
        processes={"gen": ProcessConfig(num_replicas=4, replica_devices=[0, 0, 1, 1])},
    )
    specs = [
        _spec("gen@r0", stage_name="engine@r0", gpu_id=0),
        _spec("gen@r1", stage_name="engine@r1", gpu_id=0),
        _spec("gen@r2", stage_name="engine@r2", gpu_id=1),
        _spec("gen@r3", stage_name="engine@r3", gpu_id=1),
    ]

    plan = _plan(config, specs, tmp_path)

    assert plan is not None
    assert [(g.gpu_id, g.leader, g.followers) for g in plan.groups] == [
        (0, "gen@r0", ("gen@r1",)),
        (1, "gen@r2", ("gen@r3",)),
    ]
    assert len({group.store_dir for group in plan.groups}) == 2


def test_every_process_gets_the_reduction_compat_flag(tmp_path) -> None:
    """A relay-only stage unpickles the engine's CUDA tensors and needs it too."""
    config = _make_config(
        tmp_path,
        stages=[
            _engine_stage(),
            StageConfig(
                name="vocoder",
                process="vocoder",
                factory_path=f"{__name__}.noop_factory",
                gpu=0,
                terminal=True,
            ),
        ],
        processes={"gen": ProcessConfig(num_replicas=2, replica_devices=[0, 0])},
    )
    specs = [
        _spec("gen@r0", stage_name="engine@r0", gpu_id=0),
        _spec("gen@r1", stage_name="engine@r1", gpu_id=0),
        _spec("vocoder", stage_name="vocoder", gpu_id=0),
    ]

    plan = _plan(config, specs, tmp_path)

    assert plan is not None
    assert all(
        env[ENV_WEIGHT_SHARE_COMPAT] == "1" for env in plan.env_by_process.values()
    )
    assert set(plan.env_by_process) == {"gen@r0", "gen@r1", "vocoder"}


def test_a_replicated_tp_process_is_skipped_not_rejected(tmp_path) -> None:
    """Sharing one process must not fail the pipeline over an unrelated TP one."""
    config = _make_config(
        tmp_path,
        stages=[
            _engine_stage(),
            StageConfig(
                name="thinker",
                process="thinker",
                factory_path=f"{__name__}.noop_factory",
                gpu=[1, 2],
                tp_size=2,
                terminal=True,
            ),
        ],
        processes={
            "gen": ProcessConfig(num_replicas=2, replica_devices=[0, 0]),
            "thinker": ProcessConfig(num_replicas=2, replica_devices=[1, 2, 1, 2]),
        },
    )
    specs = [
        _spec("gen@r0", stage_name="engine@r0", gpu_id=0),
        _spec("gen@r1", stage_name="engine@r1", gpu_id=0),
        _spec("thinker@r0_tp0", stage_name="thinker@r0", gpu_id=1, tp_size=2),
        _spec("thinker@r0_tp1", stage_name="thinker@r0", gpu_id=2, tp_size=2),
        _spec("thinker@r1_tp0", stage_name="thinker@r1", gpu_id=1, tp_size=2),
        _spec("thinker@r1_tp1", stage_name="thinker@r1", gpu_id=2, tp_size=2),
    ]

    plan = _plan(config, specs, tmp_path)

    assert plan is not None
    assert [group.logical_process for group in plan.groups] == ["gen"]
    assert plan.follower_process_names == {"gen@r1"}


def test_cpu_only_replicas_are_not_a_sharing_group(tmp_path) -> None:
    config = _make_config(
        tmp_path,
        stages=[
            _engine_stage(),
            StageConfig(
                name="preprocess",
                process="pre",
                factory_path=f"{__name__}.noop_factory",
                next="engine",
            ),
        ],
        processes={
            "gen": ProcessConfig(num_replicas=2, replica_devices=[0, 0]),
            "pre": ProcessConfig(num_replicas=2),
        },
    )
    specs = [
        _spec("gen@r0", stage_name="engine@r0", gpu_id=0),
        _spec("gen@r1", stage_name="engine@r1", gpu_id=0),
        _spec("pre@r0", stage_name="preprocess@r0", gpu_id=None),
        _spec("pre@r1", stage_name="preprocess@r1", gpu_id=None),
    ]

    plan = _plan(config, specs, tmp_path)

    assert plan is not None
    assert [group.logical_process for group in plan.groups] == ["gen"]
    assert plan.follower_process_names == {"gen@r1"}


def test_on_without_any_same_gpu_replica_pair_is_rejected(tmp_path) -> None:
    config = _make_config(
        tmp_path,
        stages=[_engine_stage()],
        processes={"gen": ProcessConfig(num_replicas=2, replica_devices=[0, 1])},
    )
    specs = [
        _spec("gen@r0", stage_name="engine@r0", gpu_id=0),
        _spec("gen@r1", stage_name="engine@r1", gpu_id=1),
    ]

    with pytest.raises(WeightShareError, match="two or more replicas"):
        _plan(config, specs, tmp_path)


def test_engine_stage_without_a_pinned_kv_cap_is_rejected(tmp_path) -> None:
    config = _make_config(
        tmp_path,
        stages=[_engine_stage(max_total_tokens=None)],
        processes={"gen": ProcessConfig(num_replicas=2, replica_devices=[0, 0])},
    )
    specs = [
        _spec("gen@r0", stage_name="engine@r0", gpu_id=0),
        _spec("gen@r1", stage_name="engine@r1", gpu_id=0),
    ]

    with pytest.raises(WeightShareError, match="max_total_tokens"):
        _plan(config, specs, tmp_path)


def test_two_engine_stages_in_one_sharing_process_are_rejected(tmp_path) -> None:
    config = _make_config(
        tmp_path,
        stages=[
            _engine_stage(),
            _engine_stage(name="second_engine", process="gen"),
        ],
        processes={"gen": ProcessConfig(num_replicas=2, replica_devices=[0, 0])},
    )
    specs = [
        _spec("gen@r0", stage_name="engine@r0", gpu_id=0),
        _spec("gen@r1", stage_name="engine@r1", gpu_id=0),
    ]

    with pytest.raises(WeightShareError, match="exactly one SGLang engine stage"):
        _plan(config, specs, tmp_path)


def test_externally_assigned_roles_are_rejected(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv(ENV_WEIGHT_SHARE, "leader:/tmp/elsewhere")
    config = _make_config(
        tmp_path,
        stages=[_engine_stage()],
        processes={"gen": ProcessConfig(num_replicas=2, replica_devices=[0, 0])},
    )
    specs = [
        _spec("gen@r0", stage_name="engine@r0", gpu_id=0),
        _spec("gen@r1", stage_name="engine@r1", gpu_id=0),
    ]

    with pytest.raises(WeightShareError, match="parent environment"):
        _plan(config, specs, tmp_path)


def test_a_stage_env_default_cannot_assign_a_role(tmp_path) -> None:
    config = _make_config(
        tmp_path,
        stages=[_engine_stage()],
        processes={"gen": ProcessConfig(num_replicas=2, replica_devices=[0, 0])},
    )
    specs = [
        _spec(
            "gen@r0",
            stage_name="engine@r0",
            gpu_id=0,
            env_defaults={ENV_WEIGHT_SHARE: "leader:/tmp/elsewhere"},
        ),
        _spec("gen@r1", stage_name="engine@r1", gpu_id=0),
    ]

    with pytest.raises(WeightShareError, match="environment defaults"):
        _plan(config, specs, tmp_path)


def test_the_weight_share_flag_resolves_from_the_command_line(tmp_path) -> None:
    from sglang_omni.config.manager import ConfigManager

    config = _make_config(
        tmp_path,
        stages=[_engine_stage()],
        processes={"gen": ProcessConfig(num_replicas=2, replica_devices=[0, 0])},
        weight_share="off",
    )
    manager = ConfigManager(config)
    merged = manager.merge_config(manager.parse_extra_args(["--weight-share", "on"]))

    assert merged.weight_share == "on"
