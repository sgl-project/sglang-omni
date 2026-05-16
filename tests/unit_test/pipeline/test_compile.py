# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

from sglang_omni.config.compiler import compile_pipeline, prepare_pipeline_runtime
from sglang_omni.config.schema import EndpointsConfig, PipelineConfig
from sglang_omni.pipeline.mp_runner import _build_stage_groups
from sglang_omni.pipeline.stage.input import AggregatedInput
from sglang_omni.pipeline.stage.stream_queue import StreamQueue
from sglang_omni.pipeline.stage_process import get_stage_process_env
from tests.unit_test.fixtures.pipeline_fakes import (
    FakeMpContext,
    FakeRelay,
    fake_factory_path,
)
from tests.unit_test.pipeline.helpers import stage


def test_pipeline_schema_keeps_topology_and_validation_contracts() -> None:
    """Preserves topology helpers and rejects invalid stage graphs early."""
    config = PipelineConfig(
        model_path="model",
        stages=[
            stage("preprocess", next="thinker"),
            stage("thinker", next="decode", gpu=[0, 1], tp_size=2),
            stage("decode", terminal=True),
        ],
    )

    assert config.resolved_entry_stage == "preprocess"
    assert config.terminal_stages == ["decode"]
    assert config.gpu_placement == {"thinker": [0, 1]}

    with pytest.raises(ValueError, match="unknown stages"):
        PipelineConfig(model_path="model", stages=[stage("a", next="missing")])
    with pytest.raises(ValueError, match="wait_for but no merge_fn"):
        PipelineConfig(
            model_path="model",
            stages=[
                stage("a", wait_for=["b"], terminal=True),
                stage("b", terminal=True),
            ],
        )
    with pytest.raises(ValueError, match="gpu has 1 entries"):
        PipelineConfig(
            model_path="model",
            stages=[stage("tp", gpu=[0], tp_size=2, terminal=True)],
        )


def test_compile_pipeline_wires_routes_overrides_aggregation_and_streams(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Preserves config-to-runtime wiring for routes, overrides, fan-in, and streams."""
    import sglang_omni.pipeline.stage.runtime as runtime

    monkeypatch.setattr(
        runtime,
        "create_relay",
        lambda relay_type, **kwargs: FakeRelay(device=kwargs.get("device", "cpu")),
    )
    config = PipelineConfig(
        model_path="global-model",
        name="contract",
        endpoints=EndpointsConfig(scheme="tcp"),
        runtime_overrides={"thinker": {"model_path": "runtime-model", "extra": "rt"}},
        stages=[
            stage("preprocess", next=["thinker", "aggregate"]),
            stage(
                "thinker",
                factory=fake_factory_path("make_scheduler_accepting_model_path"),
                factory_args={"extra": "factory"},
                gpu=0,
                next="aggregate",
                stream_to=["talker"],
            ),
            stage(
                "aggregate",
                wait_for=["preprocess", "thinker"],
                merge_fn=fake_factory_path("merge_payloads"),
                terminal=True,
            ),
            stage("talker", gpu=0, terminal=True),
        ],
    )

    coordinator, stages = compile_pipeline(config)
    stage_map = {compiled.name: compiled for compiled in stages}

    assert coordinator.entry_stage == "preprocess"
    assert stage_map["preprocess"].get_next("req", None) == ["thinker", "aggregate"]
    assert isinstance(stage_map["aggregate"].input_handler, AggregatedInput)
    assert isinstance(stage_map["talker"]._stream_queue, StreamQueue)
    assert stage_map["thinker"]._same_gpu_targets == {"talker"}
    assert stage_map["thinker"].scheduler.model_path == "runtime-model"
    assert stage_map["thinker"].scheduler.factory_kwargs["extra"] == "rt"


def test_mp_runner_preserves_tp_rank_and_visible_device_contracts() -> None:
    """Preserves TP process specs and one-visible-device env mapping."""
    config = PipelineConfig(
        model_path="model",
        name="mp",
        endpoints=EndpointsConfig(scheme="tcp"),
        relay_backend="nccl",
        stages=[
            stage(
                "thinker",
                factory=fake_factory_path("make_scheduler_accepting_gpu_id"),
                gpu=[1, 3],
                tp_size=2,
                terminal=True,
            )
        ],
    )
    prep = prepare_pipeline_runtime(config)

    group = _build_stage_groups(
        config,
        ctx=FakeMpContext(),
        stages_cfg=prep.stages_cfg,
        name_map=prep.name_map,
        endpoints=prep.endpoints,
    )[0]
    leader, follower = group.specs
    env = get_stage_process_env(follower, env={"CUDA_VISIBLE_DEVICES": "4,5,6,7"})

    assert leader.role == "leader"
    assert follower.role == "follower"
    assert leader.factory_args["tp_rank"] == 0
    assert follower.factory_args["tp_rank"] == 1
    assert leader.factory_args["nccl_port"] == follower.factory_args["nccl_port"]
    assert env["CUDA_VISIBLE_DEVICES"] == "7"
