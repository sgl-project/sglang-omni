# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

import pytest

from sglang_omni.config.runtime import resolve_factory_signature_args
from sglang_omni.config.schema import (
    EndpointsConfig,
    PDConfig,
    PDStagePlacement,
    PipelineConfig,
)
from sglang_omni.pipeline.mp_runner import (
    _build_stage_groups,
    _resolve_same_process_targets,
)
from sglang_omni.pipeline.runtime_config import prepare_pipeline_runtime
from sglang_omni.platforms.cuda import CUDAOmniPlatform
from sglang_omni.utils.imports import import_string
from tests.unit_test.fixtures.pipeline_fakes import FakeMpContext, fake_factory_path
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
    with pytest.raises(ValueError, match="route_fn on a terminal stage"):
        PipelineConfig(
            model_path="model",
            stages=[
                stage(
                    "decode",
                    terminal=True,
                    route_fn=fake_factory_path("identity_route"),
                )
            ],
        )
    with pytest.raises(ValueError, match="stream_done_to_fn without stream_to"):
        PipelineConfig(
            model_path="model",
            stages=[
                stage(
                    "thinker",
                    next="decode",
                    stream_done_to_fn=fake_factory_path("identity_stream_targets"),
                ),
                stage("decode", terminal=True),
            ],
        )
    with pytest.raises(ValueError, match="wait_for_fn but no wait_for"):
        PipelineConfig(
            model_path="model",
            stages=[
                stage(
                    "aggregate",
                    terminal=True,
                    wait_for_fn=fake_factory_path("identity_wait_sources"),
                )
            ],
        )


class _KwargSeedingPipelineConfig(PipelineConfig):
    """Seeds an author constructor kwarg, so both channels appear in specs."""

    def stage_factory_kwargs(self, stage_name: str) -> dict[str, Any]:
        if stage_name == "thinker":
            return {"extra": "factory"}
        return {}


def test_runner_specs_wire_routes_overrides_aggregation_and_streams(tmp_path) -> None:
    """Preserves config-to-runtime wiring for routes, overrides, fan-in, and streams."""
    config = _KwargSeedingPipelineConfig(
        model_path="global-model",
        name="contract",
        endpoints=EndpointsConfig(base_path=str(tmp_path)),
        stages=[
            stage("preprocess", next=["thinker", "aggregate"]),
            stage(
                "thinker",
                factory_path=fake_factory_path("make_scheduler_accepting_model_path"),
                factory={"extra": "rt"},
                gpu=0,
                next="aggregate",
                route_fn=fake_factory_path("identity_route"),
                stream_to=["talker"],
                stream_done_to_fn=fake_factory_path("identity_stream_targets"),
            ),
            stage(
                "aggregate",
                wait_for=["preprocess", "thinker"],
                wait_for_fn=fake_factory_path("identity_wait_sources"),
                merge_fn=fake_factory_path("merge_payloads"),
                terminal=True,
            ),
            stage("talker", gpu=0, terminal=True),
        ],
    )

    prep = prepare_pipeline_runtime(config)
    try:
        group = _build_stage_groups(
            config,
            ctx=FakeMpContext(),
            stages_cfg=prep.stages_cfg,
            endpoints=prep.endpoints,
            placement_plan=prep.placement_plan,
            process_plan=prep.process_plan,
        )[0]
    finally:
        assert prep.runtime_dir is not None
        prep.runtime_dir.close()
    specs = {spec.stage_name: spec for spec in group.specs}

    assert prep.entry_stage == "preprocess"
    assert specs["preprocess"].next_stages == ["thinker", "aggregate"]
    assert specs["thinker"].route_fn == fake_factory_path("identity_route")
    assert specs["thinker"].stream_done_to_fn == fake_factory_path(
        "identity_stream_targets"
    )
    assert specs["aggregate"].wait_for == ["preprocess", "thinker"]
    assert specs["aggregate"].wait_for_fn == fake_factory_path("identity_wait_sources")
    assert specs["aggregate"].merge_fn == fake_factory_path("merge_payloads")
    assert specs["talker"].is_stream_receiver
    assert specs["thinker"].gpu_stage_names == {"thinker", "talker"}
    assert specs["thinker"].stage_gpu_ids["thinker"] == (0,)
    assert specs["thinker"].stage_gpu_ids["talker"] == (0,)
    assert specs["preprocess"].same_process_targets == {"thinker", "aggregate"}
    assert specs["thinker"].same_process_targets == {"aggregate", "talker"}
    assert specs["thinker"].factory_arg_defaults["model_path"] == "global-model"
    assert specs["thinker"].factory_kwargs["extra"] == "factory"
    assert specs["thinker"].typed_kwargs["extra"] == "rt"


def test_runner_specs_defer_factory_signature_import_to_child(
    tmp_path,
    monkeypatch,
) -> None:
    import sglang_omni.config.runtime as runtime_config

    def fail_parent_factory_import(path: str):
        raise AssertionError(f"factory imported in parent process: {path}")

    monkeypatch.setattr(runtime_config, "import_string", fail_parent_factory_import)

    config = PipelineConfig(
        model_path="global-model",
        name="contract",
        endpoints=EndpointsConfig(base_path=str(tmp_path)),
        stages=[
            stage(
                "thinker",
                factory_path=fake_factory_path("runtime_factory"),
                gpu=1,
                terminal=True,
            ),
        ],
    )
    prep = prepare_pipeline_runtime(config)
    try:
        group = _build_stage_groups(
            config,
            ctx=FakeMpContext(),
            stages_cfg=prep.stages_cfg,
            endpoints=prep.endpoints,
            placement_plan=prep.placement_plan,
            process_plan=prep.process_plan,
        )[0]
    finally:
        assert prep.runtime_dir is not None
        prep.runtime_dir.close()

    spec = group.specs[0]
    assert spec.factory == fake_factory_path("runtime_factory")
    assert spec.factory_arg_defaults["model_path"] == "global-model"
    assert spec.factory_arg_defaults["gpu_id"] == 1
    assert spec.gpu_id == 1
    assert "model_path" not in spec.factory_kwargs
    assert "gpu_id" not in spec.factory_kwargs


def test_runner_specs_wire_same_process_targets_only_for_local_edges() -> None:
    config = PipelineConfig(
        model_path="model",
        stages=[
            stage("a", next="b", process="p0"),
            stage("b", next="c", process="p0"),
            stage("c", terminal=True, process="p1"),
        ],
    )
    prep = prepare_pipeline_runtime(config)
    groups = _build_stage_groups(
        config,
        ctx=FakeMpContext(),
        stages_cfg=prep.stages_cfg,
        endpoints=prep.endpoints,
        placement_plan=prep.placement_plan,
        process_plan=prep.process_plan,
    )
    specs = {spec.stage_name: spec for group in groups for spec in group.specs}

    assert specs["a"].same_process_targets == {"b"}
    assert specs["b"].same_process_targets == set()


@pytest.mark.parametrize(
    ("vocoder_process", "expected_fractions"),
    [
        ("vocoder", [0.15, 0.82, 0.18]),
        ("pipeline", [0.15, 0.82, 1.0]),
    ],
    ids=["isolated-vocoder", "merged-vocoder"],
)
def test_runner_specs_expose_process_total_in_construction_order(
    vocoder_process: str,
    expected_fractions: list[float],
) -> None:
    config = PipelineConfig(
        model_path="model",
        stages=[
            stage(
                "preprocess",
                next="engine",
                process="pipeline",
                gpu=0,
                gpu_memory_fraction=0.15,
            ),
            stage(
                "engine",
                next="vocoder",
                process="pipeline",
                gpu=0,
                gpu_memory_fraction=0.67,
            ),
            stage(
                "vocoder",
                terminal=True,
                process=vocoder_process,
                gpu=0,
                gpu_memory_fraction=0.18,
            ),
        ],
    )
    prep = prepare_pipeline_runtime(config)
    try:
        groups = _build_stage_groups(
            config,
            ctx=FakeMpContext(),
            stages_cfg=prep.stages_cfg,
            endpoints=prep.endpoints,
            placement_plan=prep.placement_plan,
            process_plan=prep.process_plan,
        )
    finally:
        assert prep.runtime_dir is not None
        prep.runtime_dir.close()
    specs = {spec.stage_name: spec for group in groups for spec in group.specs}

    assert [
        specs[stage_name].factory_arg_defaults["process_total_gpu_memory_fraction"]
        for stage_name in ("preprocess", "engine", "vocoder")
    ] == pytest.approx(expected_fractions)


def test_same_process_stages_compile_to_local_edges() -> None:
    config = PipelineConfig(
        model_path="model",
        stages=[
            stage("preprocess", next="encoder", process="frontend"),
            stage("encoder", next="decode", gpu=0, process="frontend"),
            stage("decode", terminal=True, process="decode"),
        ],
    )
    prep = prepare_pipeline_runtime(config)

    assert [stage_cfg.name for stage_cfg in prep.stages_cfg] == [
        "preprocess",
        "encoder",
        "decode",
    ]
    assert prep.entry_stage == "preprocess"
    assert prep.process_plan.stage_to_process["preprocess"] == (
        prep.process_plan.stage_to_process["encoder"]
    )
    assert prep.process_plan.stage_to_process["decode"] != (
        prep.process_plan.stage_to_process["encoder"]
    )

    groups = _build_stage_groups(
        config,
        ctx=FakeMpContext(),
        stages_cfg=prep.stages_cfg,
        endpoints=prep.endpoints,
        placement_plan=prep.placement_plan,
        process_plan=prep.process_plan,
    )
    specs = {spec.stage_name: spec for group in groups for spec in group.specs}

    assert specs["preprocess"].same_process_targets == {"encoder"}
    assert specs["encoder"].same_process_targets == set()


def test_runner_specs_wire_same_process_stream_targets() -> None:
    config = PipelineConfig(
        model_path="model",
        stages=[
            stage("thinker", next="decode", stream_to=["decode"]),
            stage("decode", terminal=True, can_accept_stream_before_payload=True),
        ],
    )
    prep = prepare_pipeline_runtime(config)
    groups = _build_stage_groups(
        config,
        ctx=FakeMpContext(),
        stages_cfg=prep.stages_cfg,
        endpoints=prep.endpoints,
        placement_plan=prep.placement_plan,
        process_plan=prep.process_plan,
    )
    specs = {spec.stage_name: spec for group in groups for spec in group.specs}

    assert specs["thinker"].same_process_targets == {"decode"}


def test_runner_specs_wire_direct_cuda_ipc_payload_disable_flag() -> None:
    config = PipelineConfig(
        model_path="model",
        stages=[
            stage(
                "mm_aggregate",
                next="thinker",
                disable_direct_cuda_ipc_payload=True,
            ),
            stage("thinker", terminal=True, gpu=0),
        ],
    )
    prep = prepare_pipeline_runtime(config)
    groups = _build_stage_groups(
        config,
        ctx=FakeMpContext(),
        stages_cfg=prep.stages_cfg,
        endpoints=prep.endpoints,
        placement_plan=prep.placement_plan,
        process_plan=prep.process_plan,
    )
    specs = {spec.stage_name: spec for group in groups for spec in group.specs}

    assert specs["mm_aggregate"].disable_direct_cuda_ipc_payload is True
    assert specs["thinker"].disable_direct_cuda_ipc_payload is False


def test_runner_specs_do_not_wire_same_process_targets_to_tp_stages() -> None:
    config = PipelineConfig(
        model_path="model",
        stages=[
            stage("preprocess", next="thinker"),
            stage("thinker", gpu=[0, 1], tp_size=2, terminal=True),
        ],
    )
    prep = prepare_pipeline_runtime(config)
    stage_cfg_by_name = {stage_cfg.name: stage_cfg for stage_cfg in prep.stages_cfg}
    preprocess = stage_cfg_by_name["preprocess"]
    thinker = stage_cfg_by_name["thinker"]

    assert (
        _resolve_same_process_targets(
            preprocess,
            stage_cfg_by_name,
            prep.process_plan,
        )
        == set()
    )
    assert (
        _resolve_same_process_targets(
            thinker,
            stage_cfg_by_name,
            prep.process_plan,
        )
        == set()
    )


def test_mp_runner_preserves_tp_rank_and_visible_device_contracts(tmp_path) -> None:
    """Preserves TP process specs and one-visible-device env mapping."""
    config = PipelineConfig(
        model_path="model",
        name="mp",
        endpoints=EndpointsConfig(base_path=str(tmp_path)),
        env_defaults={"SGLANG_TEST_STAGE_ENV": "1"},
        stages=[
            stage(
                "thinker",
                factory_path=fake_factory_path("make_scheduler_accepting_gpu_id"),
                gpu=[1, 3],
                tp_size=2,
                terminal=True,
            )
        ],
    )
    prep = prepare_pipeline_runtime(config)
    try:
        group = _build_stage_groups(
            config,
            ctx=FakeMpContext(),
            stages_cfg=prep.stages_cfg,
            endpoints=prep.endpoints,
            placement_plan=prep.placement_plan,
            process_plan=prep.process_plan,
        )[0]
    finally:
        assert prep.runtime_dir is not None
        prep.runtime_dir.close()
    leader, follower = group.specs
    env = CUDAOmniPlatform().get_stage_process_env(
        follower, env={"CUDA_VISIBLE_DEVICES": "4,5,6,7"}
    )

    assert leader.role == "leader"
    assert follower.role == "follower"
    assert leader.factory_kwargs["tp_rank"] == 0
    assert follower.factory_kwargs["tp_rank"] == 1
    assert leader.factory_kwargs["nccl_port"] == follower.factory_kwargs["nccl_port"]
    assert leader.env_defaults == {"SGLANG_TEST_STAGE_ENV": "1"}
    assert follower.env_defaults == {"SGLANG_TEST_STAGE_ENV": "1"}
    assert env["CUDA_VISIBLE_DEVICES"] == "7"


def test_mp_runner_keeps_cpu_stage_without_gpu_identity(tmp_path) -> None:
    config = PipelineConfig(
        model_path="model",
        name="mp",
        endpoints=EndpointsConfig(base_path=str(tmp_path)),
        stages=[stage("preprocess", next="decode"), stage("decode", terminal=True)],
    )
    prep = prepare_pipeline_runtime(config)
    try:
        group = _build_stage_groups(
            config,
            ctx=FakeMpContext(),
            stages_cfg=prep.stages_cfg,
            endpoints=prep.endpoints,
            placement_plan=prep.placement_plan,
            process_plan=prep.process_plan,
        )[0]
    finally:
        assert prep.runtime_dir is not None
        prep.runtime_dir.close()

    assert group.specs[0].gpu_id is None
    assert "gpu_id" not in group.specs[0].comm_config


def _pd(prefill_gpu: int, decode_gpu: int) -> PDConfig:
    return PDConfig(
        prefill=PDStagePlacement(gpu=prefill_gpu),
        decode=PDStagePlacement(gpu=decode_gpu),
    )


def _pd_pipeline(
    tmp_path,
    *,
    factory_path: str = fake_factory_path("pd_capable_factory"),
    terminal: bool = False,
) -> PipelineConfig:
    stages = [
        stage("pre", next="thinker"),
        stage(
            "thinker",
            factory_path=factory_path,
            terminal=terminal,
            next=None if terminal else "post",
            pd_disaggregation=_pd(1, 2),
        ),
    ]
    if not terminal:
        stages.append(stage("post", terminal=True))
    return PipelineConfig(
        model_path="dummy",
        name="pd",
        endpoints=EndpointsConfig(base_path=str(tmp_path)),
        entry_stage="thinker",
        stages=stages,
    )


def _pd_specs_by_name(config, prep):
    groups = _build_stage_groups(
        config,
        ctx=FakeMpContext(),
        stages_cfg=prep.stages_cfg,
        name_map=prep.name_map,
        endpoints=prep.endpoints,
        placement_plan=prep.placement_plan,
        process_plan=prep.process_plan,
    )
    return {spec.stage_name: spec for group in groups for spec in group.specs}


def test_pd_runtime_prep_rewrites_entry_and_terminal_identity(tmp_path) -> None:
    config = _pd_pipeline(tmp_path, terminal=True)
    prep = prepare_pipeline_runtime(config)
    with prep.runtime_dir:
        assert config.terminal_stages == ["thinker"]
        assert prep.entry_stage == "thinker_prefill"
        assert prep.terminal_stages == ["thinker_decode"]
        assert prep.terminal_name_map == {"thinker": "thinker_decode"}


def test_pd_stage_specs_carry_typed_pd_execution_outside_factory_args(
    tmp_path,
) -> None:
    config = _pd_pipeline(tmp_path)
    prep = prepare_pipeline_runtime(config)
    with prep.runtime_dir:
        specs = _pd_specs_by_name(config, prep)

    prefill, decode = specs["thinker_prefill"], specs["thinker_decode"]
    assert (prefill.pd_execution.role, prefill.pd_execution.partner) == (
        "prefill",
        "thinker_decode",
    )
    assert (decode.pd_execution.role, decode.pd_execution.partner) == (
        "decode",
        "thinker_prefill",
    )
    assert prefill.pd_execution.scheduler_class.endswith(".OmniPrefillScheduler")
    assert decode.pd_execution.scheduler_class.endswith(".OmniDecodeScheduler")
    assert prefill.pd_execution.scheduler_class != decode.pd_execution.scheduler_class
    assert "pd_role" not in prefill.factory_kwargs
    assert "pd_partner" not in decode.factory_kwargs

    assert prefill.name_map["thinker"] == "thinker_prefill"
    assert prefill.name_map["post"] == "post"
    assert prefill.name_map["thinker_prefill"] == "thinker_prefill"


def test_pd_decode_launches_and_shares_abort_endpoint_without_inbound_edge(
    tmp_path,
) -> None:
    config = _pd_pipeline(tmp_path)
    prep = prepare_pipeline_runtime(config)
    with prep.runtime_dir:
        groups = _build_stage_groups(
            config,
            ctx=FakeMpContext(),
            stages_cfg=prep.stages_cfg,
            name_map=prep.name_map,
            endpoints=prep.endpoints,
            placement_plan=prep.placement_plan,
            process_plan=prep.process_plan,
        )
    registered = {
        stage_name for group in groups for stage_name in group.stage_control_endpoints
    }
    assert {"thinker_prefill", "thinker_decode"} <= registered
    assert all(s.next != "thinker_decode" for s in prep.stages_cfg)

    specs = _pd_specs_by_name(config, prep)
    assert specs["thinker_prefill"].abort_endpoint == prep.endpoints["abort"]
    assert specs["thinker_decode"].abort_endpoint == prep.endpoints["abort"]


def test_pd_non_capable_factory_is_rejected_before_launch(tmp_path) -> None:
    config = _pd_pipeline(tmp_path, factory_path=fake_factory_path("dummy_factory"))
    with pytest.raises(ValueError, match="not PD-capable"):
        prepare_pipeline_runtime(config)


def test_pd_strict_factory_receives_no_pd_metadata(tmp_path) -> None:
    config = _pd_pipeline(
        tmp_path, factory_path=fake_factory_path("strict_pd_capable_factory")
    )
    prep = prepare_pipeline_runtime(config)
    with prep.runtime_dir:
        specs = _pd_specs_by_name(config, prep)

    decode = specs["thinker_decode"]
    factory = import_string(decode.factory)
    resolved = resolve_factory_signature_args(
        factory,
        decode.factory_kwargs,
        defaults=decode.factory_arg_defaults,
    )
    assert "pd_role" not in resolved
    assert "pd_partner" not in resolved
    assert factory(**resolved) == {"model_path": "dummy", "gpu_id": resolved["gpu_id"]}


def test_non_pd_pipeline_is_unaffected_by_pd_capability_check(tmp_path) -> None:
    config = PipelineConfig(
        model_path="dummy",
        name="plain",
        endpoints=EndpointsConfig(base_path=str(tmp_path)),
        stages=[stage("a", next="b"), stage("b", terminal=True)],
    )
    prep = prepare_pipeline_runtime(config)
    with prep.runtime_dir:
        specs = _pd_specs_by_name(config, prep)
    assert all(spec.pd_execution is None for spec in specs.values())
