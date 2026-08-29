# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging

import pytest
from sglang.srt import runtime_context

if not hasattr(runtime_context, "get_model"):
    runtime_context.get_model = lambda: None
if not hasattr(runtime_context, "get_serving"):
    runtime_context.get_serving = lambda: None

from sglang_omni.config import (
    PDConfig,
    PDExecution,
    PDStagePlacement,
    PipelineConfig,
    expand_pd_stages,
)
from sglang_omni.pipeline import stage_workers
from sglang_omni.pipeline.mp_runner import _build_stage_groups
from sglang_omni.pipeline.runtime_config import prepare_pipeline_runtime
from sglang_omni.pipeline.stage_workers import StageLaunchConfig
from sglang_omni.scheduling.omni_scheduler import OmniScheduler
from sglang_omni.scheduling.pd_scheduler import (
    OmniDecodeScheduler,
    OmniPrefillScheduler,
)
from tests.unit_test.fixtures.pipeline_fakes import fake_factory_path
from tests.unit_test.pipeline.helpers import stage


def _pd() -> PDConfig:
    return PDConfig(
        prefill=PDStagePlacement(gpu=0),
        decode=PDStagePlacement(gpu=1),
    )


def _config(*, factory: str = "make_pd_scheduler") -> PipelineConfig:
    return _PDTestConfig(
        model_path="model",
        entry_stage="pre",
        stages=[
            stage("pre", next="thinker"),
            stage(
                "thinker",
                factory_path=fake_factory_path(factory),
                wait_for=["pre"],
                merge_fn=fake_factory_path("merge_payloads"),
                route_fn=fake_factory_path("identity_route"),
                next="post",
                stream_to=["sink"],
                project_payload={"post": fake_factory_path("project_payload")},
                pd_disaggregation=_pd(),
            ),
            stage("post", terminal=True),
            stage("sink", terminal=True),
        ],
    )


class _PDTestConfig(PipelineConfig):
    def stage_factory_kwargs(self, stage_name: str):
        return {"logical_marker": stage_name} if stage_name == "thinker" else {}


def test_pd_rewrite_assigns_input_to_prefill_and_output_to_decode() -> None:
    config = _config()
    expansion = expand_pd_stages(
        list(config.stages), entry_stage=config.resolved_entry_stage
    )
    stages = {item.name: item for item in expansion.stages}

    assert [item.name for item in expansion.stages] == [
        "pre",
        "thinker_prefill",
        "thinker_decode",
        "post",
        "sink",
    ]
    assert stages["pre"].next == "thinker_prefill"
    assert stages["thinker_prefill"].wait_for == ["pre"]
    assert stages["thinker_prefill"].next is None
    assert stages["thinker_prefill"].stream_to == []
    assert stages["thinker_decode"].wait_for is None
    assert stages["thinker_decode"].next == "post"
    assert stages["thinker_decode"].stream_to == ["sink"]
    assert stages["thinker_decode"].route_fn == fake_factory_path("identity_route")
    assert stages["thinker_prefill"].pd_execution == PDExecution(
        decode_targets=("thinker_decode",), role="prefill", partner="thinker_decode"
    )
    assert stages["thinker_decode"].pd_execution == PDExecution(
        role="decode", partner="thinker_prefill"
    )
    assert "thinker_decode" not in (stages["thinker_prefill"].next or [])


def test_pd_rewrite_is_idempotent_and_non_pd_is_unchanged() -> None:
    config = _config()
    once = expand_pd_stages(
        list(config.stages), entry_stage=config.resolved_entry_stage
    )
    twice = expand_pd_stages(once.stages, entry_stage=once.entry_stage)
    assert twice.stages == once.stages
    assert twice.entry_stage == once.entry_stage
    assert twice.routing_map == once.routing_map
    assert twice.output_map == once.output_map

    plain = PipelineConfig(
        model_path="model",
        stages=[stage("one", next="two"), stage("two", terminal=True)],
    )
    unchanged = expand_pd_stages(
        list(plain.stages), entry_stage=plain.resolved_entry_stage
    )
    assert unchanged.stages == plain.stages
    assert unchanged.entry_stage == "one"


def test_logical_pd_entry_and_terminal_identity_belong_to_role_halves() -> None:
    config = PipelineConfig(
        model_path="model",
        entry_stage="thinker",
        stages=[
            stage(
                "thinker",
                factory_path=fake_factory_path("make_pd_scheduler"),
                terminal=True,
                pd_disaggregation=_pd(),
            )
        ],
    )
    expansion = expand_pd_stages(
        list(config.stages), entry_stage=config.resolved_entry_stage
    )

    assert expansion.entry_stage == "thinker_prefill"
    assert [item.name for item in expansion.stages if item.terminal] == [
        "thinker_decode"
    ]
    assert expansion.stages[0].next is None


def test_runtime_prep_maps_entry_terminal_and_output_sources(tmp_path) -> None:
    config = _config()
    config.endpoints.base_path = str(tmp_path)
    prep = prepare_pipeline_runtime(config)
    try:
        assert prep.entry_stage == "pre"
        assert prep.name_map["thinker"] == "thinker_prefill"
        assert prep.source_name_map["thinker"] == "thinker_decode"
        assert {item.name for item in prep.stages_cfg} == {
            "pre",
            "thinker_prefill",
            "thinker_decode",
            "post",
            "sink",
        }
        assert "thinker_decode" not in prep.terminal_stages
    finally:
        prep.runtime_dir.close()


def test_compiled_specs_construct_concrete_scheduler_types(tmp_path) -> None:
    config = _config()
    config.endpoints.base_path = str(tmp_path)
    prep = prepare_pipeline_runtime(config)
    try:
        groups = _build_stage_groups(
            config,
            stages_cfg=prep.stages_cfg,
            endpoints=prep.endpoints,
            placement_plan=prep.placement_plan,
            process_plan=prep.process_plan,
            name_map=prep.name_map,
            source_name_map=prep.source_name_map,
            replica_topology=prep.replica_topology,
        )
        specs = {
            spec.stage_name: spec
            for group in groups
            for process in group.process_specs
            for spec in process.stage_specs
        }
        prefill = stage_workers._construct_scheduler(
            specs["thinker_prefill"], None, logging.getLogger(__name__)
        )
        decode = stage_workers._construct_scheduler(
            specs["thinker_decode"], None, logging.getLogger(__name__)
        )
        plain = stage_workers._construct_scheduler(
            specs["pre"], None, logging.getLogger(__name__)
        )
        assert type(prefill) is OmniPrefillScheduler
        assert type(decode) is OmniDecodeScheduler
        assert prefill.factory_observed_type is OmniPrefillScheduler
        assert decode.factory_observed_type is OmniDecodeScheduler
        assert not isinstance(plain, OmniScheduler)
        assert prefill.factory_kwargs["partner_stage"] == "thinker_decode"
        assert decode.factory_kwargs["partner_stage"] == "thinker_prefill"
        assert prefill.factory_kwargs["logical_marker"] == "thinker"
        assert decode.factory_kwargs["logical_marker"] == "thinker"
    finally:
        prep.runtime_dir.close()


def test_stage_receives_scheduler_after_factory_selected_final_type(
    monkeypatch,
) -> None:
    captured = {}

    class _Stage:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(stage_workers, "Stage", _Stage)
    spec = StageLaunchConfig(
        stage_name="thinker_prefill",
        factory=fake_factory_path("make_pd_scheduler"),
        pd_execution=PDExecution(role="prefill", partner="thinker_decode"),
    )
    stage_workers._construct_stage(spec, logging.getLogger(__name__))

    assert type(captured["scheduler"]) is OmniPrefillScheduler
    assert captured["kv_registrations"] == ()


def test_concrete_scheduler_types_isolate_role_state() -> None:
    assert not hasattr(OmniScheduler, "_handoff_prefilled_requests")
    assert not hasattr(OmniScheduler, "_drain_decode_admissions")
    assert hasattr(OmniPrefillScheduler, "_handoff_prefilled_requests")
    assert not hasattr(OmniPrefillScheduler, "_drain_decode_admissions")
    assert hasattr(OmniDecodeScheduler, "_drain_decode_admissions")
    assert not hasattr(OmniDecodeScheduler, "_handoff_prefilled_requests")
    assert "_pd_role" not in OmniScheduler.__dict__


def test_pd_factory_wrong_type_and_missing_capability_fail_at_startup(tmp_path) -> None:
    wrong = StageLaunchConfig(
        stage_name="thinker_decode",
        factory=fake_factory_path("make_wrong_pd_scheduler"),
        pd_execution=PDExecution(role="decode", partner="thinker_prefill"),
    )
    with pytest.raises(TypeError, match="expected OmniDecodeScheduler"):
        stage_workers._construct_scheduler(wrong, None, logging.getLogger(__name__))

    # The declaration is config data, so a stage that never made it is
    # rejected before any process starts.
    config = _config(factory="make_scheduler")
    config.endpoints.base_path = str(tmp_path)
    for item in config.stages:
        item.pd_capable = False
    with pytest.raises(ValueError, match="has not declared pd_capable"):
        prepare_pipeline_runtime(config)


@pytest.mark.parametrize(
    "pd,message",
    [
        (PDConfig(prefill=PDStagePlacement(gpu=0)), "requires explicit"),
        (
            PDConfig(
                prefill=PDStagePlacement(gpu=0),
                decode=PDStagePlacement(gpu=0),
            ),
            "must declare memory_fraction",
        ),
        (
            PDConfig(
                prefill=PDStagePlacement(gpu=0, memory_fraction=0.45),
                decode=PDStagePlacement(gpu=0),
            ),
            "decode must declare memory_fraction",
        ),
    ],
)
def test_invalid_pd_placement_fails_in_config(pd, message) -> None:
    with pytest.raises(ValueError, match=message):
        PipelineConfig(
            model_path="model",
            stages=[stage("thinker", terminal=True, pd_disaggregation=pd)],
        )


def test_two_halves_may_share_a_gpu_when_both_declare_a_budget() -> None:
    """The split PD needs is between processes, not between cards.

    Sharing is also the only shape in which the halves can share one copy of
    the weights, and the only one a one-GPU CI box can run.
    """
    config = PipelineConfig(
        model_path="model",
        stages=[
            stage(
                "thinker",
                terminal=True,
                pd_disaggregation=PDConfig(
                    prefill=PDStagePlacement(gpu=0, memory_fraction=0.45),
                    decode=PDStagePlacement(gpu=0, memory_fraction=0.45),
                ),
            )
        ],
    )
    expansion = expand_pd_stages(
        list(config.stages), entry_stage=config.resolved_entry_stage
    )
    halves = {item.name: item for item in expansion.stages}

    assert halves["thinker_prefill"].gpu == 0
    assert halves["thinker_decode"].gpu == 0
    assert halves["thinker_prefill"].gpu_memory_fraction == 0.45
    assert halves["thinker_decode"].gpu_memory_fraction == 0.45


def test_each_half_may_override_the_engine_arguments() -> None:
    """Prefill sizes batches for one forward, Decode for many steps."""
    config = PipelineConfig(
        model_path="model",
        stages=[
            stage(
                "thinker",
                terminal=True,
                pd_disaggregation=PDConfig(
                    prefill=PDStagePlacement(gpu=0, engine={"max_running_requests": 8}),
                    decode=PDStagePlacement(gpu=1, engine={"max_running_requests": 64}),
                ),
            )
        ],
    )
    expansion = expand_pd_stages(
        list(config.stages), entry_stage=config.resolved_entry_stage
    )
    halves = {item.name: item for item in expansion.stages}

    assert halves["thinker_prefill"].engine.max_running_requests == 8
    assert halves["thinker_decode"].engine.max_running_requests == 64


def test_an_unsplit_budget_still_reaches_both_halves() -> None:
    """Separate cards need no per-half budget; the logical one stands."""
    logical = stage(
        "thinker",
        terminal=True,
        pd_disaggregation=PDConfig(
            prefill=PDStagePlacement(gpu=0),
            decode=PDStagePlacement(gpu=1),
        ),
    )
    logical.gpu_memory_fraction = 0.8
    expansion = expand_pd_stages([logical], entry_stage="thinker")
    halves = {item.name: item for item in expansion.stages}

    assert halves["thinker_prefill"].gpu_memory_fraction == 0.8
    assert halves["thinker_decode"].gpu_memory_fraction == 0.8


@pytest.mark.parametrize(
    "engine,message",
    [
        ({"page_size": 32}, "requires 1"),
        ({"disable_radix_cache": False}, "requires True"),
    ],
)
def test_a_pd_stage_is_rejected_for_what_the_runtime_cannot_do(engine, message) -> None:
    """The scheduler refuses these in its constructor, after the model loads."""
    from sglang_omni.config.pd_capability import validate_pd_engine_args

    logical = stage(
        "thinker",
        terminal=True,
        pd_disaggregation=PDConfig(
            prefill=PDStagePlacement(gpu=0, engine=engine),
            decode=PDStagePlacement(gpu=1),
        ),
    )
    expansion = expand_pd_stages([logical], entry_stage="thinker")

    with pytest.raises(ValueError, match=message):
        validate_pd_engine_args(expansion.stages)


def test_a_supported_engine_argument_passes() -> None:
    from sglang_omni.config.pd_capability import validate_pd_engine_args

    logical = stage(
        "thinker",
        terminal=True,
        pd_disaggregation=PDConfig(
            prefill=PDStagePlacement(gpu=0, engine={"page_size": 1}),
            decode=PDStagePlacement(gpu=1, engine={"max_running_requests": 64}),
        ),
    )
    expansion = expand_pd_stages([logical], entry_stage="thinker")

    validate_pd_engine_args(expansion.stages)
