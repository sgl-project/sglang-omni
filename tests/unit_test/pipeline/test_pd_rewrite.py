# SPDX-License-Identifier: Apache-2.0
"""Structural tests for the PD graph rewrite."""

from __future__ import annotations

import pytest

from sglang_omni.config import expand_pd_stages
from sglang_omni.config.schema import PDConfig, PDStagePlacement, PipelineConfig
from sglang_omni.pipeline.runtime_config import prepare_pipeline_runtime
from tests.unit_test.fixtures.pipeline_fakes import fake_factory_path
from tests.unit_test.pipeline.helpers import stage


def _pd(prefill_gpu: int, decode_gpu: int) -> PDConfig:
    return PDConfig(
        prefill=PDStagePlacement(gpu=prefill_gpu),
        decode=PDStagePlacement(gpu=decode_gpu),
    )


def _by_name(stages: list) -> dict[str, object]:
    return {s.name: s for s in stages}


def test_pd_expansion_rewrites_edges_and_metadata() -> None:
    """A single representative pipeline exercises all structural invariants."""
    config = PipelineConfig(
        model_path="dummy",
        stages=[
            stage(
                "pre",
                next="thinker",
                project_payload={
                    "thinker": "tests.unit_test.fixtures.pipeline_fakes.project_payload"
                },
            ),
            stage(
                "thinker",
                next="post",
                wait_for=["pre"],
                merge_fn="tests.unit_test.fixtures.pipeline_fakes.merge_payloads",
                stream_to=["sink"],
                stream_done_to_fn=fake_factory_path("identity_stream_targets"),
                pd_disaggregation=_pd(1, 2),
            ),
            stage("post", terminal=True),
            stage("sink", terminal=True),
        ],
    )

    expansion = expand_pd_stages(
        list(config.stages), entry_stage=config.resolved_entry_stage
    )
    stages = _by_name(expansion.stages)

    assert [s.name for s in expansion.stages] == [
        "pre",
        "thinker_prefill",
        "thinker_decode",
        "post",
        "sink",
    ]

    assert expansion.routing_map == {"thinker": "thinker_prefill"}
    assert expansion.output_map == {"thinker": "thinker_decode"}

    assert stages["pre"].next == "thinker_prefill"
    assert stages["pre"].project_payload == {
        "thinker_prefill": "tests.unit_test.fixtures.pipeline_fakes.project_payload"
    }

    prefill = stages["thinker_prefill"]
    decode = stages["thinker_decode"]

    assert prefill.wait_for == ["pre"]
    assert prefill.merge_fn == "tests.unit_test.fixtures.pipeline_fakes.merge_payloads"
    assert prefill.gpu == 1
    assert prefill.process == "thinker_prefill"
    assert prefill.next == "post"
    assert not prefill.terminal
    assert prefill.stream_to == ["sink"]
    assert prefill.stream_done_to_fn == fake_factory_path("identity_stream_targets")

    assert decode.next == "post"
    assert decode.stream_to == ["sink"]
    assert decode.stream_done_to_fn == fake_factory_path("identity_stream_targets")
    assert decode.gpu == 2
    assert decode.process == "thinker_decode"
    assert not decode.wait_for
    assert not decode.merge_fn

    assert prefill.pd_execution.role == "prefill"
    assert prefill.pd_execution.partner == "thinker_decode"
    assert decode.pd_execution.role == "decode"
    assert decode.pd_execution.partner == "thinker_prefill"
    assert prefill.pd_execution.scheduler_class.endswith(".OmniPrefillScheduler")
    assert decode.pd_execution.scheduler_class.endswith(".OmniDecodeScheduler")
    assert "pd_role" not in prefill.factory.model_dump()
    assert "pd_partner" not in decode.factory.model_dump()


def test_pd_entry_stage_rewritten_to_prefill() -> None:
    config = PipelineConfig(
        model_path="dummy",
        entry_stage="thinker",
        stages=[stage("thinker", terminal=True, pd_disaggregation=_pd(0, 1))],
    )
    expansion = expand_pd_stages(
        list(config.stages), entry_stage=config.resolved_entry_stage
    )

    assert expansion.entry_stage == "thinker_prefill"
    assert expansion.routing_map == {"thinker": "thinker_prefill"}
    assert expansion.output_map == {"thinker": "thinker_decode"}
    assert [s.name for s in expansion.stages if s.terminal] == ["thinker_decode"]


def test_pd_expansion_is_idempotent() -> None:
    config = PipelineConfig(
        model_path="dummy",
        stages=[
            stage("pre", next="thinker"),
            stage("thinker", next="post", pd_disaggregation=_pd(0, 1)),
            stage("post", terminal=True),
        ],
    )
    once = expand_pd_stages(
        list(config.stages), entry_stage=config.resolved_entry_stage
    )
    twice = expand_pd_stages(once.stages, entry_stage=once.entry_stage)

    assert [s.name for s in twice.stages] == [s.name for s in once.stages]
    assert twice.entry_stage == once.entry_stage
    assert twice.routing_map == twice.output_map == {}


def test_non_pd_pipeline_is_unchanged() -> None:
    config = PipelineConfig(
        model_path="dummy",
        stages=[stage("pre", next="post"), stage("post", terminal=True)],
    )
    expansion = expand_pd_stages(
        list(config.stages), entry_stage=config.resolved_entry_stage
    )

    assert [s.name for s in expansion.stages] == ["pre", "post"]
    assert expansion.entry_stage == "pre"
    assert expansion.routing_map == expansion.output_map == {}


def test_pd_output_identity_drives_fan_in_sources() -> None:
    config = PipelineConfig(
        model_path="dummy",
        stages=[
            stage(
                "source",
                factory_path=fake_factory_path("pd_capable_factory"),
                next="fanin",
                pd_disaggregation=_pd(0, 1),
            ),
            stage(
                "fanin",
                wait_for=["source"],
                wait_for_fn=fake_factory_path("identity_wait_sources"),
                merge_fn=fake_factory_path("merge_payloads"),
                terminal=True,
            ),
        ],
    )

    prep = prepare_pipeline_runtime(config)
    try:
        stages = _by_name(prep.stages_cfg)
        assert stages["source_decode"].next == "fanin"
        assert stages["fanin"].wait_for == ["source_decode"]
        assert prep.name_map["source"] == "source_prefill"
        assert prep.source_name_map["source"] == "source_decode"
        assert prep.terminal_name_map["source"] == "source_decode"
    finally:
        prep.runtime_dir.close()


def test_pd_source_and_destination_use_opposite_identities() -> None:
    projector = fake_factory_path("project_payload")
    config = PipelineConfig(
        model_path="dummy",
        stages=[
            stage(
                "source",
                next="target",
                project_payload={"target": projector},
                pd_disaggregation=_pd(0, 1),
            ),
            stage(
                "target",
                wait_for=["source"],
                merge_fn=fake_factory_path("merge_payloads"),
                next="sink",
                pd_disaggregation=_pd(2, 3),
            ),
            stage("sink", terminal=True),
        ],
    )

    expansion = expand_pd_stages(
        list(config.stages), entry_stage=config.resolved_entry_stage
    )
    stages = _by_name(expansion.stages)

    for source_name in ("source_prefill", "source_decode"):
        assert stages[source_name].next == "target_prefill"
        assert stages[source_name].project_payload == {"target_prefill": projector}
    assert stages["target_prefill"].wait_for == ["source_decode"]
    assert expansion.routing_map == {
        "source": "source_prefill",
        "target": "target_prefill",
    }
    assert expansion.output_map == {
        "source": "source_decode",
        "target": "target_decode",
    }


@pytest.mark.parametrize(
    "kwargs,message",
    [
        (
            {"terminal": True, "pd_disaggregation": _pd(0, 0)},
            "cannot share the same GPU",
        ),
        (
            {
                "terminal": True,
                "pd_disaggregation": PDConfig(prefill=PDStagePlacement(gpu=0)),
            },
            "requires explicit",
        ),
        (
            {"next": "thinker_decode", "pd_disaggregation": _pd(0, 1)},
            "collides with an existing stage",
        ),
    ],
)
def test_pd_validation_rejects_invalid_configs(kwargs, message) -> None:
    with pytest.raises(ValueError, match=message):
        PipelineConfig(
            model_path="dummy",
            stages=[
                stage("thinker", **kwargs),
                stage("thinker_decode", terminal=True),
            ],
        )
