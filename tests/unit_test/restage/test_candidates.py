"""Candidate export must honor runtime process boundaries and GPU budgets."""

import pytest

from sglang_omni.config.placement import build_stage_placement_plan
from sglang_omni.config.schema import PipelineConfig
from sglang_omni.config.topology import (
    build_process_topology_plan,
    compile_logical_processes,
)
from sglang_omni.pipeline.replicas import expand_replica_stages
from sglang_omni.restage.candidates import materialize_candidate
from tests.unit_test.restage.conftest_seed import pipeline


def test_single_replica_preserves_normal_serving_colocation_policy():
    source = pipeline()
    source.placement.require_memory_fraction_for_colocation = False
    source.stages[-1].gpu_memory_fraction = None
    logical, stages = compile_logical_processes(source)
    expanded, topology = expand_replica_stages(stages, logical)
    placement = build_stage_placement_plan(source, stages_cfg=expanded)
    build_process_topology_plan(source, placement, stages_cfg=expanded)
    candidate = materialize_candidate(
        source, {"engine": ((2,),), "tail": ((2,),)}, (2, 5)
    )
    assert candidate.stage_named("engine").gpu == 2
    assert candidate.stage_named("tail").gpu == 2
    assert candidate.processes["engine"].replica_devices is None


def test_actual_replicas_still_require_colocation_memory_budgets():
    source = pipeline()
    source.placement.require_memory_fraction_for_colocation = False
    source.stages[-1].gpu_memory_fraction = None
    with pytest.raises(ValueError, match="replica-induced GPU sharing"):
        materialize_candidate(source, {"engine": ((0,),), "tail": ((1,), (1,))}, (0, 1))


def test_model_specific_placement_constraints_run_before_export():
    from sglang_omni.models.ming_omni.config import MingOmniSpeechPipelineConfig

    source = MingOmniSpeechPipelineConfig(model_path="unused-checkpoint")
    with pytest.raises(ValueError, match="collides"):
        materialize_candidate(
            source,
            {
                "audio_encoder": ((0,),),
                "image_encoder": ((0,),),
                "thinker": ((0,),),
                "talker": ((0,),),
            },
            (0, 1),
        )


def test_tp_replicas_match_exported_and_expanded_device_mapping():
    source = pipeline(2)
    source.stages[1].gpu_memory_fraction = 0.2
    candidate = materialize_candidate(
        source,
        {
            "engine": ((2, 5), (7, 9)),
            "tail": ((9,),),
        },
        (2, 5, 7, 9),
    )
    assert candidate.stage_named("engine").gpu == [2, 5]
    assert candidate.stage_named("tail").gpu == 9
    logical, stages = compile_logical_processes(candidate)
    expanded, topology = expand_replica_stages(stages, logical)
    placements = build_stage_placement_plan(
        candidate, stages_cfg=expanded, replica_instances=topology.replicas
    )
    assert [p.gpu_ids for p in placements.instances_of("engine")] == [(2, 5), (7, 9)]
    assert placements.stages["tail"].gpu_ids == (9,)


def test_model_candidate_roundtrips_through_serving_yaml_loader(tmp_path):
    import yaml

    from sglang_omni.config.resolver import ConfigResolver
    from sglang_omni.config.sources import dump_user_config, sources_from_config_file
    from sglang_omni.models.qwen3_tts.config import Qwen3TTSPipelineConfig

    source = Qwen3TTSPipelineConfig(model_path="unused-local-checkpoint")
    result = materialize_candidate(source, {"pipeline": ((2,), (5,))}, (2, 5))
    output = tmp_path / "candidate.yaml"
    output.write_text(yaml.safe_dump(dump_user_config(result)), encoding="utf-8")
    baseline, patches = sources_from_config_file(str(output))
    restored = ConfigResolver(baseline).resolve(patches).config
    assert type(restored) is Qwen3TTSPipelineConfig
    assert restored.model_dump() == result.model_dump()


def test_replicas_roundtrip_with_noncontiguous_device_budget():
    source = pipeline()
    before = source.model_dump()
    result = materialize_candidate(
        source, {"engine": ((2,),), "tail": ((5,), (5,))}, (2, 5)
    )
    restored = PipelineConfig.model_validate_json(result.model_dump_json())
    logical, _ = compile_logical_processes(restored)
    assert logical.get("tail").replica_devices == ((5,), (5,))
    assert logical.get("engine").replica_devices is None
    assert restored.stage_named("engine").gpu == 2
    assert restored.stages[0].gpu is None
    assert source.model_dump() == before


def test_shared_process_members_are_replicated_together():
    source = pipeline()
    source.stages[0].process = "engine"
    result = materialize_candidate(
        source, {"engine": ((0,), (1,)), "tail": ((2,),)}, (0, 1, 2, 3)
    )
    logical, stages = compile_logical_processes(result)
    expanded, topology = expand_replica_stages(stages, logical)
    assert len(topology.replicas["cpu"]) == 2
    assert len(topology.replicas["engine"]) == 2
    cpu_copies = [stage for stage in expanded if stage.name in topology.replicas["cpu"]]
    assert all(stage.gpu is None for stage in cpu_copies)
    assert {stage.process for stage in cpu_copies} == {"engine@r0", "engine@r1"}


@pytest.mark.parametrize(
    "assignments,match",
    [
        ({"engine": ((0,),), "tail": ((2,),)}, "budget"),
        ({"engine": ((0,),)}, "GPU processes"),
        ({"engine": ((0,),), "tail": ((1,),), "front": ((0,),)}, "GPU processes"),
        ({"engine": ((0, 1),), "tail": ((1,),)}, "tp_size"),
        ({"engine": (), "tail": ((1,),)}, "replica"),
    ],
)
def test_unlaunchable_assignments_are_rejected(assignments, match):
    with pytest.raises(ValueError, match=match):
        materialize_candidate(pipeline(), assignments, (0, 1))
