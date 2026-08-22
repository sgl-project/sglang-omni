# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import pytest

from sglang_omni.config import (
    PipelineConfig,
    ProcessConfig,
    StageConfig,
    StageResourceConfig,
    StageRuntimeConfig,
    build_process_topology_plan,
    build_stage_placement_plan,
    compile_logical_processes,
)
from sglang_omni.config.manager import ConfigManager
from sglang_omni.pipeline.replicas import expand_replica_stages

_FACTORY = "tests.unit_test.fixtures.pipeline_fakes.dummy_factory"


class _ProcessLocalEdgePipelineConfig(PipelineConfig):
    @classmethod
    def process_local_edges(cls) -> frozenset[tuple[str, str]]:
        return frozenset({("a", "b")})


def _stage(
    name: str,
    *,
    gpu: int | list[int] | None = None,
    fraction: float | None = None,
    process: str | None = None,
    tp_size: int = 1,
    terminal: bool = False,
    next_stage: str | None = None,
) -> StageConfig:
    return StageConfig(
        name=name,
        factory=_FACTORY,
        gpu=gpu,
        process=process,
        tp_size=tp_size,
        runtime=StageRuntimeConfig(
            resources=StageResourceConfig(total_gpu_memory_fraction=fraction)
        ),
        next=next_stage,
        terminal=terminal,
    )


def _compiled_topology_inputs(config: PipelineConfig):
    plan, stages = compile_logical_processes(config)
    stages, replica_topology = expand_replica_stages(stages, plan)
    gpu_placement = build_stage_placement_plan(
        config,
        stages_cfg=stages,
        replica_instances=replica_topology.replicas,
    )
    return plan, stages, gpu_placement


def _compiled_topology(config: PipelineConfig):
    """Physical topology from the compiled logical plan, as the runtime builds it."""
    _, stages, gpu_placement = _compiled_topology_inputs(config)
    return build_process_topology_plan(
        config,
        gpu_placement,
        stages_cfg=stages,
    )


def _process_names(config: PipelineConfig) -> list[str]:
    return [stage.process for stage in config.stages]


def _isolate(config: PipelineConfig, **stage_processes: str) -> PipelineConfig:
    """Move stages into their own processes the way a user's YAML would."""
    return ConfigManager(config).merge_config(
        {
            f"stages.{stage}.process": process
            for stage, process in stage_processes.items()
        }
    )


def _compiled_fractions(config: PipelineConfig) -> dict[str, float | None]:
    _, stages = compile_logical_processes(config)
    return {
        stage.name: stage.runtime.resources.total_gpu_memory_fraction
        for stage in stages
        if stage.gpu is not None
    }


# --- Logical process membership -------------------------------------------------


def test_stage_process_parses_from_schema_and_dotted_overrides() -> None:
    config = PipelineConfig(
        model_path="dummy",
        stages=[
            _stage("a", process="old0", next_stage="b"),
            _stage("b", process="old1", terminal=True),
        ],
    )

    merged = ConfigManager(config).merge_config(
        {"stages.0.process": "p0", "stages.1.process": "p1"}
    )

    assert _process_names(merged) == ["p0", "p1"]


@pytest.mark.parametrize("stage_ref", ["0", "a"])
def test_config_manager_rejects_dotted_gpu_override_with_replica_devices(
    stage_ref: str,
) -> None:
    config = PipelineConfig(
        model_path="dummy",
        stages=[_stage("a", gpu=0, process="p0", terminal=True)],
        processes={
            "p0": ProcessConfig(
                num_replicas=1,
                replica_devices=[1],
            )
        },
    )

    with pytest.raises(
        ValueError,
        match="stages.*gpu.*process 'p0' declares replica_devices",
    ):
        ConfigManager(config).merge_config({f"stages.{stage_ref}.gpu": 2})


def test_config_manager_allows_dotted_gpu_override_without_replica_devices() -> None:
    config = PipelineConfig(
        model_path="dummy",
        stages=[_stage("a", gpu=0, process="p0", terminal=True)],
    )

    merged = ConfigManager(config).merge_config({"stages.a.gpu": 2})

    assert merged.stages[0].gpu == 2


def test_shared_process_name_groups_stages_in_declaration_order() -> None:
    config = PipelineConfig(
        model_path="dummy",
        stages=[
            _stage("a", process="tail", next_stage="b"),
            _stage("b", process="tail", next_stage="c"),
            _stage("c", process="head", terminal=True),
        ],
    )

    plan, _ = compile_logical_processes(config)

    assert [(p.name, p.stage_names) for p in plan.processes] == [
        ("tail", ("a", "b")),
        ("head", ("c",)),
    ]
    assert plan.stage_to_process == {"a": "tail", "b": "tail", "c": "head"}


def test_unreplicated_processes_default_to_one_replica() -> None:
    config = PipelineConfig(
        model_path="dummy",
        stages=[
            _stage("a", process="p0", next_stage="b"),
            _stage("b", process="p1", terminal=True),
        ],
    )

    plan, _ = compile_logical_processes(config)

    assert plan.has_replicas() is False
    assert [p.num_replicas for p in plan.processes] == [1, 1]
    assert plan.replicated_process_names() == ()


def test_processes_policy_attaches_to_declared_process_names() -> None:
    config = PipelineConfig(
        model_path="dummy",
        stages=[
            _stage("a", process="p0", next_stage="b"),
            _stage("b", process="p1", terminal=True),
        ],
        processes={"p1": ProcessConfig(num_replicas=3)},
    )

    plan, _ = compile_logical_processes(config)

    assert plan.num_replicas("p1") == 3
    assert plan.replicated_process_names() == ("p1",)


def test_processes_policy_rejects_unknown_process_name() -> None:
    with pytest.raises(ValueError, match="unknown process name"):
        PipelineConfig(
            model_path="dummy",
            stages=[_stage("a", process="p0", terminal=True)],
            processes={"p1": ProcessConfig(num_replicas=2)},
        )


def test_process_name_cannot_use_the_replica_suffix() -> None:
    with pytest.raises(ValueError, match="reserved"):
        PipelineConfig(
            model_path="dummy",
            stages=[_stage("a", process="p@r1", terminal=True)],
        )


def test_tp_stage_cannot_share_its_process_with_another_stage() -> None:
    with pytest.raises(ValueError, match="cannot be shared"):
        PipelineConfig(
            model_path="dummy",
            stages=[
                _stage("a", process="model", next_stage="thinker"),
                _stage(
                    "thinker", gpu=[0, 1], tp_size=2, process="model", terminal=True
                ),
            ],
        )


def test_tp_stage_process_name_falls_back_to_stage_name() -> None:
    config = PipelineConfig(
        model_path="dummy",
        stages=[
            _stage("a", process="p0", next_stage="b"),
            _stage("b", gpu=[0, 1], tp_size=2, terminal=True),
        ],
    )

    plan, _ = compile_logical_processes(config)

    assert plan.get("b").tp_size == 2
    assert plan.stage_to_process["b"] == "b"


# --- Cross-process edge validation ----------------------------------------------


def test_cross_process_edge_compiles_by_default() -> None:
    config = PipelineConfig(
        model_path="dummy",
        stages=[
            _stage("a", process="p0", next_stage="b"),
            _stage("b", process="p1", terminal=True),
        ],
    )

    plan, _ = compile_logical_processes(config)

    assert [process.name for process in plan.processes] == ["p0", "p1"]


def test_declared_process_local_edge_is_rejected() -> None:
    config = _ProcessLocalEdgePipelineConfig(
        model_path="dummy",
        stages=[
            _stage("a", process="p0", next_stage="b"),
            _stage("b", process="p1", terminal=True),
        ],
    )

    with pytest.raises(ValueError, match="require source and destination"):
        compile_logical_processes(config)


def test_process_local_edge_compiles_inside_one_process() -> None:
    config = _ProcessLocalEdgePipelineConfig(
        model_path="dummy",
        stages=[
            _stage("a", process="p", next_stage="b"),
            _stage("b", process="p", terminal=True),
        ],
    )

    plan, _ = compile_logical_processes(config)

    assert plan.get("p").stage_names == ("a", "b")


def test_tp_stage_edges_compile_by_default() -> None:
    config = PipelineConfig(
        model_path="dummy",
        stages=[
            _stage("a", process="p0", next_stage="b"),
            _stage("b", gpu=[0, 1], tp_size=2, terminal=True),
        ],
    )

    plan, _ = compile_logical_processes(config)

    assert plan.stage_to_process == {"a": "p0", "b": "b"}


def test_stream_and_wait_for_edges_respect_process_local_constraint() -> None:
    stages = [
        StageConfig(
            name="a",
            factory=_FACTORY,
            process="p0",
            next="b",
            stream_to=["b"],
        ),
        StageConfig(
            name="b",
            factory=_FACTORY,
            process="p1",
            terminal=True,
            wait_for=["a"],
            merge_fn=_FACTORY,
        ),
    ]

    config = _ProcessLocalEdgePipelineConfig(model_path="dummy", stages=stages)
    with pytest.raises(ValueError, match="require source and destination"):
        compile_logical_processes(config)


# --- Model topologies -----------------------------------------------------------


def test_moss_tts_local_default_isolates_vocoder_with_declared_fractions() -> None:
    from sglang_omni.models.moss_tts_local.config import MossTTSLocalPipelineConfig

    config = MossTTSLocalPipelineConfig(model_path="dummy")

    assert _process_names(config) == ["pipeline", "pipeline", "vocoder"]
    assert _compiled_fractions(config) == {
        "preprocessing": 0.15,
        "tts_engine": 0.67,
        "vocoder": 0.18,
    }
    assert [
        (group.name, group.stage_names) for group in _compiled_topology(config).groups
    ] == [
        ("pipeline", ("preprocessing", "tts_engine")),
        ("vocoder", ("vocoder",)),
    ]
    assert build_stage_placement_plan(config).gpus[
        0
    ].total_gpu_memory_fraction == pytest.approx(1.0)


def test_moss_tts_local_split_rejects_splitting_the_pipeline() -> None:
    from sglang_omni.models.moss_tts_local.config import MossTTSLocalSplitPipelineConfig

    config = MossTTSLocalSplitPipelineConfig(model_path="dummy")
    assert _process_names(config) == ["pipeline"] * 3

    isolated = _isolate(config, preprocessing="frontend")

    with pytest.raises(ValueError, match="Cross-process edge"):
        compile_logical_processes(isolated)


def test_ming_tts_default_stays_in_one_process() -> None:
    from sglang_omni.models.ming_tts.config import MingTTSPipelineConfig

    config = MingTTSPipelineConfig(model_path="dummy")
    assert _process_names(config) == ["pipeline"] * 4
    compile_logical_processes(config)


def test_fishaudio_default_splits_preprocessing_from_the_pipeline() -> None:
    from sglang_omni.models.fishaudio_s2_pro.config import S2ProPipelineConfig

    config = S2ProPipelineConfig(model_path="dummy")
    assert _process_names(config) == ["preprocessing", "pipeline", "pipeline"]
    compile_logical_processes(config)


def test_qwen3_tts_rejects_splitting_preprocessing_from_the_engine() -> None:
    from sglang_omni.models.qwen3_tts.config import Qwen3TTSPipelineConfig

    config = Qwen3TTSPipelineConfig(model_path="dummy")
    compile_logical_processes(config)

    isolated = _isolate(config, tts_engine="engine")

    with pytest.raises(ValueError, match="'preprocessing' -> 'tts_engine'"):
        compile_logical_processes(isolated)


def test_qwen3_tts_can_isolate_vocoder_with_explicit_fractions() -> None:
    from sglang_omni.models.qwen3_tts.config import Qwen3TTSPipelineConfig

    config = ConfigManager(Qwen3TTSPipelineConfig(model_path="dummy")).merge_config(
        {
            "stages.vocoder.process": "vocoder",
            "stages.tts_engine.runtime.resources.total_gpu_memory_fraction": 0.85,
            "stages.vocoder.runtime.resources.total_gpu_memory_fraction": 0.10,
        }
    )

    assert _compiled_fractions(config) == {"tts_engine": 0.85, "vocoder": 0.10}
    assert [
        (group.name, group.stage_names) for group in _compiled_topology(config).groups
    ] == [
        ("pipeline", ("preprocessing", "tts_engine")),
        ("vocoder", ("vocoder",)),
    ]


def test_minimax_music3_allows_ar_acoustic_process_boundary() -> None:
    from sglang_omni.models.minimax_music3.config import (
        MiniMaxMusic3DualGPUPipelineConfig,
    )

    config = MiniMaxMusic3DualGPUPipelineConfig(model_path="dummy")

    plan, _ = compile_logical_processes(config)

    assert plan.stage_to_process == {
        "preprocessing": "minimax_music3_ar",
        "minimax_music3_ar": "minimax_music3_ar",
        "dit_dav": "minimax_music3_dit_dav",
    }


def test_minimax_music3_rejects_splitting_preprocessing_from_ar() -> None:
    from sglang_omni.models.minimax_music3.config import (
        MiniMaxMusic3DualGPUPipelineConfig,
    )

    config = MiniMaxMusic3DualGPUPipelineConfig(model_path="dummy")
    isolated = _isolate(config, preprocessing="preprocessing")

    with pytest.raises(ValueError, match="'preprocessing' -> 'minimax_music3_ar'"):
        compile_logical_processes(isolated)


def test_higgs_default_groups_the_frontend_and_can_split_it_further() -> None:
    from sglang_omni.models.higgs_tts.config import HiggsTtsPipelineConfig

    config = HiggsTtsPipelineConfig(model_path="dummy")
    assert [
        (group.name, group.stage_names) for group in _compiled_topology(config).groups
    ] == [
        ("tts_frontend", ("preprocessing", "audio_encoder")),
        ("pipeline", ("tts_engine", "vocoder")),
    ]

    isolated = _isolate(config, audio_encoder="audio_encoder")

    assert [
        (group.name, group.stage_names) for group in _compiled_topology(isolated).groups
    ] == [
        ("tts_frontend", ("preprocessing",)),
        ("audio_encoder", ("audio_encoder",)),
        ("pipeline", ("tts_engine", "vocoder")),
    ]
    compile_logical_processes(isolated)


def test_qwen3_omni_speech_default_topology_compiles() -> None:
    from sglang_omni.models.qwen3_omni.config import Qwen3OmniSpeechPipelineConfig

    config = Qwen3OmniSpeechPipelineConfig(model_path="dummy")

    plan, _ = compile_logical_processes(config)

    assert [p.name for p in plan.processes] == [
        "preprocessing",
        "image_encoder",
        "audio_encoder",
        "thinker",
        "decode",
        "talker_ar",
        "code2wav",
    ]


def test_ming_omni_speech_default_topology_compiles() -> None:
    from sglang_omni.models.ming_omni.config import MingOmniSpeechPipelineConfig

    config = MingOmniSpeechPipelineConfig(model_path="dummy")

    plan, _ = compile_logical_processes(config)

    assert plan.stage_to_process["talker"] == "talker"


# --- Removed configuration entry points -----------------------------------------


@pytest.mark.parametrize(
    ("flag", "value", "config_field"),
    [
        ("--isolate-stage", "b", "isolate_stage"),
        ("--stage-process", "a=frontend", "stage_process"),
    ],
)
def test_serve_cli_rejects_removed_process_flags_from_valid_config(
    monkeypatch: pytest.MonkeyPatch,
    flag: str,
    value: str,
    config_field: str,
) -> None:
    from typer.testing import CliRunner

    from sglang_omni.cli import app

    config = PipelineConfig(
        model_path="dummy",
        stages=[_stage("a", process="pipeline", terminal=True)],
    )
    monkeypatch.setattr(
        ConfigManager,
        "from_file",
        staticmethod(lambda _path: ConfigManager(config)),
    )

    result = CliRunner().invoke(
        app,
        ["serve", "--config", "valid.yaml", flag, value],
    )

    assert result.exit_code != 0
    assert result.exception is not None
    assert config_field in str(result.exception)
    assert "Extra inputs are not permitted" in str(result.exception)


def test_fused_stages_config_entry_is_removed() -> None:
    with pytest.raises(ValueError, match="fused_stages") as exc_info:
        PipelineConfig(
            model_path="dummy",
            stages=[
                _stage("a", process="p", next_stage="b"),
                _stage("b", process="p", terminal=True),
            ],
            fused_stages=[["a", "b"]],
        )
    assert "Extra inputs are not permitted" in str(exc_info.value)


# --- Physical process topology --------------------------------------------------


def test_non_tp_stages_must_declare_process() -> None:
    with pytest.raises(ValueError, match="Non-TP stages must declare process"):
        PipelineConfig(
            model_path="dummy",
            stages=[
                _stage("a", process="p0", next_stage="b"),
                _stage("b", terminal=True),
            ],
        )


def test_missing_non_tp_process_declaration_is_rejected() -> None:
    with pytest.raises(ValueError, match="Non-TP stages must declare process"):
        PipelineConfig(
            model_path="dummy",
            stages=[_stage("a", next_stage="b"), _stage("b", terminal=True)],
        )


def test_tp_process_names_are_derived_when_process_is_missing() -> None:
    config = PipelineConfig(
        model_path="dummy",
        stages=[_stage("thinker", gpu=[0, 1], tp_size=2, terminal=True)],
    )

    topology = _compiled_topology(config)

    assert topology.groups == ()
    assert topology.tp_stage_to_processes == {"thinker": ("thinker_tp0", "thinker_tp1")}


def test_tp_process_field_is_used_as_rank_process_prefix() -> None:
    config = PipelineConfig(
        model_path="dummy",
        stages=[
            _stage(
                "thinker",
                gpu=[0, 1],
                tp_size=2,
                process="model",
                terminal=True,
            )
        ],
    )

    topology = _compiled_topology(config)

    assert topology.tp_stage_to_processes == {"thinker": ("model_tp0", "model_tp1")}


def test_same_process_same_gpu_does_not_require_memory_budgets() -> None:
    config = PipelineConfig(
        model_path="dummy",
        stages=[
            _stage("a", gpu=0, process="p0", next_stage="b"),
            _stage("b", gpu=0, process="p0", terminal=True),
        ],
    )

    topology = _compiled_topology(config)

    assert [
        (group.name, group.stage_names, group.gpu_id) for group in topology.groups
    ] == [("p0", ("a", "b"), 0)]


def test_same_gpu_multiple_processes_accepts_explicit_budgets() -> None:
    config = PipelineConfig(
        model_path="dummy",
        stages=[
            _stage("a", gpu=0, fraction=0.20, process="p0", next_stage="b"),
            _stage("b", gpu=0, fraction=0.30, process="p0", next_stage="c"),
            _stage("c", gpu=0, fraction=0.40, process="p1", terminal=True),
        ],
    )

    topology = _compiled_topology(config)

    assert [
        (group.name, group.stage_names, group.gpu_id) for group in topology.groups
    ] == [
        ("p0", ("a", "b"), 0),
        ("p1", ("c",), 0),
    ]


def test_same_gpu_multiple_processes_rejects_missing_budget() -> None:
    config = PipelineConfig(
        model_path="dummy",
        stages=[
            _stage("a", gpu=0, fraction=0.20, process="p0", next_stage="b"),
            _stage("b", gpu=0, process="p1", terminal=True),
        ],
    )
    _, stages, gpu_placement = _compiled_topology_inputs(config)

    with pytest.raises(ValueError, match="total_gpu_memory_fraction"):
        build_process_topology_plan(
            config,
            gpu_placement,
            stages_cfg=stages,
        )


def test_replica_memory_fraction_opt_out_still_rejects_over_budget() -> None:
    from sglang_omni.config import PlacementConfig

    config = PipelineConfig(
        model_path="dummy",
        stages=[_stage("a", gpu=0, fraction=0.6, process="p0", terminal=True)],
        processes={"p0": ProcessConfig(num_replicas=2, replica_devices=[0, 0])},
        placement=PlacementConfig(require_memory_fraction_for_colocation=False),
    )

    with pytest.raises(ValueError, match="exceeds placement limit"):
        _compiled_topology(config)


def test_one_process_group_cannot_span_multiple_gpus() -> None:
    config = PipelineConfig(
        model_path="dummy",
        stages=[
            _stage("a", gpu=0, process="p0", next_stage="b"),
            _stage("b", gpu=1, process="p0", terminal=True),
        ],
    )
    _, stages, gpu_placement = _compiled_topology_inputs(config)

    with pytest.raises(ValueError, match="spans multiple GPUs"):
        build_process_topology_plan(
            config,
            gpu_placement,
            stages_cfg=stages,
        )


def test_tp_process_names_must_not_collide_with_non_tp_process_group() -> None:
    config = PipelineConfig(
        model_path="dummy",
        stages=[
            _stage("a", process="thinker_tp0", next_stage="thinker"),
            _stage("thinker", gpu=[0, 1], tp_size=2, terminal=True),
        ],
    )
    _, stages, gpu_placement = _compiled_topology_inputs(config)

    with pytest.raises(ValueError, match="collide"):
        build_process_topology_plan(
            config,
            gpu_placement,
            stages_cfg=stages,
        )


def test_tp_process_names_must_be_unique_across_tp_stages() -> None:
    with pytest.raises(ValueError, match="claimed by multiple TP stages"):
        PipelineConfig(
            model_path="dummy",
            stages=[
                _stage("a", gpu=[0, 1], tp_size=2, process="model", next_stage="b"),
                _stage("b", gpu=[2, 3], tp_size=2, process="model", terminal=True),
            ],
        )


def test_replica_induced_gpu_sharing_requires_memory_fractions_despite_opt_out() -> (
    None
):
    from sglang_omni.config import PlacementConfig

    config = PipelineConfig(
        model_path="dummy",
        stages=[_stage("a", gpu=0, process="p0", terminal=True)],
        processes={"p0": ProcessConfig(num_replicas=2, replica_devices=[0, 0])},
        placement=PlacementConfig(require_memory_fraction_for_colocation=False),
    )

    with pytest.raises(ValueError, match="replica-induced GPU sharing"):
        _compiled_topology(config)


def test_single_replica_device_sharing_requires_memory_fractions_despite_opt_out() -> (
    None
):
    from sglang_omni.config import PlacementConfig

    config = PipelineConfig(
        model_path="dummy",
        stages=[
            _stage("a", gpu=1, process="p0", next_stage="b"),
            _stage("b", gpu=0, process="p1", terminal=True),
        ],
        processes={"p0": ProcessConfig(num_replicas=1, replica_devices=[0])},
        placement=PlacementConfig(require_memory_fraction_for_colocation=False),
    )

    with pytest.raises(ValueError, match="replica-induced GPU sharing"):
        _compiled_topology(config)


def test_replica_induced_gpu_sharing_accepts_declared_fractions() -> None:
    from sglang_omni.config import PlacementConfig

    config = PipelineConfig(
        model_path="dummy",
        stages=[_stage("a", gpu=0, fraction=0.4, process="p0", terminal=True)],
        processes={"p0": ProcessConfig(num_replicas=2, replica_devices=[0, 0])},
        placement=PlacementConfig(require_memory_fraction_for_colocation=False),
    )
    plan, stages = compile_logical_processes(config)
    expanded, topology = expand_replica_stages(stages, plan)
    gpu_placement = build_stage_placement_plan(
        config,
        stages_cfg=expanded,
        replica_instances=topology.replicas,
    )
    process_plan = build_process_topology_plan(
        config,
        gpu_placement,
        stages_cfg=expanded,
    )

    assert [group.name for group in process_plan.groups] == ["p0@r0", "p0@r1"]


def test_pre_existing_gpu_sharing_still_honours_the_model_opt_out() -> None:
    from sglang_omni.config import PlacementConfig

    config = PipelineConfig(
        model_path="dummy",
        stages=[
            _stage("a", gpu=0, process="p0", next_stage="b"),
            _stage("b", gpu=0, process="p1", terminal=True),
        ],
        placement=PlacementConfig(require_memory_fraction_for_colocation=False),
    )
    plan, stages = compile_logical_processes(config)
    expanded, topology = expand_replica_stages(stages, plan)
    gpu_placement = build_stage_placement_plan(
        config,
        stages_cfg=expanded,
        replica_instances=topology.replicas,
    )
    process_plan = build_process_topology_plan(
        config,
        gpu_placement,
        stages_cfg=expanded,
    )

    assert [group.name for group in process_plan.groups] == ["p0", "p1"]
