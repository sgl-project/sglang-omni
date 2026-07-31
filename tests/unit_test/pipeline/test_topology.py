# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import pytest

from sglang_omni.config import (
    PipelineConfig,
    StageConfig,
    StageResourceConfig,
    StageRuntimeConfig,
    apply_stage_process_overrides,
    build_process_topology_plan,
    build_stage_placement_plan,
    parse_stage_process_assignment,
)
from sglang_omni.config.manager import ConfigManager

_FACTORY = "tests.unit_test.fixtures.pipeline_fakes.dummy_factory"

_PLACEMENT_POLICY_CALLS: list[str] = []


def _recording_placement_policy(config: PipelineConfig, plan: object) -> None:
    _PLACEMENT_POLICY_CALLS.append(config.name or "")


class _IsolationPipelineConfig(PipelineConfig):
    @classmethod
    def process_safe_edges(cls) -> frozenset[tuple[str, str]]:
        return frozenset(
            {
                ("a", "b"),
                ("b", "c"),
                ("a", "audio_decode"),
                ("vocoder", "audio_decode"),
            }
        )

    @classmethod
    def process_edge_resources(
        cls,
    ) -> dict[tuple[str, str], dict[str, float]]:
        return {}


class _IsolationAliasPipelineConfig(_IsolationPipelineConfig):
    @classmethod
    def isolation_role_to_stage(cls) -> dict[str, str]:
        return {
            "vocoder": "audio_decode",
            "tp_role": "thinker",
            "missing_role": "missing",
        }


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


def _topology(config: PipelineConfig):
    gpu_placement = build_stage_placement_plan(config)
    return build_process_topology_plan(config, gpu_placement)


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

    assert [stage.process for stage in merged.stages] == ["p0", "p1"]


def test_process_override_none_preserves_declared_topology() -> None:
    config = _IsolationPipelineConfig(
        model_path="dummy",
        stages=[
            _stage("a", process="pipeline", next_stage="b"),
            _stage("b", process="pipeline", terminal=True),
        ],
    )

    overridden = apply_stage_process_overrides(config, isolate_stages=None)

    assert overridden is config
    assert [stage.process for stage in overridden.stages] == [
        "pipeline",
        "pipeline",
    ]


def test_process_override_isolates_named_stage_without_mutating_source() -> None:
    config = _IsolationPipelineConfig(
        model_path="dummy",
        stages=[
            _stage("a", process="pipeline", next_stage="b"),
            _stage("b", process="pipeline", terminal=True),
        ],
    )

    overridden = apply_stage_process_overrides(config, isolate_stages=["b"])

    assert overridden is not config
    assert [stage.process for stage in config.stages] == ["pipeline", "pipeline"]
    assert [stage.process for stage in overridden.stages] == ["pipeline", "b"]


def test_process_override_isolates_multiple_stages_separately() -> None:
    config = _IsolationPipelineConfig(
        model_path="dummy",
        stages=[
            _stage("a", process="pipeline", next_stage="b"),
            _stage("b", process="pipeline", next_stage="c"),
            _stage("c", process="pipeline", terminal=True),
        ],
    )

    overridden = apply_stage_process_overrides(
        config,
        isolate_stages=["b", "c"],
    )

    assert [stage.process for stage in overridden.stages] == ["pipeline", "b", "c"]


def test_process_override_rejects_unknown_stage() -> None:
    config = _IsolationPipelineConfig(
        model_path="dummy",
        stages=[_stage("a", process="pipeline", terminal=True)],
    )

    with pytest.raises(ValueError, match="Unknown stage or isolation role: missing"):
        apply_stage_process_overrides(config, isolate_stages=["missing"])


def test_process_override_literal_stage_takes_precedence_over_alias() -> None:
    config = _IsolationAliasPipelineConfig(
        model_path="dummy",
        stages=[
            _stage("vocoder", process="pipeline", next_stage="audio_decode"),
            _stage("audio_decode", process="pipeline", terminal=True),
        ],
    )

    overridden = apply_stage_process_overrides(config, isolate_stages=["vocoder"])

    assert [stage.process for stage in overridden.stages] == [
        "vocoder",
        "pipeline",
    ]


def test_process_override_alias_does_not_mutate_source_config() -> None:
    config = _IsolationAliasPipelineConfig(
        model_path="dummy",
        stages=[
            _stage("a", process="pipeline", next_stage="audio_decode"),
            _stage("audio_decode", process="pipeline", terminal=True),
        ],
    )

    overridden = apply_stage_process_overrides(config, isolate_stages=["vocoder"])

    assert [stage.process for stage in config.stages] == ["pipeline", "pipeline"]
    assert [stage.process for stage in overridden.stages] == [
        "pipeline",
        "audio_decode",
    ]


def test_process_override_rejects_alias_with_missing_stage() -> None:
    config = _IsolationAliasPipelineConfig(
        model_path="dummy",
        stages=[_stage("audio_decode", process="pipeline", terminal=True)],
    )

    with pytest.raises(
        ValueError,
        match="Unknown stage or isolation role: missing_role",
    ):
        apply_stage_process_overrides(config, isolate_stages=["missing_role"])


def test_process_override_rejects_stage_not_declared_process_safe() -> None:
    config = PipelineConfig(
        model_path="dummy",
        stages=[
            _stage("preprocessing", process="pipeline", next_stage="tts_engine"),
            _stage("tts_engine", process="pipeline", terminal=True),
        ],
    )

    with pytest.raises(
        ValueError,
        match="Process split across 'preprocessing' -> 'tts_engine' is not supported",
    ):
        apply_stage_process_overrides(config, isolate_stages=["preprocessing"])


def test_process_override_rejects_tp_stage() -> None:
    config = PipelineConfig(
        model_path="dummy",
        stages=[
            _stage(
                "thinker",
                gpu=[0, 1],
                tp_size=2,
                terminal=True,
            )
        ],
    )

    with pytest.raises(ValueError, match="already uses one process per TP rank"):
        apply_stage_process_overrides(config, isolate_stages=["thinker"])


def test_process_override_rejects_alias_to_tp_stage() -> None:
    config = _IsolationAliasPipelineConfig(
        model_path="dummy",
        stages=[
            _stage(
                "thinker",
                gpu=[0, 1],
                tp_size=2,
                terminal=True,
            )
        ],
    )

    with pytest.raises(
        ValueError,
        match="Stage 'thinker' already uses one process per TP rank",
    ):
        apply_stage_process_overrides(config, isolate_stages=["tp_role"])


def test_process_override_same_gpu_requires_memory_fractions() -> None:
    config = _IsolationPipelineConfig(
        model_path="dummy",
        stages=[
            _stage("a", gpu=0, process="pipeline", next_stage="b"),
            _stage("b", gpu=0, process="pipeline", terminal=True),
        ],
    )
    with pytest.raises(ValueError, match="total_gpu_memory_fraction"):
        apply_stage_process_overrides(config, isolate_stages=["b"])


def test_process_override_same_gpu_accepts_valid_memory_fractions() -> None:
    config = _IsolationPipelineConfig(
        model_path="dummy",
        stages=[
            _stage(
                "a",
                gpu=0,
                fraction=0.40,
                process="pipeline",
                next_stage="b",
            ),
            _stage(
                "b",
                gpu=0,
                fraction=0.40,
                process="pipeline",
                terminal=True,
            ),
        ],
    )
    overridden = apply_stage_process_overrides(config, isolate_stages=["b"])

    topology = _topology(overridden)

    assert [(group.name, group.stage_names) for group in topology.groups] == [
        ("pipeline", ("a",)),
        ("b", ("b",)),
    ]


def test_process_override_rejects_process_name_collision() -> None:
    # Note (Akazaakane): 'b' must share a process with another stage, otherwise it is
    # already isolated and the override is a no-op before the collision can happen.
    config = _IsolationPipelineConfig(
        model_path="dummy",
        stages=[
            _stage("a", process="b", next_stage="b"),
            _stage("b", process="pipeline", next_stage="c"),
            _stage("c", process="pipeline", terminal=True),
        ],
    )

    with pytest.raises(
        ValueError,
        match="Stage 'b' cannot be isolated because process group 'b'",
    ):
        apply_stage_process_overrides(config, isolate_stages=["b"])


def test_process_override_rejects_fused_stage_component() -> None:
    config = _IsolationPipelineConfig(
        model_path="dummy",
        stages=[
            _stage("a", process="pipeline", next_stage="b"),
            _stage("b", process="pipeline", terminal=True),
        ],
        fused_stages=[["a", "b"]],
    )

    with pytest.raises(
        ValueError,
        match="Stage 'b' cannot be isolated because process group",
    ):
        apply_stage_process_overrides(config, isolate_stages=["b"])


def test_moss_tts_local_default_already_isolates_vocoder() -> None:
    from sglang_omni.models.moss_tts_local.config import MossTTSLocalPipelineConfig

    config = MossTTSLocalPipelineConfig(model_path="dummy")
    assert {
        stage.name: stage.runtime.resources.total_gpu_memory_fraction
        for stage in config.stages
        if stage.gpu is not None
    } == {
        "preprocessing": 0.15,
        "tts_engine": 0.67,
        "vocoder": 0.18,
    }
    assert [stage.process for stage in config.stages] == [
        "pipeline",
        "pipeline",
        "vocoder",
    ]
    expected_groups = [
        ("pipeline", ("preprocessing", "tts_engine")),
        ("vocoder", ("vocoder",)),
    ]
    assert [
        (group.name, group.stage_names) for group in _topology(config).groups
    ] == expected_groups

    isolated = apply_stage_process_overrides(config, isolate_stages=["vocoder"])

    assert [stage.process for stage in config.stages] == [
        "pipeline",
        "pipeline",
        "vocoder",
    ]
    assert [
        (group.name, group.stage_names) for group in _topology(isolated).groups
    ] == expected_groups
    assert build_stage_placement_plan(config).gpus[
        0
    ].total_gpu_memory_fraction == pytest.approx(1.0)
    assert {
        stage.name: stage.runtime.resources.total_gpu_memory_fraction
        for stage in isolated.stages
        if stage.gpu is not None
    } == {
        "preprocessing": 0.15,
        "tts_engine": 0.67,
        "vocoder": 0.18,
    }
    assert build_stage_placement_plan(isolated).gpus[
        0
    ].total_gpu_memory_fraction == pytest.approx(1.0)


def test_ming_tts_preserves_default_and_can_isolate_vocoder_role() -> None:
    from sglang_omni.models.ming_tts.config import MingTTSPipelineConfig

    config = MingTTSPipelineConfig(model_path="dummy")
    assert {
        stage.name: stage.runtime.resources.total_gpu_memory_fraction
        for stage in config.stages
        if stage.gpu is not None
    } == {
        "reference_encode": None,
        "tts_engine": None,
        "audio_decode": None,
    }
    assert [stage.process for stage in config.stages] == ["pipeline"] * 4

    isolated = apply_stage_process_overrides(config, isolate_stages=["vocoder"])

    assert [stage.process for stage in config.stages] == ["pipeline"] * 4
    assert [
        (group.name, group.stage_names) for group in _topology(isolated).groups
    ] == [
        (
            "pipeline",
            ("preprocessing", "reference_encode", "tts_engine"),
        ),
        ("audio_decode", ("audio_decode",)),
    ]
    assert build_stage_placement_plan(config).gpus[
        0
    ].total_gpu_memory_fraction == pytest.approx(0.0)
    assert {
        stage.name: stage.runtime.resources.total_gpu_memory_fraction
        for stage in isolated.stages
        if stage.gpu is not None
    } == {
        "reference_encode": 0.08,
        "tts_engine": 0.72,
        "audio_decode": 0.12,
    }
    assert build_stage_placement_plan(isolated).gpus[
        0
    ].total_gpu_memory_fraction == pytest.approx(0.92)


def test_ming_isolation_resources_do_not_change_default_placement_limit() -> None:
    from sglang_omni.config import PlacementConfig
    from sglang_omni.models.ming_tts.config import MingTTSPipelineConfig

    config = MingTTSPipelineConfig(
        model_path="dummy",
        placement=PlacementConfig(max_total_gpu_memory_fraction_per_gpu=0.90),
    )

    _topology(config)

    with pytest.raises(ValueError, match="exceeds placement limit 0.900"):
        apply_stage_process_overrides(config, isolate_stages=["vocoder"])


def test_fishaudio_preserves_default_and_can_isolate_vocoder() -> None:
    from sglang_omni.models.fishaudio_s2_pro.config import S2ProPipelineConfig

    config = S2ProPipelineConfig(model_path="dummy")
    assert [stage.process for stage in config.stages] == [
        "preprocessing",
        "pipeline",
        "pipeline",
    ]

    isolated = apply_stage_process_overrides(config, isolate_stages=["vocoder"])

    assert [stage.process for stage in config.stages] == [
        "preprocessing",
        "pipeline",
        "pipeline",
    ]
    assert [
        (group.name, group.stage_names) for group in _topology(isolated).groups
    ] == [
        ("preprocessing", ("preprocessing",)),
        ("pipeline", ("tts_engine",)),
        ("vocoder", ("vocoder",)),
    ]
    assert build_stage_placement_plan(config).gpus[
        0
    ].total_gpu_memory_fraction == pytest.approx(0.0)
    assert [
        stage.runtime.resources.total_gpu_memory_fraction
        for stage in isolated.stages
        if stage.gpu is not None
    ] == [0.85, 0.10]
    assert build_stage_placement_plan(isolated).gpus[
        0
    ].total_gpu_memory_fraction == pytest.approx(0.95)


def test_process_override_warns_but_preserves_understated_resources(caplog) -> None:
    from sglang_omni.models.fishaudio_s2_pro.config import S2ProPipelineConfig

    config = S2ProPipelineConfig(model_path="dummy")
    stages = {stage.name: stage for stage in config.stages}
    stages["tts_engine"].runtime.resources.total_gpu_memory_fraction = 0.01
    stages["vocoder"].runtime.resources.total_gpu_memory_fraction = 0.15

    isolated = apply_stage_process_overrides(config, isolate_stages=["vocoder"])
    isolated_stages = {stage.name: stage for stage in isolated.stages}

    assert isolated_stages[
        "tts_engine"
    ].runtime.resources.total_gpu_memory_fraction == pytest.approx(0.01)
    assert isolated_stages[
        "vocoder"
    ].runtime.resources.total_gpu_memory_fraction == pytest.approx(0.15)
    assert (
        "Stage 'tts_engine' declares total_gpu_memory_fraction=0.01 below "
        "process-edge resource contract 0.85"
    ) in caplog.text


def test_voxtral_preserves_default_and_can_isolate_vocoder() -> None:
    from sglang_omni.models.voxtral_tts.config import VoxtralTTSPipelineConfig

    config = VoxtralTTSPipelineConfig(model_path="dummy")
    assert [stage.process for stage in config.stages] == ["pipeline"] * 3

    isolated = apply_stage_process_overrides(config, isolate_stages=["vocoder"])

    assert [stage.process for stage in config.stages] == ["pipeline"] * 3
    assert [
        (group.name, group.stage_names) for group in _topology(isolated).groups
    ] == [
        ("pipeline", ("preprocessing", "tts_generation")),
        ("vocoder", ("vocoder",)),
    ]
    assert build_stage_placement_plan(config).gpus[
        0
    ].total_gpu_memory_fraction == pytest.approx(0.0)
    assert [
        stage.runtime.resources.total_gpu_memory_fraction
        for stage in isolated.stages
        if stage.gpu is not None
    ] == [0.85, 0.10]
    assert build_stage_placement_plan(isolated).gpus[
        0
    ].total_gpu_memory_fraction == pytest.approx(0.95)


def test_moss_split_rejects_vocoder_isolation_as_not_process_safe() -> None:
    from sglang_omni.models.moss_tts_local.config import MossTTSLocalSplitPipelineConfig

    config = MossTTSLocalSplitPipelineConfig(model_path="dummy")

    with pytest.raises(
        ValueError,
        match="Process split across 'tts_engine' -> 'vocoder' is not supported",
    ):
        apply_stage_process_overrides(config, isolate_stages=["vocoder"])


@pytest.mark.parametrize(
    ("config_path", "expected_total_fraction"),
    [
        (
            "sglang_omni.models.qwen3_tts.config.Qwen3TTSPipelineConfig",
            0.95,
        ),
        (
            "sglang_omni.models.moss_tts.config.MossTTSPipelineConfig",
            1.0,
        ),
    ],
)
def test_ar_to_vocoder_boundary_isolates_with_declared_contract(
    config_path: str,
    expected_total_fraction: float,
) -> None:
    """Preprocessing is process-local for these models but the vocoder is not."""
    from sglang_omni.utils.imports import import_string

    config = import_string(config_path)(model_path="dummy")

    isolated = apply_stage_process_overrides(config, isolate_stages=["vocoder"])

    assert [stage.process for stage in config.stages] == ["pipeline"] * 3
    assert [
        (group.name, group.stage_names) for group in _topology(isolated).groups
    ] == [
        ("pipeline", ("preprocessing", "tts_engine")),
        ("vocoder", ("vocoder",)),
    ]
    assert build_stage_placement_plan(isolated).gpus[
        0
    ].total_gpu_memory_fraction == pytest.approx(expected_total_fraction)


def test_stage_process_singleton_matches_isolate_stage_resources() -> None:
    from sglang_omni.models.qwen3_tts.config import Qwen3TTSPipelineConfig

    config = Qwen3TTSPipelineConfig(model_path="dummy")

    isolated = apply_stage_process_overrides(
        config,
        isolate_stages=["vocoder"],
    )
    assigned = apply_stage_process_overrides(
        config,
        stage_processes=["vocoder=vocoder"],
    )

    assert assigned.model_dump() == isolated.model_dump()


def test_resource_contract_depends_on_split_edge_not_moved_stage() -> None:
    from sglang_omni.models.fishaudio_s2_pro.config import S2ProPipelineConfig

    config = S2ProPipelineConfig(model_path="dummy")

    overridden = apply_stage_process_overrides(
        config,
        stage_processes=["tts_engine=tts_engine"],
    )

    stages = {stage.name: stage for stage in overridden.stages}
    assert stages[
        "tts_engine"
    ].runtime.resources.total_gpu_memory_fraction == pytest.approx(0.85)
    assert stages[
        "vocoder"
    ].runtime.resources.total_gpu_memory_fraction == pytest.approx(0.10)


def test_qwen_tts_engine_assignment_rejects_unsafe_preprocessing_edge() -> None:
    from sglang_omni.models.qwen3_tts.config import Qwen3TTSPipelineConfig

    config = Qwen3TTSPipelineConfig(model_path="dummy")

    with pytest.raises(
        ValueError,
        match="Process split across 'preprocessing' -> 'tts_engine' is not supported",
    ):
        apply_stage_process_overrides(
            config,
            stage_processes=["tts_engine=tts_engine"],
        )


def test_higgs_can_isolate_vocoder_from_declared_fractions() -> None:
    """The process-safe vocoder boundary can be split without mutating the source."""
    from sglang_omni.models.higgs_tts.config import HiggsTtsPipelineConfig

    config = HiggsTtsPipelineConfig(model_path="dummy")

    isolated = apply_stage_process_overrides(config, isolate_stages=["vocoder"])

    assert [
        (group.name, group.stage_names) for group in _topology(isolated).groups
    ] == [
        ("tts_frontend", ("preprocessing", "audio_encoder")),
        ("pipeline", ("tts_engine",)),
        ("vocoder", ("vocoder",)),
    ]
    assert [stage.process for stage in config.stages] == [
        "tts_frontend",
        "tts_frontend",
        "pipeline",
        "pipeline",
    ]
    assert [
        stage.runtime.resources.total_gpu_memory_fraction for stage in isolated.stages
    ] == [None, 0.03, 0.85, 0.10]


def test_higgs_can_isolate_audio_encoder_from_declared_fractions() -> None:
    """A process-safe stage needs no contract when the config declares fractions."""
    from sglang_omni.models.higgs_tts.config import HiggsTtsPipelineConfig

    config = HiggsTtsPipelineConfig(model_path="dummy")
    assert HiggsTtsPipelineConfig.process_edge_resources() == {}

    isolated = apply_stage_process_overrides(config, isolate_stages=["audio_encoder"])

    assert [
        (group.name, group.stage_names) for group in _topology(isolated).groups
    ] == [
        ("tts_frontend", ("preprocessing",)),
        ("audio_encoder", ("audio_encoder",)),
        ("pipeline", ("tts_engine", "vocoder")),
    ]
    assert build_stage_placement_plan(isolated).gpus[
        0
    ].total_gpu_memory_fraction == pytest.approx(0.98)


def test_fishaudio_preprocessing_isolation_is_an_unchanged_no_op() -> None:
    from sglang_omni.models.fishaudio_s2_pro.config import S2ProPipelineConfig

    config = S2ProPipelineConfig(model_path="dummy")

    isolated = apply_stage_process_overrides(config, isolate_stages=["preprocessing"])

    assert [
        (group.name, group.stage_names) for group in _topology(isolated).groups
    ] == [
        ("preprocessing", ("preprocessing",)),
        ("pipeline", ("tts_engine", "vocoder")),
    ]


@pytest.mark.parametrize(
    ("config_path", "stage_name"),
    [
        ("sglang_omni.models.audar_tts.config.AudarTTSPipelineConfig", "vocoder"),
        ("sglang_omni.models.zonos2.config.Zonos2PipelineConfig", "vocoder"),
    ],
)
def test_process_safe_stage_without_contract_reports_the_missing_fractions(
    config_path: str,
    stage_name: str,
) -> None:
    """Process safety and resource declarations fail with distinct messages."""
    from sglang_omni.utils.imports import import_string

    config = import_string(config_path)(model_path="dummy")

    with pytest.raises(
        ValueError,
        match="shared by multiple process groups without",
    ):
        apply_stage_process_overrides(config, isolate_stages=[stage_name])


def test_process_edge_contract_fraction_is_validated_before_placement() -> None:
    """Contract values must go through the field validator, not raw assignment."""

    class _NegativeContractConfig(_IsolationPipelineConfig):
        @classmethod
        def process_edge_resources(
            cls,
        ) -> dict[tuple[str, str], dict[str, float]]:
            return {("a", "b"): {"a": -0.5, "b": 0.10}}

    config = _NegativeContractConfig(
        model_path="dummy",
        stages=[
            _stage("a", gpu=0, process="pipeline", next_stage="b"),
            _stage("b", gpu=0, process="pipeline", terminal=True),
        ],
    )

    with pytest.raises(ValueError, match="total_gpu_memory_fraction"):
        apply_stage_process_overrides(config, isolate_stages=["b"])


def test_process_edge_contract_rejects_conflicting_stage_recommendations() -> None:
    class _ConflictingContractConfig(_IsolationPipelineConfig):
        @classmethod
        def process_edge_resources(
            cls,
        ) -> dict[tuple[str, str], dict[str, float]]:
            return {
                ("a", "b"): {"b": 0.30},
                ("b", "c"): {"b": 0.40},
            }

    config = _ConflictingContractConfig(
        model_path="dummy",
        stages=[
            _stage("a", process="pipeline", next_stage="b"),
            _stage("b", process="pipeline", next_stage="c"),
            _stage("c", process="pipeline", terminal=True),
        ],
    )

    with pytest.raises(
        ValueError,
        match="Conflicting process-edge resources for stage 'b'",
    ):
        apply_stage_process_overrides(
            config,
            stage_processes=["b=b"],
        )


def test_process_override_does_not_run_placement_policy() -> None:
    """Policy execution belongs to prepare_pipeline_runtime, not the override chain."""
    _PLACEMENT_POLICY_CALLS.clear()
    config = _IsolationPipelineConfig(
        model_path="dummy",
        placement_policy=f"{__name__}._recording_placement_policy",
        stages=[
            _stage("a", gpu=0, fraction=0.50, process="pipeline", next_stage="b"),
            _stage("b", gpu=0, fraction=0.40, process="pipeline", terminal=True),
        ],
    )

    apply_stage_process_overrides(config, isolate_stages=["b"])

    assert _PLACEMENT_POLICY_CALLS == []

    build_stage_placement_plan(config)

    assert _PLACEMENT_POLICY_CALLS == ["dummy"]


@pytest.mark.parametrize(
    ("config_path", "stage_name"),
    [
        (
            "sglang_omni.models.moss_tts_local.config.MossTTSLocalPipelineConfig",
            "preprocessing",
        ),
        (
            "sglang_omni.models.qwen3_tts.config.Qwen3TTSPipelineConfig",
            "preprocessing",
        ),
        (
            "sglang_omni.models.moss_tts.config.MossTTSPipelineConfig",
            "preprocessing",
        ),
    ],
)
def test_process_override_rejects_process_local_handoff_stages(
    config_path: str,
    stage_name: str,
) -> None:
    from sglang_omni.utils.imports import import_string

    config_cls = import_string(config_path)
    config = config_cls(model_path="dummy")

    with pytest.raises(
        ValueError,
        match=f"Process split across '{stage_name}' -> 'tts_engine' is not supported",
    ):
        apply_stage_process_overrides(config, isolate_stages=[stage_name])


def test_higgs_can_isolate_preprocessing_and_audio_encoder_separately() -> None:
    """Singleton isolation of both frontend stages leaves tts_engine alone."""
    from sglang_omni.models.higgs_tts.config import HiggsTtsPipelineConfig

    config = HiggsTtsPipelineConfig(model_path="dummy")

    isolated = apply_stage_process_overrides(
        config,
        isolate_stages=["preprocessing", "audio_encoder"],
    )

    assert [
        (group.name, group.stage_names) for group in _topology(isolated).groups
    ] == [
        ("preprocessing", ("preprocessing",)),
        ("audio_encoder", ("audio_encoder",)),
        ("pipeline", ("tts_engine", "vocoder")),
    ]
    assert build_stage_placement_plan(isolated).gpus[
        0
    ].total_gpu_memory_fraction == pytest.approx(0.98)


def test_higgs_groups_preprocessing_and_audio_encoder_by_default() -> None:
    from sglang_omni.models.higgs_tts.config import HiggsTtsPipelineConfig

    config = HiggsTtsPipelineConfig(model_path="dummy")

    assert [(group.name, group.stage_names) for group in _topology(config).groups] == [
        ("tts_frontend", ("preprocessing", "audio_encoder")),
        ("pipeline", ("tts_engine", "vocoder")),
    ]
    assert build_stage_placement_plan(config).gpus[
        0
    ].total_gpu_memory_fraction == pytest.approx(0.98)


def test_higgs_can_rename_grouped_frontend() -> None:
    from sglang_omni.models.higgs_tts.config import HiggsTtsPipelineConfig

    config = HiggsTtsPipelineConfig(model_path="dummy")

    grouped = apply_stage_process_overrides(
        config,
        stage_processes=[
            "preprocessing=frontend",
            "audio_encoder=frontend",
        ],
    )

    assert [stage.process for stage in config.stages] == [
        "tts_frontend",
        "tts_frontend",
        "pipeline",
        "pipeline",
    ]
    assert [(group.name, group.stage_names) for group in _topology(grouped).groups] == [
        ("frontend", ("preprocessing", "audio_encoder")),
        ("pipeline", ("tts_engine", "vocoder")),
    ]
    assert build_stage_placement_plan(grouped).gpus[
        0
    ].total_gpu_memory_fraction == pytest.approx(0.98)


def test_stage_process_rejects_unsafe_grouped_split() -> None:
    """Grouping is validated by the edge it crosses, not by the stage that moved."""
    from sglang_omni.models.qwen3_tts.config import Qwen3TTSPipelineConfig

    config = Qwen3TTSPipelineConfig(model_path="dummy")

    with pytest.raises(
        ValueError,
        match="Process split across 'preprocessing' -> 'tts_engine' is not supported",
    ):
        apply_stage_process_overrides(
            config,
            stage_processes=["preprocessing=frontend"],
        )


def test_stage_process_conflicts_with_isolate_stage() -> None:
    from sglang_omni.models.higgs_tts.config import HiggsTtsPipelineConfig

    config = HiggsTtsPipelineConfig(model_path="dummy")

    with pytest.raises(
        ValueError,
        match="cannot use both --isolate-stage and --stage-process",
    ):
        apply_stage_process_overrides(
            config,
            isolate_stages=["audio_encoder"],
            stage_processes=["audio_encoder=frontend"],
        )


@pytest.mark.parametrize(
    ("config_path", "assignments", "stage_name"),
    [
        (
            "sglang_omni.models.higgs_tts.config.HiggsTtsPipelineConfig",
            ["preprocessing=frontend", "preprocessing=other"],
            "preprocessing",
        ),
        (
            "sglang_omni.models.ming_tts.config.MingTTSPipelineConfig",
            ["vocoder=g1", "audio_decode=g2"],
            "audio_decode",
        ),
    ],
)
def test_stage_process_rejects_duplicate_resolved_stage(
    config_path: str,
    assignments: list[str],
    stage_name: str,
) -> None:
    from sglang_omni.utils.imports import import_string

    config = import_string(config_path)(model_path="dummy")

    with pytest.raises(
        ValueError,
        match=f"Stage '{stage_name}' has multiple --stage-process assignments",
    ):
        apply_stage_process_overrides(
            config,
            stage_processes=assignments,
        )


@pytest.mark.parametrize("value", ["preprocessing", "preprocessing=", "=frontend", ""])
def test_parse_stage_process_assignment_rejects_malformed_values(value: str) -> None:
    with pytest.raises(ValueError, match="expected STAGE=PROCESS"):
        parse_stage_process_assignment(value)


def test_parse_stage_process_assignment_strips_whitespace() -> None:
    assert parse_stage_process_assignment(" preprocessing = frontend ") == (
        "preprocessing",
        "frontend",
    )


def test_serve_cli_accepts_repeatable_isolate_stage(monkeypatch) -> None:
    import importlib

    from typer.testing import CliRunner

    from sglang_omni.cli import app

    config = _IsolationPipelineConfig(
        model_path="dummy",
        stages=[
            _stage("a", process="pipeline", next_stage="b"),
            _stage("b", process="pipeline", next_stage="c"),
            _stage("c", process="pipeline", terminal=True),
        ],
    )
    launched: list[PipelineConfig] = []
    serve_module = importlib.import_module("sglang_omni.cli.serve")
    monkeypatch.setattr(
        ConfigManager,
        "from_file",
        staticmethod(lambda _path: ConfigManager(config)),
    )
    monkeypatch.setattr(
        serve_module,
        "launch_server",
        lambda pipeline_config, **_kwargs: launched.append(pipeline_config),
    )

    result = CliRunner().invoke(
        app,
        [
            "serve",
            "--config",
            "ignored.yaml",
            "--isolate-stage",
            "b",
            "--isolate-stage",
            "c",
        ],
    )

    assert result.exit_code == 0, result.output
    assert len(launched) == 1
    assert [stage.process for stage in launched[0].stages] == ["pipeline", "b", "c"]


def test_serve_cli_accepts_repeatable_stage_process(monkeypatch) -> None:
    import importlib

    from typer.testing import CliRunner

    from sglang_omni.cli import app

    config = _IsolationPipelineConfig(
        model_path="dummy",
        stages=[
            _stage("a", process="pipeline", next_stage="b"),
            _stage("b", process="pipeline", next_stage="c"),
            _stage("c", process="pipeline", terminal=True),
        ],
    )
    launched: list[PipelineConfig] = []
    serve_module = importlib.import_module("sglang_omni.cli.serve")
    monkeypatch.setattr(
        ConfigManager,
        "from_file",
        staticmethod(lambda _path: ConfigManager(config)),
    )
    monkeypatch.setattr(
        serve_module,
        "launch_server",
        lambda pipeline_config, **_kwargs: launched.append(pipeline_config),
    )

    result = CliRunner().invoke(
        app,
        [
            "serve",
            "--config",
            "ignored.yaml",
            "--stage-process",
            "a=frontend",
            "--stage-process",
            "b=frontend",
        ],
    )

    assert result.exit_code == 0, result.output
    assert len(launched) == 1
    assert [stage.process for stage in launched[0].stages] == [
        "frontend",
        "frontend",
        "pipeline",
    ]


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

    topology = _topology(config)

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

    topology = _topology(config)

    assert topology.tp_stage_to_processes == {"thinker": ("model_tp0", "model_tp1")}


def test_same_process_same_gpu_does_not_require_memory_budgets() -> None:
    config = PipelineConfig(
        model_path="dummy",
        stages=[
            _stage("a", gpu=0, process="p0", next_stage="b"),
            _stage("b", gpu=0, process="p0", terminal=True),
        ],
    )

    topology = _topology(config)

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

    topology = _topology(config)

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
    gpu_placement = build_stage_placement_plan(config)

    with pytest.raises(ValueError, match="total_gpu_memory_fraction"):
        build_process_topology_plan(config, gpu_placement)


def test_same_gpu_multiple_processes_rejects_over_budget() -> None:
    config = PipelineConfig(
        model_path="dummy",
        stages=[
            _stage("a", gpu=0, fraction=0.70, process="p0", next_stage="b"),
            _stage("b", gpu=0, fraction=0.40, process="p1", terminal=True),
        ],
    )

    with pytest.raises(ValueError, match="exceeds placement limit"):
        build_stage_placement_plan(config)


def test_one_process_group_cannot_span_multiple_gpus() -> None:
    config = PipelineConfig(
        model_path="dummy",
        stages=[
            _stage("a", gpu=0, process="p0", next_stage="b"),
            _stage("b", gpu=1, process="p0", terminal=True),
        ],
    )
    gpu_placement = build_stage_placement_plan(config)

    with pytest.raises(ValueError, match="spans multiple GPUs"):
        build_process_topology_plan(config, gpu_placement)


def test_tp_process_names_must_not_collide_with_non_tp_process_group() -> None:
    config = PipelineConfig(
        model_path="dummy",
        stages=[
            _stage("a", process="thinker_tp0", next_stage="thinker"),
            _stage("thinker", gpu=[0, 1], tp_size=2, terminal=True),
        ],
    )
    gpu_placement = build_stage_placement_plan(config)

    with pytest.raises(ValueError, match="collide"):
        build_process_topology_plan(config, gpu_placement)


def test_tp_process_names_must_be_unique_across_tp_stages() -> None:
    config = PipelineConfig(
        model_path="dummy",
        stages=[
            _stage("a", gpu=[0, 1], tp_size=2, process="model", next_stage="b"),
            _stage("b", gpu=[2, 3], tp_size=2, process="model", terminal=True),
        ],
    )
    gpu_placement = build_stage_placement_plan(config)

    with pytest.raises(ValueError, match="Duplicate TP process names"):
        build_process_topology_plan(config, gpu_placement)
