# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os

import pytest
import torch

from sglang_omni.config.manager import ConfigManager
from sglang_omni.config.runtime import (
    apply_typed_stage_kwargs,
    resolve_stage_factory_kwargs,
    resolve_stage_typed_kwargs,
)
from sglang_omni.models.fun_cosyvoice3 import CAPABILITIES
from sglang_omni.models.fun_cosyvoice3.config import (
    FUN_COSYVOICE3_DEFAULT_FLOW_CUDA_GRAPH_CAPTURE_SHAPES,
    FunCosyVoice3PipelineConfig,
)
from sglang_omni.models.fun_cosyvoice3.payload_types import FunCosyVoice3State
from sglang_omni.models.fun_cosyvoice3.stages import create_vocoder_executor
from sglang_omni.models.registry import PIPELINE_CONFIG_REGISTRY
from sglang_omni.pipeline.mp_runner import build_stage_groups
from sglang_omni.pipeline.runtime_config import prepare_pipeline_runtime
from sglang_omni.pipeline.stage_workers import patched_spawn_env
from tests.unit_test.fixtures.pipeline_fakes import FakeMpContext
from tests.unit_test.pipeline.helpers import build_compiled_process_topology


@pytest.mark.parametrize(
    ("env_defaults", "expected_omp_threads"),
    [({}, "1"), ({"OMP_NUM_THREADS": "3"}, "3")],
)
def test_fun_cosyvoice3_engine_process_resolves_spawn_omp_default(
    monkeypatch: pytest.MonkeyPatch,
    env_defaults: dict[str, str],
    expected_omp_threads: str,
) -> None:
    monkeypatch.delenv("OMP_NUM_THREADS", raising=False)
    config = FunCosyVoice3PipelineConfig(
        model_path="model",
        env_defaults=env_defaults,
    )

    prep = prepare_pipeline_runtime(config)
    try:
        groups = build_stage_groups(
            config,
            ctx=FakeMpContext(),
            stages_cfg=prep.stages_cfg,
            endpoints=prep.endpoints,
            placement_plan=prep.placement_plan,
            process_plan=prep.process_plan,
            replica_topology=prep.replica_topology,
        )
        engine_stage_names = {
            stage_cfg.name
            for stage_cfg in prep.stages_cfg
            if type(config).stage_config_cls(stage_cfg.name).engine_stage
        }
        engine_process_specs = [
            process_spec
            for group in groups
            for process_spec in group.process_specs
            if any(
                stage_spec.stage_name in engine_stage_names
                for stage_spec in process_spec.stage_specs
            )
        ]
        assert len(engine_process_specs) == 1

        with patched_spawn_env(engine_process_specs[0]):
            assert os.environ["OMP_NUM_THREADS"] == expected_omp_threads
    finally:
        prep.runtime_dir.close()


@pytest.mark.parametrize(
    "written_omp_setting",
    [
        ("env_defaults.OMP_NUM_THREADS", "3"),
        ("vocoder.env.OMP_NUM_THREADS", "3"),
    ],
)
def test_fun_cosyvoice3_engine_process_spawns_with_a_written_omp_setting(
    monkeypatch: pytest.MonkeyPatch,
    written_omp_setting: tuple[str, str],
) -> None:
    monkeypatch.delenv("OMP_NUM_THREADS", raising=False)
    config = ConfigManager(
        FunCosyVoice3PipelineConfig(model_path="model")
    ).merge_config([written_omp_setting])
    prepared_runtime = prepare_pipeline_runtime(config)
    try:
        stage_groups = build_stage_groups(
            config,
            ctx=FakeMpContext(),
            stages_cfg=prepared_runtime.stages_cfg,
            endpoints=prepared_runtime.endpoints,
            placement_plan=prepared_runtime.placement_plan,
            process_plan=prepared_runtime.process_plan,
            replica_topology=prepared_runtime.replica_topology,
        )
        (engine_process_spec,) = [
            process_spec
            for stage_group in stage_groups
            for process_spec in stage_group.process_specs
            if any(
                stage_spec.stage_name == "tts_engine"
                for stage_spec in process_spec.stage_specs
            )
        ]

        with patched_spawn_env(engine_process_spec):
            assert os.environ["OMP_NUM_THREADS"] == "3"
    finally:
        prepared_runtime.runtime_dir.close()


def test_fun_cosyvoice3_config_and_registry_contract() -> None:
    config = FunCosyVoice3PipelineConfig(model_path="model")

    assert [stage.name for stage in config.stages] == [
        "preprocessing",
        "vocoder",
        "tts_engine",
    ]
    assert [stage.process for stage in config.stages] == [
        "pipeline",
        "pipeline",
        "pipeline",
    ]
    assert config.terminal_stages == ["vocoder"]
    assert config.required_speech_reference_count == 1
    assert config.speech_reference_text_excludes_instructions is True
    assert config.gpu_placement == {"tts_engine": 0, "vocoder": 0}
    assert type(config).stage_config_cls("tts_engine").engine_stage
    assert config.process_local_edges() == frozenset({("preprocessing", "tts_engine")})
    assert CAPABILITIES.supports_streaming_vocoder is True
    stages_by_name = {stage.name: stage for stage in config.stages}
    assert stages_by_name["tts_engine"].stream_to == ["vocoder"]
    assert stages_by_name["vocoder"].can_accept_stream_before_payload is True

    vocoder = next(stage for stage in config.stages if stage.name == "vocoder")
    assert vocoder.factory.dtype == "bfloat16"
    # max_batch_size / max_batch_wait_ms are declared fields on FactoryArgs, so
    # they are validated eagerly rather than passing through as extras.
    assert vocoder.factory.max_batch_size == 16
    assert vocoder.factory.max_batch_wait_ms == 30
    assert vocoder.factory.model_extra == {
        "flow_batch_admission_frames": 8000,
        "flow_merge_max_gap_frames": 384,
        "flow_merge_pad_budget_percent": 25.0,
        "flow_cuda_graph_capture_shapes": FUN_COSYVOICE3_DEFAULT_FLOW_CUDA_GRAPH_CAPTURE_SHAPES,
        "enable_flow_cuda_graph": True,
        "enable_flow_prefix_cuda_graph": True,
        "enable_flow_estimator_trt": False,
        "token_hop_len": 25,
        "token_max_hop_len": 100,
        "disable_hop_growth": False,
        "flow_prefix_cache_gb": 24.0,
    }

    build_compiled_process_topology(config)
    assert (
        PIPELINE_CONFIG_REGISTRY.get_config("FunCosyVoice3SGLangModel")
        is FunCosyVoice3PipelineConfig
    )


def test_fun_cosyvoice3_flow_factory_overrides_use_typed_path() -> None:
    config = FunCosyVoice3PipelineConfig(model_path="model")
    manager = ConfigManager(config)
    merged = manager.merge_config(
        {
            "vocoder.factory.flow_batch_admission_frames": 4000,
            "vocoder.factory.flow_merge_max_gap_frames": 40,
            "vocoder.factory.flow_merge_pad_budget_percent": 3,
            "vocoder.factory.flow_cuda_graph_capture_shapes": [
                [1, 496],
                [5, 544],
                [7, 576],
            ],
        }
    )
    vocoder = next(stage for stage in merged.stages if stage.name == "vocoder")

    assert vocoder.factory.model_extra == {
        "flow_batch_admission_frames": 4000,
        "flow_merge_max_gap_frames": 40,
        "flow_merge_pad_budget_percent": 3,
        "flow_cuda_graph_capture_shapes": [[1, 496], [5, 544], [7, 576]],
        "enable_flow_cuda_graph": True,
        "enable_flow_prefix_cuda_graph": True,
        "enable_flow_estimator_trt": False,
        "token_hop_len": 25,
        "token_max_hop_len": 100,
        "disable_hop_growth": False,
        "flow_prefix_cache_gb": 24.0,
    }
    args = resolve_stage_typed_kwargs(vocoder)
    assert args["flow_batch_admission_frames"] == 4000
    assert args["flow_merge_max_gap_frames"] == 40
    assert args["flow_merge_pad_budget_percent"] == 3
    assert args["flow_cuda_graph_capture_shapes"] == [
        [1, 496],
        [5, 544],
        [7, 576],
    ]


@pytest.mark.parametrize(
    ("overrides", "expected_compile"),
    [
        ({}, True),
        ({"vocoder.factory.enable_flow_estimator_trt": True}, False),
        ({"vocoder.factory.enable_dit_torch_compile": False}, False),
    ],
)
def test_fun_cosyvoice3_dit_compile_default_yields_to_tensorrt(
    overrides: dict[str, bool], expected_compile: bool
) -> None:
    merged = ConfigManager(
        FunCosyVoice3PipelineConfig(model_path="model")
    ).merge_config(overrides)
    vocoder = next(stage for stage in merged.stages if stage.name == "vocoder")

    kwargs = apply_typed_stage_kwargs(
        create_vocoder_executor,
        resolve_stage_factory_kwargs(vocoder, merged),
        resolve_stage_typed_kwargs(vocoder),
        stage_name="vocoder",
    )

    assert kwargs["enable_dit_torch_compile"] is expected_compile


def test_fun_cosyvoice3_rejects_explicit_tensorrt_and_dit_compile() -> None:
    manager = ConfigManager(FunCosyVoice3PipelineConfig(model_path="model"))

    with pytest.raises(ValueError, match="enable only one"):
        manager.merge_config(
            {
                "vocoder.factory.enable_flow_estimator_trt": True,
                "vocoder.factory.enable_dit_torch_compile": True,
            }
        )


def test_fun_cosyvoice3_state_round_trip_preserves_wire_contract() -> None:
    state = FunCosyVoice3State(
        text="hello",
        language="en",
        instructions="speak brightly",
        ref_text="reference",
        stream=True,
        speed=1.25,
        seed=7,
        generation_kwargs={"max_new_tokens": 32},
        flow_embedding=torch.tensor([[1.0, 2.0]]),
        flow_prompt_speech_token=torch.tensor([[10, 11]], dtype=torch.int32),
        flow_prompt_speech_feat=torch.ones(1, 2, 80),
        audio_codes=torch.tensor([[20], [21]], dtype=torch.long),
    )

    wire = state.to_dict()
    restored = FunCosyVoice3State.from_dict(wire)

    assert wire["flow_embedding"] == [[1.0, 2.0]]
    assert restored.text == state.text
    assert restored.language == state.language
    assert restored.instructions == state.instructions
    assert restored.stream is True
    assert restored.speed == 1.25
    assert restored.seed == 7
    assert restored.generation_kwargs == {"max_new_tokens": 32}
    assert restored.flow_prompt_speech_token == [[10, 11]]
    torch.testing.assert_close(
        torch.as_tensor(restored.flow_prompt_speech_feat),
        torch.ones(1, 2, 80),
        rtol=0,
        atol=0,
    )
    assert restored.audio_codes == [[20], [21]]
