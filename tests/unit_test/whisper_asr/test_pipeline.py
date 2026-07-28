# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import inspect
import sys
from pathlib import Path
from types import SimpleNamespace

import sglang_omni.model_runner.base as model_runner_base
import sglang_omni.models.whisper_asr.stages as whisper_asr_stages
import sglang_omni.scheduling.bootstrap as bootstrap
import sglang_omni.scheduling.omni_scheduler as omni_scheduler
import sglang_omni.scheduling.sglang_backend as sglang_backend
from sglang_omni.config.manager import ConfigManager
from sglang_omni.config.runtime import resolve_stage_static_factory_args
from sglang_omni.models.registry import PIPELINE_CONFIG_REGISTRY
from sglang_omni.models.whisper_asr import request_builders as whisper_request_builders
from sglang_omni.models.whisper_asr.config import WhisperASRPipelineConfig


def test_whisper_asr_config_uses_single_batched_stage() -> None:
    config = WhisperASRPipelineConfig(model_path="openai/whisper-large-v3")

    assert config.entry_stage == "asr"
    assert [stage.name for stage in config.stages] == ["asr"]
    assert config.terminal_stages == ["asr"]
    assert config.gpu_placement == {"asr": 0}
    assert config.stages[0].factory.endswith("create_sglang_whisper_asr_executor")
    assert config.stages[0].factory_args["device"] == "cuda:0"
    assert WhisperASRPipelineConfig.mem_fraction_role_to_stage() == {"asr": "asr"}
    assert WhisperASRPipelineConfig.generation_sglang_role_to_stage() == {
        "generation": "asr"
    }
    assert (
        PIPELINE_CONFIG_REGISTRY.get_config("WhisperForConditionalGeneration")
        is WhisperASRPipelineConfig
    )


def test_whisper_asr_compile_default_is_configurable() -> None:
    signature = inspect.signature(whisper_asr_stages.create_sglang_whisper_asr_executor)

    assert signature.parameters["enable_torch_compile"].default is True


def test_whisper_asr_rtx4090_profile_is_bf16_and_bounded() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    config = ConfigManager.from_file(
        str(repo_root / "examples/configs/whisper_asr_rtx4090.yaml")
    ).config
    stage = config.stages[0]

    factory_args = resolve_stage_static_factory_args(stage, config)

    assert factory_args["dtype"] == "bfloat16"
    assert factory_args["max_running_requests"] == 16
    assert factory_args["enable_torch_compile"] is False
    assert factory_args["server_args_overrides"]["mem_fraction_static"] == 0.65


def test_whisper_asr_threads_explicit_cuda_graph_bs(monkeypatch) -> None:
    build_kwargs: dict[str, object] = {}
    fake_processor = SimpleNamespace(
        tokenizer=object(),
        feature_extractor=SimpleNamespace(nb_max_frames=3000),
    )
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(
            AutoProcessor=SimpleNamespace(
                from_pretrained=lambda *args, **kwargs: fake_processor
            ),
            GenerationConfig=SimpleNamespace(
                from_pretrained=lambda *args, **kwargs: object()
            ),
        ),
    )
    monkeypatch.setattr(
        whisper_request_builders,
        "make_whisper_scheduler_adapters",
        lambda **kwargs: (object(), object()),
    )
    monkeypatch.setattr(
        whisper_asr_stages,
        "get_visible_gpu_sm_version",
        lambda gpu_id: 89,
    )
    monkeypatch.setattr(
        whisper_asr_stages,
        "get_process_gpu_memory_bytes",
        lambda gpu_id: 0,
    )
    monkeypatch.setattr(
        model_runner_base,
        "ModelRunner",
        lambda *args, **kwargs: object(),
    )
    monkeypatch.setattr(
        sglang_backend,
        "SGLangOutputProcessor",
        lambda **kwargs: object(),
    )
    monkeypatch.setattr(
        omni_scheduler,
        "OmniScheduler",
        lambda **kwargs: SimpleNamespace(**kwargs),
    )

    def _fake_server_args_builder(model_path, context_length, **overrides):
        build_kwargs.update(overrides)
        return SimpleNamespace(**overrides)

    def _fake_create_infrastructure(server_args, gpu_id, **kwargs):
        model_worker = SimpleNamespace(model_runner=SimpleNamespace(model=object()))
        return False, (
            model_worker,
            object(),
            object(),
            object(),
            object(),
            object(),
            object(),
        )

    monkeypatch.setattr(
        sglang_backend,
        "build_sglang_server_args",
        _fake_server_args_builder,
    )
    monkeypatch.setattr(
        bootstrap,
        "create_sglang_infrastructure_defer_cuda_graph",
        _fake_create_infrastructure,
    )

    whisper_asr_stages.create_sglang_whisper_asr_executor("dummy")

    assert build_kwargs["cuda_graph_max_bs"] == 16
    assert build_kwargs["cuda_graph_bs"] == [1, 2, 4, 8, 12, 16]
    assert build_kwargs["enable_torch_compile"] is True
