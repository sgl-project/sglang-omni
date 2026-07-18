# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import inspect
from types import SimpleNamespace

import sglang_omni.models.fun_asr.stages as fun_asr_stages
from sglang_omni.models.fun_asr.config import FunASRPipelineConfig
from sglang_omni.models.registry import PIPELINE_CONFIG_REGISTRY


def test_fun_asr_config_uses_batched_stage_with_32_running_requests() -> None:
    config = FunASRPipelineConfig(model_path="FunAudioLLM/Fun-ASR-Nano-2512-hf")

    assert config.entry_stage == "asr"
    assert [stage.name for stage in config.stages] == ["asr"]
    assert config.terminal_stages == ["asr"]
    assert config.gpu_placement == {"asr": 0}
    assert config.stages[0].factory.endswith("create_sglang_fun_asr_executor")
    assert config.stages[0].factory_args["device"] == "cuda:0"
    assert config.stages[0].factory_args["max_running_requests"] == 32
    assert config.stages[0].factory_args["max_new_tokens"] == 200
    assert config.stages[0].factory_args["request_build_max_workers"] == 2
    assert config.stages[0].factory_args["request_build_max_pending"] == 16
    assert (
        PIPELINE_CONFIG_REGISTRY.get_config("FunAsrNanoForConditionalGeneration")
        is FunASRPipelineConfig
    )


def test_fun_asr_stage_default_allows_32_running_requests() -> None:
    signature = inspect.signature(fun_asr_stages.create_sglang_fun_asr_executor)

    assert signature.parameters["max_running_requests"].default == 32
    assert signature.parameters["max_new_tokens"].default == 200
    assert signature.parameters["request_build_max_workers"].default == 2
    assert signature.parameters["request_build_max_pending"].default == 16


def test_fun_asr_stage_default_uses_auto_static_kv_budget() -> None:
    signature = inspect.signature(fun_asr_stages.create_sglang_fun_asr_executor)

    assert signature.parameters["mem_fraction_static"].default is None


def test_fun_asr_stage_default_disables_multimodal_embedding_cache() -> None:
    signature = inspect.signature(fun_asr_stages.create_sglang_fun_asr_executor)

    assert signature.parameters["mm_embedding_cache_size_bytes"].default == 0


def test_fun_asr_stage_default_disables_torch_compile() -> None:
    signature = inspect.signature(fun_asr_stages.create_sglang_fun_asr_executor)

    assert signature.parameters["enable_torch_compile"].default is False


def test_fun_asr_threads_generation_batch_and_request_build_policy(monkeypatch) -> None:
    build_kwargs: dict[str, object] = {}
    validations: list[dict[str, object]] = []

    monkeypatch.setattr(
        fun_asr_stages.AutoTokenizer,
        "from_pretrained",
        lambda *args, **kwargs: lambda text, add_special_tokens=False: SimpleNamespace(
            input_ids=[0] * len(text)
        ),
    )
    monkeypatch.setattr(
        fun_asr_stages.AutoFeatureExtractor,
        "from_pretrained",
        lambda *args, **kwargs: SimpleNamespace(nb_max_frames=500),
    )
    monkeypatch.setattr(
        fun_asr_stages,
        "get_visible_gpu_sm_version",
        lambda gpu_id: None,
    )
    monkeypatch.setattr(fun_asr_stages, "init_mm_embedding_cache", lambda size: None)
    monkeypatch.setattr(
        fun_asr_stages,
        "make_fun_asr_scheduler_adapters",
        lambda **kwargs: (object(), object()),
    )
    monkeypatch.setattr(fun_asr_stages, "ModelRunner", lambda *args, **kwargs: object())
    monkeypatch.setattr(
        fun_asr_stages,
        "SGLangOutputProcessor",
        lambda **kwargs: object(),
    )
    monkeypatch.setattr(
        fun_asr_stages,
        "OmniScheduler",
        SimpleNamespace,
    )

    def _fake_server_args_builder(model_path, context_length, **overrides):
        build_kwargs.update(overrides)
        return SimpleNamespace(**overrides)

    model_worker = SimpleNamespace(model_runner=SimpleNamespace(model=object()))
    infrastructure = (
        model_worker,
        object(),
        object(),
        object(),
        object(),
        object(),
        object(),
    )

    monkeypatch.setattr(
        fun_asr_stages,
        "build_sglang_server_args",
        _fake_server_args_builder,
    )
    monkeypatch.setattr(
        fun_asr_stages,
        "create_sglang_infrastructure_defer_cuda_graph",
        lambda *args, **kwargs: (False, infrastructure),
        raising=False,
    )
    monkeypatch.setattr(
        fun_asr_stages,
        "create_sglang_infrastructure",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("legacy bootstrap must not be used")
        ),
        raising=False,
    )
    monkeypatch.setattr(
        fun_asr_stages,
        "validate_generation_batch_policy",
        lambda **kwargs: validations.append(kwargs),
        raising=False,
    )

    scheduler = fun_asr_stages.create_sglang_fun_asr_executor("dummy")

    assert build_kwargs["cuda_graph_max_bs"] == 32
    assert build_kwargs["cuda_graph_bs"] == [1, 2, 4, 8, 12, 16, 24, 32]
    assert validations == [
        {"model_name": "Fun-ASR", "server_args": scheduler.server_args}
    ]
    assert scheduler.request_build_max_workers == 2
    assert scheduler.request_build_max_pending == 16
