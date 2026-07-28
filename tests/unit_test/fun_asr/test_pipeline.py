# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import inspect
from pathlib import Path
from types import SimpleNamespace

import pytest

import sglang_omni.models.fun_asr.stages as fun_asr_stages
from sglang_omni.config.manager import ConfigManager
from sglang_omni.config.runtime import resolve_stage_static_factory_args
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
    assert config.stages[0].factory_args["enable_pre_lm_encoder"] is True
    assert config.stages[0].factory_args["pre_lm_cache_max_entries"] == 4096
    assert config.stages[0].factory_args["pre_lm_cache_size_bytes"] == 2 * 1024**3
    assert config.stages[0].factory_args["request_build_max_workers"] == 8
    assert config.stages[0].factory_args["request_build_max_pending"] == 16
    assert FunASRPipelineConfig.mem_fraction_role_to_stage() == {"asr": "asr"}
    assert FunASRPipelineConfig.generation_sglang_role_to_stage() == {
        "generation": "asr"
    }
    assert (
        PIPELINE_CONFIG_REGISTRY.get_config("FunAsrNanoForConditionalGeneration")
        is FunASRPipelineConfig
    )


def test_fun_asr_stage_default_allows_32_running_requests() -> None:
    signature = inspect.signature(fun_asr_stages.create_sglang_fun_asr_executor)

    assert signature.parameters["max_running_requests"].default == 32
    assert signature.parameters["max_new_tokens"].default == 200
    assert signature.parameters["enable_pre_lm_encoder"].default is True
    assert signature.parameters["pre_lm_cache_max_entries"].default == 4096
    assert signature.parameters["pre_lm_cache_size_bytes"].default == 2 * 1024**3
    assert signature.parameters["pre_lm_max_batch_size"].default == 8
    assert signature.parameters["pre_lm_max_batch_wait_ms"].default == 4
    assert signature.parameters["request_build_max_workers"].default == 8
    assert signature.parameters["request_build_max_pending"].default == 16


@pytest.mark.parametrize(
    ("batch_size", "wait_ms", "match"),
    [
        (0, 4, "pre_lm_max_batch_size"),
        (-1, 4, "pre_lm_max_batch_size"),
        (8, -1, "pre_lm_max_batch_wait_ms"),
    ],
)
def test_fun_asr_stage_rejects_invalid_pre_lm_batch_knobs(
    batch_size: int, wait_ms: int, match: str
) -> None:
    # Validation runs before any model/tokenizer load.
    with pytest.raises(ValueError, match=match):
        fun_asr_stages.create_sglang_fun_asr_executor(
            "dummy",
            pre_lm_max_batch_size=batch_size,
            pre_lm_max_batch_wait_ms=wait_ms,
        )


def test_fun_asr_stage_default_uses_auto_static_kv_budget() -> None:
    signature = inspect.signature(fun_asr_stages.create_sglang_fun_asr_executor)

    assert signature.parameters["mem_fraction_static"].default is None


def test_fun_asr_stage_default_disables_multimodal_embedding_cache() -> None:
    signature = inspect.signature(fun_asr_stages.create_sglang_fun_asr_executor)

    assert signature.parameters["mm_embedding_cache_size_bytes"].default == 0


def test_fun_asr_stage_default_disables_torch_compile() -> None:
    signature = inspect.signature(fun_asr_stages.create_sglang_fun_asr_executor)

    assert signature.parameters["enable_torch_compile"].default is False


def test_fun_asr_stage_default_enables_async_decode() -> None:
    signature = inspect.signature(fun_asr_stages.create_sglang_fun_asr_executor)

    assert signature.parameters["enable_async_decode"].default is True
    assert signature.parameters["async_decode_min_batch_size"].default == 2


def test_fun_asr_rtx4090_profile_is_bf16_and_bounded() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    config = ConfigManager.from_file(
        str(repo_root / "examples/configs/fun_asr_rtx4090.yaml")
    ).config
    stage = config.stages[0]

    factory_args = resolve_stage_static_factory_args(stage, config)

    assert factory_args["dtype"] == "bfloat16"
    assert factory_args["max_running_requests"] == 16
    assert factory_args["enable_torch_compile"] is False
    assert factory_args["mm_attention_backend"] == "triton_attn"
    assert factory_args["server_args_overrides"]["mem_fraction_static"] == 0.65


def test_fun_asr_threads_generation_batch_and_request_build_policy(monkeypatch) -> None:
    build_kwargs: dict[str, object] = {}
    validations: list[dict[str, object]] = []
    adapter_kwargs: dict[str, object] = {}
    fake_tokenizer = lambda text, add_special_tokens=False: SimpleNamespace(
        input_ids=[0] * len(text)
    )

    monkeypatch.setattr(
        fun_asr_stages.AutoTokenizer,
        "from_pretrained",
        lambda *args, **kwargs: fake_tokenizer,
    )
    monkeypatch.setattr(
        fun_asr_stages.AutoFeatureExtractor,
        "from_pretrained",
        lambda *args, **kwargs: SimpleNamespace(nb_max_frames=500),
    )
    monkeypatch.setattr(
        fun_asr_stages,
        "get_visible_gpu_sm_version",
        lambda gpu_id: 89,
    )
    monkeypatch.setattr(
        fun_asr_stages,
        "get_process_gpu_memory_bytes",
        lambda gpu_id: 0,
    )
    monkeypatch.setattr(fun_asr_stages, "init_mm_embedding_cache", lambda size: None)
    monkeypatch.setattr(
        fun_asr_stages,
        "make_fun_asr_scheduler_adapters",
        lambda **kwargs: (adapter_kwargs.update(kwargs) or object(), object()),
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
    encoder_services = []

    class _EncoderService:
        def __init__(self) -> None:
            self.close_calls = 0

        def close(self) -> None:
            self.close_calls += 1

    monkeypatch.setattr(
        fun_asr_stages,
        "build_cache_namespace",
        lambda *args, **kwargs: "test-namespace",
    )

    def _make_encoder_service(*args, **kwargs):
        service = _EncoderService()
        encoder_services.append(service)
        return service

    monkeypatch.setattr(
        fun_asr_stages,
        "FunASRPreLMEncoderService",
        _make_encoder_service,
    )

    def _fake_server_args_builder(model_path, context_length, **overrides):
        build_kwargs.update(overrides)
        build_kwargs["context_length"] = context_length
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
    assert build_kwargs["mm_attention_backend"] == "triton_attn"
    max_prompt_overhead = max(
        fun_asr_stages.fun_asr_prompt_overhead_tokens(
            fake_tokenizer,
            language=language,
            itn=itn,
        )
        for language in (None, "英文")
        for itn in (True, False)
    )
    assert build_kwargs["context_length"] == (
        fun_asr_stages.fun_asr_low_frame_rate_length(500) + 200 + max_prompt_overhead
    )
    assert validations == [
        {"model_name": "Fun-ASR", "server_args": scheduler.server_args}
    ]
    assert adapter_kwargs["audio_encoder_service"] is encoder_services[0]
    assert scheduler.request_build_max_workers == 8
    assert scheduler.request_build_max_pending == 16
    assert scheduler.enable_async_decode is True
    assert scheduler.async_decode_min_batch_size == 2
    scheduler.shutdown_callback()
    assert encoder_services[0].close_calls == 1

    monkeypatch.setattr(
        fun_asr_stages,
        "OmniScheduler",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("factory failed")),
    )

    with pytest.raises(RuntimeError, match="factory failed"):
        fun_asr_stages.create_sglang_fun_asr_executor("dummy")

    assert encoder_services[1].close_calls == 1
