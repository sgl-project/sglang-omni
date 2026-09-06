# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest

import sglang_omni.models.fun_asr.engine_builder as fun_asr_builder
import sglang_omni.models.fun_asr.stages as fun_asr_stages
import sglang_omni.platforms as platforms
import sglang_omni.scheduling.bootstrap as bootstrap
import sglang_omni.scheduling.engine_factory as engine_factory
import sglang_omni.scheduling.omni_scheduler as omni_scheduler
import sglang_omni.scheduling.sglang_backend as sglang_backend
from sglang_omni.models.fun_asr import request_builders
from sglang_omni.models.fun_asr.config import FunASRPipelineConfig
from sglang_omni.models.registry import PIPELINE_CONFIG_REGISTRY


@pytest.fixture(autouse=True)
def _select_non_mlx_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    import sglang.srt.utils.tensor_bridge as tensor_bridge

    # Backend-specific tests (e.g. _apple_builder) opt into MLX explicitly.
    # Keep CUDA/ROCm/Torch MPS profile tests independent of the caller's
    # SGLANG_USE_MLX environment.
    monkeypatch.setattr(tensor_bridge, "use_mlx", lambda: False)


def test_fun_asr_config_uses_batched_stage_with_64_running_requests() -> None:
    config = FunASRPipelineConfig(model_path="FunAudioLLM/Fun-ASR-Nano-2512-hf")

    assert config.entry_stage == "asr"
    assert [stage.name for stage in config.stages] == ["asr"]
    assert config.terminal_stages == ["asr"]
    assert config.gpu_placement == {"asr": 0}
    assert config.stages[0].factory_path.endswith("create_sglang_fun_asr_executor")
    # Constructor defaults live on the factory signature (pinned below); the
    # config declares no overrides of its own.
    assert config.stages[0].factory.model_dump(exclude_none=True) == {}
    assert not config.stages[0].factory.model_extra
    assert type(config).stage_config_cls("asr").engine_stage
    assert config.stage_factory_kwargs("asr") == {"enable_encoder_cuda_graph": True}
    assert (
        PIPELINE_CONFIG_REGISTRY.get_config("FunAsrNanoForConditionalGeneration")
        is FunASRPipelineConfig
    )


def test_fun_asr_stage_default_allows_64_running_requests() -> None:
    signature = inspect.signature(fun_asr_stages.create_sglang_fun_asr_executor)

    assert signature.parameters["max_running_requests"].default == 64
    assert signature.parameters["max_new_tokens"].default == 200
    assert signature.parameters["enable_pre_lm_encoder"].default is True
    assert signature.parameters["pre_lm_cache_max_entries"].default == 4096
    assert signature.parameters["pre_lm_cache_size_bytes"].default == 2 * 1024**3
    assert signature.parameters["pre_lm_max_batch_size"].default == 8
    assert signature.parameters["pre_lm_max_batch_wait_ms"].default == 10
    assert signature.parameters["request_build_max_workers"].default == 8
    assert signature.parameters["request_build_max_pending"].default == 32
    assert signature.parameters["stream_emit_interval_s"].default == 0.05


def test_fun_asr_stage_defaults_enable_pending_build_aware_coalescing() -> None:
    signature = inspect.signature(fun_asr_stages.create_sglang_fun_asr_executor)

    assert signature.parameters["prefill_coalesce_requests"].default == 16
    assert signature.parameters["prefill_coalesce_wait_ms"].default == 24.0
    assert signature.parameters["prefill_coalesce_when_idle"].default is True
    assert (
        signature.parameters["prefill_coalesce_requires_pending_builds"].default is True
    )
    assert (
        signature.parameters["prefill_coalesce_after_builds_during_decode"].default
        is True
    )
    assert signature.parameters["enable_async_decode"].default is True
    assert signature.parameters["async_decode_min_batch_size"].default == 2


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


def test_fun_asr_threads_generation_batch_and_request_build_policy(monkeypatch) -> None:
    from sglang_omni.scheduling.generation_batch_policy import (
        build_default_prefill_cuda_graph_bs,
    )

    build_kwargs: dict[str, object] = {}
    infra_kwargs: list[dict[str, object]] = []
    validations: list[dict[str, object]] = []
    stream_builder_calls: list[dict[str, object]] = []
    stream_output_builder = object()

    # Exercise the existing CUDA policy regardless of the host platform.
    monkeypatch.setattr(
        "sglang_omni.utils.device.resolve_device_spec", lambda *args: "cuda:0"
    )

    def tokenizer(text, add_special_tokens=False):
        return SimpleNamespace(input_ids=[0] * len(text))

    adapter_kwargs: dict[str, object] = {}

    monkeypatch.setattr(
        platforms.current_platform, "get_device", lambda index: "cpu", raising=False
    )

    monkeypatch.setattr(
        fun_asr_builder.AutoTokenizer,
        "from_pretrained",
        lambda *args, **kwargs: tokenizer,
    )
    monkeypatch.setattr(
        fun_asr_builder.AutoFeatureExtractor,
        "from_pretrained",
        lambda *args, **kwargs: SimpleNamespace(nb_max_frames=500),
    )
    monkeypatch.setattr(
        fun_asr_builder,
        "get_visible_gpu_sm_version",
        lambda gpu_id: None,
    )
    monkeypatch.setattr(fun_asr_builder, "init_mm_embedding_cache", lambda size: None)
    monkeypatch.setattr(
        request_builders,
        "make_fun_asr_scheduler_adapters",
        lambda **kwargs: (adapter_kwargs.update(kwargs) or object(), object()),
    )
    monkeypatch.setattr(
        request_builders,
        "make_fun_asr_stream_output_builder",
        lambda **kwargs: stream_builder_calls.append(kwargs) or stream_output_builder,
    )
    monkeypatch.setattr(
        sglang_backend,
        "SGLangOutputProcessor",
        lambda **kwargs: object(),
    )
    monkeypatch.setattr(
        omni_scheduler,
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
        fun_asr_builder,
        "build_cache_namespace",
        lambda *args, **kwargs: "test-namespace",
    )

    def _make_encoder_service(*args, **kwargs):
        service = _EncoderService()
        encoder_services.append(service)
        return service

    monkeypatch.setattr(
        fun_asr_builder,
        "FunASRPreLMEncoderService",
        _make_encoder_service,
    )

    def _fake_server_args_builder(model_path, context_length, **overrides):
        build_kwargs.clear()
        build_kwargs.update(overrides)
        prefill_bs = overrides.get("cuda_graph_bs_prefill")
        server_args = SimpleNamespace(**overrides)
        server_args.mm_attention_backend = None
        server_args.cuda_graph_config = SimpleNamespace(
            prefill=SimpleNamespace(
                backend=overrides.get("cuda_graph_backend_prefill", "disabled"),
                bs=prefill_bs,
                max_bs=overrides.get("cuda_graph_max_bs_prefill"),
            )
        )
        server_args._cuda_graph_config_locked = {
            ("prefill", "backend"),
            ("prefill", "bs"),
        }
        return server_args

    model_worker = SimpleNamespace(
        gpu_id=0,
        model_runner=SimpleNamespace(model=object()),
    )
    infrastructure = (
        model_worker,
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
        lambda *args, **kwargs: (
            infra_kwargs.append(dict(kwargs)) or False,
            infrastructure,
        ),
    )
    monkeypatch.setattr(
        engine_factory,
        "validate_generation_batch_policy",
        lambda **kwargs: validations.append(kwargs),
    )

    scheduler = fun_asr_stages.create_sglang_fun_asr_executor("dummy")

    assert build_kwargs["cuda_graph_max_bs"] == 64
    assert build_kwargs["cuda_graph_bs"] == [
        1,
        2,
        4,
        8,
        12,
        16,
        24,
        32,
        40,
        48,
        56,
        64,
    ]
    assert build_kwargs["cuda_graph_backend_prefill"] == "breakable"
    assert build_kwargs["cuda_graph_bs_prefill"] == build_default_prefill_cuda_graph_bs(
        256
    )
    assert scheduler.server_args._cuda_graph_config_locked == {("prefill", "bs")}
    assert infra_kwargs[-1]["enable_prefill_input_embeds"] is True
    assert validations == [
        {"model_name": "Fun-ASR", "server_args": scheduler.server_args}
    ]
    assert adapter_kwargs["audio_encoder_service"] is encoder_services[0]
    assert scheduler.request_build_max_workers == 8
    assert scheduler.request_build_max_pending == 32
    assert stream_builder_calls == [
        {"tokenizer": tokenizer, "min_emit_interval_s": 0.05}
    ]
    assert scheduler.stream_output_builder is stream_output_builder
    assert scheduler.enable_async_decode is True
    assert scheduler.async_decode_min_batch_size == 2
    assert scheduler.prefill_coalesce_requests == 16
    assert scheduler.prefill_coalesce_wait_ms == 24.0
    assert scheduler.prefill_coalesce_when_idle is True
    assert scheduler.prefill_coalesce_requires_pending_builds is True
    assert scheduler.prefill_coalesce_after_builds_during_decode is True
    scheduler.shutdown_callback()
    assert encoder_services[0].close_calls == 1

    scheduler_without_service = fun_asr_stages.create_sglang_fun_asr_executor(
        "dummy", enable_pre_lm_encoder=False
    )
    assert scheduler_without_service.shutdown_callback is None

    scheduler_graph_disabled = fun_asr_stages.create_sglang_fun_asr_executor(
        "dummy", server_args_overrides={"disable_cuda_graph": True}
    )
    assert build_kwargs["cuda_graph_backend_prefill"] == "disabled"
    assert "cuda_graph_bs_prefill" not in build_kwargs
    assert "enable_prefill_input_embeds" not in infra_kwargs[-1]
    scheduler_graph_disabled.shutdown_callback()

    monkeypatch.setattr(
        omni_scheduler,
        "OmniScheduler",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("factory failed")),
    )

    with pytest.raises(RuntimeError, match="factory failed"):
        fun_asr_stages.create_sglang_fun_asr_executor("dummy")

    assert encoder_services[2].close_calls == 1


def test_fun_asr_declares_breakable_prefill_cuda_graph_support() -> None:
    assert fun_asr_builder.FunASREngineBuilder.supports_breakable_prefill_cuda_graph


def _apple_builder(monkeypatch, *, mlx):
    from sglang.srt.utils import tensor_bridge

    monkeypatch.setattr(tensor_bridge, "use_mlx", lambda: mlx)
    monkeypatch.setattr(fun_asr_builder.current_platform, "is_mps", lambda: True)
    captured = []
    monkeypatch.setattr(
        fun_asr_builder.FunASREngineBuilder,
        "build",
        lambda self, *args, **kwargs: captured.append(self),
    )
    fun_asr_stages.create_sglang_fun_asr_executor("dummy")
    builder = captured[0]
    builder.device = "mps"
    builder.context_length = 300
    return builder


@pytest.mark.parametrize("mlx", [False, True])
def test_apple_profile_skips_cuda_resources_and_uses_greedy_requests(monkeypatch, mlx):
    builder = _apple_builder(monkeypatch, mlx=mlx)
    defaults = builder.generation_defaults(dtype="bfloat16")
    assert defaults["max_running_requests"] == 1
    assert defaults["disable_cuda_graph"]
    assert defaults["disable_radix_cache"]
    assert defaults["chunked_prefill_size"] == -1
    overrides = {"enable_torch_compile": True}
    builder.adjust_overrides(overrides)
    assert overrides["enable_torch_compile"] is False
    builder.enable_encoder_cuda_graph = True
    builder.enable_encoder_torch_compile = True
    # Dummy models cannot compile or run an encoder; neither Apple path should try.
    builder.setup_model_resources(
        object(), object(), generation_cuda_graph_enabled=False
    )
    builder.setup_runtime_resources(object(), object())
    assert builder.audio_encoder_service is None
    options = builder.extra_scheduler_kwargs()
    assert options["enable_async_decode"] is False
    assert options["prefill_coalesce_requests"] == 0
    assert options["request_build_max_workers"] == 1
    monkeypatch.setattr(
        request_builders, "make_fun_asr_scheduler_adapters", lambda **kwargs: kwargs
    )
    assert builder.make_adapters(object())["greedy_only"]


@pytest.mark.parametrize("mlx", [False, True])
@pytest.mark.parametrize(
    "override,match",
    [
        ({"max_running_requests": 2}, "max_running_requests=1"),
        ({"disable_radix_cache": False}, "disabled radix cache"),
        ({"chunked_prefill_size": 128}, "chunked prefill"),
        ({"mlx_enable_sampling": True}, "mlx_enable_sampling=False"),
        ({"quantization": "mlx_q4"}, "unquantized"),
    ],
)
def test_apple_rejects_unsupported_runtime_options(monkeypatch, mlx, override, match):
    builder = _apple_builder(monkeypatch, mlx=mlx)
    args = {
        **builder.generation_defaults(dtype="bfloat16"),
        "mlx_enable_sampling": False,
        "quantization": None,
        **override,
    }
    with pytest.raises(ValueError, match=match):
        builder.validate_before_infrastructure(SimpleNamespace(**args))
