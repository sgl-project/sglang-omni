# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib
import inspect
from types import SimpleNamespace
from typing import Any

import pytest

import sglang_omni.platforms as platforms
from tests.unit_test.fakes import FakeServerArgs

TEST_MAX_TOTAL_TOKENS = 82000


def test_engine_builder_import_is_cpu_only() -> None:
    from sglang_omni.scheduling.engine_factory import (
        AsrEngineBuilder,
        SGLangGenerationEngineBuilder,
        TtsEngineBuilder,
    )

    assert SGLangGenerationEngineBuilder.__name__ == "SGLangGenerationEngineBuilder"
    assert AsrEngineBuilder.__name__ == "AsrEngineBuilder"
    assert TtsEngineBuilder.__name__ == "TtsEngineBuilder"
    assert issubclass(AsrEngineBuilder, SGLangGenerationEngineBuilder)
    assert issubclass(TtsEngineBuilder, SGLangGenerationEngineBuilder)


def test_asr_engine_builder_preserves_model_path() -> None:
    from sglang_omni.scheduling.engine_factory import AsrEngineBuilder

    assert AsrEngineBuilder.resolve_checkpoint(object(), "repo/id") == "repo/id"


def test_legacy_tts_engine_factory_paths_remain_importable() -> None:
    module_names = (
        "sglang_omni.models.moss_tts.stages",
        "sglang_omni.models.moss_tts_local.stages",
        "sglang_omni.models.qwen3_tts.stages",
    )

    for module_name in module_names:
        module = importlib.import_module(module_name)
        assert (
            module.create_tts_engine_executor
            is module.create_sglang_tts_engine_executor
        )


def test_tts_engine_builder_uses_shared_checkpoint_resolver(monkeypatch) -> None:
    from sglang_omni.scheduling import engine_factory

    calls: list[str] = []

    def fake_resolve_checkpoint(model_path: str) -> str:
        calls.append(model_path)
        return "/resolved/checkpoint"

    monkeypatch.setattr(engine_factory, "_resolve_checkpoint", fake_resolve_checkpoint)

    resolved = engine_factory.TtsEngineBuilder.resolve_checkpoint(object(), "repo/id")

    assert resolved == "/resolved/checkpoint"
    assert calls == ["repo/id"]


def test_tts_engine_builder_hook_contract_is_narrow() -> None:
    from sglang_omni.scheduling.engine_factory import TtsEngineBuilder

    build_signature = inspect.signature(TtsEngineBuilder.build)
    assert not any(
        param.kind is inspect.Parameter.VAR_KEYWORD
        for param in build_signature.parameters.values()
    )

    generation_defaults_signature = inspect.signature(
        TtsEngineBuilder.generation_defaults
    )
    assert list(generation_defaults_signature.parameters) == ["self", "dtype"]

    adjust_overrides_signature = inspect.signature(TtsEngineBuilder.adjust_overrides)
    assert list(adjust_overrides_signature.parameters) == ["self", "overrides"]

    resolve_context_length_signature = inspect.signature(
        TtsEngineBuilder.resolve_context_length
    )
    assert list(resolve_context_length_signature.parameters) == [
        "self",
        "checkpoint_dir",
        "server_args_overrides",
    ]


def test_context_length_override_is_capability_gated() -> None:
    from sglang_omni.models.arkasr.engine_builder import ArkasrEngineBuilder
    from sglang_omni.models.moss_tts.engine_builder import MossTtsEngineBuilder
    from sglang_omni.models.moss_tts_local.engine_builder import (
        MossTtsLocalEngineBuilder,
    )
    from sglang_omni.scheduling.engine_factory import TtsEngineBuilder

    assert TtsEngineBuilder.supports_context_length_override is False
    assert ArkasrEngineBuilder.supports_context_length_override is False
    assert MossTtsEngineBuilder.supports_context_length_override is True
    assert MossTtsLocalEngineBuilder.supports_context_length_override is True


@pytest.mark.parametrize(
    "value",
    [True, False, 1.5, 3.0, float("inf"), float("-inf"), float("nan"), "8192", None],
)
def test_normalize_context_length_rejects_non_integral_values(value: Any) -> None:
    from sglang_omni.scheduling.engine_factory import _normalize_context_length

    with pytest.raises(ValueError, match="context length must be a positive integer"):
        _normalize_context_length(value, model_name="MOSS-TTS")


@pytest.mark.parametrize("value", [0, -1])
def test_normalize_context_length_rejects_non_positive_values(value: int) -> None:
    from sglang_omni.scheduling.engine_factory import _normalize_context_length

    with pytest.raises(ValueError, match="resolved an invalid context length"):
        _normalize_context_length(value, model_name="MOSS-TTS")


@pytest.mark.parametrize("value", [1, 8192])
def test_normalize_context_length_preserves_integral_values(value: int) -> None:
    from sglang_omni.scheduling.engine_factory import _normalize_context_length

    assert _normalize_context_length(value, model_name="MOSS-TTS") == value


def test_tts_engine_builder_phase_order_and_override_contract(monkeypatch) -> None:
    from sglang_omni.scheduling import bootstrap, sglang_backend
    from sglang_omni.scheduling.engine_factory import TtsEngineBuilder

    monkeypatch.setattr(
        platforms.current_platform, "device_type", "cuda", raising=False
    )
    monkeypatch.setattr(
        "sglang.srt.utils.get_device", lambda device_id=None: f"cuda:{device_id}"
    )

    events: list[str] = []
    build_kwargs: dict[str, Any] = {}
    init_graph_calls: list[bool] = []

    class FakeModel:
        pass

    class FakeSGLangRunner:
        def __init__(self, server_args: Any) -> None:
            self.server_args = server_args
            self.model = FakeModel()

        def init_cuda_graphs(self) -> None:
            events.append("init_graphs")
            init_graph_calls.append(True)

    class FakeWorker:
        def __init__(self, server_args: Any) -> None:
            self.model_runner = FakeSGLangRunner(server_args)
            self.model_config = SimpleNamespace(is_multimodal=False)
            self.enable_prefill_input_embeds = False

    def fake_build_sglang_server_args(
        checkpoint_dir: str,
        *,
        context_length: int,
        **kwargs: Any,
    ) -> Any:
        events.append("build_server_args")
        build_kwargs.update(kwargs)
        return SimpleNamespace(
            checkpoint_dir=checkpoint_dir,
            context_length=context_length,
            cuda_graph_bs=kwargs["cuda_graph_bs"],
            cuda_graph_max_bs=kwargs["cuda_graph_max_bs"],
            cuda_graph_config=SimpleNamespace(
                decode=SimpleNamespace(
                    max_bs=kwargs["cuda_graph_max_bs"],
                    bs=kwargs["cuda_graph_bs"],
                ),
                prefill=SimpleNamespace(backend="disabled", bs=None, max_bs=None),
            ),
            disable_cuda_graph=kwargs["disable_cuda_graph"],
            enable_torch_compile=kwargs["enable_torch_compile"],
            max_running_requests=kwargs["max_running_requests"],
            mem_fraction_static=kwargs["mem_fraction_static"],
            torch_compile_max_bs=kwargs["torch_compile_max_bs"],
        )

    def fake_create_sglang_infrastructure(
        server_args: Any,
        gpu_id: int,
        **kwargs: Any,
    ) -> tuple[Any, ...]:
        events.append("infrastructure")
        assert gpu_id == 2
        assert kwargs == {
            "defer_cuda_graph_capture": True,
            "model_arch_override": "TestArch",
        }
        return (
            FakeWorker(server_args),
            "tree_cache",
            "req_pool",
            "kv_pool",
            "model_config",
        )

    def fake_output_processor(**kwargs: Any) -> Any:
        events.append("output_processor")
        assert kwargs["capture_hidden"] is False
        assert kwargs["capture_hidden_layers"] is None
        assert isinstance(kwargs["model"], FakeModel)
        return SimpleNamespace(**kwargs)

    monkeypatch.setattr(
        sglang_backend,
        "build_sglang_server_args",
        fake_build_sglang_server_args,
    )
    monkeypatch.setattr(
        bootstrap,
        "create_sglang_infrastructure",
        fake_create_sglang_infrastructure,
    )
    monkeypatch.setattr(sglang_backend, "SGLangOutputProcessor", fake_output_processor)

    class RecordingBuilder(TtsEngineBuilder):
        model_name = "Test TTS"
        context_length = 123
        model_arch_override = "TestArch"

        def resolve_checkpoint(self, model_path: str) -> str:
            events.append("resolve_checkpoint")
            return f"{model_path}-resolved"

        def pre_infra_setup(self, checkpoint_dir: str) -> None:
            events.append("pre_infra_setup")
            assert checkpoint_dir == "model-resolved"

        def resolve_context_length(
            self,
            checkpoint_dir: str,
            *,
            server_args_overrides: dict[str, Any] | None = None,
        ) -> int:
            events.append("resolve_context_length")
            assert checkpoint_dir == "model-resolved"
            assert server_args_overrides == {
                "cuda_graph_max_bs": 8,
                "torch_compile_max_bs": 8,
                "mem_fraction_static": 0.7,
                "max_total_tokens": TEST_MAX_TOTAL_TOKENS,
                "max_running_requests": 2,
                "trust_remote_code": False,
                "model_config_parser": "hf",
                "json_model_override_args": (
                    '{"language_config": {"max_position_embeddings": 4096}}'
                ),
                "decrypted_config_file": "/tmp/override.json",
            }
            return super().resolve_context_length(
                checkpoint_dir,
                server_args_overrides=server_args_overrides,
            )

        def generation_defaults(
            self,
            *,
            dtype: str,
        ) -> dict[str, Any]:
            events.append("generation_defaults")
            assert dtype == "bfloat16"
            return {
                "max_running_requests": 4,
                "cuda_graph_max_bs": 4,
                "torch_compile_max_bs": 4,
                "dtype": dtype,
                "disable_cuda_graph": False,
                "enable_torch_compile": True,
                "mem_fraction_static": 0.5,
            }

        def adjust_overrides(self, overrides: dict[str, Any]) -> None:
            events.append("adjust_overrides")
            assert overrides["mem_fraction_static"] == 0.7

        def customize_server_args(self, server_args: Any) -> None:
            events.append("customize_server_args")
            assert server_args.context_length == 123

        def setup_model(
            self,
            *,
            model_worker: Any,
            checkpoint_dir: str,
            device: str,
            gpu_id: int,
            server_args: Any,
        ) -> None:
            events.append("setup_model")
            assert isinstance(model_worker.model_runner.model, FakeModel)
            assert checkpoint_dir == "model-resolved"
            assert device == "cuda:2"
            assert gpu_id == 2
            assert server_args.disable_cuda_graph is False

        def get_model_buffer_bs(self, model: Any) -> int | None:
            events.append("get_model_buffer_bs")
            assert isinstance(model, FakeModel)
            return 2

        def compile_model(self, model: Any, server_args: Any) -> None:
            events.append("compile_model")
            assert isinstance(model, FakeModel)
            assert server_args.torch_compile_max_bs == 8

        def post_cuda_graph_setup(self, model: Any, server_args: Any) -> None:
            events.append("post_cuda_graph_setup")
            assert isinstance(model, FakeModel)
            assert server_args.disable_cuda_graph is False

        def make_model_runner(self, model_worker: Any, output_proc: Any) -> Any:
            events.append("make_model_runner")
            return SimpleNamespace(model_worker=model_worker, output_proc=output_proc)

        def make_adapters(self, model: Any) -> tuple[Any, Any]:
            events.append("make_adapters")
            assert isinstance(model, FakeModel)
            return "request_builder", "result_adapter"

        def make_scheduler(
            self,
            *,
            model_worker: Any,
            tree_cache: Any,
            req_to_token_pool: Any,
            token_to_kv_pool_allocator: Any,
            server_args: Any,
            model_config: Any,
            model_runner: Any,
            request_builder: Any,
            result_adapter: Any,
        ) -> Any:
            events.append("make_scheduler")
            assert request_builder == "request_builder"
            assert result_adapter == "result_adapter"
            return SimpleNamespace(
                outbox="outbox",
                kwargs={
                    "model_worker": model_worker,
                    "tree_cache": tree_cache,
                    "req_to_token_pool": req_to_token_pool,
                    "token_to_kv_pool_allocator": token_to_kv_pool_allocator,
                    "server_args": server_args,
                    "model_config": model_config,
                    "model_runner": model_runner,
                },
            )

        def post_scheduler_setup(self, scheduler: Any, model_runner: Any) -> None:
            events.append("post_scheduler_setup")
            model_runner.outbox = scheduler.outbox

    scheduler = RecordingBuilder().build(
        "model",
        device="cuda:0",
        gpu_id=2,
        server_args_overrides={
            "cuda_graph_max_bs": 8,
            "torch_compile_max_bs": 8,
            "mem_fraction_static": 0.7,
            "max_total_tokens": TEST_MAX_TOTAL_TOKENS,
            "max_running_requests": 2,
            "trust_remote_code": False,
            "model_config_parser": "hf",
            "json_model_override_args": (
                '{"language_config": {"max_position_embeddings": 4096}}'
            ),
            "decrypted_config_file": "/tmp/override.json",
        },
    )

    assert events == [
        "resolve_checkpoint",
        "pre_infra_setup",
        "resolve_context_length",
        "generation_defaults",
        "adjust_overrides",
        "build_server_args",
        "customize_server_args",
        "infrastructure",
        "setup_model",
        "get_model_buffer_bs",
        "compile_model",
        "init_graphs",
        "post_cuda_graph_setup",
        "output_processor",
        "make_model_runner",
        "make_adapters",
        "make_scheduler",
        "post_scheduler_setup",
    ]
    assert build_kwargs["max_running_requests"] == 2
    assert build_kwargs["device"] == "cuda"
    assert build_kwargs["cuda_graph_max_bs"] == 8
    assert build_kwargs["torch_compile_max_bs"] == 8
    assert build_kwargs["mem_fraction_static"] == 0.7
    assert build_kwargs["max_total_tokens"] == TEST_MAX_TOTAL_TOKENS
    assert build_kwargs["trust_remote_code"] is False
    assert build_kwargs["model_config_parser"] == "hf"
    assert build_kwargs["json_model_override_args"] == (
        '{"language_config": {"max_position_embeddings": 4096}}'
    )
    assert build_kwargs["decrypted_config_file"] == "/tmp/override.json"
    assert init_graph_calls == [True]
    assert scheduler.kwargs["server_args"].disable_cuda_graph is False
    assert scheduler.kwargs["model_runner"].outbox == "outbox"


def _build_minimal_tts_builder_harness(monkeypatch):
    """Fakes for exercising ``build()`` without CUDA graphs or a real engine."""
    from sglang_omni.scheduling import bootstrap, sglang_backend
    from sglang_omni.scheduling.engine_factory import TtsEngineBuilder
    from sglang_omni.scheduling.stage_kv_budget import consume_stage_kv_cache_bytes

    monkeypatch.setattr(
        platforms.current_platform, "device_type", "cuda", raising=False
    )
    monkeypatch.setattr(
        "sglang.srt.utils.get_device", lambda device_id=None: f"cuda:{device_id}"
    )

    build_kwargs: dict[str, Any] = {}
    consumed: list[int | None] = []

    class FakeModel:
        pass

    class FakeWorker:
        def __init__(self, server_args: Any) -> None:
            self.model_runner = SimpleNamespace(
                model=FakeModel(), server_args=server_args
            )
            self.model_config = SimpleNamespace(is_multimodal=False)
            self.enable_prefill_input_embeds = False

    def fake_build_sglang_server_args(
        checkpoint_dir: str, *, context_length: int, **kwargs: Any
    ) -> Any:
        build_kwargs.update(kwargs)
        return FakeServerArgs(
            checkpoint_dir=checkpoint_dir,
            context_length=context_length,
            cuda_graph_bs=None,
            cuda_graph_max_bs=None,
            cuda_graph_config=SimpleNamespace(
                decode=SimpleNamespace(max_bs=None, bs=None),
                prefill=SimpleNamespace(backend="disabled", bs=None, max_bs=None),
            ),
            disable_cuda_graph=True,
            enable_torch_compile=False,
            max_running_requests=kwargs["max_running_requests"],
            mem_fraction_static=kwargs.get("mem_fraction_static"),
            torch_compile_max_bs=None,
        )

    def fake_create_sglang_infrastructure(
        server_args: Any, gpu_id: int, **kwargs: Any
    ) -> tuple[Any, ...]:
        consumed.append(consume_stage_kv_cache_bytes())
        return (
            FakeWorker(server_args),
            "tree_cache",
            "req_pool",
            "kv_pool",
            "model_config",
        )

    monkeypatch.setattr(
        sglang_backend, "build_sglang_server_args", fake_build_sglang_server_args
    )
    monkeypatch.setattr(
        bootstrap, "create_sglang_infrastructure", fake_create_sglang_infrastructure
    )
    monkeypatch.setattr(
        sglang_backend,
        "SGLangOutputProcessor",
        lambda **kwargs: SimpleNamespace(**kwargs),
    )

    class MinimalBuilder(TtsEngineBuilder):
        model_name = "Test TTS"
        context_length = 123

        def resolve_checkpoint(self, model_path: str) -> str:
            return model_path

        def generation_defaults(self, *, dtype: str) -> dict[str, Any]:
            return {
                "max_running_requests": 4,
                "dtype": dtype,
                "disable_cuda_graph": True,
                "mem_fraction_static": 0.2,
            }

        def setup_model(self, **kwargs: Any) -> None:
            pass

        def make_model_runner(self, model_worker: Any, output_proc: Any) -> Any:
            return SimpleNamespace(model_worker=model_worker)

        def make_adapters(self, model: Any) -> tuple[Any, Any]:
            return "request_builder", "result_adapter"

        def make_scheduler(self, **kwargs: Any) -> Any:
            return SimpleNamespace(outbox="outbox")

    return MinimalBuilder, build_kwargs, consumed


def test_byte_budget_clears_builder_default_mem_fraction(monkeypatch, caplog) -> None:
    """A KV-tuned builder default (e.g. dots 0.20) must not shape ServerArgs when
    the byte budget owns KV sizing, or post-capture headroom inherits it."""
    import logging

    from sglang_omni.scheduling import engine_factory
    from sglang_omni.scheduling.stage_kv_budget import stage_kv_cache_budget

    MinimalBuilder, build_kwargs, consumed = _build_minimal_tts_builder_harness(
        monkeypatch
    )

    with caplog.at_level(logging.INFO, logger=engine_factory.logger.name):
        with stage_kv_cache_budget("tts_engine", 2 * 1024**3):
            MinimalBuilder().build("model", device="cuda:0", gpu_id=0)

    assert "mem_fraction_static" not in build_kwargs
    assert consumed == [2 * 1024**3]
    assert "clearing builder default mem_fraction_static=0.2" in caplog.text


def test_without_byte_budget_builder_default_mem_fraction_is_kept(
    monkeypatch,
) -> None:
    MinimalBuilder, build_kwargs, consumed = _build_minimal_tts_builder_harness(
        monkeypatch
    )

    MinimalBuilder().build("model", device="cuda:0", gpu_id=0)

    assert build_kwargs["mem_fraction_static"] == 0.2
    assert consumed == [None]


def test_asr_engine_builder_phase_order_and_failure_cleanup(monkeypatch) -> None:
    from sglang_omni.scheduling import bootstrap, engine_factory, sglang_backend
    from sglang_omni.scheduling.engine_factory import AsrEngineBuilder

    events: list[str] = []

    class FakeModel:
        pass

    class FakeSGLangRunner:
        def __init__(self) -> None:
            self.model = FakeModel()

        def init_cuda_graphs(self) -> None:
            events.append("init_cuda_graphs")

    model_worker = SimpleNamespace(
        model_runner=FakeSGLangRunner(),
        model_config=SimpleNamespace(is_multimodal=False),
        enable_prefill_input_embeds=False,
    )

    def fake_server_args(*args: Any, **kwargs: Any) -> Any:
        del args, kwargs
        events.append("server_args")
        return SimpleNamespace(
            cuda_graph_config=SimpleNamespace(
                prefill=SimpleNamespace(backend="disabled")
            ),
            _cuda_graph_config_locked=set(),
        )

    def fake_validate(**kwargs: Any) -> None:
        assert kwargs["model_name"] == "Test ASR"
        events.append("validate")

    def fake_infrastructure(*args: Any, **kwargs: Any) -> tuple[Any, ...]:
        del args, kwargs
        events.append("infrastructure")
        return True, (
            model_worker,
            "tree_cache",
            "req_pool",
            "kv_pool",
            "model_config",
        )

    def fake_output_processor(**kwargs: Any) -> Any:
        assert isinstance(kwargs["model"], FakeModel)
        events.append("output_processor")
        return object()

    monkeypatch.setattr(sglang_backend, "build_sglang_server_args", fake_server_args)
    monkeypatch.setattr(
        bootstrap,
        "create_sglang_infrastructure_defer_cuda_graph",
        fake_infrastructure,
    )
    monkeypatch.setattr(sglang_backend, "SGLangOutputProcessor", fake_output_processor)
    monkeypatch.setattr(
        engine_factory, "validate_generation_batch_policy", fake_validate
    )

    class RecordingBuilder(AsrEngineBuilder):
        model_name = "Test ASR"
        context_length = 256

        def generation_defaults(self, *, dtype: str) -> dict[str, Any]:
            events.append("generation_defaults")
            return {"max_running_requests": 4, "dtype": dtype}

        def setup_model(self, **kwargs: Any) -> None:
            del kwargs
            events.append("setup_model")

        def compile_model(self, model: Any, server_args: Any) -> None:
            del model, server_args
            events.append("compile_model")

        def post_cuda_graph_setup(self, model: Any, server_args: Any) -> None:
            del model, server_args
            events.append("post_cuda_graph_setup")

        def setup_model_resources(
            self,
            model: Any,
            server_args: Any,
            *,
            generation_cuda_graph_enabled: bool,
        ) -> None:
            del model, server_args
            assert generation_cuda_graph_enabled is True
            events.append("setup_model_resources")

        def setup_runtime_resources(self, model: Any, server_args: Any) -> None:
            del model, server_args
            events.append("setup_runtime_resources")

        def make_adapters(self, model: Any) -> tuple[Any, Any]:
            del model
            events.append("make_adapters")
            return "request_builder", "result_adapter"

        def make_model_runner(self, model_worker: Any, output_proc: Any) -> Any:
            del model_worker, output_proc
            events.append("make_model_runner")
            return "model_runner"

        def extra_scheduler_kwargs(self) -> dict[str, Any]:
            events.append("extra_scheduler_kwargs")
            return {"stream_output_builder": "stream_builder"}

        def _make_scheduler(self, **kwargs: Any) -> Any:
            assert kwargs["extra_scheduler_kwargs"] == {
                "stream_output_builder": "stream_builder"
            }
            events.append("make_scheduler")
            raise RuntimeError("scheduler failed")

        def cleanup_build_failure(self) -> None:
            events.append("cleanup_build_failure")

    with pytest.raises(RuntimeError, match="scheduler failed"):
        RecordingBuilder().build("repo/id")

    assert events == [
        "generation_defaults",
        "server_args",
        "validate",
        "infrastructure",
        "setup_model",
        "compile_model",
        "init_cuda_graphs",
        "post_cuda_graph_setup",
        "setup_model_resources",
        "output_processor",
        "setup_runtime_resources",
        "make_adapters",
        "extra_scheduler_kwargs",
        "make_model_runner",
        "make_scheduler",
        "cleanup_build_failure",
    ]


def test_tts_engine_builder_base_scheduler_preserves_abort_with_extra_kwargs(
    monkeypatch,
) -> None:
    from sglang_omni.scheduling import omni_scheduler
    from sglang_omni.scheduling.engine_factory import TtsEngineBuilder

    captured_kwargs: dict[str, Any] = {}

    class FakeScheduler:
        def __init__(self, **kwargs: Any) -> None:
            captured_kwargs.update(kwargs)

    monkeypatch.setattr(omni_scheduler, "OmniScheduler", FakeScheduler)

    def abort_callback(request_id: str) -> None:
        del request_id

    class SchedulerKwargsBuilder(TtsEngineBuilder):
        model_name = "Test TTS"
        context_length = 123

        def resolve_checkpoint(self, model_path: str) -> str:
            return model_path

        def generation_defaults(
            self,
            *,
            dtype: str,
        ) -> dict[str, Any]:
            del dtype
            return {}

        def setup_model(
            self,
            *,
            model_worker: Any,
            checkpoint_dir: str,
            device: str,
            gpu_id: int,
            server_args: Any,
        ) -> None:
            del model_worker, checkpoint_dir, device, gpu_id, server_args

        def make_model_runner(self, model_worker: Any, output_proc: Any) -> Any:
            del output_proc
            return model_worker

        def make_adapters(self, model: Any) -> tuple[Any, Any]:
            del model
            return object(), object()

        def make_abort_callback(self) -> Any | None:
            return abort_callback

        def extra_scheduler_kwargs(self) -> dict[str, Any]:
            return {
                "enable_async_decode": True,
                "async_decode_min_batch_size": 3,
            }

    scheduler = SchedulerKwargsBuilder().make_scheduler(
        model_worker="worker",
        tree_cache="tree_cache",
        req_to_token_pool="req_pool",
        token_to_kv_pool_allocator="kv_pool",
        server_args="server_args",
        model_config="model_config",
        model_runner="runner",
        request_builder="request_builder",
        result_adapter="result_adapter",
    )

    assert isinstance(scheduler, FakeScheduler)
    assert captured_kwargs["abort_callback"] is abort_callback
    assert captured_kwargs["enable_async_decode"] is True
    assert captured_kwargs["async_decode_min_batch_size"] == 3
    assert captured_kwargs["tp_worker"] == "worker"
    assert captured_kwargs["request_builder"] == "request_builder"
    assert captured_kwargs["result_adapter"] == "result_adapter"
