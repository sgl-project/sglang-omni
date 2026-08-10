# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import inspect
from pathlib import Path
from types import SimpleNamespace

import pytest

import sglang_omni.model_runner.base as model_runner_base
import sglang_omni.models.qwen3_asr.engine_builder as qwen3_asr_builder
import sglang_omni.models.qwen3_asr.stages as qwen3_asr_stages
import sglang_omni.scheduling.bootstrap as bootstrap
import sglang_omni.scheduling.omni_scheduler as omni_scheduler
import sglang_omni.scheduling.sglang_backend as sglang_backend
from sglang_omni.config.manager import ConfigManager
from sglang_omni.config.runtime import resolve_stage_static_factory_args
from sglang_omni.models.qwen3_asr import request_builders
from sglang_omni.models.qwen3_asr.config import Qwen3ASRPipelineConfig
from sglang_omni.models.qwen3_asr.stages import create_sglang_qwen3_asr_executor
from sglang_omni.models.registry import PIPELINE_CONFIG_REGISTRY
from tests.unit_test.fakes import FakeServerArgs


def _make_engine_builder(
    *, mm_attention_backend: str | None = None
) -> qwen3_asr_builder.Qwen3ASREngineBuilder:
    return qwen3_asr_builder.Qwen3ASREngineBuilder(
        max_running_requests=64,
        max_new_tokens=128,
        enable_async_decode=True,
        async_decode_min_batch_size=2,
        mem_fraction_static=None,
        mm_embedding_cache_size_bytes=0,
        enable_torch_compile=False,
        mm_attention_backend=mm_attention_backend,
        request_build_max_workers=8,
        request_build_max_pending=32,
        prefill_coalesce_requests=16,
        prefill_coalesce_wait_ms=24.0,
        prefill_coalesce_when_idle=True,
        prefill_coalesce_requires_pending_builds=True,
        prefill_coalesce_after_builds_during_decode=True,
    )


@pytest.mark.parametrize(
    ("sm_version", "expected_backend"),
    [(89, None), (100, "triton_attn"), (120, "triton_attn")],
)
def test_qwen3_asr_default_mm_attention_backend_by_sm(
    monkeypatch: pytest.MonkeyPatch,
    sm_version: int,
    expected_backend: str | None,
) -> None:
    queried_gpu_ids: list[int] = []
    monkeypatch.setattr(
        qwen3_asr_builder,
        "get_visible_gpu_sm_version",
        lambda gpu_id: queried_gpu_ids.append(gpu_id) or sm_version,
        raising=False,
    )
    builder = _make_engine_builder()
    builder.gpu_id = 3

    defaults = builder.generation_defaults(dtype="bfloat16")

    assert queried_gpu_ids == [3]
    if expected_backend is None:
        assert "mm_attention_backend" not in defaults
    else:
        assert defaults["mm_attention_backend"] == expected_backend


def test_qwen3_asr_explicit_mm_attention_backend_overrides_sm_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        qwen3_asr_builder,
        "get_visible_gpu_sm_version",
        lambda gpu_id: pytest.fail(f"unexpected SM lookup for GPU {gpu_id}"),
        raising=False,
    )
    builder = _make_engine_builder(mm_attention_backend="fa3")
    builder.gpu_id = 0

    defaults = builder.generation_defaults(dtype="bfloat16")

    assert defaults["mm_attention_backend"] == "fa3"


def test_qwen3_asr_config_uses_batched_stage_with_64_running_requests() -> None:
    config = Qwen3ASRPipelineConfig(model_path="Qwen/Qwen3-ASR-1.7B")

    assert config.entry_stage == "asr"
    assert [stage.name for stage in config.stages] == ["asr"]
    assert config.terminal_stages == ["asr"]
    assert config.gpu_placement == {"asr": 0}
    assert config.stages[0].factory.endswith("create_sglang_qwen3_asr_executor")
    assert config.stages[0].factory_args["device"] == "cuda:0"
    assert config.stages[0].factory_args["max_running_requests"] == 64
    assert config.stages[0].factory_args["request_build_max_workers"] == 8
    assert config.stages[0].factory_args["request_build_max_pending"] == 32
    assert config.stages[0].factory_args["prefill_coalesce_requests"] == 16
    assert config.stages[0].factory_args["prefill_coalesce_wait_ms"] == 24
    assert config.stages[0].factory_args["prefill_coalesce_when_idle"] is True
    assert (
        config.stages[0].factory_args["prefill_coalesce_requires_pending_builds"]
        is True
    )
    assert (
        config.stages[0].factory_args["prefill_coalesce_after_builds_during_decode"]
        is True
    )
    assert "request_build_max_backlog" not in config.stages[0].factory_args
    assert config.stages[0].factory_args["enable_pre_lm_encoder"] is True
    assert config.stages[0].factory_args["pre_lm_cache_max_entries"] == 4096
    assert config.stages[0].factory_args["pre_lm_cache_size_bytes"] == 2 * 1024**3
    assert config.stages[0].factory_args["pre_lm_max_batch_size"] == 8
    assert config.stages[0].factory_args["pre_lm_max_batch_wait_ms"] == 0
    assert Qwen3ASRPipelineConfig.mem_fraction_role_to_stage() == {"asr": "asr"}
    assert Qwen3ASRPipelineConfig.generation_sglang_role_to_stage() == {
        "generation": "asr"
    }
    assert (
        PIPELINE_CONFIG_REGISTRY.get_config("Qwen3ASRForConditionalGeneration")
        is Qwen3ASRPipelineConfig
    )


def test_qwen3_asr_stage_default_allows_64_running_requests() -> None:
    signature = inspect.signature(create_sglang_qwen3_asr_executor)

    assert signature.parameters["max_running_requests"].default == 64
    assert signature.parameters["request_build_max_workers"].default == 8
    assert signature.parameters["request_build_max_pending"].default == 32
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
    assert "request_build_max_backlog" not in signature.parameters


def test_qwen3_asr_stage_default_uses_auto_dtype() -> None:
    signature = inspect.signature(create_sglang_qwen3_asr_executor)

    assert signature.parameters["dtype"].default == "auto"


def test_qwen3_asr_stage_default_enables_pre_lm_encoder() -> None:
    signature = inspect.signature(create_sglang_qwen3_asr_executor)

    assert signature.parameters["enable_pre_lm_encoder"].default is True
    assert signature.parameters["pre_lm_cache_max_entries"].default == 4096
    assert signature.parameters["pre_lm_cache_size_bytes"].default == 2 * 1024**3
    assert signature.parameters["pre_lm_max_batch_size"].default == 8
    assert signature.parameters["pre_lm_max_batch_wait_ms"].default == 0


@pytest.mark.parametrize(
    ("batch_size", "wait_ms", "match"),
    [
        (0, 4, "pre_lm_max_batch_size"),
        (-1, 4, "pre_lm_max_batch_size"),
        (8, -1, "pre_lm_max_batch_wait_ms"),
    ],
)
def test_qwen3_asr_stage_rejects_invalid_pre_lm_batch_knobs(
    batch_size: int, wait_ms: int, match: str
) -> None:
    # Validation runs before any model/tokenizer load.
    with pytest.raises(ValueError, match=match):
        create_sglang_qwen3_asr_executor(
            "dummy",
            pre_lm_max_batch_size=batch_size,
            pre_lm_max_batch_wait_ms=wait_ms,
        )


def test_qwen3_asr_stage_default_uses_auto_static_kv_budget() -> None:
    signature = inspect.signature(create_sglang_qwen3_asr_executor)

    assert signature.parameters["mem_fraction_static"].default is None


def test_qwen3_asr_stage_default_disables_multimodal_embedding_cache() -> None:
    signature = inspect.signature(create_sglang_qwen3_asr_executor)

    assert signature.parameters["mm_embedding_cache_size_bytes"].default == 0


def test_qwen3_asr_stage_default_disables_torch_compile() -> None:
    signature = inspect.signature(create_sglang_qwen3_asr_executor)

    assert signature.parameters["enable_torch_compile"].default is False


def test_qwen3_asr_stage_default_enables_async_decode() -> None:
    signature = inspect.signature(create_sglang_qwen3_asr_executor)

    assert signature.parameters["enable_async_decode"].default is True
    assert signature.parameters["async_decode_min_batch_size"].default == 1


def test_qwen3_asr_rtx4090_profile_is_bf16_and_bounded() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    config = ConfigManager.from_file(
        str(repo_root / "examples/configs/qwen3_asr_rtx4090.yaml")
    ).config
    stage = config.stages[0]

    factory_args = resolve_stage_static_factory_args(stage, config)

    assert factory_args["dtype"] == "bfloat16"
    assert factory_args["max_running_requests"] == 16
    assert factory_args["server_args_overrides"]["mem_fraction_static"] == 0.65


def test_qwen3_asr_threads_explicit_cuda_graph_bs(monkeypatch, caplog) -> None:
    build_kwargs: dict[str, object] = {}
    adapter_kwargs: dict[str, object] = {}
    memory_queries: list[int] = []

    monkeypatch.setattr(
        qwen3_asr_builder.AutoTokenizer,
        "from_pretrained",
        lambda *args, **kwargs: object(),
    )
    monkeypatch.setattr(
        qwen3_asr_builder.AutoFeatureExtractor,
        "from_pretrained",
        lambda *args, **kwargs: SimpleNamespace(nb_max_frames=3000),
    )
    monkeypatch.setattr(
        qwen3_asr_builder,
        "get_visible_gpu_sm_version",
        lambda gpu_id: 89,
    )
    monkeypatch.setattr(
        qwen3_asr_builder,
        "get_process_gpu_memory_bytes",
        lambda gpu_id: memory_queries.append(gpu_id) or 0,
    )
    monkeypatch.setattr(qwen3_asr_builder, "init_mm_embedding_cache", lambda size: None)
    fake_encoder_service = SimpleNamespace(close=lambda: None)
    monkeypatch.setattr(
        qwen3_asr_builder,
        "Qwen3ASRPreLMEncoderService",
        lambda *args, **kwargs: fake_encoder_service,
    )
    monkeypatch.setattr(
        qwen3_asr_builder,
        "build_cache_namespace",
        lambda *args, **kwargs: "testns",
    )
    monkeypatch.setattr(
        request_builders,
        "make_qwen3_asr_scheduler_adapters",
        lambda **kwargs: (adapter_kwargs.update(kwargs) or object(), object()),
    )
    monkeypatch.setattr(
        sglang_backend,
        "SGLangOutputProcessor",
        lambda **kwargs: object(),
    )
    monkeypatch.setattr(
        model_runner_base,
        "ModelRunner",
        lambda *args, **kwargs: object(),
    )
    monkeypatch.setattr(
        omni_scheduler,
        "OmniScheduler",
        lambda **kwargs: SimpleNamespace(**kwargs),
    )

    def _fake_server_args_builder(model_path, context_length, **overrides):
        build_kwargs.update(overrides)
        normalized_overrides = {
            key: value
            for key, value in overrides.items()
            if key not in {"cuda_graph_bs", "cuda_graph_max_bs"}
        }
        server_args = FakeServerArgs(
            context_length=context_length, **normalized_overrides
        )
        server_args.cuda_graph_config = SimpleNamespace(
            decode=SimpleNamespace(
                max_bs=overrides["cuda_graph_max_bs"],
                bs=overrides["cuda_graph_bs"],
            ),
            prefill=SimpleNamespace(backend="disabled", bs=None, max_bs=None),
        )
        return server_args

    def _fake_create_infrastructure(server_args, gpu_id, **kwargs):
        model_worker = SimpleNamespace(
            gpu_id=gpu_id,
            model_runner=SimpleNamespace(model=object()),
        )
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

    with caplog.at_level("INFO", logger=qwen3_asr_builder.__name__):
        scheduler = qwen3_asr_stages.create_sglang_qwen3_asr_executor(
            "dummy",
            enable_async_decode=False,
            async_decode_min_batch_size=4,
            server_args_overrides={"context_length": 2048},
        )

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
    assert "cuda_graph_bs=[1, 2, 4, 8, 12, 16, 24, 32, 40, 48, 56, 64]" in caplog.text
    assert "mm_attention_backend" not in build_kwargs
    assert memory_queries == [0, 0, 0]
    assert adapter_kwargs["context_length"] == 2048
    assert scheduler.enable_async_decode is False
    assert scheduler.async_decode_min_batch_size == 4
    assert scheduler.prefill_coalesce_requests == 16
    assert scheduler.prefill_coalesce_wait_ms == 24.0
    assert scheduler.prefill_coalesce_when_idle is True
    assert scheduler.prefill_coalesce_requires_pending_builds is True
    assert scheduler.prefill_coalesce_after_builds_during_decode is True
    assert scheduler.shutdown_callback is fake_encoder_service.close
