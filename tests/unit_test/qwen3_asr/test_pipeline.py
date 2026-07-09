# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import inspect
from types import SimpleNamespace

import sglang_omni.models.qwen3_asr.stages as qwen3_asr_stages
from sglang_omni.models.qwen3_asr.config import Qwen3ASRPipelineConfig
from sglang_omni.models.qwen3_asr.stages import create_sglang_qwen3_asr_executor
from sglang_omni.models.registry import PIPELINE_CONFIG_REGISTRY


def test_qwen3_asr_config_uses_batched_stage_with_auto_batch_caps() -> None:
    config = Qwen3ASRPipelineConfig(model_path="Qwen/Qwen3-ASR-1.7B")

    assert config.entry_stage == "asr"
    assert [stage.name for stage in config.stages] == ["asr"]
    assert config.terminal_stages == ["asr"]
    assert config.gpu_placement == {"asr": 0}
    assert config.stages[0].factory.endswith("create_sglang_qwen3_asr_executor")
    assert config.stages[0].factory_args["device"] == "cuda:0"
    # Batch caps auto-tier to the GPU's total memory (small GPUs keep 32 / 16).
    assert config.stages[0].factory_args["max_running_requests"] == "auto"
    assert config.stages[0].factory_args["request_build_max_workers"] == 2
    assert config.stages[0].factory_args["request_build_max_pending"] == "auto"
    assert "request_build_max_backlog" not in config.stages[0].factory_args
    assert (
        PIPELINE_CONFIG_REGISTRY.get_config("Qwen3ASRForConditionalGeneration")
        is Qwen3ASRPipelineConfig
    )


def test_qwen3_asr_stage_defaults_to_auto_batch_caps() -> None:
    signature = inspect.signature(create_sglang_qwen3_asr_executor)

    assert signature.parameters["max_running_requests"].default == "auto"
    assert signature.parameters["request_build_max_workers"].default == 2
    assert signature.parameters["request_build_max_pending"].default == "auto"
    assert "request_build_max_backlog" not in signature.parameters


def test_qwen3_asr_stage_default_uses_auto_static_kv_budget() -> None:
    signature = inspect.signature(create_sglang_qwen3_asr_executor)

    assert signature.parameters["mem_fraction_static"].default is None


def test_qwen3_asr_stage_default_disables_multimodal_embedding_cache() -> None:
    signature = inspect.signature(create_sglang_qwen3_asr_executor)

    assert signature.parameters["mm_embedding_cache_size_bytes"].default == 0


def test_qwen3_asr_stage_default_disables_torch_compile() -> None:
    signature = inspect.signature(create_sglang_qwen3_asr_executor)

    assert signature.parameters["enable_torch_compile"].default is False


def test_qwen3_asr_threads_explicit_cuda_graph_bs(monkeypatch) -> None:
    build_kwargs: dict[str, object] = {}

    monkeypatch.setattr(
        qwen3_asr_stages.AutoTokenizer,
        "from_pretrained",
        lambda *args, **kwargs: object(),
    )
    monkeypatch.setattr(
        qwen3_asr_stages.AutoFeatureExtractor,
        "from_pretrained",
        lambda *args, **kwargs: SimpleNamespace(nb_max_frames=3000),
    )
    monkeypatch.setattr(
        qwen3_asr_stages,
        "get_visible_gpu_sm_version",
        lambda gpu_id: None,
    )
    # Force the "auto" batch caps to the small-GPU floor (32 / 16) so the
    # cuda_graph_max_bs assertions below are deterministic regardless of the
    # host GPU the unit tests happen to run on.
    monkeypatch.setattr(
        qwen3_asr_stages,
        "get_gpu_device_info",
        lambda gpu_id: SimpleNamespace(total_memory_bytes=None),
    )
    monkeypatch.setattr(qwen3_asr_stages, "init_mm_embedding_cache", lambda size: None)
    monkeypatch.setattr(
        qwen3_asr_stages,
        "make_qwen3_asr_scheduler_adapters",
        lambda **kwargs: (object(), object()),
    )
    monkeypatch.setattr(
        qwen3_asr_stages,
        "ModelRunner",
        lambda *args, **kwargs: object(),
    )
    monkeypatch.setattr(
        qwen3_asr_stages,
        "SGLangOutputProcessor",
        lambda **kwargs: object(),
    )
    monkeypatch.setattr(
        qwen3_asr_stages,
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
        qwen3_asr_stages,
        "build_sglang_server_args",
        _fake_server_args_builder,
    )
    monkeypatch.setattr(
        qwen3_asr_stages,
        "create_sglang_infrastructure_defer_cuda_graph",
        _fake_create_infrastructure,
    )

    qwen3_asr_stages.create_sglang_qwen3_asr_executor("dummy")

    assert build_kwargs["cuda_graph_max_bs"] == 32
    assert build_kwargs["cuda_graph_bs"] == [1, 2, 4, 8, 12, 16, 24, 32]


def test_qwen3_asr_auto_batch_caps_scale_on_large_gpu(monkeypatch) -> None:
    build_kwargs: dict[str, object] = {}
    scheduler_kwargs: dict[str, object] = {}

    monkeypatch.setattr(
        qwen3_asr_stages.AutoTokenizer,
        "from_pretrained",
        lambda *args, **kwargs: object(),
    )
    monkeypatch.setattr(
        qwen3_asr_stages.AutoFeatureExtractor,
        "from_pretrained",
        lambda *args, **kwargs: SimpleNamespace(nb_max_frames=3000),
    )
    monkeypatch.setattr(
        qwen3_asr_stages,
        "get_visible_gpu_sm_version",
        lambda gpu_id: None,
    )
    # An 80 GiB GPU (H100-class) must lift the auto caps to 128 / 256, and that
    # must flow through to cuda_graph_max_bs and the scheduler's build backlog.
    monkeypatch.setattr(
        qwen3_asr_stages,
        "get_gpu_device_info",
        lambda gpu_id: SimpleNamespace(total_memory_bytes=80 * (1024**3)),
    )
    monkeypatch.setattr(qwen3_asr_stages, "init_mm_embedding_cache", lambda size: None)
    monkeypatch.setattr(
        qwen3_asr_stages,
        "make_qwen3_asr_scheduler_adapters",
        lambda **kwargs: (object(), object()),
    )
    monkeypatch.setattr(
        qwen3_asr_stages,
        "ModelRunner",
        lambda *args, **kwargs: object(),
    )
    monkeypatch.setattr(
        qwen3_asr_stages,
        "SGLangOutputProcessor",
        lambda **kwargs: object(),
    )

    def _capture_scheduler(**kwargs):
        scheduler_kwargs.update(kwargs)
        return SimpleNamespace(**kwargs)

    monkeypatch.setattr(qwen3_asr_stages, "OmniScheduler", _capture_scheduler)

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
        qwen3_asr_stages,
        "build_sglang_server_args",
        _fake_server_args_builder,
    )
    monkeypatch.setattr(
        qwen3_asr_stages,
        "create_sglang_infrastructure_defer_cuda_graph",
        _fake_create_infrastructure,
    )

    qwen3_asr_stages.create_sglang_qwen3_asr_executor("dummy")

    assert build_kwargs["max_running_requests"] == 128
    assert build_kwargs["cuda_graph_max_bs"] == 128
    assert max(build_kwargs["cuda_graph_bs"]) == 128
    # request_build_max_workers stays at its conservative default (GIL headroom).
    assert scheduler_kwargs["request_build_max_workers"] == 2
    assert scheduler_kwargs["request_build_max_pending"] == 256
