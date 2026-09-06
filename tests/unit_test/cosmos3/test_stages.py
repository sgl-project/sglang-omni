# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace

from sglang_omni.models.cosmos3 import stages, vision_encoder_scheduler
from sglang_omni.scheduling.simple_scheduler import SimpleScheduler


def test_preprocessing_factory_returns_simple_scheduler(monkeypatch) -> None:
    sentinel = object()
    calls: list[tuple[str, int | None, str | None, bool]] = []

    def fake_preprocessor(
        model_path: str,
        max_seq_len: int | None,
        revision: str | None,
        enable_vision: bool,
    ):
        calls.append((model_path, max_seq_len, revision, enable_vision))
        return sentinel

    monkeypatch.setattr(stages, "Cosmos3TextPreprocessor", fake_preprocessor)

    scheduler = stages.create_preprocessing_executor(
        "nvidia/Cosmos3-Nano",
        revision="cosmos-revision",
        max_seq_len=8192,
    )

    assert isinstance(scheduler, SimpleScheduler)
    assert scheduler._fn is sentinel
    assert calls == [("nvidia/Cosmos3-Nano", 8192, "cosmos-revision", True)]


def test_vision_encoder_factory_delegates_to_scheduler(monkeypatch) -> None:
    sentinel = object()
    calls: list[tuple[str, str | None, str, str | None]] = []

    def fake_create(
        model_path: str,
        *,
        revision: str | None = None,
        device: str,
        dtype: str | None,
    ):
        calls.append((model_path, revision, device, dtype))
        return sentinel

    monkeypatch.setattr(
        vision_encoder_scheduler,
        "create_vision_encoder_scheduler",
        fake_create,
    )

    result = stages.create_vision_encoder_executor(
        "nvidia/Cosmos3-Nano",
        revision="cosmos-revision",
        device="cuda:1",
        dtype="bfloat16",
    )

    assert result is sentinel
    assert calls == [("nvidia/Cosmos3-Nano", "cosmos-revision", "cuda:1", "bfloat16")]


def test_text_factory_passes_through_tensor_parallelism(
    monkeypatch,
) -> None:
    captured: dict[str, object] = {}
    server_args = SimpleNamespace()
    scheduler = object()

    def fake_build(model_path: str, context_length: int, **kwargs):
        captured.update(
            model_path=model_path,
            context_length=context_length,
            kwargs=kwargs,
        )
        return server_args

    def fake_create(received_server_args, gpu_id: int, **kwargs):
        captured.update(
            server_args=received_server_args,
            gpu_id=gpu_id,
            scheduler_kwargs=kwargs,
        )
        return scheduler

    def fake_validate(**kwargs):
        captured["validation_kwargs"] = kwargs

    monkeypatch.setattr(stages, "build_sglang_server_args", fake_build)
    monkeypatch.setattr(stages, "create_thinker_scheduler", fake_create)
    monkeypatch.setattr(stages, "validate_generation_batch_policy", fake_validate)

    result = stages.create_sglang_text_executor_from_config(
        "nvidia/Cosmos3-Nano",
        max_seq_len=4096,
        gpu_id=2,
        tp_rank=1,
        tp_size=2,
        revision="cosmos-revision",
        server_args_overrides={"max_running_requests": 8},
    )

    assert result is scheduler
    assert captured["model_path"] == "nvidia/Cosmos3-Nano"
    assert captured["context_length"] == 4096
    assert captured["kwargs"] == {
        "max_running_requests": 8,
        "cuda_graph_max_bs": 8,
        "torch_compile_max_bs": 8,
        "cuda_graph_bs": [1, 2, 4, 8],
        "disable_cuda_graph": False,
        "sampling_backend": "pytorch",
        "revision": "cosmos-revision",
        "tp_size": 2,
    }
    assert captured["validation_kwargs"] == {
        "model_name": "Cosmos3 thinker",
        "server_args": server_args,
    }
    assert captured["server_args"] is server_args
    assert captured["gpu_id"] == 2
    assert captured["scheduler_kwargs"]["tp_rank"] == 1
