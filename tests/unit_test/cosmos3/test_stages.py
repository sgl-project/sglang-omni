# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace

from sglang_omni.models.cosmos3 import stages
from sglang_omni.scheduling.simple_scheduler import SimpleScheduler


def test_preprocessing_factory_returns_simple_scheduler(monkeypatch) -> None:
    sentinel = object()
    calls: list[tuple[str, int | None]] = []

    def fake_preprocessor(model_path: str, max_seq_len: int | None):
        calls.append((model_path, max_seq_len))
        return sentinel

    monkeypatch.setattr(stages, "Cosmos3TextPreprocessor", fake_preprocessor)

    scheduler = stages.create_preprocessing_executor(
        "nvidia/Cosmos3-Nano",
        thinker_max_seq_len=8192,
    )

    assert isinstance(scheduler, SimpleScheduler)
    assert scheduler._fn is sentinel
    assert calls == [("nvidia/Cosmos3-Nano", 8192)]


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

    monkeypatch.setattr(stages, "build_sglang_server_args", fake_build)
    monkeypatch.setattr(stages, "create_thinker_scheduler", fake_create)

    result = stages.create_sglang_text_executor_from_config(
        "nvidia/Cosmos3-Nano",
        thinker_max_seq_len=4096,
        gpu_id=2,
        tp_rank=1,
        tp_size=2,
        server_args_overrides={"disable_cuda_graph": True},
    )

    assert result is scheduler
    assert captured["model_path"] == "nvidia/Cosmos3-Nano"
    assert captured["context_length"] == 4096
    assert captured["kwargs"] == {
        "disable_cuda_graph": True,
        "tp_size": 2,
    }
    assert captured["server_args"] is server_args
    assert captured["gpu_id"] == 2
    assert captured["scheduler_kwargs"]["tp_rank"] == 1
