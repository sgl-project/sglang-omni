# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace

import pytest

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


def test_text_factory_builds_single_rank_language_only_server_args(
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

    monkeypatch.setattr(
        "sglang_omni.scheduling.sglang_backend.build_sglang_server_args",
        fake_build,
    )
    monkeypatch.setattr(
        "sglang_omni.models.cosmos3.bootstrap.create_text_scheduler",
        fake_create,
    )

    result = stages.create_sglang_text_executor_from_config(
        "nvidia/Cosmos3-Nano",
        thinker_max_seq_len=4096,
        gpu_id=2,
    )

    assert result is scheduler
    assert captured["model_path"] == "nvidia/Cosmos3-Nano"
    assert captured["context_length"] == 4096
    assert captured["kwargs"] == {
        "tp_size": 1,
        "enable_multimodal": False,
        "language_only": True,
        "sampling_backend": "pytorch",
    }
    assert captured["server_args"] is server_args
    assert captured["gpu_id"] == 2


def test_text_factory_rejects_parallelism() -> None:
    with pytest.raises(ValueError, match="tp_size=1"):
        stages.create_sglang_text_executor_from_config(
            "nvidia/Cosmos3-Nano",
            tp_size=2,
        )
