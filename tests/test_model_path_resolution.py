# SPDX-License-Identifier: Apache-2.0
"""Regression tests for Qwen3 preprocessor recovery and encoder cache wiring."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import torch

from sglang_omni.models.qwen3_omni.components import preprocessor
from sglang_omni.models.qwen3_omni.pipeline import stages


def test_qwen3_preprocessor_recovers_missing_preprocessor_config(
    monkeypatch, tmp_path: Path
) -> None:
    model_dir = tmp_path / "snapshot"
    model_dir.mkdir()

    calls: list[Path] = []

    class DummyProcessorFactory:
        @classmethod
        def from_pretrained(cls, model_path: str, **kwargs):
            path = Path(model_path)
            calls.append(path)
            if not (path / "preprocessor_config.json").exists():
                raise OSError("missing preprocessor_config.json")
            return SimpleNamespace(
                tokenizer=SimpleNamespace(chat_template="dummy-template")
            )

    def fake_resolve_model_path(model_path: str, *, local_files_only: bool = False):
        return model_dir

    def fake_hf_hub_download(repo_id: str, *, filename: str) -> str:
        cfg_path = model_dir / filename
        cfg_path.write_text("{}", encoding="utf-8")
        return str(cfg_path)

    monkeypatch.setattr(preprocessor, "resolve_model_path", fake_resolve_model_path)
    monkeypatch.setattr(preprocessor, "hf_hub_download", fake_hf_hub_download)
    monkeypatch.setattr(preprocessor, "Qwen3OmniMoeProcessor", DummyProcessorFactory)

    proc = preprocessor.Qwen3OmniPreprocessor("Qwen/Qwen3-Omni-30B-A3B-Instruct")

    assert proc.processor.tokenizer.chat_template == "dummy-template"
    assert len(calls) == 2
    assert calls[0] == model_dir
    assert calls[1] == model_dir
    assert (model_dir / "preprocessor_config.json").exists()


def test_qwen3_encoder_executor_passes_cache_settings(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_create_single_pass_engine(model, **kwargs):
        captured["model"] = model
        captured.update(kwargs)
        return object()

    class DummyExecutor:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    monkeypatch.setattr(
        stages, "create_single_pass_engine", fake_create_single_pass_engine
    )
    monkeypatch.setattr(stages, "EngineExecutor", DummyExecutor)

    executor = stages._create_encoder_executor(
        stage_name="image",
        model=torch.nn.Identity(),
        device="cpu",
        use_cache=True,
        cache_size=7,
    )

    assert isinstance(executor, DummyExecutor)
    assert captured["device"] == "cpu"
    assert captured["use_cache"] is True
    assert captured["cache_size"] == 7
