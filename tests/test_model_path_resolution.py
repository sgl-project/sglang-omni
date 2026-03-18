# SPDX-License-Identifier: Apache-2.0
"""Regression tests for Qwen3 preprocessor recovery and encoder cache wiring."""

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

from sglang_omni.models.qwen3_omni.components import preprocessor


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


def test_qwen3_encoder_executor_forwards_cache_settings() -> None:
    stages_path = Path(__file__).resolve().parent.parent / "sglang_omni" / "models" / "qwen3_omni" / "pipeline" / "stages.py"
    tree = ast.parse(stages_path.read_text(encoding="utf-8"))

    target_fn = None
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "_create_encoder_executor":
            target_fn = node
            break

    assert target_fn is not None, "_create_encoder_executor() not found"

    engine_call = None
    for node in ast.walk(target_fn):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id == "create_single_pass_engine":
                engine_call = node
                break

    assert engine_call is not None, "create_single_pass_engine() call not found"

    keywords = {kw.arg: kw.value for kw in engine_call.keywords if kw.arg is not None}
    assert isinstance(keywords.get("use_cache"), ast.Name)
    assert keywords["use_cache"].id == "use_cache"
    assert isinstance(keywords.get("cache_size"), ast.Name)
    assert keywords["cache_size"].id == "cache_size"
