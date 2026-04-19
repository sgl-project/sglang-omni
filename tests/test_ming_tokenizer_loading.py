# SPDX-License-Identifier: Apache-2.0
"""Tests for Ming tokenizer loading subdirectory behavior."""

from __future__ import annotations

from pathlib import Path

from sglang_omni.models.ming_omni.components import common


def test_load_ming_tokenizer_probes_hub_subdir_without_full_snapshot(
    monkeypatch, tmp_path: Path
):
    sentinel = object()

    def fail_resolve_model_path(*args, **kwargs):
        raise AssertionError(
            "resolve_model_path should not be used for tokenizer probes"
        )

    def fake_hf_hub_download(repo_id: str, filename: str, subfolder: str | None = None):
        assert repo_id == "org/model"
        assert subfolder == "talker/llm"
        if filename != "tokenizer.json":
            raise FileNotFoundError(filename)
        cached_file = tmp_path / "hub" / "snap" / "talker" / "llm" / filename
        cached_file.parent.mkdir(parents=True, exist_ok=True)
        cached_file.write_text("{}", encoding="utf-8")
        return str(cached_file)

    def fake_auto_from_pretrained(path: str, trust_remote_code: bool = False, **kwargs):
        if path == "org/model":
            raise OSError("root tokenizer not found")
        if path.endswith("talker/llm"):
            return sentinel
        raise OSError(path)

    def fake_fast_from_pretrained(path: str, **kwargs):
        if path == "org/model":
            raise OSError("root tokenizer not found")
        raise OSError(path)

    monkeypatch.setattr(common, "resolve_model_path", fail_resolve_model_path)
    monkeypatch.setattr(common, "hf_hub_download", fake_hf_hub_download)

    import transformers

    monkeypatch.setattr(
        transformers.AutoTokenizer, "from_pretrained", fake_auto_from_pretrained
    )
    monkeypatch.setattr(
        transformers.PreTrainedTokenizerFast,
        "from_pretrained",
        fake_fast_from_pretrained,
    )

    tokenizer = common.load_ming_tokenizer("org/model")
    assert tokenizer is sentinel


def test_load_ming_tokenizer_prefers_local_subdir_without_hub_probe(
    monkeypatch, tmp_path: Path
):
    sentinel = object()
    model_dir = tmp_path / "local_model"
    tokenizer_dir = model_dir / "talker" / "llm"
    tokenizer_dir.mkdir(parents=True)
    (tokenizer_dir / "tokenizer.json").write_text("{}", encoding="utf-8")

    def fail_resolve_model_path(*args, **kwargs):
        raise AssertionError(
            "resolve_model_path should not be used for tokenizer probes"
        )

    def fail_hf_hub_download(*args, **kwargs):
        raise AssertionError("hf_hub_download should not be called for local paths")

    def fake_auto_from_pretrained(path: str, trust_remote_code: bool = False, **kwargs):
        if path == str(model_dir):
            raise OSError("root tokenizer not found")
        if path == str(tokenizer_dir):
            return sentinel
        raise OSError(path)

    def fake_fast_from_pretrained(path: str, **kwargs):
        if path == str(model_dir):
            raise OSError("root tokenizer not found")
        raise OSError(path)

    monkeypatch.setattr(common, "resolve_model_path", fail_resolve_model_path)
    monkeypatch.setattr(common, "hf_hub_download", fail_hf_hub_download)

    import transformers

    monkeypatch.setattr(
        transformers.AutoTokenizer, "from_pretrained", fake_auto_from_pretrained
    )
    monkeypatch.setattr(
        transformers.PreTrainedTokenizerFast,
        "from_pretrained",
        fake_fast_from_pretrained,
    )

    tokenizer = common.load_ming_tokenizer(str(model_dir))
    assert tokenizer is sentinel
