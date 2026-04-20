# SPDX-License-Identifier: Apache-2.0
"""Tests for Ming thinker tokenizer loading."""

from __future__ import annotations

from sglang_omni.models.ming_omni.components import common


def test_load_ming_tokenizer_falls_back_to_preview_repo(monkeypatch):
    sentinel = object()
    calls: list[str] = []

    def fake_auto_from_pretrained(path: str, trust_remote_code: bool = False, **kwargs):
        calls.append(f"auto:{path}")
        raise OSError("root tokenizer not found")

    def fake_fast_from_pretrained(path: str, **kwargs):
        calls.append(f"fast:{path}")
        if path == common._TOKENIZER_FALLBACK:
            return sentinel
        raise OSError("root tokenizer not found")

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
    assert calls == [
        "auto:org/model",
        "fast:org/model",
        f"fast:{common._TOKENIZER_FALLBACK}",
    ]
