# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace

from sglang_omni.preprocessing.text import ensure_chat_template


def test_ensure_chat_template_uses_remote_fallback(monkeypatch) -> None:
    calls: list[tuple[str, bool]] = []

    def fake_load_chat_template(
        model_path: str, *, local_files_only: bool = True
    ) -> str | None:
        calls.append((model_path, local_files_only))
        if model_path == "base-model":
            return "template"
        return None

    monkeypatch.setattr(
        "sglang_omni.preprocessing.text.load_chat_template",
        fake_load_chat_template,
    )
    tokenizer = SimpleNamespace(chat_template=None)

    ensure_chat_template(
        tokenizer,
        model_path="fp8-model",
        fallback_model_paths=("base-model",),
    )

    assert tokenizer.chat_template == "template"
    assert calls == [("fp8-model", True), ("base-model", False)]


def test_ensure_chat_template_does_not_fetch_when_template_exists(monkeypatch) -> None:
    def fail_load_chat_template(
        model_path: str, *, local_files_only: bool = True
    ) -> str | None:
        raise AssertionError("chat template should not be loaded")

    monkeypatch.setattr(
        "sglang_omni.preprocessing.text.load_chat_template",
        fail_load_chat_template,
    )
    tokenizer = SimpleNamespace(chat_template="existing")

    ensure_chat_template(
        tokenizer,
        model_path="fp8-model",
        fallback_model_paths=("base-model",),
    )

    assert tokenizer.chat_template == "existing"
