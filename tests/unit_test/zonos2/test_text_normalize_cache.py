# SPDX-License-Identifier: Apache-2.0
"""GPU-free tests for the text-normalization cache (StageOutputCache-backed).

normalize_text() memoizes (text, language) -> normalized string on the shared
StageOutputCache. These cover the model-local glue: language gating, a cache hit
skipping the NeMo normalizer, and distinct-input misses. The normalizer is stubbed,
so no NeMo/FST grammars are loaded.
"""

from sglang_omni.models.zonos2.components import text_frontend
from sglang_omni.models.zonos2.components.text_frontend import normalize_text


class _CountingNormalizer:
    def __init__(self) -> None:
        self.calls = 0

    def normalize(self, text: str, language: str) -> str:
        self.calls += 1
        return f"NORM[{text}]"


def _install(monkeypatch):
    text_frontend._NORMALIZE_CACHE.clear()
    norm = _CountingNormalizer()
    monkeypatch.setattr(text_frontend, "_get_normalizer", lambda: norm)
    return norm


def test_repeated_text_hits_cache(monkeypatch) -> None:
    norm = _install(monkeypatch)
    out1 = normalize_text("hello world", "en_us")
    out2 = normalize_text("hello world", "en_us")
    assert out1 == out2 == "NORM[hello world]"
    assert norm.calls == 1  # second lookup served from cache


def test_distinct_text_misses(monkeypatch) -> None:
    norm = _install(monkeypatch)
    normalize_text("one", "en_us")
    normalize_text("two", "en_us")
    assert norm.calls == 2


def test_unsupported_language_bypasses_normalizer(monkeypatch) -> None:
    norm = _install(monkeypatch)
    # Unknown server language / None returns text unchanged, normalizer untouched.
    assert normalize_text("hello", "xx") == "hello"
    assert normalize_text("hello", None) == "hello"
    assert norm.calls == 0
