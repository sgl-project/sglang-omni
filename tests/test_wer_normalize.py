# SPDX-License-Identifier: Apache-2.0
"""Tests for Chinese inverse text normalization in WER benchmark.

Verifies that normalize_text correctly converts Chinese numerals to Arabic
numerals before character-level WER comparison, so that spoken-form ASR output
(e.g., 四百六十五) matches written-form reference text (e.g., 465).
"""

import sys
import warnings
from unittest.mock import MagicMock

import pytest

# benchmark_tts_wer.py has heavy module-level imports (torch, scipy, etc.)
# that normalize_text doesn't use. Stub them so we can import the function
# without installing the full benchmark stack.
for _mod in ["torch", "soundfile", "scipy", "scipy.signal", "aiohttp", "tqdm", "jiwer"]:
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()

from benchmarks.performance.tts.benchmark_tts_wer import normalize_text  # noqa: E402


@pytest.fixture(autouse=True)
def _suppress_cn2an_warnings():
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", module="cn2an")
        yield


# --- Group 1: Chinese ITN (new behavior) ---


@pytest.mark.parametrize(
    "spoken, written, expected",
    [
        ("四百六十五", "465", "4 6 5"),
        ("二零零二年一月二十八日", "2002年1月28日", "2 0 0 2 年 1 月 2 8 日"),
        ("一二三四五", "12345", "1 2 3 4 5"),
        ("十三点五", "13.5", "1 3 5"),
        ("百分之六点三", "6.3%", "6 3"),
        ("共四百六十五篇", "共465篇", "共 4 6 5 篇"),
        ("2002年", "2002年", "2 0 0 2 年"),
        ("你好世界", "你好世界", "你 好 世 界"),
        ("二〇二二年", "2022年", "2 0 2 2 年"),
        ("两百五十", "250", "2 5 0"),
        ("三分之一", "1/3", "1 3"),
    ],
    ids=[
        "cardinal",
        "date",
        "sequential_digits",
        "decimal",
        "percentage",
        "mixed_text",
        "already_arabic",
        "no_numbers",
        "circle_zero",
        "liang_variant",
        "fraction",
    ],
)
def test_zh_itn_spoken_matches_written(spoken, written, expected):
    """Spoken-form and written-form should normalize to the same output."""
    assert normalize_text(spoken, "zh") == expected
    assert normalize_text(written, "zh") == expected


# --- Group 2: Chinese regression tests (existing behavior preserved) ---


@pytest.mark.parametrize(
    "text, expected",
    [
        ("你好，世界！", "你 好 世 界"),
        ("你 好\u3000世界", "你 好 世 界"),
        ("，。！", ""),
    ],
    ids=["punctuation_stripped", "whitespace_stripped", "empty_after_strip"],
)
def test_zh_existing_behavior(text, expected):
    assert normalize_text(text, "zh") == expected


# --- Group 3: English path not broken ---


def test_en_unchanged():
    result = normalize_text("Hello World!", "en")
    assert "hello" in result
    assert "world" in result
