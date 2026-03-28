# SPDX-License-Identifier: Apache-2.0
"""Tests for Chinese text normalization in WER benchmark.

Verifies that normalize_text converts Arabic numerals to spoken-form Chinese
so that reference text aligns with ASR output (which is always spoken-form).
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


# --- Group 1: Arabic → spoken-form normalization ---
# Reference text with Arabic numerals should normalize to the same result
# as spoken-form Chinese (what ASR outputs).


@pytest.mark.parametrize(
    "written, spoken, expected",
    [
        ("465", "四百六十五", "四 百 六 十 五"),
        ("2002年1月28日", "二零零二年一月二十八日", "二 零 零 二 年 一 月 二 十 八 日"),
        ("13.5", "十三点五", "十 三 点 五"),
        ("6.3%", "百分之六点三", "百 分 之 六 点 三"),
        ("共465篇", "共四百六十五篇", "共 四 百 六 十 五 篇"),
        ("2022年", "二零二二年", "二 零 二 二 年"),
        ("250", "二百五十", "二 百 五 十"),
        ("25公里", "二十五公里", "二 十 五 公 里"),
        ("1/3", "三分之一", "三 分 之 一"),
    ],
    ids=[
        "cardinal",
        "date",
        "decimal",
        "percentage",
        "mixed_text",
        "year",
        "round_number",
        "with_unit",
        "fraction",
    ],
)
def test_zh_arabic_matches_spoken(written, spoken, expected):
    """Arabic-numeral reference and spoken-form ASR should normalize identically."""
    assert normalize_text(written, "zh") == expected
    assert normalize_text(spoken, "zh") == expected


# --- Group 2: Spoken-form passthrough (seed-tts-eval zh dataset uses this) ---


@pytest.mark.parametrize(
    "text, expected",
    [
        ("导航开始，全程二十五公里，预计需要十二分钟。", "导 航 开 始 全 程 二 十 五 公 里 预 计 需 要 十 二 分 钟"),
        ("一点五公里后，右转驶入京藏高速辅路。", "一 点 五 公 里 后 右 转 驶 入 京 藏 高 速 辅 路"),
        ("你好世界", "你 好 世 界"),
    ],
    ids=["navigation_distance", "navigation_decimal", "no_numbers"],
)
def test_zh_spoken_form_passthrough(text, expected):
    """Spoken-form text (no Arabic digits) passes through unchanged."""
    assert normalize_text(text, "zh") == expected


# --- Group 3: Existing behavior preserved ---


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


# --- Group 4: English path not broken ---


def test_en_unchanged():
    result = normalize_text("Hello World!", "en")
    assert "hello" in result
    assert "world" in result
