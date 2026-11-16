# SPDX-License-Identifier: Apache-2.0
"""CPU-only cleanup and segmentation contracts for Ming 3 speech text."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from sglang_omni.models.ming3_audio.text_processing import (
    DEFAULT_MAX_FRAGMENT_CHARS,
    DEFAULT_MAX_FRAGMENTS,
    DEFAULT_MAX_SEMANTIC_LENGTH,
    TextPlanningError,
    TextTooLargeError,
    TextUnbreakableError,
    plan_speech_text,
    tokenize_text,
    validate_text_limits,
)

FIXTURE_ROOT = Path(__file__).parent / "fixtures"
FINAL_ANCHOR = "成为全球数字生态的重要参与者。"
GOLDEN_SECTION_BOUNDARIES = (
    "1. 基本概况\n",
    "2. 核心业务\n",
    "3. 发展历程与里程碑\n",
    "4. 社会价值与公益\n",
    "5. 全球影响力\n",
    "6. 近期动态\n",
)
GOLDEN_FRAGMENT_LENGTHS = (
    46,
    8,
    48,
    18,
    33,
    8,
    44,
    17,
    62,
    15,
    50,
    12,
    49,
    33,
    11,
    57,
    32,
    9,
    65,
    8,
    52,
    47,
)
GOLDEN_FRAGMENT_SHA256S = (
    "ee8d8acee0f5a6e9f3fea76710db0f2f53639c689c8bafde917b600f592777db",
    "2b469cbed90b3b5de3922498dc1720174038785ab02cc9ed362e556dc86cc356",
    "df11325c52eeff82c850aa3085b597a7f122c0c07e406233f97ec9d75cabed26",
    "3c6f07134a58aedeae73ad85f02bb7a492266c419e28d2f7be3c478d34902c9b",
    "86f1390e174a27d70aec92e6690d1e68b228213b12f2605619c5a6fee97d641a",
    "cdfb098905d1b83238251a46ee5c48ac4c800b968177640393e751e7bee4a3cb",
    "e95a607690189b14c3998ad9417538300a7601bf5f595580fca32115d22f27a4",
    "d02b0308d48eec1cc1232a234e45a540e38a3fffe6f2e755867abc518231f742",
    "e661679c4c02876f43665faebaae4b3a609d9249a22083dc0e6f5d94b4ee3e3a",
    "79818c755135c0e459c3043d9200118b0626e0a171b0f9f56840c5413b8fd274",
    "a5c0fb0d10d13c89159f7e97d64910479ec5aeb098a1906703ee6f2288772b7a",
    "fddf33f8092de40fda24b46364b71e88391f1d8d463dd2892769221cb07b96ca",
    "daeda80d423bd677dae693515ae2f328d07937a0ff54e04c14daa8ef63574007",
    "0dcabf5c5f147701f839004dc5c8cd3be60d0789d6c704647cf34e0bd0b3ad06",
    "303e56aecd5306f2a7aa1fbe36048f16254c2744a5602508b4c874db27528531",
    "7143d7b41dcc50e63af42bb931e21f1277b099a7c61c3527e8ac7bc1f19a48e5",
    "8908f63b9fc65efef7b43ae96a6f45ca4f3161c9b90aec819314b2723f13682b",
    "ea3d449cd477a883c91fd1fc5f8d15e3190233b49881e5616377ca37c1e73d9d",
    "24a6cd737f8aa0cd2f099376d35cf41acbb86fb802d1b51940081e7affdbc1f1",
    "a6e495411881be52dfcd9d11ba613322d8b2417e2091f49733dd8f64cd1b3c2a",
    "caa7e86aa831814b8f54ad6271ed7dc17120278ebe69af6f5200f7bc054785f7",
    "bc83d2d212f1851cf42170e1cc1735212400bcbf758630a29291af94a7281a68",
)


def fixture_text(name: str) -> str:
    return (FIXTURE_ROOT / name).read_text(encoding="utf-8").rstrip("\n")


def meaningful_text(text: str) -> str:
    return "".join(character for character in text if not character.isspace())


def assert_default_fragment_limits(fragments: tuple[str, ...]) -> None:
    for fragment in fragments:
        assert 0 < len(fragment) <= DEFAULT_MAX_FRAGMENT_CHARS
        assert sum(token.semantic_length for token in tokenize_text(fragment)) <= (
            DEFAULT_MAX_SEMANTIC_LENGTH
        )


def test_validate_text_limits_remains_a_public_compatibility_helper() -> None:
    validate_text_limits("spoken text", 16, 16, 1, 16)

    with pytest.raises(TextTooLargeError):
        validate_text_limits("spoken text is too long", 16, 16, 1, 16)


@pytest.mark.parametrize(
    "raw_text",
    [
        "明天早上八点在会议室见。",
        " 明天早上八点在会议室见。\n",
        "Line one\r\nLine two.",
    ],
)
def test_short_plain_text_is_byte_for_byte_unchanged(raw_text: str) -> None:
    plan = plan_speech_text(raw_text)

    assert plan.cleaned_text == raw_text
    assert plan.fragments == (raw_text,)


@pytest.mark.parametrize(
    ("raw_text", "cleaned_text"),
    [
        (
            "### 标题\n- **加粗**文本和[链接](https://example.com/a_b)。",
            "标题\n加粗文本和链接。",
        ),
        (
            "第一段。\n\n第二段，含有、顿号。",
            "第一段。\n\n第二段，含有、顿号。",
        ),
        ("1. 第一项\n2. 第二项", "1. 第一项\n2. 第二项"),
        ("![图片说明](https://example.com/image.png)", "图片说明"),
        ("```python\nprint('hello')\n```", "print('hello')"),
        ("~~删除线~~和 _强调_", "删除线和 强调"),
    ],
)
def test_markdown_cleanup_retains_speakable_content(
    raw_text: str, cleaned_text: str
) -> None:
    plan = plan_speech_text(raw_text)

    assert plan.cleaned_text == cleaned_text
    assert "".join(plan.fragments) == cleaned_text


def test_nested_bullets_retain_order_and_content() -> None:
    raw_text = "### 1. 标题\n- 一级\n  - 二级\n    - 三级"

    plan = plan_speech_text(raw_text, max_semantic_length=2)

    assert plan.cleaned_text == "1. 标题\n一级\n  二级\n    三级"
    assert meaningful_text("".join(plan.fragments)) == meaningful_text(
        plan.cleaned_text
    )
    assert [fragment.strip() for fragment in plan.fragments] == [
        "1.",
        "标题",
        "一级",
        "二级",
        "三级",
    ]


def test_numbers_abbreviations_words_and_bare_url_are_safe() -> None:
    raw_text = (
        "3.14 7.5% 2023-10-20 -3 v1.2 1,234.56 "
        "U.S. Dr. Lee OceanBase https://example.com/a_b?q=1#part."
    )

    plan = plan_speech_text(raw_text, max_semantic_length=8, max_fragment_chars=40)

    assert "".join(plan.fragments) == plan.cleaned_text
    assert "https://example.com/a_b?q=1#part" not in plan.cleaned_text
    assert plan.cleaned_text.endswith("OceanBase link.")
    assert any("3.14" in fragment for fragment in plan.fragments)
    assert any("7.5%" in fragment for fragment in plan.fragments)
    assert any("2023-10-20" in fragment for fragment in plan.fragments)
    assert any("-3" in fragment for fragment in plan.fragments)
    assert any("v1.2" in fragment for fragment in plan.fragments)
    assert any("1,234.56" in fragment for fragment in plan.fragments)
    assert any(
        "U.S." in fragment and "Dr. Lee" in fragment for fragment in plan.fragments
    )
    assert any("OceanBase" in fragment for fragment in plan.fragments)
    assert any("link." in fragment for fragment in plan.fragments)


def test_ipv6_urls_are_cleaned_and_valid_markdown_links_are_accepted() -> None:
    raw_url = "https://[2001:db8::1]:8443/api"

    bare_url_plan = plan_speech_text(f"See {raw_url}", max_semantic_length=2)
    markdown_link_plan = plan_speech_text(f"See [router](<{raw_url}>)")

    assert bare_url_plan.cleaned_text == "See link"
    assert markdown_link_plan.cleaned_text == "See router"
    assert [token.text for token in tokenize_text(raw_url)] == [raw_url]


def test_version_token_is_atomic() -> None:
    plan = plan_speech_text("v1.2", max_semantic_length=1, max_fragment_chars=4)

    assert plan.fragments == ("v1.2",)


def test_issue_corpus_is_frozen_and_cleanup_matches_no_markdown_evidence() -> None:
    """Pin the issue corpus and require review before changing fragment hashes."""
    markdown_text = fixture_text("issue_ant_group_markdown.txt")
    no_markdown_text = fixture_text("issue_ant_group_no_markdown.txt")

    assert len(markdown_text) == 882
    assert len(no_markdown_text) == 724
    assert (
        hashlib.sha256(no_markdown_text.encode("utf-8")).hexdigest()
        == "3bf472e45ea5a36fafc51b19dbd2f1f0e1736728d65848975c1626e830a73598"
    )

    markdown_plan = plan_speech_text(markdown_text)
    no_markdown_plan = plan_speech_text(no_markdown_text)

    assert markdown_plan.cleaned_text == no_markdown_text
    assert "".join(markdown_plan.fragments) == markdown_plan.cleaned_text
    assert no_markdown_plan.cleaned_text == no_markdown_text
    assert "".join(no_markdown_plan.fragments) == no_markdown_text
    assert markdown_plan.fragments == tuple(
        fragment for fragment in markdown_plan.fragments if fragment.strip()
    )
    assert no_markdown_plan.fragments == tuple(
        fragment for fragment in no_markdown_plan.fragments if fragment.strip()
    )
    assert markdown_plan.fragments == no_markdown_plan.fragments
    assert len(markdown_plan.fragments) == len(GOLDEN_FRAGMENT_LENGTHS)
    assert 1 < len(markdown_plan.fragments) <= DEFAULT_MAX_FRAGMENTS
    assert 1 < len(no_markdown_plan.fragments) <= DEFAULT_MAX_FRAGMENTS
    assert tuple(map(len, markdown_plan.fragments)) == GOLDEN_FRAGMENT_LENGTHS
    assert (
        tuple(
            hashlib.sha256(fragment.encode("utf-8")).hexdigest()
            for fragment in markdown_plan.fragments
        )
        == GOLDEN_FRAGMENT_SHA256S
    )
    assert all(
        0 < len(fragment) <= DEFAULT_MAX_FRAGMENT_CHARS
        for fragment in markdown_plan.fragments
    )
    assert all(
        0 < len(fragment) <= DEFAULT_MAX_FRAGMENT_CHARS
        for fragment in no_markdown_plan.fragments
    )
    assert markdown_plan.fragments[0].endswith("核心介绍：\n\n")
    assert all(
        boundary in markdown_plan.fragments and boundary in no_markdown_plan.fragments
        for boundary in GOLDEN_SECTION_BOUNDARIES
    )
    assert markdown_plan.cleaned_text.startswith("蚂蚁集团（Ant Group）")
    assert "OceanBase" in markdown_plan.cleaned_text
    assert markdown_plan.cleaned_text.endswith(FINAL_ANCHOR)
    assert markdown_plan.fragments[-1].endswith(FINAL_ANCHOR)
    assert no_markdown_plan.fragments[-1].endswith(FINAL_ANCHOR)
    for fragment in markdown_plan.fragments + no_markdown_plan.fragments:
        assert 0 < len(fragment) <= DEFAULT_MAX_FRAGMENT_CHARS
        assert sum(token.semantic_length for token in tokenize_text(fragment)) <= 48


def test_long_text_is_deterministic_and_bounded() -> None:
    raw_text = "明天早上八点在会议室见。" * 64

    first_plan = plan_speech_text(raw_text)
    second_plan = plan_speech_text(raw_text)

    assert first_plan == second_plan
    assert len(first_plan.fragments) > 1
    assert len(first_plan.fragments) <= DEFAULT_MAX_FRAGMENTS
    assert all(
        0 < len(fragment) <= DEFAULT_MAX_FRAGMENT_CHARS
        for fragment in first_plan.fragments
    )
    assert all(fragment.strip() for fragment in first_plan.fragments)
    assert "".join(first_plan.fragments) == raw_text


@pytest.mark.parametrize("raw_text", ["", "   \n\t", "。。。、！？", "😀🤝✨"])
def test_empty_or_unpronounceable_text_fails_closed(raw_text: str) -> None:
    with pytest.raises(TextPlanningError):
        plan_speech_text(raw_text)


def test_overlong_bare_url_is_replaced_with_a_short_spoken_token() -> None:
    raw_text = "Before https://" + "a" * (DEFAULT_MAX_FRAGMENT_CHARS * 2) + " after."

    plan = plan_speech_text(raw_text)

    assert plan.cleaned_text == "Before link after."
    assert "".join(plan.fragments) == plan.cleaned_text


def test_overlong_unbreakable_word_is_rejected() -> None:
    raw_text = "a" * (DEFAULT_MAX_FRAGMENT_CHARS + 1)

    with pytest.raises(TextUnbreakableError):
        plan_speech_text(raw_text)


def test_input_and_fragment_bounds_are_rejected() -> None:
    with pytest.raises(TextTooLargeError):
        plan_speech_text("a" * 20, max_input_chars=10)
    with pytest.raises(TextTooLargeError):
        plan_speech_text("你好。" * 20, max_semantic_length=1, max_fragments=2)


def test_default_fragment_cap_rejects_the_thirty_third_fragment() -> None:
    with pytest.raises(TextTooLargeError, match="planned 33 fragments"):
        plan_speech_text("中。" * 33, max_semantic_length=1, max_fragment_chars=2)


@pytest.mark.parametrize(
    "raw_text",
    [
        "<b>unsupported HTML</b>",
        "[reference][id]",
        "```unclosed\ncontent",
        "`unclosed inline code",
        "Read <https://example.com/unclosed",
    ],
)
def test_unsupported_formats_fail_closed(raw_text: str) -> None:
    with pytest.raises(TextPlanningError):
        plan_speech_text(raw_text)


def test_private_use_and_surrogate_codepoints_fail_closed() -> None:
    for raw_text in ("a\ue001", "a\ue002", "a\uf8ff", "a\ud800"):
        with pytest.raises(TextPlanningError):
            plan_speech_text(raw_text)


def test_unknown_markdown_does_not_drop_visible_content() -> None:
    raw_text = "Unknown [syntax without destination and #inline markers"

    plan = plan_speech_text(raw_text)

    assert meaningful_text("".join(plan.fragments)) == meaningful_text(raw_text)


def test_natural_newline_boundaries_precede_greedy_filling() -> None:
    raw_text = "标题\n第一条内容包含足够多的中文字符。\n第二条内容也应独立发音。"

    plan = plan_speech_text(raw_text, max_semantic_length=16)

    assert plan.fragments == (
        "标题\n",
        "第一条内容包含足够多的中文字符。\n",
        "第二条内容也应独立发音。",
    )


def test_many_short_lines_are_grouped_without_losing_newlines() -> None:
    raw_text = "中\n" * 60

    plan = plan_speech_text(raw_text)

    assert "".join(plan.fragments) == raw_text
    assert len(plan.fragments) < 10
    assert all(fragment.strip() for fragment in plan.fragments)


def test_longer_lines_share_fragments_up_to_the_semantic_limit() -> None:
    raw_text = ("甲" * 20 + "\n") * 60

    plan = plan_speech_text(raw_text)

    assert len(plan.fragments) <= DEFAULT_MAX_FRAGMENTS
    assert "".join(plan.fragments) == raw_text
    assert all(fragment.endswith("\n") for fragment in plan.fragments)


@pytest.mark.parametrize(
    "raw_text",
    [
        ("甲" * 16 + "\n") * 40,
        ("### 标题\n" + "甲" * 16 + "。\n\n") * 40,
        ("甲" * 16 + "。\n\n") * 40,
    ],
    ids=("single-newlines", "headings", "meaningful-paragraphs"),
)
def test_structural_boundaries_do_not_force_overfragmentation(raw_text: str) -> None:
    plan = plan_speech_text(raw_text)

    assert "".join(plan.fragments) == plan.cleaned_text
    assert len(plan.fragments) <= DEFAULT_MAX_FRAGMENTS
    assert all(fragment.strip() for fragment in plan.fragments)
    assert_default_fragment_limits(plan.fragments)


def test_short_paragraphs_fill_fragments_without_losing_text() -> None:
    cleaned_text = "甲\n\n" * 40

    plan = plan_speech_text(cleaned_text)

    assert plan.cleaned_text == cleaned_text
    assert "".join(plan.fragments) == cleaned_text
    assert len(plan.fragments) <= DEFAULT_MAX_FRAGMENTS
    assert_default_fragment_limits(plan.fragments)


def test_short_sentences_fill_fragments_without_losing_text() -> None:
    cleaned_text = ("甲" * 16 + "。") * 40

    plan = plan_speech_text(cleaned_text)

    assert plan.cleaned_text == cleaned_text
    assert "".join(plan.fragments) == cleaned_text
    assert len(plan.fragments) <= DEFAULT_MAX_FRAGMENTS
    assert_default_fragment_limits(plan.fragments)


def test_coalescing_obeys_a_custom_fragment_count_cap() -> None:
    cleaned_text = "甲\n\n" * 6

    plan = plan_speech_text(
        cleaned_text,
        max_semantic_length=6,
        max_fragment_chars=10,
        max_fragments=2,
    )

    assert "".join(plan.fragments) == cleaned_text
    assert len(plan.fragments) == 2
    assert all(len(fragment) <= 10 for fragment in plan.fragments)
    assert all(
        sum(token.semantic_length for token in tokenize_text(fragment)) <= 6
        for fragment in plan.fragments
    )


def test_coalescing_keeps_protected_tokens_atomic() -> None:
    text = ("2023-10-20 v1.2 👩\u200d💻 甲。\n\n") * 40

    plan = plan_speech_text(text)

    assert "".join(plan.fragments) == plan.cleaned_text
    assert len(plan.fragments) <= DEFAULT_MAX_FRAGMENTS
    assert sum(fragment.count("2023-10-20") for fragment in plan.fragments) == 40
    assert sum(fragment.count("v1.2") for fragment in plan.fragments) == 40
    assert sum(fragment.count("👩\u200d💻") for fragment in plan.fragments) == 40
    assert_default_fragment_limits(plan.fragments)


def test_unicode_graphemes_are_not_split_across_fragments() -> None:
    raw_text = "甲👩\u200d💻👍🏽乙"

    plan = plan_speech_text(raw_text, max_semantic_length=2, max_fragment_chars=5)

    assert plan.fragments == ("甲👩\u200d💻", "👍🏽乙")
    assert "".join(plan.fragments) == raw_text


def test_keycap_emoji_stays_with_its_variation_selector_and_combining_mark() -> None:
    raw_text = "甲1\ufe0f\u20e3乙"

    plan = plan_speech_text(raw_text, max_semantic_length=1, max_fragment_chars=3)

    assert plan.fragments == ("甲", "1\ufe0f\u20e3", "乙")
    assert "".join(plan.fragments) == raw_text


def test_cjk_variation_selector_stays_with_its_base_character() -> None:
    raw_text = "甲禰\U000e0100乙"

    plan = plan_speech_text(raw_text, max_semantic_length=1, max_fragment_chars=3)

    assert plan.fragments == ("甲", "禰\U000e0100", "乙")
    assert "".join(plan.fragments) == raw_text


def test_non_ascii_combining_mark_stays_with_its_word() -> None:
    raw_text = "甲a\u093e乙"

    plan = plan_speech_text(raw_text, max_semantic_length=1, max_fragment_chars=3)

    assert plan.fragments == ("甲", "a\u093e", "乙")
    assert "".join(plan.fragments) == raw_text


@pytest.mark.parametrize(
    ("raw_text", "cleaned_text"),
    [
        (
            "Read https://example.com/path?a=1&b=2, then continue.",
            "Read link, then continue.",
        ),
        (
            "Read <https://example.com/a_(b)?x=1&y=(2)>.",
            "Read link.",
        ),
        (
            "Read [the nested docs](https://example.com/a_(b)?x=1&y=(2)) now.",
            "Read the nested docs now.",
        ),
        (
            'Read [the docs](https://example.com/a_(b) "official docs") now.',
            "Read the docs now.",
        ),
        (
            "Read [the docs](<https://example.com/docs> 'official docs') now.",
            "Read the docs now.",
        ),
        (
            "Read [the docs](https://example.com/docs (official docs)) now.",
            "Read the docs now.",
        ),
        (
            "Read [the [nested] docs](https://example.com/docs) now.",
            "Read the [nested] docs now.",
        ),
        (
            "![diagram](https://example.com/image_(1).png) is shown.",
            "diagram is shown.",
        ),
        (
            "`https://example.com/inline` and\n```text\nhttps://example.com/code\n```",
            "link and\nlink",
        ),
    ],
)
def test_markdown_and_url_cleanup_preserves_audio_context(
    raw_text: str, cleaned_text: str
) -> None:
    plan = plan_speech_text(raw_text)

    assert plan.cleaned_text == cleaned_text
    assert "".join(plan.fragments) == cleaned_text


def test_malformed_nested_markdown_link_fails_closed() -> None:
    with pytest.raises(TextPlanningError, match="malformed Markdown link"):
        plan_speech_text("Read [the docs](https://example.com/a_(b)")


def test_malformed_markdown_link_title_fails_closed() -> None:
    with pytest.raises(TextPlanningError, match="malformed Markdown link title"):
        plan_speech_text('Read [the docs](https://example.com "missing close).')


def test_nested_markdown_link_labels_beyond_supported_depth_fail_closed() -> None:
    with pytest.raises(TextPlanningError, match="malformed Markdown link"):
        plan_speech_text("Read [a [b [c]]](https://example.com) now.")


def test_shorter_code_fence_does_not_close_a_longer_fence() -> None:
    raw_text = "````python\nbefore\n```\nafter\n````"

    plan = plan_speech_text(raw_text)

    assert plan.cleaned_text == "before\n```\nafter"
    assert "".join(plan.fragments) == plan.cleaned_text
