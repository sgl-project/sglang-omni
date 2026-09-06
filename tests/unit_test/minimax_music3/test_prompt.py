# SPDX-License-Identifier: Apache-2.0
"""Prompt construction contract for MiniMax Music 3."""

from __future__ import annotations

import logging

import pytest

from sglang_omni.models.minimax_music3.prompt import (
    build_prompt,
    clean_caption,
    dropped_lyric_lines,
    normalize_lyrics,
)


@pytest.mark.parametrize(
    "lyrics, expected",
    [
        pytest.param("[Verse] Walking down the street", [1], id="tag-then-words"),
        pytest.param("  [Verse]\tWalking down", [1], id="leading-and-inner-space"),
        pytest.param("[Intro] [Verse] hello", [1], id="two-tags-then-words"),
        pytest.param("[Verse]\nWalking down the street", [], id="tag-on-its-own-line"),
        pytest.param("[Verse]   ", [], id="tag-with-trailing-space"),
        pytest.param("Walking down the street", [], id="no-tag"),
        pytest.param("[Verse] one\n[Chorus]\ntwo\n[Bridge] three", [1, 4], id="mixed"),
    ],
)
def test_dropped_lyric_lines_finds_words_next_to_a_tag(lyrics, expected) -> None:
    assert dropped_lyric_lines(lyrics) == expected


def test_build_prompt_warns_about_the_words_it_drops(caplog) -> None:
    """The cookbook documents this loss; it should not be silent.

    See docs/cookbook/minimax_music3.md, "Put a tag on its own line".
    """
    with caplog.at_level(logging.WARNING):
        prompt = build_prompt("Bright J-pop", "[Verse] Walking down the street")

    assert "Walking down the street" not in prompt
    assert len(caplog.records) == 1
    message = caplog.records[0].getMessage()
    assert "line 1" in message
    # The warning must not echo the lyrics themselves into the server log.
    assert "Walking down the street" not in message


def test_build_prompt_stays_quiet_when_nothing_is_dropped(caplog) -> None:
    with caplog.at_level(logging.WARNING):
        prompt = build_prompt("Bright J-pop", "[Verse]\nWalking down the street")

    assert "Walking down the street" in prompt
    assert caplog.records == []


def test_normalize_lyrics_matches_the_documented_examples() -> None:
    """Both forms from the cookbook table, unchanged by this PR."""
    assert (
        normalize_lyrics("[Verse]\nWalking down the street")
        == "[start]\n[verse]\nWalking down the street"
    )
    assert normalize_lyrics("[Verse] Walking down the street") == "[start]\n[verse]"


def test_normalize_lyrics_splits_tags_onto_their_own_lines() -> None:
    assert normalize_lyrics("first line [chorus]") == "[start]\nfirst line\n[chorus]"
    assert normalize_lyrics("one ^ two") == "[start]\none\ntwo"


def test_clean_caption_rewrites_special_tags_and_strips_markdown() -> None:
    assert clean_caption("<|genre rock|> with drums") == "genre is rock with drums"
    assert clean_caption("A **bright** J-pop track") == "A bright J-pop track"
    assert clean_caption("- lofi\n- hiphop") == "lofi\nhiphop"
    assert clean_caption("line1\n\n\nline2") == "line1\nline2"


def test_build_prompt_frames_the_caption_and_lyrics() -> None:
    prompt = build_prompt("Bright J-pop", "[Verse]\nWalking down the street")

    assert prompt.startswith("<|im_start|><|caption_start|>Bright J-pop<|caption_end|>")
    assert prompt.endswith("<|lyrics_end|><|im_end|><|audio_start|>")
