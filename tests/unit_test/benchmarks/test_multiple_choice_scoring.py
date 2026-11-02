# SPDX-License-Identifier: Apache-2.0
"""Contracts for scoring multiple-choice answers in the MMMU and video benchmarks."""

from __future__ import annotations

import pytest

from benchmarks.benchmarker.data import RequestResult
from benchmarks.dataset.mmmu import MMMUSample
from benchmarks.dataset.videomme import VideoMMESample
from benchmarks.metrics.mmmu import compute_mmmu_metrics
from benchmarks.metrics.video import compute_videomme_metrics
from benchmarks.tasks.video_understanding import build_videomme_result_records
from benchmarks.tasks.visual_understand import (
    build_mmmu_result_records,
    parse_multi_choice_response,
)

CHOICES = ["A", "B", "C", "D"]
OPTIONS = {"A": "pine", "B": "oak", "C": "maple", "D": "birch"}
UNPARSEABLE_TEXT = "the reasoning stops before it names"


def mmmu_sample(sample_id: str, answer: str) -> MMMUSample:
    return MMMUSample(
        sample_id=sample_id,
        question="which tree",
        options=list(OPTIONS.values()),
        answer=answer,
        image_data_uris=(),
        subject="Biology",
        prompt="which tree",
        all_choices=CHOICES,
        index2ans=OPTIONS,
    )


def video_sample(sample_id: str, answer: str) -> VideoMMESample:
    return VideoMMESample(
        sample_id=sample_id,
        video_path="clip.mp4",
        question="which tree",
        options=list(OPTIONS.values()),
        answer=answer,
        all_choices=CHOICES,
        index2ans=OPTIONS,
    )


def answered(text: str) -> RequestResult:
    return RequestResult(text=text, is_success=True)


def test_unparseable_answer_names_no_option() -> None:
    """An answer that names no option parses to no choice, not a drawn letter."""
    assert parse_multi_choice_response(UNPARSEABLE_TEXT, CHOICES, OPTIONS) == ("", True)


@pytest.mark.parametrize(
    ("text", "expected_choice"),
    [
        ("the last tree in the clip\nAnswer: d", "D"),
        ("Answer: a birch, the tall tree here", "D"),
    ],
)
def test_lowercase_answer_letter_names_its_option(
    text: str, expected_choice: str
) -> None:
    """A lowercase answer letter that no word follows names its option; an article does not."""
    assert parse_multi_choice_response(text, CHOICES, OPTIONS) == (
        expected_choice,
        False,
    )


@pytest.mark.parametrize("earlier_text", ["Answer: A", UNPARSEABLE_TEXT])
def test_unparseable_mmmu_answer_is_wrong_whatever_came_before(
    earlier_text: str,
) -> None:
    """An MMMU answer that names no option scores wrong whatever the earlier answers were."""
    records = build_mmmu_result_records(
        [mmmu_sample("first", "A"), mmmu_sample("second", "C")],
        [answered(earlier_text), answered(UNPARSEABLE_TEXT)],
    )

    second = records[1]
    assert (second["predicted"], second["is_correct"], second["is_unparseable"]) == (
        "",
        False,
        True,
    )


def test_mmmu_summary_counts_unparseable() -> None:
    records = build_mmmu_result_records(
        [mmmu_sample("first", "A"), mmmu_sample("second", "C")],
        [answered("Answer: A"), answered(UNPARSEABLE_TEXT)],
    )

    summary = compute_mmmu_metrics(records)
    assert (summary["correct"], summary["unparseable"]) == (1, 1)


def test_unparseable_video_answer_is_wrong() -> None:
    """A video answer that names no option scores wrong and counts as unparseable."""
    records = build_videomme_result_records(
        [video_sample("first", "A"), video_sample("second", "C")],
        [answered(UNPARSEABLE_TEXT), answered("Answer: C")],
    )

    assert [record["is_correct"] for record in records] == [False, True]
    assert compute_videomme_metrics(records)["unparseable"] == 1
