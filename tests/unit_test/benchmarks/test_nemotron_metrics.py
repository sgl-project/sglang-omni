from __future__ import annotations

from benchmarks.nemotron_metrics import (
    aggregate_quality,
    batch_distribution,
    percentile,
    score_transcript,
)


def test_percentile_uses_nearest_rank() -> None:
    assert percentile([3, 1, 2], 0.5) == 2
    assert percentile([], 0.5) is None


def test_score_transcript_is_token_aware_for_wer() -> None:
    row = score_transcript("hello world", "hello word", language="en-US")
    assert row["metric"] == "WER"
    assert row["reference_units"] == 2
    assert row["edit_errors"] == 1
    assert row["error_rate"] == 0.5


def test_score_transcript_uses_characters_for_cer() -> None:
    row = score_transcript("你好世界", "你好世", language="zh-CN")
    assert row["metric"] == "CER"
    assert row["reference_units"] == 4
    assert row["error_rate"] == 0.25


def test_quality_aggregates_edit_counts() -> None:
    rows = [
        score_transcript("hello world", "hello word", language="en-US"),
        score_transcript("good morning", "good morning", language="en-US"),
    ]
    result = aggregate_quality(rows)
    assert result["scored_samples"] == 2
    assert result["wer_cer"] == 1 / 4


def test_batch_distribution_keeps_counts_and_fractions() -> None:
    result = batch_distribution([1, 2, 2])
    assert result["counts"] == {"1": 1, "2": 2}
    assert result["fractions"] == {"1": 1 / 3, "2": 2 / 3}
