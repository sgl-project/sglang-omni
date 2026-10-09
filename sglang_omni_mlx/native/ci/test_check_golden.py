# SPDX-License-Identifier: Apache-2.0
"""Unit tests for check_golden.py's per-chip gate (no runtime needed)."""

from __future__ import annotations

import check_golden

CLIP = {
    "text": "hello",
    "language": "English",
    "generated_token_count": 2,
    "finish_reason": "stop",
}
METRICS = {"wer_en": 0.03, "cer_zh": 0.06, "mer_mixed": 0.16}


def golden(clip_count: int = 10) -> dict:
    return {
        "model": "m",
        "baseline": METRICS,
        "tolerance": {"metric_pp": 1.5, "min_identical": 0.8},
        "devices": {
            "Apple M5": {
                "metrics": METRICS,
                "clips": {f"c{i}": CLIP for i in range(clip_count)},
            }
        },
    }


def results(changed: int, clip_count: int = 10) -> dict:
    return {
        f"c{i}": {**CLIP, "text": "other"} if i < changed else CLIP
        for i in range(clip_count)
    }


def test_a_recorded_chip_requires_every_clip() -> None:
    assert check_golden.check(golden(), "Apple M5", results(0), METRICS)[1] == []
    failures = check_golden.check(golden(), "Apple M5", results(1), METRICS)[1]
    assert failures == ["`c0` differs from its golden output"]


def test_another_chip_is_held_to_the_error_rates() -> None:
    assert check_golden.check(golden(), "Apple M4 Pro", results(2), METRICS)[1] == []
    worse = {**METRICS, "cer_zh": METRICS["cer_zh"] + 0.02}
    failures = check_golden.check(golden(), "Apple M4 Pro", results(2), worse)[1]
    assert failures == ["cer_zh is +2.00 pp from Apple M5's"]


def test_another_chip_needs_most_clips_identical() -> None:
    failures = check_golden.check(golden(), "Apple M4 Pro", results(3), METRICS)[1]
    assert failures == ["only 7/10 clips match Apple M5's outputs"]


def test_an_empty_golden_file_fails() -> None:
    assert check_golden.check(golden(0), "Apple M5", {}, METRICS)[1] == [
        "the golden file has no clips"
    ]
