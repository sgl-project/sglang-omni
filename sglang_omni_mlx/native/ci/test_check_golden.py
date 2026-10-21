# SPDX-License-Identifier: Apache-2.0
"""Unit tests for check_golden.py's per-chip gate (no runtime needed)."""

from __future__ import annotations

import check_golden
import pytest
import voxtral_golden

CLIP = {
    "text": "hello",
    "language": "English",
    "generated_token_count": 2,
    "finish_reason": "stop",
}
METRICS = {"wer_en": 0.03, "cer_zh": 0.06, "mer_mixed": 0.16}
SPEECH = {"prompt_ids": "7 8 9", "finish_reason": "stop", "codes": ["1 2", "3 4"]}


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


def test_moss_tags_are_dropped_before_scoring() -> None:
    text = "[0.00][S01] Hello [sniff] world.[1.25][1.30][S02]你好[2.00]"
    assert (
        check_golden.spoken_text(text, "moss_transcribe_diarize") == "Hello world. 你好"
    )
    assert check_golden.spoken_text("[S01] kept", None) == "[S01] kept"


def test_a_pending_baseline_reports_no_delta() -> None:
    pending = {**golden(), "baseline": {"source": "pending", **METRICS}}
    lines, failures = check_golden.check(pending, "Apple M5", results(0), METRICS)
    assert failures == []
    assert "| wer_en | pending | 3.00% | n/a |" in lines


def speech_golden(case_ids: tuple[str, ...] = ("en", "fr")) -> dict:
    return {
        "model": "v",
        "kind": "voxtral_tts",
        "cases": {case_id: {} for case_id in case_ids},
        "devices": {"Apple M5": {"metrics": {}, "clips": {"en": SPEECH, "fr": SPEECH}}},
    }


def test_a_recorded_chip_requires_every_voxtral_code() -> None:
    same = {"en": SPEECH, "fr": SPEECH}
    assert voxtral_golden.check(speech_golden(), "Apple M5", same)[1] == []
    changed = {
        "en": {**SPEECH, "codes": ["1 2", "3 0"]},
        "fr": {**SPEECH, "codes": ["1 2"], "finish_reason": "length"},
    }
    failures = voxtral_golden.check(speech_golden(), "Apple M5", changed)[1]
    assert failures == ["`en` differs from frame 1", "`fr` has frame count 1, not 2"]


def test_another_chip_is_held_to_voxtral_prompt_ids() -> None:
    sampled = {"en": {**SPEECH, "codes": ["5 6"]}, "fr": SPEECH}
    assert voxtral_golden.check(speech_golden(), "Apple M4 Pro", sampled)[1] == []
    retokenized = {"en": {**SPEECH, "prompt_ids": "7 8"}, "fr": SPEECH}
    failures = voxtral_golden.check(speech_golden(), "Apple M4 Pro", retokenized)[1]
    assert failures == ["`en` has other prompt ids than Apple M5's"]


@pytest.mark.parametrize("device", ["Apple M5", "Apple M4 Pro"])
def test_a_voxtral_case_without_golden_outputs_fails(device: str) -> None:
    results = {"en": SPEECH, "fr": SPEECH, "de": SPEECH}
    golden = speech_golden(("en", "fr", "de"))
    failures = voxtral_golden.check(golden, device, results)[1]
    assert failures == ["`de` has no golden output"]
    empty = {**speech_golden(), "devices": {}}
    assert voxtral_golden.check(empty, "Apple M5", results)[1] == [
        "the golden file has no outputs"
    ]
