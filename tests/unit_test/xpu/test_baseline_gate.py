# SPDX-License-Identifier: Apache-2.0
"""How the XPU baseline folds samples and judges a run; pure functions, so no card and no server."""

from __future__ import annotations

import pytest

from scripts.xpu.baseline.gate import (
    BUILTIN_TOLERANCES,
    DEFAULT_WARMUPS,
    card_pool,
    compare,
    fold_launches,
    fold_samples,
    merged_serve_extra,
    resolve_tolerances,
    speed_tolerance_for,
    warmups_for,
    with_recorded_judge_flags,
)
from scripts.xpu.baseline.models import MODEL_BY_KEY, BaselineModel, Record

LATENCIES = [1.0, 1.1, 1.2, 1.3, 2.0]


def test_repeats_fold_to_the_median_and_a_ceiling_to_the_worst_sample() -> None:
    folded = fold_samples(
        [
            {
                "exact": {"transcript": "cars"},
                "atmost": {"word_error_rate": rate},
                "speed": {"latency_s": latency},
            }
            for latency, rate in zip(LATENCIES, [0.0, 0.0, 0.1, 0.0, 0.0])
        ]
    )
    assert folded["exact"] == {"transcript": "cars"}
    assert folded["atmost"] == {"word_error_rate": 0.1}
    assert folded["speed"] == {"latency_s": 1.2}
    assert folded["spread"]["speed.latency_s"] == pytest.approx(0.8333, abs=1e-4)
    assert folded["trimmed_spread"]["speed.latency_s"] == pytest.approx(
        0.1667, abs=1e-4
    )


def test_an_exact_field_that_differs_between_repeats_fails_the_fold() -> None:
    with pytest.raises(AssertionError, match="not reproducible"):
        fold_samples([{"exact": {"content": "42"}}, {"exact": {"content": "41"}}])


def test_fewer_than_five_repeats_record_no_trimmed_spread() -> None:
    folded = fold_samples([{"speed": {"latency_s": value}} for value in LATENCIES[:4]])
    assert "trimmed_spread" not in folded


def test_an_observed_field_is_kept_beside_its_span_and_never_gated() -> None:
    folded = fold_samples(
        [{"observed": {"frames": frames}} for frames in (90.0, 100.0, 110.0)]
    )
    assert folded["observed"] == {"frames": 100.0}
    assert folded["observed_span"] == {"frames": 0.2}


def test_one_launch_is_its_own_record() -> None:
    run: Record = {"speed": {"latency_s": 1.0}, "cards": "3"}
    assert fold_launches([run]) is run


def test_launches_fold_to_the_slowest_speed_and_keep_every_card_layout() -> None:
    runs: list[Record] = [
        {
            "exact": {"content": "42"},
            "speed": {"latency_s": latency, "decode_tokens_per_second": rate},
            "spread": {"speed.latency_s": 0.5},
            "trimmed_spread": {"speed.latency_s": trimmed},
            "cards": cards,
            "judge_cards": "7",
            "judge_serve_extra": ["--mem-fraction-static", "0.6"],
            "warmups": 6,
        }
        for latency, rate, trimmed, cards in (
            (1.0, 30.0, 0.02, "1"),
            (1.4, 20.0, 0.05, "2"),
            (1.2, 25.0, 0.01, "1"),
        )
    ]
    folded = fold_launches(runs)
    assert folded["speed"] == {"latency_s": 1.4, "decode_tokens_per_second": 20.0}
    assert folded["cards"] == "1 | 2"
    assert folded["judge_cards"] == "7"
    assert folded["within_run_spread"] == {"speed.latency_s": 0.05}
    assert folded["launches"] == 3
    assert folded["judge_serve_extra"] == ["--mem-fraction-static", "0.6"]
    assert folded["warmups"] == 6


def test_a_single_launch_gate_is_sized_from_its_trimmed_spread() -> None:
    record: Record = {
        "spread": {"speed.latency_s": 0.9},
        "trimmed_spread": {"speed.latency_s": 0.05},
    }
    assert speed_tolerance_for(
        record, "latency_s", BUILTIN_TOLERANCES
    ) == pytest.approx(0.15)


def test_a_multi_launch_gate_charges_the_within_run_spread_and_not_the_launch_spread() -> (
    None
):
    record: Record = {
        "launches": 3,
        "spread": {"speed.latency_s": 0.4},
        "within_run_spread": {"speed.latency_s": 0.02},
    }
    assert speed_tolerance_for(record, "latency_s", BUILTIN_TOLERANCES) == 0.10


def test_a_spread_too_wide_to_see_a_doubling_is_capped_at_the_ceiling() -> None:
    record: Record = {"launches": 3, "within_run_spread": {"speed.latency_s": 0.6425}}
    assert speed_tolerance_for(record, "latency_s", BUILTIN_TOLERANCES) == 0.5


def test_a_field_with_no_recorded_spread_gets_the_floor() -> None:
    assert speed_tolerance_for({}, "latency_s", BUILTIN_TOLERANCES) == 0.10


BASELINE: Record = {
    "exact": {"content": "42"},
    "atmost": {"round_trip_wer": 0.1},
    "close": {"rms": 0.05},
    "speed": {"latency_s": 1.0, "audio_seconds_per_wall_second": 4.0},
    "cards": "1",
}


def judged(observed: Record) -> tuple[list[str], list[str], list[str]]:
    return compare("model", BASELINE, observed, BUILTIN_TOLERANCES)


def test_a_run_matching_its_baseline_passes_without_notes() -> None:
    assert judged(BASELINE) == ([], [], [])


def test_garbage_after_the_answer_is_a_changed_exact_field() -> None:
    failures, _, _ = judged({**BASELINE, "exact": {"content": "42!!!!"}})
    assert failures and "content changed" in failures[0]


def test_a_ceiling_fails_past_its_absolute_headroom_and_a_lower_value_is_a_gain() -> (
    None
):
    failures, _, _ = judged({**BASELINE, "atmost": {"round_trip_wer": 0.2}})
    assert failures and "rose to 0.2000" in failures[0]
    failures, gains, _ = judged({**BASELINE, "atmost": {"round_trip_wer": 0.0}})
    assert not failures and "fell to 0.0000" in gains[0]


def test_loudness_outside_its_band_fails() -> None:
    failures, _, _ = judged({**BASELINE, "close": {"rms": 0.06}})
    assert failures and "rms moved +20.00%" in failures[0]


@pytest.mark.parametrize(
    "speed, verdict",
    [
        ({"latency_s": 1.2, "audio_seconds_per_wall_second": 4.0}, "regressed"),
        ({"latency_s": 1.0, "audio_seconds_per_wall_second": 3.0}, "regressed"),
        ({"latency_s": 0.8, "audio_seconds_per_wall_second": 4.0}, "improved"),
        ({"latency_s": 1.05, "audio_seconds_per_wall_second": 3.9}, ""),
    ],
)
def test_latency_must_not_grow_and_a_rate_must_not_shrink(
    speed: dict[str, float], verdict: str
) -> None:
    failures, gains, _ = judged({**BASELINE, "speed": speed})
    said = " ".join(failures + gains)
    assert (verdict in said) if verdict else not said


def test_a_field_missing_from_the_run_fails() -> None:
    failures, _, _ = judged({**BASELINE, "speed": {"latency_s": 1.0}})
    assert failures == ["model: audio_seconds_per_wall_second missing from this run"]


def test_a_card_change_or_a_missing_card_record_is_a_note_not_a_failure() -> None:
    failures, _, notes = judged({**BASELINE, "cards": "4"})
    assert not failures and "measured on 4" in notes[0]
    failures, _, notes = compare(
        "model", {**BASELINE, "cards": ""}, BASELINE, BUILTIN_TOLERANCES
    )
    assert not failures and "re-record" in notes[0]


SPEECH = MODEL_BY_KEY["qwen3-tts-cv"]
RECORDED_JUDGE: Record = {"judge_serve_extra": ["--mem-fraction-static", "0.6"]}
ENTRY_JUDGE = {"qwen3-asr": ("--entry",)}


def test_a_judge_runs_with_the_flags_its_dependent_was_recorded_with() -> None:
    restored = with_recorded_judge_flags(SPEECH, RECORDED_JUDGE, ENTRY_JUDGE, set())
    assert restored["qwen3-asr"] == ("--mem-fraction-static", "0.6")


def test_a_judge_recorded_bare_runs_bare() -> None:
    restored = with_recorded_judge_flags(
        SPEECH, {"judge_serve_extra": []}, ENTRY_JUDGE, set()
    )
    assert restored["qwen3-asr"] == ()


def test_a_judge_flag_named_on_this_run_wins() -> None:
    restored = with_recorded_judge_flags(
        SPEECH, RECORDED_JUDGE, ENTRY_JUDGE, {"qwen3-asr"}
    )
    assert restored is ENTRY_JUDGE


@pytest.mark.parametrize(
    "model, recorded",
    [(SPEECH, {}), (MODEL_BY_KEY["qwen3-asr"], RECORDED_JUDGE)],
    ids=["entry-from-before-the-field", "model-with-no-judge"],
)
def test_nothing_is_restored_without_a_judge_and_a_record_of_its_flags(
    model: BaselineModel, recorded: Record
) -> None:
    assert with_recorded_judge_flags(model, recorded, ENTRY_JUDGE, set()) is ENTRY_JUDGE


@pytest.mark.parametrize(
    "recorded, explicit, want",
    [({}, None, DEFAULT_WARMUPS), ({"warmups": 6}, None, 6), ({"warmups": 6}, 3, 3)],
)
def test_a_check_repeats_the_warmups_its_entry_discarded(
    recorded: Record, explicit: int | None, want: int
) -> None:
    assert warmups_for(recorded, explicit) == want


def test_the_card_pool_defaults_to_the_visible_cards_and_refuses_outside_or_repeated_ones() -> (
    None
):
    assert card_pool("", [2, 3]) == [2, 3]
    assert card_pool("3", [2, 3]) == [3]
    with pytest.raises(ValueError, match="cannot use"):
        card_pool("1", [2, 3])
    with pytest.raises(ValueError, match="more than once"):
        card_pool("2,2", [2, 3])


def test_a_tolerance_flag_beats_the_entry_and_the_entry_beats_the_builtin() -> None:
    tolerances, sources = resolve_tolerances(
        {"speed": 0.2, "close": None}, {"close": 0.05, "speed": 0.15}
    )
    assert (tolerances["speed"], sources["speed"]) == (0.2, "the flag")
    assert (tolerances["close"], sources["close"]) == (0.05, "this entry")
    assert (tolerances["absolute"], sources["absolute"]) == (
        0.05,
        "the built-in default",
    )


def test_a_re_record_replaces_the_flags_of_what_it_measured_and_keeps_the_rest() -> (
    None
):
    merged = merged_serve_extra(
        {"qwen3-tts": ["--old"], "auk": ["--kept"]},
        {"whisper": ("--unmeasured",)},
        {"qwen3-tts"},
    )
    assert merged == {"auk": ["--kept"]}
