# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from benchmarks.dataset.seedtts import SampleInput
from benchmarks.eval import benchmark_tts_consistency as consistency
from benchmarks.eval.benchmark_tts_consistency import (
    MODE_CROSS_TURN,
    MODE_REPEAT,
    _build_arg_parser,
    _build_sessions,
    _generation_kwargs,
    adjacent_abs_deltas,
    load_consistency_samples,
)


def test_defaults_use_cross_turn_english() -> None:
    args = _build_arg_parser().parse_args([])
    assert args.mode == MODE_CROSS_TURN
    assert args.turns_per_session == 4
    assert args.repeats == 3
    assert _generation_kwargs(args)["language"] == "English"
    zh = _build_arg_parser().parse_args(["--lang", "zh"])
    assert _generation_kwargs(zh)["language"] == "Chinese"


def test_cross_turn_reuses_the_first_reference_and_drops_a_short_tail() -> None:
    samples = [
        SampleInput(str(index), f"ref {index}", f"{index}.wav", f"text {index}")
        for index in range(5)
    ]
    sessions = _build_sessions(
        _build_arg_parser().parse_args(
            ["--mode", MODE_CROSS_TURN, "--turns-per-session", "2"]
        ),
        samples,
    )
    assert len(sessions) == 2
    assert [sample.target_text for sample in sessions[0]] == ["text 0", "text 1"]
    assert {sample.ref_audio for sample in sessions[0]} == {"0.wav"}
    assert {sample.ref_audio for sample in sessions[1]} == {"2.wav"}


def test_repeat_copies_each_row() -> None:
    sessions = _build_sessions(
        _build_arg_parser().parse_args(["--mode", MODE_REPEAT, "--repeats", "3"]),
        [
            SampleInput("a", "ref a", "a.wav", "hello"),
            SampleInput("b", "ref b", "b.wav", "world"),
        ],
    )
    assert [sample.sample_id for sample in sessions[0]] == ["a__r0", "a__r1", "a__r2"]
    assert {sample.target_text for sample in sessions[0]} == {"hello"}
    assert {sample.ref_audio for sample in sessions[0]} == {"a.wav"}


def test_sample_window_is_offset_then_limit(monkeypatch) -> None:
    calls: list[tuple] = []

    def fake_load(meta, max_samples, split):
        calls.append((meta, max_samples, split))
        return [
            SampleInput(str(index), "ref", f"{index}.wav", f"text {index}")
            for index in range(max_samples or 5)
        ]

    monkeypatch.setattr(consistency, "load_seedtts_samples", fake_load)
    window = load_consistency_samples("meta", "zh", 2, 1)
    assert calls == [("meta", 3, "zh")]
    assert [sample.sample_id for sample in window] == ["1", "2"]


def test_adjacent_deltas_skip_missing_values() -> None:
    assert adjacent_abs_deltas([100.0, None, 80.0, 90.0]) == [10.0]
