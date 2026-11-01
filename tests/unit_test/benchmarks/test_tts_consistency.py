# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import io

import numpy as np
import soundfile as sf

from benchmarks.dataset.seedtts import SampleInput
from benchmarks.eval import benchmark_tts_consistency as consistency
from benchmarks.eval.benchmark_tts_consistency import (
    _build_arg_parser,
    _build_sessions,
    _concat_wav_blobs,
    _generation_kwargs,
    _session_config,
    adjacent_abs_deltas,
    load_consistency_samples,
)


def test_defaults_use_cross_turn_english() -> None:
    args = _build_arg_parser().parse_args([])
    assert args.mode == "cross-turn"
    assert args.endpoint == "ws"
    assert args.turns_per_session == 4
    assert args.repeats == 3
    assert "seed" not in _generation_kwargs(args)
    assert _generation_kwargs(args)["language"] == "English"
    zh = _build_arg_parser().parse_args(["--lang", "zh"])
    assert _generation_kwargs(zh)["language"] == "Chinese"


def test_session_config_omits_seed_unless_set() -> None:
    sample = SampleInput("a", "ref a", "a.wav", "hello")
    omitted = _session_config(sample, _build_arg_parser().parse_args([]))
    seeded = _session_config(sample, _build_arg_parser().parse_args(["--seed", "42"]))
    assert omitted["type"] == "session.config"
    assert "input" not in omitted
    assert "seed" not in omitted
    assert "carry_previous_audio" not in omitted
    assert seeded["seed"] == 42


def test_prev_ref_sets_carry_flag_and_groups_like_session() -> None:
    sample = SampleInput("a", "ref a", "a.wav", "hello")
    config = _session_config(
        sample, _build_arg_parser().parse_args(["--mode", "prev-ref"])
    )
    assert config["carry_previous_audio"] is True
    sessions = _build_sessions(
        _build_arg_parser().parse_args(
            ["--mode", "prev-ref", "--turns-per-session", "2"]
        ),
        [
            SampleInput(str(index), f"ref {index}", f"{index}.wav", f"text {index}")
            for index in range(4)
        ],
    )
    assert {sample.ref_audio for sample in sessions[0]} == {"0.wav"}


def test_cross_turn_reuses_the_first_reference_and_drops_a_short_tail() -> None:
    samples = [
        SampleInput(str(index), f"ref {index}", f"{index}.wav", f"text {index}")
        for index in range(5)
    ]
    sessions = _build_sessions(
        _build_arg_parser().parse_args(
            ["--mode", "cross-turn", "--turns-per-session", "2"]
        ),
        samples,
    )
    assert len(sessions) == 2
    assert [sample.target_text for sample in sessions[0]] == ["text 0", "text 1"]
    assert {sample.ref_audio for sample in sessions[0]} == {"0.wav"}
    assert {sample.ref_audio for sample in sessions[1]} == {"2.wav"}


def test_session_groups_like_cross_turn() -> None:
    samples = [
        SampleInput(str(index), f"ref {index}", f"{index}.wav", f"text {index}")
        for index in range(4)
    ]
    sessions = _build_sessions(
        _build_arg_parser().parse_args(
            ["--mode", "session", "--turns-per-session", "2"]
        ),
        samples,
    )
    assert len(sessions) == 2
    assert {sample.ref_audio for sample in sessions[0]} == {"0.wav"}
    assert [sample.target_text for sample in sessions[0]] == ["text 0", "text 1"]


def test_repeat_copies_each_row() -> None:
    sessions = _build_sessions(
        _build_arg_parser().parse_args(["--mode", "repeat", "--repeats", "3"]),
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


def test_concat_wav_blobs_joins_turns() -> None:
    first = io.BytesIO()
    second = io.BytesIO()
    sf.write(first, np.ones(2400, dtype=np.float32), 24000, format="WAV")
    sf.write(second, np.full(1200, 0.5, dtype=np.float32), 24000, format="WAV")
    merged, rate = sf.read(
        io.BytesIO(_concat_wav_blobs([first.getvalue(), second.getvalue()])),
        dtype="float32",
    )
    assert rate == 24000
    assert merged.shape[0] == 3600
