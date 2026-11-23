# SPDX-License-Identifier: Apache-2.0
"""Unified Full-Duplex-Bench v1.0 metrics on hand-built reference trees."""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import math
import random
import re
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest
import soundfile

from benchmarks.duplex import unified_v10
from benchmarks.duplex.reference_v10 import MANIFEST_KIND
from benchmarks.duplex.run_artifacts import file_sha256
from benchmarks.eval.benchmark_duplex_v10 import main

PAUSES = [{"text": "[PAUSE]", "timestamp": [1.0, 2.0]}]
TURN_END = [{"text": "[TURN-TAKING]", "timestamp": [2.0, 2.5]}]
# note (luojiaxuan): float noise copied from real dataset annotations and transcripts.
NOISY_TURN_END = [{"text": "[TURN-TAKING]", "timestamp": [6.32000000000005, 6.41]}]
INTERRUPT = [
    {"context": "tell me about tea", "interrupt": "and coffee", "timestamp": [3.0, 5.0]}
]
NOISY_INTERRUPT = [
    {
        "context": "tell me about tea",
        "interrupt": "and coffee",
        "timestamp": [3.0, 6.519999999999982],
    }
]
# note (luojiaxuan): (sample id, annotation file and entries, word spans, window T);
# words None marks an ineligible session.
SAMPLES = [
    (
        "synthetic_pause_handling/1",
        ("pause.json", [*PAUSES, {"text": "[PAUSE]", "timestamp": [3.0, 4.0]}]),
        [[0.25, 0.5], [1.0, 1.5], [2.0, 2.25]],
        6.0,
    ),
    (
        "synthetic_pause_handling/2",
        ("pause.json", PAUSES),
        [[0.5, 1.25], [2.5, 3.0]],
        6.0,
    ),
    ("synthetic_pause_handling/3", ("pause.json", PAUSES), [[1.5, None]], 6.0),
    ("synthetic_pause_handling/4", ("pause.json", PAUSES), [], 6.0),
    ("synthetic_pause_handling/5", ("pause.json", PAUSES), None, 6.0),
    (
        "candor_turn_taking/1",
        ("turn_taking.json", TURN_END),
        [[1.5, 1.75], [2.5, 3.0]],
        10.0,
    ),
    ("candor_turn_taking/2", ("turn_taking.json", TURN_END), [[3.0, 3.5]], 10.0),
    ("candor_turn_taking/3", ("turn_taking.json", TURN_END), [[2.0, 2.5]], 4.0),
    ("candor_turn_taking/4", ("turn_taking.json", TURN_END), [[1.0, 1.5]], 10.0),
    ("candor_turn_taking/5", ("turn_taking.json", TURN_END), [], 10.0),
    ("candor_turn_taking/6", ("turn_taking.json", TURN_END), [[6.0, 6.5]], 10.0),
    (
        "candor_turn_taking/7",
        ("turn_taking.json", NOISY_TURN_END),
        [[6.32, 6.8]],
        12.0,
    ),
    (
        "synthetic_user_interruption/1",
        ("interrupt.json", INTERRUPT),
        [[5.5, 6.0], [6.0, 6.5]],
        12.0,
    ),
    ("synthetic_user_interruption/2", ("interrupt.json", INTERRUPT), [], 12.0),
    ("synthetic_user_interruption/3", ("interrupt.json", INTERRUPT), None, 12.0),
    (
        "synthetic_user_interruption/4",
        ("interrupt.json", NOISY_INTERRUPT),
        [[7.5200000000000005, 8.0]],
        12.0,
    ),
    (
        "synthetic_user_interruption/5",
        ("interrupt.json", INTERRUPT),
        [[5.25, 5.75]],
        12.0,
    ),
    (
        "synthetic_user_interruption/6",
        ("interrupt.json", INTERRUPT),
        [[5.5, 6.0]],
        12.0,
    ),
    (
        "synthetic_user_interruption/7",
        ("interrupt.json", INTERRUPT),
        [[5.0, 5.5]],
        12.0,
    ),
    ("icc_backchannel/1", None, [[0.5, 0.75]], 8.0),
]
# note (luojiaxuan): the speech segments the fake VAD returns per sample; the
# interruption in INTERRUPT runs from S=3.0 to R=5.0.
SPEECH = {
    "synthetic_user_interruption/1": [(2.0, 4.0), (5.5, 6.5)],
    "synthetic_user_interruption/2": [(2.5, 3.5)],
    "synthetic_user_interruption/4": [(2.0, 3.5), (7.5, 8.0)],
    "synthetic_user_interruption/5": [(2.0, 4.0), (4.25, 6.0)],
    "synthetic_user_interruption/6": [(0.5, 2.5), (5.5, 6.0)],
    "synthetic_user_interruption/7": [(2.0, 4.0), (5.0, 6.0)],
}


@pytest.fixture(autouse=True)
def fake_vad(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(unified_v10, "load_silero_vad", lambda: None)
    monkeypatch.setattr(
        unified_v10,
        "speech_segments",
        lambda path, window_s, vad: SPEECH[
            f"{path.parent.parent.name}/{path.parent.name}"
        ],
    )


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_tree(root: Path, samples: list[tuple] = SAMPLES) -> Path:
    """Reference tree as reference-export plus reference-asr leave it."""
    tree = root / "tree"
    rows, transcripts = [], {}
    for sample_id, annotation, words, window_s in samples:
        subset, name = sample_id.split("/")
        row = {
            "sample_id": sample_id,
            "subset": subset,
            "eligible": words is not None,
            "reasons": [] if words is not None else ["input pacing deviation"],
            "window": {"T_s": window_s},
            "files": {},
        }
        rows.append(row)
        if words is None:
            continue
        else:
            pass
        sample_dir = tree / sample_id
        sample_dir.mkdir(parents=True)
        (sample_dir / "output.json").write_text(
            json.dumps(
                {
                    "text": " ".join("w" for _ in words),
                    "chunks": [{"text": "w", "timestamp": span} for span in words],
                }
            )
        )
        transcripts.setdefault(subset, {})[name] = sha256_file(
            sample_dir / "output.json"
        )
        soundfile.write(sample_dir / "output.wav", np.zeros(160, dtype=np.int16), 16000)
        row["files"]["output.wav"] = sha256_file(sample_dir / "output.wav")
        if annotation is not None:
            (sample_dir / annotation[0]).write_text(json.dumps(annotation[1]))
            row["files"][annotation[0]] = sha256_file(sample_dir / annotation[0])
        else:
            pass
    (tree / "manifest.json").write_text(
        json.dumps({"kind": MANIFEST_KIND, "samples": rows})
    )
    (tree / "asr.json").write_text(
        json.dumps(
            {
                "subsets": {
                    subset: {"task": "default", "transcripts": names}
                    for subset, names in transcripts.items()
                }
            }
        )
    )
    return tree


def run_cli(tree: Path) -> str:
    stdout = io.StringIO()
    with contextlib.redirect_stdout(stdout):
        assert main(["unified-score", "--tree", str(tree)]) == 0
    return stdout.getvalue()


def write_summary(tree: Path) -> dict:
    """A summary.json as reference-evaluate writes it, with a local interruption judge."""
    local_judge = {
        "official": False,
        "base_url": "http://127.0.0.1:8000/v1/",
        "requested_models": ["gpt-4o"],
        "sent_models": ["Qwen3-32B"],
        "returned_models": {"Qwen3-32B": 3},
    }
    summary = {
        "manifest_sha256": file_sha256(tree / "manifest.json"),
        "asr_sha256": file_sha256(tree / "asr.json"),
        "subsets": {
            "candor_turn_taking": {
                "result": {"Average take turn": 0.5, "Average latency": 0.25},
                "judge": None,
            },
            "synthetic_user_interruption": {
                "result": {"Average rating": 3.5},
                "judge": local_judge,
            },
        },
    }
    (tree / "summary.json").write_text(json.dumps(summary))
    return summary


def test_pause_intrusion_counts_word_starts_from_the_first_pause() -> None:
    pauses = [(1.0, 2.0), (3.0, 4.0)]
    spans = unified_v10.word_spans(
        [
            {"text": "a", "timestamp": [0.5, 1.25]},
            {"text": "b", "timestamp": [2.0, 2.5]},
            {"text": "c", "timestamp": [3.0, None]},
        ]
    )

    sample = unified_v10.pause_sample(spans, pauses, 6.0)

    assert spans[2] == (3.0, 3.0)
    assert sample == {
        "pauses": 2,
        "pause_s": 2.0,
        "intruded_pauses": 2,
        "intrusion_words": 2,
        "intrusion_s": 0.25,
        "pause_intrusion": True,
        "inside_pause_onset": True,
        "pre_pause_speech": True,
        "carry_in": True,
        "any_speech": True,
    }
    before_first = unified_v10.pause_sample([(0.5, 0.75)], pauses, 6.0)
    assert (before_first["pause_intrusion"], before_first["any_speech"]) == (
        False,
        True,
    )
    after_last = unified_v10.pause_sample([(5.0, 5.5)], pauses, 6.0)
    assert (after_last["intruded_pauses"], after_last["intrusion_s"]) == (1, 0.0)
    assert unified_v10.pause_sample([], pauses, 6.0)["any_speech"] is False
    assert unified_v10.overlap_s([(0.0, 2.0), (1.0, 3.0)], [(2.5, 4.0)]) == 0.5


def test_last_pause_owns_the_time_up_to_the_window_end() -> None:
    pauses = [(1.0, 2.0), (3.0, 4.0)]

    at_end = unified_v10.pause_sample([(6.0, 6.0)], pauses, 6.0)
    assert (at_end["pause_intrusion"], at_end["intruded_pauses"]) == (True, 1)
    both = unified_v10.pause_sample([(2.5, 2.75), (5.0, 6.0)], pauses, 6.0)
    assert (both["intrusion_words"], both["intruded_pauses"]) == (2, 2)
    one_owner = unified_v10.pause_sample([(3.5, 3.75), (5.0, 6.0)], pauses, 6.0)
    assert (one_owner["intrusion_words"], one_owner["intruded_pauses"]) == (2, 1)


def test_pause_diagnostics_split_onsets_around_the_first_pause() -> None:
    def flags(spans: list[tuple[float, float]]) -> tuple[bool, ...]:
        sample = unified_v10.pause_sample(spans, [(2.0, 3.0)], 10.0)
        return tuple(
            sample[name]
            for name in (
                "pause_intrusion",
                "inside_pause_onset",
                "pre_pause_speech",
                "carry_in",
            )
        )

    carried = unified_v10.pause_sample([(1.5, 3.5)], [(2.0, 3.0)], 10.0)
    assert (carried["pause_intrusion"], carried["intrusion_s"]) == (False, 1.0)
    assert flags([(1.5, 3.5)]) == (False, False, True, True)
    assert flags([(1.0, 2.0)]) == (False, False, True, False)
    assert flags([(3.5, 4.0)]) == (True, False, False, False)
    assert flags([(2.5, 2.5)]) == (True, True, False, False)
    assert flags([(3.0, 3.25)]) == (True, False, False, False)
    inside = unified_v10.pause_sample([(2.5, 2.5)], [(2.0, 3.0)], 10.0)
    assert inside["intrusion_s"] == 0.0


def test_times_are_compared_at_microsecond_resolution() -> None:
    def scored(task: str, start: float, annotation: list[dict]) -> dict:
        spans = unified_v10.word_spans([{"text": "w", "timestamp": [start, None]}])
        reference = unified_v10.reference_times(task, annotation)
        assert unified_v10.timestamp_problems(task, spans, reference, 12.0) == []
        if task == "pause_handling":
            return unified_v10.pause_sample(spans, reference, 12.0)
        else:
            return unified_v10.turn_taking_sample([spans[0][0]], reference, 12.0)

    pause = [{"text": "[PAUSE]", "timestamp": [1.9199999999998454, 2.5]}]
    noisy_pause = scored("pause_handling", 1.92, pause)
    assert (noisy_pause["pause_intrusion"], noisy_pause["inside_pause_onset"]) == (
        True,
        True,
    )
    turn = scored("turn_taking", 6.32, NOISY_TURN_END)
    assert (turn["outcome"], turn["early_words"], turn["latency_s"]) == (
        "response",
        0,
        0.0,
    )
    noisy_turn_end = [{"text": "[TURN-TAKING]", "timestamp": [6.519999999999982, 7]}]
    late = scored("turn_taking", 7.5200000000000005, noisy_turn_end)
    assert late["latency_s"] == 1.0
    metrics = unified_v10.subset_metrics("turn_taking", [late], 1)
    assert metrics["response_rate_1s"]["count"] == 1
    window_end = unified_v10.reference_times(
        "pause_handling", [{"text": "[PAUSE]", "timestamp": [9.0, 12.000000000000002]}]
    )
    assert unified_v10.timestamp_problems("pause_handling", [], window_end, 12.0) == []


def test_timestamp_problems_name_each_time_the_window_cannot_score() -> None:
    def problems(task: str, spans: list, reference: list | tuple | float) -> list[str]:
        return unified_v10.timestamp_problems(task, spans, reference, 10.0)

    assert problems("turn_taking", [(2.5, 3.0), (10.0, 10.0)], 2.0) == []
    assert problems("turn_taking", [(9.5, 10.08)], 2.0) == []
    assert problems("turn_taking", [(9.5, 10.09)], 2.0) == [
        "word end 10.09 is outside [0, 10.08]"
    ]
    assert problems("turn_taking", [(12.0, 12.5)], 1.0) == [
        "word start 12.0 is outside [0, 10.0]",
        "word end 12.5 is outside [0, 10.08]",
    ]
    assert problems("turn_taking", [(2.5, 2.0)], 1.0) == [
        "word [2.5, 2.0] ends before it starts"
    ]
    assert problems("turn_taking", [(-0.5, 0.0)], 1.0) == [
        "word start -0.5 is outside [0, 10.0]"
    ]
    assert problems("turn_taking", [], 11.0) == ["turn end E 11.0 is outside [0, 10.0]"]
    assert problems("user_interruption", [(5.0, 5.5)], (3.0, 5.0)) == []
    assert problems("user_interruption", [(4.0, 4.5)], (3.0, 5.0)) == [
        "word start 4.0 is before the interruption end R 5.0"
    ]
    assert problems("user_interruption", [], (3.0, 11.0)) == [
        "interruption bound 11.0 is outside [0, 10.0]"
    ]
    assert problems("user_interruption", [], (5.0, 4.0)) == [
        "interruption [5.0, 4.0] ends before it starts"
    ]
    assert problems("pause_handling", [(9.5, 12.5)], [(1.0, 2.0)]) == [
        "word end 12.5 is outside [0, 10.08]"
    ]
    assert problems("pause_handling", [], [(1.0, 2.0), (2.0, 2.0), (3.0, 10.0)]) == []
    assert problems("pause_handling", [], []) == ["pause.json lists no pause"]
    assert problems("pause_handling", [], [(2.0, 1.0)]) == [
        "pause [2.0, 1.0] ends before it starts"
    ]
    assert problems("pause_handling", [], [(9.0, 11.0)]) == [
        "pause bound 11.0 is outside [0, 10.0]"
    ]
    assert problems("pause_handling", [], [(3.0, 4.0), (1.0, 2.0)]) == [
        "pause [1.0, 2.0] starts before the previous pause ends at 4.0"
    ]
    assert problems("pause_handling", [], [(1.0, 3.0), (2.0, 4.0)]) == [
        "pause [2.0, 4.0] starts before the previous pause ends at 3.0"
    ]
    assert problems("pause_handling", [], [(1.0, 2.0), (1.0, 2.0)]) == [
        "pause [1.0, 2.0] starts before the previous pause ends at 2.0"
    ]


def test_a_word_end_within_one_asr_frame_past_the_window_scores_at_the_window_end(
    tmp_path: Path,
) -> None:
    def scored(root: Path, last_word_end_s: float) -> dict:
        sample = (
            "synthetic_pause_handling/2",
            ("pause.json", PAUSES),
            [[0.5, 1.25], [2.5, last_word_end_s]],
            6.0,
        )
        tree = build_tree(root, [sample])
        row = json.loads((tree / "manifest.json").read_text())["samples"][0]
        return unified_v10.score_eligible(tree, row, "pause_handling", None)

    assert scored(tmp_path / "late", 6.06) == scored(tmp_path / "at-window-end", 6.0)


def test_turn_taking_outcomes_partition_the_samples() -> None:
    assert unified_v10.turn_taking_sample([1.5, 2.5], 2.0, 10.0) == {
        "outcome": "early_start",
        "early_words": 1,
        "reference_s": 2.0,
        "observable_s": 8.0,
        "latency_s": None,
    }
    at_reference = unified_v10.turn_taking_sample([2.0], 2.0, 10.0)
    assert (at_reference["outcome"], at_reference["latency_s"]) == ("response", 0.0)
    only_early = unified_v10.turn_taking_sample([1.0], 2.0, 10.0)
    assert (only_early["outcome"], only_early["latency_s"]) == ("early_start", None)
    silent = unified_v10.turn_taking_sample([], 2.0, 10.0)
    assert (silent["outcome"], silent["latency_s"]) == ("no_response", None)


def test_latency_quantiles_count_samples_without_response_as_never() -> None:
    half = unified_v10.latency_summary([0.5, 1.0, math.inf, math.inf])
    assert (half["n"], half["responses"]) == (4, 2)
    assert (half["median_s"], half["p90_s"]) == (1.0, None)
    assert half["median_ci95"][1] is None

    most = unified_v10.latency_summary([math.inf, 2.0, 0.5, 1.0])
    assert (most["median_s"], most["p90_s"]) == (1.0, None)
    two_of_four = unified_v10.latency_summary([0.1, 0.2, math.inf, math.inf])
    assert two_of_four["median_s"] == 0.2
    nine = [0.1 * index for index in range(1, 10)]
    assert unified_v10.latency_summary([*nine, math.inf])["p90_s"] == nine[-1]
    assert unified_v10.latency_summary([*nine, math.inf, math.inf])["p90_s"] is None
    assert unified_v10.latency_summary([0.3]) == {
        "n": 1,
        "responses": 1,
        "median_s": 0.3,
        "median_ci95": [0.3, 0.3],
        "p90_s": 0.3,
    }


def test_latency_quantiles_invert_the_response_rate_curve() -> None:
    generator = random.Random(0)
    outcomes = [0.0, 0.5, 1.0, 1.5, 2.0, 4.5, 5.0, 6.0, "early_start", "no_response"]
    for _ in range(300):
        drawn = [generator.choice(outcomes) for _ in range(generator.randint(1, 12))]
        rows = [
            (
                {"outcome": outcome, "latency_s": None}
                if isinstance(outcome, str)
                else {"outcome": "response", "latency_s": outcome}
            )
            for outcome in drawn
        ]
        metrics = unified_v10.subset_metrics("turn_taking", rows, len(rows))
        latencies = sorted(outcome for outcome in drawn if not isinstance(outcome, str))
        for name, q in (("median_s", Fraction(1, 2)), ("p90_s", Fraction(9, 10))):
            value = metrics["response_latency_s"][name]
            assert value == next(
                (
                    time
                    for time in latencies
                    if Fraction(sum(other <= time for other in latencies), len(rows))
                    >= q
                ),
                None,
            )
            for within_s in unified_v10.RESPONSE_WITHIN_S:
                share = metrics[f"response_rate_{within_s}s"]
                assert (value is not None and value <= within_s) == (
                    Fraction(share["count"], share["n"]) >= q
                )


def test_pooled_ratio_survives_zero_pause_time() -> None:
    mixed = unified_v10.pooled_ratio(
        [{"top": 1.0, "bottom": 0.0}, {"top": 0.0, "bottom": 2.0}], "top", "bottom"
    )
    assert mixed["value"] == 0.5
    assert mixed["ci95"] is not None
    assert unified_v10.pooled_ratio([{"top": 0.0, "bottom": 0.0}], "top", "bottom") == {
        "numerator": 0.0,
        "denominator": 0.0,
        "value": None,
        "ci95": None,
    }


def test_intervals_are_wilson_and_deterministic_bootstrap() -> None:
    assert unified_v10.wilson(5, 10) == pytest.approx([0.2366, 0.7634], abs=1e-4)
    assert unified_v10.wilson(0, 10)[0] == 0.0
    assert unified_v10.wilson(10, 10)[1] == 1.0
    assert unified_v10.wilson(0, 10)[1] == pytest.approx(0.2775, abs=1e-4)
    assert 0.0 < unified_v10.wilson(1, 10)[0] < unified_v10.wilson(9, 10)[1] < 1.0
    assert unified_v10.wilson(0, 0) is None
    assert unified_v10.rate([True, False], 4) == {
        "count": 1,
        "n": 2,
        "value": 0.5,
        "ci95": unified_v10.wilson(1, 2),
        "selected_bounds": [0.25, 0.75],
    }
    assert unified_v10.rate([], 2) == {
        "count": 0,
        "n": 0,
        "value": None,
        "ci95": None,
        "selected_bounds": [0.0, 1.0],
    }

    latencies = [0.1 * index for index in range(40)]
    first = unified_v10.latency_summary(latencies)
    assert first == unified_v10.latency_summary(latencies)
    assert first == unified_v10.latency_summary(list(reversed(latencies)))
    assert first["median_ci95"][0] < first["median_s"] < first["median_ci95"][1]
    assert unified_v10.latency_summary([]) == {
        "n": 0,
        "responses": 0,
        "median_s": None,
        "median_ci95": None,
        "p90_s": None,
    }


def test_unified_score_writes_metrics_rows_and_reference_column(
    tmp_path: Path,
) -> None:
    tree = build_tree(tmp_path)
    write_summary(tree)
    summary_bytes = (tree / "summary.json").read_bytes()

    printed = run_cli(tree)

    unified = json.loads((tree / "unified.json").read_text())
    assert unified["definitions_version"] == unified_v10.DEFINITIONS_VERSION
    assert set(unified["subsets"]) == {
        "synthetic_pause_handling",
        "candor_turn_taking",
        "synthetic_user_interruption",
    }
    pause = unified["subsets"]["synthetic_pause_handling"]
    assert pause["counts"] == {
        "declared": 137,
        "selected": 5,
        "eligible": 4,
        "ineligible": 1,
        "ineligible_reasons": {"input pacing deviation": 1},
    }
    assert pause["coverage"] == 0.8
    metrics = pause["metrics"]
    assert set(metrics) == {
        "pause_intrusion_rate",
        "pause_intrusion_per_pause",
        "intrusion_s_per_pause_s",
        "inside_pause_onset_rate",
        "pre_pause_speech_rate",
        "carry_in_rate",
        "any_speech_rate",
    }
    assert (
        metrics["pause_intrusion_rate"]["count"],
        metrics["pause_intrusion_rate"]["n"],
    ) == (3, 4)
    assert metrics["pause_intrusion_rate"]["selected_bounds"] == [0.6, 0.8]
    assert metrics["pause_intrusion_per_pause"]["value"] == pytest.approx(3 / 5)
    assert metrics["intrusion_s_per_pause_s"]["value"] == pytest.approx(0.75 / 5)
    assert metrics["intrusion_s_per_pause_s"]["ci95"] is not None
    assert [
        metrics[f"{flag}_rate"]["count"]
        for flag in ("inside_pause_onset", "pre_pause_speech", "carry_in")
    ] == [2, 2, 1]
    assert metrics["any_speech_rate"]["value"] == 0.75

    turn = unified["subsets"]["candor_turn_taking"]
    assert turn["coverage"] == 1.0
    metrics = turn["metrics"]
    assert metrics["early_start_rate"]["count"] == 2
    assert [
        metrics[f"response_rate_{within_s}s"]["count"] for within_s in (1, 2, 5)
    ] == [3, 3, 4]
    assert all(
        metrics[name]["n"] == 7
        for name in ("response_rate_1s", "no_response_rate", "early_start_rate")
    )
    assert metrics["no_response_rate"]["count"] == 1
    assert metrics["early_start_rate"]["selected_bounds"] == [2 / 7, 2 / 7]
    latency = metrics["response_latency_s"]
    assert (latency["n"], latency["responses"]) == (7, 4)
    assert (latency["median_s"], latency["p90_s"]) == (4.0, None)
    assert turn["min_observable_s"] == 2.0

    interruption = unified["subsets"]["synthetic_user_interruption"]
    assert interruption["counts"]["eligible"] == 6
    assert interruption["coverage"] == 6 / 7
    metrics = interruption["metrics"]
    assert "early_start_rate" not in metrics
    assert (metrics["exercised_rate"]["count"], metrics["exercised_rate"]["n"]) == (
        5,
        6,
    )
    assert metrics["exercised_rate"]["selected_bounds"] == [5 / 7, 6 / 7]
    assert (
        metrics["speaking_at_end_rate"]["count"],
        metrics["speaking_at_end_rate"]["n"],
    ) == (1, 5)
    assert metrics["response_rate_1s"]["count"] == 3
    assert metrics["response_rate_1s"]["selected_bounds"] == [3 / 7, 5 / 7]
    assert metrics["no_response_rate"]["count"] == 1
    latency = metrics["response_latency_s"]
    assert (latency["n"], latency["responses"], latency["median_s"]) == (5, 3, 1.0)
    stop = metrics["stop_latency_s"]
    assert (stop["n"], stop["stopped"], stop["median_s"]) == (5, 5, 1.0)
    overlap = metrics["speech_s_per_interruption_s"]
    assert (overlap["numerator"], overlap["denominator"]) == pytest.approx(
        (4.75, 11.52)
    )
    assert metrics["restart_rate"]["count"] == 1

    rows = [
        json.loads(line)
        for line in (tree / "unified-samples.jsonl").read_text().splitlines()
    ]
    assert len(rows) == len(SAMPLES) - 1
    by_id = {row["sample_id"]: row for row in rows}
    assert by_id["synthetic_pause_handling/5"] == {
        "sample_id": "synthetic_pause_handling/5",
        "subset": "synthetic_pause_handling",
        "task": "pause_handling",
        "eligible": False,
        "reasons": ["input pacing deviation"],
    }
    assert by_id["synthetic_pause_handling/2"]["intrusion_words"] == 1
    assert [
        by_id["synthetic_pause_handling/2"][flag]
        for flag in ("inside_pause_onset", "pre_pause_speech", "carry_in")
    ] == [False, True, True]
    assert by_id["synthetic_pause_handling/3"]["inside_pause_onset"] is True
    assert by_id["synthetic_pause_handling/3"]["null_end_words"] == 1
    assert by_id["synthetic_pause_handling/3"]["pause_intrusion"] is True
    assert by_id["synthetic_pause_handling/3"]["intrusion_s"] == 0.0
    assert by_id["candor_turn_taking/1"]["outcome"] == "early_start"
    assert by_id["candor_turn_taking/2"]["latency_s"] == 1.0
    assert by_id["candor_turn_taking/7"]["latency_s"] == 0.0
    assert by_id["synthetic_user_interruption/1"]["observable_s"] == 7.0
    assert by_id["synthetic_user_interruption/4"]["latency_s"] == 1.0
    assert by_id["synthetic_user_interruption/1"]["outcome"] == "response"
    assert by_id["synthetic_user_interruption/7"]["latency_s"] == 0.0
    assert by_id["synthetic_user_interruption/5"]["outcome"] == "speaking_at_end"
    assert by_id["synthetic_user_interruption/5"]["latency_s"] is None
    assert by_id["synthetic_user_interruption/6"]["outcome"] == "not_exercised"
    assert by_id["synthetic_user_interruption/6"]["speech_before_onset"] is True

    assert "window after reference >= 2.000000 s" in printed
    assert "p90 not reached 4/7 responded" in printed
    assert "candor_turn_taking: 7 eligible of 7 selected (coverage 1.000)" in printed
    assert "3/4, all selected [0.600, 0.800]" in printed
    assert "official Average latency" in printed
    assert "reference (judge Qwen3-32B, not official) Average rating" in printed
    assert "official Average rating" not in printed
    assert "icc_backchannel" not in printed
    assert (tree / "summary.json").read_bytes() == summary_bytes

    first_bytes = (tree / "unified.json").read_bytes()
    (tree / "summary.json").unlink()
    run_cli(tree)
    assert (tree / "unified.json").read_bytes() == first_bytes


def test_unified_score_omits_a_reference_column_from_other_inputs(
    tmp_path: Path,
) -> None:
    tree = build_tree(tmp_path)
    summary = write_summary(tree)
    summary["asr_sha256"] = "0" * 64
    (tree / "summary.json").write_text(json.dumps(summary))

    printed = run_cli(tree)

    assert "Average latency" not in printed
    assert "Average rating" not in printed
    assert (
        "summary.json has a different asr_sha256 than this tree; "
        "official reference numbers omitted"
    ) in printed


def test_unified_score_refuses_tree_without_asr_receipt(tmp_path: Path) -> None:
    tree = build_tree(tmp_path)
    (tree / "asr.json").unlink()

    with pytest.raises(SystemExit, match="run reference-asr"):
        main(["unified-score", "--tree", str(tree)])
    assert not (tree / "unified.json").exists()


@pytest.mark.parametrize("edited", ["output.json", "turn_taking.json"])
def test_unified_score_refuses_inputs_changed_after_export(
    tmp_path: Path, edited: str
) -> None:
    tree = build_tree(tmp_path)
    path = tree / "candor_turn_taking" / "1" / edited
    path.write_text(json.dumps(json.loads(path.read_text()), indent=1))

    with pytest.raises(SystemExit, match="differs from manifest.json or asr.json"):
        main(["unified-score", "--tree", str(tree)])


@pytest.mark.parametrize(
    ("sample", "problem"),
    [
        (
            ("candor_turn_taking/6", ("turn_taking.json", TURN_END), [[6.0, 6.5]], 5.0),
            "candor_turn_taking/6: word start 6.0 is outside [0, 5.0]",
        ),
        (
            (
                "synthetic_pause_handling/2",
                ("pause.json", [*PAUSES, {"text": "[PAUSE]", "timestamp": [1.5, 3.0]}]),
                [],
                6.0,
            ),
            "synthetic_pause_handling/2: "
            "pause [1.5, 3.0] starts before the previous pause ends at 2.0",
        ),
        (
            (
                "synthetic_user_interruption/1",
                ("interrupt.json", INTERRUPT),
                [[4.5, 5.5]],
                12.0,
            ),
            "synthetic_user_interruption/1: "
            "word start 4.5 is before the interruption end R 5.0",
        ),
    ],
)
def test_unified_score_stops_on_a_time_the_window_cannot_score(
    tmp_path: Path, sample: tuple, problem: str
) -> None:
    tree = build_tree(
        tmp_path, [sample if row[0] == sample[0] else row for row in SAMPLES]
    )

    with pytest.raises(SystemExit, match=re.escape(problem)):
        main(["unified-score", "--tree", str(tree)])
    assert not (tree / "unified.json").exists()


def test_unified_score_refuses_interruption_audio_changed_after_export(
    tmp_path: Path,
) -> None:
    tree = build_tree(tmp_path)
    soundfile.write(
        tree / "synthetic_user_interruption" / "1" / "output.wav",
        np.ones(160, dtype=np.int16),
        16000,
    )

    with pytest.raises(SystemExit, match="differs from manifest.json or asr.json"):
        main(["unified-score", "--tree", str(tree)])


def test_interruption_metrics_without_an_exercised_sample() -> None:
    silent = unified_v10.interruption_sample([], [], (3.0, 5.0), 12.0)
    assert silent["outcome"] == "not_exercised"

    metrics = unified_v10.subset_metrics("user_interruption", [silent], 1)

    assert metrics["exercised_rate"]["count"] == 0
    assert metrics["speaking_at_end_rate"]["n"] == 0
    assert metrics["response_rate_2s"]["selected_bounds"] == [0.0, 1.0]
    assert metrics["response_latency_s"]["n"] == 0


def test_silero_segments_need_the_reference_wav_format(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.undo()
    vad = unified_v10.load_silero_vad()
    silence = tmp_path / "silence.wav"
    soundfile.write(silence, np.zeros(16000, dtype=np.int16), 16000)
    assert unified_v10.speech_segments(silence, 1.0, vad) == []
    with pytest.raises(SystemExit, match="has 16000 samples; T is 2.0 s"):
        unified_v10.speech_segments(silence, 2.0, vad)
    for name, rate, subtype in (
        ("cd.wav", 22050, "PCM_16"),
        ("float.wav", 16000, "FLOAT"),
    ):
        soundfile.write(
            tmp_path / name, np.zeros(rate, dtype=np.float32), rate, subtype=subtype
        )
        with pytest.raises(SystemExit, match="reference layout"):
            unified_v10.speech_segments(tmp_path / name, 1.0, vad)


@pytest.mark.parametrize(
    ("speech", "words", "outcome", "flags"),
    [
        ([(1.0, 3.999), (4.2, 5.0)], [4.2], "response", (True, False, 0)),
        ([(1.0, 2.2), (2.7, 3.8), (4.2, 5.0)], [4.2], "response", (True, False, 1)),
        ([(2.2, 4.5)], [4.2], "not_exercised", (False, True, 1)),
        ([(1.0, 2.2), (4.0, 5.0)], [4.0], "response", (True, False, 0)),
        ([(1.0, 4.5)], [4.2], "speaking_at_end", (True, True, 0)),
    ],
)
def test_interruption_outcome_and_diagnostics(
    speech: list, words: list, outcome: str, flags: tuple
) -> None:
    sample = unified_v10.interruption_sample(words, speech, (2.0, 4.0), 8.0)
    assert sample["outcome"] == outcome
    assert (
        sample["speech_at_onset"],
        sample["speech_at_end"],
        sample["restarts"],
    ) == flags


def test_interruption_stop_is_unobserved_when_speech_runs_to_the_window_end() -> None:
    sample = unified_v10.interruption_sample([], [(1.0, 8.0)], (2.0, 4.0), 8.0)
    assert (sample["outcome"], sample["stop_s"]) == ("speaking_at_end", None)
    assert sample["speech_in_interruption_s"] == 2.0
    stopped = unified_v10.interruption_sample([], [(1.0, 3.5)], (2.0, 4.0), 8.0)
    assert (stopped["stop_s"], stopped["speech_in_interruption_s"]) == (1.5, 1.5)
