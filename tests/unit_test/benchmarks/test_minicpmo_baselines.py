# SPDX-License-Identifier: Apache-2.0
"""MiniCPM-o baseline trials: composed inputs, resumable recording, replies read from traces and logs, and the stop and greeting scores."""

from __future__ import annotations

import argparse
import asyncio
import base64
import hashlib
import json
import math
from pathlib import Path
from typing import Literal

import numpy as np
import pytest
import soundfile
from pydantic import JsonValue

from benchmarks.duplex.client import SAMPLE_RATE, SEND_RECEIPTS_FILE
from benchmarks.duplex.minicpmo_scores import (
    OUTPUT_SAMPLE_RATE,
    Reply,
    ReplyUnit,
    StopScore,
    counted_to,
    greeting_table,
    official_replies,
    score_events,
    score_stop,
    sglang_replies,
    stop_table,
)
from benchmarks.duplex.minicpmo_trials import (
    CLIPS,
    TRIALS,
    Placement,
    SpokenEvent,
    Trial,
    TrialInput,
    compose_trial,
    write_inputs,
)
from benchmarks.eval import benchmark_duplex, benchmark_minicpmo_baselines

FIRST_SEND_S = 100.0
COUNT_WORDS = (
    "one, two, three, four, five, six, seven, eight, nine, ten, eleven, twelve, "
    "thirteen, fourteen, fifteen, sixteen, seventeen, eighteen, nineteen, twenty, "
    "twenty-one, twenty-two, twenty-three, twenty-four, twenty-five"
)
COUNT_TO_TWENTY = COUNT_WORDS.split(", twenty-one")[0]
COUNT_TO_THIRTY = (
    f"{COUNT_WORDS}, twenty-six, twenty-seven, twenty-eight, twenty-nine, thirty."
)
GREETING_EVENTS = [
    SpokenEvent(role="greeting", start_s=2.0, end_s=2.42),
    SpokenEvent(role="greeting", start_s=10.0, end_s=10.42),
    SpokenEvent(role="question", start_s=23.0, end_s=24.78),
]


def reply(texts_by_unit: dict[int, str], audio_end_s: float | None = None) -> Reply:
    return Reply(
        units=tuple(
            ReplyUnit(float(unit), text) for unit, text in texts_by_unit.items()
        ),
        audio_end_s=audio_end_s,
    )


@pytest.mark.parametrize(
    ("transcript", "expected_count"),
    [
        ("Okay, I'll count from one to thirty slowly for you. One, two,", 2),
        ("1, 2, 3, 4, 5, 6, 7, 8, 9, 1 0, 11,", 11),
        ("1 2 3 4 5 6 7 8 9 10 11 12", 12),
        (f"{COUNT_WORDS}, twenty six, twenty-seven", 27),
        (
            f"{COUNT_WORDS}, twenty-six, twenty-seven, twenty-eight, twenty-nine, thirty. One more time?",
            30,
        ),
    ],
)
def test_counted_to_follows_the_count_in_order(
    transcript: str, expected_count: int
) -> None:
    """A preamble naming thirty counted as thirty, "1 0" split across deltas stopped the count at nine, a count without commas read as zero, "twenty six" stopped it at twenty-five, and "thirty. One" read as thirty-one."""
    assert counted_to(transcript) == expected_count


@pytest.mark.parametrize(
    ("replies", "expected_score"),
    [
        (
            [
                reply(
                    {
                        3: "Okay, one, two,",
                        4: " three,",
                        5: " four,",
                        6: " five,",
                        7: " six,",
                    },
                    9.5,
                )
            ],
            StopScore(
                status="kept",
                counted_to_thirty=False,
                reply_end_after_stop_unit_s=0.0,
                delivered_end_after_stop_s=2.5,
            ),
        ),
        (
            [reply({3: COUNT_WORDS, 6: ", twenty-six", 7: ", twenty-seven"})],
            StopScore(
                status="past twenty",
                counted_to_thirty=False,
                reply_end_after_stop_unit_s=0.0,
            ),
        ),
        (
            [reply({3: COUNT_TO_TWENTY, 7: ", twenty-one"})],
            StopScore(
                status="kept", counted_to_thirty=False, reply_end_after_stop_unit_s=0.0
            ),
        ),
        (
            [reply({3: "Okay, one, two,", 6: " three."})],
            StopScore(
                status="ended before stop",
                counted_to_thirty=False,
                reply_end_after_stop_unit_s=-1.0,
            ),
        ),
        (
            [
                reply({2: "Hm?"}),
                reply({3: "Okay, one, two,", 7: " three,", 8: " four."}),
            ],
            StopScore(
                status="kept", counted_to_thirty=False, reply_end_after_stop_unit_s=1.0
            ),
        ),
        (
            [reply({7: "One,", 8: " two."})],
            StopScore(
                status="started after stop",
                counted_to_thirty=False,
                reply_end_after_stop_unit_s=1.0,
            ),
        ),
        (
            [reply({8: "Okay, I'll stop."})],
            StopScore(status="no count", counted_to_thirty=False),
        ),
    ],
)
def test_score_stop_scores_the_counting_reply_against_a_stop_at_seven_seconds(
    replies: list[Reply], expected_score: StopScore
) -> None:
    """A reply that only answered the stop counted as a stop, and a backchannel before the count excluded the run."""
    assert score_stop(7.0, replies) == expected_score


@pytest.mark.parametrize("stop_s", [7.0, 7.25, 7.5, 7.75])
def test_score_stop_gives_every_offset_in_a_unit_the_same_window(stop_s: float) -> None:
    """Timing the window from the stop's onset left a stop late in its unit less time to take effect."""
    assert score_stop(
        stop_s, [reply({3: "One, two,", 7: " three,", 10: " four."})]
    ) == StopScore(
        status="kept", counted_to_thirty=False, reply_end_after_stop_unit_s=3.0
    )


def test_score_stop_without_a_stop_reads_only_the_count() -> None:
    assert score_stop(None, [reply({3: COUNT_TO_THIRTY})]) == StopScore(
        status="kept", counted_to_thirty=True
    )


def test_sglang_replies_read_units_from_media_time_and_delivery_from_the_first_send(
    tmp_path: Path,
) -> None:
    one_second = base64.b64encode(bytes(2 * OUTPUT_SAMPLE_RATE)).decode()
    half_second = base64.b64encode(bytes(OUTPUT_SAMPLE_RATE)).decode()

    def delta(
        time_s: float,
        delta_kind: Literal["audio", "transcript"],
        response_id: str,
        unit_start_s: float,
        delta_text: str,
    ) -> dict[str, JsonValue]:
        return {
            "direction": "receive",
            "time_s": FIRST_SEND_S + time_s,
            "event": {
                "type": (
                    "response.output_audio.delta"
                    if delta_kind == "audio"
                    else "response.output_audio_transcript.delta"
                ),
                "event_id": f"evt_{time_s}_{delta_kind}",
                "response_id": response_id,
                "delta": delta_text,
                "sglang": {
                    "media_time": {
                        "t_start_ms": unit_start_s * 1000,
                        "duration_ms": 1000.0,
                    }
                },
            },
        }

    trace = [
        {
            "direction": "send",
            "time_s": FIRST_SEND_S,
            "event": {"type": "input_audio_buffer.append", "audio": "AAAA"},
        },
        delta(4.1, "transcript", "response_a", 3.0, " one,"),
        delta(4.1, "audio", "response_a", 3.0, one_second),
        delta(5.1, "transcript", "response_a", 4.0, " two,"),
        delta(5.1, "audio", "response_a", 4.0, one_second),
        delta(11.0, "transcript", "response_b", 10.0, " Hi."),
        delta(11.0, "audio", "response_b", 10.0, half_second),
    ]
    (tmp_path / "continuous.jsonl").write_text(
        "".join(json.dumps(trace_record) + "\n" for trace_record in trace)
    )
    (tmp_path / SEND_RECEIPTS_FILE).write_text(
        json.dumps(
            {
                "appends": [
                    {
                        "event_id": "append_0",
                        "seq": 0,
                        "start_s": FIRST_SEND_S,
                        "completed_s": FIRST_SEND_S,
                    }
                ]
            }
        )
    )
    assert sglang_replies(tmp_path) == [
        Reply(
            units=(ReplyUnit(3.0, " one,"), ReplyUnit(4.0, " two,")),
            audio_end_s=pytest.approx(6.1),
        ),
        Reply(
            units=(ReplyUnit(10.0, " Hi."),),
            audio_end_s=pytest.approx(11.5),
        ),
    ]


def test_official_replies_stay_open_through_listen_units_until_an_end_of_turn(
    tmp_path: Path,
) -> None:
    """Splitting at a listen unit ended a count that paused for one unit, and an empty unit opened a reply; the session does neither."""
    units = [
        (2, True, "", False),
        (3, False, " One,", False),
        (4, True, "", False),
        (5, False, " two.", False),
        (6, False, "", True),
        (7, True, "", False),
        (8, False, " Hi.", True),
        (9, False, "", True),
    ]
    log_path = tmp_path / "control_r1.jsonl"
    log_path.write_text(
        "".join(
            json.dumps(
                {
                    "unit_index": unit_index,
                    "is_listen": is_listen,
                    "text": text,
                    "end_of_turn": end_of_turn,
                }
            )
            + "\n"
            for unit_index, is_listen, text, end_of_turn in units
        )
    )
    assert official_replies(log_path) == [
        reply({3: " One,", 5: " two.", 6: ""}),
        reply({8: " Hi."}),
    ]


@pytest.mark.parametrize(
    ("events", "replies", "expected_after_end_s"),
    [
        (
            [SpokenEvent(role="question", start_s=0.0, end_s=0.9), *GREETING_EVENTS],
            [reply({1: "Canberra."}), reply({3: "Hi."}), reply({24: "Canberra."})],
            [1.1, 1.58, None, 0.22],
        ),
        ([GREETING_EVENTS[0]], [reply({1: "Canberra."})], [None]),
        (
            [SpokenEvent(role="greeting", start_s=9.5, end_s=10.0)],
            [reply({14: "Hi again."})],
            [5.0],
        ),
        (
            [SpokenEvent(role="greeting", start_s=9.5, end_s=10.0)],
            [reply({15: "Hi again."})],
            [None],
        ),
        (
            [
                GREETING_EVENTS[2],
                SpokenEvent(role="greeting", start_s=24.9, end_s=25.0),
            ],
            [reply({25: "Canberra, and hi."})],
            [1.22, None],
        ),
    ],
)
def test_score_events_needs_a_reply_starting_within_the_window(
    events: list[SpokenEvent],
    replies: list[Reply],
    expected_after_end_s: list[float | None],
) -> None:
    """A reply that ended as a greeting began answered it, and one reply answered two events."""
    assert score_events(events, replies) == [
        pytest.approx(seconds) if seconds is not None else None
        for seconds in expected_after_end_s
    ]


def test_compose_trial_adds_each_clip_at_its_sample_offset_over_seeded_noise() -> None:
    clip_samples_by_name = {"hi": np.full(800, 1000, dtype=np.int16)}
    placements = (Placement(clip_name="hi", start_s=0.5, role="greeting", gain=0.5),)
    quiet = compose_trial(
        Trial(name="quiet", duration_s=1.0, placements=placements),
        clip_samples_by_name,
    )
    expected = np.zeros(SAMPLE_RATE, dtype=np.int16)
    expected[SAMPLE_RATE // 2 : SAMPLE_RATE // 2 + 800] = 500
    assert np.array_equal(quiet, expected)

    noisy_trial = Trial(
        name="noisy", duration_s=1.0, placements=placements, noise_dbfs=-50.0
    )
    noisy = compose_trial(noisy_trial, clip_samples_by_name)
    assert np.array_equal(noisy, compose_trial(noisy_trial, clip_samples_by_name))
    noise = noisy[: SAMPLE_RATE // 2].astype(np.float64)
    assert 20 * math.log10(math.sqrt(np.mean(noise**2)) / 32768) == pytest.approx(
        -50.0, abs=0.2
    )
    with pytest.raises(ValueError, match="runs past the end of late"):
        compose_trial(
            Trial(
                name="late",
                duration_s=1.0,
                placements=(Placement(clip_name="hi", start_s=0.98, role="greeting"),),
            ),
            clip_samples_by_name,
        )


def write_short_clip_inputs(tmp_path: Path) -> Path:
    clips_dir = tmp_path / "clips"
    clips_dir.mkdir()
    for clip_name in CLIPS:
        soundfile.write(
            str(clips_dir / f"{clip_name}.wav"),
            np.full(SAMPLE_RATE // 4, 1000, dtype=np.int16),
            SAMPLE_RATE,
            "PCM_16",
        )
    write_inputs(clips_dir, tmp_path / "inputs")
    return tmp_path / "inputs"


def test_record_keeps_finished_runs_and_records_failed_and_cut_off_ones_again(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failed session was kept as recorded, so the report scored a broken trace and a rerun never replaced it, even after a gap in the failed-run numbers; a cut-off one stopped the rerun."""
    inputs_dir = write_short_clip_inputs(tmp_path)
    recorded_names: list[str] = []

    async def fake_run(
        url: str,
        pcm: bytes,
        server: dict[str, JsonValue],
        *,
        output: Path,
        profile: str,
        timeout_s: float,
    ) -> dict[str, JsonValue]:
        run_dir = output
        run_dir.mkdir(parents=True)
        (run_dir / "manifest.json").write_text(
            json.dumps({"input": {"sha256": hashlib.sha256(pcm).hexdigest()}})
        )
        status = (
            "fail" if run_dir.name == "control_r1" and not recorded_names else "pass"
        )
        (run_dir / "report.json").write_text(
            json.dumps({"cases": [{"status": status}]})
        )
        recorded_names.append(run_dir.name)
        return {}

    monkeypatch.setattr(benchmark_duplex, "run", fake_run)
    arguments = argparse.Namespace(
        inputs=inputs_dir,
        output=tmp_path / "runs",
        url="ws://127.0.0.1:8000/v1/realtime",
        timeout=90.0,
        rounds=1,
        first_round=1,
    )
    asyncio.run(benchmark_minicpmo_baselines.record_rounds(arguments, {}))
    assert len(recorded_names) == len(TRIALS)
    assert not (tmp_path / "runs" / "control_r1").exists()
    assert (tmp_path / "runs" / "control_r1.failed-1" / "report.json").exists()

    asyncio.run(benchmark_minicpmo_baselines.record_rounds(arguments, {}))
    assert recorded_names[len(TRIALS) :] == ["control_r1"]

    runs_dir = tmp_path / "runs"
    (runs_dir / "long_o000_r1.failed-2").mkdir()
    (runs_dir / "long_o000_r1" / "report.json").write_text(
        json.dumps({"cases": [{"status": "fail"}]})
    )
    asyncio.run(benchmark_minicpmo_baselines.record_rounds(arguments, {}))
    assert recorded_names[len(TRIALS) + 1 :] == ["long_o000_r1"]
    assert (runs_dir / "long_o000_r1.failed-3" / "report.json").exists()

    (runs_dir / "short_o000_r1" / "report.json").unlink()
    asyncio.run(benchmark_minicpmo_baselines.record_rounds(arguments, {}))
    assert recorded_names[len(TRIALS) + 2 :] == ["short_o000_r1"]
    assert (runs_dir / "short_o000_r1.failed-1" / "manifest.json").exists()

    manifest_path = runs_dir / "short_o250_r1" / "manifest.json"
    recorded_manifest = manifest_path.read_text()
    manifest_path.write_text(json.dumps({"input": {"sha256": "0" * 64}}))
    with pytest.raises(
        ValueError, match="short_o250_r1 was recorded from another input"
    ):
        asyncio.run(benchmark_minicpmo_baselines.record_rounds(arguments, {}))
    manifest_path.write_text(recorded_manifest)

    soundfile.write(
        str(inputs_dir / "long_o250.wav"),
        np.zeros(SAMPLE_RATE, dtype=np.int16),
        SAMPLE_RATE,
        "PCM_16",
    )
    with pytest.raises(ValueError, match="long_o250.wav no longer matches"):
        asyncio.run(benchmark_minicpmo_baselines.record_rounds(arguments, {}))

    arguments.timeout = 25.0
    with pytest.raises(ValueError, match="timeout must exceed"):
        asyncio.run(benchmark_minicpmo_baselines.record_rounds(arguments, {}))


def test_tables_count_exclusions_and_runs_with_no_greeting_answered() -> None:
    trial_inputs = [
        TrialInput(
            name="short_o000",
            duration_s=30.0,
            sha256="",
            events=[
                SpokenEvent(role="prompt", start_s=1.0, end_s=3.0),
                SpokenEvent(role="stop", start_s=7.0, end_s=7.5),
            ],
        ),
        TrialInput(
            name="cold_soft_hi", duration_s=32.0, sha256="", events=GREETING_EVENTS
        ),
    ]
    replies_by_trial = {
        "short_o000": [
            [reply({3: "One, two,", 7: " three."})],
            [
                reply(
                    {
                        3: "One, two,",
                        7: " three,",
                        8: " four,",
                        9: " five,",
                        10: " six.",
                    }
                )
            ],
            [reply({3: COUNT_TO_THIRTY})],
        ],
        "cold_soft_hi": [
            [reply({3: "Hi."}), reply({24: "Canberra."})],
            [reply({24: "Canberra."})],
        ],
    }
    assert stop_table(trial_inputs, replies_by_trial).splitlines()[2] == (
        "| short_o000 | 2 | 1 | 2 | 0 | 1.5 / 2.7 / 3.0 |  |"
    )
    assert greeting_table(trial_inputs, replies_by_trial).splitlines()[2] == (
        "| cold_soft_hi | 2 | 1/4 | 1 | 2/2 | 0.2 |"
    )


def test_report_refuses_runs_and_logs_of_other_inputs(tmp_path: Path) -> None:
    """Rebuilt inputs moved the event times under runs and logs recorded from the old ones, and an empty source printed empty tables."""
    inputs_dir = write_short_clip_inputs(tmp_path)
    run_dir = tmp_path / "runs" / "control_r1"
    run_dir.mkdir(parents=True)
    (run_dir / "manifest.json").write_text(json.dumps({"input": {"sha256": "0" * 64}}))
    (run_dir / "report.json").write_text(json.dumps({"cases": [{"status": "pass"}]}))
    with pytest.raises(ValueError, match="control_r1 was recorded from another input"):
        benchmark_minicpmo_baselines.main(
            ["report", "--inputs", str(inputs_dir), "--runs", str(tmp_path / "runs")]
        )

    official_dir = tmp_path / "official"
    official_dir.mkdir()
    configuration_path = official_dir / "configuration.json"
    configuration_path.write_text(json.dumps({"trials_sha256": "0" * 64}))
    with pytest.raises(ValueError, match="ran other inputs"):
        benchmark_minicpmo_baselines.main(
            ["report", "--inputs", str(inputs_dir), "--official", str(official_dir)]
        )

    configuration_path.write_text(
        json.dumps(
            {
                "trials_sha256": hashlib.sha256(
                    (inputs_dir / "trials.json").read_bytes()
                ).hexdigest()
            }
        )
    )
    with pytest.raises(ValueError, match="No finished runs"):
        benchmark_minicpmo_baselines.main(
            ["report", "--inputs", str(inputs_dir), "--official", str(official_dir)]
        )
