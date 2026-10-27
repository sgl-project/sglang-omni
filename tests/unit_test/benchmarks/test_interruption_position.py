# SPDX-License-Identifier: Apache-2.0
"""Interruption-position trials: the simulated player, the run score, the tables and the report."""

from __future__ import annotations

import base64
import json
from pathlib import Path

import numpy as np
import pytest
from pydantic import JsonValue

from benchmarks.duplex.interruption_position import (
    TRANSCRIBED_WORDS,
    PositionScore,
    TranscribedWord,
    first_number,
    position_tables,
    recorded_replies,
    score_run,
)
from benchmarks.duplex.playback import ReplyAudioChunk, played_sample_count
from benchmarks.eval.benchmark_interruption_position import main

OUTPUT_RATE = 24000
COUNT_TEXT = "One, two, three, four."
# note (Jisen Li): Whisper ends each word where the next starts; the speech of each is its first 240 ms.
WORDS = [
    TranscribedWord(word=" One,", start_s=0.0, end_s=0.4),
    TranscribedWord(word=" two,", start_s=0.4, end_s=0.8),
    TranscribedWord(word=" three,", start_s=0.8, end_s=1.2),
    TranscribedWord(word=" four.", start_s=1.2, end_s=1.6),
]
SPEECH_SPANS_S = ((0.0, 0.24), (0.4, 0.64), (0.8, 1.04), (1.2, 1.44))


@pytest.mark.parametrize(
    ("chunks", "start_buffer_ms", "at_s", "played_samples"),
    [
        ([(0.0, 100), (0.05, 100)], 0.0, 0.12, 120),
        ([(0.0, 100), (0.05, 100)], 150.0, 0.12, 70),
        ([(0.0, 50), (0.2, 100)], 0.0, 0.25, 100),
        ([(0.0, 100), (0.2, 40), (0.25, 40)], 60.0, 0.3, 150),
        ([(0.0, 50)], 100.0, 1.0, 0),
        ([(0.0, 100), (0.5, 100)], 0.0, 0.3, 100),
    ],
)
def test_player_starts_after_its_buffer_and_buffers_again_after_running_dry(
    chunks: list[tuple[float, int]],
    start_buffer_ms: float,
    at_s: float,
    played_samples: int,
) -> None:
    assert (
        played_sample_count(
            [
                ReplyAudioChunk(received_s=received_s, sample_count=sample_count)
                for received_s, sample_count in chunks
            ],
            sample_rate=1000,
            start_buffer_ms=start_buffer_ms,
            at_s=at_s,
        )
        == played_samples
    )


@pytest.mark.parametrize(
    ("transcript", "number"),
    [
        ("The last number I said was seven.", 7),
        ("The last one was twenty-three.", 23),
        ("twenty one", 21),
        ("I said 14, so next is 15.", 14),
        ("I was counting to thirty and got to seven.", 7),
        ("One.", 1),
        ("I'm not sure where I was.", None),
    ],
)
def test_first_number_skips_one_and_the_count_target(
    transcript: str, number: int | None
) -> None:
    assert first_number(transcript) == number


def write_session(
    run_dir: Path,
    *,
    count_text: str = COUNT_TEXT,
    counting_status: str = "cancelled",
    cut_s: float = 11.125,
    truncated_ms: int | None = 1125,
    answer: str = "The last number was three.",
) -> None:
    """A counting reply of 2 s audio in two chunks from 10 s, cut at cut_s; a reply to the question's first words is cut too, then the answer comes."""
    records: list[dict[str, JsonValue]] = []

    def add(
        time_s: float, direction: str, event_type: str, **fields: JsonValue
    ) -> None:
        records.append(
            {
                "direction": direction,
                "time_s": time_s,
                "event": {
                    "type": event_type,
                    "event_id": f"event_{len(records)}",
                    **fields,
                },
            }
        )

    def reply_part(
        time_s: float, event_type: str, response_id: str, **fields: JsonValue
    ) -> None:
        add(
            time_s,
            "receive",
            event_type,
            response_id=response_id,
            item_id=f"item_{response_id}",
            output_index=0,
            content_index=0,
            **fields,
        )

    reply_samples = np.zeros(2 * OUTPUT_RATE, dtype="<i2")
    for start_s, end_s in SPEECH_SPANS_S:
        reply_samples[round(start_s * OUTPUT_RATE) : round(end_s * OUTPUT_RATE)] = 3000
    chunks = [
        base64.b64encode(chunk_samples.tobytes()).decode()
        for chunk_samples in np.split(reply_samples, 2)
    ]
    add(
        9.0,
        "receive",
        "response.created",
        response={"id": "resp_0", "status": "in_progress"},
    )
    reply_part(
        9.5, "response.output_audio_transcript.delta", "resp_0", delta=count_text
    )
    reply_part(10.0, "response.output_audio.delta", "resp_0", delta=chunks[0])
    reply_part(10.5, "response.output_audio.delta", "resp_0", delta=chunks[1])
    reply_part(10.9, "sglang.response.progress", "resp_0", text_end=9, audio_ms=700.0)
    add(
        cut_s,
        "receive",
        "response.done",
        response={"id": "resp_0", "status": counting_status},
    )
    if truncated_ms is not None:
        add(
            cut_s,
            "send",
            "conversation.item.truncate",
            item_id="item_resp_0",
            content_index=0,
            audio_end_ms=truncated_ms,
        )
    else:
        pass
    reply_part(11.3, "sglang.response.heard", "resp_0", text_end=16)
    add(
        12.0,
        "receive",
        "response.created",
        response={"id": "resp_wait", "status": "in_progress"},
    )
    reply_part(
        12.1, "response.output_audio_transcript.delta", "resp_wait", delta="Sure."
    )
    add(
        12.2,
        "receive",
        "response.done",
        response={"id": "resp_wait", "status": "cancelled"},
    )
    add(
        14.0,
        "receive",
        "response.created",
        response={"id": "resp_1", "status": "in_progress"},
    )
    reply_part(14.1, "response.output_audio_transcript.delta", "resp_1", delta=answer)
    add(
        14.5,
        "receive",
        "response.done",
        response={"id": "resp_1", "status": "completed"},
    )
    run_dir.mkdir(parents=True)
    (run_dir / "continuous.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records)
    )


@pytest.mark.parametrize(
    ("cut_s", "truncated_ms", "played_s", "is_mid_word"),
    [(11.125, 1125, 1.125, False), (11.25, 1250, 1.25, True)],
)
def test_a_cut_count_is_scored_against_the_speech_that_played(
    tmp_path: Path,
    cut_s: float,
    truncated_ms: int,
    played_s: float,
    is_mid_word: bool,
) -> None:
    """A cut in the pause after "three" has heard three, though Whisper's span of "three" runs to 1.2 s."""
    write_session(tmp_path / "run", cut_s=cut_s, truncated_ms=truncated_ms)
    assert score_run(
        recorded_replies(tmp_path / "run" / "continuous.jsonl"),
        WORDS,
        start_buffer_ms=0.0,
    ) == PositionScore(
        status="scored",
        played_s=played_s,
        delivered_s=2.0,
        heard_count=3,
        delivered_count=4,
        progress_count=2,
        kept_count=3,
        answer_number=3,
        is_mid_word=is_mid_word,
        is_progress_behind=True,
    )


@pytest.mark.parametrize(
    ("session", "status"),
    [
        ({"counting_status": "completed", "truncated_ms": None}, "not interrupted"),
        ({"count_text": "Sure, here we go."}, "no count"),
    ],
)
def test_a_count_that_was_not_cut_is_not_scored(
    tmp_path: Path, session: dict[str, JsonValue], status: str
) -> None:
    write_session(tmp_path / "run", **session)
    score = score_run(
        recorded_replies(tmp_path / "run" / "continuous.jsonl"),
        WORDS,
        start_buffer_ms=0.0,
    )
    assert score == PositionScore(status=status)


def test_a_truncate_from_another_start_buffer_is_refused(tmp_path: Path) -> None:
    write_session(tmp_path / "run")
    with pytest.raises(
        ValueError, match="not where a 1500.0 ms start buffer had played"
    ):
        score_run(
            recorded_replies(tmp_path / "run" / "continuous.jsonl"),
            WORDS,
            start_buffer_ms=1500.0,
        )


def test_tables_compare_history_and_answers_with_what_was_heard() -> None:
    def cut(
        heard_count: int,
        answer_number: int | None,
        *,
        kept_count: int | None = None,
        progress_count: int | None = None,
        is_mid_word: bool = False,
        is_progress_behind: bool | None = None,
    ) -> PositionScore:
        return PositionScore(
            status="scored",
            played_s=1.0,
            delivered_s=2.0,
            heard_count=heard_count,
            delivered_count=heard_count + 2,
            progress_count=progress_count,
            kept_count=kept_count,
            answer_number=answer_number,
            is_mid_word=is_mid_word,
            is_progress_behind=is_progress_behind,
        )

    tables = position_tables(
        {
            "truncating": [
                cut(3, 3, kept_count=3, progress_count=3, is_progress_behind=False),
                cut(
                    3,
                    4,
                    kept_count=4,
                    progress_count=2,
                    is_mid_word=True,
                    is_progress_behind=True,
                ),
                cut(5, 7, kept_count=4, progress_count=5, is_progress_behind=False),
                PositionScore(status="not interrupted"),
            ],
            "silent": [cut(2, 1), cut(2, None)],
        }
    ).splitlines()
    assert tables[2:4] == [
        "| truncating | 3 | 1 | 1/3 | 1/3 | 1/3 | 1/3 | 2/3 | 2.0 / 2.0 / 2.0 |",
        "| silent | 2 | 0 |  |  |  |  |  | 2.0 / 2.0 / 2.0 |",
    ]
    assert tables[7:] == [
        "| truncating | 3 | 1 | 1 | 1 | 0 | 0 |",
        "| silent | 2 | 0 | 0 | 0 | 1 | 1 |",
    ]


def write_run(runs_dir: Path, run_name: str, input_sha256: str) -> Path:
    run_dir = runs_dir / run_name
    write_session(run_dir)
    (run_dir / "manifest.json").write_text(
        json.dumps(
            {
                "input": {"sha256": input_sha256},
                "config": {
                    "playback": {"start_buffer_ms": 0.0, "sends_truncate": True}
                },
            }
        )
    )
    (run_dir / "report.json").write_text(json.dumps({"cases": [{"status": "pass"}]}))
    return run_dir


def test_report_scores_transcribed_runs_and_refuses_other_inputs(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    inputs_dir = tmp_path / "inputs"
    inputs_dir.mkdir()
    (inputs_dir / "trials.json").write_text(
        json.dumps(
            [
                {
                    "name": "position_o000",
                    "duration_s": 30.0,
                    "sha256": "a" * 64,
                    "events": [],
                }
            ]
        )
    )
    words_dir = tmp_path / "words"
    words_dir.mkdir()
    write_run(tmp_path / "runs", "position_o000_truncating_r1", "a" * 64)
    (words_dir / "position_o000_truncating_r1.json").write_bytes(
        TRANSCRIBED_WORDS.dump_json(WORDS)
    )
    write_run(tmp_path / "runs", "position_o000_silent_r1", "a" * 64)
    arguments = [
        "report",
        "--inputs",
        str(inputs_dir),
        "--runs",
        str(tmp_path / "runs"),
        "--words",
        str(words_dir),
    ]

    assert main(arguments) == 0
    output = capsys.readouterr().out
    assert (
        "| truncating | 1 | 0 | 1/1 | 1/1 | 0/1 | 0/1 | 0/1 | 1.0 / 1.0 / 1.0 |"
        in output
    )
    assert (
        "Not transcribed, so left out of the tables: position_o000_silent_r1" in output
    )

    write_run(tmp_path / "runs", "position_o000_truncating_r2", "b" * 64)
    with pytest.raises(ValueError, match="recorded from another input"):
        main(arguments)
