# SPDX-License-Identifier: Apache-2.0
"""Scores of the MiniCPM-o baseline trials on the input unit clock, for sglang-omni sessions and official-loop logs."""

from __future__ import annotations

import base64
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, NamedTuple

from pydantic import BaseModel, ConfigDict

from benchmarks.duplex.client import PACKET_MS, SEND_RECEIPTS_FILE
from benchmarks.duplex.minicpmo_trials import SpokenEvent, TrialInput
from benchmarks.duplex.oracle_models import TraceRecord, WireEvent
from benchmarks.duplex.profiles import PROFILES, ProfileName
from benchmarks.duplex.serving import SCORED_STATUSES, quantiles, score_session
from benchmarks.duplex.v15_audio import AUDIO_DELTA, TRANSCRIPT_DELTAS

PROFILE: ProfileName = "minicpmo-native-pr2377"
UNIT_DURATION_S = PROFILES[PROFILE].native_unit_ms / 1000
OUTPUT_SAMPLE_RATE = PROFILES[PROFILE].output_sample_rate
PCM16_BYTES = 2
COUNT_TARGET = 30
# note (Jisen Li): #2895 reads a stop as working when the reply ends within about 3 s; the window starts when the stop's unit ends, so every offset gets the same time to react.
STOP_WITHIN_S = 3.0
# note (Jisen Li): A count past twenty when the stop arrives ends on its own about as soon as a stop would end it.
MAX_COUNT_AT_STOP = 20
# note (Jisen Li): #2896 saw answered greetings start within 2 s; 4 s keeps slow answers apart from none.
REPLY_WINDOW_S = 4.0
NUMBER_WORDS = {
    word: value
    for value, word in enumerate(
        "zero one two three four five six seven eight nine ten eleven twelve "
        "thirteen fourteen fifteen sixteen seventeen eighteen nineteen".split()
    )
}
TENS_WORDS = {"twenty": 20, "thirty": 30}

StopStatus = Literal[
    "kept", "no count", "started after stop", "ended before stop", "past twenty"
]


class OfficialUnit(BaseModel):
    model_config = ConfigDict(extra="forbid")
    unit_index: int
    is_listen: bool
    text: str
    end_of_turn: bool


class ReplyUnit(NamedTuple):
    start_s: float
    text: str


class AudioArrival(NamedTuple):
    arrival_s: float
    audio_s: float


@dataclass(frozen=True, kw_only=True)
class Reply:
    """A reply's speaking units on the input clock, and when its last audio finished arriving at the client."""

    units: tuple[ReplyUnit, ...]
    audio_end_s: float | None = None


@dataclass(frozen=True, kw_only=True)
class StopScore:
    status: StopStatus
    counted_to_thirty: bool
    reply_end_after_stop_unit_s: float | None = None
    delivered_end_after_stop_s: float | None = None


def run_passed(run_dir: Path) -> bool:
    """Whether every case of a recorded run kept the protocol, so its trace is whole enough to score."""
    run_report = json.loads((run_dir / "report.json").read_text())
    return all(case["status"] in SCORED_STATUSES for case in run_report["cases"])


def sglang_replies(run_dir: Path) -> list[Reply]:
    """Replies of one recorded session, in order of their first unit."""
    first_send_s = json.loads((run_dir / SEND_RECEIPTS_FILE).read_text())["appends"][0][
        "start_s"
    ]
    trace_records = [
        TraceRecord.model_validate(json.loads(line))
        for line in (run_dir / "continuous.jsonl").read_text().splitlines()
    ]
    texts_by_response: dict[str, dict[float, str]] = {}
    audio_arrivals_by_response: dict[str, list[AudioArrival]] = {}
    for trace_record in (
        trace_record
        for trace_record in trace_records
        if trace_record.direction == "receive"
    ):
        event = WireEvent.model_validate(trace_record.event)
        if event.type == AUDIO_DELTA:
            assert event.response_id is not None and event.delta is not None
            assert event.sglang.media_time is not None
            texts_by_response.setdefault(event.response_id, {}).setdefault(
                event.sglang.media_time.t_start_ms / 1000, ""
            )
            audio_arrivals_by_response.setdefault(event.response_id, []).append(
                AudioArrival(
                    trace_record.time_s - first_send_s,
                    len(base64.b64decode(event.delta))
                    / PCM16_BYTES
                    / OUTPUT_SAMPLE_RATE,
                )
            )
        elif event.type in TRANSCRIPT_DELTAS:
            assert event.response_id is not None and event.delta is not None
            assert event.sglang.media_time is not None
            texts_by_unit_start_s = texts_by_response.setdefault(event.response_id, {})
            unit_start_s = event.sglang.media_time.t_start_ms / 1000
            texts_by_unit_start_s[unit_start_s] = (
                texts_by_unit_start_s.get(unit_start_s, "") + event.delta
            )
        else:
            pass
    replies = []
    for response_id, texts_by_unit_start_s in texts_by_response.items():
        audio_arrivals = audio_arrivals_by_response.get(response_id, [])
        replies.append(
            Reply(
                units=tuple(
                    ReplyUnit(*unit) for unit in sorted(texts_by_unit_start_s.items())
                ),
                audio_end_s=(
                    audio_arrivals[-1].arrival_s + audio_arrivals[-1].audio_s
                    if audio_arrivals
                    else None
                ),
            )
        )
    return sorted(replies, key=lambda reply: reply.units[0].start_s)


def official_replies(log_path: Path) -> list[Reply]:
    """Replies of one official-loop log, split as the session splits responses.

    A speaking unit with text opens a reply, any speaking unit joins the open one, a listen unit leaves it open, and an end of turn closes it.
    """
    replies: list[list[ReplyUnit]] = []
    is_reply_open = False
    for line in log_path.read_text().splitlines():
        unit = OfficialUnit.model_validate_json(line)
        unit_start_s = unit.unit_index * UNIT_DURATION_S
        if unit.is_listen:
            pass
        elif is_reply_open:
            replies[-1].append(ReplyUnit(unit_start_s, unit.text))
        elif unit.text:
            replies.append([ReplyUnit(unit_start_s, unit.text)])
        else:
            pass
        is_reply_open = (
            is_reply_open or (not unit.is_listen and bool(unit.text))
        ) and not unit.end_of_turn
    return [Reply(units=tuple(units)) for units in replies]


def reply_transcript(reply: Reply) -> str:
    return "".join(unit.text for unit in reply.units)


def counted_to(transcript: str) -> int:
    """Highest n counted as 1, 2, ..., n in order, so "count from one to thirty" in a preamble does not count."""
    tokens = re.findall(r"[a-z]+|\d+", transcript.lower().replace("-", " "))
    count = 0
    index = 0
    while index < len(tokens):
        token = tokens[index]
        next_token = tokens[index + 1] if index + 1 < len(tokens) else ""
        ones_value = NUMBER_WORDS.get(next_token, 0)
        # note (Jisen Li): Deltas can split "10" into "1 0", and "twenty-six" can come as "twenty six"; a pair is joined only when that continues the count.
        if (
            token.isdigit()
            and next_token.isdigit()
            and int(token + next_token) == count + 1
        ) or (
            token in TENS_WORDS
            and 0 < ones_value < 10
            and TENS_WORDS[token] + ones_value == count + 1
        ):
            count += 1
            index += 2
        elif token.isdigit() and int(token) == count + 1:
            count += 1
            index += 1
        elif TENS_WORDS.get(token, NUMBER_WORDS.get(token)) == count + 1:
            count += 1
            index += 1
        else:
            index += 1
    return count


def score_stop(stop_s: float | None, replies: list[Reply]) -> StopScore:
    """Score the counting reply, the first that counts at least to one, against a stop whose onset is at stop_s, or without one."""
    counting_reply = next(
        (reply for reply in replies if counted_to(reply_transcript(reply)) > 0),
        None,
    )
    if counting_reply is None:
        return StopScore(status="no count", counted_to_thirty=False)
    else:
        pass
    counted_to_thirty = counted_to(reply_transcript(counting_reply)) >= COUNT_TARGET
    if stop_s is None:
        return StopScore(status="kept", counted_to_thirty=counted_to_thirty)
    else:
        pass
    reply_start_s = counting_reply.units[0].start_s
    reply_end_s = counting_reply.units[-1].start_s + UNIT_DURATION_S
    stop_unit_end_s = (math.floor(stop_s / UNIT_DURATION_S) + 1) * UNIT_DURATION_S
    count_at_stop = counted_to(
        "".join(unit.text for unit in counting_reply.units if unit.start_s < stop_s)
    )
    if reply_start_s >= stop_s:
        status = "started after stop"
    elif reply_end_s <= stop_s:
        status = "ended before stop"
    elif count_at_stop > MAX_COUNT_AT_STOP:
        status = "past twenty"
    else:
        status = "kept"
    return StopScore(
        status=status,
        counted_to_thirty=counted_to_thirty,
        reply_end_after_stop_unit_s=reply_end_s - stop_unit_end_s,
        delivered_end_after_stop_s=(
            counting_reply.audio_end_s - stop_s
            if counting_reply.audio_end_s is not None
            else None
        ),
    )


def score_events(events: list[SpokenEvent], replies: list[Reply]) -> list[float | None]:
    """Per event, seconds from its end to the end of the first unit of the reply that answers it, if any.

    A reply answers an event when its first unit ends after the event starts and begins within REPLY_WINDOW_S of the event's end; each reply answers at most one event.
    """
    unused_replies = list(replies)
    reply_after_end_s: list[float | None] = []
    for event in events:
        answer = next(
            (
                reply
                for reply in unused_replies
                if reply.units[0].start_s + UNIT_DURATION_S > event.start_s
                and reply.units[0].start_s <= event.end_s + REPLY_WINDOW_S
            ),
            None,
        )
        if answer is None:
            reply_after_end_s.append(None)
        else:
            unused_replies.remove(answer)
            reply_after_end_s.append(
                answer.units[0].start_s + UNIT_DURATION_S - event.end_s
            )
    return reply_after_end_s


def format_quantiles(values: list[float]) -> str:
    if values:
        summary = quantiles(values)
        return f"{summary['p50']:.1f} / {summary['p90']:.1f} / {summary['max']:.1f}"
    else:
        return ""


def stop_table(
    trial_inputs: list[TrialInput], replies_by_trial: dict[str, list[list[Reply]]]
) -> str:
    rows = [
        f"| Trial | Kept | Excluded | Stopped within {STOP_WITHIN_S:g} s of the stop's unit | Counted to {COUNT_TARGET} | "
        "Reply end after the stop's unit p50 / p90 / max (s) | Delivered audio end after the stop p50 / p90 / max (s) |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for trial_input in trial_inputs:
        roles = [event.role for event in trial_input.events]
        runs = replies_by_trial.get(trial_input.name, [])
        if "prompt" in roles and runs:
            stop_s = next(
                (event.start_s for event in trial_input.events if event.role == "stop"),
                None,
            )
            stop_scores = [score_stop(stop_s, replies) for replies in runs]
            kept_scores = [score for score in stop_scores if score.status == "kept"]
            reply_ends_after_stop_unit_s = [
                score.reply_end_after_stop_unit_s
                for score in kept_scores
                if score.reply_end_after_stop_unit_s is not None
            ]
            delivered_ends_after_stop_s = [
                score.delivered_end_after_stop_s
                for score in kept_scores
                if score.delivered_end_after_stop_s is not None
            ]
            stopped = (
                str(
                    sum(
                        seconds <= STOP_WITHIN_S
                        for seconds in reply_ends_after_stop_unit_s
                    )
                )
                if stop_s is not None
                else ""
            )
            rows.append(
                f"| {trial_input.name} | {len(kept_scores)} | {len(stop_scores) - len(kept_scores)} | {stopped} | "
                f"{sum(score.counted_to_thirty for score in kept_scores)} | "
                f"{format_quantiles(reply_ends_after_stop_unit_s)} | {format_quantiles(delivered_ends_after_stop_s)} |"
            )
        else:
            pass
    return "\n".join(rows)


def greeting_table(
    trial_inputs: list[TrialInput], replies_by_trial: dict[str, list[list[Reply]]]
) -> str:
    rows = [
        "| Trial | Runs | Greetings answered | Runs with no greeting answered | Questions answered | Reply after question end p50 (s) |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for trial_input in trial_inputs:
        roles = [event.role for event in trial_input.events]
        runs = replies_by_trial.get(trial_input.name, [])
        if "prompt" not in roles and runs:
            greetings_after_end_s: list[float | None] = []
            questions_after_end_s: list[float | None] = []
            unanswered_runs = 0
            for replies in runs:
                events_after_end_s = score_events(trial_input.events, replies)
                run_greetings_after_end_s = [
                    seconds
                    for role, seconds in zip(roles, events_after_end_s)
                    if role == "greeting"
                ]
                greetings_after_end_s += run_greetings_after_end_s
                questions_after_end_s += [
                    seconds
                    for role, seconds in zip(roles, events_after_end_s)
                    if role == "question"
                ]
                unanswered_runs += bool(run_greetings_after_end_s) and all(
                    seconds is None for seconds in run_greetings_after_end_s
                )
            answered_questions_s = [
                seconds for seconds in questions_after_end_s if seconds is not None
            ]
            if greetings_after_end_s:
                greetings_answered = f"{sum(seconds is not None for seconds in greetings_after_end_s)}/{len(greetings_after_end_s)}"
                unanswered = str(unanswered_runs)
            else:
                greetings_answered = ""
                unanswered = ""
            question_p50 = (
                f"{quantiles(answered_questions_s)['p50']:.1f}"
                if answered_questions_s
                else ""
            )
            rows.append(
                f"| {trial_input.name} | {len(runs)} | {greetings_answered} | {unanswered} | "
                f"{len(answered_questions_s)}/{len(questions_after_end_s)} | {question_p50} |"
            )
        else:
            pass
    return "\n".join(rows)


def serving_table(run_dirs: list[Path], playback_reserve_s: float) -> str:
    """Unit lag and first-audio latency as benchmarks.duplex.serving scores a session, and how far each packet send strayed from its schedule."""
    unit_lags_s: list[float] = []
    first_audio_latencies_s: list[float] = []
    send_deviations_ms: list[float] = []
    units_missed = 0
    for run_dir in run_dirs:
        session_score = score_session(run_dir, PROFILE, playback_reserve_s)
        unit_lags_s += session_score["unit_lag_s"]
        units_missed += session_score["units_missed"]
        first_audio_latencies_s += [
            reply["first_audio_latency_s"] for reply in session_score["replies"]
        ]
        appends = json.loads((run_dir / SEND_RECEIPTS_FILE).read_text())["appends"]
        send_deviations_ms += [
            abs(
                append["start_s"]
                - appends[0]["start_s"]
                - append["seq"] * PACKET_MS / 1000
            )
            * 1000
            for append in appends
        ]
    if unit_lags_s:
        unit_lag_quantiles = quantiles(unit_lags_s)
        unit_lag_column = (
            f"{unit_lag_quantiles['p50']:.2f} / {unit_lag_quantiles['p90']:.2f} / "
            f"{unit_lag_quantiles['p99']:.2f} / {unit_lag_quantiles['max']:.2f}"
        )
    else:
        unit_lag_column = ""
    if first_audio_latencies_s:
        first_audio_latency_quantiles = quantiles(first_audio_latencies_s)
        first_audio_latency_column = (
            f"{first_audio_latency_quantiles['p50']:.2f} / {first_audio_latency_quantiles['p90']:.2f} / "
            f"{first_audio_latency_quantiles['max']:.2f}"
        )
    else:
        first_audio_latency_column = ""
    if send_deviations_ms:
        send_deviation_quantiles = quantiles(send_deviations_ms)
        send_deviation_column = f"{send_deviation_quantiles['p99']:.1f} / {send_deviation_quantiles['max']:.1f}"
    else:
        send_deviation_column = ""
    return "\n".join(
        [
            "| Sessions | Units | Units missed | Unit lag p50 / p90 / p99 / max (s) | First audio latency p50 / p90 / max (s) | "
            f"Packets | Send deviation from the {PACKET_MS} ms schedule p99 / max (ms) |",
            "| --- | --- | --- | --- | --- | --- | --- |",
            f"| {len(run_dirs)} | {len(unit_lags_s)} | {units_missed} | {unit_lag_column} | {first_audio_latency_column} | "
            f"{len(send_deviations_ms)} | {send_deviation_column} |",
        ]
    )
