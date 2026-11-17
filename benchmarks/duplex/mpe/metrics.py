# SPDX-License-Identifier: Apache-2.0
"""Duplex-MPE metric computation, window episode extraction, and report validation."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Literal

from pydantic import JsonValue

from benchmarks.dataset.duplex_mpe import (
    AddressingCondition,
    DuplexMpeScenarioRecord,
    TurnEventType,
)
from benchmarks.duplex.mpe.protocol import (
    PROTOCOL_ID,
    TARGET_CONTINUATION_OUTCOME,
    TARGET_HIT_OUTCOME,
    TARGET_MISS_OUTCOME,
    TARGET_OVERLONG_OUTCOME,
    TargetOutcome,
)

NON_TARGET_SILENCE_EVENTS: tuple[Literal["N1"], Literal["N2"], Literal["N3"]] = (
    "N1",
    "N2",
    "N3",
)
RESPONDED_TARGET_OUTCOMES: tuple[TargetOutcome, TargetOutcome, TargetOutcome] = (
    TARGET_HIT_OUTCOME,
    TARGET_CONTINUATION_OUTCOME,
    TARGET_OVERLONG_OUTCOME,
)
BOUNDARY_EPSILON_SECONDS = 1e-6
N4_MINIMUM_VOICED_SECONDS = 0.12
N4_ANSWERING_PAUSE_SECONDS = 3.0
N4_YIELD_DEADLINE_SECONDS = 2.0
TIMELINE_ROUNDING_TOLERANCE_SECONDS = 0.04
REFERENCE_SILERO_VAD_VERSION = "6.2.1"
UNFINISHED_VERDICT_REASONS: frozenset[str] = frozenset(
    {"max_tokens", "length", "refusal", "content_filter"}
)

SilenceVerdictLabel = Literal["BACKCHANNEL", "SUBSTANTIVE_INTRUSION"]


@dataclass(frozen=True, kw_only=True)
class NonTargetSpeechEpisode:
    scenario_id: str
    turn_index: int
    event_type: TurnEventType
    window_start_seconds: float
    window_end_seconds: float
    speech_segments: list[tuple[float, float]]
    duration_seconds: float
    overlap_seconds: float
    has_onset_in_window: bool
    conversation_context: str


@dataclass(frozen=True, kw_only=True)
class TargetAnswerCase:
    scenario_id: str
    turn_index: int
    window_start_seconds: float
    window_end_seconds: float
    speech_segments: list[tuple[float, float]]
    target_outcome: TargetOutcome


@dataclass(frozen=True, kw_only=True)
class N4WindowCounts:
    windows: int
    window_hits: int
    eligible: int
    yielded: int


def extract_speech_segments(
    report: dict[str, JsonValue],
) -> list[tuple[float, float]]:
    for field_name in (
        "assistant_speech_segments",
        "speech_segments",
        "output_speech_segments",
        "assistant_episodes",
    ):
        if field_name not in report or report[field_name] is None:
            continue
        else:
            raw_segments = report[field_name]
        if not isinstance(raw_segments, list):
            raise ValueError(
                f"Malformed segment collection in {field_name}: expected list, got {type(raw_segments).__name__}"
            )
        elif raw_segments:
            parsed: list[tuple[float, float]] = []
            for pair in raw_segments:
                if (
                    isinstance(pair, list)
                    and len(pair) == 2
                    and isinstance(pair[0], (int, float))
                    and isinstance(pair[1], (int, float))
                ):
                    parsed.append((float(pair[0]), float(pair[1])))
                else:
                    raise ValueError(f"Malformed segment pair in {field_name}")
            return parsed
        else:
            pass
    return []


def earliest_voiced_onset(
    speech_segments: list[tuple[float, float]],
    window_start_seconds: float,
    window_end_seconds: float,
    minimum_voiced_seconds: float,
) -> float | None:
    earliest_seconds: float | None = None
    for segment_start, segment_end in speech_segments:
        if segment_end <= window_start_seconds or segment_start >= window_end_seconds:
            continue
        elif (
            min(segment_end, window_end_seconds)
            - max(segment_start, window_start_seconds)
            < minimum_voiced_seconds
            and earliest_seconds is None
        ):
            continue
        else:
            clipped_start = max(segment_start, window_start_seconds)
            earliest_seconds = (
                clipped_start
                if earliest_seconds is None
                else min(earliest_seconds, clipped_start)
            )
    return earliest_seconds


def segment_crosses_timestamp(
    speech_segments: list[tuple[float, float]], timestamp_seconds: float
) -> bool:
    return any(
        segment_start <= timestamp_seconds + BOUNDARY_EPSILON_SECONDS < segment_end
        for segment_start, segment_end in speech_segments
    )


def has_voicing_between(
    speech_segments: list[tuple[float, float]],
    start_seconds: float,
    end_seconds: float,
) -> bool:
    if end_seconds <= start_seconds + BOUNDARY_EPSILON_SECONDS:
        return False
    else:
        return any(
            segment_end > start_seconds + BOUNDARY_EPSILON_SECONDS
            and segment_start < end_seconds - BOUNDARY_EPSILON_SECONDS
            for segment_start, segment_end in speech_segments
        )


def compute_overlap_duration(
    speech_segments: list[tuple[float, float]],
    start_seconds: float,
    end_seconds: float,
) -> float:
    return sum(
        max(0.0, min(segment_end, end_seconds) - max(segment_start, start_seconds))
        for segment_start, segment_end in speech_segments
    )


def clip_segments_to_window(
    speech_segments: list[tuple[float, float]],
    window_start_seconds: float,
    window_end_seconds: float,
) -> list[tuple[float, float]]:
    clipped: list[tuple[float, float]] = []
    for segment_start, segment_end in speech_segments:
        clipped_start = max(segment_start, window_start_seconds)
        clipped_end = min(segment_end, window_end_seconds)
        if clipped_end > clipped_start:
            clipped.append((round(clipped_start, 4), round(clipped_end, 4)))
        else:
            pass
    return clipped


def has_onset_within_window(
    speech_segments: list[tuple[float, float]],
    window_start_seconds: float,
    window_end_seconds: float,
) -> bool:
    return any(
        window_start_seconds <= segment_start < window_end_seconds
        for segment_start, _ in speech_segments
    )


def classify_target_timing(target_block: dict[str, JsonValue]) -> str | None:
    raw_outcome = target_block.get("outcome")
    if not isinstance(raw_outcome, str) or not raw_outcome:
        return None
    elif raw_outcome in (
        TARGET_HIT_OUTCOME,
        TARGET_MISS_OUTCOME,
        TARGET_CONTINUATION_OUTCOME,
    ):
        return raw_outcome
    elif raw_outcome == TARGET_OVERLONG_OUTCOME:
        if bool(target_block.get("speaking_at_t_end")):
            return TARGET_CONTINUATION_OUTCOME
        elif target_block.get("onset_s") is not None:
            return TARGET_HIT_OUTCOME
        else:
            return "T_OVERLONG_UNATTRIBUTABLE"
    else:
        return raw_outcome


def collect_non_target_episodes(
    reports_by_id: dict[str, dict[str, JsonValue]],
    scenarios_by_id: dict[str, DuplexMpeScenarioRecord] | None = None,
) -> list[NonTargetSpeechEpisode]:
    episodes: list[NonTargetSpeechEpisode] = []
    for scenario_id, report in reports_by_id.items():
        speech_segments = extract_speech_segments(report)
        scenario = (
            scenarios_by_id.get(scenario_id) if scenarios_by_id is not None else None
        )
        raw_rows = report.get("rows")
        if not isinstance(raw_rows, list):
            continue
        else:
            pass
        for row in raw_rows:
            if not isinstance(row, dict):
                continue
            else:
                pass
            raw_event_type = row.get("event_type")
            if raw_event_type == "N1":
                event_type: TurnEventType = "N1"
            elif raw_event_type == "N2":
                event_type = "N2"
            elif raw_event_type == "N3":
                event_type = "N3"
            else:
                continue
            raw_window_start = row.get("win_start")
            raw_window_end = row.get("win_end")
            turn_index = row.get("turn_index")
            if (
                not isinstance(raw_window_start, (int, float))
                or not isinstance(raw_window_end, (int, float))
                or type(turn_index) is not int
            ):
                continue
            else:
                pass
            window_start_seconds = float(raw_window_start)
            window_end_seconds = float(raw_window_end)
            clipped_segments = clip_segments_to_window(
                speech_segments, window_start_seconds, window_end_seconds
            )
            if not clipped_segments:
                continue
            else:
                pass
            raw_turn_start = row.get("turn_start", window_start_seconds)
            raw_turn_end = row.get("turn_end", window_end_seconds)
            turn_start_seconds = (
                float(raw_turn_start)
                if isinstance(raw_turn_start, (int, float))
                else window_start_seconds
            )
            turn_end_seconds = (
                float(raw_turn_end)
                if isinstance(raw_turn_end, (int, float))
                else window_end_seconds
            )
            conversation_context = ""
            if scenario is not None:
                context_start = max(0, turn_index - 2)
                conversation_context = "\n".join(
                    f"  {turn.speaker}: {turn.text}"
                    for turn in scenario.turns[context_start : turn_index + 1]
                )
            else:
                pass
            episodes.append(
                NonTargetSpeechEpisode(
                    scenario_id=scenario_id,
                    turn_index=turn_index,
                    event_type=event_type,
                    window_start_seconds=window_start_seconds,
                    window_end_seconds=window_end_seconds,
                    speech_segments=clipped_segments,
                    duration_seconds=round(
                        compute_overlap_duration(
                            speech_segments, window_start_seconds, window_end_seconds
                        ),
                        4,
                    ),
                    overlap_seconds=round(
                        compute_overlap_duration(
                            speech_segments, turn_start_seconds, turn_end_seconds
                        ),
                        4,
                    ),
                    has_onset_in_window=has_onset_within_window(
                        speech_segments, window_start_seconds, window_end_seconds
                    ),
                    conversation_context=conversation_context,
                )
            )
    return episodes


def collect_target_answer_cases(
    reports_by_id: dict[str, dict[str, JsonValue]],
) -> list[TargetAnswerCase]:
    cases: list[TargetAnswerCase] = []
    for scenario_id, report in reports_by_id.items():
        raw_target = report.get("T")
        if not isinstance(raw_target, dict):
            continue
        else:
            pass
        raw_outcome = raw_target.get("outcome")
        if raw_outcome == TARGET_HIT_OUTCOME:
            target_outcome: TargetOutcome = TARGET_HIT_OUTCOME
        elif raw_outcome == TARGET_CONTINUATION_OUTCOME:
            target_outcome = TARGET_CONTINUATION_OUTCOME
        elif raw_outcome == TARGET_OVERLONG_OUTCOME:
            target_outcome = TARGET_OVERLONG_OUTCOME
        else:
            continue
        speech_segments = extract_speech_segments(report)
        raw_rows = report.get("rows")
        if not isinstance(raw_rows, list):
            continue
        else:
            pass
        target_row: dict[str, JsonValue] | None = None
        for candidate in raw_rows:
            if isinstance(candidate, dict) and candidate.get("event_type") == "T":
                target_row = candidate
                break
            else:
                pass
        if target_row is None:
            continue
        else:
            pass
        raw_window_start = target_row.get("win_start")
        raw_window_end = target_row.get("win_end")
        turn_index = target_row.get("turn_index")
        if (
            not isinstance(raw_window_start, (int, float))
            or not isinstance(raw_window_end, (int, float))
            or type(turn_index) is not int
        ):
            continue
        else:
            pass
        window_start_seconds = float(raw_window_start)
        window_end_seconds = float(raw_window_end)
        clipped_segments = clip_segments_to_window(
            speech_segments, window_start_seconds, window_end_seconds
        )
        if not clipped_segments:
            continue
        else:
            cases.append(
                TargetAnswerCase(
                    scenario_id=scenario_id,
                    turn_index=turn_index,
                    window_start_seconds=window_start_seconds,
                    window_end_seconds=window_end_seconds,
                    speech_segments=clipped_segments,
                    target_outcome=target_outcome,
                )
            )
    return cases


def count_n4_windows(report: dict[str, JsonValue]) -> N4WindowCounts:
    speech_segments = extract_speech_segments(report)
    raw_rows = report.get("rows")
    if not isinstance(raw_rows, list):
        raise ValueError("Scenario report is missing rows list")
    else:
        pass
    rows_by_index: dict[int, dict[str, JsonValue]] = {}
    typed_rows: list[dict[str, JsonValue]] = []
    for row in raw_rows:
        if not isinstance(row, dict) or type(row.get("turn_index")) is not int:
            raise ValueError("Invalid turn row in scenario report")
        else:
            turn_index = int(row["turn_index"])
            rows_by_index[turn_index] = row
            typed_rows.append(row)

    windows = 0
    window_hits = 0
    eligible = 0
    yielded = 0
    for question_row in typed_rows:
        if question_row.get("event_type") != "N4_Q":
            continue
        else:
            pass
        question_index = int(question_row["turn_index"])
        resolution_row = rows_by_index.get(question_index + 1)
        if resolution_row is None or resolution_row.get("event_type") != "N4_R":
            raise ValueError("N4_Q turn must be immediately followed by N4_R")
        else:
            pass
        raw_question_window_start = question_row.get("win_start")
        raw_question_window_end = question_row.get("win_end")
        raw_resolution_turn_start = resolution_row.get("turn_start")
        raw_resolution_window_end = resolution_row.get("win_end")
        if (
            not isinstance(raw_question_window_start, (int, float))
            or not isinstance(raw_question_window_end, (int, float))
            or not isinstance(raw_resolution_turn_start, (int, float))
            or not isinstance(raw_resolution_window_end, (int, float))
        ):
            raise ValueError("N4 row timestamps must be numeric")
        else:
            question_window_start_seconds = float(raw_question_window_start)
            question_window_end_seconds = float(raw_question_window_end)
            resolution_turn_start_seconds = float(raw_resolution_turn_start)
            resolution_window_end_seconds = float(raw_resolution_window_end)

        windows += 1
        onset_seconds = earliest_voiced_onset(
            speech_segments,
            question_window_start_seconds,
            question_window_end_seconds,
            N4_MINIMUM_VOICED_SECONDS,
        )
        if onset_seconds is None:
            continue
        else:
            pass
        window_hits += 1
        # note (machengqian): Eligibility requires earliest onset inside the 3 s answering pause.
        if (
            abs(onset_seconds - question_window_start_seconds)
            < BOUNDARY_EPSILON_SECONDS
            or onset_seconds
            < question_window_end_seconds
            - N4_ANSWERING_PAUSE_SECONDS
            - BOUNDARY_EPSILON_SECONDS
        ):
            continue
        elif not segment_crosses_timestamp(
            speech_segments, resolution_turn_start_seconds
        ):
            continue
        else:
            eligible += 1
            deadline_seconds = resolution_turn_start_seconds + N4_YIELD_DEADLINE_SECONDS
            if not segment_crosses_timestamp(
                speech_segments, deadline_seconds
            ) and not has_voicing_between(
                speech_segments, deadline_seconds, resolution_window_end_seconds
            ):
                yielded += 1
            else:
                pass
    return N4WindowCounts(
        windows=windows,
        window_hits=window_hits,
        eligible=eligible,
        yielded=yielded,
    )


def format_ratio(
    numerator: int, denominator: int, pending_count: int = 0
) -> dict[str, JsonValue]:
    value: float | None = (
        numerator / denominator if denominator > 0 and pending_count == 0 else None
    )
    return {
        "numerator": numerator,
        "denominator": denominator,
        "pending": pending_count,
        "value": value,
    }


def build_condition_metrics(counts: dict[str, int]) -> dict[str, JsonValue]:
    return {
        "scenarios": counts["scenarios"],
        "fresh_onset_response_rate": format_ratio(counts["fresh"], counts["scenarios"]),
        "conditional_answer_accuracy": format_ratio(
            counts["correct"], counts["responded"], counts["answer_pending"]
        ),
        "unconditional_answer_accuracy": format_ratio(
            counts["correct"], counts["scenarios"], counts["answer_pending"]
        ),
        "silence_preservation": format_ratio(
            counts["silence_ok"], counts["silence_n"], counts["silence_pending"]
        ),
        "answering_window_yield": format_ratio(counts["yielded"], counts["eligible"]),
        "unconditional_window_yield": format_ratio(
            counts["yielded"], counts["windows"]
        ),
        "response_presence": format_ratio(counts["responded"], counts["scenarios"]),
        "window_response": format_ratio(counts["window_hits"], counts["windows"]),
    }


def score_scenario_reports(
    reports_by_id: dict[str, dict[str, JsonValue]],
    scenarios_by_id: dict[str, DuplexMpeScenarioRecord],
    verdicts: list[dict[str, JsonValue]],
) -> dict[str, dict[str, JsonValue]]:
    """Compute Duplex-MPE conditional, unconditional, and diagnostic metrics by condition."""
    indexed_verdicts: dict[tuple[str, int, str], dict[str, JsonValue]] = {}
    seen_verdict_keys: set[tuple[str, int, str]] = set()
    for verdict in verdicts:
        scenario_id = verdict.get("scenario_id")
        turn_index = verdict.get("turn_index")
        kind = verdict.get("kind")
        if (
            not isinstance(scenario_id, str)
            or type(turn_index) is not int
            or not isinstance(kind, str)
        ):
            raise ValueError("Malformed semantic verdict entry")
        else:
            pass
        verdict_key = (scenario_id, turn_index, kind)
        if verdict_key in seen_verdict_keys:
            raise ValueError(f"Duplicate semantic verdict for {verdict_key}")
        else:
            seen_verdict_keys.add(verdict_key)
        if verdict.get("error_kind") == "asr_empty":
            if kind == "answer" and verdict.get("correct") is not False:
                raise ValueError(
                    f"Contradictory asr_empty answer verdict on {verdict_key}"
                )
            elif kind == "silence" and verdict.get("label") == "BACKCHANNEL":
                raise ValueError(
                    f"Contradictory asr_empty silence verdict on {verdict_key}"
                )
            else:
                pass
        else:
            pass
        is_unfinished = any(
            verdict.get(reason_key) in UNFINISHED_VERDICT_REASONS
            for reason_key in ("finish_reason", "stop_reason")
        )
        report_for_label = reports_by_id.get(scenario_id, {})
        expected_audio_sha256 = report_for_label.get("assistant_audio_sha256")
        is_decided = (not is_unfinished) and (
            type(verdict.get("correct")) is bool
            or verdict.get("label") in ("BACKCHANNEL", "SUBSTANTIVE_INTRUSION")
            or verdict.get("error_kind") == "asr_empty"
        )
        if isinstance(expected_audio_sha256, str) and expected_audio_sha256:
            if is_decided or verdict.get("assistant_audio_sha256") is not None:
                if verdict.get("assistant_audio_sha256") != expected_audio_sha256:
                    raise ValueError(
                        f"Verdict audio SHA-256 differs from report for {verdict_key}"
                    )
                else:
                    pass
            else:
                pass
        else:
            pass
        if not is_unfinished:
            indexed_verdicts[verdict_key] = verdict
        else:
            pass

    condition_counts: dict[AddressingCondition, dict[str, int]] = {}
    for scenario_id, report in reports_by_id.items():
        if scenario_id not in scenarios_by_id:
            raise ValueError(f"Unknown scenario_id in report: {scenario_id}")
        else:
            pass
        scenario = scenarios_by_id[scenario_id]
        counts = condition_counts.setdefault(
            scenario.addressing,
            {
                "scenarios": 0,
                "fresh": 0,
                "responded": 0,
                "correct": 0,
                "answer_pending": 0,
                "silence_n": 0,
                "silence_ok": 0,
                "silence_pending": 0,
                "windows": 0,
                "window_hits": 0,
                "eligible": 0,
                "yielded": 0,
            },
        )
        counts["scenarios"] += 1
        raw_target = report.get("T")
        if not isinstance(raw_target, dict):
            raise ValueError(f"Missing T block in report: {scenario_id}")
        else:
            pass
        target_class = classify_target_timing(raw_target)
        if target_class not in (
            TARGET_HIT_OUTCOME,
            TARGET_CONTINUATION_OUTCOME,
            TARGET_MISS_OUTCOME,
        ):
            raise ValueError(f"Missing or invalid T outcome in {scenario_id}")
        else:
            pass
        counts["fresh"] += int(target_class == TARGET_HIT_OUTCOME)
        has_responded = target_class != TARGET_MISS_OUTCOME
        counts["responded"] += int(has_responded)
        if has_responded:
            answer_cases = collect_target_answer_cases({scenario_id: report})
            if len(answer_cases) != 1:
                raise ValueError(
                    f"Response marked present but no scored speech in {scenario_id}"
                )
            else:
                pass
            verdict_record = indexed_verdicts.get(
                (scenario_id, answer_cases[0].turn_index, "answer"), {}
            )
            correct_flag = verdict_record.get("correct")
            if type(correct_flag) is not bool:
                counts["answer_pending"] += 1
            else:
                counts["correct"] += int(correct_flag)
        else:
            pass

        raw_rows = report.get("rows")
        if not isinstance(raw_rows, list):
            raise ValueError(f"Missing rows list in report: {scenario_id}")
        else:
            pass
        silence_windows = [
            row
            for row in raw_rows
            if isinstance(row, dict)
            and row.get("event_type") in NON_TARGET_SILENCE_EVENTS
        ]
        counts["silence_n"] += len(silence_windows)
        voiced_episodes = collect_non_target_episodes(
            {scenario_id: report}, {scenario_id: scenario}
        )
        counts["silence_ok"] += len(silence_windows) - len(voiced_episodes)
        for episode in voiced_episodes:
            verdict_record = indexed_verdicts.get(
                (scenario_id, episode.turn_index, "silence"), {}
            )
            silence_label = verdict_record.get("label")
            if silence_label == "BACKCHANNEL":
                counts["silence_ok"] += 1
            elif (
                silence_label == "SUBSTANTIVE_INTRUSION"
                or verdict_record.get("error_kind") == "asr_empty"
            ):
                pass
            else:
                counts["silence_pending"] += 1

        n4_summary = count_n4_windows(report)
        counts["windows"] += n4_summary.windows
        counts["window_hits"] += n4_summary.window_hits
        counts["eligible"] += n4_summary.eligible
        counts["yielded"] += n4_summary.yielded

    results: dict[str, dict[str, JsonValue]] = {
        condition: build_condition_metrics(counts)
        for condition, counts in condition_counts.items()
    }
    if condition_counts:
        pooled_counts: dict[str, int] = {
            metric_key: sum(counts[metric_key] for counts in condition_counts.values())
            for metric_key in next(iter(condition_counts.values()))
        }
        results["overall_pooled"] = build_condition_metrics(pooled_counts)
    else:
        pass
    return results


def is_supported_silero_vad_version(version_text: str) -> bool:
    match = re.fullmatch(r"(\d+)\.(\d+)(?:\.(\d+))?", version_text)
    if match is None:
        return False
    else:
        major_version = int(match.group(1))
        minor_version = int(match.group(2))
        return (major_version, minor_version) >= (5, 1)


def validate_scenario_reports(
    reports_by_id: dict[str, dict[str, JsonValue]],
    scenarios_by_id: dict[str, DuplexMpeScenarioRecord],
    *,
    strict_reference_vad_version: bool = False,
) -> None:
    """Reject incomplete conversations, out-of-range windows, or altered VAD parameters."""
    if not reports_by_id:
        raise ValueError("No completed scenario reports to validate")
    else:
        pass
    observed_vad_versions: set[str] = set()
    for scenario_id, report in reports_by_id.items():
        if scenario_id not in scenarios_by_id:
            raise ValueError(f"Unknown scenario_id in report: {scenario_id}")
        elif report.get("protocol_id") != PROTOCOL_ID:
            raise ValueError(
                f"Incompatible evaluation protocol in {scenario_id}: "
                f"{report.get('protocol_id')!r}"
            )
        else:
            pass
        expected_turns = [
            (index, turn.event_type)
            for index, turn in enumerate(scenarios_by_id[scenario_id].turns)
        ]
        raw_rows = report.get("rows")
        if not isinstance(raw_rows, list):
            raise ValueError(f"Missing rows in {scenario_id}")
        else:
            pass
        actual_turns = [
            (row.get("turn_index"), row.get("event_type"))
            for row in raw_rows
            if isinstance(row, dict)
        ]
        if actual_turns != expected_turns:
            raise ValueError(
                f"Missing, reordered, or mislabeled turns in {scenario_id}"
            )
        else:
            pass
        raw_duration = report.get("total_duration_s", 0.0)
        if (
            not isinstance(raw_duration, (int, float))
            or not math.isfinite(float(raw_duration))
            or float(raw_duration) <= 0.0
        ):
            raise ValueError(f"Invalid total_duration_s in {scenario_id}")
        else:
            duration_seconds = float(raw_duration)

        for row in raw_rows:
            assert isinstance(row, dict)
            numeric_values: list[float] = []
            for field_name in ("turn_start", "turn_end", "win_start", "win_end"):
                raw_time = row.get(field_name)
                if not isinstance(raw_time, (int, float)) or not math.isfinite(
                    float(raw_time)
                ):
                    raise ValueError(f"Invalid {field_name} in {scenario_id}")
                else:
                    numeric_values.append(float(raw_time))
            (
                turn_start_seconds,
                turn_end_seconds,
                window_start_seconds,
                window_end_seconds,
            ) = numeric_values
            upper_limit = duration_seconds + TIMELINE_ROUNDING_TOLERANCE_SECONDS
            if not 0.0 <= turn_start_seconds <= turn_end_seconds <= upper_limit:
                raise ValueError(f"Human turn outside audio timeline in {scenario_id}")
            elif not 0.0 <= window_start_seconds <= window_end_seconds <= upper_limit:
                raise ValueError(
                    f"Scoring window outside audio timeline in {scenario_id}"
                )
            else:
                pass

        previous_end_seconds = 0.0
        upper_limit = duration_seconds + TIMELINE_ROUNDING_TOLERANCE_SECONDS
        for segment_start, segment_end in extract_speech_segments(report):
            if (
                not 0.0 <= segment_start < segment_end <= upper_limit
                or segment_start < previous_end_seconds - BOUNDARY_EPSILON_SECONDS
            ):
                raise ValueError(f"Invalid assistant_speech_segments in {scenario_id}")
            else:
                previous_end_seconds = segment_end

        aed_config = report.get("aed_config")
        if not isinstance(aed_config, dict):
            raise ValueError(f"Missing aed_config in {scenario_id}")
        else:
            pass
        required_settings: dict[str, JsonValue] = {
            "detector": "silero_streaming_hysteresis_hangover",
            "silero_sr": 16000,
            "onset_prob": 0.5,
            "offset_prob": 0.35,
            "hangover_ms": 800.0,
            "t_endpoint_hangover_ms": 2000.0,
            "min_speech_ms": 120.0,
            "silero_window": 512,
        }
        raw_vad_version = aed_config.get("silero_vad_version")
        if not isinstance(raw_vad_version, str) or not is_supported_silero_vad_version(
            raw_vad_version
        ):
            raise ValueError(
                f"Invalid or unsupported silero_vad_version in {scenario_id}: {raw_vad_version!r}"
            )
        else:
            observed_vad_versions.add(raw_vad_version)
        if strict_reference_vad_version:
            required_settings["silero_vad_version"] = REFERENCE_SILERO_VAD_VERSION
        else:
            pass
        if any(
            aed_config.get(setting_key) != setting_value
            for setting_key, setting_value in required_settings.items()
        ):
            raise ValueError(f"Incompatible Silero AED settings in {scenario_id}")
        else:
            pass
    if len(observed_vad_versions) > 1:
        raise ValueError(
            f"Mixed silero_vad_version values across scenario reports: {sorted(observed_vad_versions)}"
        )
    else:
        pass


def format_percent_cell(metric_entry: JsonValue) -> str:
    if not isinstance(metric_entry, dict):
        return "n/a"
    else:
        raw_value = metric_entry.get("value")
        numerator = metric_entry.get("numerator")
        denominator = metric_entry.get("denominator")
        if isinstance(raw_value, (int, float)):
            return f"{100.0 * float(raw_value):.1f}% ({numerator}/{denominator})"
        elif denominator == 0:
            return f"n/a ({numerator}/{denominator})"
        else:
            pending = metric_entry.get("pending", 0)
            return f"pending ({pending} unresolved; {numerator}/{denominator})"


def format_markdown_score_table(
    scored_results: dict[str, dict[str, JsonValue]],
) -> str:
    """Render a concise Markdown summary table across explicit, implicit, and pooled splits."""
    header = (
        "| Split | Scenarios | TR (Fresh Onset) | AA (Cond.) | AA (Uncond.) | "
        "SP (Silence) | AY (Cond.) | AY (Uncond.) | RP | WR |\n"
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
    )
    lines = [header]
    for split_name in ("explicit", "implicit", "overall_pooled"):
        if split_name not in scored_results:
            continue
        else:
            pass
        row = scored_results[split_name]
        lines.append(
            f"| {split_name} | {row['scenarios']} | "
            f"{format_percent_cell(row['fresh_onset_response_rate'])} | "
            f"{format_percent_cell(row['conditional_answer_accuracy'])} | "
            f"{format_percent_cell(row['unconditional_answer_accuracy'])} | "
            f"{format_percent_cell(row['silence_preservation'])} | "
            f"{format_percent_cell(row['answering_window_yield'])} | "
            f"{format_percent_cell(row['unconditional_window_yield'])} | "
            f"{format_percent_cell(row['response_presence'])} | "
            f"{format_percent_cell(row['window_response'])} |"
        )
    return "\n".join(lines) + "\n"
