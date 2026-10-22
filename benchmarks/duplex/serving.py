# SPDX-License-Identifier: Apache-2.0
"""Client-observed serving timing of recorded duplex sessions."""

from __future__ import annotations

import base64
import itertools
import json
import math
from pathlib import Path

import numpy as np
from pydantic import JsonValue

from benchmarks.duplex.client import PACKET_MS, SAMPLE_RATE, SEND_RECEIPTS_FILE
from benchmarks.duplex.oracle_models import (
    ADMISSION_REJECTED_CODE,
    MediaTime,
    TraceRecord,
    WireEvent,
)
from benchmarks.duplex.profiles import PROFILES, ProfileName
from benchmarks.duplex.run_artifacts import load_run
from benchmarks.duplex.v15_audio import AUDIO_DELTA, LEGACY_AUDIO_DELTA

INPUT_BYTES_PER_MS = SAMPLE_RATE * 2 / 1000
# note (jeffro): a not_exercised session kept the protocol; its model output just
# never overlapped the input, so its timing is complete and it is scored.
SCORED_STATUSES = ("pass", "not_exercised")
FAILED_STATUSES = ("fail", "error")
MAX_UNIT_MISS_RATE = 0.01
# note (jeffro): a turn-based server sends no unit receipt tied to generation, so it
# keeps up when its replies stream at least as fast as they play.
MAX_REPLY_REAL_TIME_FACTOR = 1.0


def input_units(variant_dir: Path, unit_ms: int) -> int:
    """Units the session's input fills, counting the padded last one."""
    input_ms = (variant_dir / "input.pcm").stat().st_size / INPUT_BYTES_PER_MS
    return math.ceil(input_ms / unit_ms)


def quantiles(values: list[float]) -> dict[str, JsonValue]:
    return {
        "count": len(values),
        **{
            f"p{percent}": float(np.percentile(values, percent)) if values else None
            for percent in (50, 90, 99)
        },
        "max": max(values, default=None),
    }


def score_session(
    variant_dir: Path, profile: ProfileName, playback_reserve_s: float
) -> dict[str, JsonValue]:
    """Compute 'Unit lag', 'deadline misses' and 'per-reply audio timing' as the client saw them."""
    unit_ms = PROFILES[profile].native_unit_ms
    output_sample_rate = PROFILES[profile].output_sample_rate
    is_native = PROFILES[profile].protocol == "native"
    packet_completed_s = {
        receipt["seq"]: receipt["completed_s"]
        for receipt in json.loads((variant_dir / SEND_RECEIPTS_FILE).read_text())[
            "appends"
        ]
    }
    records = [
        TraceRecord.model_validate(json.loads(line))
        for line in (variant_dir / "continuous.jsonl").read_text().splitlines()
    ]
    input_end_s = next(
        (
            record.time_s
            for record in records
            if record.direction == "send"
            and record.event["type"] == "sglang.input_audio.end"
        ),
        None,
    )

    def unit_media_time(event: WireEvent) -> MediaTime:
        if event.sglang.media_time is None:
            raise ValueError(f"{variant_dir}: {event.type} has no media time")
        else:
            return event.sglang.media_time

    # note (jeffro): lag starts when the server has all of a unit's input: when the
    # client finished sending the packet holding the unit's last audio or, for the
    # short last unit that the server fills with silence, when it sent the input end.
    def ready_s(media_time: MediaTime) -> float:
        end_ms = media_time.t_start_ms + media_time.duration_ms
        packet_ready_s = packet_completed_s[math.ceil(end_ms / PACKET_MS) - 1]
        if media_time.duration_ms >= unit_ms:
            return packet_ready_s
        elif input_end_s is None:
            raise ValueError(f"{variant_dir}: a padded unit finished before input end")
        else:
            return max(packet_ready_s, input_end_s)

    unit_lags_s: list[float] = []
    reply_created_s: dict[str, float] = {}
    reply_opening_unit: dict[str, MediaTime] = {}
    reply_first_audio_unit: dict[str, MediaTime] = {}
    reply_chunks: dict[str, list[tuple[float, float]]] = {}
    for record in (record for record in records if record.direction == "receive"):
        event = WireEvent.model_validate(record.event)
        if event.type == "sglang.unit.done" and is_native:
            media_time = unit_media_time(event)
            if media_time.duration_ms > 0:
                unit_lags_s.append(record.time_s - ready_s(media_time))
            else:
                pass
        elif event.type == "response.created":
            assert event.response is not None
            reply_created_s[event.response.id] = record.time_s
            if is_native:
                reply_opening_unit[event.response.id] = unit_media_time(event)
            else:
                pass
        elif event.type in (AUDIO_DELTA, LEGACY_AUDIO_DELTA):
            assert event.response_id is not None and event.delta is not None
            if is_native:
                reply_first_audio_unit.setdefault(
                    event.response_id, unit_media_time(event)
                )
            else:
                pass
            reply_chunks.setdefault(event.response_id, []).append(
                (
                    record.time_s,
                    len(base64.b64decode(event.delta)) / 2 / output_sample_rate,
                )
            )
        else:
            pass

    replies: list[dict[str, JsonValue]] = []
    for response_id, chunks in reply_chunks.items():
        # note (jeffro): the simulated player starts a reply playback_reserve_s after
        # its first audio arrives.
        playback_s = chunks[0][0] + playback_reserve_s
        underrun_s = 0.0
        unbuffered_start_s = chunks[0][0]
        startup_buffer_needed_s = 0.0
        for arrival_s, audio_s in chunks:
            underrun_s += max(0.0, arrival_s - playback_s)
            playback_s = max(playback_s, arrival_s) + audio_s
            startup_buffer_needed_s = max(
                startup_buffer_needed_s, arrival_s - unbuffered_start_s
            )
            unbuffered_start_s += audio_s
        # note (jeffro): the last chunk's own playback is not part of the generation
        # time, so the factor covers the audio that arrived before it.
        streamed_audio_s = sum(audio_s for _, audio_s in chunks[:-1])
        if is_native:
            first_audio_unit = reply_first_audio_unit[response_id]
            # note (jeffro): a reply's first text can come one unit before its first audio,
            # Latency starts at the unit carrying the first audio, so that unit is not counted as serving time.
            reply_ready_s = ready_s(first_audio_unit)
            units_before_first_audio = round(
                (
                    first_audio_unit.t_start_ms
                    - reply_opening_unit[response_id].t_start_ms
                )
                / unit_ms
            )
        else:
            # note (jeffro): response.created arrives with the server's end of turn,
            # or at the input end when the server flushes an open turn.
            reply_ready_s = reply_created_s[response_id]
            units_before_first_audio = None
        replies.append(
            {
                "response_id": response_id,
                "first_audio_latency_s": chunks[0][0] - reply_ready_s,
                "units_before_first_audio": units_before_first_audio,
                "audio_s": sum(audio_s for _, audio_s in chunks),
                "underrun_s": underrun_s,
                "startup_buffer_needed_s": startup_buffer_needed_s,
                "real_time_factor": (
                    (chunks[-1][0] - chunks[0][0]) / streamed_audio_s
                    if len(chunks) > 1
                    else None
                ),
            }
        )

    units_expected = input_units(variant_dir, unit_ms) if is_native else None
    return {
        "unit_ms": unit_ms,
        "units_expected": units_expected,
        "units_done": len(unit_lags_s) if is_native else None,
        "units_missed": (
            units_expected
            - len(unit_lags_s)
            + sum(lag_s > unit_ms / 1000 for lag_s in unit_lags_s)
            if is_native
            else None
        ),
        "unit_lag_s": unit_lags_s,
        "replies": replies,
    }


def summarize_level(run_dir: Path, playback_reserve_s: float) -> dict[str, JsonValue]:
    """Pool every attempted session of one recording, made at one concurrency."""
    manifest, run, _ = load_run(run_dir)
    if run["status"] != "complete":
        raise ValueError(f"{run_dir} is not a complete recording")
    else:
        pass
    profile = manifest["profile"]
    unit_ms = PROFILES[profile].native_unit_ms
    is_native = PROFILES[profile].protocol == "native"
    attempted = [
        variant_state
        for sample in run["samples"]
        for variant_state in sample["variants"].values()
        if variant_state["status"] in SCORED_STATUSES + FAILED_STATUSES
    ]
    sessions = [
        score_session(run_dir / variant_state["directory"], profile, playback_reserve_s)
        for variant_state in attempted
        if variant_state["status"] in SCORED_STATUSES
    ]
    # note (jeffro): every unit of a session that broke off is missed, so a failure
    # cannot lower the miss rate.
    failed_units = sum(
        input_units(run_dir / variant_state["directory"], unit_ms)
        for variant_state in attempted
        if variant_state["status"] in FAILED_STATUSES
    )
    trace_spans_s: list[tuple[float, float]] = []
    admission_rejections = 0
    for variant_state in attempted:
        variant_dir = run_dir / variant_state["directory"]
        trace_lines = (variant_dir / "continuous.jsonl").read_text().splitlines()
        if trace_lines:
            trace_spans_s.append(
                (
                    json.loads(trace_lines[0])["time_s"],
                    json.loads(trace_lines[-1])["time_s"],
                )
            )
        else:
            pass
        rejection_records = [
            json.loads(line) for line in trace_lines if ADMISSION_REJECTED_CODE in line
        ]
        admission_rejections += any(
            record["direction"] == "receive"
            and (WireEvent.model_validate(record["event"]).error or {}).get("code")
            == ADMISSION_REJECTED_CODE
            for record in rejection_records
        )
    replies = [reply for session in sessions for reply in session["replies"]]
    load = run["load"]
    sessions_failed = sum(
        variant_state["status"] in FAILED_STATUSES for variant_state in attempted
    )
    # note (jeffro): a native reply comes out one unit per input unit, so its
    # real-time factor tracks the input pace.
    real_time_factor = (
        None
        if is_native
        else quantiles(
            [
                reply["real_time_factor"]
                for reply in replies
                if reply["real_time_factor"] is not None
            ]
        )
    )
    if is_native:
        units_expected = failed_units + sum(
            session["units_expected"] for session in sessions
        )
        units_missed = failed_units + sum(
            session["units_missed"] for session in sessions
        )
        unit_miss_rate = units_missed / units_expected
        pacing_target = (
            f"unit miss rate at most {MAX_UNIT_MISS_RATE:.0%}",
            unit_miss_rate > MAX_UNIT_MISS_RATE,
        )
    else:
        units_expected = units_missed = unit_miss_rate = None
        pacing_target = (
            f"reply real-time factor p90 below {MAX_REPLY_REAL_TIME_FACTOR:g}",
            real_time_factor["p90"] is None
            or real_time_factor["p90"] >= MAX_REPLY_REAL_TIME_FACTOR,
        )
    return {
        "run": str(run_dir),
        "profile": profile,
        "concurrency": load["concurrency"],
        # note (jeffro): sessions start as slots free up, so the first and last ones
        # of a recording run with fewer neighbours than the concurrency.
        "mean_open_sessions": sum(end_s - start_s for start_s, end_s in trace_spans_s)
        / (
            max(end_s for _, end_s in trace_spans_s)
            - min(start_s for start_s, _ in trace_spans_s)
        ),
        "sessions": len(attempted),
        "sessions_failed": sessions_failed,
        "admission_denials": load["variants_with_admission_denials"],
        "admission_rejections": admission_rejections,
        "input_pacing_violations": load["input_pacing_violations"],
        "units_expected": units_expected,
        "units_missed": units_missed,
        "unit_miss_rate": unit_miss_rate,
        "unit_lag_s": quantiles(
            [lag_s for session in sessions for lag_s in session["unit_lag_s"]]
        ),
        "replies": len(replies),
        "replies_with_audio_after_text": (
            sum(reply["units_before_first_audio"] > 0 for reply in replies)
            if is_native
            else None
        ),
        "first_audio_latency_s": quantiles(
            [reply["first_audio_latency_s"] for reply in replies]
        ),
        "startup_buffer_needed_s": quantiles(
            [reply["startup_buffer_needed_s"] for reply in replies]
        ),
        "playback_reserve_s": playback_reserve_s,
        "replies_with_underrun": sum(reply["underrun_s"] > 0 for reply in replies),
        "underrun_s": sum(reply["underrun_s"] for reply in replies),
        "real_time_factor": real_time_factor,
        "unmet_targets": [
            target
            for target, is_unmet in (
                ("no failed sessions", sessions_failed > 0),
                ("no admission denials", load["variants_with_admission_denials"] > 0),
                ("no admission rejections", admission_rejections > 0),
                ("no input pacing violations", load["input_pacing_violations"] > 0),
                pacing_target,
            )
            if is_unmet
        ],
    }


def serving_report(
    run_dirs: list[Path], playback_reserve_s: float
) -> dict[str, JsonValue]:
    """Concurrency levels of one system and the highest one that meets the targets."""
    levels = sorted(
        (summarize_level(run_dir, playback_reserve_s) for run_dir in run_dirs),
        key=lambda level: level["concurrency"],
    )
    concurrencies = [level["concurrency"] for level in levels]
    if len({level["profile"] for level in levels}) != 1:
        raise ValueError("every recording must use the same profile")
    elif len(set(concurrencies)) != len(concurrencies):
        raise ValueError(f"each concurrency needs one recording, got {concurrencies}")
    else:
        pass
    meeting = list(
        itertools.takewhile(lambda level: not level["unmet_targets"], levels)
    )
    return {
        "levels": levels,
        "max_concurrency_meeting_targets": (
            meeting[-1]["concurrency"] if meeting else None
        ),
        # note (jeffro): when the highest recorded level still meets the targets,
        # the capacity is at least that level and a higher one must be recorded.
        "capacity_bounded": len(meeting) < len(levels),
    }
