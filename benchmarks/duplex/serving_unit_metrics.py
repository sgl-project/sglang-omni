# SPDX-License-Identifier: Apache-2.0
"""Compute serving metrics for models that answer in native units, such as MiniCPM-o."""

from __future__ import annotations

import base64
import json
import math
from pathlib import Path

from pydantic import JsonValue

from benchmarks.duplex.client import (
    PACKET_BYTES,
    PACKET_MS,
    SAMPLE_RATE,
    SEND_RECEIPTS_FILE,
)
from benchmarks.duplex.profiles import PROFILES, ProfileName
from benchmarks.duplex.serving_metrics import (
    LATE_SEND_THRESHOLD_S,
    distribution,
    playback_underruns,
)


def unit_session_metrics(
    trace_path: Path,
    *,
    session_id: str,
    input_duration_s: float,
    profile: ProfileName,
    reserve_s: float,
) -> dict[str, JsonValue]:
    """Score one session unit by unit; silence while the model listens is not a fault."""
    records = [json.loads(line) for line in trace_path.read_text().splitlines()]
    receipts = json.loads(
        trace_path.with_name(SEND_RECEIPTS_FILE).read_text(encoding="utf-8")
    )
    appends = receipts["appends"]
    unit_s = PROFILES[profile].native_unit_ms / 1000
    output_sample_rate = PROFILES[profile].output_sample_rate
    expected_units = math.ceil(input_duration_s / unit_s)
    expected_appends = math.ceil(input_duration_s * SAMPLE_RATE * 2 / PACKET_BYTES)

    errors: list[str] = []
    event_types: list[str] = []
    unit_done_s: list[float] = []
    unit_has_audio: list[bool] = []
    unit_has_text: list[bool] = []
    pending_audio = False
    pending_text = False
    responses: dict[str, list[tuple[float, float]]] = {}
    response_unit: dict[str, int] = {}
    output_samples = 0
    for record in records:
        event = record["event"]
        event_type = event.get("type")
        if record["direction"] == "error":
            errors.append(str(event.get("message", "client error")))
        elif record["direction"] == "receive":
            event_types.append(event_type)
            if event_type == "error":
                errors.append(str(event.get("error", "server error")))
            elif event_type == "response.created":
                response_unit[event["response"]["id"]] = len(unit_done_s)
            elif event_type == "response.done":
                if event["response"]["status"] != "completed":
                    errors.append(f"response did not complete: {event['response']}")
                else:
                    pass
            elif event_type == "response.output_audio.delta":
                pcm = base64.b64decode(event["delta"], validate=True)
                output_samples += len(pcm) // 2
                responses.setdefault(event["response_id"], []).append(
                    (record["time_s"], len(pcm) / (2 * output_sample_rate))
                )
                pending_audio = True
            elif event_type in (
                "response.output_audio_transcript.delta",
                "response.output_text.delta",
            ):
                pending_text = True
            elif event_type == "sglang.unit.done":
                unit_done_s.append(record["time_s"])
                unit_has_audio.append(pending_audio)
                unit_has_text.append(pending_text)
                pending_audio = False
                pending_text = False
            else:
                pass
        else:
            pass
    if len(appends) != expected_appends:
        errors.append(f"sent {len(appends)}/{expected_appends} input frames")
    else:
        pass
    if "sglang.input_audio.drained" not in event_types:
        errors.append("input did not drain")
    else:
        pass
    if "session.closed" not in event_types:
        errors.append("session did not close")
    else:
        pass
    if len(unit_done_s) != expected_units:
        errors.append(f"completed {len(unit_done_s)}/{expected_units} units")
    else:
        pass

    # note (Junnan Li): A unit can start once the packet holding its last sample is sent.
    unit_ready_s = [
        appends[
            min(
                math.ceil((index + 1) * unit_s * 1000 / PACKET_MS) - 1,
                len(appends) - 1,
            )
        ]["completed_s"]
        for index in range(len(unit_done_s) if appends else 0)
    ]
    unit_lag = [done_s - ready_s for done_s, ready_s in zip(unit_done_s, unit_ready_s)]
    listen_lag: list[float] = []
    speak_lag: list[float] = []
    for index, lag_s in enumerate(unit_lag):
        if unit_has_audio[index] or unit_has_text[index]:
            speak_lag.append(lag_s)
        else:
            listen_lag.append(lag_s)
    missed_units = sum(lag_s > unit_s for lag_s in unit_lag) + max(
        0, expected_units - len(unit_lag)
    )
    # note (Junnan Li): A reply's latency runs from the unit that opened it to its first audio.
    response_audio_lag = [
        packets[0][0] - unit_ready_s[response_unit[response_id]]
        for response_id, packets in responses.items()
        if response_unit[response_id] < len(unit_ready_s)
    ]
    underrun_count = 0
    underrun_total_s = 0.0
    underrun_worst_s = 0.0
    for packets in responses.values():
        count, total_s, worst_s = playback_underruns(packets, packets[-1][0], reserve_s)
        underrun_count += count
        underrun_total_s += total_s
        underrun_worst_s = max(underrun_worst_s, worst_s)
    output_duration_s = output_samples / output_sample_rate
    units_per_s = (
        len(unit_done_s) / (unit_done_s[-1] - appends[0]["start_s"])
        if unit_done_s and appends
        else None
    )
    lateness = [max(0.0, r["start_s"] - r["scheduled_s"]) for r in appends]
    late_send_count = sum(value > LATE_SEND_THRESHOLD_S for value in lateness)
    return {
        "session_id": session_id,
        "input_duration_s": input_duration_s,
        "trace_file": str(trace_path),
        "receipts_file": str(trace_path.with_name(SEND_RECEIPTS_FILE)),
        "success": not errors,
        "errors": errors,
        "expected_units": expected_units,
        "completed_units": len(unit_done_s),
        "speak_units": len(speak_lag),
        "missed_units": missed_units,
        "unit_miss_rate": missed_units / expected_units,
        "units_per_s": units_per_s,
        "unit_lag_s": distribution(unit_lag),
        "listen_unit_lag_s": distribution(listen_lag),
        "speak_unit_lag_s": distribution(speak_lag),
        "response_audio_lag_s": distribution(response_audio_lag),
        "send_lateness_s": distribution(lateness),
        "late_send_count": late_send_count,
        "late_send_rate": late_send_count / len(lateness) if lateness else None,
        "unit_lag_values_s": unit_lag,
        "listen_unit_lag_values_s": listen_lag,
        "speak_unit_lag_values_s": speak_lag,
        "response_audio_lag_values_s": response_audio_lag,
        "send_lateness_values_s": lateness,
        "responses": len(responses),
        "output_samples": output_samples,
        "output_duration_s": output_duration_s,
        "underrun_count": underrun_count,
        "underrun_total_s": underrun_total_s,
        "underrun_worst_s": underrun_worst_s,
        "underrun_ratio": (
            underrun_total_s / output_duration_s if output_duration_s else None
        ),
    }


def failed_unit_session(
    session_id: str, trace_path: Path, input_duration_s: float, profile: ProfileName
) -> dict[str, JsonValue]:
    """Count every unit of an unreadable session as missed."""
    expected_units = math.ceil(
        input_duration_s / (PROFILES[profile].native_unit_ms / 1000)
    )
    return {
        "session_id": session_id,
        "input_duration_s": input_duration_s,
        "trace_file": str(trace_path),
        "success": False,
        "errors": [],
        "expected_units": expected_units,
        "completed_units": 0,
        "speak_units": 0,
        "missed_units": expected_units,
        "unit_miss_rate": 1.0,
        "units_per_s": None,
        "unit_lag_s": distribution([]),
        "listen_unit_lag_s": distribution([]),
        "speak_unit_lag_s": distribution([]),
        "response_audio_lag_s": distribution([]),
        "send_lateness_s": distribution([]),
        "late_send_count": 0,
        "late_send_rate": None,
        "unit_lag_values_s": [],
        "listen_unit_lag_values_s": [],
        "speak_unit_lag_values_s": [],
        "response_audio_lag_values_s": [],
        "send_lateness_values_s": [],
        "responses": 0,
        "output_samples": 0,
        "output_duration_s": 0.0,
        "underrun_count": 0,
        "underrun_total_s": 0.0,
        "underrun_worst_s": 0.0,
        "underrun_ratio": None,
    }


def aggregate_unit_sessions(
    sessions: list[dict[str, JsonValue]]
) -> dict[str, JsonValue]:
    expected_units = sum(s["expected_units"] for s in sessions)
    missed_units = sum(s["missed_units"] for s in sessions)
    send_count = sum(len(s["send_lateness_values_s"]) for s in sessions)
    late_send_count = sum(s["late_send_count"] for s in sessions)
    output_duration_s = sum(s["output_duration_s"] for s in sessions)
    underrun_total_s = sum(s["underrun_total_s"] for s in sessions)

    def pooled(name: str) -> dict[str, float | int | None]:
        return distribution([value for s in sessions for value in s[name]])

    return {
        "attempted_sessions": len(sessions),
        "successful_sessions": sum(bool(s["success"]) for s in sessions),
        "expected_units": expected_units,
        "completed_units": sum(s["completed_units"] for s in sessions),
        "speak_units": sum(s["speak_units"] for s in sessions),
        "missed_units": missed_units,
        "unit_miss_rate": missed_units / expected_units,
        "session_units_per_s": distribution(
            [s["units_per_s"] for s in sessions if s["units_per_s"] is not None]
        ),
        "total_units_per_s": sum(s["units_per_s"] or 0.0 for s in sessions),
        "unit_lag_s": pooled("unit_lag_values_s"),
        "listen_unit_lag_s": pooled("listen_unit_lag_values_s"),
        "speak_unit_lag_s": pooled("speak_unit_lag_values_s"),
        "response_audio_lag_s": pooled("response_audio_lag_values_s"),
        "send_lateness_s": pooled("send_lateness_values_s"),
        "late_send_threshold_s": LATE_SEND_THRESHOLD_S,
        "late_send_count": late_send_count,
        "late_send_rate": late_send_count / send_count if send_count else None,
        "responses": sum(s["responses"] for s in sessions),
        "output_duration_s": output_duration_s,
        "underrun_sessions": sum(bool(s["underrun_count"]) for s in sessions),
        "underrun_count": sum(s["underrun_count"] for s in sessions),
        "underrun_total_s": underrun_total_s,
        "underrun_worst_s": max((s["underrun_worst_s"] for s in sessions), default=0.0),
        "underrun_ratio": (
            underrun_total_s / output_duration_s if output_duration_s else None
        ),
    }
