# SPDX-License-Identifier: Apache-2.0
"""Reconstruct bounded-input-jitter reference windows from immutable captures."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import soundfile
from numpy.typing import NDArray
from pydantic import JsonValue
from scipy.signal import resample_poly

from benchmarks.duplex.profiles import PROFILES
from benchmarks.duplex.reference_capture import (
    AUDIO_DELTA,
    TraceFormat,
    parse_pcm16_trace,
)
from benchmarks.duplex.run_artifacts import file_sha256
from benchmarks.duplex.v15_audio import PACING_TOLERANCE_S

POLICY_VERSION = 2
POLICY = "derived-analysis-v2-bounded-input-jitter"
COMPLETION_GRACE_S = 0.080
APPEND = "input_audio_buffer.append"
RATE = 16000
PACKET_S = 0.08
PACKET_SAMPLES = 1280
FILES = {
    "overlap": ("input.wav", "output.wav"),
    "clean": ("clean_input.wav", "clean_output.wav"),
}
SCHEMA = "full-duplex-bench-v1.5-reference-audio"
INCOMPLETE = {"pending", "not_run", "error", "invalid", "window_invalid"}
SEND_RECEIPTS = "input-send-receipts.json"


def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def response_id(event: dict[str, JsonValue]) -> JsonValue:
    return event.get("response_id") or (event.get("response") or {}).get("id")


def deadline(t0: float, index: int, window_s: float) -> float:
    """v2 send-completion deadline of append index (absolute trace clock)."""
    return t0 + min((index + 1) * PACKET_S, window_s) + COMPLETION_GRACE_S


def check_completions(
    completions: list[float | None],
    append_times: list[float],
    deadlines: list[float],
    reasons: list[str],
) -> None:
    """Shared v2 check of per-append completion times (None = not recorded)."""
    previous = None
    for index, (completed, start) in enumerate(zip(completions, append_times)):
        if completed is None:
            reasons.append(f"append {index} send completion not recorded")
            continue
        else:
            pass
        if completed < start or (previous is not None and completed < previous):
            reasons.append(f"append {index} send completion out of order")
        else:
            pass
        previous = completed
        if completed > deadlines[index]:
            reasons.append(f"append {index} send completed after v2 deadline")
        else:
            pass


def check_send_receipts(
    path: Path,
    append_ids: list[JsonValue],
    append_times: list[float],
    deadlines: list[float],
    reasons: list[str],
) -> tuple[str | None, list[float | None] | None]:
    """Verify SGLang send completions; return (sidecar sha256, per-append completions)."""
    if not path.is_file():
        reasons.append(f"{SEND_RECEIPTS} missing for campaign-adapter capture")
        return None, None
    else:
        pass
    try:
        rows = json.loads(path.read_text())["appends"]
        if not isinstance(rows, list):
            raise TypeError("appends is not a list")
        else:
            pass
    except (ValueError, KeyError, TypeError) as exc:
        reasons.append(f"{SEND_RECEIPTS} unreadable: {type(exc).__name__}")
        rows = []
    if [
        row.get("event_id") if isinstance(row, dict) else None for row in rows
    ] != append_ids:
        reasons.append(f"{SEND_RECEIPTS} append event_ids differ from trace")
    else:
        pass
    completions = [None] * len(append_times)
    for index, (row, start) in enumerate(zip(rows, append_times)):
        row = row if isinstance(row, dict) else {}
        completed = row.get("completed_s")
        if row.get("seq") != index or row.get("start_s") != start:
            reasons.append(f"append {index} send receipt seq/start differs from trace")
        else:
            pass
        if type(completed) in (int, float) and math.isfinite(completed):
            completions[index] = completed
        else:
            pass
    check_completions(completions, append_times, deadlines, reasons)
    return file_sha256(path), completions


def lateness(
    times: list[float | None] | None, end: float | None
) -> tuple[int | None, float | None]:
    """(count after T, max positive lateness s) of recorded times; None if any is unknown."""
    if end is None or times is None or any(t is None for t in times):
        return None, None
    else:
        pass
    late = [t - end for t in times if t > end]
    return len(late), max(late, default=0.0)


def analyze_variant(
    variant_dir: Path,
    trace_format: TraceFormat,
    expected_input_sha: str | None,
    receipts_required: bool = False,
    profile_rate: int | None = None,
) -> tuple[dict[str, JsonValue], bytes | None, NDArray[np.int16] | None]:
    """Stream one trace; return (record, input_pcm or None, output int16 at 16 kHz or None).

    profile_rate is the recorded profile's output rate for the legacy format.
    """
    reasons, post_window_events, anomalies = [], [], []
    pcm_path, trace_path = variant_dir / "input.pcm", variant_dir / "continuous.jsonl"
    if not pcm_path.is_file() or not trace_path.is_file():
        return (
            {
                "window": {
                    "valid": False,
                    "reasons": ["input.pcm or continuous.jsonl missing"],
                    "policy_version": POLICY_VERSION,
                    "policy": POLICY,
                }
            },
            None,
            None,
        )
    else:
        pass
    pcm = pcm_path.read_bytes()
    samples = np.frombuffer(pcm, "<i2") if len(pcm) % 2 == 0 else np.zeros(0, "<i2")
    if not len(samples):
        reasons.append("input.pcm empty or odd length")
    else:
        pass
    input_sha = sha_bytes(pcm)
    if input_sha != expected_input_sha:
        reasons.append("input.pcm sha256 differs from run.json input.sha256")
    else:
        pass
    sample_count = len(samples)
    window_s = sample_count / RATE
    packet = PACKET_SAMPLES
    expected_appends = -(-sample_count // packet)
    first_append_s = window_end_s = None
    out_rate = None
    appends, deviations, append_ids, append_times = 0, [], [], []
    last_time = None
    playout = cursor = None
    chunks = {
        "in_window": 0,
        "in_window_samples": 0,
        "after_window": 0,
        "after_window_samples": 0,
    }
    first_audio_s = last_audio_s = None
    live_after_t = False
    interrupts_on_speech = None
    barge_in_cuts = 0
    created_responses, response_terminals = set(), {}
    closed_count, close_times_s = 0, []
    accepted_receipts = {}

    def at(time_s: float) -> float | None:
        return None if first_append_s is None else time_s - first_append_s

    def problem(message: str, time_s: float) -> None:
        """Window-invalidating if at/before T (or before t0), else a post-window diagnostic."""
        if window_end_s is None or time_s <= window_end_s:
            reasons.append(message)
        else:
            post_window_events.append({"elapsed_s": at(time_s), "message": message})

    with trace_path.open(encoding="utf-8") as handle:
        for capture in parse_pcm16_trace(
            handle, samples, packet, trace_format, profile_rate
        ):
            line_number = capture.line_number
            if capture.row is None:
                reasons.append(capture.read_error)
                continue
            else:
                direction, time_s, event = (
                    capture.row.direction,
                    capture.row.time_s,
                    capture.row.event,
                )
            kind = event.get("type")
            if last_time is not None and time_s < last_time:
                reasons.append(f"trace clock not monotonic at line {line_number}")
            else:
                pass
            last_time = time_s if last_time is None else max(last_time, time_s)
            if direction == "send" and kind == "input_audio_buffer.append":
                if first_append_s is None:
                    first_append_s, window_end_s = time_s, time_s + window_s
                else:
                    pass
                index = appends
                appends += 1
                append_ids.append(event.get("event_id"))
                append_times.append(time_s)
                if index >= expected_appends:
                    reasons.append(f"extra input append {index}")
                    continue
                else:
                    pass
                start, error = capture.source_start_s, capture.payload_error
                if error is None and (
                    start is None
                    or not math.isclose(start, index * PACKET_S, abs_tol=1e-9)
                ):
                    error = "append source start is not index * 80 ms"
                else:
                    pass
                if error:
                    reasons.append(f"append {index} (line {line_number}): {error}")
                else:
                    pass
                if start is not None:
                    deviations.append(abs(at(time_s) - start))
                else:
                    pass
                continue
            else:
                pass
            if direction == "send":
                if first_append_s is not None:
                    accepted_receipts.setdefault(f"sent:{kind}", at(time_s))
                else:
                    pass
                if (
                    trace_format is TraceFormat.LEGACY_PCM16
                    and kind == "session.update"
                    and interrupts_on_speech is None
                ):
                    turn_detection = event["session"].get("turn_detection") or {}
                    interrupts_on_speech = (
                        turn_detection.get("interrupt_response") is not False
                    )
                else:
                    pass
                continue
            else:
                pass
            if direction in ("error", "diagnostic"):
                message = f"client {direction}: {event.get('message')}"
                if direction == "error":
                    problem(message, time_s)
                else:
                    anomalies.append({"elapsed_s": at(time_s), "message": message})
                continue
            else:
                pass
            if direction != "receive":
                continue
            else:
                pass
            if window_end_s is not None and time_s > window_end_s:
                live_after_t = True
            else:
                pass
            if capture.declares_output_rate:
                out_rate = capture.output_rate
            elif kind == "session.closed":
                closed_count += 1
                close_times_s.append(at(time_s))
                if window_end_s is None or time_s <= window_end_s:
                    reasons.append("session.closed at or before window end")
                else:
                    pass
            elif kind == "error":
                problem(f"native server error: {event.get('error', event)}", time_s)
            elif kind == "response.created":
                created_responses.add(response_id(event))
            elif kind == "response.done":
                response_terminals[response_id(event)] = (
                    event.get("response") or {}
                ).get("status")
            elif kind in (
                "input_audio_buffer.committed",
                "sglang.input_audio.drained",
                "sglang.input_audio.ended",
            ):
                accepted_receipts.setdefault(kind, at(time_s))
            elif trace_format is TraceFormat.LEGACY_PCM16 and (
                kind == "output_audio_buffer.cleared"
                or (
                    kind == "input_audio_buffer.speech_started" and interrupts_on_speech
                )
            ):
                if playout is None:
                    continue
                else:
                    pass
                cut = round(at(time_s) * out_rate)
                if cut < cursor:
                    playout[cut : min(cursor, len(playout))] = 0
                    cursor = cut
                    barge_in_cuts += 1
                else:
                    pass
            elif kind == AUDIO_DELTA[trace_format]:
                try:
                    if first_append_s is None:
                        raise ValueError("audio before first input append")
                    else:
                        pass
                    rate = capture.output_rate
                    if capture.payload_error is not None:
                        raise ValueError(capture.payload_error)
                    else:
                        pass
                    if type(rate) is not int or rate <= 0:
                        raise ValueError("audio output rate undeclared")
                    else:
                        pass
                    if playout is not None and rate != out_rate:
                        raise ValueError("audio output rate changed")
                    else:
                        pass
                    data = capture.output_pcm
                    if not data or len(data) % 2:
                        raise ValueError("empty or truncated PCM16 audio delta")
                    else:
                        pass
                except ValueError as exc:
                    problem(f"trace line {line_number}: {exc}", time_s)
                    continue
                if time_s > window_end_s:
                    chunks["after_window"] += 1
                    chunks["after_window_samples"] += len(data) // 2
                    continue
                else:
                    pass
                if playout is None:
                    out_rate = rate
                    playout = np.zeros(math.ceil(sample_count * rate / RATE), "<i2")
                    cursor = 0
                else:
                    pass
                pcm16 = np.frombuffer(data, "<i2")
                start = max(cursor, round(at(time_s) * rate))
                stop = min(start + len(pcm16), len(playout))
                if stop > start:
                    playout[start:stop] = pcm16[: stop - start]
                else:
                    pass
                cursor = start + len(pcm16)
                chunks["in_window"] += 1
                chunks["in_window_samples"] += len(pcm16)
                first_audio_s = start / rate if first_audio_s is None else first_audio_s
                last_audio_s = at(time_s)
            else:
                pass

    if first_append_s is None:
        reasons.append("no input append sent")
    elif appends < expected_appends:
        reasons.append(f"incomplete append population: {appends}/{expected_appends}")
    else:
        pass
    for index, event_id in enumerate(append_ids):
        if not isinstance(event_id, str) or not event_id:
            reasons.append(f"append {index} event_id missing or not a non-empty string")
        else:
            pass
    valid_ids = [
        event_id for event_id in append_ids if isinstance(event_id, str) and event_id
    ]
    if len(set(valid_ids)) != len(valid_ids):
        reasons.append("append event_ids are not unique")
    else:
        pass
    deadlines = (
        [deadline(first_append_s, i, window_s) for i in range(appends)]
        if first_append_s is not None
        else []
    )
    receipts_sha, completions = None, None
    receipts_path = variant_dir / SEND_RECEIPTS
    if receipts_required or receipts_path.is_file():
        receipts_sha, completions = check_send_receipts(
            receipts_path, append_ids, append_times, deadlines, reasons
        )
    else:
        pass
    legacy = receipts_sha is None and not receipts_required
    if legacy:
        for index, time_s in enumerate(append_times):
            if time_s > window_end_s:
                reasons.append(f"append {index} sent after window end")
            else:
                pass
    else:
        pass
    starts_after, start_late = lateness(append_times, window_end_s)
    completions_after, completion_late = lateness(
        None if legacy else completions, window_end_s
    )
    max_dev = max(deviations, default=None)
    if max_dev is None or max_dev > PACING_TOLERANCE_S:
        reasons.append(
            f"input pacing deviation {max_dev} exceeds {PACING_TOLERANCE_S}s"
        )
    else:
        pass
    if first_append_s is not None and not live_after_t:
        reasons.append("receiver liveness after window end unproven")
    else:
        pass
    valid = not reasons
    output = None
    record = {
        "trace_format": trace_format.value,
        "source": {
            "directory": str(variant_dir),
            "trace_sha256": file_sha256(trace_path),
            "input_pcm_sha256": input_sha,
            "input_send_receipts_sha256": receipts_sha,
        },
        "input": {"samples": sample_count, "sample_rate": RATE, "duration_s": window_s},
        "window": {
            "t0": "send start of first input_audio_buffer.append",
            "T_s": window_s,
            "valid": valid,
            "reasons": reasons,
            "policy_version": POLICY_VERSION,
            "policy": POLICY,
        },
        "input_check": {
            "appends": appends,
            "expected_appends": expected_appends,
            "legacy_inference": legacy,
            "send_completion_evidence": (
                "legacy: inferred from absence of client "
                f"error before T and append starts <= T (no {SEND_RECEIPTS})"
                if legacy
                else (
                    f"{SEND_RECEIPTS} completed_s in order <= per-append v2 deadline"
                    if receipts_sha is not None
                    else "missing"
                )
            ),
            "completion_deadline": "t0 + min((i+1)*0.080, T) + 0.080 s",
            "completions_recorded": sum(c is not None for c in completions or []),
            "append_starts_after_T": starts_after,
            "max_append_start_lateness_s": start_late,
            "append_completions_after_T": completions_after,
            "max_append_completion_lateness_s": completion_late,
            "after_T_evidence": (
                "legacy: completions unobserved"
                if legacy
                else (
                    "observed"
                    if completions_after is not None
                    else "incomplete: completions missing, counts unknown"
                )
            ),
            "max_abs_pacing_deviation_s": max_dev,
            "pacing_tolerance_s": PACING_TOLERANCE_S,
        },
        "lifecycle": {
            "session_closed_count": closed_count,
            "session_closed_elapsed_s": close_times_s,
            "responses_created": len(created_responses),
            "responses_terminal": len(response_terminals),
            "responses_missing_terminal": sorted(
                str(r) for r in created_responses - response_terminals.keys()
            ),
            "terminal_statuses": sorted({str(s) for s in response_terminals.values()}),
            "receipts_elapsed_s": accepted_receipts,
            "post_window_errors": post_window_events,
            "protocol_anomalies": anomalies,
            "all_input_processed_receipt": None,
            "natural_completion": None,
        },
    }
    if valid:
        native = playout if playout is not None else np.zeros(0, "<i2")
        rate = out_rate if playout is not None else RATE
        if playout is None:
            out = np.zeros(sample_count, "<i2")
            clipped = 0
        elif rate == RATE:
            out = native[:sample_count].copy()
            clipped = 0
        else:
            g = math.gcd(RATE, rate)
            resampled = resample_poly(native.astype(np.float64), RATE // g, rate // g)
            if len(resampled) < sample_count:
                raise AssertionError("resampled timeline shorter than input")
            else:
                pass
            scaled = np.round(resampled[:sample_count])
            clipped = int(np.count_nonzero((scaled < -32768) | (scaled > 32767)))
            out = np.clip(scaled, -32768, 32767).astype("<i2")
        crop_at = window_s * rate
        record["output"] = {
            "timeline": "fifo_zero_buffer_receipt_playout",
            "native_rate": out_rate,
            "resample": (
                None
                if rate == RATE
                else {
                    "method": "scipy.signal.resample_poly",
                    "up": RATE // math.gcd(RATE, rate),
                    "down": rate // math.gcd(RATE, rate),
                }
            ),
            "samples": int(len(out)),
            "pcm_sha256": sha_bytes(out.tobytes()),
            "clipped_samples": clipped,
            "silent": chunks["in_window"] == 0,
            "packets_in_window": chunks["in_window"],
            "packet_audio_in_window_s": (
                chunks["in_window_samples"] / rate if out_rate else 0.0
            ),
            "packets_after_window_excluded": chunks["after_window"],
            "audio_after_window_excluded_s": (
                chunks["after_window_samples"] / out_rate if out_rate else None
            ),
            "first_playout_start_s": first_audio_s,
            "last_contributing_receipt_s": last_audio_s,
        }
        if trace_format is TraceFormat.LEGACY_PCM16:
            record["output"]["barge_in_cuts"] = barge_in_cuts
        else:
            pass
        record["boundary"] = {
            "playout_active_at_T": bool(cursor is not None and cursor > crop_at),
            "queued_audio_cropped_s": (
                max(0.0, cursor - crop_at) / rate if cursor else 0.0
            ),
            "audio_received_after_T": chunks["after_window"] > 0,
            "speech_at_boundary": "decided later by VAD on output.wav",
        }
        output = out
    else:
        pass
    return record, (pcm if input_sha == expected_input_sha else None), output


def load_runs(runs: list[Path], trace_format: TraceFormat) -> tuple[
    dict[str, tuple[Path, dict[str, JsonValue]]],
    list[dict[str, JsonValue]],
    list[dict[str, JsonValue]],
]:
    """Map sample id -> (run dir, entry); a later run may supersede an incomplete entry."""
    chosen, sources, superseded = {}, [], []
    for run in runs:
        run = run.resolve()
        data = json.loads((run / "run.json").read_text())
        manifest = json.loads((run / "manifest.json").read_text())
        declared_format = manifest.get("trace_format")
        if declared_format is not None and TraceFormat(declared_format) != trace_format:
            raise ValueError(
                f"{run} declares a different trace format: {declared_format}"
            )
        else:
            pass
        profile = manifest.get("profile")
        protocol = PROFILES[profile].protocol if profile in PROFILES else None
        if trace_format is TraceFormat.LEGACY_PCM16 and protocol != "legacy":
            raise ValueError(
                f"{run} profile {profile!r} did not record the legacy protocol"
            )
        elif trace_format is TraceFormat.PCM16 and protocol == "legacy":
            raise ValueError(
                f"{run} profile {profile!r} recorded the legacy protocol; export it "
                f"with --trace-format {TraceFormat.LEGACY_PCM16.value}"
            )
        else:
            pass
        sources.append(
            {
                "run": str(run),
                "status": data.get("status"),
                "run_json_sha256": file_sha256(run / "run.json"),
                "manifest_sha256": file_sha256(run / "manifest.json"),
                "validation_scope": manifest.get("validation_scope"),
                "profile": manifest.get("profile"),
                "server": manifest.get("server"),
            }
        )
        for entry in data["samples"]:
            previous = chosen.get(entry["id"])
            if previous is not None:
                if not any(
                    v["status"] in INCOMPLETE for v in previous[1]["variants"].values()
                ):
                    raise ValueError(f"{entry['id']} captured completely in two runs")
                else:
                    pass
                superseded.append({"id": entry["id"], "run": str(previous[0])})
            else:
                pass
            chosen[entry["id"]] = (run, entry)
    return chosen, sources, superseded


def diagnostics(run: Path, state: dict[str, JsonValue]) -> dict[str, JsonValue]:
    """Preserve recorder verdicts verbatim-by-field; only the legacy export gates on
    the protocol verdict, and it reads run.json for that."""
    kept = {k: v for k, v in state.items() if k not in ("files", "input_timing")}
    directory = run / state["directory"]
    if (directory / "report.json").is_file():
        kept["report_sha256"] = file_sha256(directory / "report.json")
    else:
        pass
    return kept


def write_wav(path: Path, pcm16: NDArray[np.int16]) -> str:
    soundfile.write(str(path), pcm16, RATE, subtype="PCM_16")
    return file_sha256(path)
