# SPDX-License-Identifier: Apache-2.0
"""Score duplex v1.5 overlap-event timing from speech segments and shared validation."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import statistics
from collections import Counter
from math import gcd
from pathlib import Path
from typing import Literal

import numpy as np
import soundfile
from pydantic import JsonValue
from scipy.signal import resample_poly

TIMING_VERSION = "fdb-v15-event-v1"
CATEGORIES = ("interruption", "backchannel", "talking_to_other", "background_speech")
TIMELINES = ("media", "simulated_playout")
EVALUATIONS = ("overlap", "clean_reference")

# Note (wenyao): Float rounding can put segment ends just past the audio duration.
DURATION_TOLERANCE_S = 1e-3
TIMING_CONFIG = {
    "version": TIMING_VERSION,
    "onset_span": "start <= event_start < end",
    "stop_latency": "onset-spanning span_end - event_start, also after event_end",
    "response_onset": "segment start >= event_end",
    "stop_censoring": "onset span ending within eof_tolerance_s of output audio EOF",
    "response_censoring": "missing response when the observation is incomplete",
    "eof_tolerance_s": 0.05,
    "clean_reference": "same event anchor, no event speech required in clean input",
    "timeline_boundary": "media or simulated client playout, never acoustic audibility",
}
SILERO_VAD_CONFIG = {
    "sampling_rate": 16000,
    "threshold": 0.5,
    "min_speech_duration_ms": 250,
    "min_silence_duration_ms": 100,
    "speech_pad_ms": 30,
    "onnx": True,
}

Category = Literal[
    "interruption", "backchannel", "talking_to_other", "background_speech"
]


def canonical_hash(value: JsonValue) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


TIMING_CONFIG_HASH = canonical_hash(TIMING_CONFIG)


def check_category(category: str) -> None:
    if category not in CATEGORIES:
        raise ValueError(f"unknown category {category!r}")


def check_interval(start_s: float, end_s: float, duration_s: float, name: str) -> None:
    if not all(math.isfinite(value) for value in (start_s, end_s, duration_s)):
        raise ValueError(f"{name} has a non-finite time")
    elif not 0 <= start_s < end_s <= duration_s + DURATION_TOLERANCE_S:
        raise ValueError(
            f"{name} [{start_s}, {end_s}] is reversed or outside [0, {duration_s}]"
        )


def validate_segments(
    segments: list[list[float]], duration_s: float, name: str
) -> list[tuple[float, float]]:
    """Return ordered, disjoint, in-range speech segments or raise ValueError."""
    checked = []
    previous_end = 0.0
    for index, segment in enumerate(segments):
        if len(segment) != 2:
            raise ValueError(f"{name} segment {index} must be [start, end]")
        start_s, end_s = float(segment[0]), float(segment[1])
        check_interval(start_s, end_s, duration_s, f"{name} segment {index}")
        if start_s < previous_end:
            raise ValueError(f"{name} segment {index} overlaps or is out of order")
        checked.append((start_s, end_s))
        previous_end = end_s
    return checked


def silero_speech_segments(wav_path: str | Path) -> dict[str, JsonValue]:
    """Detect speech in a PCM WAV with the frozen Silero VAD configuration."""
    # Note (wenyao): Silero pulls torch; segment-input scoring and tests must not need it.
    import torch
    from silero_vad import get_speech_timestamps, load_silero_vad

    info = soundfile.info(str(wav_path))
    if info.format != "WAV" or not info.subtype.startswith("PCM"):
        raise ValueError(
            f"{wav_path} is {info.format}/{info.subtype}, expected PCM WAV"
        )
    audio, sample_rate = soundfile.read(str(wav_path), dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)
    duration_s = len(audio) / sample_rate
    target_rate = SILERO_VAD_CONFIG["sampling_rate"]
    if sample_rate != target_rate:
        divisor = gcd(sample_rate, target_rate)
        audio = resample_poly(audio, target_rate // divisor, sample_rate // divisor)
    timestamps = get_speech_timestamps(
        torch.from_numpy(np.ascontiguousarray(audio, dtype=np.float32)),
        load_silero_vad(onnx=SILERO_VAD_CONFIG["onnx"]),
        sampling_rate=target_rate,
        threshold=SILERO_VAD_CONFIG["threshold"],
        min_speech_duration_ms=SILERO_VAD_CONFIG["min_speech_duration_ms"],
        min_silence_duration_ms=SILERO_VAD_CONFIG["min_silence_duration_ms"],
        speech_pad_ms=SILERO_VAD_CONFIG["speech_pad_ms"],
    )
    segments = [
        [item["start"] / target_rate, min(item["end"] / target_rate, duration_s)]
        for item in timestamps
    ]
    return {
        "segments": [
            list(pair) for pair in validate_segments(segments, duration_s, "vad")
        ],
        "duration_s": duration_s,
        "sample_rate": sample_rate,
        "vad": {
            "package": "silero-vad",
            "version": importlib.metadata.version("silero-vad"),
            "config": SILERO_VAD_CONFIG,
            "config_hash": canonical_hash(SILERO_VAD_CONFIG),
        },
    }


def score_event_timing(
    *,
    sample_id: str,
    category: Category,
    input_segments: list[list[float]],
    output_segments: list[list[float]],
    event_start_s: float,
    event_end_s: float,
    input_duration_s: float,
    observed_end_s: float,
    protocol_valid: bool,
    timeline: Literal["media", "simulated_playout"],
    segment_source: dict[str, JsonValue],
    observation_complete: bool = False,
    evaluation: Literal["overlap", "clean_reference"] = "overlap",
    output_duration_s: float | None = None,
) -> dict[str, JsonValue]:
    """Score model stop and response timing around one metadata overlap event.

    output_duration_s is the output audio EOF (default observed_end_s); speech reaching it
    is right-censored. observation_complete only turns a missing response into no_response.
    """
    check_category(category)
    if timeline not in TIMELINES:
        raise ValueError(f"unknown timeline {timeline!r}")
    elif evaluation not in EVALUATIONS:
        raise ValueError(f"unknown evaluation {evaluation!r}")
    check_interval(event_start_s, event_end_s, input_duration_s, "event")
    if not math.isfinite(observed_end_s) or observed_end_s <= 0:
        raise ValueError(
            f"observed_end_s must be positive and finite: {observed_end_s}"
        )
    if output_duration_s is None:
        output_duration_s = observed_end_s
    elif not math.isfinite(output_duration_s) or not (
        0 < output_duration_s <= observed_end_s + DURATION_TOLERANCE_S
    ):
        raise ValueError(
            f"output_duration_s {output_duration_s} must be in (0, {observed_end_s}]"
        )
    inputs = validate_segments(input_segments, input_duration_s, "input")
    outputs = validate_segments(output_segments, output_duration_s, "output")
    stop = {
        "status": None,
        "latency_s": None,
        "span_end_s": None,
        "stopped_during_event": None,
    }
    response = {
        "status": None,
        "latency_s": None,
        "onset_s": None,
        "observed_after_event_s": observed_end_s - event_end_s,
    }
    record = {
        "version": TIMING_VERSION,
        "config_hash": TIMING_CONFIG_HASH,
        "sample_id": sample_id,
        "category": category,
        "timeline": timeline,
        "evaluation": evaluation,
        "segment_source": segment_source,
        "event": [event_start_s, event_end_s],
        "observed_end_s": observed_end_s,
        "observation_complete": observation_complete,
        "output_duration_s": output_duration_s,
        "protocol_valid": protocol_valid,
        "has_output_speech": bool(outputs),
        "speaking_at_onset": None,
        "stop": stop,
        "response": response,
    }
    if not protocol_valid:
        record["status"] = "protocol_failure"
    elif evaluation == "overlap" and not any(
        start < event_end_s and end > event_start_s for start, end in inputs
    ):
        record["status"] = "input_speech_absent"
    elif observed_end_s <= event_start_s:
        record["status"] = "not_observed"
    else:
        record["status"] = "eligible"
        span_end = next(
            (end for start, end in outputs if start <= event_start_s < end), None
        )
        record["speaking_at_onset"] = span_end is not None
        stop["span_end_s"] = span_end
        if span_end is None:
            stop["status"] = "not_speaking_at_onset"
        # Note (wenyao): Native completion drains finite output; an EOF cut is not a voice stop.
        elif span_end >= output_duration_s - TIMING_CONFIG["eof_tolerance_s"]:
            stop["status"] = "right_censored"
        else:
            stop.update(
                status="stopped",
                latency_s=span_end - event_start_s,
                stopped_during_event=span_end <= event_end_s,
            )
        onset = next((start for start, _ in outputs if start >= event_end_s), None)
        if onset is not None:
            response.update(
                status="responded", latency_s=onset - event_end_s, onset_s=onset
            )
        elif any(start < event_end_s < end for start, end in outputs):
            response["status"] = "speaking_through_event_end"
        elif not observation_complete:
            response["status"] = "right_censored"
        else:
            response["status"] = "no_response"
    return record


def describe(values: list[float]) -> dict[str, JsonValue]:
    if not values:
        return {"n": 0, "mean": None, "median": None, "min": None, "max": None}
    else:
        return {
            "n": len(values),
            "mean": statistics.fmean(values),
            "median": statistics.median(values),
            "min": min(values),
            "max": max(values),
        }


def index_records(
    records: list[dict[str, JsonValue]],
    selected: dict[str, str],
    name: str,
    evaluation: str | None = None,
) -> dict[str, dict[str, JsonValue]]:
    indexed = {}
    for record in records:
        sample_id = record["sample_id"]
        if sample_id in indexed:
            raise ValueError(f"duplicate {name} record for {sample_id}")
        elif sample_id not in selected:
            raise ValueError(f"{name} record {sample_id} is not selected")
        elif record["category"] != selected[sample_id]:
            raise ValueError(f"{name} record {sample_id} has a mismatched category")
        elif evaluation is not None and record["evaluation"] != evaluation:
            raise ValueError(f"{name} record {sample_id} is not a {evaluation} record")
        indexed[sample_id] = record
    return indexed


def summarize_timing(
    overlap_records: list[dict[str, JsonValue]],
    clean_records: list[dict[str, JsonValue]],
    selected: dict[str, str],
) -> dict[str, JsonValue]:
    """Aggregate timing per category; selected samples without records count as missing."""
    for category in selected.values():
        check_category(category)
    overlap = index_records(overlap_records, selected, "overlap", "overlap")
    clean = index_records(clean_records, selected, "clean", "clean_reference")
    categories = {}
    pairs = []
    for category in CATEGORIES:
        sample_ids = sorted(key for key, value in selected.items() if value == category)
        present = [overlap[key] for key in sample_ids if key in overlap]
        eligible = [record for record in present if record["status"] == "eligible"]
        paired = [
            (overlap[key], clean[key])
            for key in sample_ids
            if key in overlap
            and key in clean
            and overlap[key]["status"] == "eligible"
            and clean[key]["status"] == "eligible"
        ]
        stop_deltas, response_deltas = [], []
        onset_transitions = Counter()
        for noisy, reference in paired:
            stop_delta = response_delta = None
            if noisy["stop"]["status"] == "stopped" == reference["stop"]["status"]:
                stop_delta = noisy["stop"]["latency_s"] - reference["stop"]["latency_s"]
                stop_deltas.append(stop_delta)
            if (
                noisy["response"]["status"]
                == "responded"
                == reference["response"]["status"]
            ):
                response_delta = (
                    noisy["response"]["latency_s"] - reference["response"]["latency_s"]
                )
                response_deltas.append(response_delta)
            onset_transitions[
                f"clean_{reference['speaking_at_onset']}_overlap_{noisy['speaking_at_onset']}"
            ] += 1
            pairs.append(
                {
                    "sample_id": noisy["sample_id"],
                    "category": category,
                    "stop_latency_delta_s": stop_delta,
                    "response_latency_delta_s": response_delta,
                }
            )
        categories[category] = {
            "selected": len(sample_ids),
            "missing": len(sample_ids) - len(present),
            "status_counts": dict(Counter(record["status"] for record in present)),
            "eligible": len(eligible),
            "speaking_at_onset": sum(
                record["speaking_at_onset"] for record in eligible
            ),
            "stop_status_counts": dict(Counter(r["stop"]["status"] for r in eligible)),
            "response_status_counts": dict(
                Counter(r["response"]["status"] for r in eligible)
            ),
            "stop_latency_s": describe(
                [
                    r["stop"]["latency_s"]
                    for r in eligible
                    if r["stop"]["latency_s"] is not None
                ]
            ),
            "response_latency_s": describe(
                [
                    r["response"]["latency_s"]
                    for r in eligible
                    if r["response"]["latency_s"] is not None
                ]
            ),
            "clean_missing": sum(key not in clean for key in sample_ids),
            "paired_eligible": len(paired),
            "paired_onset_transitions": dict(onset_transitions),
            "paired_stop_latency_delta_s": describe(stop_deltas),
            "paired_response_latency_delta_s": describe(response_deltas),
        }
    return {
        "version": TIMING_VERSION,
        "config": TIMING_CONFIG,
        "config_hash": TIMING_CONFIG_HASH,
        "timelines": sorted({record["timeline"] for record in overlap.values()}),
        "categories": categories,
        "pairs": pairs,
    }
