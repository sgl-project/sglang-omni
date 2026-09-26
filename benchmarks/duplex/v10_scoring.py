# SPDX-License-Identifier: Apache-2.0
"""Score Full-Duplex-Bench v1.0 turn-taking tasks from word timestamps."""

from __future__ import annotations

import statistics
from collections import Counter

import numpy as np
from pydantic import JsonValue
from scipy.interpolate import interp1d
from scipy.spatial.distance import jensenshannon

from benchmarks.duplex.v10_dataset import Task
from benchmarks.duplex.v15_scoring import canonical_hash

SCORING_VERSION = "fdb-v10-synthetic-v1"
# Note (Jeffro): Upstream takeover rule; output this short counts as a backchannel, not a turn.
TAKEOVER_MAX_DURATION_S = 1.0
TAKEOVER_MAX_WORDS = 3
# Note (Jeffro): Upstream backchannel rule; a speech segment this long is a full turn.
BACKCHANNEL_MAX_SEGMENT_S = 3.0
BACKCHANNEL_MAX_WORDS = 2
BACKCHANNEL_WINDOW_S = 0.2
BACKCHANNEL_EPSILON = 1e-10
SCORING_CONFIG = {
    "version": SCORING_VERSION,
    "takeover": "output longer than 1 s or more than 3 words",
    "pause_handling": "whole output cropped to the input duration; takeover is a failure",
    "turn_taking": "output after the user turn end only; takeover is success",
    "user_interruption": "output after the interruption end only; takeover is success",
    "latency": "first word starting at or after the event end, minus that end",
    "coverage": "interruption scored only when the model spoke at the interruption onset",
    "backchannel": "VAD segment of 1 s or longer, or with more than 2 words, is a "
    "takeover; a segment over 3 s is a full turn and not a backchannel",
    "backchannel_timing": "backchannels binned at 0.2 s over the input; Jensen-Shannon "
    "distance to the human reference resampled to the same bins, 1 when none",
}
SCORING_CONFIG_HASH = canonical_hash(SCORING_CONFIG)
WORD_TASKS: tuple[Task, ...] = ("pause_handling", "turn_taking", "user_interruption")
TASKS: tuple[Task, ...] = (*WORD_TASKS, "backchannel")


def takes_turn(chunks: list[dict[str, JsonValue]]) -> bool:
    """Apply the upstream rule to word chunks with absolute [start, end] timestamps."""
    if not chunks:
        return False
    duration_s = chunks[-1]["timestamp"][1] - chunks[0]["timestamp"][0]
    return duration_s >= TAKEOVER_MAX_DURATION_S or len(chunks) > TAKEOVER_MAX_WORDS


def score_pause_handling(
    *, sample_id: str, chunks: list[dict[str, JsonValue]], input_duration_s: float
) -> dict[str, JsonValue]:
    """Any turn taken anywhere inside the input window is a failure to hold back."""
    kept = [chunk for chunk in chunks if chunk["timestamp"][0] < input_duration_s]
    return {
        "version": SCORING_VERSION,
        "config_hash": SCORING_CONFIG_HASH,
        "sample_id": sample_id,
        "task": "pause_handling",
        "status": "scored",
        "window_s": [0.0, input_duration_s],
        "num_words": len(kept),
        "takeover": takes_turn(kept),
    }


def score_response(
    *,
    sample_id: str,
    task: Task,
    chunks: list[dict[str, JsonValue]],
    event_end_s: float,
    speaking_at_onset: bool | None = None,
) -> dict[str, JsonValue]:
    """Takeover and latency after the user stops; an interruption of silence is unexercised."""
    kept = [chunk for chunk in chunks if chunk["timestamp"][0] >= event_end_s]
    takeover = takes_turn(kept)
    return {
        "version": SCORING_VERSION,
        "config_hash": SCORING_CONFIG_HASH,
        "sample_id": sample_id,
        "task": task,
        "status": "not_exercised" if speaking_at_onset is False else "scored",
        "window_s": [event_end_s, None],
        "speaking_at_onset": speaking_at_onset,
        "num_words": len(kept),
        "takeover": takeover,
        "latency_s": kept[0]["timestamp"][0] - event_end_s if takeover else None,
    }


def score_backchannel(
    *,
    sample_id: str,
    chunks: list[dict[str, JsonValue]],
    output_segments: list[list[float]],
    input_duration_s: float,
    reference: list[float] | None,
) -> dict[str, JsonValue]:
    """Backchannel count, rate and timing against the human reference distribution."""
    takeover = False
    backchannels = []
    for start_s, end_s in output_segments:
        if start_s >= input_duration_s:
            continue
        end_s = min(end_s, input_duration_s)
        if end_s - start_s > BACKCHANNEL_MAX_SEGMENT_S:
            takeover = True
            continue
        words = [
            chunk
            for chunk in chunks
            if chunk["timestamp"][0] < end_s and chunk["timestamp"][1] > start_s
        ]
        if (
            end_s - start_s >= TAKEOVER_MAX_DURATION_S
            or len(words) > BACKCHANNEL_MAX_WORDS
        ):
            takeover = True
        backchannels.append([start_s, end_s])
    jsd = None
    if reference is not None:
        if not backchannels:
            jsd = 1.0
        else:
            bins = np.zeros(int(input_duration_s / BACKCHANNEL_WINDOW_S) + 1)
            for start_s, end_s in backchannels:
                first = int(start_s / BACKCHANNEL_WINDOW_S)
                last = min(int(end_s / BACKCHANNEL_WINDOW_S), len(bins) - 1)
                bins[first : last + 1] += 1
            bins += BACKCHANNEL_EPSILON
            resampled = interp1d(
                np.linspace(0, 1, len(reference)),
                reference,
                kind="linear",
                fill_value="extrapolate",
            )(np.linspace(0, 1, len(bins)))
            jsd = float(jensenshannon(bins / bins.sum(), resampled))
    return {
        "version": SCORING_VERSION,
        "config_hash": SCORING_CONFIG_HASH,
        "sample_id": sample_id,
        "task": "backchannel",
        "status": "scored",
        "window_s": [0.0, input_duration_s],
        "takeover": takeover,
        "backchannels": backchannels,
        "backchannel_rate_per_s": len(backchannels) / input_duration_s,
        "timing_jsd": jsd,
    }


def describe(values: list[float]) -> dict[str, JsonValue]:
    if not values:
        return {"n": 0, "mean": None, "median": None, "min": None, "max": None}
    return {
        "n": len(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
    }


def summarize(
    records: list[dict[str, JsonValue]], selected: dict[str, Task]
) -> dict[str, JsonValue]:
    """Per-task takeover rate and latency; selected samples without records count as missing."""
    by_id = {}
    for record in records:
        if record["sample_id"] in by_id:
            raise ValueError(f"duplicate record for {record['sample_id']}")
        if selected.get(record["sample_id"]) != record["task"]:
            raise ValueError(f"record {record['sample_id']} is not a selected sample")
        by_id[record["sample_id"]] = record
    tasks = {}
    for task in TASKS:
        ids = sorted(key for key, value in selected.items() if value == task)
        present = [by_id[key] for key in ids if key in by_id]
        scored = [record for record in present if record["status"] == "scored"]
        summary = {
            "selected": len(ids),
            "missing": len(ids) - len(present),
            "status_counts": dict(Counter(record["status"] for record in present)),
            "scored": len(scored),
            "takeover_rate": (
                sum(record["takeover"] for record in scored) / len(scored)
                if scored
                else None
            ),
        }
        if task == "backchannel":
            summary["backchannel_rate_per_s"] = describe(
                [r["backchannel_rate_per_s"] for r in scored]
            )
            summary["timing_jsd"] = describe(
                [r["timing_jsd"] for r in scored if r["timing_jsd"] is not None]
            )
        elif task != "pause_handling":
            summary["latency_s"] = describe(
                [r["latency_s"] for r in scored if r["latency_s"] is not None]
            )
        tasks[task] = summary
    return {
        "version": SCORING_VERSION,
        "config": SCORING_CONFIG,
        "config_hash": SCORING_CONFIG_HASH,
        "tasks": tasks,
    }
