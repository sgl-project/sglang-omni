# SPDX-License-Identifier: Apache-2.0
"""Score Full-Duplex-Bench v1.0 turn-taking tasks from word timestamps."""

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
from scipy.interpolate import interp1d
from scipy.signal import resample_poly
from scipy.spatial.distance import jensenshannon

from benchmarks.duplex.v10_dataset import Task

SCORING_VERSION = "fdb-v10-synthetic-v3"
# note (Jeffro): Upstream takeover rule; output this short counts as a backchannel, not a turn.
TAKEOVER_MAX_DURATION_S = 1.0
TAKEOVER_MAX_WORDS = 3
# note (Jeffro): Upstream backchannel rule; a speech segment this long is a full turn.
BACKCHANNEL_MAX_SEGMENT_S = 3.0
BACKCHANNEL_MAX_WORDS = 2
BACKCHANNEL_WINDOW_S = 0.2
BACKCHANNEL_EPSILON = 1e-10
SCORING_CONFIG = {
    "version": SCORING_VERSION,
    "input": "a word list with timestamps, transcribed by ASR from "
    "output-playout.wav, where each audio chunk sits at the later of its arrival "
    "time and the end of the previous chunk.",
    "takeover": "the model took the turn when its output lasts 1 s or longer or has "
    "more than 3 words, anything shorter counts as a backchannel",
    "scoring_window": "only words that start before the input audio ends are "
    "scored, the playout keeps recording while the server drains after EOS, and "
    "that tail is outside the benchmark",
    "pause_handling": "the user pauses mid-sentence; a takeover anywhere in the input "
    "window means the model wrongly treated the pause as the end of the turn; "
    "lower takeover rate is better; words outside Silero VAD speech in the output "
    "are dropped first, because Whisper invents words on silent output",
    "turn_taking": "the user finishes; words starting at or after the annotated turn "
    "end count, a takeover is the wanted response, and latency is the first such "
    "word minus the turn end; if Silero VAD shows the model already speaking at the "
    "turn end it talked over the user, so the sample is spoke_before_turn_end and excluded",
    "user_interruption": "the user interrupts the model's answer; words starting at "
    "or after the interruption end count, a takeover means the model spoke after "
    "it, and latency is the first such word minus the interruption end; if VAD "
    "shows no model speech at the interruption onset nothing was interrupted, so "
    "the sample is not_exercised; if one VAD segment runs from before the onset "
    "to after the interruption end the model never stopped, so the sample is "
    "talked_through; both are excluded from the rate and latency",
    "right_censored": "flag only: the model's output speech reaches within 50 ms of "
    "the input end, so the window may have cut a response short; the score still "
    "counts",
    "backchannel": "the user talks for 20-80 s and the model should acknowledge "
    "without taking over; each Silero VAD segment of the output is a takeover when "
    "it lasts 1 s or longer or has more than 2 words, a segment over 3 s is a full "
    "turn and never a backchannel, and the remaining short segments are "
    "backchannels reported as a rate per second",
    "backchannel_timing": "backchannel segments are binned at 0.2 s across the "
    "input and compared with the human timing distribution from upstream "
    "icc_gt_distribution.json by Jensen-Shannon distance, lower is closer to "
    "human timing; a sample with no backchannels scores 1",
}
WORD_TASKS: tuple[Task, ...] = ("pause_handling", "turn_taking", "user_interruption")
TASKS: tuple[Task, ...] = (*WORD_TASKS, "backchannel")
# Only "scored" records enter a task's averages, the other statuses are counted.
ScoreStatus = Literal[
    "scored", "spoke_before_turn_end", "not_exercised", "talked_through"
]
SCORED: ScoreStatus = "scored"
CENSOR_TOLERANCE_S = 0.05


# note (wenyao): Float rounding can put segment ends just past the audio duration.
DURATION_TOLERANCE_S = 1e-3
SILERO_VAD_CONFIG = {
    "sampling_rate": 16000,
    "threshold": 0.5,
    "min_speech_duration_ms": 250,
    "min_silence_duration_ms": 100,
    "speech_pad_ms": 30,
    "onnx": True,
}


def canonical_hash(value: JsonValue) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


SCORING_CONFIG_HASH = canonical_hash(SCORING_CONFIG)


def check_interval(start_s: float, end_s: float, duration_s: float, name: str) -> None:
    if not all(math.isfinite(value) for value in (start_s, end_s, duration_s)):
        raise ValueError(f"{name} has a non-finite time")
    elif not 0 <= start_s < end_s <= duration_s + DURATION_TOLERANCE_S:
        raise ValueError(
            f"{name} [{start_s}, {end_s}] is reversed or outside [0, {duration_s}]"
        )
    else:
        pass


def validate_segments(
    segments: list[list[float]], duration_s: float, name: str
) -> list[tuple[float, float]]:
    """Return ordered, disjoint, in-range speech segments or raise ValueError."""
    checked = []
    previous_end = 0.0
    for index, segment in enumerate(segments):
        if len(segment) != 2:
            raise ValueError(f"{name} segment {index} must be [start, end]")
        else:
            pass
        start_s, end_s = float(segment[0]), float(segment[1])
        check_interval(start_s, end_s, duration_s, f"{name} segment {index}")
        if start_s < previous_end:
            raise ValueError(f"{name} segment {index} overlaps or is out of order")
        else:
            pass
        checked.append((start_s, end_s))
        previous_end = end_s
    return checked


def silero_speech_segments(wav_path: str | Path) -> dict[str, JsonValue]:
    """Detect speech in a PCM WAV with the frozen Silero VAD configuration."""
    # note (wenyao): Word-timestamp scoring and tests do not need torch or Silero.
    import torch
    from silero_vad import get_speech_timestamps, load_silero_vad

    source_audio = soundfile.info(str(wav_path))
    if source_audio.format != "WAV" or not source_audio.subtype.startswith("PCM"):
        raise ValueError(
            f"{wav_path} is {source_audio.format}/{source_audio.subtype}, expected PCM WAV"
        )
    else:
        pass
    audio, sample_rate = soundfile.read(str(wav_path), dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)
    duration_s = len(audio) / sample_rate
    target_rate = SILERO_VAD_CONFIG["sampling_rate"]
    if sample_rate != target_rate:
        divisor = gcd(sample_rate, target_rate)
        audio = resample_poly(audio, target_rate // divisor, sample_rate // divisor)
    else:
        pass
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
        [
            timestamp_span["start"] / target_rate,
            min(timestamp_span["end"] / target_rate, duration_s),
        ]
        for timestamp_span in timestamps
    ]
    return {
        "segments": [
            list(interval)
            for interval in validate_segments(segments, duration_s, "vad")
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


def takes_turn(chunks: list[dict[str, JsonValue]]) -> bool:
    """Apply the upstream rule to word chunks with absolute [start, end] timestamps."""
    if not chunks:
        return False
    else:
        pass
    duration_s = chunks[-1]["timestamp"][1] - chunks[0]["timestamp"][0]
    return duration_s >= TAKEOVER_MAX_DURATION_S or len(chunks) > TAKEOVER_MAX_WORDS


def score_pause_handling(
    *,
    sample_id: str,
    chunks: list[dict[str, JsonValue]],
    input_duration_s: float,
    output_segments: list[list[float]],
) -> dict[str, JsonValue]:
    """Any turn taken anywhere inside the input window is a failure to hold back."""
    # note (Junnan Li): Whisper returns words such as "Thank you." for silent output.
    window_chunks = [
        chunk
        for chunk in chunks
        if chunk["timestamp"][0] < input_duration_s
        and any(
            chunk["timestamp"][0] < end_s and chunk["timestamp"][1] > start_s
            for start_s, end_s in output_segments
        )
    ]
    return {
        "version": SCORING_VERSION,
        "config_hash": SCORING_CONFIG_HASH,
        "sample_id": sample_id,
        "task": "pause_handling",
        "status": SCORED,
        "window_s": [0.0, input_duration_s],
        "num_words": len(window_chunks),
        "takeover": takes_turn(window_chunks),
    }


def speaking_at(output_segments: list[list[float]], moment_s: float) -> bool:
    return any(start_s <= moment_s < end_s for start_s, end_s in output_segments)


def score_response_after(
    *,
    sample_id: str,
    task: Task,
    status: ScoreStatus,
    speaking_at_event: bool,
    chunks: list[dict[str, JsonValue]],
    user_end_s: float,
    input_duration_s: float,
    output_segments: list[list[float]],
) -> dict[str, JsonValue]:
    """Takeover and latency of the words after user_end_s, inside the input window."""
    window_chunks = [
        chunk
        for chunk in chunks
        if user_end_s <= chunk["timestamp"][0] < input_duration_s
    ]
    takeover = takes_turn(window_chunks)
    return {
        "version": SCORING_VERSION,
        "config_hash": SCORING_CONFIG_HASH,
        "sample_id": sample_id,
        "task": task,
        "status": status,
        "window_s": [user_end_s, input_duration_s],
        "speaking_at_event": speaking_at_event,
        "right_censored": any(
            end_s >= input_duration_s - CENSOR_TOLERANCE_S
            for _, end_s in output_segments
        ),
        "num_words": len(window_chunks),
        "takeover": takeover,
        "latency_s": (
            window_chunks[0]["timestamp"][0] - user_end_s if takeover else None
        ),
    }


def score_turn_taking(
    *,
    sample_id: str,
    chunks: list[dict[str, JsonValue]],
    turn_end_s: float,
    input_duration_s: float,
    output_segments: list[list[float]],
) -> dict[str, JsonValue]:
    """A model already speaking when the user turn ends talked over the user."""
    speaking = speaking_at(output_segments, turn_end_s)
    return score_response_after(
        sample_id=sample_id,
        task="turn_taking",
        status="spoke_before_turn_end" if speaking else SCORED,
        speaking_at_event=speaking,
        chunks=chunks,
        user_end_s=turn_end_s,
        input_duration_s=input_duration_s,
        output_segments=output_segments,
    )


def score_interruption(
    *,
    sample_id: str,
    chunks: list[dict[str, JsonValue]],
    interruption_start_s: float,
    interruption_end_s: float,
    input_duration_s: float,
    output_segments: list[list[float]],
) -> dict[str, JsonValue]:
    """A silent model was not interrupted; one that never paused did not respond."""
    speaking = speaking_at(output_segments, interruption_start_s)
    status: ScoreStatus
    if not speaking:
        status = "not_exercised"
    elif any(
        start_s <= interruption_start_s and end_s > interruption_end_s
        for start_s, end_s in output_segments
    ):
        status = "talked_through"
    else:
        status = SCORED
    return score_response_after(
        sample_id=sample_id,
        task="user_interruption",
        status=status,
        speaking_at_event=speaking,
        chunks=chunks,
        user_end_s=interruption_end_s,
        input_duration_s=input_duration_s,
        output_segments=output_segments,
    )


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
        else:
            pass
        end_s = min(end_s, input_duration_s)
        if end_s - start_s > BACKCHANNEL_MAX_SEGMENT_S:
            takeover = True
            continue
        else:
            pass
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
        else:
            pass
        backchannels.append([start_s, end_s])
    timing_jsd = None
    if reference is not None:
        if not backchannels:
            timing_jsd = 1.0
        else:
            backchannel_bins = np.zeros(
                int(input_duration_s / BACKCHANNEL_WINDOW_S) + 1
            )
            for start_s, end_s in backchannels:
                first = int(start_s / BACKCHANNEL_WINDOW_S)
                last = min(int(end_s / BACKCHANNEL_WINDOW_S), len(backchannel_bins) - 1)
                backchannel_bins[first : last + 1] += 1
            backchannel_bins += BACKCHANNEL_EPSILON
            resampled = interp1d(
                np.linspace(0, 1, len(reference)),
                reference,
                kind="linear",
                fill_value="extrapolate",
            )(np.linspace(0, 1, len(backchannel_bins)))
            timing_jsd = float(
                jensenshannon(backchannel_bins / backchannel_bins.sum(), resampled)
            )
    else:
        pass
    return {
        "version": SCORING_VERSION,
        "config_hash": SCORING_CONFIG_HASH,
        "sample_id": sample_id,
        "task": "backchannel",
        "status": SCORED,
        "window_s": [0.0, input_duration_s],
        "takeover": takeover,
        "backchannels": backchannels,
        "backchannel_rate_per_s": len(backchannels) / input_duration_s,
        "timing_jsd": timing_jsd,
    }


def summarize(
    records: list[dict[str, JsonValue]], selected: dict[str, Task]
) -> dict[str, JsonValue]:
    """Per-task takeover rate and latency; selected samples without records count as missing."""
    by_id = {}
    for score_record in records:
        if score_record["sample_id"] in by_id:
            raise ValueError(f"duplicate record for {score_record['sample_id']}")
        else:
            pass
        if selected.get(score_record["sample_id"]) != score_record["task"]:
            raise ValueError(
                f"record {score_record['sample_id']} is not a selected sample"
            )
        else:
            pass
        by_id[score_record["sample_id"]] = score_record
    tasks = {}
    for task in TASKS:
        sample_ids = sorted(key for key, value in selected.items() if value == task)
        present = [by_id[key] for key in sample_ids if key in by_id]
        scored = [
            score_record for score_record in present if score_record["status"] == SCORED
        ]
        summary = {
            "selected": len(sample_ids),
            "missing": len(sample_ids) - len(present),
            "status_counts": dict(
                Counter(score_record["status"] for score_record in present)
            ),
            "scored": len(scored),
            "takeover_rate": (
                sum(score_record["takeover"] for score_record in scored) / len(scored)
                if scored
                else None
            ),
        }
        if task in ("turn_taking", "user_interruption"):
            summary["right_censored"] = sum(
                bool(score_record["right_censored"]) for score_record in scored
            )
        else:
            pass
        if task == "backchannel":
            summary["backchannel_rate_per_s"] = describe(
                [score_record["backchannel_rate_per_s"] for score_record in scored]
            )
            summary["timing_jsd"] = describe(
                [
                    score_record["timing_jsd"]
                    for score_record in scored
                    if score_record["timing_jsd"] is not None
                ]
            )
        elif task != "pause_handling":
            summary["latency_s"] = describe(
                [
                    score_record["latency_s"]
                    for score_record in scored
                    if score_record["latency_s"] is not None
                ]
            )
        else:
            pass
        tasks[task] = summary
    return {
        "version": SCORING_VERSION,
        "config": SCORING_CONFIG,
        "config_hash": SCORING_CONFIG_HASH,
        "tasks": tasks,
    }
