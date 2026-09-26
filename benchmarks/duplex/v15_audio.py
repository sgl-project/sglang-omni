# SPDX-License-Identifier: Apache-2.0
"""Normalize v1.5 input audio and rebuild model output audio from native traces."""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np
import soundfile
from pydantic import JsonValue
from scipy.signal import resample_poly

from benchmarks.duplex.client import PACKET_MS, SAMPLE_RATE
from benchmarks.duplex.profiles import DEFAULT_PROFILE, PROFILES, ProfileName

PLAYOUT_CAVEATS = [
    "physical_acoustics",
    "client_jitter_buffer",
    "paper_identical_latency",
]
TRANSCRIPT_DELTAS = (
    "response.output_audio_transcript.delta",
    "response.output_text.delta",
)
# Note (wenyao): Playout and event spans assume paced source time within one packet.
PACING_TOLERANCE_S = PACKET_MS / 1000
INPUT_TIMING_CAVEAT = (
    "client send clock only; network, server receipt and acoustic onset are unmeasured"
)


def write_json(path: Path, value: JsonValue) -> None:
    """Replace path atomically so readers never observe a partial document."""
    partial = path.with_name(f".{path.name}.partial")
    partial.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(partial, path)


def normalize_audio(path: Path) -> tuple[bytes, dict[str, JsonValue]]:
    """Convert to mono 16 kHz PCM16 without trimming, padding or channel picking."""
    info = soundfile.info(str(path))
    record: dict[str, JsonValue] = {
        "source_sample_rate": info.samplerate,
        "source_channels": info.channels,
        "source_subtype": info.subtype,
        "source_frames": info.frames,
        "downmix": None,
        "resample": None,
        "pcm16": "exact",
        "clipped_samples": 0,
    }
    if (info.samplerate, info.channels, info.subtype) == (SAMPLE_RATE, 1, "PCM_16"):
        pcm = soundfile.read(str(path), dtype="int16")[0].astype("<i2").tobytes()
    else:
        audio = soundfile.read(str(path), dtype="float64", always_2d=True)[0]
        nonfinite = int(np.count_nonzero(~np.isfinite(audio)))
        if nonfinite:
            raise ValueError(f"{path.name} has {nonfinite} non-finite source samples")
        if info.channels > 1:
            audio = audio.mean(axis=1, keepdims=True)
            record["downmix"] = f"mean of {info.channels} channels"
        samples = audio[:, 0]
        if info.samplerate != SAMPLE_RATE:
            divisor = math.gcd(SAMPLE_RATE, info.samplerate)
            up, down = SAMPLE_RATE // divisor, info.samplerate // divisor
            samples = resample_poly(samples, up, down)
            record["resample"] = {
                "method": "scipy.signal.resample_poly",
                "up": up,
                "down": down,
            }
        scaled = np.round(samples * 32768)
        if not np.isfinite(scaled).all():
            raise ValueError(f"{path.name} became non-finite after downmix/resampling")
        record["pcm16"] = "float scaled by 32768, rounded, clipped to int16"
        record["clipped_samples"] = int(
            np.count_nonzero((scaled < -32768) | (scaled > 32767))
        )
        pcm = np.clip(scaled, -32768, 32767).astype("<i2").tobytes()
    record["frames"] = len(pcm) // 2
    return pcm, record


def reconstruct_output(
    variant_dir: Path, *, profile: ProfileName = DEFAULT_PROFILE
) -> dict[str, JsonValue]:
    """Rebuild model audio and a SIMULATED zero-buffer client playout from a trace."""
    contract = PROFILES[profile]
    output_sample_rate = contract.output_sample_rate
    errors: list[str] = []
    records = []
    with (variant_dir / "continuous.jsonl").open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                records.append((line_number, json.loads(line)))
            except ValueError as exc:
                errors.append(f"trace line {line_number} unreadable: {exc}")
    origin_s = next(
        (
            record["time_s"]
            for _, record in records
            if record["direction"] == "send"
            and record["event"].get("type") == "input_audio_buffer.append"
        ),
        None,
    )
    if origin_s is None:
        errors.append("no input_audio_buffer.append was sent")
    media = bytearray()
    playout = bytearray()
    chunks = []
    transcript_events = []
    appends = []
    for line_number, record in records:
        event = record["event"]
        kind = event.get("type")
        if origin_s is None:
            break
        elapsed_s = record["time_s"] - origin_s
        if record["direction"] == "send" and kind == "input_audio_buffer.append":
            source_start_ms = event.get("sglang", {}).get("t_start_ms")
            if type(source_start_ms) not in (int, float):
                errors.append(f"trace line {line_number}: append lacks t_start_ms")
                continue
            appends.append(
                {
                    "index": len(appends),
                    "trace_line": line_number,
                    "source_start_s": source_start_ms / 1000,
                    "send_elapsed_s": elapsed_s,
                    "send_deviation_s": elapsed_s - source_start_ms / 1000,
                }
            )
        elif record["direction"] != "receive":
            continue
        elif kind in TRANSCRIPT_DELTAS:
            transcript_events.append(
                {
                    "trace_line": line_number,
                    "type": kind,
                    "response_id": event.get("response_id"),
                    "receipt_s": elapsed_s,
                    "delta": event.get("delta"),
                }
            )
        elif kind == "response.output_audio.delta":
            try:
                pcm = base64.b64decode(event.get("delta") or "", validate=True)
            except (binascii.Error, TypeError) as exc:
                errors.append(f"trace line {line_number}: invalid audio base64: {exc}")
                continue
            if not pcm or len(pcm) % 2:
                errors.append(f"trace line {line_number}: truncated PCM16 audio delta")
                continue
            receive_sample = round(elapsed_s * output_sample_rate)
            playout_start = max(len(playout) // 2, receive_sample)
            chunks.append(
                {
                    "index": len(chunks),
                    "trace_line": line_number,
                    "response_id": event.get("response_id"),
                    "receipt_s": elapsed_s,
                    "receive_sample": receive_sample,
                    "samples": len(pcm) // 2,
                    "media_start_sample": len(media) // 2,
                    "playout_start_sample": playout_start,
                }
            )
            playout.extend(bytes(2 * (playout_start - len(playout) // 2)))
            playout.extend(pcm)
            media.extend(pcm)
    if origin_s is not None and not media and contract.continuous_output:
        errors.append("no model output audio")
    worst = max(appends, key=lambda item: abs(item["send_deviation_s"]), default=None)
    input_timing = {
        "tolerance_s": PACING_TOLERANCE_S,
        "reference": "sglang.t_start_ms versus client send time after the first append",
        "uncertainty": INPUT_TIMING_CAVEAT,
        "max_abs_deviation_s": abs(worst["send_deviation_s"]) if worst else None,
        "max_deviation_index": worst["index"] if worst else None,
        "within_tolerance": bool(
            worst and abs(worst["send_deviation_s"]) <= PACING_TOLERANCE_S
        ),
    }
    if worst and not input_timing["within_tolerance"]:
        errors.append(
            f"input pacing deviation {worst['send_deviation_s']:.3f}s at append "
            f"{worst['index']} (trace line {worst['trace_line']}) exceeds "
            f"{PACING_TOLERANCE_S}s"
        )
    if not media and not contract.continuous_output and appends:
        accepted = [
            r["event"]["accepted_end_ms"]
            for _, r in records
            if r["direction"] == "receive"
            and r["event"].get("type") == "sglang.input_audio.accepted"
        ]
        if accepted:
            playout.extend(bytes(round(max(accepted) * output_sample_rate / 1000) * 2))
    pcm_sha256 = {}
    for name, pcm in (("output-media.wav", media), ("output-playout.wav", playout)):
        if pcm or not contract.continuous_output:
            soundfile.write(
                str(variant_dir / name),
                np.frombuffer(bytes(pcm), dtype="<i2"),
                output_sample_rate,
                subtype="PCM_16",
            )
            pcm_sha256[name] = hashlib.sha256(pcm).hexdigest()
    texts: dict[str, str] = {}
    for item in transcript_events:
        texts[item["type"]] = texts.get(item["type"], "") + (item["delta"] or "")
    write_json(
        variant_dir / "transcript.json",
        {
            "provenance": "native server transcript deltas from continuous.jsonl",
            "independent_asr": False,
            "events": transcript_events,
            "text": texts,
        },
    )
    summary = {
        "kind": "simulated_zero_buffer_client_playout",
        "not": PLAYOUT_CAVEATS,
        "sample_rate": output_sample_rate,
        "origin": "first input_audio_buffer.append send time",
        "media_samples": len(media) // 2,
        "playout_samples": len(playout) // 2,
        "initial_delay_s": (
            chunks[0]["playout_start_sample"] / output_sample_rate if chunks else None
        ),
        "pcm_sha256": pcm_sha256,
        "input_timing": input_timing,
        "errors": errors,
    }
    write_json(
        variant_dir / "playout.json",
        {
            **summary,
            "input_timing": {**input_timing, "appends": appends},
            "chunks": chunks,
        },
    )
    return summary
