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
AUDIO_DELTA = "response.output_audio.delta"
LEGACY_TRANSCRIPT_DELTAS = ("response.text.delta",)
LEGACY_AUDIO_DELTA = "response.audio.delta"
# note (luojiaxuan): The facade clears the output buffer whenever it interrupts a
# response; a client that asked for interruption also stops at speech_started.
LEGACY_BARGE_IN_CUT = (
    "playout cursor cut at the client receipt of output_audio_buffer.cleared, or of "
    "input_audio_buffer.speech_started when turn_detection.interrupt_response is not "
    "false"
)
# note (wenyao): Playout and event spans assume paced source time within one packet.
PACING_TOLERANCE_S = PACKET_MS / 1000
INPUT_TIMING_CAVEAT = (
    "client send clock only; network, server receipt and acoustic onset are unmeasured"
)


def write_json(path: Path, value: JsonValue) -> None:
    """Replace path atomically so readers never observe a partial document."""
    partial_path = path.with_name(f".{path.name}.partial")
    partial_path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(partial_path, path)


def normalize_audio(path: Path) -> tuple[bytes, dict[str, JsonValue]]:
    """Convert to mono 16 kHz PCM16 without trimming, padding or channel picking."""
    source_audio = soundfile.info(str(path))
    record: dict[str, JsonValue] = {
        "source_sample_rate": source_audio.samplerate,
        "source_channels": source_audio.channels,
        "source_subtype": source_audio.subtype,
        "source_frames": source_audio.frames,
        "downmix": None,
        "resample": None,
        "pcm16": "exact",
        "clipped_samples": 0,
    }
    if (source_audio.samplerate, source_audio.channels, source_audio.subtype) == (
        SAMPLE_RATE,
        1,
        "PCM_16",
    ):
        pcm = soundfile.read(str(path), dtype="int16")[0].astype("<i2").tobytes()
    else:
        audio = soundfile.read(str(path), dtype="float64", always_2d=True)[0]
        nonfinite = int(np.count_nonzero(~np.isfinite(audio)))
        if nonfinite:
            raise ValueError(f"{path.name} has {nonfinite} non-finite source samples")
        else:
            pass
        if source_audio.channels > 1:
            audio = audio.mean(axis=1, keepdims=True)
            record["downmix"] = f"mean of {source_audio.channels} channels"
        else:
            pass
        samples = audio[:, 0]
        if source_audio.samplerate != SAMPLE_RATE:
            divisor = math.gcd(SAMPLE_RATE, source_audio.samplerate)
            upsample_factor, downsample_factor = (
                SAMPLE_RATE // divisor,
                source_audio.samplerate // divisor,
            )
            samples = resample_poly(samples, upsample_factor, downsample_factor)
            record["resample"] = {
                "method": "scipy.signal.resample_poly",
                "up": upsample_factor,
                "down": downsample_factor,
            }
        else:
            pass
        scaled = np.round(samples * 32768)
        if not np.isfinite(scaled).all():
            raise ValueError(f"{path.name} became non-finite after downmix/resampling")
        else:
            pass
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
    profile_contract = PROFILES[profile]
    output_sample_rate = profile_contract.output_sample_rate
    legacy = profile_contract.protocol == "legacy"
    audio_delta = LEGACY_AUDIO_DELTA if legacy else AUDIO_DELTA
    transcript_deltas = LEGACY_TRANSCRIPT_DELTAS if legacy else TRANSCRIPT_DELTAS
    errors: list[str] = []
    records = []
    with (variant_dir / "continuous.jsonl").open(encoding="utf-8") as trace_file:
        for line_number, line in enumerate(trace_file, start=1):
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
    else:
        pass
    media = bytearray()
    playout = bytearray()
    chunks = []
    cuts = []
    transcript_events = []
    appends = []
    input_end_s = 0.0
    interrupts_on_speech = None
    for line_number, record in records:
        event = record["event"]
        event_type = event.get("type")
        if origin_s is None:
            break
        else:
            pass
        elapsed_s = record["time_s"] - origin_s
        if record["direction"] == "send" and event_type == "input_audio_buffer.append":
            source_start_ms = event.get("sglang", {}).get("t_start_ms")
            if type(source_start_ms) not in (int, float):
                errors.append(f"trace line {line_number}: append lacks t_start_ms")
                continue
            else:
                pass
            appends.append(
                {
                    "index": len(appends),
                    "trace_line": line_number,
                    "source_start_s": source_start_ms / 1000,
                    "send_elapsed_s": elapsed_s,
                    "send_deviation_s": elapsed_s - source_start_ms / 1000,
                }
            )
            if legacy:
                input_end_s = source_start_ms / 1000 + len(
                    base64.b64decode(event["audio"], validate=True)
                ) / (2 * SAMPLE_RATE)
            else:
                pass
        elif (
            legacy
            and record["direction"] == "send"
            and event_type == "session.update"
            and interrupts_on_speech is None
        ):
            turn_detection = event["session"].get("turn_detection") or {}
            interrupts_on_speech = turn_detection.get("interrupt_response") is not False
        elif record["direction"] != "receive":
            continue
        elif legacy and (
            event_type == "output_audio_buffer.cleared"
            or (
                event_type == "input_audio_buffer.speech_started"
                and interrupts_on_speech
            )
        ):
            cut_sample = round(elapsed_s * output_sample_rate)
            dropped = len(playout) // 2 - cut_sample
            if dropped > 0:
                del playout[2 * cut_sample :]
                cuts.append(
                    {
                        "trace_line": line_number,
                        "type": event_type,
                        "receipt_s": elapsed_s,
                        "cut_sample": cut_sample,
                        "dropped_samples": dropped,
                    }
                )
            else:
                pass
        elif event_type in transcript_deltas:
            transcript_events.append(
                {
                    "trace_line": line_number,
                    "type": event_type,
                    "response_id": event.get("response_id"),
                    "receipt_s": elapsed_s,
                    "delta": event.get("delta"),
                }
            )
        elif event_type == audio_delta:
            try:
                pcm = base64.b64decode(event.get("delta") or "", validate=True)
            except (binascii.Error, TypeError) as exc:
                errors.append(f"trace line {line_number}: invalid audio base64: {exc}")
                continue
            if not pcm or len(pcm) % 2:
                errors.append(f"trace line {line_number}: truncated PCM16 audio delta")
                continue
            else:
                pass
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
        else:
            pass
    if origin_s is not None and not media and profile_contract.continuous_output:
        errors.append("no model output audio")
    else:
        pass
    largest_deviation = max(
        appends, key=lambda append: abs(append["send_deviation_s"]), default=None
    )
    input_timing = {
        "tolerance_s": PACING_TOLERANCE_S,
        "reference": "sglang.t_start_ms versus client send time after the first append",
        "uncertainty": INPUT_TIMING_CAVEAT,
        "max_abs_deviation_s": (
            abs(largest_deviation["send_deviation_s"]) if largest_deviation else None
        ),
        "max_deviation_index": (
            largest_deviation["index"] if largest_deviation else None
        ),
        "within_tolerance": bool(
            largest_deviation
            and abs(largest_deviation["send_deviation_s"]) <= PACING_TOLERANCE_S
        ),
    }
    if largest_deviation and not input_timing["within_tolerance"]:
        errors.append(
            f"input pacing deviation {largest_deviation['send_deviation_s']:.3f}s at append "
            f"{largest_deviation['index']} (trace line {largest_deviation['trace_line']}) exceeds "
            f"{PACING_TOLERANCE_S}s"
        )
    else:
        pass
    if media or profile_contract.continuous_output or not appends:
        pass
    elif legacy:
        playout.extend(bytes(round(input_end_s * output_sample_rate) * 2))
    else:
        accepted = [
            trace_record["event"]["accepted_end_ms"]
            for _, trace_record in records
            if trace_record["direction"] == "receive"
            and trace_record["event"].get("type") == "sglang.input_audio.accepted"
        ]
        if accepted:
            playout.extend(bytes(round(max(accepted) * output_sample_rate / 1000) * 2))
        else:
            pass
    pcm_sha256 = {}
    for name, pcm in (("output-media.wav", media), ("output-playout.wav", playout)):
        if pcm or not profile_contract.continuous_output:
            soundfile.write(
                str(variant_dir / name),
                np.frombuffer(bytes(pcm), dtype="<i2"),
                output_sample_rate,
                subtype="PCM_16",
            )
            pcm_sha256[name] = hashlib.sha256(pcm).hexdigest()
        else:
            pass
    transcript_texts: dict[str, str] = {}
    for transcript_event in transcript_events:
        transcript_texts[transcript_event["type"]] = transcript_texts.get(
            transcript_event["type"], ""
        ) + (transcript_event["delta"] or "")
    write_json(
        variant_dir / "transcript.json",
        {
            "provenance": (
                f"{profile_contract.protocol} server transcript deltas from "
                "continuous.jsonl"
            ),
            "independent_asr": False,
            "events": transcript_events,
            "text": transcript_texts,
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
    if legacy:
        summary["barge_in"] = {"rule": LEGACY_BARGE_IN_CUT, "cuts": len(cuts)}
    else:
        pass
    write_json(
        variant_dir / "playout.json",
        {
            **summary,
            "input_timing": {**input_timing, "appends": appends},
            "chunks": chunks,
            **({"cuts": cuts} if legacy else {}),
        },
    )
    return summary
