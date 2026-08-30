#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Prepare and benchmark the Nemotron VoiceChat realtime workload."""

from __future__ import annotations

import argparse
import asyncio
import base64
import hashlib
import json
import math
import statistics
import time
import wave
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import soxr
import websockets

INPUT_RATE = 16_000
SOURCE_RATE = 24_000
SGLANG_OUTPUT_RATE = 22_050
NIM_OUTPUT_RATE = 24_000
FRAME_SAMPLES = 1_280
FRAME_BYTES = FRAME_SAMPLES * 2
FRAME_SECONDS = FRAME_SAMPLES / INPUT_RATE
SOURCE_FRAME_SAMPLES = round(SOURCE_RATE * FRAME_SECONDS)
TRAILING_SILENCE_SECONDS = 2.0
DEFAULT_PROMPT = "You are a helpful, concise voice assistant."
AUDIO_EVENT_TYPES = {
    "response.audio.delta",
    "response.output_audio.delta",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def percentile(values: list[float], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = (len(ordered) - 1) * probability
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - index) + ordered[upper] * (index - lower)


def summarize(values: list[float]) -> dict[str, float | int | None]:
    return {
        "count": len(values),
        "mean": statistics.fmean(values) if values else None,
        "p50": percentile(values, 0.50),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
        "min": min(values) if values else None,
        "max": max(values) if values else None,
    }


def aggregate_report(report: dict[str, Any]) -> dict[str, Any]:
    first_audio = []
    native_intervals = []
    paired_intervals = []
    transcripts = set()
    audio_event_counts = set()
    output_sample_counts = set()

    for run in report["runs"]:
        if run["first_audio_ms"] is not None:
            first_audio.append(run["first_audio_ms"])
        transcripts.add(run["text"])
        audio_event_counts.add(run["audio_event_count"])
        output_sample_counts.add(run["output_samples"])

        arrivals = [
            event["arrival_ms"]
            for event in run["events"]
            if event["type"] in AUDIO_EVENT_TYPES
        ]
        native_intervals.extend(
            current - previous for previous, current in zip(arrivals, arrivals[1:])
        )

        # Each response event contains 80 ms of audio for both backends. Pair
        # adjacent events to compare an equivalent 160 ms media duration.
        paired_arrivals = arrivals[1::2]
        close_sent_ms = run.get("session_close_sent_ms")
        paired_intervals.extend(
            current - previous
            for previous, current in zip(paired_arrivals, paired_arrivals[1:])
            if close_sent_ms is None or not (previous < close_sent_ms <= current)
        )

    return {
        "first_audio_ms": summarize(first_audio),
        "native_audio_event_interval_ms": summarize(native_intervals),
        "paired_160ms_interval_ms": summarize(paired_intervals),
        "transcripts": sorted(transcripts),
        "audio_event_counts": sorted(audio_event_counts),
        "output_sample_counts": sorted(output_sample_counts),
    }


def prepare_input(source_path: Path, output_path: Path) -> dict[str, Any]:
    """Match the NIM file client and produce exact 80 ms model frames."""
    audio, sample_rate = sf.read(source_path, dtype="float32")
    if sample_rate != SOURCE_RATE or audio.ndim != 1:
        raise ValueError(
            f"Expected mono {SOURCE_RATE} Hz source, got "
            f"shape={audio.shape}, rate={sample_rate}"
        )

    audio = np.concatenate(
        (
            audio,
            np.zeros(
                round(TRAILING_SILENCE_SECONDS * SOURCE_RATE),
                dtype=np.float32,
            ),
        )
    )
    source_frames = []
    for offset in range(0, len(audio), SOURCE_FRAME_SAMPLES):
        frame = audio[offset : offset + SOURCE_FRAME_SAMPLES]
        if len(frame) < SOURCE_FRAME_SAMPLES:
            frame = np.pad(frame, (0, SOURCE_FRAME_SAMPLES - len(frame)))
        pcm = (frame * 32768.0).astype("<i2")
        source_frames.append(pcm.astype(np.float32) / 32768.0)

    resampler = soxr.ResampleStream(SOURCE_RATE, INPUT_RATE, 1, dtype="float32")
    converted = [resampler.resample_chunk(frame) for frame in source_frames]
    converted.append(
        resampler.resample_chunk(np.array([], dtype=np.float32), last=True)
    )
    model_audio = np.concatenate(converted)
    expected_samples = len(source_frames) * FRAME_SAMPLES
    if len(model_audio) != expected_samples:
        raise RuntimeError(
            f"Expected {expected_samples} converted samples, got {len(model_audio)}"
        )
    model_pcm = (model_audio * 32768.0).astype("<i2")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(output_path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(INPUT_RATE)
        output.writeframes(model_pcm.tobytes())

    return {
        "source_path": str(source_path),
        "source_sha256": sha256(source_path),
        "output_path": str(output_path),
        "output_sha256": sha256(output_path),
        "trailing_silence_seconds": TRAILING_SILENCE_SECONDS,
        "frame_samples": FRAME_SAMPLES,
        "frames": len(source_frames),
        "samples": len(model_pcm),
    }


def load_frames(path: Path) -> list[bytes]:
    with wave.open(str(path), "rb") as source:
        metadata = (
            source.getnchannels(),
            source.getsampwidth(),
            source.getframerate(),
        )
        raw = source.readframes(source.getnframes())
    expected = (1, 2, INPUT_RATE)
    if metadata != expected:
        raise ValueError(f"Expected WAV metadata {expected}, got {metadata}")
    if len(raw) == 0 or len(raw) % FRAME_BYTES:
        raise ValueError(f"Input must contain complete {FRAME_SAMPLES}-sample frames")
    return [
        raw[offset : offset + FRAME_BYTES] for offset in range(0, len(raw), FRAME_BYTES)
    ]


def write_pcm16(path: Path, pcm: bytes, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(sample_rate)
        output.writeframes(pcm)


async def run_sglang_session(
    *, url: str, frames: list[bytes], prompt: str, drain_timeout: float
) -> tuple[dict[str, Any], bytes]:
    output = bytearray()
    events = []
    audio_arrival_ms = []
    committed = asyncio.Event()

    async with websockets.connect(
        url, max_size=8 * 1024 * 1024, ping_interval=20, ping_timeout=60
    ) as socket:
        created = json.loads(await socket.recv())
        session = created.get("session", {})
        expected = (
            "session.created",
            INPUT_RATE,
            SGLANG_OUTPUT_RATE,
            FRAME_SAMPLES,
        )
        actual = (
            created.get("type"),
            session.get("input_sample_rate"),
            session.get("output_sample_rate"),
            session.get("frame_samples"),
        )
        if actual != expected:
            raise RuntimeError(f"Unexpected realtime session: {created}")
        await socket.send(
            json.dumps(
                {
                    "type": "session.update",
                    "session": {"instructions": prompt},
                }
            )
        )
        updated = json.loads(await socket.recv())
        if updated.get("type") != "session.updated":
            raise RuntimeError(f"Expected session.updated, got {updated}")

        origin_ns = time.perf_counter_ns()

        async def receive_events() -> None:
            async for raw_event in socket:
                received_ns = time.perf_counter_ns()
                event = json.loads(raw_event)
                event_type = event.get("type")
                record = {
                    "type": event_type,
                    "arrival_ms": (received_ns - origin_ns) / 1e6,
                }
                if "frame_index" in event:
                    record["frame_index"] = event["frame_index"]
                if event_type == "response.audio.delta":
                    pcm = base64.b64decode(event["delta"], validate=True)
                    output.extend(pcm)
                    audio_arrival_ms.append(record["arrival_ms"])
                    record["samples"] = len(pcm) // 2
                elif event_type == "response.text.delta":
                    record["delta"] = event.get("delta", "")
                elif event_type == "input_audio_buffer.committed":
                    committed.set()
                elif event_type == "error":
                    raise RuntimeError(event.get("error", {}).get("message", event))
                events.append(record)

        receiver = asyncio.create_task(receive_events())
        frame_records = []
        try:
            for index, frame in enumerate(frames):
                deadline_ns = origin_ns + int(index * FRAME_SECONDS * 1e9)
                remaining_ns = deadline_ns - time.perf_counter_ns()
                if remaining_ns > 0:
                    await asyncio.sleep(remaining_ns / 1e9)
                send_start_ns = time.perf_counter_ns()
                await socket.send(
                    json.dumps(
                        {
                            "type": "input_audio_buffer.append",
                            "audio": base64.b64encode(frame).decode("ascii"),
                        }
                    )
                )
                sent_ns = time.perf_counter_ns()
                frame_records.append(
                    {
                        "index": index,
                        "send_lateness_ms": max(0, send_start_ns - deadline_ns) / 1e6,
                        "send_duration_ms": (sent_ns - send_start_ns) / 1e6,
                    }
                )
            await socket.send(json.dumps({"type": "input_audio_buffer.commit"}))
            await asyncio.wait_for(committed.wait(), timeout=drain_timeout)
            stream_done_ns = time.perf_counter_ns()
            await socket.send(json.dumps({"type": "session.close"}))
            close_sent_ns = time.perf_counter_ns()
            await asyncio.wait_for(receiver, timeout=drain_timeout)
        finally:
            if not receiver.done():
                receiver.cancel()
                await asyncio.gather(receiver, return_exceptions=True)

    intervals = [
        current - previous
        for previous, current in zip(audio_arrival_ms, audio_arrival_ms[1:])
    ]
    return (
        {
            "first_audio_ms": audio_arrival_ms[0] if audio_arrival_ms else None,
            "stream_total_ms": (stream_done_ns - origin_ns) / 1e6,
            "session_close_sent_ms": (close_sent_ns - origin_ns) / 1e6,
            "audio_event_count": len(audio_arrival_ms),
            "output_samples": len(output) // 2,
            "audio_interval_ms": summarize(intervals),
            "send_lateness_ms": summarize(
                [record["send_lateness_ms"] for record in frame_records]
            ),
            "text": "".join(
                event.get("delta", "")
                for event in events
                if event["type"] == "response.text.delta"
            ),
            "frames": frame_records,
            "events": events,
        },
        bytes(output),
    )


async def benchmark_sglang(args: argparse.Namespace) -> dict[str, Any]:
    frames = load_frames(args.input_wav)
    report = benchmark_metadata(
        args, "sglang-omni-websocket", frames, SGLANG_OUTPUT_RATE
    )
    for index in range(args.warmup_runs):
        record, _ = await run_sglang_session(
            url=args.url,
            frames=frames,
            prompt=args.prompt,
            drain_timeout=args.drain_timeout,
        )
        record["run"] = index + 1
        report["warmup"].append(record)
    for index in range(args.runs):
        record, pcm = await run_sglang_session(
            url=args.url,
            frames=frames,
            prompt=args.prompt,
            drain_timeout=args.drain_timeout,
        )
        record["run"] = index + 1
        report["runs"].append(record)
        write_pcm16(
            args.output_dir / f"run-{index + 1}.wav",
            pcm,
            SGLANG_OUTPUT_RATE,
        )
        print_run(record)
    return report


async def run_nim_session(
    *, url: str, frames: list[bytes], prompt: str, drain_timeout: float
) -> tuple[dict[str, Any], bytes]:
    output = bytearray()
    events = []
    audio_arrival_ms = []
    session_end = asyncio.Event()

    async with websockets.connect(
        url, max_size=8 * 1024 * 1024, ping_interval=20, ping_timeout=60
    ) as socket:
        created = json.loads(await socket.recv())
        if created.get("type") != "session.created":
            raise RuntimeError(f"Expected session.created, got {created}")
        await socket.send(
            json.dumps(
                {
                    "type": "session.update",
                    "session": {
                        "audio": {
                            "input": {
                                "format": {
                                    "type": "audio/pcm",
                                    "rate": INPUT_RATE,
                                }
                            },
                            "output": {"format": "pcm16"},
                        },
                        "instructions": prompt,
                        "tools": [],
                    },
                }
            )
        )
        updated = json.loads(await socket.recv())
        if updated.get("type") != "session.updated":
            raise RuntimeError(f"Expected session.updated, got {updated}")

        origin_ns = time.perf_counter_ns()

        async def receive_events() -> None:
            async for raw_event in socket:
                received_ns = time.perf_counter_ns()
                event = json.loads(raw_event)
                event_type = event.get("type")
                record = {
                    "type": event_type,
                    "arrival_ms": (received_ns - origin_ns) / 1e6,
                }
                if event_type == "response.output_audio.delta":
                    pcm = base64.b64decode(event["delta"], validate=True)
                    output.extend(pcm)
                    audio_arrival_ms.append(record["arrival_ms"])
                    record["samples"] = len(pcm) // 2
                elif event_type == "response.output_audio_transcript.delta":
                    record["delta"] = event.get("delta", "")
                elif event_type == "response.output_audio_transcript.done":
                    record["transcript"] = event.get("transcript", "")
                elif event_type == "session.end":
                    record["stats"] = event.get("stats", {})
                    session_end.set()
                elif event_type == "error":
                    raise RuntimeError(event.get("error", {}).get("message", event))
                events.append(record)

        receiver = asyncio.create_task(receive_events())
        frame_records = []
        try:
            for index, frame in enumerate(frames):
                deadline_ns = origin_ns + int(index * FRAME_SECONDS * 1e9)
                remaining_ns = deadline_ns - time.perf_counter_ns()
                if remaining_ns > 0:
                    await asyncio.sleep(remaining_ns / 1e9)
                send_start_ns = time.perf_counter_ns()
                await socket.send(
                    json.dumps(
                        {
                            "type": "input_audio_buffer.append",
                            "audio": base64.b64encode(frame).decode("ascii"),
                        }
                    )
                )
                sent_ns = time.perf_counter_ns()
                frame_records.append(
                    {
                        "index": index,
                        "send_lateness_ms": max(0, send_start_ns - deadline_ns) / 1e6,
                        "send_duration_ms": (sent_ns - send_start_ns) / 1e6,
                    }
                )
            await asyncio.sleep(2.0)
            await socket.send(json.dumps({"type": "session.close"}))
            close_sent_ns = time.perf_counter_ns()
            await asyncio.wait_for(session_end.wait(), timeout=drain_timeout)
            stream_done_ns = time.perf_counter_ns()
            receiver.cancel()
            await asyncio.gather(receiver, return_exceptions=True)
        finally:
            if not receiver.done():
                receiver.cancel()
                await asyncio.gather(receiver, return_exceptions=True)

    intervals = [
        current - previous
        for previous, current in zip(audio_arrival_ms, audio_arrival_ms[1:])
    ]
    transcripts = [
        event["transcript"]
        for event in events
        if event["type"] == "response.output_audio_transcript.done"
        and event.get("transcript")
    ]
    text = " ".join(transcripts) or "".join(
        event.get("delta", "")
        for event in events
        if event["type"] == "response.output_audio_transcript.delta"
    )
    return (
        {
            "first_audio_ms": audio_arrival_ms[0] if audio_arrival_ms else None,
            "stream_total_ms": (stream_done_ns - origin_ns) / 1e6,
            "session_close_sent_ms": (close_sent_ns - origin_ns) / 1e6,
            "audio_event_count": len(audio_arrival_ms),
            "output_samples": len(output) // 2,
            "audio_interval_ms": summarize(intervals),
            "send_lateness_ms": summarize(
                [record["send_lateness_ms"] for record in frame_records]
            ),
            "text": text,
            "frames": frame_records,
            "events": events,
        },
        bytes(output),
    )


async def benchmark_nim(args: argparse.Namespace) -> dict[str, Any]:
    frames = load_frames(args.input_wav)
    report = benchmark_metadata(args, "nvidia-nim-websocket", frames, NIM_OUTPUT_RATE)
    for index in range(args.warmup_runs):
        record, _ = await run_nim_session(
            url=args.url,
            frames=frames,
            prompt=args.prompt,
            drain_timeout=args.drain_timeout,
        )
        record["run"] = index + 1
        report["warmup"].append(record)
    for index in range(args.runs):
        record, pcm = await run_nim_session(
            url=args.url,
            frames=frames,
            prompt=args.prompt,
            drain_timeout=args.drain_timeout,
        )
        record["run"] = index + 1
        report["runs"].append(record)
        write_pcm16(
            args.output_dir / f"run-{index + 1}.wav",
            pcm,
            NIM_OUTPUT_RATE,
        )
        print_run(record)
    return report


def benchmark_metadata(
    args,
    engine: str,
    frames: list[bytes],
    output_sample_rate: int,
) -> dict[str, Any]:
    return {
        "engine": engine,
        "url": args.url,
        "input_wav": str(args.input_wav),
        "input_sha256": sha256(args.input_wav),
        "input_sample_rate": INPUT_RATE,
        "input_frame_samples": FRAME_SAMPLES,
        "input_frame_count": len(frames),
        "output_sample_rate": output_sample_rate,
        "prompt": args.prompt,
        "paced": True,
        "warmup_runs": args.warmup_runs,
        "warmup": [],
        "runs": [],
    }


def print_run(record: dict[str, Any]) -> None:
    print(
        f"run={record['run']} first_audio_ms={record['first_audio_ms']:.3f} "
        f"audio_events={record['audio_event_count']} text={record['text']!r}",
        flush=True,
    )


def add_benchmark_arguments(parser: argparse.ArgumentParser, default_url: str) -> None:
    parser.add_argument("--url", default=default_url)
    parser.add_argument("--input-wav", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--drain-timeout", type=float, default=300.0)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("source_wav", type=Path)
    prepare.add_argument("output_wav", type=Path)

    sglang = subparsers.add_parser("sglang")
    add_benchmark_arguments(sglang, "ws://127.0.0.1:18080/v1/realtime")

    nim = subparsers.add_parser("nim")
    add_benchmark_arguments(nim, "ws://127.0.0.1:9000/v1/realtime")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "prepare":
        print(json.dumps(prepare_input(args.source_wav, args.output_wav), indent=2))
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.command == "sglang":
        report = asyncio.run(benchmark_sglang(args))
    else:
        report = asyncio.run(benchmark_nim(args))
    report["summary"] = aggregate_report(report)
    output_json = args.output_dir / "raw.json"
    output_json.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["summary"], indent=2))
    print(f"wrote {output_json}")


if __name__ == "__main__":
    main()
