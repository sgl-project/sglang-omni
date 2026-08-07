# SPDX-License-Identifier: Apache-2.0
"""Benchmark single-request MOSS-TTS-Realtime serving."""

from __future__ import annotations

import argparse
import json
import statistics
import threading
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import requests

SAMPLE_RATE = 24000
BYTES_PER_SAMPLE = 2
DEFAULT_TEXT = (
    "Hello, this is a real-time speech synthesis benchmark running on one request."
)


@dataclass
class Sample:
    index: int
    warmup: bool
    ttfa_s: float
    latency_s: float
    audio_s: float
    rtf: float


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("cannot summarize an empty sample")
    position = (len(ordered) - 1) * percentile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def _sample(
    *,
    index: int,
    warmup: bool,
    started: float,
    first_audio: float | None,
    ended: float,
    audio_bytes: int,
) -> Sample:
    if first_audio is None or audio_bytes <= 0:
        raise RuntimeError("serving response contained no audio")
    audio_s = audio_bytes / (BYTES_PER_SAMPLE * SAMPLE_RATE)
    latency_s = ended - started
    return Sample(
        index=index,
        warmup=warmup,
        ttfa_s=first_audio - started,
        latency_s=latency_s,
        audio_s=audio_s,
        rtf=latency_s / audio_s,
    )


def _benchmark_omni(args: argparse.Namespace) -> list[Sample]:
    payload = {
        "model": "OpenMOSS-Team/MOSS-TTS-Realtime",
        "voice": "default",
        "input": args.text,
        "ref_audio": args.prompt_audio,
        "response_format": "pcm",
        "stream": True,
        "temperature": 0.8,
        "top_p": 0.6,
        "top_k": 30,
        "repetition_penalty": 1.1,
        "max_new_tokens": args.max_new_tokens,
    }
    samples = []
    with requests.Session() as client:
        for index in range(args.warmup + args.requests):
            started = time.perf_counter()
            first_audio = None
            audio_bytes = 0
            with client.post(
                f"{args.base_url}/v1/audio/speech",
                json=payload,
                stream=True,
                timeout=args.timeout,
            ) as response:
                response.raise_for_status()
                for chunk in response.iter_content(chunk_size=args.chunk_bytes):
                    if not chunk:
                        continue
                    if first_audio is None:
                        first_audio = time.perf_counter()
                    audio_bytes += len(chunk)
            samples.append(
                _sample(
                    index=index,
                    warmup=index < args.warmup,
                    started=started,
                    first_audio=first_audio,
                    ended=time.perf_counter(),
                    audio_bytes=audio_bytes,
                )
            )
    return samples


class _NativeSession:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.client = requests.Session()
        self.session_id = str(uuid.uuid4())

    def close(self) -> None:
        try:
            self.client.post(
                f"{self.args.base_url}/tts/session/close",
                json={"session_id": self.session_id},
                timeout=self.args.timeout,
            ).raise_for_status()
        finally:
            self.client.close()

    def run(self, index: int) -> Sample:
        started = time.perf_counter()
        response = self.client.post(
            f"{self.args.base_url}/tts/session/start",
            json={
                "session_id": self.session_id,
                "assistant_text": self.args.text,
                "user_text": None,
                "prompt_audio": self.args.prompt_audio,
                "user_audio": None,
                "new_turn": True,
            },
            timeout=self.args.timeout,
        )
        response.raise_for_status()
        result: dict[str, Any] = {
            "first_audio": None,
            "audio_bytes": 0,
            "error": None,
        }

        def read_audio() -> None:
            try:
                with requests.get(
                    f"{self.args.base_url}/tts/session/{self.session_id}/audio",
                    stream=True,
                    timeout=self.args.timeout,
                ) as audio_response:
                    audio_response.raise_for_status()
                    for chunk in audio_response.iter_content(
                        chunk_size=self.args.chunk_bytes
                    ):
                        if not chunk:
                            continue
                        if result["first_audio"] is None:
                            result["first_audio"] = time.perf_counter()
                        result["audio_bytes"] += len(chunk)
            except requests.RequestException as exc:
                result["error"] = exc

        reader = threading.Thread(target=read_audio, daemon=True)
        reader.start()
        pushed = self.client.post(
            f"{self.args.base_url}/tts/session/push",
            json={"session_id": self.session_id, "text": "", "is_final": True},
            timeout=self.args.timeout,
        )
        pushed.raise_for_status()
        reader.join(timeout=self.args.timeout)
        if reader.is_alive():
            raise TimeoutError("native audio stream did not finish")
        if result["error"] is not None:
            raise RuntimeError("native audio stream failed") from result["error"]
        return _sample(
            index=index,
            warmup=index < self.args.warmup,
            started=started,
            first_audio=result["first_audio"],
            ended=time.perf_counter(),
            audio_bytes=result["audio_bytes"],
        )


def _benchmark_native(args: argparse.Namespace) -> list[Sample]:
    session = _NativeSession(args)
    try:
        return [session.run(index) for index in range(args.warmup + args.requests)]
    finally:
        session.close()


def _summarize(backend: str, samples: list[Sample]) -> dict[str, Any]:
    measured = [sample for sample in samples if not sample.warmup]
    summary: dict[str, Any] = {
        "backend": backend,
        "concurrency": 1,
        "requests": len(measured),
        "warmup": len(samples) - len(measured),
        "samples": [asdict(sample) for sample in samples],
    }
    for field in ("ttfa_s", "latency_s", "audio_s", "rtf"):
        values = [float(getattr(sample, field)) for sample in measured]
        summary[f"{field}_mean"] = statistics.mean(values)
        summary[f"{field}_median"] = statistics.median(values)
        summary[f"{field}_p95"] = _percentile(values, 0.95)
    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("omni", "native"), required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--prompt-audio", required=True)
    parser.add_argument("--text", default=DEFAULT_TEXT)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--requests", type=int, default=5)
    parser.add_argument("--max-new-tokens", type=int, default=300)
    parser.add_argument("--chunk-bytes", type=int, default=4096)
    parser.add_argument("--timeout", type=float, default=900)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    args.base_url = args.base_url.rstrip("/")
    if args.warmup < 0 or args.requests < 1:
        parser.error("--warmup must be >= 0 and --requests must be >= 1")
    return args


def main() -> None:
    args = _parse_args()
    runner = _benchmark_omni if args.backend == "omni" else _benchmark_native
    samples = runner(args)
    for sample in samples:
        print(json.dumps(asdict(sample), sort_keys=True))
    summary = _summarize(args.backend, samples)
    output = json.dumps(summary, indent=2, sort_keys=True)
    print(output)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
