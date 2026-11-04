# SPDX-License-Identifier: Apache-2.0
"""Measure interleaved greedy and sampled ASR requests against one endpoint."""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import httpx
from jiwer import wer

REQUEST_MIX_PERIOD = 100
REQUEST_MIX_STRIDE = 37


@dataclass(kw_only=True)
class RequestMeasurement:
    sampled: bool
    first_token_seconds: float
    latency_seconds: float
    word_error_rate: float


@dataclass(kw_only=True)
class SweepMeasurement:
    concurrency: int
    fallback_percentage: int
    requests_per_second: float
    sampled_requests: int
    word_error_rate: float
    requests: list[RequestMeasurement]


async def run_sweep(arguments: argparse.Namespace) -> None:
    audio_bytes = arguments.audio.read_bytes()
    reference = re.sub(r"[^\w\s]", "", arguments.reference_text.casefold())
    measurements: list[SweepMeasurement] = []
    async with httpx.AsyncClient(
        timeout=180, limits=httpx.Limits(max_connections=max(arguments.concurrencies))
    ) as client:

        async def transcribe(
            sampled: bool, semaphore: asyncio.Semaphore
        ) -> RequestMeasurement:
            async with semaphore:
                started = time.perf_counter()
                first_token_seconds = None
                transcript = ""
                completed = False
                async with client.stream(
                    "POST",
                    f"{arguments.base_url.rstrip('/')}/v1/audio/transcriptions",
                    data={
                        "model": arguments.model,
                        "temperature": str(0.7 if sampled else 0.0),
                        "stream": "true",
                    },
                    files={"file": (arguments.audio.name, audio_bytes, "audio/wav")},
                ) as response:
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        if line.startswith("data: ") and line != "data: [DONE]":
                            event = json.loads(line[6:])
                            if (
                                event.get("type") == "error"
                                or event.get("error") is not None
                            ):
                                raise RuntimeError(
                                    f"Transcription stream failed: {event}"
                                )
                            elif event.get("type") == "transcript.text.delta":
                                if first_token_seconds is None:
                                    first_token_seconds = time.perf_counter() - started
                                else:
                                    pass
                                transcript += event["delta"]
                            elif event.get("type") == "transcript.text.done":
                                transcript = event["text"]
                                completed = True
                            else:
                                pass
                        else:
                            pass
                if first_token_seconds is None or not completed:
                    raise RuntimeError(f"Incomplete transcription stream: {transcript}")
                else:
                    return RequestMeasurement(
                        sampled=sampled,
                        first_token_seconds=first_token_seconds,
                        latency_seconds=time.perf_counter() - started,
                        word_error_rate=wer(
                            reference, re.sub(r"[^\w\s]", "", transcript.casefold())
                        ),
                    )

        await transcribe(False, asyncio.Semaphore(1))
        if any(arguments.fallback_percentages):
            await transcribe(True, asyncio.Semaphore(1))
        else:
            pass
        for concurrency in arguments.concurrencies:
            for fallback_percentage in arguments.fallback_percentages:
                semaphore = asyncio.Semaphore(concurrency)
                started = time.perf_counter()
                requests = await asyncio.gather(
                    *(
                        transcribe(
                            index * REQUEST_MIX_STRIDE % REQUEST_MIX_PERIOD
                            < fallback_percentage,
                            semaphore,
                        )
                        for index in range(arguments.requests)
                    )
                )
                elapsed_seconds = time.perf_counter() - started
                row = SweepMeasurement(
                    concurrency=concurrency,
                    fallback_percentage=fallback_percentage,
                    requests_per_second=len(requests) / elapsed_seconds,
                    sampled_requests=sum(request.sampled for request in requests),
                    word_error_rate=sum(request.word_error_rate for request in requests)
                    / len(requests),
                    requests=requests,
                )
                measurements.append(row)
                arguments.output.write_text(
                    json.dumps(
                        [asdict(measurement) for measurement in measurements], indent=2
                    )
                    + "\n"
                )
                print(
                    json.dumps(
                        {
                            name: value
                            for name, value in asdict(row).items()
                            if name != "requests"
                        }
                    ),
                    flush=True,
                )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--reference-text", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--requests", type=int, default=100)
    parser.add_argument("--concurrencies", type=int, nargs="+", default=[1, 16, 32, 64])
    parser.add_argument(
        "--fallback-percentages", type=int, nargs="+", default=[0, 5, 100]
    )
    arguments = parser.parse_args()
    if arguments.requests <= 0 or any(value <= 0 for value in arguments.concurrencies):
        parser.error("requests and concurrency must be positive")
    elif any(value < 0 or value > 100 for value in arguments.fallback_percentages):
        parser.error("fallback percentages must be between 0 and 100")
    else:
        asyncio.run(run_sweep(arguments))


if __name__ == "__main__":
    main()
