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
REQUEST_TIMEOUT_SECONDS = 180
FALLBACK_TEMPERATURE = 0.7


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


async def run_sweep(
    *,
    audio_path: Path,
    reference_text: str,
    base_url: str,
    model_identifier: str,
    concurrency_counts: list[int],
    fallback_percentages: list[int],
    request_count: int,
    output_path: Path,
) -> None:
    audio_bytes = audio_path.read_bytes()
    normalized_reference_text = re.sub(r"[^\w\s]", "", reference_text.casefold())
    measurements: list[SweepMeasurement] = []
    async with httpx.AsyncClient(
        timeout=REQUEST_TIMEOUT_SECONDS,
        limits=httpx.Limits(max_connections=max(concurrency_counts)),
    ) as transcription_client:

        async def transcribe(
            sampled: bool, semaphore: asyncio.Semaphore
        ) -> RequestMeasurement:
            async with semaphore:
                started_at_seconds = time.perf_counter()
                first_token_seconds = None
                transcript = ""
                has_completed_transcript = False
                async with transcription_client.stream(
                    "POST",
                    f"{base_url.rstrip('/')}/v1/audio/transcriptions",
                    data={
                        "model": model_identifier,
                        "temperature": str(FALLBACK_TEMPERATURE if sampled else 0.0),
                        "stream": "true",
                    },
                    files={"file": (audio_path.name, audio_bytes, "audio/wav")},
                ) as response:
                    response.raise_for_status()
                    async for stream_line in response.aiter_lines():
                        if (
                            stream_line.startswith("data: ")
                            and stream_line != "data: [DONE]"
                        ):
                            transcript_event = json.loads(stream_line[6:])
                            if (
                                transcript_event.get("type") == "error"
                                or transcript_event.get("error") is not None
                            ):
                                raise RuntimeError(
                                    f"Transcription stream failed: {transcript_event}"
                                )
                            elif (
                                transcript_event.get("type") == "transcript.text.delta"
                            ):
                                if first_token_seconds is None:
                                    first_token_seconds = (
                                        time.perf_counter() - started_at_seconds
                                    )
                                else:
                                    pass
                                transcript += transcript_event["delta"]
                            elif transcript_event.get("type") == "transcript.text.done":
                                transcript = transcript_event["text"]
                                has_completed_transcript = True
                            else:
                                pass
                        else:
                            pass
                if first_token_seconds is None or not has_completed_transcript:
                    raise RuntimeError(f"Incomplete transcription stream: {transcript}")
                else:
                    return RequestMeasurement(
                        sampled=sampled,
                        first_token_seconds=first_token_seconds,
                        latency_seconds=time.perf_counter() - started_at_seconds,
                        word_error_rate=wer(
                            normalized_reference_text,
                            re.sub(r"[^\w\s]", "", transcript.casefold()),
                        ),
                    )

        await transcribe(False, asyncio.Semaphore(1))
        if any(fallback_percentages):
            await transcribe(True, asyncio.Semaphore(1))
        else:
            pass
        for concurrent_request_count in concurrency_counts:
            for fallback_percentage in fallback_percentages:
                semaphore = asyncio.Semaphore(concurrent_request_count)
                started_at_seconds = time.perf_counter()
                request_measurements = await asyncio.gather(
                    *(
                        transcribe(
                            request_index * REQUEST_MIX_STRIDE % REQUEST_MIX_PERIOD
                            < fallback_percentage,
                            semaphore,
                        )
                        for request_index in range(request_count)
                    )
                )
                elapsed_seconds = time.perf_counter() - started_at_seconds
                sweep_measurement = SweepMeasurement(
                    concurrency=concurrent_request_count,
                    fallback_percentage=fallback_percentage,
                    requests_per_second=len(request_measurements) / elapsed_seconds,
                    sampled_requests=sum(
                        request.sampled for request in request_measurements
                    ),
                    word_error_rate=sum(
                        request.word_error_rate for request in request_measurements
                    )
                    / len(request_measurements),
                    requests=request_measurements,
                )
                measurements.append(sweep_measurement)
                output_path.write_text(
                    json.dumps(
                        [asdict(measurement) for measurement in measurements], indent=2
                    )
                    + "\n"
                )
                print(
                    json.dumps(
                        {
                            measurement_field: measurement_value
                            for measurement_field, measurement_value in asdict(
                                sweep_measurement
                            ).items()
                            if measurement_field != "requests"
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
        asyncio.run(
            run_sweep(
                audio_path=arguments.audio,
                reference_text=arguments.reference_text,
                base_url=arguments.base_url,
                model_identifier=arguments.model,
                concurrency_counts=arguments.concurrencies,
                fallback_percentages=arguments.fallback_percentages,
                request_count=arguments.requests,
                output_path=arguments.output,
            )
        )


if __name__ == "__main__":
    main()
