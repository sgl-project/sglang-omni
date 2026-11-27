# SPDX-License-Identifier: Apache-2.0
"""Evaluate native audio tool calls on the five ServiceNow BFCL v3 audio subsets."""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import aiohttp
from pydantic import BaseModel, Json, JsonValue

from benchmarks.benchmarker.data import RequestResult
from benchmarks.benchmarker.runner import BenchmarkRunner, RunConfig
from benchmarks.benchmarker.utils import save_json_results
from benchmarks.dataset.bfcl_audio import (
    DATASET_ID,
    DATASET_REVISION,
    SUBSETS,
    AudioSample,
    load_samples,
)
from benchmarks.metrics.bfcl_audio import matches_calls


class ReturnedFunction(BaseModel):
    name: str
    arguments: Json[dict[str, JsonValue]]


class ReturnedCall(BaseModel):
    type: Literal["function"]
    function: ReturnedFunction


class AssistantMessage(BaseModel):
    role: Literal["assistant"]
    content: str | None = None
    tool_calls: list[ReturnedCall] | None = None


@dataclass(kw_only=True)
class CallResult(RequestResult):
    message: dict[str, JsonValue] | None = None
    completion_reason: str | None = None
    correct: bool = False


async def benchmark(options: argparse.Namespace) -> None:
    endpoint = f"{options.base_url.rstrip('/')}/v1/chat/completions"
    runner = BenchmarkRunner(RunConfig(max_concurrency=options.concurrency, warmup=0))

    async def send(session: aiohttp.ClientSession, sample: AudioSample) -> CallResult:
        result = CallResult(request_id=sample.sample_id)
        audio = base64.b64encode(sample.audio_bytes).decode("ascii")
        content = [
            {"type": "input_audio", "input_audio": {"data": audio, "format": "wav"}}
        ]
        payload = {
            "model": options.model,
            "messages": [{"role": "user", "content": content}],
            "tools": [
                {"type": "function", "function": function}
                for function in sample.functions
            ],
            "tool_choice": "auto",
            "modalities": ["text"],
            "temperature": 0,
            "max_tokens": options.max_tokens,
            "stream": False,
        }
        try:
            async with session.post(endpoint, json=payload) as response:
                response.raise_for_status()
                completion = await response.json()
            choice = completion["choices"][0]
            result.message = choice["message"]
            result.completion_reason = choice["finish_reason"]
            message = AssistantMessage.model_validate(result.message)
            calls = [
                (call.function.name, call.function.arguments)
                for call in message.tool_calls or []
            ]
            required = {
                function["name"]: set(function["parameters"].get("required", []))
                for function in sample.functions
            }
            if calls:
                result.correct = (
                    result.completion_reason == "tool_calls"
                    and matches_calls(calls, sample.references, required)
                )
            else:
                result.correct = (
                    result.completion_reason == "stop"
                    and bool(message.content and message.content.strip())
                    and not sample.references
                )
            result.is_success = True
        except (
            aiohttp.ClientError,
            TimeoutError,
            ValueError,
            KeyError,
            IndexError,
            TypeError,
        ) as error:
            result.error = f"{type(error).__name__}: {error}"
        return result

    records = []
    by_subset = {}
    for subset in options.subsets:
        samples = load_samples(subset, options.max_samples_per_subset)
        results = await runner.run(samples, send)
        subset_records = []
        for sample, result in zip(samples, results, strict=True):
            assert isinstance(result, CallResult)
            subset_records.append(
                {
                    "id": sample.sample_id,
                    "subset": subset,
                    "reference": sample.references,
                    "message": result.message,
                    "finish_reason": result.completion_reason,
                    "correct": result.correct,
                    "error": result.error,
                }
            )
        by_subset[subset] = summarize(subset_records)
        records.extend(subset_records)
    summary = {"overall": summarize(records), "by_subset": by_subset}
    save_json_results(
        {
            "dataset": DATASET_ID,
            "revision": DATASET_REVISION,
            "model": options.model,
            "metric": "structured_call_accuracy",
            "max_tokens": options.max_tokens,
            "concurrency": options.concurrency,
            **summary,
            "results": records,
        },
        str(options.output_dir),
        "bfcl_audio.json",
    )
    print(json.dumps(summary, indent=2))
    if all(record["error"] for record in records):
        raise SystemExit(
            f"All requests failed. See {options.output_dir}/bfcl_audio.json"
        )
    else:
        pass


def summarize(records: list[dict[str, JsonValue]]) -> dict[str, int | float]:
    correct = sum(record["correct"] is True for record in records)
    return {
        "samples": len(records),
        "correct": correct,
        "errors": sum(bool(record["error"]) for record in records),
        "accuracy": correct / len(records),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-url", required=True, help="Server root URL, e.g. http://localhost:8000."
    )
    parser.add_argument("--model", required=True, help="Served model name.")
    parser.add_argument("--subsets", nargs="+", choices=SUBSETS, default=SUBSETS)
    parser.add_argument(
        "--max-samples-per-subset",
        type=int,
        help="Evaluate the first N examples of each selected subset.",
    )
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--output-dir", type=Path, required=True)
    options = parser.parse_args()
    if min(options.concurrency, options.max_tokens) < 1 or (
        options.max_samples_per_subset is not None
        and options.max_samples_per_subset < 1
    ):
        parser.error("Sample limit, concurrency, and token budget must be positive.")
    elif len(set(options.subsets)) != len(options.subsets):
        parser.error("Select each subset once.")
    else:
        asyncio.run(benchmark(options))


if __name__ == "__main__":
    main()
