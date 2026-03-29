# SPDX-License-Identifier: Apache-2.0
"""TTS Speed case -- /v1/audio/speech request handling (streaming + non-streaming)."""

from __future__ import annotations

import asyncio
import csv
import json
import logging
import os
import time

import aiohttp

from benchmarks.benchmarker.data import RequestResult
from benchmarks.benchmarker.runner import SendFn
from benchmarks.benchmarker.utils import get_wav_duration, process_sse_line
from benchmarks.dataset.seedtts import SampleInput

logger = logging.getLogger(__name__)

TEXT_PREVIEW_LENGTH = 60
SUMMARY_LABEL_WIDTH = 30
SUMMARY_LINE_WIDTH = 60


def _build_tts_payload(
    sample: SampleInput,
    model_name: str,
    *,
    stream: bool = False,
    no_ref_audio: bool = False,
    **gen_kwargs,
) -> dict:
    payload: dict = {
        "model": model_name,
        "input": sample.target_text,
        "response_format": "wav",
    }
    if not no_ref_audio:
        payload["ref_audio"] = sample.ref_audio
        payload["ref_text"] = sample.ref_text
    for key, value in gen_kwargs.items():
        if value is not None:
            payload[key] = value
    if stream:
        payload["stream"] = True
    return payload


def _parse_response_headers(result: RequestResult, headers: dict) -> None:
    prompt_tok = headers.get("X-Prompt-Tokens")
    comp_tok = headers.get("X-Completion-Tokens")
    eng_time = headers.get("X-Engine-Time")
    if prompt_tok is not None:
        result.prompt_tokens = int(prompt_tok)
    if comp_tok is not None:
        result.completion_tokens = int(comp_tok)
    if eng_time is not None:
        result.engine_time_s = float(eng_time)
    if result.completion_tokens > 0 and result.engine_time_s > 0:
        result.tok_per_s = result.completion_tokens / result.engine_time_s


async def _handle_streaming_response(
    response: aiohttp.ClientResponse,
    result: RequestResult,
    start_time: float,
) -> None:
    total_audio_duration = 0.0
    usage_data: dict | None = None
    buffer = bytearray()
    async for chunk in response.content.iter_any():
        buffer.extend(chunk)
        while b"\n" in buffer:
            idx = buffer.index(b"\n")
            raw_line = bytes(buffer[:idx])
            del buffer[: idx + 1]
            line = raw_line.decode("utf-8", errors="replace").strip()
            total_audio_duration, usage_data = process_sse_line(
                line, total_audio_duration, usage_data
            )
    if buffer.strip():
        line = bytes(buffer).decode("utf-8", errors="replace").strip()
        total_audio_duration, usage_data = process_sse_line(
            line, total_audio_duration, usage_data
        )
    result.audio_duration_s = total_audio_duration
    if total_audio_duration > 0:
        elapsed = time.perf_counter() - start_time
        result.rtf = elapsed / total_audio_duration
    result.is_success = total_audio_duration > 0
    if usage_data:
        prompt_tok = usage_data.get("prompt_tokens")
        comp_tok = usage_data.get("completion_tokens")
        eng_time = usage_data.get("engine_time_s")
        if prompt_tok is not None:
            result.prompt_tokens = int(prompt_tok)
        if comp_tok is not None:
            result.completion_tokens = int(comp_tok)
        if eng_time is not None:
            result.engine_time_s = float(eng_time)
        if result.completion_tokens > 0 and result.engine_time_s > 0:
            result.tok_per_s = result.completion_tokens / result.engine_time_s


async def _handle_non_streaming_response(
    response: aiohttp.ClientResponse,
    result: RequestResult,
    start_time: float,
    save_audio_dir: str | None,
) -> None:
    audio_bytes = await response.read()
    result.audio_duration_s = get_wav_duration(audio_bytes)
    elapsed = time.perf_counter() - start_time
    if result.audio_duration_s > 0:
        result.is_success = True
        result.rtf = elapsed / result.audio_duration_s
    else:
        result.error = f"Empty or invalid audio response ({len(audio_bytes)} bytes)"
        return
    _parse_response_headers(result, response.headers)
    if save_audio_dir and audio_bytes:
        audio_path = os.path.join(save_audio_dir, f"{result.request_id}.wav")
        with open(audio_path, "wb") as f:
            f.write(audio_bytes)
        result.wav_path = audio_path


def make_tts_send_fn(
    model_name: str,
    api_url: str,
    *,
    stream: bool = False,
    no_ref_audio: bool = False,
    save_audio_dir: str | None = None,
    **gen_kwargs,
) -> SendFn:
    """Return a *send_fn(session, sample) -> RequestResult* for the runner."""

    async def send_fn(
        session: aiohttp.ClientSession, sample: SampleInput
    ) -> RequestResult:
        result = RequestResult(
            request_id=sample.sample_id,
            text=sample.target_text[:TEXT_PREVIEW_LENGTH],
        )
        payload = _build_tts_payload(
            sample, model_name, stream=stream, no_ref_audio=no_ref_audio, **gen_kwargs
        )

        start_time = time.perf_counter()
        try:
            async with session.post(api_url, json=payload) as response:
                if response.status != 200:
                    result.error = f"HTTP {response.status}: {await response.text()}"
                elif stream:
                    await _handle_streaming_response(response, result, start_time)
                else:
                    await _handle_non_streaming_response(
                        response, result, start_time, save_audio_dir
                    )
        except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
            result.error = str(exc)
        finally:
            result.latency_s = time.perf_counter() - start_time

        return result

    return send_fn


def print_speed_summary(metrics: dict, model_name: str) -> None:
    lw = SUMMARY_LABEL_WIDTH
    w = SUMMARY_LINE_WIDTH
    print(f"\n{'=' * w}")
    print(f"{'TTS Benchmark Result':^{w}}")
    print(f"{'=' * w}")
    print(f"  {'Model:':<{lw}} {model_name}")
    print(f"  {'Completed requests:':<{lw}} {metrics['completed_requests']}")
    print(f"  {'Failed requests:':<{lw}} {metrics['failed_requests']}")
    print(f"{'-' * w}")
    print(f"  {'Latency mean (s):':<{lw}} {metrics.get('latency_mean_s', 'N/A')}")
    print(f"  {'Latency median (s):':<{lw}} {metrics.get('latency_median_s', 'N/A')}")
    print(f"  {'Latency p95 (s):':<{lw}} {metrics.get('latency_p95_s', 'N/A')}")
    print(f"  {'Latency p99 (s):':<{lw}} {metrics.get('latency_p99_s', 'N/A')}")
    if metrics.get("rtf_mean") is not None:
        print(f"  {'RTF mean:':<{lw}} {metrics['rtf_mean']}")
        print(f"  {'RTF median:':<{lw}} {metrics['rtf_median']}")
    if metrics.get("audio_duration_mean_s"):
        print(
            f"  {'Audio duration mean (s):':<{lw}} {metrics['audio_duration_mean_s']}"
        )
    if metrics.get("tok_per_s_mean") is not None:
        print(f"  {'Tok/s (per-req mean):':<{lw}} {metrics['tok_per_s_mean']}")
        print(f"  {'Tok/s (per-req median):':<{lw}} {metrics['tok_per_s_median']}")
    if metrics.get("tok_per_s_agg") is not None:
        print(f"  {'Tok/s (aggregate):':<{lw}} {metrics['tok_per_s_agg']}")
    if metrics.get("gen_tokens_mean") is not None:
        print(f"  {'Gen tokens (mean):':<{lw}} {metrics['gen_tokens_mean']:.0f}")
        print(f"  {'Gen tokens (total):':<{lw}} {metrics['gen_tokens_total']}")
    if metrics.get("prompt_tokens_mean") is not None:
        print(f"  {'Prompt tokens (mean):':<{lw}} {metrics['prompt_tokens_mean']:.0f}")
        print(f"  {'Prompt tokens (total):':<{lw}} {metrics['prompt_tokens_total']}")
    print(f"  {'Throughput (req/s):':<{lw}} {metrics.get('throughput_qps', 'N/A')}")
    print(f"{'=' * w}")


def save_speed_results(
    outputs: list[RequestResult],
    metrics: dict,
    config: dict,
    output_dir: str,
) -> None:
    os.makedirs(output_dir, exist_ok=True)

    json_results = {
        "summary": metrics,
        "config": config,
        "per_request": [
            {
                "id": o.request_id,
                "text": o.text,
                "is_success": o.is_success,
                "latency_s": round(o.latency_s, 4),
                "audio_duration_s": round(o.audio_duration_s, 4),
                "rtf": round(o.rtf, 4) if o.rtf < float("inf") else None,
                "prompt_tokens": o.prompt_tokens or None,
                "completion_tokens": o.completion_tokens or None,
                "tok_per_s": round(o.tok_per_s, 1) if o.tok_per_s > 0 else None,
                "error": o.error or None,
            }
            for o in outputs
        ],
    }
    json_path = os.path.join(output_dir, "speed_results.json")
    with open(json_path, "w") as f:
        json.dump(json_results, f, indent=2, ensure_ascii=False)
    logger.info("Results saved to %s", json_path)

    csv_path = os.path.join(output_dir, "results.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "id",
                "text",
                "latency_s",
                "audio_duration_s",
                "rtf",
                "prompt_tokens",
                "completion_tokens",
                "tok_per_s",
                "is_success",
                "error",
            ]
        )
        for o in outputs:
            writer.writerow(
                [
                    o.request_id,
                    o.text,
                    f"{o.latency_s:.4f}",
                    f"{o.audio_duration_s:.4f}",
                    f"{o.rtf:.4f}" if o.rtf < float("inf") else "",
                    o.prompt_tokens or "",
                    o.completion_tokens or "",
                    f"{o.tok_per_s:.1f}" if o.tok_per_s > 0 else "",
                    o.is_success,
                    o.error or "",
                ]
            )
    logger.info("CSV saved to %s", csv_path)
