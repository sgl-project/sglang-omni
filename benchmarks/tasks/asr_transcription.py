# SPDX-License-Identifier: Apache-2.0
"""Shared Qwen3-ASR transcription client helpers for benchmarks."""

from __future__ import annotations

import asyncio
import os
import time
from typing import Protocol

import aiohttp

from benchmarks.benchmarker.data import RequestResult
from benchmarks.benchmarker.runner import BenchmarkRunner, RunConfig, SendFn
from benchmarks.benchmarker.utils import get_wav_duration

QWEN3_ASR_MODEL_PATH = os.getenv("QWEN3_ASR_MODEL_PATH", "Qwen/Qwen3-ASR-1.7B")
QWEN3_ASR_REQUEST_TIMEOUT_S = 300
QWEN3_ASR_MAX_NEW_TOKENS = int(os.getenv("QWEN3_ASR_MAX_NEW_TOKENS", "128"))
# ASR transcription fan-out for WER, not TTS generation concurrency.
DEFAULT_ASR_TRANSCRIBE_CONCURRENCY = int(
    os.getenv("QWEN3_ASR_CONCURRENCY", os.getenv("SEEDTTS_ASR_CONCURRENCY", "32"))
)


class ASRTranscriptionSample(Protocol):
    sample_id: str
    ref_audio: str


def make_asr_send_fn(
    model_name: str,
    api_url: str,
    *,
    lang: str = "en",
    max_new_tokens: int = QWEN3_ASR_MAX_NEW_TOKENS,
) -> SendFn:
    """Return a send function for Omni ``/v1/audio/transcriptions``.

    Note: do NOT send temperature=0 — Qwen3-ASR degenerates under pure greedy
    (the server bumps it to 0.01). ``language`` selects the forced prefix.
    """

    async def send_fn(
        session: aiohttp.ClientSession, sample: ASRTranscriptionSample
    ) -> RequestResult:
        result = RequestResult(request_id=sample.sample_id)
        try:
            with open(sample.ref_audio, "rb") as audio_file:
                audio_bytes = audio_file.read()
        except OSError as exc:
            result.error = str(exc)
            return result
        result.audio_duration_s = get_wav_duration(audio_bytes)

        form = aiohttp.FormData()
        form.add_field("model", model_name)
        form.add_field("language", lang)
        form.add_field("response_format", "json")
        form.add_field("max_new_tokens", str(max_new_tokens))
        form.add_field(
            "file",
            audio_bytes,
            filename=os.path.basename(sample.ref_audio),
            content_type="audio/wav",
        )

        start_time = time.perf_counter()
        try:
            async with session.post(api_url, data=form) as response:
                if response.status != 200:
                    result.error = f"HTTP {response.status}: {await response.text()}"
                else:
                    payload = await response.json()
                    result.text = str(payload.get("text", ""))
                    result.is_success = True
        except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
            result.error = str(exc)
        finally:
            result.latency_s = time.perf_counter() - start_time
        if result.is_success and result.audio_duration_s > 0:
            result.rtf = result.latency_s / result.audio_duration_s
        return result

    return send_fn


async def run_asr_transcription(
    samples: list[ASRTranscriptionSample],
    *,
    host: str = "127.0.0.1",
    port: int,
    model_path: str = QWEN3_ASR_MODEL_PATH,
    lang: str = "en",
    concurrency: int = DEFAULT_ASR_TRANSCRIBE_CONCURRENCY,
    warmup: int = 0,
    request_timeout_s: int = QWEN3_ASR_REQUEST_TIMEOUT_S,
    disable_tqdm: bool = True,
) -> tuple[list[RequestResult], float]:
    """Transcribe ``samples`` against a running ASR router at one concurrency."""
    api_url = f"http://{host}:{port}/v1/audio/transcriptions"
    send_fn = make_asr_send_fn(model_path, api_url, lang=lang)
    runner = BenchmarkRunner(
        RunConfig(
            max_concurrency=concurrency,
            warmup=warmup,
            disable_tqdm=disable_tqdm,
            timeout_s=request_timeout_s,
        )
    )
    outputs = await runner.run(samples, send_fn)
    return outputs, runner.wall_clock_s
