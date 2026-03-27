"""
TTS benchmarker — measures speed and optional accuracy for text-to-speech.

Supports two modes:
  1. Voice cloning (default) — reference audio + transcript provided per sample.
  2. Plain TTS (--no-ref-audio) — no reference audio, default voice.
"""

from __future__ import annotations

import base64
import os
import time

import aiohttp
from benchmark_kit.utils import get_wav_duration, iter_sse_lines, parse_sse_event
from datasets import load_dataset

from .benchmarker import Benchmarker, BenchmarkRequest, BenchmarkResult

META_FIELD_COUNT = 4


class TTSBenchmarker(Benchmarker):
    """Benchmarker for TTS models via /v1/audio/speech."""

    def __init__(
        self,
        dataset: str,
        no_ref_audio: bool = False,
        max_samples: int | None = None,
        max_new_tokens: int = 2048,
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        repetition_penalty: float | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.dataset = dataset
        self.no_ref_audio = no_ref_audio
        self.max_samples = max_samples
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.repetition_penalty = repetition_penalty

    @property
    def task_type(self) -> str:
        return "tts"

    def load_dataset(self) -> list[dict]:
        dataset = load_dataset("json", data_files=self.dataset)["train"]
        dataset_dir = os.path.dirname(self.dataset)
        samples: list[dict] = []
        for idx, sample in enumerate(dataset):
            prompt_audio = os.path.join(dataset_dir, sample["prompt_audio"])
            target_audio = os.path.join(dataset_dir, sample["target_audio"])
            samples.append(
                {
                    "id": idx,
                    "prompt_text": sample["prompt_text"],
                    "prompt_audio": prompt_audio,
                    "target_text": sample["target_text"],
                    "target_audio": target_audio,
                }
            )
            if self.max_samples and len(samples) >= self.max_samples:
                break
        return samples

    def build_request(self, sample: dict) -> BenchmarkRequest:
        api_url = f"{self.base_url}/v1/audio/speech"
        payload: dict = {
            "model": "",
            "input": sample["prompt_text"],
            "response_format": "wav",
        }

        if not self.no_ref_audio:
            if sample.get("prompt_audio"):
                payload["ref_audio"] = sample["prompt_audio"]
            if sample.get("prompt_text"):
                payload["ref_text"] = sample["prompt_text"]

        for key, value in [
            ("max_new_tokens", self.max_new_tokens),
            ("temperature", self.temperature),
            ("top_p", self.top_p),
            ("top_k", self.top_k),
            ("repetition_penalty", self.repetition_penalty),
        ]:
            if value is not None:
                payload[key] = value

        if self.stream:
            payload["stream"] = True

        return BenchmarkRequest(
            request_id=sample["id"],
            task_type="tts",
            payload=payload,
            api_url=api_url,
            stream=self.stream,
            expected_text=sample.get("target_text"),
        )

    def get_config(self) -> dict:
        config = super().get_config()
        config.update(
            {
                "dataset": self.dataset,
                "no_ref_audio": self.no_ref_audio,
                "max_samples": self.max_samples,
                "max_new_tokens": self.max_new_tokens,
            }
        )
        return config

    async def handle_non_streaming_response(
        self,
        response: aiohttp.ClientResponse,
        result: BenchmarkResult,
        start_time: float,
    ) -> None:
        """Read full TTS audio response, compute metrics."""
        audio_bytes = await response.read()
        result.generated_audio_bytes = audio_bytes
        result.audio_duration_s = get_wav_duration(audio_bytes)
        elapsed = time.perf_counter() - start_time
        # For non-streaming, TTFT equals the full latency (first token = complete response)
        result.ttft = elapsed
        if result.audio_duration_s > 0:
            result.is_success = True
            result.rtf = elapsed / result.audio_duration_s
        else:
            result.error = f"Empty or invalid audio response ({len(audio_bytes)} bytes)"
            return
        self._apply_headers(result, response.headers)

    async def handle_streaming_response(
        self,
        response: aiohttp.ClientResponse,
        result: BenchmarkResult,
        start_time: float,
    ) -> None:
        """Parse SSE stream for TTS: decode audio chunks, compute metrics."""
        total_audio_duration = 0.0
        usage_data: dict | None = None
        first_chunk_time: float | None = None

        async for line in iter_sse_lines(response):
            event = parse_sse_event(line)
            if event is None:
                continue
            if first_chunk_time is None:
                first_chunk_time = time.perf_counter()
            audio = event.get("audio")
            if isinstance(audio, dict) and audio.get("data"):
                total_audio_duration += get_wav_duration(
                    base64.b64decode(audio["data"])
                )
            event_usage = event.get("usage")
            if isinstance(event_usage, dict):
                usage_data = event_usage

        result.audio_duration_s = total_audio_duration
        if first_chunk_time is not None:
            result.ttft = first_chunk_time - start_time
        if total_audio_duration > 0:
            elapsed = time.perf_counter() - start_time
            result.rtf = elapsed / total_audio_duration
        result.is_success = total_audio_duration > 0

        if usage_data:
            self._apply_usage(result, usage_data)
