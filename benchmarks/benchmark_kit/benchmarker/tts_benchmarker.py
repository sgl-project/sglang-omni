"""
TTS benchmarker — measures speed and optional accuracy for text-to-speech.

Supports two modes:
  1. Voice cloning (default) — reference audio + transcript provided per sample.
  2. Plain TTS (--no-ref-audio) — no reference audio, default voice.

Usage:

    python -m benchmarks.benchmarker.tts_benchmarker \
        --model fishaudio/s2-pro --port 8000 \
        --testset seedtts_testset/en/meta.lst \
        --max-samples 10

    # streaming mode
    python -m benchmarks.benchmarker.tts_benchmarker \
        --model fishaudio/s2-pro --port 8000 \
        --testset seedtts_testset/en/meta.lst \
        --max-samples 10 --stream

    # plain TTS (no voice cloning)
    python -m benchmarks.benchmarker.tts_benchmarker \
        --model fishaudio/s2-pro --port 8000 \
        --testset seedtts_testset/en/meta.lst \
        --max-samples 10 --no-ref-audio
"""

from __future__ import annotations

import os

from datasets import load_dataset

from .benchmarker import Benchmarker, BenchmarkRequest

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
            if sample.get("target_text"):
                payload["ref_text"] = sample["target_text"]

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
            expected_text=sample.get("text"),
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
