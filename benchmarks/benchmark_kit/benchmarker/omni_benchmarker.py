"""
Omni benchmarker — measures speed and accuracy for multi-modal chat completions.

Supports text, image, audio, and video inputs via /v1/chat/completions.
Measures TTFT, TPOT, end-to-end latency, throughput, and optional accuracy.

Usage:

    # Text-only benchmark
    python -m benchmarks.benchmarker.omni_benchmarker \
        --model Qwen/Qwen3-Omni --port 8000 \
        --dataset dataset.jsonl --stream

    # Multi-modal with accuracy evaluation
    python -m benchmarks.benchmarker.omni_benchmarker \
        --model Qwen/Qwen3-Omni --port 8000 \
        --dataset dataset.jsonl --stream \
        --modalities text audio \
        --max-concurrency 4

Dataset format (JSONL, one JSON object per line):

    {
        "id": "sample_001",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Describe this image."}
        ],
        "images": ["path/to/image.jpg"],       // optional
        "audios": ["path/to/audio.wav"],        // optional
        "videos": ["path/to/video.mp4"],        // optional
        "expected_text": "A cat sitting on...", // optional, for accuracy
        "expected_audio_path": "path/to/ref.wav" // optional
    }
"""

from __future__ import annotations

import json
import os

from .benchmarker import Benchmarker, BenchmarkRequest, BenchmarkResult


class OmniBenchmarker(Benchmarker):
    """Benchmarker for omni models via /v1/chat/completions."""

    def __init__(
        self,
        dataset_path: str,
        max_samples: int | None = None,
        max_tokens: int = 256,
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        system_prompt: str | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.dataset_path = dataset_path
        self.max_samples = max_samples
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.system_prompt = system_prompt

    @property
    def task_type(self) -> str:
        return "omni"

    def load_dataset(self) -> list[dict]:
        """Load a JSONL dataset file."""
        if not os.path.isfile(self.dataset_path):
            raise FileNotFoundError(f"Dataset not found: {self.dataset_path}")

        samples: list[dict] = []
        with open(self.dataset_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                sample = json.loads(line)
                if "id" not in sample:
                    sample["id"] = f"sample_{len(samples)}"
                samples.append(sample)
                if self.max_samples and len(samples) >= self.max_samples:
                    break
        return samples

    def build_request(self, sample: dict) -> BenchmarkRequest:
        api_url = f"{self.base_url}/v1/chat/completions"

        messages = sample.get("messages", [])
        if self.system_prompt and not any(m.get("role") == "system" for m in messages):
            messages = [{"role": "system", "content": self.system_prompt}] + messages

        # If no messages but has a prompt field, create a simple user message
        if not messages and "prompt" in sample:
            messages = [{"role": "user", "content": sample["prompt"]}]

        payload: dict = {
            "model": "",
            "messages": messages,
            "stream": self.stream,
        }

        # Optional multi-modal inputs
        for key in ("images", "audios", "videos"):
            if sample.get(key):
                payload[key] = sample[key]

        # Sampling parameters
        if self.max_tokens is not None:
            payload["max_tokens"] = self.max_tokens
        if self.temperature is not None:
            payload["temperature"] = self.temperature
        if self.top_p is not None:
            payload["top_p"] = self.top_p
        if self.top_k is not None:
            payload["top_k"] = self.top_k

        return BenchmarkRequest(
            request_id=sample["id"],
            task_type="omni",
            payload=payload,
            api_url=api_url,
            stream=self.stream,
            expected_text=sample.get("expected_text"),
            expected_audio_path=sample.get("expected_audio_path"),
        )

    def compute_accuracy(
        self,
        result: BenchmarkResult,
        sample: dict,
    ) -> float | None:
        """Simple exact-match accuracy against expected_text.

        Override this method with a more sophisticated scorer
        (e.g., BLEU, ROUGE, WER, or LLM-as-judge) for real evaluations.
        """
        expected = sample.get("expected_text")
        if expected is None or not result.generated_text:
            return None
        # Normalize whitespace for comparison
        gen = result.generated_text.strip().lower()
        exp = expected.strip().lower()
        if gen == exp:
            return 1.0
        # Partial credit: check if expected is contained in generated
        if exp in gen:
            return 0.5
        return 0.0

    def get_config(self) -> dict:
        config = super().get_config()
        config.update(
            {
                "dataset_path": self.dataset_path,
                "max_samples": self.max_samples,
                "max_tokens": self.max_tokens,
                "modalities": self.modalities,
            }
        )
        return config
