"""
Generic benchmarker framework for omni and TTS tasks.

Measures both accuracy and system performance metrics:
  - TTFT (Time To First Token)
  - TPOT (Time Per Output Token)
  - End-to-end wall time / latency
  - RTF (Real-Time Factor) for audio outputs
  - Throughput (requests/second, tokens/second)
  - Accuracy via pluggable scorer
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import List

import aiohttp
import numpy as np
from benchmark_kit.utils import wait_for_server
from tqdm.asyncio import tqdm

from .metrics import Metrics

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

SUMMARY_LABEL_WIDTH = 32
SUMMARY_LINE_WIDTH = 64


@dataclass
class BenchmarkRequest:
    """A single benchmark request to send to the server."""

    request_id: str
    task_type: str  # "tts" or "omni"
    payload: dict  # JSON body for the API
    api_url: str
    stream: bool = False
    # Ground-truth for accuracy evaluation
    expected_text: str | None = None
    expected_audio_path: str | None = None


@dataclass
class BenchmarkResult:
    """Result of a single benchmark request with timing and accuracy info."""

    request_id: str = ""
    task_type: str = ""
    is_success: bool = False
    error: str = ""

    # ── Timing metrics ───────────────────────────────────────────────────
    latency: float = 0.0  # end-to-end wall time (seconds)
    ttft: float = 0.0  # time to first token (seconds)
    tpot: float = 0.0  # time per output token (seconds)

    # ── Token metrics ────────────────────────────────────────────────────
    prompt_tokens: int = 0
    completion_tokens: int = 0
    engine_time_s: float = 0.0
    tok_per_s: float = 0.0

    # ── Audio metrics (TTS / omni with audio output) ─────────────────────
    audio_duration_s: float = None
    rtf: float = None  # real-time factor = latency / audio_duration

    # -- Video and Image metrics ----
    # TODO (shenggui): add vision output metrics
    video_duration_s: float = None
    height: int = None
    width: int = None
    fps: int = None

    # ── Generated content ────────────────────────────────────────────────
    generated_text: str = ""
    generated_audio_bytes: bytes = field(default_factory=bytes, repr=False)
    generated_image_bytes: bytes = field(default_factory=bytes, repr=False)
    generated_video_bytes: bytes = field(default_factory=bytes, repr=False)


# ── Result saving ────────────────────────────────────────────────────────────
def compute_system_performance(outputs: list[BenchmarkResult]) -> dict:
    """Compute system performance metrics from a list of BenchmarkResult."""
    return {
        "avg_latency_s": sum(o.latency for o in outputs) / len(outputs),
        "avg_ttft_s": sum(o.ttft for o in outputs) / len(outputs),
        "avg_tpot_ms": sum(o.tpot * 1000 for o in outputs) / len(outputs),
        "avg_generated_tokens": sum(o.completion_tokens for o in outputs)
        / len(outputs),
        "avg_tok_per_s": sum(o.tok_per_s for o in outputs) / len(outputs),
    }


def save_results(
    outputs: list[BenchmarkResult],
    system_performance: dict,
    metrics_summary: dict,
    config: dict,
    output_dir: str,
) -> None:
    """Save JSON results to output_dir."""
    os.makedirs(output_dir, exist_ok=True)
    json_results = {
        "system_performance": system_performance,
        "metrics": metrics_summary,
        "config": config,
        "per_request": [
            {
                "id": o.request_id,
                "task_type": o.task_type,
                "is_success": o.is_success,
                "latency_s": round(o.latency, 4),
                "ttft_s": round(o.ttft, 4) if o.ttft > 0 else None,
                "tpot_ms": round(o.tpot * 1000, 2) if o.tpot > 0 else None,
                "audio_duration_s": (
                    round(o.audio_duration_s, 4) if o.audio_duration_s > 0 else None
                ),
                "rtf": (round(o.rtf, 4) if 0 < o.rtf < float("inf") else None),
                "prompt_tokens": o.prompt_tokens or None,
                "completion_tokens": o.completion_tokens or None,
                "tok_per_s": round(o.tok_per_s, 1) if o.tok_per_s > 0 else None,
                "generated_text": o.generated_text or None,
                "error": o.error or None,
            }
            for o in outputs
        ],
    }
    json_path = os.path.join(output_dir, "benchmark_results.json")
    with open(json_path, "w") as f:
        json.dump(json_results, f, indent=2, ensure_ascii=False)
    logger.info("JSON results saved to %s", json_path)


def _detect_image_extension(content: bytes) -> str:
    """Infer image extension from magic bytes, defaulting to .img."""
    if content.startswith(b"\x89PNG\r\n\x1a\n"):
        return ".png"
    if content.startswith(b"\xff\xd8\xff"):
        return ".jpg"
    if content.startswith((b"GIF87a", b"GIF89a")):
        return ".gif"
    if content.startswith(b"BM"):
        return ".bmp"
    if content.startswith((b"II*\x00", b"MM\x00*")):
        return ".tiff"
    if len(content) >= 12 and content[:4] == b"RIFF" and content[8:12] == b"WEBP":
        return ".webp"
    if (
        len(content) >= 12
        and content[4:8] == b"ftyp"
        and content[8:12]
        in {
            b"avif",
            b"avis",
        }
    ):
        return ".avif"
    return ".img"


def _detect_video_extension(content: bytes) -> str:
    """Infer video/container extension from magic bytes, defaulting to .mp4."""
    if len(content) >= 12 and content[4:8] == b"ftyp":
        major_brand = content[8:12]
        if major_brand == b"qt  ":
            return ".mov"
        return ".mp4"
    if content.startswith(b"\x1a\x45\xdf\xa3"):
        return ".webm"
    if len(content) >= 12 and content[:4] == b"RIFF" and content[8:12] == b"AVI ":
        return ".avi"
    if content.startswith(b"OggS"):
        return ".ogv"
    return ".mp4"


class Benchmarker(ABC):
    """Abstract base class for task-specific benchmarkers.

    Subclasses implement dataset loading, request building, and optional
    accuracy scoring. The base class handles server readiness, warmup,
    concurrent request dispatch, metrics computation, and result saving.
    """

    def __init__(
        self,
        base_url: str,
        stream: bool = False,
        max_concurrency: int = 1,
        request_rate: float = float("inf"),
        warmup: int = 1,
        metrics: List[Metrics] = None,
    ) -> None:
        self.base_url = base_url
        self.stream = stream
        self.max_concurrency = max_concurrency
        self.request_rate = request_rate
        self.warmup = warmup
        self.metrics = metrics or []

    @property
    @abstractmethod
    def task_type(self) -> str:
        """Return the task type identifier (e.g. 'tts', 'omni')."""

    @abstractmethod
    def load_dataset(self) -> list[dict]:
        """Load and return the dataset as a list of sample dicts."""

    @abstractmethod
    def build_request(self, sample: dict) -> BenchmarkRequest:
        """Convert a single dataset sample into a BenchmarkRequest."""

    def get_config(self) -> dict:
        """Return a dict of benchmark configuration for saving."""
        return {
            "base_url": self.base_url,
            "task_type": self.task_type,
            "stream": self.stream,
            "max_concurrency": self.max_concurrency,
            "request_rate": self.request_rate,
            "warmup": self.warmup,
            "metrics": [m.name for m in self.metrics],
        }

    async def run(self, output_dir: str):
        """Execute the full benchmark pipeline and return the metrics dict."""
        # ensure the http server is up and running
        wait_for_server(self.base_url)

        # load the dataset and build the requests
        samples = self.load_dataset()
        requests = [self.build_request(s) for s in samples]
        logger.info(
            "Prepared %d requests for task_type=%s", len(requests), self.task_type
        )
        os.makedirs(output_dir, exist_ok=True)

        # Warmup
        if self.warmup > 0:
            logger.info("Warmup (%d requests)...", self.warmup)
            async with aiohttp.ClientSession() as session:
                for i in range(min(self.warmup, len(requests))):
                    out = await self.send_request(requests[i], session)
                    status = "ok" if out.is_success else out.error
                    logger.info("  warmup %d/%d: %s", i + 1, self.warmup, status)

        # Benchmark
        logger.info(
            "Benchmarking %d requests (max_concurrency=%s)...",
            len(requests),
            self.max_concurrency,
        )
        outputs = await self._run_requests(requests, output_dir)

        # Accuracy scoring
        for result, sample in zip(outputs, samples):
            for metric in self.metrics:
                metric_result = metric.compute(sample, result)
                metric.append_result(metric_result)

        # print system performance
        system_performance = compute_system_performance(outputs)
        metrics_summary = {m.name: asdict(m.compute_summary()) for m in self.metrics}

        # print the system perf and metrics
        print("=" * 10, "System Performance", "=" * 10)
        for k, v in system_performance.items():
            print(f"{k}: {v:.2f}")
        print("\n")

        print("=" * 10, "Metrics Summary", "=" * 10)
        for k, v in metrics_summary.items():
            print(f"{k}: {v}")
        print("\n")

        # save the benchmarking results
        save_results(
            outputs, system_performance, metrics_summary, self.get_config(), output_dir
        )

    async def _run_requests(
        self,
        requests: list[BenchmarkRequest],
        output_dir: str | None,
    ) -> list[BenchmarkResult]:
        """Send all requests with concurrency and rate limiting."""
        semaphore = (
            asyncio.Semaphore(self.max_concurrency) if self.max_concurrency else None
        )

        async def _limited(
            req: BenchmarkRequest,
            session: aiohttp.ClientSession,
            pbar: tqdm,
        ) -> BenchmarkResult:
            if semaphore:
                async with semaphore:
                    return await self.send_request(req, session, output_dir, pbar)
            return await self.send_request(req, session, output_dir, pbar)

        pbar = tqdm(total=len(requests))
        async with aiohttp.ClientSession() as session:
            tasks = []
            for req in requests:
                if self.request_rate != float("inf"):
                    interval = np.random.exponential(1.0 / self.request_rate)
                    await asyncio.sleep(interval)
                tasks.append(asyncio.create_task(_limited(req, session, pbar)))
            outputs: list[BenchmarkResult] = await asyncio.gather(*tasks)
        pbar.close()
        return outputs

    async def send_request(
        self,
        request: BenchmarkRequest,
        session: aiohttp.ClientSession,
        output_dir: str | None = None,
        pbar: tqdm | None = None,
    ) -> BenchmarkResult:
        """Send a single benchmark request and return the result with timing."""
        result = BenchmarkResult(
            request_id=request.request_id,
            task_type=request.task_type,
        )

        start_time = time.perf_counter()
        try:
            async with session.post(request.api_url, json=request.payload) as response:
                if response.status != 200:
                    result.error = f"HTTP {response.status}: {await response.text()}"
                else:
                    if request.stream:
                        handler = self.handle_streaming_response
                    else:
                        handler = self.handle_non_streaming_response
                    await handler(response, result, start_time)
        except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
            result.error = str(exc)
        finally:
            result.latency = time.perf_counter() - start_time

        # Save audio if requested
        if output_dir is not None:
            if result.generated_audio_bytes:
                audio_path = os.path.join(output_dir, f"{request.request_id}.wav")
                with open(audio_path, "wb") as f:
                    f.write(result.generated_audio_bytes)

            if result.generated_image_bytes:
                image_ext = _detect_image_extension(result.generated_image_bytes)
                image_path = os.path.join(
                    output_dir, f"{request.request_id}{image_ext}"
                )
                with open(image_path, "wb") as f:
                    f.write(result.generated_image_bytes)

            if result.generated_video_bytes:
                video_ext = _detect_video_extension(result.generated_video_bytes)
                video_path = os.path.join(
                    output_dir, f"{request.request_id}{video_ext}"
                )
                with open(video_path, "wb") as f:
                    f.write(result.generated_video_bytes)
        if pbar:
            pbar.update(1)
        return result

    @abstractmethod
    def handle_streaming_response(
        self,
        response: aiohttp.ClientResponse,
        result: BenchmarkResult,
        start_time: float,
    ) -> None:
        pass

    @abstractmethod
    def handle_non_streaming_response(
        self,
        response: aiohttp.ClientResponse,
        result: BenchmarkResult,
        start_time: float,
    ) -> None:
        pass

    def _apply_usage(self, result: BenchmarkResult, usage: dict) -> None:
        """Apply usage dict (from SSE or JSON body) to result."""
        prompt_tok = usage.get("prompt_tokens")
        comp_tok = usage.get("completion_tokens")
        eng_time = usage.get("engine_time_s")
        if prompt_tok is not None:
            result.prompt_tokens = int(prompt_tok)
        if comp_tok is not None:
            result.completion_tokens = int(comp_tok)
        if eng_time is not None:
            result.engine_time_s = float(eng_time)
        if result.completion_tokens > 0 and result.engine_time_s > 0:
            result.tok_per_s = result.completion_tokens / result.engine_time_s
        if result.completion_tokens > 1 and result.latency > 0:
            result.tpot = result.engine_time_s / (result.completion_tokens - 1)

    def _apply_headers(self, result: BenchmarkResult, headers: dict) -> None:
        """Extract token counts and engine time from HTTP response headers."""
        usage = {}
        if headers.get("X-Prompt-Tokens") is not None:
            usage["prompt_tokens"] = headers["X-Prompt-Tokens"]
        if headers.get("X-Completion-Tokens") is not None:
            usage["completion_tokens"] = headers["X-Completion-Tokens"]
        if headers.get("X-Engine-Time") is not None:
            usage["engine_time_s"] = headers["X-Engine-Time"]
        if usage:
            self._apply_usage(result, usage)
