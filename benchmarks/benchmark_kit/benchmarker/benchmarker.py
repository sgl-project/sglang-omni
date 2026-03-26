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
import base64
import csv
import json
import logging
import os
import struct
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import aiohttp
import numpy as np
from tqdm.asyncio import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────

WAV_HEADER_SIZE = 44
SSE_DATA_PREFIX = "data: "
SSE_DONE_MARKER = "data: [DONE]"
SUMMARY_LABEL_WIDTH = 32
SUMMARY_LINE_WIDTH = 64


# ── Data classes ─────────────────────────────────────────────────────────────


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
    audio_duration_s: float = 0.0
    rtf: float = 0.0  # real-time factor = latency / audio_duration

    # ── Generated content ────────────────────────────────────────────────
    generated_text: str = ""
    generated_audio_bytes: bytes = field(default_factory=bytes, repr=False)

    # ── Accuracy ─────────────────────────────────────────────────────────
    accuracy_score: float | None = None


# ── Utility helpers ──────────────────────────────────────────────────────────


def wav_duration(wav_bytes: bytes) -> float:
    """Return duration in seconds from a WAV byte buffer."""
    if len(wav_bytes) <= WAV_HEADER_SIZE:
        return 0.0
    sample_rate = struct.unpack_from("<I", wav_bytes, 24)[0]
    num_channels = struct.unpack_from("<H", wav_bytes, 22)[0]
    bits_per_sample = struct.unpack_from("<H", wav_bytes, 34)[0]
    if sample_rate == 0 or num_channels == 0 or bits_per_sample == 0:
        return 0.0
    bytes_per_sample = num_channels * bits_per_sample // 8
    pcm_size = len(wav_bytes) - WAV_HEADER_SIZE
    return pcm_size / (sample_rate * bytes_per_sample)


def wait_for_service(base_url: str, timeout: int = 1200) -> None:
    """Block until the server health endpoint returns 200."""
    import requests as requests_lib

    logger.info("Waiting for service at %s ...", base_url)
    start = time.time()
    while True:
        try:
            resp = requests_lib.get(f"{base_url}/health", timeout=1)
            if resp.status_code == 200:
                logger.info("Service is ready.")
                return
        except requests_lib.exceptions.RequestException:
            pass
        if time.time() - start > timeout:
            raise TimeoutError(f"Service at {base_url} not ready within {timeout}s")
        time.sleep(1)


def _parse_sse_event(line: str) -> dict | None:
    """Parse an SSE data line into a JSON dict, or None."""
    if not line.startswith(SSE_DATA_PREFIX) or line == SSE_DONE_MARKER:
        return None
    return json.loads(line[len(SSE_DATA_PREFIX) :])


# ── Streaming helpers ────────────────────────────────────────────────────────


async def _iter_sse_lines(response: aiohttp.ClientResponse):
    """Yield decoded SSE lines from an aiohttp streaming response."""
    buffer = bytearray()
    async for chunk in response.content.iter_any():
        buffer.extend(chunk)
        while b"\n" in buffer:
            idx = buffer.index(b"\n")
            raw_line = bytes(buffer[:idx])
            del buffer[: idx + 1]
            line = raw_line.decode("utf-8", errors="replace").strip()
            if line:
                yield line
    if buffer.strip():
        yield bytes(buffer).decode("utf-8", errors="replace").strip()


async def handle_tts_streaming(
    response: aiohttp.ClientResponse,
    result: BenchmarkResult,
    start_time: float,
) -> None:
    """Parse SSE stream for TTS: decode audio chunks, compute metrics."""
    total_audio_duration = 0.0
    usage_data: dict | None = None
    first_chunk_time: float | None = None

    async for line in _iter_sse_lines(response):
        event = _parse_sse_event(line)
        if event is None:
            continue
        if first_chunk_time is None:
            first_chunk_time = time.perf_counter()
        audio = event.get("audio")
        if isinstance(audio, dict) and audio.get("data"):
            total_audio_duration += wav_duration(base64.b64decode(audio["data"]))
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
        _apply_usage(result, usage_data)


async def handle_omni_streaming(
    response: aiohttp.ClientResponse,
    result: BenchmarkResult,
    start_time: float,
) -> None:
    """Parse SSE stream for omni chat completions: collect text/audio, compute timing."""
    text_parts: list[str] = []
    audio_chunks: list[bytes] = []
    first_chunk_time: float | None = None
    chunk_count = 0
    usage_data: dict | None = None

    async for line in _iter_sse_lines(response):
        event = _parse_sse_event(line)
        if event is None:
            continue

        if first_chunk_time is None:
            first_chunk_time = time.perf_counter()
        chunk_count += 1

        for choice in event.get("choices", []):
            delta = choice.get("delta", {})
            content = delta.get("content")
            if content:
                text_parts.append(content)
            audio = delta.get("audio")
            if isinstance(audio, dict) and audio.get("data"):
                audio_chunks.append(base64.b64decode(audio["data"]))

        event_usage = event.get("usage")
        if isinstance(event_usage, dict):
            usage_data = event_usage

    result.generated_text = "".join(text_parts)
    if audio_chunks:
        result.generated_audio_bytes = b"".join(audio_chunks)
        result.audio_duration_s = wav_duration(result.generated_audio_bytes)

    if first_chunk_time is not None:
        result.ttft = first_chunk_time - start_time

    if chunk_count > 1 and first_chunk_time is not None:
        elapsed = time.perf_counter() - first_chunk_time
        result.tpot = elapsed / (chunk_count - 1)

    if result.audio_duration_s > 0:
        elapsed = time.perf_counter() - start_time
        result.rtf = elapsed / result.audio_duration_s

    result.is_success = bool(text_parts or audio_chunks)

    if usage_data:
        _apply_usage(result, usage_data)


# ── Non-streaming helpers ────────────────────────────────────────────────────


async def handle_tts_non_streaming(
    response: aiohttp.ClientResponse,
    result: BenchmarkResult,
    start_time: float,
) -> None:
    """Read full TTS audio response, compute metrics."""
    audio_bytes = await response.read()
    result.generated_audio_bytes = audio_bytes
    result.audio_duration_s = wav_duration(audio_bytes)
    elapsed = time.perf_counter() - start_time
    # For non-streaming, TTFT equals the full latency (first token = complete response)
    result.ttft = elapsed
    if result.audio_duration_s > 0:
        result.is_success = True
        result.rtf = elapsed / result.audio_duration_s
    else:
        result.error = f"Empty or invalid audio response ({len(audio_bytes)} bytes)"
        return
    _apply_headers(result, response.headers)


async def handle_omni_non_streaming(
    response: aiohttp.ClientResponse,
    result: BenchmarkResult,
    start_time: float,
) -> None:
    """Read full omni chat completion response, compute metrics."""
    body = await response.json()
    elapsed = time.perf_counter() - start_time
    result.ttft = elapsed  # non-streaming: TTFT = full latency

    choices = body.get("choices", [])
    if choices:
        message = choices[0].get("message", {})
        result.generated_text = message.get("content", "")
        audio = message.get("audio")
        if isinstance(audio, dict) and audio.get("data"):
            result.generated_audio_bytes = base64.b64decode(audio["data"])
            result.audio_duration_s = wav_duration(result.generated_audio_bytes)

    usage = body.get("usage")
    if isinstance(usage, dict):
        _apply_usage(result, usage)

    if result.audio_duration_s > 0:
        result.rtf = elapsed / result.audio_duration_s

    result.is_success = bool(result.generated_text or result.generated_audio_bytes)
    if not result.is_success:
        result.error = "No content in response"


# ── Shared helpers ───────────────────────────────────────────────────────────


def _apply_usage(result: BenchmarkResult, usage: dict) -> None:
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
    if result.completion_tokens > 1 and result.latency > 0 and result.ttft > 0:
        generation_time = result.latency - result.ttft
        if generation_time > 0:
            result.tpot = generation_time / (result.completion_tokens - 1)


def _apply_headers(result: BenchmarkResult, headers: dict) -> None:
    """Extract token counts and engine time from HTTP response headers."""
    usage = {}
    if headers.get("X-Prompt-Tokens") is not None:
        usage["prompt_tokens"] = headers["X-Prompt-Tokens"]
    if headers.get("X-Completion-Tokens") is not None:
        usage["completion_tokens"] = headers["X-Completion-Tokens"]
    if headers.get("X-Engine-Time") is not None:
        usage["engine_time_s"] = headers["X-Engine-Time"]
    if usage:
        _apply_usage(result, usage)


# ── Request sender ───────────────────────────────────────────────────────────

# Handler dispatch: (task_type, stream) -> handler function
_HANDLERS = {
    ("tts", True): handle_tts_streaming,
    ("tts", False): handle_tts_non_streaming,
    ("omni", True): handle_omni_streaming,
    ("omni", False): handle_omni_non_streaming,
}


async def send_request(
    request: BenchmarkRequest,
    session: aiohttp.ClientSession,
    save_audio_dir: str | None = None,
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
                handler = _HANDLERS.get((request.task_type, request.stream))
                if handler is None:
                    result.error = (
                        f"Unknown task_type={request.task_type!r}, "
                        f"stream={request.stream}"
                    )
                else:
                    await handler(response, result, start_time)
    except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
        result.error = str(exc)
    finally:
        result.latency = time.perf_counter() - start_time

    # Save audio if requested
    if save_audio_dir and result.generated_audio_bytes:
        audio_path = os.path.join(save_audio_dir, f"{request.request_id}.wav")
        with open(audio_path, "wb") as f:
            f.write(result.generated_audio_bytes)

    if pbar:
        pbar.update(1)
    return result


# ── Metrics computation ──────────────────────────────────────────────────────


def calculate_metrics(outputs: list[BenchmarkResult]) -> dict:
    """Compute aggregate performance and accuracy metrics."""
    successes = [o for o in outputs if o.is_success]
    if not successes:
        return {"completed_requests": 0, "failed_requests": len(outputs)}

    latencies = [o.latency for o in successes]
    total_wall = sum(latencies)

    metrics: dict = {
        "completed_requests": len(successes),
        "failed_requests": len(outputs) - len(successes),
        # Latency
        "latency_mean_s": round(float(np.mean(latencies)), 3),
        "latency_median_s": round(float(np.median(latencies)), 3),
        "latency_p95_s": round(float(np.percentile(latencies, 95)), 3),
        "latency_p99_s": round(float(np.percentile(latencies, 99)), 3),
        # Throughput
        "throughput_qps": (
            round(len(successes) / total_wall, 3) if total_wall > 0 else 0
        ),
    }

    # TTFT
    ttfts = [o.ttft for o in successes if o.ttft > 0]
    if ttfts:
        metrics["ttft_mean_s"] = round(float(np.mean(ttfts)), 4)
        metrics["ttft_median_s"] = round(float(np.median(ttfts)), 4)
        metrics["ttft_p95_s"] = round(float(np.percentile(ttfts, 95)), 4)
        metrics["ttft_p99_s"] = round(float(np.percentile(ttfts, 99)), 4)

    # TPOT
    tpots = [o.tpot for o in successes if o.tpot > 0]
    if tpots:
        metrics["tpot_mean_ms"] = round(float(np.mean(tpots)) * 1000, 2)
        metrics["tpot_median_ms"] = round(float(np.median(tpots)) * 1000, 2)
        metrics["tpot_p95_ms"] = round(float(np.percentile(tpots, 95)) * 1000, 2)
        metrics["tpot_p99_ms"] = round(float(np.percentile(tpots, 99)) * 1000, 2)

    # RTF (for audio outputs)
    rtfs = [o.rtf for o in successes if 0 < o.rtf < float("inf")]
    if rtfs:
        metrics["rtf_mean"] = round(float(np.mean(rtfs)), 4)
        metrics["rtf_median"] = round(float(np.median(rtfs)), 4)

    # Audio duration
    audio_durations = [o.audio_duration_s for o in successes if o.audio_duration_s > 0]
    if audio_durations:
        metrics["audio_duration_mean_s"] = round(float(np.mean(audio_durations)), 3)

    # Token metrics
    tokens_per_sec = [o.tok_per_s for o in successes if o.tok_per_s > 0]
    gen_token_counts = [
        o.completion_tokens for o in successes if o.completion_tokens > 0
    ]
    prompt_token_counts = [o.prompt_tokens for o in successes if o.prompt_tokens > 0]
    total_tokens = sum(gen_token_counts)
    total_engine_time = sum(o.engine_time_s for o in successes if o.engine_time_s > 0)

    if tokens_per_sec:
        metrics["tok_per_s_mean"] = round(float(np.mean(tokens_per_sec)), 1)
        metrics["tok_per_s_median"] = round(float(np.median(tokens_per_sec)), 1)
    if total_engine_time > 0 and total_tokens > 0:
        metrics["tok_per_s_agg"] = round(total_tokens / total_engine_time, 1)
    if gen_token_counts:
        metrics["gen_tokens_mean"] = round(float(np.mean(gen_token_counts)), 0)
        metrics["gen_tokens_total"] = total_tokens
    if prompt_token_counts:
        metrics["prompt_tokens_mean"] = round(float(np.mean(prompt_token_counts)), 0)
        metrics["prompt_tokens_total"] = sum(prompt_token_counts)

    # Accuracy
    scores = [o.accuracy_score for o in successes if o.accuracy_score is not None]
    if scores:
        metrics["accuracy_mean"] = round(float(np.mean(scores)), 4)
        metrics["accuracy_median"] = round(float(np.median(scores)), 4)

    return metrics


# ── Summary printer ──────────────────────────────────────────────────────────


def print_summary(metrics: dict, task_type: str) -> None:
    """Print a formatted summary table of benchmark results."""
    lw = SUMMARY_LABEL_WIDTH
    w = SUMMARY_LINE_WIDTH
    print(f"\n{'=' * w}")
    print(f"{'Benchmark Result':^{w}}")
    print(f"{'=' * w}")
    print(f"  {'Task type:':<{lw}} {task_type}")
    print(f"  {'Completed requests:':<{lw}} {metrics['completed_requests']}")
    print(f"  {'Failed requests:':<{lw}} {metrics['failed_requests']}")

    print(f"{'-' * w}")
    print(f"  {'--- Latency ---':<{lw}}")
    for key, label in [
        ("latency_mean_s", "Latency mean (s):"),
        ("latency_median_s", "Latency median (s):"),
        ("latency_p95_s", "Latency p95 (s):"),
        ("latency_p99_s", "Latency p99 (s):"),
    ]:
        if key in metrics:
            print(f"  {label:<{lw}} {metrics[key]}")

    if "ttft_mean_s" in metrics:
        print(f"{'-' * w}")
        print(f"  {'--- TTFT ---':<{lw}}")
        for key, label in [
            ("ttft_mean_s", "TTFT mean (s):"),
            ("ttft_median_s", "TTFT median (s):"),
            ("ttft_p95_s", "TTFT p95 (s):"),
            ("ttft_p99_s", "TTFT p99 (s):"),
        ]:
            if key in metrics:
                print(f"  {label:<{lw}} {metrics[key]}")

    if "tpot_mean_ms" in metrics:
        print(f"{'-' * w}")
        print(f"  {'--- TPOT ---':<{lw}}")
        for key, label in [
            ("tpot_mean_ms", "TPOT mean (ms):"),
            ("tpot_median_ms", "TPOT median (ms):"),
            ("tpot_p95_ms", "TPOT p95 (ms):"),
            ("tpot_p99_ms", "TPOT p99 (ms):"),
        ]:
            if key in metrics:
                print(f"  {label:<{lw}} {metrics[key]}")

    if "rtf_mean" in metrics:
        print(f"{'-' * w}")
        print(f"  {'--- Audio ---':<{lw}}")
        for key, label in [
            ("rtf_mean", "RTF mean:"),
            ("rtf_median", "RTF median:"),
            ("audio_duration_mean_s", "Audio duration mean (s):"),
        ]:
            if key in metrics:
                print(f"  {label:<{lw}} {metrics[key]}")

    if "tok_per_s_mean" in metrics or "gen_tokens_mean" in metrics:
        print(f"{'-' * w}")
        print(f"  {'--- Tokens ---':<{lw}}")
        for key, label in [
            ("tok_per_s_mean", "Tok/s (per-req mean):"),
            ("tok_per_s_median", "Tok/s (per-req median):"),
            ("tok_per_s_agg", "Tok/s (aggregate):"),
            ("gen_tokens_mean", "Gen tokens (mean):"),
            ("gen_tokens_total", "Gen tokens (total):"),
            ("prompt_tokens_mean", "Prompt tokens (mean):"),
            ("prompt_tokens_total", "Prompt tokens (total):"),
        ]:
            val = metrics.get(key)
            if val is not None:
                if "mean" in key and "tokens" in key:
                    print(f"  {label:<{lw}} {val:.0f}")
                else:
                    print(f"  {label:<{lw}} {val}")

    if "accuracy_mean" in metrics:
        print(f"{'-' * w}")
        print(f"  {'--- Accuracy ---':<{lw}}")
        print(f"  {'Accuracy mean:':<{lw}} {metrics['accuracy_mean']}")
        print(f"  {'Accuracy median:':<{lw}} {metrics['accuracy_median']}")

    print(f"  {'Throughput (req/s):':<{lw}} {metrics.get('throughput_qps', 'N/A')}")
    print(f"{'=' * w}")


# ── Result saving ────────────────────────────────────────────────────────────


def save_results(
    outputs: list[BenchmarkResult],
    metrics: dict,
    config: dict,
    output_dir: str,
) -> None:
    """Save JSON and CSV results to output_dir."""
    os.makedirs(output_dir, exist_ok=True)
    _save_json(outputs, metrics, config, output_dir)
    _save_csv(outputs, output_dir)


def _save_json(
    outputs: list[BenchmarkResult],
    metrics: dict,
    config: dict,
    output_dir: str,
) -> None:
    json_results = {
        "summary": metrics,
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
                "accuracy_score": o.accuracy_score,
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


def _save_csv(outputs: list[BenchmarkResult], output_dir: str) -> None:
    csv_path = os.path.join(output_dir, "benchmark_results.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "id",
                "task_type",
                "latency_s",
                "ttft_s",
                "tpot_ms",
                "audio_duration_s",
                "rtf",
                "prompt_tokens",
                "completion_tokens",
                "tok_per_s",
                "accuracy_score",
                "is_success",
                "error",
            ]
        )
        for o in outputs:
            writer.writerow(
                [
                    o.request_id,
                    o.task_type,
                    f"{o.latency:.4f}",
                    f"{o.ttft:.4f}" if o.ttft > 0 else "",
                    f"{o.tpot * 1000:.2f}" if o.tpot > 0 else "",
                    f"{o.audio_duration_s:.4f}" if o.audio_duration_s > 0 else "",
                    f"{o.rtf:.4f}" if 0 < o.rtf < float("inf") else "",
                    o.prompt_tokens or "",
                    o.completion_tokens or "",
                    f"{o.tok_per_s:.1f}" if o.tok_per_s > 0 else "",
                    o.accuracy_score if o.accuracy_score is not None else "",
                    o.is_success,
                    o.error or "",
                ]
            )
    logger.info("CSV results saved to %s", csv_path)


# ── Abstract benchmarker base ────────────────────────────────────────────────


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
        output_dir: str = "results",
        save_audio: bool = False,
        disable_tqdm: bool = False,
    ) -> None:
        self.base_url = base_url
        self.stream = stream
        self.max_concurrency = max_concurrency
        self.request_rate = request_rate
        self.warmup = warmup
        self.output_dir = output_dir
        self.save_audio = save_audio
        self.disable_tqdm = disable_tqdm

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

    def compute_accuracy(
        self,
        result: BenchmarkResult,
        sample: dict,
    ) -> float | None:
        """Compute accuracy score for a single result.

        Override in subclasses to enable accuracy measurement.
        Returns None if accuracy is not applicable.
        """
        return None

    def get_config(self) -> dict:
        """Return a dict of benchmark configuration for saving."""
        return {
            "base_url": self.base_url,
            "task_type": self.task_type,
            "stream": self.stream,
            "max_concurrency": self.max_concurrency,
            "request_rate": self.request_rate,
            "warmup": self.warmup,
        }

    async def run(self) -> dict:
        """Execute the full benchmark pipeline and return the metrics dict."""
        wait_for_service(self.base_url)

        samples = self.load_dataset()
        requests = [self.build_request(s) for s in samples]
        logger.info(
            "Prepared %d requests for task_type=%s", len(requests), self.task_type
        )

        save_audio_dir = None
        if self.save_audio and self.output_dir:
            save_audio_dir = os.path.join(self.output_dir, "audio")
            os.makedirs(save_audio_dir, exist_ok=True)

        # Warmup
        if self.warmup > 0:
            logger.info("Warmup (%d requests)...", self.warmup)
            async with aiohttp.ClientSession() as session:
                for i in range(min(self.warmup, len(requests))):
                    out = await send_request(requests[i], session)
                    status = "ok" if out.is_success else out.error
                    logger.info("  warmup %d/%d: %s", i + 1, self.warmup, status)

        # Benchmark
        logger.info(
            "Benchmarking %d requests (max_concurrency=%s)...",
            len(requests),
            self.max_concurrency,
        )
        outputs = await self._run_requests(requests, save_audio_dir)

        # Accuracy scoring
        for result, sample in zip(outputs, samples):
            score = self.compute_accuracy(result, sample)
            if score is not None:
                result.accuracy_score = score

        metrics = calculate_metrics(outputs)
        print_summary(metrics, self.task_type)

        if self.output_dir:
            save_results(outputs, metrics, self.get_config(), self.output_dir)

        return metrics

    async def _run_requests(
        self,
        requests: list[BenchmarkRequest],
        save_audio_dir: str | None,
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
                    return await send_request(req, session, save_audio_dir, pbar)
            return await send_request(req, session, save_audio_dir, pbar)

        pbar = tqdm(total=len(requests), disable=self.disable_tqdm)
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
