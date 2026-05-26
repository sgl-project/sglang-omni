# SPDX-License-Identifier: Apache-2.0
"""Streaming time-to-first-audio-chunk (TTFT) benchmark for Qwen3-Omni speech.

Measures wall-clock latency from request submission to the first audio
delta returned by ``POST /v1/chat/completions`` with
``modalities=["text","audio"]`` and ``stream=true``. Designed to surface
the gain from partial-prefix talker startup (``partial_start_min_chunks``),
which MMMU end-to-end accuracy benchmarks cannot observe because their
total-request latency is dominated by the thinker.

Usage:
    # Baseline server (partial-start disabled):
    python examples/run_qwen3_omni_speech_server.py --port 8001 ...
    python benchmarks/eval/benchmark_omni_streaming_ttft.py \\
        --base-url http://localhost:8001 \\
        --label baseline --repeats 5

    # Treatment server (partial-start enabled):
    python examples/run_qwen3_omni_speech_server.py --port 8001 \\
        --enable-partial-start --partial-start-min-chunks 5 ...
    python benchmarks/eval/benchmark_omni_streaming_ttft.py \\
        --base-url http://localhost:8001 \\
        --label partial5 --repeats 5
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import statistics
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmarks.benchmarker.utils import wait_for_service  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

PROMPTS: dict[str, str] = {
    "short": "Please reply: Hello, how are you today?",
    "medium": (
        "Please respond with the following sentence verbatim: "
        "The quick brown fox jumps over the lazy dog while the sun sets "
        "over the quiet hills, and the river continues to flow gently "
        "through the valley."
    ),
}


@dataclass
class RunResult:
    label: str
    prompt_id: str
    repeat: int
    ttft_seconds: float
    total_seconds: float
    audio_chunks: int
    status_code: int


@dataclass
class Summary:
    label: str
    base_url: str
    per_run: list[RunResult] = field(default_factory=list)
    aggregate: dict[str, dict[str, float]] = field(default_factory=dict)


async def _measure_one(
    client: httpx.AsyncClient,
    base_url: str,
    model: str,
    prompt: str,
    *,
    request_id_hint: str,
    seed: int,
    timeout_s: float,
) -> tuple[float, float, int, int]:
    """Stream a chat completion with modalities=[text, audio] and time the
    first audio delta. The talker_ar pipeline owns the audio output; this is
    the metric ``partial_start_min_chunks`` is designed to move.
    """
    url = f"{base_url.rstrip('/')}/v1/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "modalities": ["text", "audio"],
        "audio": {"voice": "alloy", "format": "wav"},
        "stream": True,
        "seed": seed,
        "max_tokens": 256,
        "metadata": {"client_label": request_id_hint},
    }

    start = time.perf_counter()
    ttft: float | None = None
    audio_chunks = 0
    status_code = 0

    async with client.stream("POST", url, json=payload, timeout=timeout_s) as response:
        status_code = response.status_code
        if status_code >= 400:
            text = await response.aread()
            raise RuntimeError(f"server returned {status_code}: {text[:512]!r}")
        async for raw_line in response.aiter_lines():
            line = raw_line.strip()
            if not line.startswith("data:"):
                continue
            body = line[len("data:") :].strip()
            if body == "[DONE]":
                continue
            try:
                evt = json.loads(body)
            except json.JSONDecodeError:
                continue
            for choice in evt.get("choices", []):
                delta = choice.get("delta") or {}
                audio = delta.get("audio")
                if audio and audio.get("data"):
                    if ttft is None:
                        ttft = time.perf_counter() - start
                    audio_chunks += 1

    total = time.perf_counter() - start
    if ttft is None:
        raise RuntimeError("server returned 200 but no audio delta arrived")
    return ttft, total, audio_chunks, status_code


async def _run(args: argparse.Namespace) -> Summary:
    summary = Summary(label=args.label, base_url=args.base_url)
    async with httpx.AsyncClient(http2=False) as client:
        for prompt_id, prompt_text in PROMPTS.items():
            # Discard warmup runs from aggregates but keep their raw timings
            # for diagnostic context.
            for warm in range(args.warmup):
                seed = 9000 + warm
                hint = f"{args.label}-{prompt_id}-warmup{warm}"
                ttft, total, audio_chunks, status_code = await _measure_one(
                    client,
                    args.base_url,
                    args.model,
                    prompt_text,
                    request_id_hint=hint,
                    seed=seed,
                    timeout_s=args.timeout_s,
                )
                logger.info(
                    "[%s] WARMUP prompt=%s repeat=%d ttft=%.3fs total=%.3fs "
                    "audio_chunks=%d status_code=%d",
                    args.label,
                    prompt_id,
                    warm,
                    ttft,
                    total,
                    audio_chunks,
                    status_code,
                )

            ttfts: list[float] = []
            totals: list[float] = []
            for repeat in range(args.repeats):
                seed = 1000 + repeat
                hint = f"{args.label}-{prompt_id}-{repeat}"
                ttft, total, audio_chunks, status_code = await _measure_one(
                    client,
                    args.base_url,
                    args.model,
                    prompt_text,
                    request_id_hint=hint,
                    seed=seed,
                    timeout_s=args.timeout_s,
                )
                summary.per_run.append(
                    RunResult(
                        label=args.label,
                        prompt_id=prompt_id,
                        repeat=repeat,
                        ttft_seconds=ttft,
                        total_seconds=total,
                        audio_chunks=audio_chunks,
                        status_code=status_code,
                    )
                )
                ttfts.append(ttft)
                totals.append(total)
                logger.info(
                    "[%s] prompt=%s repeat=%d ttft=%.3fs total=%.3fs audio_chunks=%d",
                    args.label,
                    prompt_id,
                    repeat,
                    ttft,
                    total,
                    audio_chunks,
                )

            summary.aggregate[prompt_id] = {
                "ttft_mean": statistics.fmean(ttfts),
                "ttft_min": min(ttfts),
                "ttft_max": max(ttfts),
                "ttft_stdev": statistics.pstdev(ttfts) if len(ttfts) > 1 else 0.0,
                "total_mean": statistics.fmean(totals),
            }
    return summary


def _print_summary(summary: Summary) -> None:
    print("\n" + "=" * 60)
    print(f"  Streaming TTFT — label={summary.label}")
    print(f"  base_url={summary.base_url}")
    print("=" * 60)
    for prompt_id, agg in summary.aggregate.items():
        print(
            f"  prompt={prompt_id:<7} ttft_mean={agg['ttft_mean']:.3f}s  "
            f"min={agg['ttft_min']:.3f}s  max={agg['ttft_max']:.3f}s  "
            f"stdev={agg['ttft_stdev']:.3f}s  total_mean={agg['total_mean']:.3f}s"
        )
    print("=" * 60)


def _default_output_path(label: str) -> Path:
    run_id = time.strftime("%Y%m%d-%H%M%S")
    return Path("results") / f"ttft_{label}_{run_id}.json"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Measure streaming TTFT for Qwen3-Omni speech.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--model", default="qwen3-omni")
    parser.add_argument(
        "--label",
        required=True,
        help="Label for this run, e.g. 'baseline' or 'partial5'.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Path to write JSON results. Defaults to results/ttft_<label>_<run-id>.json.",
    )
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--timeout-s", type=float, default=300.0)
    args = parser.parse_args(argv)
    if args.output is None:
        args.output = _default_output_path(args.label)
    elif args.output.exists():
        raise FileExistsError(f"output path already exists: {args.output}")

    wait_for_service(args.base_url)
    summary = asyncio.run(_run(args))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "label": summary.label,
                "base_url": summary.base_url,
                "per_run": [asdict(r) for r in summary.per_run],
                "aggregate": summary.aggregate,
            },
            indent=2,
        )
    )
    _print_summary(summary)
    logger.info("wrote %s", args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
