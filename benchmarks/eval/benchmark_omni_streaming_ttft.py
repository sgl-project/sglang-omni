# SPDX-License-Identifier: Apache-2.0
"""Streaming time-to-first-audio-chunk (TTFT) benchmark for Qwen3-Omni speech.

Measures wall-clock latency from request submission to the first audio
delta returned by POST /v1/chat/completions with
modalities=["text","audio"] and stream=true. Designed to surface
the gain from partial-prefix talker startup (partial_start_min_chunks),
which MMMU end-to-end accuracy benchmarks cannot observe because their
total-request latency is dominated by the thinker.

Usage:
    python examples/run_qwen3_omni_speech_server.py --port 8001 \
        --no-enable-partial-start

    python -m benchmarks.eval.benchmark_omni_streaming_ttft \
        --base-url http://localhost:8001 \
        --label baseline --repeats 5

    # Treatment server (partial-start enabled; explicit flags for the A/B).
    python examples/run_qwen3_omni_speech_server.py --port 8001 \
        --enable-partial-start --partial-start-min-chunks 5

    python -m benchmarks.eval.benchmark_omni_streaming_ttft \
        --base-url http://localhost:8001 \
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
from typing import TypedDict

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmarks.benchmarker.conditions import (  # noqa: E402
    add_fingerprint_argument,
    add_talker_sampling_arguments,
    fingerprint_fields,
)
from benchmarks.benchmarker.utils import wait_for_service  # noqa: E402
from benchmarks.tasks.tts import (  # noqa: E402
    TalkerSamplingParams,
    talker_sampling_params,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

DEFAULT_STREAMING_TTFT_SEED = 1000
MAX_COMPLETION_TOKENS = 256

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
    seed: int
    talker_temperature: float | None
    talker_top_p: float | None
    talker_top_k: int | None
    talker_repetition_penalty: float | None
    per_run: list[RunResult] = field(default_factory=list)
    aggregate: dict[str, dict[str, float]] = field(default_factory=dict)


class StreamingTtftMessage(TypedDict):
    role: str
    content: str


class StreamingTtftAudio(TypedDict):
    voice: str
    format: str


class StreamingTtftMetadata(TypedDict):
    client_label: str


class StreamingTtftPayload(TypedDict, total=False):
    model: str
    messages: list[StreamingTtftMessage]
    modalities: list[str]
    audio: StreamingTtftAudio
    stream: bool
    seed: int
    max_tokens: int
    metadata: StreamingTtftMetadata
    talker_temperature: float
    talker_top_p: float
    talker_top_k: int
    talker_repetition_penalty: float


def streaming_ttft_payload(
    *,
    model: str,
    prompt: str,
    seed: int,
    request_id_hint: str,
    talker_params: TalkerSamplingParams,
) -> StreamingTtftPayload:
    payload: StreamingTtftPayload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "modalities": ["text", "audio"],
        "audio": {"voice": "alloy", "format": "wav"},
        "stream": True,
        "seed": seed,
        "max_tokens": MAX_COMPLETION_TOKENS,
        "metadata": {"client_label": request_id_hint},
    }
    payload.update(talker_params)
    return payload


async def _measure_one(
    client: httpx.AsyncClient,
    base_url: str,
    model: str,
    prompt: str,
    *,
    request_id_hint: str,
    seed: int,
    timeout_s: float,
    talker_params: TalkerSamplingParams,
) -> tuple[float, float, int, int]:
    """Time the first audio delta of one streaming chat completion."""
    url = f"{base_url.rstrip('/')}/v1/chat/completions"
    payload = streaming_ttft_payload(
        model=model,
        prompt=prompt,
        seed=seed,
        request_id_hint=request_id_hint,
        talker_params=talker_params,
    )

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
    talker_params = talker_sampling_params(
        talker_temperature=args.talker_temperature,
        talker_top_p=args.talker_top_p,
        talker_top_k=args.talker_top_k,
        talker_repetition_penalty=args.talker_repetition_penalty,
    )
    summary = Summary(
        label=args.label,
        base_url=args.base_url,
        seed=args.seed,
        talker_temperature=args.talker_temperature,
        talker_top_p=args.talker_top_p,
        talker_top_k=args.talker_top_k,
        talker_repetition_penalty=args.talker_repetition_penalty,
    )
    async with httpx.AsyncClient(http2=False) as client:
        for prompt_id, prompt_text in PROMPTS.items():
            # Warmup replays the measured seed so timed repeats are not cold.
            for warm in range(args.warmup):
                hint = f"{args.label}-{prompt_id}-warmup{warm}"
                ttft, total, audio_chunks, status_code = await _measure_one(
                    client,
                    args.base_url,
                    args.model,
                    prompt_text,
                    request_id_hint=hint,
                    seed=args.seed,
                    timeout_s=args.timeout_s,
                    talker_params=talker_params,
                )
                logger.info(
                    f"[{args.label}] WARMUP prompt={prompt_id} repeat={warm} "
                    f"ttft={ttft:.3f}s total={total:.3f}s "
                    f"audio_chunks={audio_chunks} status_code={status_code}"
                )

            ttfts: list[float] = []
            totals: list[float] = []
            for repeat in range(args.repeats):
                hint = f"{args.label}-{prompt_id}-{repeat}"
                ttft, total, audio_chunks, status_code = await _measure_one(
                    client,
                    args.base_url,
                    args.model,
                    prompt_text,
                    request_id_hint=hint,
                    seed=args.seed,
                    timeout_s=args.timeout_s,
                    talker_params=talker_params,
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
                    f"[{args.label}] prompt={prompt_id} repeat={repeat} "
                    f"ttft={ttft:.3f}s total={total:.3f}s audio_chunks={audio_chunks}"
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
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_STREAMING_TTFT_SEED,
        help="Sampler seed used for warmup and every measured repeat.",
    )
    add_talker_sampling_arguments(parser)
    add_fingerprint_argument(parser)
    parser.add_argument("--timeout-s", type=float, default=300.0)
    args = parser.parse_args(argv)
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    if args.warmup < 0:
        parser.error("--warmup must be non-negative")
    if args.output is None:
        args.output = _default_output_path(args.label)
    elif args.output.exists():
        raise FileExistsError(f"output path already exists: {args.output}")

    wait_for_service(args.base_url)
    summary = asyncio.run(_run(args))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    document = {
        "label": summary.label,
        "base_url": summary.base_url,
        "seed": summary.seed,
        "talker_temperature": summary.talker_temperature,
        "talker_top_p": summary.talker_top_p,
        "talker_top_k": summary.talker_top_k,
        "talker_repetition_penalty": summary.talker_repetition_penalty,
        "per_run": [asdict(run) for run in summary.per_run],
        "aggregate": summary.aggregate,
        **fingerprint_fields(args.fingerprint, args.base_url),
    }
    args.output.write_text(json.dumps(document, indent=2))
    _print_summary(summary)
    logger.info(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
