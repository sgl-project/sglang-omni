# SPDX-License-Identifier: Apache-2.0
"""MMMU benchmark for sglang-omni models.

Evaluates VLM accuracy and performance on the MMMU validation set via
/v1/chat/completions with image input.

Usage:
    # Text-only
    python benchmarks/eval/benchmark_omni_mmmu.py \
        --model qwen3-omni --port 8000 --max-samples 20

    # With concurrency
    python benchmarks/eval/benchmark_omni_mmmu.py \
        --model qwen3-omni --port 8000 --max-samples 50 --max-concurrency 16

    # With audio (requires speech server)
    # Note (Yifei, Chenyang): Concurrency=1 only for now since code_predictor and
    # code2wav modules serialize GPU access, so they run serially even when
    # concurrency > 1. And, audio output is still slow at this stage.

    python benchmarks/eval/benchmark_omni_mmmu.py \
        --model qwen3-omni --port 8000 --max-samples 5 --enable-audio --max-tokens 50


H200 Full-Set Reference Results

Reproducibility references for the FULL eval set — NOT CI thresholds.
CI runs on a subset and has its own thresholds elsewhere (see tasks/*.py).

Benchmark: MMMU     |  Dataset: MMMU_val (900 samples, all 30 subjects)
Hardware:  1 x H200 (default; non-H200 sources are tagged in Source column)
Last verified: 2026-04-25

Accuracy (summary)

| Model      | Config             | accuracy | correct | failed | mc_fallback | Source                                                 |
| ---------- | ------------------ | -------- | ------- | ------ | ----------- | ------------------------------------------------------ |
| Qwen3-Omni | enable_audio=False | 67.22%   | 605/900 | 0      | 21          | PR #316 [H200, full-set, c=8, max_tokens=2048]         |
| Qwen3-Omni | enable_audio=True  | 46.00%   | 23/50   | 15     | 0           | PR #316 [H200, 50-sample subset, c=1, max_tokens=2048] |
| Qwen3-Omni | enable_audio=False | 66.11%   | 595/900 | 0      | 28          | PR #351 [H100, full-set, c=8, max_tokens=2048, text-only server] |
| Qwen3-Omni | enable_audio=True  | 18.00%   | 9/50    | 21     | 20          | PR #351 [H100, 50-sample subset, c=1, max_tokens=64, timeout=120s] |

Note (Xuesong): full 900 not runfor enable_audio = True — Issue #276 talker is c=1 only and ~2 min/sample (~30 h for full set). 15/50 requests failed
 in audio generation (Issue #276); on the 35 completed requests accuracy = 65.7%.

Speed (speed)

| Model      | Config             | latency_mean_s | latency_p95_s | throughput_qps | tok_per_s_mean | tok_per_s_agg | Source                                                     |
| ---------- | ------------------ | -------------- | ------------- | -------------- | -------------- | ------------- | ---------------------------------------------------------- |
| Qwen3-Omni | enable_audio=False | 25.70          | 96.38         | 0.308          | 19.6           | 19.9          | PR #316 [H200, full-set, c=8, max_tokens=2048]             |
| Qwen3-Omni | enable_audio=True  | 123.13         | 221.52        | 0.004          | 2.2            | 2.1           | PR #316 [H200, **50-sample subset**, c=1, max_tokens=2048] |
| Qwen3-Omni | enable_audio=False | 20.297         | 74.122        | 0.392          | 24.9           | 25.4          | PR #351 [H100, full-set, c=8, max_tokens=2048, text-only server] |
| Qwen3-Omni | enable_audio=True  | 19.579         | 23.147        | 0.009          | 3.3            | 3.3           | PR #351 [H100, 50-sample subset, c=1, max_tokens=64, timeout=120s] |
"""


from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmarks.benchmarker.runner import BenchmarkRunner, RunConfig
from benchmarks.benchmarker.utils import save_json_results, wait_for_service
from benchmarks.dataset.mmmu import load_mmmu_samples
from benchmarks.metrics.mmmu import compute_mmmu_metrics, print_mmmu_accuracy_summary
from benchmarks.metrics.performance import compute_speed_metrics, print_speed_summary
from benchmarks.metrics.wer import print_wer_summary
from benchmarks.tasks.tts import compute_text_audio_consistency
from benchmarks.tasks.visual_understand import (
    build_mmmu_result_records,
    make_mmmu_send_fn,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class MMMUEvalConfig:
    model: str
    base_url: str | None = None
    host: str = "localhost"
    port: int = 8000
    max_samples: int | None = None
    max_tokens: int = 2048
    temperature: float = 0.0
    output_dir: str | None = None
    max_concurrency: int = 1
    warmup: int = 0
    request_rate: float = float("inf")
    disable_tqdm: bool = False
    enable_audio: bool = False
    asr_device: str = "cuda:0"
    lang: str = "en"
    repo_id: str | None = None
    prompt_override: str | None = None
    timeout_s: int = 300


def _build_base_url(config: MMMUEvalConfig) -> str:
    return config.base_url or f"http://{config.host}:{config.port}"


async def run_mmmu_eval(config: MMMUEvalConfig) -> dict:
    """Run full MMMU evaluation and return results dict.

    Returns a dict with keys: summary, speed, config,
    per_sample, and wer (only when enable_audio is True).
    """
    base_url = _build_base_url(config)
    api_url = f"{base_url}/v1/chat/completions"

    samples = load_mmmu_samples(
        config.max_samples,
        repo_id=config.repo_id,
        instruction_override=config.prompt_override,
    )
    logger.info(f"Prepared {len(samples)} MMMU samples")

    audio_dir: str | None = None
    if config.enable_audio and config.output_dir:
        audio_dir = str(Path(config.output_dir) / "audio")
        Path(audio_dir).mkdir(parents=True, exist_ok=True)

    send_fn = make_mmmu_send_fn(
        config.model,
        api_url,
        max_tokens=config.max_tokens,
        temperature=config.temperature,
        enable_audio=config.enable_audio,
        audio_dir=audio_dir,
    )

    runner = BenchmarkRunner(
        RunConfig(
            max_concurrency=config.max_concurrency,
            request_rate=config.request_rate,
            warmup=config.warmup,
            disable_tqdm=config.disable_tqdm,
            timeout_s=config.timeout_s,
        )
    )
    request_results = await runner.run(samples, send_fn)

    per_sample = build_mmmu_result_records(samples, request_results)
    summary = compute_mmmu_metrics(per_sample)
    speed_metrics = compute_speed_metrics(
        request_results, wall_clock_s=runner.wall_clock_s
    )

    config_dict = {
        "model": config.model,
        "base_url": base_url,
        "max_samples": config.max_samples,
        "max_tokens": config.max_tokens,
        "temperature": config.temperature,
        "max_concurrency": config.max_concurrency,
        "warmup": config.warmup,
        "enable_audio": config.enable_audio,
    }

    results = {
        "summary": summary,
        "speed": speed_metrics,
        "config": config_dict,
        "per_sample": per_sample,
    }

    if config.enable_audio:
        results["wer"] = compute_text_audio_consistency(
            request_results, config.lang, config.asr_device
        )

    print_mmmu_accuracy_summary(summary, config.model)
    print_speed_summary(
        speed_metrics, config.model, config.max_concurrency, title="MMMU Speed"
    )
    if "wer" in results:
        print_wer_summary(results["wer"]["summary"], config.model)

    if config.output_dir:
        save_json_results(results, config.output_dir, "mmmu_results.json")

    return results


def _config_from_args(args: argparse.Namespace) -> MMMUEvalConfig:
    return MMMUEvalConfig(
        base_url=args.base_url,
        host=args.host,
        port=args.port,
        model=args.model,
        max_samples=args.max_samples,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        output_dir=args.output_dir,
        max_concurrency=args.max_concurrency,
        warmup=args.warmup,
        request_rate=args.request_rate,
        disable_tqdm=args.disable_tqdm,
        enable_audio=args.enable_audio,
        asr_device=args.asr_device,
        lang=args.lang,
        repo_id=args.repo_id,
    )


async def benchmark(args: argparse.Namespace) -> dict:
    config = _config_from_args(args)
    return await run_mmmu_eval(config)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="MMMU benchmark for VLM models served by sglang-omni."
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=None,
        help="Base URL (e.g. http://localhost:8000). Overrides --host/--port.",
    )
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--model",
        type=str,
        default="qwen3-omni",
        help="Model name for the API request.",
    )
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=1,
        help="Maximum concurrent requests.",
    )
    parser.add_argument(
        "--request-rate",
        type=float,
        default=float("inf"),
        help="Requests per second (inf = send all at once).",
    )
    parser.add_argument("--disable-tqdm", action="store_true")
    parser.add_argument(
        "--enable-audio",
        action="store_true",
        help="Request audio output and compute text-audio WER.",
    )
    parser.add_argument(
        "--asr-device",
        type=str,
        default="cuda:0",
        help="Device for ASR model (default: cuda:0).",
    )
    parser.add_argument(
        "--lang",
        choices=["en", "zh"],
        default="en",
        help="Language for ASR transcription (default: en).",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default=None,
        help="HuggingFace dataset repo (e.g. 'zhaochenyang20/mmmu-ci-50'). "
        "Defaults to loading the full MMMU/MMMU (all 30 subjects).",
    )
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = "results/mmmu_audio" if args.enable_audio else "results/mmmu"

    base_url = args.base_url or f"http://{args.host}:{args.port}"
    wait_for_service(base_url)

    asyncio.run(benchmark(args))


if __name__ == "__main__":
    main()
