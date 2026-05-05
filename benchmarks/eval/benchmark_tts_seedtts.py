# SPDX-License-Identifier: Apache-2.0
"""SeedTTS benchmark for TTS models with performance and WER metrics.

Note (Qiujiang, Chenyang):

1. Voice-clone models (e.g. fishaudio/s2-pro): default uses ref_audio /
  ref_text from the meta file.
2. Plain TTS (e.g. mistralai/Voxtral-4B-TTS-2603): use --no-ref-audio and
  --voice for a server-side speaker preset.

Usage:

    # Download the test set:
    python -m benchmarks.dataset.prepare --dataset seedtts

    # Launch the server:
    1. For S2-Pro:
    python -m sglang_omni.cli serve \
        --model-path fishaudio/s2-pro \
        --port 8000

    2. For Voxtral-4B-TTS-2603:
    python -m sglang_omni.cli serve \
        --model-path mistralai/Voxtral-4B-TTS-2603 \
        --port 8000

    # Full pipeline (generate + transcribe) — voice cloning
    python -m benchmarks.eval.benchmark_tts_seedtts \
        --meta seedtts_testset/en/meta.lst \
        --max-concurrency 16 \
        --model fishaudio/s2-pro --port 8000

    # Full pipeline — plain TTS (no ref audio from testset)
    python -m benchmarks.eval.benchmark_tts_seedtts \
        --meta seedtts_testset/en/meta.lst \
        --model mistralai/Voxtral-4B-TTS-2603 --port 8000 \
        --max-concurrency 16 \
        --no-ref-audio --voice cheerful_female --max-samples 50

For CI settings, separate the generate and transcribe phases into two runs.

Usage (CI):

    # Generate audio only
    python -m benchmarks.eval.benchmark_tts_seedtts \
        --generate-only \
        --meta seedtts_testset/en/meta.lst \
        --max-concurrency 16 \
        --output-dir results/s2pro_en \
        --model fishaudio/s2-pro --port 8000

    # Transcribe + WER only
    python -m benchmarks.eval.benchmark_tts_seedtts \
        --transcribe-only \
        --meta seedtts_testset/en/meta.lst \
        --model fishaudio/s2-pro \
        --output-dir results/s2pro_en \
        --lang en --device cuda:0


H200 Full-Set Reference Results

Reproducibility references for the FULL eval set — NOT CI thresholds.
CI runs on a subset and has its own thresholds elsewhere (see tasks/*.py).

Benchmark: SeedTTS  |  Dataset: seed-tts-eval, full set (EN=1088, ZH=2020)
Hardware:  1 x H200 (default; non-H200 sources are tagged in Source column)
Last verified: 2026-05-04

Accuracy (accuracy.wer)

| Model  | Config           | wer_corpus | wer_per_sample_mean | wer_per_sample_median | wer_per_sample_std | evaluated | skipped | Source                         |
| ------ | ---------------- | ---------- | ------------------- | --------------------- | ------------------ | --------- | ------- | ------------------------------ |
| S2-Pro | EN, stream=False | 1.16%      | 1.11%               | 0.00%                 | 3.7%               | 1088/1088 | 0       | PR #393 [H200, full-set, c=16] |
| S2-Pro | EN, stream=True  | 1.06%      | 1.00%               | 0.00%                 | 3.4%               | 1088/1088 | 0       | PR #393 [H200, full-set, c=16] |
| S2-Pro | ZH, stream=False | 0.93%      | 0.89%               | 0.00%                 | 2.2%               | 2020/2020 | 0       | PR #393 [H200, full-set, c=16] |
| S2-Pro | ZH, stream=True  | 0.94%      | 0.89%               | 0.00%                 | 2.2%               | 2020/2020 | 0       | PR #393 [H200, full-set, c=16] |
| S2-Pro | EN, stream=False | 1.03%      | 0.98%               | 0.00%                 | 3.4%               | 1088/1088 | 0       | PR #351 [H100, full-set, c=16] |
| S2-Pro | EN, stream=True  | 0.98%      | 0.94%               | 0.00%                 | 3.3%               | 1088/1088 | 0       | PR #351 [H100, full-set, c=16] |
| S2-Pro | ZH, stream=False | 0.93%      | 0.89%               | 0.00%                 | 2.2%               | 2020/2020 | 0       | PR #351 [H100, full-set, c=16] |
| S2-Pro | ZH, stream=True  | 0.98%      | 0.94%               | 0.00%                 | 2.4%               | 2020/2020 | 0       | PR #351 [H100, full-set, c=16] |


Generation speed (generation.speed)

| Model  | Config           | latency_mean_s | latency_p95_s | rtf_mean | throughput_qps | tok_per_s_mean | tok_per_s_agg | Source                         |
| ------ | ---------------- | -------------- | ------------- | -------- | -------------- | -------------- | ------------- | ------------------------------ |
| S2-Pro | EN, stream=False | 17.016         | 28.582        | 4.501    | 0.937          | 45.3           | 42.7          | PR #393 [H200, full-set, c=16] |
| S2-Pro | EN, stream=True  | 16.189         | 27.789        | 4.297    | 0.984          | 46.7           | 44.8          | PR #393 [H200, full-set, c=16] |
| S2-Pro | ZH, stream=False | 12.428         | 21.057        | 2.318    | 1.282          | 51.7           | 50.4          | PR #393 [H200, full-set, c=16] |
| S2-Pro | ZH, stream=True  | 9.384          | 13.631        | 1.762    | 1.700          | 57.0           | 56.4          | PR #393 [H200, full-set, c=16] |
| S2-Pro | EN, stream=False | 9.38           | 14.65         | 2.48     | 1.700          | 56.6           | 56.0          | PR #351 [H100, full-set, c=16] |
| S2-Pro | EN, stream=True  | 9.92           | 15.49         | 2.62     | 1.607          | 53.9           | 53.2          | PR #351 [H100, full-set, c=16] |
| S2-Pro | ZH, stream=False | 9.64           | 13.61         | 1.80     | 1.655          | 55.7           | 55.2          | PR #351 [H100, full-set, c=16] |
| S2-Pro | ZH, stream=True  | 9.27           | 13.11         | 1.74     | 1.722          | 51.7           | 51.1          | PR #351 [H100, full-set, c=16] |

Note (Chenyang): tok_per_s_{mean,agg} here counts S2-Pro's codec tokens.  It is NOT
comparable to the tok_per_s column reported for Qwen3-Omni in benchmark_omni_seedtts.py,
whose tokens are discrete talker LM tokens emitted at audio frame rate. Cross-model
comparison of this column is not meaningful — use latency_mean_s / rtf_mean / throughput_qps
instead when comparing backends.

ASR speed (accuracy.asr_speed) — Whisper-large-v3 for EN, FunASR paraformer-zh for ZH

| Lang | asr_latency_mean_s | asr_rtf_mean | asr_throughput_samples_per_s | Source                                          |
| ---- | ------------------ | ------------ | ---------------------------- | ----------------------------------------------- |
| EN   | 0.297              | 0.0772       | 3.36                         | PR #393 [H200, from S2-Pro EN stream=False run] |
| ZH   | 0.294              | 0.0556       | 3.40                         | PR #393 [H200, from S2-Pro ZH stream=False run] |
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
from dataclasses import dataclass

from benchmarks.benchmarker.runner import BenchmarkRunner, RunConfig
from benchmarks.benchmarker.utils import wait_for_service
from benchmarks.dataset.seedtts import load_seedtts_samples
from benchmarks.metrics.performance import (
    build_speed_results,
    compute_speed_metrics,
    print_speed_summary,
)
from benchmarks.tasks.tts import (
    build_base_url,
    make_tts_send_fn,
    run_seedtts_transcribe,
    save_generated_audio_metadata,
    save_speed_results,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class TtsSeedttsBenchmarkConfig:
    model: str
    meta: str
    base_url: str | None = None
    host: str = "localhost"
    port: int = 8000
    # Optional speaker-preset name forwarded to the server as payload["voice"].
    # Voxtral-4B-TTS-2603 uses it to pick a built-in speaker (defaults to
    # "cheerful_female" server-side); voice-cloning models such as S2-Pro
    # ignore it and take the speaker from ref_audio/ref_text instead.
    voice: str | None = None
    # Default is voice-clone ON — S2-Pro's canonical flow uses the
    # seed-tts-eval reference audio.  The ``--no-ref-audio`` CLI flag flips
    # this to False for plain TTS models that do not accept ref audio.
    voice_clone: bool = True
    output_dir: str = "results/tts_seedtts"
    max_samples: int | None = None
    max_new_tokens: int | None = 2048
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    repetition_penalty: float | None = None
    warmup: int = 1
    concurrency: int = 1
    request_rate: float = float("inf")
    stream: bool = False
    disable_tqdm: bool = False
    # Transcribe phase
    lang: str = "en"
    device: str = "cuda:0"


def _build_generation_kwargs(config: TtsSeedttsBenchmarkConfig) -> dict:
    generation_kwargs: dict = {}
    if config.max_new_tokens is not None:
        generation_kwargs["max_new_tokens"] = config.max_new_tokens
    if config.temperature is not None:
        generation_kwargs["temperature"] = config.temperature
    if config.top_p is not None:
        generation_kwargs["top_p"] = config.top_p
    if config.top_k is not None:
        generation_kwargs["top_k"] = config.top_k
    if config.repetition_penalty is not None:
        generation_kwargs["repetition_penalty"] = config.repetition_penalty
    return generation_kwargs


def _build_results_config(
    config: TtsSeedttsBenchmarkConfig,
    *,
    base_url: str,
) -> dict:
    return {
        "model": config.model,
        "base_url": base_url,
        "meta": config.meta,
        "voice_clone": config.voice_clone,
        "voice": config.voice,
        "stream": config.stream,
        "max_samples": config.max_samples,
        "max_new_tokens": config.max_new_tokens,
        "warmup": config.warmup,
        "concurrency": config.concurrency,
        "request_rate": config.request_rate,
    }


async def run_tts_seedtts_benchmark(
    config: TtsSeedttsBenchmarkConfig,
) -> dict:
    """Generate audio and measure speed. Always saves audio for WER use.

    Returns a dict with keys: summary, per_request, config.
    """
    if not os.path.isfile(config.meta):
        raise FileNotFoundError(f"Meta file not found: {config.meta}")

    base_url = build_base_url(config)
    api_url = f"{base_url}/v1/audio/speech"

    samples = load_seedtts_samples(config.meta, config.max_samples)
    logger.info(f"Prepared {len(samples)} requests")

    save_audio_dir = os.path.abspath(os.path.join(config.output_dir, "audio"))
    os.makedirs(save_audio_dir, exist_ok=True)

    generation_kwargs = _build_generation_kwargs(config)
    send_fn = make_tts_send_fn(
        config.model,
        api_url,
        stream=config.stream,
        no_ref_audio=not config.voice_clone,
        voice=config.voice,
        save_audio_dir=save_audio_dir,
        **generation_kwargs,
    )

    runner = BenchmarkRunner(
        RunConfig(
            max_concurrency=config.concurrency,
            request_rate=config.request_rate,
            warmup=config.warmup,
            disable_tqdm=config.disable_tqdm,
        )
    )
    outputs = await runner.run(samples, send_fn)

    metrics = compute_speed_metrics(outputs, wall_clock_s=runner.wall_clock_s)
    results_config = _build_results_config(config, base_url=base_url)
    benchmark_results = build_speed_results(outputs, metrics, results_config)
    save_speed_results(outputs, metrics, results_config, config.output_dir)
    save_generated_audio_metadata(outputs, samples, config.output_dir)
    return benchmark_results


def run_tts_seedtts_transcribe(config: TtsSeedttsBenchmarkConfig) -> dict:
    """Transcribe saved audio and compute WER + ASR speed metrics.

    Server need not be running.

    Returns a dict with keys: wer_summary, asr_speed, per_sample.
    """
    generation_mode = "streaming" if config.stream else "non-streaming"
    wer_config = {
        "model": config.model,
        "meta": config.meta,
        "voice_clone": config.voice_clone,
        "voice": config.voice,
        "max_new_tokens": config.max_new_tokens,
        "temperature": config.temperature,
        "max_samples": config.max_samples,
        "stream": config.stream,
        "concurrency": config.concurrency,
    }
    return run_seedtts_transcribe(
        config,
        wer_config=wer_config,
        generation_mode=generation_mode,
    )


def _config_from_args(args: argparse.Namespace) -> TtsSeedttsBenchmarkConfig:
    # ``--no-ref-audio`` is preserved as a legacy CLI flag; it flips the
    # dataclass default (``voice_clone=True``) to False for plain TTS.
    voice_clone = not args.no_ref_audio
    return TtsSeedttsBenchmarkConfig(
        base_url=args.base_url,
        host=args.host,
        port=args.port,
        model=args.model,
        meta=args.meta,
        voice=args.voice,
        voice_clone=voice_clone,
        output_dir=args.output_dir,
        max_samples=args.max_samples,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        repetition_penalty=args.repetition_penalty,
        warmup=args.warmup,
        concurrency=args.concurrency,
        request_rate=args.request_rate,
        stream=args.stream,
        disable_tqdm=args.disable_tqdm,
        lang=args.lang,
        device=args.device,
    )


async def benchmark(config: TtsSeedttsBenchmarkConfig) -> dict:
    results = await run_tts_seedtts_benchmark(config)
    print_speed_summary(
        results["summary"], config.model, concurrency=config.concurrency
    )
    return results


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SeedTTS benchmark for TTS models.")
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
        default="fishaudio/s2-pro",
        help="Model name for the API request.",
    )
    parser.add_argument(
        "--voice",
        type=str,
        default=None,
        help=(
            "Built-in speaker-preset name for plain TTS models that select a "
            "voice server-side (e.g. mistralai/Voxtral-4B-TTS-2603 accepts "
            "'cheerful_female'). Has no effect on voice-cloning models such "
            "as fishaudio/s2-pro, which take the speaker from ref_audio in "
            "the meta file."
        ),
    )
    parser.add_argument(
        "--meta",
        "--testset",
        dest="meta",
        type=str,
        default="seedtts_testset/en/meta.lst",
        help="Path to a meta.lst file (seed-tts-eval format).",
    )
    parser.add_argument(
        "--no-ref-audio",
        dest="no_ref_audio",
        action="store_true",
        help="Skip ref audio/text from testset (TTS without voice cloning).",
    )
    parser.add_argument("--output-dir", type=str, default="results/tts_seedtts")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--repetition-penalty", type=float, default=None)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument(
        "--concurrency",
        "--max-concurrency",
        dest="concurrency",
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
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Use streaming SSE for TTS generation.",
    )
    parser.add_argument(
        "--save-audio",
        action="store_true",
        help="Legacy flag kept for backward compatibility. The unified "
        "benchmark always saves generated WAVs so the transcribe phase can "
        "reuse them; passing this flag is a no-op.",
    )
    parser.add_argument("--disable-tqdm", action="store_true")
    parser.add_argument(
        "--lang",
        type=str,
        choices=["en", "zh"],
        default="en",
        help="Language for ASR model (transcribe phase).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device for ASR model (transcribe phase).",
    )
    parser.add_argument(
        "--server-timeout",
        type=int,
        default=1200,
        help="Timeout in seconds to wait for server readiness.",
    )

    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--generate-only",
        action="store_true",
        help="Only synthesize audio and measure speed; skip WER transcription.",
    )
    mode.add_argument(
        "--transcribe-only",
        action="store_true",
        help="Only run ASR transcription and WER on existing output-dir.",
    )
    return parser


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()
    config = _config_from_args(args)

    if args.save_audio:
        logger.info("--save-audio is a no-op: the unified benchmark always saves WAVs.")

    if args.transcribe_only:
        run_tts_seedtts_transcribe(config)
        return

    wait_for_service(build_base_url(config), timeout=args.server_timeout)
    asyncio.run(benchmark(config))

    if args.generate_only:
        return

    run_tts_seedtts_transcribe(config)


if __name__ == "__main__":
    main()
