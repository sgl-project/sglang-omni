"""SeedTTS benchmark for TTS models with performance and WER metrics.

Note (Qiujiang, Chenyang):

1. Voice-clone models (e.g. fishaudio/s2-pro): default uses ref_audio /
  ref_text from the meta file.

2. Plain TTS (e.g. mistralai/Voxtral-4B-TTS-2603): use --no-ref-audio and
  --voice for a server-side speaker preset.

Usage:

1. Download the test set:

    python -m benchmarks.dataset.prepare --dataset seedtts

2. Full pipeline (auto start TTS → generate → stop TTS → start ASR → WER):


    python -m benchmarks.eval.benchmark_tts_seedtts \
        --meta zhaochenyang20/seed-tts-eval-arrow \
        --max-concurrency 16 \
        --model fishaudio/s2-pro \
        --port 8000

    python -m benchmarks.eval.benchmark_tts_seedtts \
        --meta zhaochenyang20/seed-tts-eval-arrow \
        --model mistralai/Voxtral-4B-TTS-2603 --port 8000 \
        --max-concurrency 16 \
        --no-ref-audio --voice cheerful_female

    python -m benchmarks.eval.benchmark_tts_seedtts \
        --meta zhaochenyang20/seed-tts-eval-arrow \
        --model bosonai/higgs-audio-v3-tts-4b --port 8000 \
        --ref-format references \
        --output-dir results/higgs_tts_en \
        --lang en --max-concurrency 16

    python -m benchmarks.eval.benchmark_tts_seedtts \
        --meta zhaochenyang20/seed-tts-eval-arrow \
        --model OpenMOSS-Team/MOSS-TTS-v1.5 --port 8000 \
        --ref-format references \
        --token-count auto \
        --output-dir results/moss_tts_en \
        --lang en --max-concurrency 16

    python -m benchmarks.eval.benchmark_tts_seedtts \
        --meta zhaochenyang20/seed-tts-eval-arrow \
        --model OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5 --port 8000 \
        --ref-format references \
        --token-count auto \
        --output-dir results/moss_tts_en \
        --lang en --max-concurrency 16


3. For CI settings, separate the generate and transcribe phases into two runs.

Usage (CI):

    # Generate audio only

    python -m benchmarks.eval.benchmark_tts_seedtts \
        --generate-only \
        --meta zhaochenyang20/seed-tts-eval-arrow \
        --max-concurrency 16 \
        --output-dir results/s2pro_en \
        --model fishaudio/s2-pro \
        --port 8000

    # Transcribe + WER only

    python -m benchmarks.eval.benchmark_tts_seedtts \
        --transcribe-only \
        --meta zhaochenyang20/seed-tts-eval-arrow \
        --model fishaudio/s2-pro \
        --output-dir results/s2pro_en \
        --lang en --port 8000
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import math
import os
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

from benchmarks.benchmarker.data import RequestResult
from benchmarks.benchmarker.runner import BenchmarkRunner, RunConfig
from benchmarks.benchmarker.utils import managed_omni_server
from benchmarks.dataset.seedtts import SampleInput, load_seedtts_samples
from benchmarks.metrics.performance import (
    build_speed_results,
    compute_speed_metrics,
    print_speed_summary,
)
from benchmarks.tasks.asr import (
    DEFAULT_ASR_TRANSCRIBE_CONCURRENCY,
    QWEN3_ASR_MODEL_PATH,
)
from benchmarks.tasks.tts import (
    MOSS_TTS_TOKEN_COUNT_AUTO,
    build_base_url,
    make_tts_send_fn,
    run_seedtts_similarity,
    run_seedtts_transcribe,
    run_seedtts_utmos,
    save_generated_audio_metadata,
    save_speed_results,
)
from sglang_omni.admission import QueueFullError

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

DEFAULT_TTS_BENCHMARK_CONCURRENCY = int(os.getenv("TTS_BENCHMARK_CONCURRENCY", "16"))


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
    task_type: str | None = None
    instructions: str | None = None
    # Default is voice-clone ON — S2-Pro's canonical flow uses the
    # seed-tts-eval reference audio.  The ``--no-ref-audio`` CLI flag flips
    # this to False for plain TTS models that do not accept ref audio.
    voice_clone: bool = True
    # Reference payload shape for voice cloning. The default keeps the original
    # ref_audio/ref_text fields; Higgs TTS should pass --ref-format references.
    ref_format: str = "flat"
    # Keeps ref_audio but drops ref_text/references[].text — for cross-lingual
    # runs where the reference speaker is cloned but no reference transcript is
    # sent. Only meaningful when voice_clone=True; ignored otherwise.
    no_ref_text: bool = False
    response_format: str = "wav"
    output_dir: str = "results/tts_seedtts"
    max_samples: int | None = None
    # Note (Yueying Li): skip this many samples before taking max_samples — lets N concurrent
    # clients replay DISJOINT dataset shards (offset i*max_samples) so shared
    # radix/fingerprint caches don't inflate multi-client throughput.
    sample_offset: int = 0
    max_new_tokens: int | None = 2048
    token_count: int | str | None = None
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    repetition_penalty: float | None = None
    seed: int | None = None
    warmup: int = 1
    concurrency: int = DEFAULT_TTS_BENCHMARK_CONCURRENCY
    request_rate: float = float("inf")
    stream: bool = False
    initial_codec_chunk_frames: int | None = None
    disable_tqdm: bool = False
    max_running_requests: int = 64
    max_queued_requests: int | None = None
    overshoot_duration_s: float = 10.0
    cuda_graph_max_bs: int = 64
    # note (luojiaxuan): optional sglang-omni pipeline config yaml forwarded
    # to the managed TTS server as ``--config`` (e.g.
    # examples/configs/dots_tts.yaml to run the canonical optimized
    # deployment).
    server_config: str | None = None
    # SGLang quantization mode (e.g. fp8) forwarded to the TTS generation
    # stage; recorded as run provenance so bf16 vs fp8 archives are
    # distinguishable. None = bf16.
    quantization: str | None = None
    # Transcribe phase
    lang: str = "en"
    device: str = "cuda:0"
    similarity_checkpoint: str | None = None
    asr_model_path: str = QWEN3_ASR_MODEL_PATH
    asr_concurrency: int = DEFAULT_ASR_TRANSCRIBE_CONCURRENCY


def _build_generation_kwargs(config: TtsSeedttsBenchmarkConfig) -> dict:
    generation_kwargs: dict = {}
    if config.max_new_tokens is not None:
        generation_kwargs["max_new_tokens"] = config.max_new_tokens
    if config.token_count is not None:
        generation_kwargs["token_count"] = config.token_count
    if config.temperature is not None:
        generation_kwargs["temperature"] = config.temperature
    if config.top_p is not None:
        generation_kwargs["top_p"] = config.top_p
    if config.top_k is not None:
        generation_kwargs["top_k"] = config.top_k
    if config.repetition_penalty is not None:
        generation_kwargs["repetition_penalty"] = config.repetition_penalty
    if config.seed is not None:
        generation_kwargs["seed"] = config.seed
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
        "ref_format": config.ref_format,
        "no_ref_text": config.no_ref_text,
        "response_format": config.response_format,
        "voice": config.voice,
        "task_type": config.task_type,
        "instructions": config.instructions,
        "stream": config.stream,
        "max_samples": config.max_samples,
        "sample_offset": config.sample_offset,
        "max_new_tokens": config.max_new_tokens,
        "seed": config.seed,
        "token_count": config.token_count,
        "warmup": config.warmup,
        "concurrency": config.concurrency,
        "request_rate": config.request_rate,
        "initial_codec_chunk_frames": config.initial_codec_chunk_frames,
        "max_running_requests": config.max_running_requests,
        "max_queued_requests": config.max_queued_requests,
        "overshoot_duration_s": config.overshoot_duration_s,
        "cuda_graph_max_bs": config.cuda_graph_max_bs,
        "server_config": config.server_config,
        "quantization": config.quantization,
    }


def _load_benchmark_samples(config: TtsSeedttsBenchmarkConfig) -> list[SampleInput]:
    # Note (Jiaxin Deng): a negative offset would silently slice from the end
    # instead of skipping the first N, contaminating the shard it claims to take.
    if config.sample_offset < 0:
        raise ValueError(
            f"--sample-offset must be non-negative, got {config.sample_offset}"
        )
    if config.sample_offset:
        head = config.sample_offset + (config.max_samples or 0)
        return load_seedtts_samples(
            config.meta, head if config.max_samples else None, split=config.lang
        )[config.sample_offset :]
    return load_seedtts_samples(config.meta, config.max_samples, split=config.lang)


async def run_tts_seedtts_benchmark(
    config: TtsSeedttsBenchmarkConfig,
    *,
    samples: list[SampleInput] | None = None,
    save_audio: bool = True,
) -> dict:
    """Generate audio and measure speed.

    Saves audio by default so the transcribe phase can reuse it.
    """
    base_url = build_base_url(config)
    api_url = f"{base_url}/v1/audio/speech"
    if samples is None:
        samples = _load_benchmark_samples(config)
    logger.info(f"Prepared {len(samples)} requests (offset {config.sample_offset})")

    save_audio_dir = None
    if save_audio:
        save_audio_dir = os.path.abspath(os.path.join(config.output_dir, "audio"))
        os.makedirs(save_audio_dir, exist_ok=True)
    else:
        os.makedirs(config.output_dir, exist_ok=True)

    generation_kwargs = _build_generation_kwargs(config)
    send_fn = make_tts_send_fn(
        config.model,
        api_url,
        response_format=config.response_format,
        stream=config.stream,
        initial_codec_chunk_frames=config.initial_codec_chunk_frames,
        no_ref_audio=not config.voice_clone,
        ref_format=config.ref_format,
        no_ref_text=config.no_ref_text,
        voice=config.voice,
        task_type=config.task_type,
        instructions=config.instructions,
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


def run_tts_seedtts_transcribe(
    config: TtsSeedttsBenchmarkConfig,
    *,
    asr_router_port: int | None = None,
) -> dict:
    """Transcribe saved audio and compute WER + ASR speed metrics.

    Server need not be running.

    Returns a dict with keys: wer_summary, asr_speed, per_sample.
    """
    generation_mode = "streaming-audio" if config.stream else "non-streaming"
    wer_config = {
        "model": config.model,
        "tts_model": config.model,
        "asr_model": config.asr_model_path,
        "meta": config.meta,
        "voice_clone": config.voice_clone,
        "ref_format": config.ref_format,
        "no_ref_text": config.no_ref_text,
        "response_format": config.response_format,
        "voice": config.voice,
        "task_type": config.task_type,
        "instructions": config.instructions,
        "max_new_tokens": config.max_new_tokens,
        "token_count": config.token_count,
        "temperature": config.temperature,
        "max_samples": config.max_samples,
        "stream": config.stream,
        "initial_codec_chunk_frames": config.initial_codec_chunk_frames,
        "concurrency": config.concurrency,
        "asr_concurrency": config.asr_concurrency,
        "quantization": config.quantization,
    }
    return run_seedtts_transcribe(
        config,
        wer_config=wer_config,
        generation_mode=generation_mode,
        asr_router_port=asr_router_port,
    )


def _config_from_args(args: argparse.Namespace) -> TtsSeedttsBenchmarkConfig:
    # ``--no-ref-audio`` is preserved as a legacy CLI flag; it flips the
    # dataclass default (``voice_clone=True``) to False for plain TTS.
    voice_clone = not args.no_ref_audio
    response_format = "pcm" if args.stream else args.response_format
    return TtsSeedttsBenchmarkConfig(
        base_url=args.base_url,
        host=args.host,
        port=args.port,
        model=args.model,
        meta=args.meta,
        voice=args.voice,
        task_type=args.task_type,
        instructions=args.instructions,
        voice_clone=voice_clone,
        ref_format=args.ref_format,
        no_ref_text=args.no_ref_text,
        response_format=response_format,
        output_dir=args.output_dir,
        max_samples=args.max_samples,
        sample_offset=args.sample_offset,
        max_new_tokens=args.max_new_tokens,
        token_count=args.token_count,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        repetition_penalty=args.repetition_penalty,
        seed=args.seed,
        warmup=args.warmup,
        concurrency=args.concurrency,
        request_rate=args.request_rate,
        stream=args.stream,
        initial_codec_chunk_frames=args.initial_codec_chunk_frames,
        disable_tqdm=args.disable_tqdm,
        max_running_requests=args.max_running_requests,
        max_queued_requests=args.max_queued_requests,
        overshoot_duration_s=args.overshoot_duration_s,
        cuda_graph_max_bs=args.cuda_graph_max_bs,
        server_config=args.server_config,
        quantization=args.quantization,
        lang=args.lang,
        device=args.device,
        similarity_checkpoint=args.similarity_checkpoint,
        asr_model_path=args.asr_model_path,
        asr_concurrency=args.asr_concurrency,
    )


def _parse_token_count(value: str) -> int | str:
    normalized = value.strip().lower()
    if normalized == MOSS_TTS_TOKEN_COUNT_AUTO:
        return MOSS_TTS_TOKEN_COUNT_AUTO
    try:
        token_count = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "token count must be a positive integer or 'auto'"
        ) from exc
    if token_count <= 0:
        raise argparse.ArgumentTypeError("token count must be positive")
    return token_count


def _parse_concurrencies(value: str) -> list[int]:
    tokens = [token.strip() for token in value.split(",") if token.strip()]
    if not tokens:
        raise argparse.ArgumentTypeError(
            "concurrencies must be a non-empty comma-separated list"
        )
    values: list[int] = []
    for token in tokens:
        try:
            parsed = int(token)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"invalid concurrency {token!r}") from exc
        if parsed <= 0:
            raise argparse.ArgumentTypeError("concurrency must be > 0")
        values.append(parsed)
    return values


@dataclass(frozen=True)
class SustainedOvershootPlan:
    capacity: int
    request_rate: float
    duration_s: float
    request_count: int
    concurrency: int
    overshoot_ratio: float


def plan_sustained_overshoot(
    *,
    max_running_requests: int,
    max_queued_requests: int,
    duration_s: float,
    request_rate: float | None = None,
    overshoot_factor: float = 2.0,
) -> SustainedOvershootPlan:
    """Open-loop arrivals above ``running + queued`` for ``duration_s``."""
    if max_running_requests < 1 or max_queued_requests < 1:
        raise ValueError("max_running_requests and max_queued_requests must be >= 1")
    if duration_s <= 0 or overshoot_factor <= 1:
        raise ValueError("overshoot duration_s must be positive and factor > 1")
    capacity = max_running_requests + max_queued_requests
    rate = overshoot_factor * capacity if request_rate is None else float(request_rate)
    if rate <= capacity:
        raise ValueError(
            "sustained overshoot requires request_rate > admission capacity "
            f"({rate} <= {capacity})"
        )
    return SustainedOvershootPlan(
        capacity=capacity,
        request_rate=rate,
        duration_s=float(duration_s),
        request_count=max(1, math.ceil(rate * duration_s)),
        concurrency=0,
        overshoot_ratio=rate / capacity,
    )


def expand_samples_for_overshoot(
    samples: list[SampleInput], request_count: int
) -> list[SampleInput]:
    if not samples or request_count < 1:
        raise ValueError("sustained overshoot requires samples and request_count >= 1")
    n = len(samples)
    expanded: list[SampleInput] = []
    for index in range(request_count):
        source = samples[index % n]
        expanded.append(replace(source, sample_id=f"{source.sample_id}#{index}"))
    return expanded


def classify_overshoot_outcomes(records: list[Any]) -> dict[str, Any]:
    success_ttfa: list[float] = []
    reject_latency: list[float] = []
    success = queue_full = other_failed = 0
    for record in records:
        if isinstance(record, RequestResult):
            ok, error, ttfa, latency = (
                record.is_success,
                record.error,
                record.audio_ttfp_s,
                record.latency_s,
            )
        else:
            ok = bool(record.get("is_success"))
            error = record.get("error")
            ttfa = record.get("audio_ttfp_s")
            latency = float(record.get("latency_s") or 0.0)
        if ok:
            success += 1
            if ttfa is not None:
                success_ttfa.append(float(ttfa))
        elif QueueFullError.matches(error):
            queue_full += 1
            reject_latency.append(float(latency))
        else:
            other_failed += 1
    return {
        "total_requests": len(records),
        "success": success,
        "queue_full": queue_full,
        "other_failed": other_failed,
        "success_ttfa_p95_s": _percentile(success_ttfa, 95),
        "queue_full_latency_p95_s": _percentile(reject_latency, 95),
    }


def _percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    rank = min(len(ordered) - 1, max(0, math.ceil(pct / 100.0 * len(ordered)) - 1))
    return ordered[rank]


async def run_tts_concurrency_sweep(
    config: TtsSeedttsBenchmarkConfig,
    concurrencies: list[int],
) -> dict[str, Any]:
    """Run generate-only once per concurrency and write a summary JSON."""
    rows: list[dict[str, Any]] = []
    for concurrency in concurrencies:
        point_output_dir = os.path.join(config.output_dir, f"c{concurrency}")
        point = replace(
            config,
            concurrency=concurrency,
            output_dir=point_output_dir,
        )
        print(f"[conc={concurrency}] generate pass")
        results = await run_tts_seedtts_benchmark(point)
        summary = results["summary"]
        success = int(summary.get("completed_requests") or 0)
        failed = int(summary.get("failed_requests") or 0)
        row = {
            "concurrency": concurrency,
            "output_dir": point_output_dir,
            "success": success,
            "failed": failed,
            "latency_p95_s": summary.get("latency_p95_s"),
            "audio_ttfp_p95_s": summary.get("audio_ttfp_p95_s"),
            "summary": summary,
        }
        rows.append(row)
        print_speed_summary(summary, config.model, concurrency=concurrency)
        print(
            f"  success={success} failed={failed} "
            f"latency_p95={row['latency_p95_s']} "
            f"ttfa_p95={row['audio_ttfp_p95_s']}"
        )

    payload = {
        "config": _build_results_config(config, base_url=build_base_url(config)),
        "concurrencies": concurrencies,
        "rows": rows,
    }
    out_path = os.path.join(config.output_dir, "concurrency_sweep.json")
    os.makedirs(config.output_dir, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    logger.info("Wrote concurrency sweep to %s", out_path)
    return payload


async def run_tts_sustained_overshoot(
    config: TtsSeedttsBenchmarkConfig,
) -> dict[str, Any]:
    """Hold offered load above admission capacity instead of stepping concurrency."""
    if config.max_queued_requests is None:
        raise ValueError("--sustained-overshoot requires --max-queued-requests")
    request_rate = None if config.request_rate == float("inf") else config.request_rate
    plan = plan_sustained_overshoot(
        max_running_requests=config.max_running_requests,
        max_queued_requests=config.max_queued_requests,
        duration_s=config.overshoot_duration_s,
        request_rate=request_rate,
    )
    point = replace(
        config,
        concurrency=plan.concurrency,
        request_rate=plan.request_rate,
        output_dir=os.path.join(config.output_dir, "overshoot"),
    )
    print(
        f"[overshoot] open-loop rate={plan.request_rate:.3f}/s "
        f"capacity={plan.capacity} duration={plan.duration_s}s "
        f"n={plan.request_count}"
    )
    corpus = _load_benchmark_samples(config)
    samples = expand_samples_for_overshoot(corpus, plan.request_count)
    results = await run_tts_seedtts_benchmark(point, samples=samples, save_audio=False)
    outcomes = classify_overshoot_outcomes(results["per_request"])
    payload = {
        "plan": asdict(plan),
        "outcomes": outcomes,
        "summary": results["summary"],
        "config": results["config"],
    }
    out_path = os.path.join(point.output_dir, "sustained_overshoot.json")
    os.makedirs(point.output_dir, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    logger.info("Wrote sustained overshoot to %s", out_path)
    print_speed_summary(results["summary"], config.model, concurrency=plan.concurrency)
    print(
        f"  success={outcomes['success']} queue_full={outcomes['queue_full']} "
        f"other_failed={outcomes['other_failed']} "
        f"ttfa_p95={outcomes['success_ttfa_p95_s']} "
        f"reject_p95={outcomes['queue_full_latency_p95_s']}"
    )
    return payload


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
        "--task-type",
        type=str,
        default=None,
        help="Model-specific TTS task type, for example Base, CustomVoice, or VoiceDesign.",
    )
    parser.add_argument(
        "--instructions",
        type=str,
        default=None,
        help="Model-specific style or voice-design instructions.",
    )
    parser.add_argument(
        "--meta",
        "--testset",
        dest="meta",
        type=str,
        default="zhaochenyang20/seed-tts-eval-arrow",
        help="HuggingFace Arrow/Parquet dataset repo id or local meta.lst path.",
    )
    parser.add_argument(
        "--no-ref-audio",
        dest="no_ref_audio",
        action="store_true",
        help="Skip ref audio/text from testset (TTS without voice cloning).",
    )
    parser.add_argument(
        "--no-ref-text",
        dest="no_ref_text",
        action="store_true",
        help=(
            "Keep ref_audio (voice cloning) but drop ref_text/references[].text "
            "from the request, for cross-lingual-style runs. Ignored when "
            "--no-ref-audio is also set."
        ),
    )
    parser.add_argument(
        "--ref-format",
        choices=["flat", "references"],
        default="flat",
        help=(
            "Reference payload shape for voice cloning. The default 'flat' sends "
            "ref_audio/ref_text, preserving the original behavior for S2-Pro "
            "and similar models. Use 'references' for Higgs TTS."
        ),
    )
    parser.add_argument(
        "--response-format",
        type=str,
        default="wav",
        help=(
            "Requested audio payload format. Streaming always sends "
            "response_format=pcm."
        ),
    )
    parser.add_argument("--output-dir", type=str, default="results/tts_seedtts")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--sample-offset", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument(
        "--token-count",
        type=_parse_token_count,
        default=None,
        help=(
            "MOSS-TTS duration token target forwarded as token_count. Pass "
            "'auto' to estimate per sample using OpenMOSS app defaults."
        ),
    )
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--repetition-penalty", type=float, default=None)
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Per-request sampler seed for reproducible generation.",
    )
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument(
        "--concurrency",
        "--max-concurrency",
        dest="concurrency",
        type=int,
        default=DEFAULT_TTS_BENCHMARK_CONCURRENCY,
        help="Maximum concurrent requests.",
    )
    parser.add_argument(
        "--concurrencies",
        type=_parse_concurrencies,
        default=None,
        help="Comma-separated concurrency levels to sweep (requires --generate-only).",
    )
    parser.add_argument(
        "--sustained-overshoot",
        action="store_true",
        help="Open-loop soak above running+queued (needs --generate-only and --max-queued-requests).",
    )
    parser.add_argument(
        "--overshoot-duration-s",
        type=float,
        default=10.0,
        help="Wall-clock seconds to keep offering overshoot load.",
    )
    parser.add_argument(
        "--request-rate",
        type=float,
        default=float("inf"),
        help="Requests/s (inf = burst). Soak defaults to 2x capacity if omitted.",
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Use streaming for TTS generation.",
    )
    parser.add_argument(
        "--initial-codec-chunk-frames",
        type=int,
        default=None,
        help=(
            "Optional model-specific first codec chunk size. With Higgs TTS "
            "this controls only the first streaming vocoder chunk."
        ),
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
        "--asr-model-path",
        type=str,
        default=QWEN3_ASR_MODEL_PATH,
        help="HuggingFace model id for the ASR server started in the "
        f"transcribe phase. Defaults to {QWEN3_ASR_MODEL_PATH}; "
        "openai/whisper-large-v3 can also be used.",
    )
    parser.add_argument(
        "--asr-concurrency",
        type=int,
        default=DEFAULT_ASR_TRANSCRIBE_CONCURRENCY,
        help="Concurrent transcription requests during WER evaluation.",
    )
    parser.add_argument(
        "--similarity-checkpoint",
        type=str,
        default=None,
        help="Optional path to a custom fine-tuned WavLM checkpoint. "
        "If omitted, the official weights are downloaded into a local cache "
        "directory (override the cache root with SEEDTTS_SIM_CACHE_DIR).",
    )
    parser.add_argument(
        "--server-timeout",
        type=int,
        default=1200,
        help="Timeout in seconds to wait for server readiness.",
    )
    parser.add_argument(
        "--max-running-requests",
        type=int,
        default=64,
        help=(
            "SGLang generation stage max_running_requests for the server "
            "started by this benchmark. Recommended to keep equal to "
            "--cuda-graph-max-bs. Defaults to 64."
        ),
    )
    parser.add_argument(
        "--max-queued-requests",
        type=int,
        default=None,
        help=(
            "SGLang generation stage max_queued_requests for the managed "
            "server. Omit to leave the pipeline default."
        ),
    )
    parser.add_argument(
        "--cuda-graph-max-bs",
        type=int,
        default=64,
        help=(
            "SGLang generation stage cuda_graph_max_bs for the server "
            "started by this benchmark. Recommended to keep equal to "
            "--max-running-requests. Defaults to 64."
        ),
    )
    parser.add_argument(
        "--server-config",
        type=str,
        default=None,
        help=(
            "Optional sglang-omni pipeline config yaml passed to the managed "
            "TTS server as --config (e.g. examples/configs/dots_tts.yaml for "
            "the canonical optimized dots.tts deployment). Ignored with "
            "--use-existing-server."
        ),
    )
    parser.add_argument(
        "--quantization",
        type=str,
        default=None,
        help=(
            "SGLang quantization mode (e.g. fp8) for the TTS generation stage "
            "of the server started by this benchmark. The ASR server is always "
            "left unquantized. Defaults to none (bf16)."
        ),
    )
    parser.add_argument(
        "--skip-gpu-cleanup",
        action="store_true",
        help=(
            "Do not run ensure_gpus_idle after stopping a server. Use when "
            "running multiple benchmark processes in parallel on different "
            "GPUs; combine with CUDA_VISIBLE_DEVICES per worker and clean up "
            "each GPU once after the worker finishes."
        ),
    )
    parser.add_argument(
        "--use-existing-server",
        action="store_true",
        help=(
            "Do not start or stop a server; send requests to the configured "
            "--base-url or --host/--port instead."
        ),
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
    mode.add_argument(
        "--similarity-only",
        action="store_true",
        help="Only run speaker similarity on existing output-dir.",
    )
    mode.add_argument(
        "--utmos-only",
        action="store_true",
        help="Only run UTMOS MOS scoring on existing output-dir.",
    )
    return parser


def _validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if (
        args.initial_codec_chunk_frames is not None
        and args.initial_codec_chunk_frames < 0
    ):
        parser.error("--initial-codec-chunk-frames must be non-negative")
    if args.max_running_requests <= 0:
        parser.error("--max-running-requests must be positive")
    if args.max_queued_requests is not None and args.max_queued_requests < 1:
        parser.error("--max-queued-requests must be >= 1")
    if args.cuda_graph_max_bs <= 0:
        parser.error("--cuda-graph-max-bs must be positive")
    if args.concurrencies is not None and not args.generate_only:
        parser.error("--concurrencies currently requires --generate-only")
    if args.sustained_overshoot and not args.generate_only:
        parser.error("--sustained-overshoot currently requires --generate-only")
    if args.sustained_overshoot and args.concurrencies is not None:
        parser.error("--sustained-overshoot cannot be combined with --concurrencies")
    if args.sustained_overshoot and args.max_queued_requests is None:
        parser.error("--sustained-overshoot requires --max-queued-requests")
    if args.overshoot_duration_s <= 0:
        parser.error("--overshoot-duration-s must be positive")
    if args.sustained_overshoot and args.request_rate != float("inf"):
        try:
            plan_sustained_overshoot(
                max_running_requests=args.max_running_requests,
                max_queued_requests=args.max_queued_requests,
                duration_s=args.overshoot_duration_s,
                request_rate=args.request_rate,
            )
        except ValueError as exc:
            parser.error(str(exc))
    if args.use_existing_server and not (args.generate_only or args.transcribe_only):
        parser.error(
            "--use-existing-server currently requires --generate-only or "
            "--transcribe-only"
        )


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()
    _validate_args(parser, args)
    config = _config_from_args(args)
    wait_for_gpu_release = not args.skip_gpu_cleanup

    if args.save_audio:
        logger.info("--save-audio is a no-op: the unified benchmark always saves WAVs.")

    if args.similarity_only:
        run_seedtts_similarity(config)
        return

    if args.utmos_only:
        run_seedtts_utmos(config, log_per_sample=True)
        return

    if args.transcribe_only:
        if args.use_existing_server:
            run_tts_seedtts_transcribe(config, asr_router_port=config.port)
        else:
            with managed_omni_server(
                model_path=config.asr_model_path,
                port=config.port,
                host=config.host,
                log_file=Path(config.output_dir) / "server_logs" / "asr_server.log",
                timeout=args.server_timeout,
                wait_for_gpu_release=wait_for_gpu_release,
            ):
                run_tts_seedtts_transcribe(config, asr_router_port=config.port)
        return

    async def _run_generate() -> None:
        if args.concurrencies is not None:
            await run_tts_concurrency_sweep(config, args.concurrencies)
        elif args.sustained_overshoot:
            await run_tts_sustained_overshoot(config)
        else:
            await benchmark(config)

    if args.use_existing_server:
        asyncio.run(_run_generate())
    else:
        with managed_omni_server(
            model_path=config.model,
            port=config.port,
            host=config.host,
            server_config=config.server_config,
            max_running_requests=config.max_running_requests,
            max_queued_requests=config.max_queued_requests,
            cuda_graph_max_bs=config.cuda_graph_max_bs,
            quantization=config.quantization,
            log_file=Path(config.output_dir) / "server_logs" / "tts_server.log",
            timeout=args.server_timeout,
            wait_for_gpu_release=wait_for_gpu_release,
        ):
            asyncio.run(_run_generate())

    if args.generate_only:
        return

    with managed_omni_server(
        model_path=config.asr_model_path,
        port=config.port,
        host=config.host,
        log_file=Path(config.output_dir) / "server_logs" / "asr_server.log",
        timeout=args.server_timeout,
        wait_for_gpu_release=wait_for_gpu_release,
    ):
        run_tts_seedtts_transcribe(config, asr_router_port=config.port)


if __name__ == "__main__":
    main()
