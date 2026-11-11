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

4. Mixed load: 48 non streaming and 16 streaming closed loop clients at once for a 120 s window, then WER over every request in generated.json. See docs/developer_reference/tts_bench.md for sweeps.

    python -m benchmarks.eval.benchmark_tts_seedtts \
        --meta zhaochenyang20/seed-tts-eval-arrow \
        --model FunAudioLLM/Fun-CosyVoice3-0.5B-2512 \
        --mixture 48+16 \
        --output-dir results/cosyvoice3_mixture \
        --port 8000
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import math
import os
import re
from contextlib import nullcontext
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

from benchmarks.benchmarker.conditions import (
    ConcurrencyAggregate,
    RepeatSpeedSummary,
    add_fingerprint_argument,
    aggregate_repeats,
    collect_run_fingerprint,
    warn_if_tail_percentile_is_thin,
)
from benchmarks.benchmarker.data import RequestClass, RequestResult
from benchmarks.benchmarker.fingerprint import BenchmarkFingerprint
from benchmarks.benchmarker.runner import (
    BenchmarkRunner,
    ClientGroup,
    RunConfig,
    SendFn,
    resolve_warmup,
    run_fixed_duration,
)
from benchmarks.benchmarker.utils import managed_omni_server
from benchmarks.dataset.seedtts import SampleInput, load_seedtts_samples
from benchmarks.metrics.mixture import (
    MixtureLoad,
    MixtureRepeatResult,
    aggregate_mixture_sweep,
    compute_mixture_summaries,
    print_mixture_summary,
)
from benchmarks.metrics.performance import (
    build_speed_results,
    compute_speed_metrics,
    load_tts_speed_results,
    print_speed_summary,
)
from benchmarks.runtime_metrics import ResourceMonitor
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
    stream_outcome_collector,
)
from sglang_omni.admission import QueueFullError
from sglang_omni.utils.logging import configure_dependency_loggers

logger = logging.getLogger(__name__)

DEFAULT_TTS_BENCHMARK_CONCURRENCY = int(os.getenv("TTS_BENCHMARK_CONCURRENCY", "16"))


DEFAULT_MAX_NEW_TOKENS = 2048
DEFAULT_MIXTURE_DURATION_S = 120.0
DEFAULT_MIXTURE_SETTLE_S = 10.0
MIXTURE_LOAD_PATTERN = re.compile(r"([0-9]+)\+([0-9]+)")


@dataclass(frozen=True)
class _ModelBenchmarkProfile:
    """Per-checkpoint CLI defaults and managed-server knobs."""

    argument_defaults: dict[str, Any] = field(default_factory=dict)
    forward_sglang_engine: bool = True
    # note (Yucheng Hu): None omits max_new_tokens so the server's own default applies.
    max_new_tokens: int | None = DEFAULT_MAX_NEW_TOKENS


_AUK_BENCHMARK_PROFILE = _ModelBenchmarkProfile(
    argument_defaults={
        "output_dir": "results/auk_seedtts",
        "concurrency": 1,
        "warmup": 1,
        "seed": 1234,
    },
    forward_sglang_engine=False,
)
_MODEL_BENCHMARK_PROFILES: dict[str, _ModelBenchmarkProfile] = {
    **{
        checkpoint: _ModelBenchmarkProfile(max_new_tokens=None)
        for checkpoint in (
            "qwen3-tts-12hz-0.6b-base",
            "qwen3-tts-12hz-0.6b-customvoice",
            "qwen3-tts-12hz-1.7b-base",
            "qwen3-tts-12hz-1.7b-customvoice",
            "qwen3-tts-12hz-1.7b-voicedesign",
        )
    },
    "auk": _AUK_BENCHMARK_PROFILE,
    "auk-flash": _AUK_BENCHMARK_PROFILE,
    # note (Yucheng Hu): Fun-CosyVoice3 caps generation at min(2048, 20x target
    # text tokens) unless the request sets max_new_tokens; a flat 2048 lets a
    # runaway run 82 s.
    "fun-cosyvoice3-0.5b-2512": _ModelBenchmarkProfile(max_new_tokens=None),
}


def _model_checkpoint_name(model: str) -> str:
    return Path(model.split("@", 1)[0]).name.lower()


def _profile_for_model(model: str) -> _ModelBenchmarkProfile:
    return _MODEL_BENCHMARK_PROFILES.get(
        _model_checkpoint_name(model), _ModelBenchmarkProfile()
    )


def _parse_args(
    parser: argparse.ArgumentParser,
) -> tuple[argparse.Namespace, _ModelBenchmarkProfile]:
    args = parser.parse_args()
    profile = _profile_for_model(args.model)
    if not profile.argument_defaults:
        return args, profile
    parser.set_defaults(**profile.argument_defaults)
    return parser.parse_args(), profile


@dataclass
class TtsSeedttsBenchmarkConfig:
    model: str
    meta: str
    base_url: str | None = None
    host: str = "localhost"
    port: int = 8000
    voice: str | None = None
    task_type: str | None = None
    instructions: str | None = None
    voice_clone: bool = True
    ref_format: str = "flat"
    no_ref_text: bool = False
    response_format: str = "wav"
    output_dir: str = "results/tts_seedtts"
    max_samples: int | None = None
    # Note (Yueying Li): skip this many samples before taking max_samples — lets N concurrent
    # clients replay DISJOINT dataset shards (offset i*max_samples) so shared
    # radix/fingerprint caches don't inflate multi-client throughput.
    sample_offset: int = 0
    max_new_tokens: int | None = None
    token_count: int | str | None = None
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    repetition_penalty: float | None = None
    seed: int | None = None
    # Note (Shulei He): Fraction of requests sent with subtalker_dosample=True (rest use False),
    # deterministically alternated by sample index so runs are reproducible.
    # None leaves subtalker_dosample unset, i.e. the server-side default applies.
    subtalker_dosample_ratio: float | None = None
    warmup: int | None = None
    concurrency: int = DEFAULT_TTS_BENCHMARK_CONCURRENCY
    request_rate: float = float("inf")
    arrival_seed: int | None = None
    stream: bool = False
    collect_stream_outcomes: bool = True
    initial_codec_chunk_frames: int | None = None
    disable_tqdm: bool = False
    max_running_requests: int = 64
    max_queued_requests: int | None = None
    overshoot_duration_s: float = 10.0
    cuda_graph_max_bs: int = 64
    # note (luojiaxuan): optional sglang-omni pipeline config yaml forwarded
    # to the managed TTS server as --config e.g.
    # examples/configs/dots_tts.yaml to run the canonical optimized
    # deployment.
    server_config: str | None = None
    quantization: str | None = None
    lang: str = "en"
    device: str = "cuda:0"
    similarity_checkpoint: str | None = None
    asr_model_path: str = QWEN3_ASR_MODEL_PATH
    asr_concurrency: int = DEFAULT_ASR_TRANSCRIBE_CONCURRENCY
    environment_fingerprint: BenchmarkFingerprint | None = None
    measure_gpu_memory: bool = False
    mixture: MixtureLoad | None = None
    mixture_duration_s: float = DEFAULT_MIXTURE_DURATION_S
    mixture_settle_s: float = DEFAULT_MIXTURE_SETTLE_S


def resolve_max_new_tokens(config: TtsSeedttsBenchmarkConfig) -> int | None:
    # note (Yucheng Hu): resolved here rather than in _parse_args so a config built
    # without the CLI, as tests/test_model/test_tts_ci.py does, follows the profile.
    if config.max_new_tokens is not None:
        return config.max_new_tokens
    else:
        return _profile_for_model(config.model).max_new_tokens


def _build_generation_kwargs(config: TtsSeedttsBenchmarkConfig) -> dict:
    generation_kwargs: dict = {}
    max_new_tokens = resolve_max_new_tokens(config)
    if max_new_tokens is not None:
        generation_kwargs["max_new_tokens"] = max_new_tokens
    else:
        pass
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


def _resolve_warmup(config: TtsSeedttsBenchmarkConfig) -> int:
    return resolve_warmup(config.warmup, config.concurrency)


def _subtalker_dosample_flags(
    samples: list[SampleInput],
    ratio: float,
) -> dict[str, bool]:
    if not 0.0 <= ratio <= 1.0:
        raise ValueError(f"--subtalker-dosample-ratio must be in [0, 1], got {ratio}")
    flags: dict[str, bool] = {}
    for index, sample in enumerate(samples):
        dosample = math.floor((index + 1) * ratio) > math.floor(index * ratio)
        flags[sample.sample_id] = dosample
    return flags


def _build_results_config(
    config: TtsSeedttsBenchmarkConfig,
    *,
    base_url: str,
) -> dict:
    recorded = {
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
        "collect_stream_outcomes": config.stream and config.collect_stream_outcomes,
        "max_samples": config.max_samples,
        "sample_offset": config.sample_offset,
        "max_new_tokens": resolve_max_new_tokens(config),
        "temperature": config.temperature,
        "top_p": config.top_p,
        "top_k": config.top_k,
        "repetition_penalty": config.repetition_penalty,
        "seed": config.seed,
        "subtalker_dosample_ratio": config.subtalker_dosample_ratio,
        "token_count": config.token_count,
        "warmup": _resolve_warmup(config),
        "concurrency": config.concurrency,
        "request_rate": config.request_rate,
        "arrival_seed": config.arrival_seed,
        "initial_codec_chunk_frames": config.initial_codec_chunk_frames,
        "max_running_requests": config.max_running_requests,
        "max_queued_requests": config.max_queued_requests,
        "overshoot_duration_s": config.overshoot_duration_s,
        "cuda_graph_max_bs": config.cuda_graph_max_bs,
        "server_config": config.server_config,
        "quantization": config.quantization,
        "mixture": None,
    }
    if config.mixture is not None:
        recorded["mixture"] = {
            "non_streaming_clients": config.mixture.non_streaming_clients,
            "streaming_clients": config.mixture.streaming_clients,
            "settle_s": config.mixture_settle_s,
            "duration_s": config.mixture_duration_s,
        }
        recorded["stream"] = None
        recorded["collect_stream_outcomes"] = (
            config.mixture.streaming_clients > 0 and config.collect_stream_outcomes
        )
        recorded["concurrency"] = config.mixture.total_clients
        recorded["warmup"] = 0 if config.warmup == 0 else config.mixture.total_clients
    else:
        pass
    if config.environment_fingerprint is None:
        return recorded
    return {
        **recorded,
        "environment_fingerprint": config.environment_fingerprint,
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


def _make_subtalker_dosample_send_fn(
    config: TtsSeedttsBenchmarkConfig,
    api_url: str,
    common_send_kwargs: dict,
    generation_kwargs: dict,
    samples: list[SampleInput],
) -> SendFn:
    # Note (Shulei He): Qwen3-TTS-only: alternate subtalker_dosample per request for mixed
    # sampled/greedy predictor traffic.
    dosample_by_id = _subtalker_dosample_flags(samples, config.subtalker_dosample_ratio)
    send_fn_true = make_tts_send_fn(
        config.model,
        api_url,
        **common_send_kwargs,
        **generation_kwargs,
        stage_params={"tts_engine": {"subtalker_dosample": True}},
    )
    send_fn_false = make_tts_send_fn(
        config.model,
        api_url,
        **common_send_kwargs,
        **generation_kwargs,
        stage_params={"tts_engine": {"subtalker_dosample": False}},
    )
    send_fn_by_id = {
        sample_id: (send_fn_true if dosample else send_fn_false)
        for sample_id, dosample in dosample_by_id.items()
    }

    async def send_fn(session, sample):
        return await send_fn_by_id[sample.sample_id](session, sample)

    return send_fn


async def run_tts_seedtts_benchmark(
    config: TtsSeedttsBenchmarkConfig,
    *,
    samples: list[SampleInput] | None = None,
    save_audio: bool = True,
) -> dict:
    """Generate audio and measure speed.

    Saves audio by default so the transcribe phase can reuse it.
    """
    if config.mixture is not None:
        return await run_tts_mixture_benchmark(
            config, samples=samples, save_audio=save_audio
        )
    else:
        pass
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
    common_send_kwargs = dict(
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
    )
    if config.subtalker_dosample_ratio is None:
        send_fn = make_tts_send_fn(
            config.model, api_url, **common_send_kwargs, **generation_kwargs
        )
    else:
        send_fn = _make_subtalker_dosample_send_fn(
            config, api_url, common_send_kwargs, generation_kwargs, samples
        )

    runner = BenchmarkRunner(
        RunConfig(
            max_concurrency=config.concurrency,
            request_rate=config.request_rate,
            warmup=_resolve_warmup(config),
            disable_tqdm=config.disable_tqdm,
            arrival_seed=config.arrival_seed,
        )
    )
    monitor = ResourceMonitor().start() if config.measure_gpu_memory else None
    try:
        if config.stream and config.collect_stream_outcomes:
            async with stream_outcome_collector(api_url) as collect:
                outputs = await runner.run(samples, send_fn, after_send=collect)
        else:
            outputs = await runner.run(samples, send_fn)
    finally:
        resources = monitor.stop() if monitor is not None else None
    warn_if_tail_percentile_is_thin(len(outputs))

    metrics = compute_speed_metrics(outputs, wall_clock_s=runner.wall_clock_s)
    metrics["wall_clock_s"] = runner.wall_clock_s
    if resources is None:
        pass
    elif resources["available"]:
        metrics["gpu_memory_peak_mib"] = resources["gpu_memory_used_mib"]["max"]
    else:
        metrics["gpu_memory_peak_mib"] = None
        logger.warning(f"TTS GPU memory unavailable: {resources['error']}")
    results_config = _build_results_config(config, base_url=base_url)
    benchmark_results = build_speed_results(outputs, metrics, results_config)
    save_speed_results(outputs, metrics, results_config, config.output_dir)
    save_generated_audio_metadata(outputs, samples, config.output_dir)
    return benchmark_results


async def run_tts_mixture_benchmark(
    config: TtsSeedttsBenchmarkConfig,
    *,
    samples: list[SampleInput] | None = None,
    save_audio: bool = True,
) -> dict:
    """Run one NS+S load closed loop for a fixed window and save its speed results."""
    unsupported_settings = [
        name
        for name, is_set in (
            ("warmup other than 0", config.warmup not in (None, 0)),
            ("stream", config.stream),
            ("request_rate", config.request_rate != float("inf")),
            ("arrival_seed", config.arrival_seed is not None),
            ("subtalker_dosample_ratio", config.subtalker_dosample_ratio is not None),
        )
        if is_set
    ]
    if unsupported_settings:
        raise ValueError(
            f"a mixture load does not support {', '.join(unsupported_settings)}"
        )
    else:
        pass
    load = config.mixture
    base_url = build_base_url(config)
    api_url = f"{base_url}/v1/audio/speech"
    if samples is None:
        samples = _load_benchmark_samples(config)
    else:
        pass
    logger.info(f"Mixture {load.name} walks a pool of {len(samples)} samples")
    if (
        config.max_queued_requests is not None
        and load.total_clients
        > config.max_running_requests + config.max_queued_requests
    ):
        logger.warning(
            f"{load.total_clients} clients exceed running plus queued requests, "
            "so server rejections will count as failures"
        )
    else:
        pass

    if save_audio:
        save_audio_dir = os.path.abspath(os.path.join(config.output_dir, "audio"))
        os.makedirs(save_audio_dir, exist_ok=True)
    else:
        save_audio_dir = None
        os.makedirs(config.output_dir, exist_ok=True)
    generation_kwargs = _build_generation_kwargs(config)
    common_send_kwargs = dict(
        initial_codec_chunk_frames=config.initial_codec_chunk_frames,
        no_ref_audio=not config.voice_clone,
        ref_format=config.ref_format,
        no_ref_text=config.no_ref_text,
        voice=config.voice,
        task_type=config.task_type,
        instructions=config.instructions,
        save_audio_dir=save_audio_dir,
    )
    send_fns: dict[RequestClass, SendFn] = {
        "non_streaming": make_tts_send_fn(
            config.model,
            api_url,
            response_format=config.response_format,
            **common_send_kwargs,
            **generation_kwargs,
        ),
        "streaming": make_tts_send_fn(
            config.model,
            api_url,
            stream=True,
            **common_send_kwargs,
            **generation_kwargs,
        ),
    }
    first_sample_indexes: dict[RequestClass, int] = {
        "non_streaming": 0,
        "streaming": len(samples) // 2,
    }
    groups = [
        ClientGroup(
            request_class=request_class,
            clients=clients,
            send_fn=send_fns[request_class],
            first_sample_index=first_sample_indexes[request_class],
        )
        for request_class, clients in load.clients_by_class.items()
    ]
    if load.streaming_clients > 0 and config.collect_stream_outcomes:
        collector = stream_outcome_collector(api_url)
    else:
        collector = nullcontext(None)
    monitor = ResourceMonitor().start() if config.measure_gpu_memory else None
    try:
        async with collector as after_send:
            run = await run_fixed_duration(
                groups,
                samples,
                settle_s=config.mixture_settle_s,
                duration_s=config.mixture_duration_s,
                should_warm_up=config.warmup != 0,
                timeout_s=RunConfig().timeout_s,
                after_send=after_send,
            )
    finally:
        resources = monitor.stop() if monitor is not None else None

    summaries = compute_mixture_summaries(
        run.results,
        load=load,
        settle_s=config.mixture_settle_s,
        duration_s=config.mixture_duration_s,
        pool_size=len(samples),
    )
    for class_summary in summaries["summary_by_class"].values():
        warn_if_tail_percentile_is_thin(class_summary["total_requests"])
    metrics = summaries["summary"]
    if resources is None:
        pass
    elif resources["available"]:
        metrics["gpu_memory_peak_mib"] = resources["gpu_memory_used_mib"]["max"]
    else:
        metrics["gpu_memory_peak_mib"] = None
        logger.warning(f"TTS GPU memory unavailable: {resources['error']}")
    results_config = _build_results_config(config, base_url=base_url)
    benchmark_results = build_speed_results(
        run.results,
        metrics,
        results_config,
        summary_by_class=summaries["summary_by_class"],
        mixture=summaries["mixture"],
    )
    save_speed_results(
        run.results,
        metrics,
        results_config,
        config.output_dir,
        summary_by_class=summaries["summary_by_class"],
        mixture=summaries["mixture"],
    )
    save_generated_audio_metadata(run.results, run.issued_samples, config.output_dir)
    return benchmark_results


def run_tts_seedtts_transcribe(
    config: TtsSeedttsBenchmarkConfig,
    *,
    asr_router_port: int | None = None,
) -> dict:
    """Transcribe saved audio and compute WER + ASR speed metrics."""
    speed_results = load_tts_speed_results(config.output_dir)
    saved_config = {} if speed_results is None else speed_results["config"]
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
        "max_new_tokens": resolve_max_new_tokens(config),
        "token_count": config.token_count,
        "temperature": config.temperature,
        "top_p": config.top_p,
        "top_k": config.top_k,
        "repetition_penalty": config.repetition_penalty,
        "seed": config.seed,
        "max_samples": config.max_samples,
        "stream": config.stream,
        "collect_stream_outcomes": config.stream and config.collect_stream_outcomes,
        "initial_codec_chunk_frames": config.initial_codec_chunk_frames,
        "concurrency": config.concurrency,
        "asr_concurrency": config.asr_concurrency,
        "quantization": config.quantization,
        "mixture": None,
    }
    if saved_config.get("mixture") is not None:
        # note (db-ol): --transcribe-only after a mixture run has no --mixture, so the saved speed results tell how the audio was generated.
        generation_mode = "mixture"
        for name in ("stream", "collect_stream_outcomes", "concurrency", "mixture"):
            wer_config[name] = saved_config[name]
    elif config.stream:
        generation_mode = "streaming-audio"
    else:
        generation_mode = "non-streaming"
    return run_seedtts_transcribe(
        config,
        wer_config=wer_config,
        generation_mode=generation_mode,
        asr_router_port=asr_router_port,
    )


def _config_from_args(args: argparse.Namespace) -> TtsSeedttsBenchmarkConfig:
    voice_clone = not args.no_ref_audio
    response_format = "pcm" if args.stream else args.response_format
    if args.mixture is not None and len(args.mixture) == 1:
        mixture = args.mixture[0]
        concurrency = mixture.total_clients
    elif args.concurrency is None:
        mixture = None
        concurrency = DEFAULT_TTS_BENCHMARK_CONCURRENCY
    else:
        mixture = None
        concurrency = args.concurrency
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
        subtalker_dosample_ratio=args.subtalker_dosample_ratio,
        warmup=args.warmup,
        concurrency=concurrency,
        request_rate=args.request_rate,
        arrival_seed=args.arrival_seed,
        stream=args.stream,
        collect_stream_outcomes=args.collect_stream_outcomes,
        initial_codec_chunk_frames=args.initial_codec_chunk_frames,
        measure_gpu_memory=args.measure_gpu_memory,
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
        mixture=mixture,
        mixture_duration_s=args.mixture_duration_s,
        mixture_settle_s=args.mixture_settle_s,
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


def parse_mixture_loads(value: str) -> list[MixtureLoad]:
    tokens = [token for token in value.replace(" ", "").split(",") if token]
    if not tokens:
        raise argparse.ArgumentTypeError(
            "mixture loads must be a comma separated list with at least one load"
        )
    else:
        pass
    loads: list[MixtureLoad] = []
    for token in tokens:
        match = MIXTURE_LOAD_PATTERN.fullmatch(token)
        if match is None:
            raise argparse.ArgumentTypeError(
                f"invalid mixture load {token!r}, expected NS+S such as 48+16"
            )
        else:
            pass
        load = MixtureLoad(
            non_streaming_clients=int(match[1]), streaming_clients=int(match[2])
        )
        if load.total_clients == 0:
            raise argparse.ArgumentTypeError("a mixture load needs at least one client")
        elif load in loads:
            raise argparse.ArgumentTypeError(f"duplicate mixture load {token!r}")
        else:
            loads.append(load)
    return loads


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
    """Open-loop arrivals above running + queued for duration_s."""
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
    *,
    repeats: int = 1,
) -> dict[str, Any]:
    """Run generate-only across concurrencies and optional repeats."""
    if repeats < 1:
        raise ValueError(f"repeats must be positive, got {repeats}")
    rows: list[ConcurrencyAggregate] = []
    for concurrency in concurrencies:
        repeat_summaries: list[RepeatSpeedSummary] = []
        for repeat_index in range(1, repeats + 1):
            if repeats == 1:
                point_name = f"c{concurrency}"
            else:
                point_name = f"c{concurrency}_r{repeat_index}"
            point_output_dir = os.path.join(config.output_dir, point_name)
            point = replace(
                config,
                concurrency=concurrency,
                output_dir=point_output_dir,
            )
            print(f"[conc={concurrency} repeat={repeat_index}/{repeats}] generate pass")
            results = await run_tts_seedtts_benchmark(point)
            summary = results["summary"]
            repeat_summaries.append(
                {
                    "repeat": repeat_index,
                    "output_dir": point_output_dir,
                    **summary,
                    "warmup": _resolve_warmup(point),
                }
            )
            print_speed_summary(summary, config.model, concurrency=concurrency)
            print(
                f"  success={summary.get('completed_requests')} "
                f"failed={summary.get('failed_requests')} "
                f"latency_p95={summary.get('latency_p95_s')} "
                f"ttfa_p95={summary.get('audio_ttfp_p95_s')}"
            )
        rows.append(aggregate_repeats(concurrency, repeat_summaries))

    payload = {
        "config": _build_results_config(config, base_url=build_base_url(config)),
        "concurrencies": concurrencies,
        "repeats": repeats,
        "rows": rows,
    }
    out_path = os.path.join(config.output_dir, "concurrency_sweep.json")
    os.makedirs(config.output_dir, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    logger.info(f"Wrote concurrency sweep to {out_path}")
    return payload


async def run_tts_mixture_sweep(
    config: TtsSeedttsBenchmarkConfig,
    loads: list[MixtureLoad],
    *,
    repeats: int,
) -> dict:
    """Run every mixture load in order on one server, each pass in its own directory."""
    repeat_results_by_load: dict[MixtureLoad, list[MixtureRepeatResult]] = {}
    for load in loads:
        repeat_results: list[MixtureRepeatResult] = []
        for repeat_index in range(1, repeats + 1):
            if repeats == 1:
                point_name = load.name
            else:
                point_name = f"{load.name}_r{repeat_index}"
            point_output_dir = os.path.join(config.output_dir, point_name)
            point = replace(
                config,
                mixture=load,
                concurrency=load.total_clients,
                output_dir=point_output_dir,
            )
            print(f"[load={load.name} repeat={repeat_index}/{repeats}] generate pass")
            results = await run_tts_seedtts_benchmark(point)
            print_mixture_summary(results, config.model)
            repeat_results.append(
                {
                    "repeat": repeat_index,
                    "output_dir": point_output_dir,
                    "summary": results["summary"],
                    "summary_by_class": results["summary_by_class"],
                    "mixture": results["mixture"],
                }
            )
        repeat_results_by_load[load] = repeat_results

    leg_config = _build_results_config(config, base_url=build_base_url(config))
    leg_config["mixture"] = {
        "loads": [load.name for load in loads],
        "settle_s": config.mixture_settle_s,
        "duration_s": config.mixture_duration_s,
    }
    # note (db-ol): concurrency and the default warmup change with the load, so the leg records None and each point's speed_results.json has its own.
    leg_config["stream"] = None
    leg_config["concurrency"] = None
    leg_config["warmup"] = 0 if config.warmup == 0 else None
    leg_config["collect_stream_outcomes"] = config.collect_stream_outcomes and any(
        load.streaming_clients > 0 for load in loads
    )
    payload = {
        "config": leg_config,
        "loads": [load.name for load in loads],
        "repeats": repeats,
        "rows": aggregate_mixture_sweep(repeat_results_by_load),
    }
    out_path = os.path.join(config.output_dir, "mixture_sweep.json")
    os.makedirs(config.output_dir, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    logger.info(f"Wrote mixture sweep to {out_path}")
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
    if config.mixture is None:
        print_speed_summary(
            results["summary"], config.model, concurrency=config.concurrency
        )
    else:
        print_mixture_summary(results, config.model)
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
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=None,
        help=(
            "Generation length cap sent with every request (default "
            f"{DEFAULT_MAX_NEW_TOKENS}). Fun-CosyVoice3 omits it by default so the "
            "server applies the model's own text-length contract; pass a value to "
            "override that contract."
        ),
    )
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
    parser.add_argument(
        "--subtalker-dosample-ratio",
        type=float,
        default=None,
        help=(
            "Optional model-specific (Qwen3-TTS only) fraction of requests "
            "sent with subtalker_dosample=True (the rest use False), "
            "deterministically interleaved by sample index. 1.0 = all "
            "sampled, 0.0 = all greedy, 0.5 = alternating mixed traffic. "
            "Omit to leave subtalker_dosample unset (server-side default)."
        ),
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=None,
        help="Warmup requests; defaults to the configured concurrency.",
    )
    parser.add_argument(
        "--concurrency",
        "--max-concurrency",
        dest="concurrency",
        type=int,
        default=None,
        help="Maximum concurrent requests.",
    )
    parser.add_argument(
        "--concurrencies",
        type=_parse_concurrencies,
        default=None,
        help="Comma-separated concurrency levels to sweep (requires --generate-only).",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="Measured passes per concurrency. Requires --generate-only when above 1.",
    )
    parser.add_argument(
        "--mixture",
        type=parse_mixture_loads,
        default=None,
        help=(
            "Comma separated NS+S loads such as 48+16,48+0,0+16. Each load runs "
            "NS non streaming and S streaming closed loop clients at once for a "
            "fixed window, in the given order on one server. More than one load "
            "needs --generate-only."
        ),
    )
    parser.add_argument(
        "--mixture-duration-s",
        type=float,
        default=DEFAULT_MIXTURE_DURATION_S,
        help="Measured window of each mixture load in seconds.",
    )
    parser.add_argument(
        "--mixture-settle-s",
        type=float,
        default=DEFAULT_MIXTURE_SETTLE_S,
        help="Unmeasured seconds from the client start to the window.",
    )
    add_fingerprint_argument(parser)
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
        "--arrival-seed",
        type=int,
        default=None,
        help="Seed for the open-loop Poisson arrivals; omit for a fresh sequence.",
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Use streaming for TTS generation.",
    )
    parser.add_argument(
        "--collect-stream-outcomes",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Collect raw PCM finish reasons with background GETs; disable to measure their overhead.",
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
    parser.add_argument(
        "--measure-gpu-memory",
        action="store_true",
        help="Sample memory used on the first visible local GPU during TTS generation.",
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
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    is_sweep = args.concurrencies is not None or args.repeats > 1
    if is_sweep and not args.generate_only:
        parser.error("--concurrencies and --repeats require --generate-only")
    if args.sustained_overshoot and not args.generate_only:
        parser.error("--sustained-overshoot currently requires --generate-only")
    if args.sustained_overshoot and is_sweep:
        parser.error(
            "--sustained-overshoot cannot be combined with --concurrencies or --repeats"
        )
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
    if args.mixture_duration_s <= 0:
        parser.error("--mixture-duration-s must be positive")
    else:
        pass
    if args.mixture_settle_s < 0:
        parser.error("--mixture-settle-s must be at least 0")
    else:
        pass
    if args.mixture is not None and parser.get_default("concurrency") is not None:
        parser.error(
            "the model profile sets concurrency and warmup, so --mixture is not "
            "supported for this model"
        )
    elif args.mixture is not None:
        conflicting_flags = [
            flag
            for flag, is_given in (
                ("--concurrencies", args.concurrencies is not None),
                ("--concurrency/--max-concurrency", args.concurrency is not None),
                ("--stream", args.stream),
                ("--request-rate", args.request_rate != float("inf")),
                ("--arrival-seed", args.arrival_seed is not None),
                ("--sustained-overshoot", args.sustained_overshoot),
                (
                    "--subtalker-dosample-ratio",
                    args.subtalker_dosample_ratio is not None,
                ),
                ("--warmup other than 0", args.warmup not in (None, 0)),
                ("--transcribe-only", args.transcribe_only),
                ("--similarity-only", args.similarity_only),
                ("--utmos-only", args.utmos_only),
            )
            if is_given
        ]
        if conflicting_flags:
            parser.error(
                f"--mixture cannot be combined with {', '.join(conflicting_flags)}"
            )
        elif len(args.mixture) > 1 and not args.generate_only:
            parser.error("--mixture with more than one load requires --generate-only")
        elif (
            args.max_queued_requests is not None
            and args.warmup != 0
            and max(load.total_clients for load in args.mixture)
            > args.max_running_requests + args.max_queued_requests
        ):
            parser.error(
                "--mixture above --max-running-requests plus --max-queued-requests "
                "needs --warmup 0, because the warmup sends one request per client "
                "at once and the server rejects the excess"
            )
        else:
            pass
    else:
        pass


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    configure_dependency_loggers()
    parser = _build_arg_parser()
    args, profile = _parse_args(parser)
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
        if args.fingerprint:
            config.environment_fingerprint = collect_run_fingerprint(
                build_base_url(config)
            )
        else:
            config.environment_fingerprint = None
        is_sweep = args.concurrencies is not None or args.repeats > 1
        if args.mixture is not None and (len(args.mixture) > 1 or is_sweep):
            await run_tts_mixture_sweep(config, args.mixture, repeats=args.repeats)
        elif is_sweep:
            if args.concurrencies is None:
                concurrencies = [config.concurrency]
            else:
                concurrencies = args.concurrencies
            await run_tts_concurrency_sweep(config, concurrencies, repeats=args.repeats)
        elif args.sustained_overshoot:
            await run_tts_sustained_overshoot(config)
        else:
            await benchmark(config)

    if args.use_existing_server:
        asyncio.run(_run_generate())
    else:
        engine_overrides = (
            dict(
                max_running_requests=config.max_running_requests,
                max_queued_requests=config.max_queued_requests,
                cuda_graph_max_bs=config.cuda_graph_max_bs,
                quantization=config.quantization,
            )
            if profile.forward_sglang_engine
            else {}
        )
        with managed_omni_server(
            model_path=config.model,
            port=config.port,
            host=config.host,
            server_config=config.server_config,
            **engine_overrides,
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
