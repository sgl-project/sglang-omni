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


Reference Results

Reproducibility references for the FULL eval set — NOT CI thresholds.
CI runs on a subset and has its own thresholds elsewhere (see tasks/*.py).

Benchmark: SeedTTS  |  Dataset: seed-tts-eval, full set (EN=1088, ZH=2020)
Hardware:  1 x H200 (default; non-H200 sources are tagged in Source column)
Last verified: 2026-05-25

Accuracy (accuracy.wer)

Note: the Higgs TTS EN raw corpus WER includes 2 samples above 50% WER; the
outlier-excluded corpus WER is 1.36%.

Note: the MOSS-TTS EN raw corpus WER includes 5 samples above 50% WER; the
outlier-excluded corpus WER is 1.54%.

| Model  | Config           | wer_corpus | wer_per_sample_mean | wer_per_sample_median | wer_per_sample_std | evaluated | skipped | Source                         |
| ------ | ---------------- | ---------- | ------------------- | --------------------- | ------------------ | --------- | ------- | ------------------------------ |
| S2-Pro | EN, stream=False | 1.07%      | 1.03%               | 0.00%                 | 3.6%               | 1088/1088 | 0       | PR #411 [H200, full-set, c=16] |
| S2-Pro | EN, stream=True  | 1.05%      | 1.00%               | 0.00%                 | 3.5%               | 1088/1088 | 0       | PR #411 [H200, full-set, c=16] |
| S2-Pro | ZH, stream=False | 1.02%      | 0.98%               | 0.00%                 | 3.3%               | 2020/2020 | 0       | PR #411 [H200, full-set, c=16] |
| S2-Pro | ZH, stream=True  | 0.93%      | 0.89%               | 0.00%                 | 2.2%               | 2020/2020 | 0       | PR #411 [H200, full-set, c=16] |
| S2-Pro | EN, stream=False | 1.03%      | 0.98%               | 0.00%                 | 3.4%               | 1088/1088 | 0       | PR #351 [H100, full-set, c=16] |
| S2-Pro | EN, stream=True  | 0.98%      | 0.94%               | 0.00%                 | 3.3%               | 1088/1088 | 0       | PR #351 [H100, full-set, c=16] |
| S2-Pro | ZH, stream=False | 0.93%      | 0.89%               | 0.00%                 | 2.2%               | 2020/2020 | 0       | PR #351 [H100, full-set, c=16] |
| S2-Pro | ZH, stream=True  | 0.98%      | 0.94%               | 0.00%                 | 2.4%               | 2020/2020 | 0       | PR #351 [H100, full-set, c=16] |
| S2-Pro | EN, stream=False | 1.16%      | 1.12%               | 0.00%                 | 3.7%               | 1088/1088 | 0       | PR #411 [H100, full-set, c=16] |
| S2-Pro | EN, stream=True  | 1.06%      | 1.02%               | 0.00%                 | 3.5%               | 1088/1088 | 0       | PR #411 [H100, full-set, c=16] |
| S2-Pro | ZH, stream=False | 0.92%      | 0.87%               | 0.00%                 | 2.1%               | 2020/2020 | 0       | PR #411 [H100, full-set, c=16] |
| S2-Pro | ZH, stream=True  | 0.90%      | 0.86%               | 0.00%                 | 2.1%               | 2020/2020 | 0       | PR #411 [H100, full-set, c=16] |
| Higgs TTS | EN, stream=False | 4.68%   | 4.16%               | 0.00%                 | 91.2%              | 1088/1088 | 0       | PR #534 [H200, full-set, c=16, CUDA Graph on, torch.compile off] |
| Higgs TTS | ZH, stream=False | 1.14%   | 1.08%               | 0.00%                 | 2.7%               | 2020/2020 | 0       | PR #534 [H200, full-set, c=16, CUDA Graph on, torch.compile off] |
| MOSS-TTS | EN, stream=False | 1.93%   | 1.98%               | 0.00%                 | 8.1%               | 1088/1088 | 0       | PR #609 [H100, full-set, c=16, token-count=auto] |

Generation speed (generation.speed)

| Model  | Config           | latency_mean_s | latency_p95_s | rtf_mean | throughput_qps | output_tok_per_req_s | Source                         |
| ------ | ---------------- | -------------- | ------------- | -------- | -------------- | ------------------------------ | ------------------------------ |
| S2-Pro | EN, stream=False | 15.836         | 21.509        | 4.269    | 1.005          | 75.0                           | PR #411 [H200, full-set, c=16] |
| S2-Pro | EN, stream=True  | 16.135         | 21.733        | 4.357    | 0.985          | 74.1                           | PR #411 [H200, full-set, c=16] |
| S2-Pro | ZH, stream=False | 16.114         | 20.092        | 3.018    | 0.990          | 73.1                           | PR #411 [H200, full-set, c=16] |
| S2-Pro | ZH, stream=True  | 17.070         | 22.120        | 3.203    | 0.935          | 71.5                           | PR #411 [H200, full-set, c=16] |
| S2-Pro | EN, stream=False | 9.38           | 14.65         | 2.48     | 1.700          | 56.0                           | PR #351 [H100, full-set, c=16] |
| S2-Pro | EN, stream=True  | 9.92           | 15.49         | 2.62     | 1.607          | 53.2                           | PR #351 [H100, full-set, c=16] |
| S2-Pro | ZH, stream=False | 9.64           | 13.61         | 1.80     | 1.655          | 55.2                           | PR #351 [H100, full-set, c=16] |
| S2-Pro | ZH, stream=True  | 9.27           | 13.11         | 1.74     | 1.722          | 51.1                           | PR #351 [H100, full-set, c=16] |
| S2-Pro | EN, stream=False | 12.247         | 16.703        | 3.289    | 1.299          | 67.8                           | PR #411 [H100, V1-pipeline, full-set, c=16] |
| S2-Pro | EN, stream=True  | 12.164         | 16.717        | 3.265    | 1.308          | 67.0                           | PR #411 [H100, V1-pipeline, full-set, c=16] |
| S2-Pro | ZH, stream=False | 12.028         | 15.526        | 2.256    | 1.327          | 65.7                           | PR #411 [H100, V1-pipeline, full-set, c=16] |
| S2-Pro | ZH, stream=True  | 11.417         | 15.020        | 2.141    | 1.398          | 65.5                           | PR #411 [H100, V1-pipeline, full-set, c=16] |
| Higgs TTS | EN, stream=False | 1.749       | 2.600         | 0.425    | 9.104          | 112.9                          | PR #534 [H200, full-set, c=16, CUDA Graph on, torch.compile off] |
| Higgs TTS | ZH, stream=False | 1.629       | 2.110         | 0.282    | 9.792          | 109.9                          | PR #534 [H200, full-set, c=16, CUDA Graph on, torch.compile off] |
| MOSS-TTS | EN, stream=False | 3.890       | 4.781         | 0.913    | 4.091          | 54.1                           | PR #609 [H100, full-set, c=16, token-count=auto] |

Higgs TTS concurrency sweep

Full SeedTTS EN set, 1088/1088 successful requests per run, 3 runs per concurrency. Scores are mean over the 3 complete runs.

| concurrency | latency_mean_s | latency_median_s | latency_p95_s | latency_p99_s | rtf_mean | rtf_median | rtf_p95 | rtf_p99 | audio_duration_mean_s | audio_throughput_s_per_s | output_throughput_tok_s | output_tok_per_req_s | throughput_qps | Source |
| ----------- | -------------- | ---------------- | ------------- | ------------- | -------- | ---------- | ------- | ------- | --------------------- | ------------------------ | ----------------------- | -------------------- | -------------- | ------ |
| 1           | 0.617          | 0.595            | 0.882         | 1.010         | 0.1473   | 0.1448     | 0.1665  | 0.1927  | 4.251                 | 6.885                    | 183.5                   | 196.2                | 1.620          | Local H100, full-set, n=3 mean, c=1|
| 2           | 0.742          | 0.721            | 1.060         | 1.264         | 0.1801   | 0.1710     | 0.2374  | 0.2931  | 4.218                 | 11.368                   | 303.1                   | 183.4                | 2.695          | Local H100, full-set, n=3 mean, c=2|
| 4           | 0.733          | 0.718            | 1.045         | 1.227         | 0.1774   | 0.1743     | 0.2056  | 0.2436  | 4.189                 | 22.837                   | 609.1                   | 166.0                | 5.452          | Local H100, full-set, n=3 mean, c=4|
| 8           | 0.898          | 0.880            | 1.289         | 1.513         | 0.2171   | 0.2131     | 0.2557  | 0.3096  | 4.198                 | 37.383                   | 996.9                   | 135.1                | 8.905          | Local H100, full-set, n=3 mean, c=8|
| 16          | 1.079          | 1.057            | 1.567         | 1.834         | 0.2616   | 0.2527     | 0.3288  | 0.4096  | 4.195                 | 61.842                   | 1649.3                  | 114.5                | 14.742         | Local H100, full-set, n=3 mean, c=16|
| 32          | 2.106          | 2.085            | 2.675         | 2.966         | 0.5273   | 0.5031     | 0.7190  | 0.9553  | 4.191                 | 63.009                   | 1680.5                  | 56.2                 | 15.034         | Local H100, full-set, n=3 mean, c=32|
| 64          | 4.236          | 4.261            | 4.959         | 5.361         | 1.0784   | 1.0226     | 1.6119  | 2.1907  | 4.214                 | 62.232                   | 1659.2                  | 28.4                 | 14.765         | Local H100, full-set, n=3 mean, c=64|

| concurrency | latency_mean_s | latency_median_s | latency_p95_s | latency_p99_s | rtf_mean | rtf_median | rtf_p95 | rtf_p99 | audio_duration_mean_s | audio_throughput_s_per_s | output_throughput_tok_s | output_tok_per_req_s | throughput_qps | Source |
| ----------- | -------------- | ---------------- | ------------- | ------------- | -------- | ---------- | ------- | ------- | --------------------- | ------------------------ | ----------------------- | -------------------- | -------------- | ------ |
| 1           | 0.581          | 0.569            | 0.835         | 0.968         | 0.1408   | 0.1379     | 0.1631  | 0.1867  | 4.186                 | 7.200                    | 192.0                   | 208.9                | 1.720          | Local H200, full-set, n=3 mean, c=1|
| 2           | 0.729          | 0.701            | 1.023         | 1.226         | 0.1749   | 0.1662     | 0.2335  | 0.2923  | 4.287                 | 11.757                   | 313.1                   | 192.4                | 2.743          | Local H200, full-set, n=3 mean, c=2|
| 4           | 0.734          | 0.715            | 1.037         | 1.203         | 0.1779   | 0.1727     | 0.2178  | 0.2620  | 4.207                 | 22.892                   | 610.4                   | 172.0                | 5.441          | Local H200, full-set, n=3 mean, c=4|
| 8           | 0.864          | 0.838            | 1.219         | 1.405         | 0.2077   | 0.2032     | 0.2517  | 0.2996  | 4.239                 | 39.198                   | 1044.7                  | 145.8                | 9.248          | Local H200, full-set, n=3 mean, c=8|
| 16          | 1.165          | 1.124            | 1.762         | 2.144         | 0.2818   | 0.2671     | 0.3779  | 0.4947  | 4.214                 | 57.640                   | 1536.8                  | 108.3                | 13.679         | Local H200, full-set, n=3 mean, c=16|
| 32          | 2.319          | 2.252            | 3.162         | 3.954         | 0.5806   | 0.5473     | 0.8384  | 1.1300  | 4.189                 | 57.277                   | 1527.7                  | 51.5                 | 13.674         | Local H200, full-set, n=3 mean, c=32|
| 64          | 4.466          | 4.494            | 5.205         | 5.597         | 1.1363   | 1.0770     | 1.6952  | 2.3054  | 4.238                 | 59.416                   | 1583.6                  | 27.3                 | 14.021         | Local H200, full-set, n=3 mean, c=64|

Higgs TTS H200 concurrency sweep (accuracy.wer)

| concurrency | wer_corpus | wer_per_sample_mean | wer_per_sample_median | wer_per_sample_std | evaluated | skipped | Source |
| ----------- | ---------- | ------------------- | --------------------- | ------------------ | --------- | ------- | ------ |
| 1           | 1.03%      | 1.01%               | 0.00%                 | 3.6%               | 1088/1088 | 0       | Local H200, full-set, n=3 mean, c=1|
| 2           | 1.11%      | 1.08%               | 0.00%                 | 3.8%               | 1088/1088 | 0       | Local H200, full-set, n=3 mean, c=2|
| 4           | 1.05%      | 1.00%               | 0.00%                 | 3.6%               | 1088/1088 | 0       | Local H200, full-set, n=3 mean, c=4|
| 8           | 1.12%      | 1.08%               | 0.00%                 | 4.6%               | 1088/1088 | 0       | Local H200, full-set, n=3 mean, c=8|
| 16          | 1.10%      | 1.08%               | 0.00%                 | 4.1%               | 1088/1088 | 0       | Local H200, full-set, n=3 mean, c=16|
| 32          | 1.07%      | 1.04%               | 0.00%                 | 3.6%               | 1088/1088 | 0       | Local H200, full-set, n=3 mean, c=32|
| 64          | 1.09%      | 1.07%               | 0.00%                 | 3.8%               | 1088/1088 | 0       | Local H200, full-set, n=3 mean, c=64|
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
from dataclasses import dataclass
from pathlib import Path

from benchmarks.benchmarker.runner import BenchmarkRunner, RunConfig
from benchmarks.benchmarker.utils import managed_omni_server
from benchmarks.dataset.seedtts import load_seedtts_samples
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
    response_format: str = "wav"
    output_dir: str = "results/tts_seedtts"
    max_samples: int | None = None
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
    cuda_graph_max_bs: int = 64
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
        "response_format": config.response_format,
        "voice": config.voice,
        "task_type": config.task_type,
        "instructions": config.instructions,
        "stream": config.stream,
        "max_samples": config.max_samples,
        "max_new_tokens": config.max_new_tokens,
        "seed": config.seed,
        "token_count": config.token_count,
        "warmup": config.warmup,
        "concurrency": config.concurrency,
        "request_rate": config.request_rate,
        "initial_codec_chunk_frames": config.initial_codec_chunk_frames,
        "max_running_requests": config.max_running_requests,
        "cuda_graph_max_bs": config.cuda_graph_max_bs,
    }


async def run_tts_seedtts_benchmark(
    config: TtsSeedttsBenchmarkConfig,
) -> dict:
    """Generate audio and measure speed. Always saves audio for WER use.

    Returns a dict with keys: summary, per_request, config.
    """
    base_url = build_base_url(config)
    api_url = f"{base_url}/v1/audio/speech"

    samples = load_seedtts_samples(config.meta, config.max_samples, split=config.lang)
    logger.info(f"Prepared {len(samples)} requests")

    save_audio_dir = os.path.abspath(os.path.join(config.output_dir, "audio"))
    os.makedirs(save_audio_dir, exist_ok=True)

    generation_kwargs = _build_generation_kwargs(config)
    send_fn = make_tts_send_fn(
        config.model,
        api_url,
        response_format=config.response_format,
        stream=config.stream,
        initial_codec_chunk_frames=config.initial_codec_chunk_frames,
        no_ref_audio=not config.voice_clone,
        ref_format=config.ref_format,
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
        response_format=response_format,
        output_dir=args.output_dir,
        max_samples=args.max_samples,
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
        cuda_graph_max_bs=args.cuda_graph_max_bs,
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
        "--request-rate",
        type=float,
        default=float("inf"),
        help="Requests per second (inf = send all at once).",
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


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()
    if (
        args.initial_codec_chunk_frames is not None
        and args.initial_codec_chunk_frames < 0
    ):
        parser.error("--initial-codec-chunk-frames must be non-negative")
    if args.max_running_requests <= 0:
        parser.error("--max-running-requests must be positive")
    if args.cuda_graph_max_bs <= 0:
        parser.error("--cuda-graph-max-bs must be positive")
    if args.use_existing_server and not (args.generate_only or args.transcribe_only):
        parser.error(
            "--use-existing-server currently requires --generate-only or "
            "--transcribe-only"
        )
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

    if args.use_existing_server:
        asyncio.run(benchmark(config))
    else:
        with managed_omni_server(
            model_path=config.model,
            port=config.port,
            host=config.host,
            max_running_requests=config.max_running_requests,
            cuda_graph_max_bs=config.cuda_graph_max_bs,
            log_file=Path(config.output_dir) / "server_logs" / "tts_server.log",
            timeout=args.server_timeout,
            wait_for_gpu_release=wait_for_gpu_release,
        ):
            asyncio.run(benchmark(config))

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
