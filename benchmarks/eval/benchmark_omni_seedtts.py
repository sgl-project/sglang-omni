# SPDX-License-Identifier: Apache-2.0
"""SeedTTS benchmark for Omni models (Qwen3-Omni, MiniCPM-o) with speed and WER metrics.

Note (chenyang):

    This benchmark is both used in CI on a subset and locally for the whole set.
    If running locally, the audio generation and transcription are run in overlap,
    thus Qwen3 Omni server and the ASR model share the same GPU.

    On the CI, to avoid GPU OOM, we run the audio generation and transcription
    sequentially.

Usage:

    # Launch the server:
    python -m sglang_omni.cli serve \
        --model-path Qwen/Qwen3-Omni-30B-A3B-Instruct \
        --port 8000

    # Download the test set:
    python -m benchmarks.dataset.prepare --dataset seedtts

    # Full pipeline (generate + transcribe)
    python -m benchmarks.eval.benchmark_omni_seedtts \
        --meta zhaochenyang20/seed-tts-eval-arrow \
        --output-dir results/qwen3_omni_en \
        --max-concurrency 16 \
        --model qwen3-omni --port 8000 --max-samples 50

CI Usage:

    # Generate audio only (server must be running)
    python -m benchmarks.eval.benchmark_omni_seedtts \
        --generate-only \
        --meta zhaochenyang20/seed-tts-eval-arrow \
        --output-dir results/qwen3_omni_en \
        --max-concurrency 16 \
        --model qwen3-omni --port 8000 --max-samples 50

    # Transcribe + WER only (ASR server must be running on --port)
    python -m benchmarks.eval.benchmark_omni_seedtts \
        --transcribe-only \
        --meta zhaochenyang20/seed-tts-eval-arrow \
        --output-dir results/qwen3_omni_en \
        --model qwen3-omni --lang en --port 8000


H200 Full-Set Reference Results

Reproducibility references for the FULL eval set — NOT CI thresholds.
CI runs on a subset and has its own thresholds elsewhere (see tasks/*.py).

Benchmark: SeedTTS  |  Dataset: seed-tts-eval, full set
Hardware:  1 x H200 (default; non-H200 sources are tagged in Source column)
Last verified: 2026-05-04

Accuracy (accuracy.wer)

| Model      | Config            | wer_corpus | wer_per_sample_mean | wer_per_sample_median | wer_per_sample_std | evaluated | skipped | Source                         |
| ---------- | ----------------- | ---------- | ------------------- | --------------------- | ------------------ | --------- | ------- | ------------------------------ |
| Qwen3-Omni | EN, voice_clone=T | 2.44%      | 2.68%               | 0.00%                 | 21.5%              | 1088/1088 | 0       | PR #411 [H200, V1-pipeline, full-set, c=16, n=3 mean] |
| Qwen3-Omni | EN, voice_clone=F | 2.60%      | 2.85%               | 0.00%                 | 12.0%              | 1088/1088 | 0       | PR #411 [H200, V1-pipeline, full-set, c=16] |
| Qwen3-Omni | ZH, voice_clone=T | 1.77%      | 1.79%               | 0.00%                 | 17.1%              | 2020/2020 | 0       | PR #411 [H200, V1-pipeline, full-set, c=16] |
| Qwen3-Omni | ZH, voice_clone=F | 1.92%      | 1.81%               | 0.00%                 | 8.5%               | 2020/2020 | 0       | PR #411 [H200, V1-pipeline, full-set, c=16] |
| Qwen3-Omni | EN, voice_clone=T | 1.86%      | 1.94%               | 0.00%                 | 5.9%               | 1088/1088 | 0       | PR #351 [H100, full-set, c=16] |
| Qwen3-Omni | EN, voice_clone=F | 2.40%      | 2.44%               | 0.00%                 | 7.3%               | 1088/1088 | 0       | PR #351 [H100, full-set, c=16] |
| Qwen3-Omni | ZH, voice_clone=T | 1.49%      | 1.45%               | 0.00%                 | 3.7%               | 2020/2020 | 0       | PR #351 [H100, full-set, c=16] |
| Qwen3-Omni | ZH, voice_clone=F | 1.76%      | 1.62%               | 0.00%                 | 8.6%               | 2018/2020 | 2       | PR #351 [H100, full-set, c=16] |
| Qwen3-Omni | EN, voice_clone=T | 2.53%      | 2.79%               | 0.00%                 | 22.2%              | 1088/1088 | 0       | PR #426 [H100, V1-pipeline, full-set, c=16] |
| Qwen3-Omni | ZH, voice_clone=T | 1.77%      | 1.82%               | 0.00%                 | 19.0%              | 2020/2020 | 0       | PR #411 [H100, V1-pipeline, full-set, c=16] |
| Qwen3-Omni | EN, voice_clone=T | 2.40%      | 2.62%               | 0.00%                 | 21.3%              | 1088/1088 | 0       | PR #411 [H100, V1-pipeline, full-set, c=16, n=3 mean] |
| Qwen3-Omni | EN, voice_clone=F | 2.57%      | 2.80%               | 0.00%                 | 11.7%              | 1088/1088 | 0       | PR #411 [H100, V1-pipeline, full-set, c=16] |
| Qwen3-Omni | ZH, voice_clone=F | 2.38%      | 2.11%               | 0.00%                 | 20.5%              | 2020/2020 | 0       | PR #411 [H100, V1-pipeline, full-set, c=16] |


Generation speed (generation.speed)

| Model      | Config            | latency_mean_s | latency_p95_s | rtf_mean | throughput_qps | output_tok_per_req_s | Source                         |
| ---------- | ----------------- | -------------- | ------------- | -------- | -------------- | ------------------------------ | ------------------------------ |
| Qwen3-Omni | EN, voice_clone=T | 3.06           | 4.32          | 0.87     | 5.224          | 4.7                            | PR #411 [H200, V1-pipeline, full-set, c=16] |
| Qwen3-Omni | EN, voice_clone=F | 2.60           | 3.70          | 0.73     | 6.151          | 5.5                            | PR #411 [H200, V1-pipeline, full-set, c=16] |
| Qwen3-Omni | ZH, voice_clone=T | 3.31           | 4.31          | 0.80     | 4.826          | 5.1                            | PR #411 [H200, V1-pipeline, full-set, c=16] |
| Qwen3-Omni | ZH, voice_clone=F | 2.92           | 3.85          | 0.70     | 5.474          | 5.8                            | PR #411 [H200, V1-pipeline, full-set, c=16] |
| Qwen3-Omni | EN, voice_clone=T | 44.94          | 67.44         | 12.43    | 0.355          | 0.3                            | PR #351 [H100, full-set, c=16] |
| Qwen3-Omni | EN, voice_clone=F | 45.73          | 68.10         | 12.69    | 0.349          | 0.3                            | PR #351 [H100, full-set, c=16] |
| Qwen3-Omni | ZH, voice_clone=T | 55.88          | 74.93         | 13.44    | 0.286          | 0.3                            | PR #351 [H100, full-set, c=16] |
| Qwen3-Omni | ZH, voice_clone=F | 55.09          | 73.68         | 13.19    | 0.288          | 0.3                            | PR #351 [H100, full-set, c=16] |
| Qwen3-Omni | EN, voice_clone=T | 2.56           | 3.70          | 0.73     | 6.225          | 5.6                            | PR #411 [H100, V1-pipeline, full-set, c=16] |
| Qwen3-Omni | ZH, voice_clone=T | 2.79           | 3.68          | 0.67     | 5.737          | 6.1                            | PR #411 [H100, V1-pipeline, full-set, c=16] |
| Qwen3-Omni | EN, voice_clone=F | 2.30           | 3.32          | 0.64     | 6.948          | 6.2                            | PR #411 [H100, V1-pipeline, full-set, c=16] |
| Qwen3-Omni | ZH, voice_clone=F | 2.52           | 3.31          | 0.61     | 6.350          | 6.8                            | PR #411 [H100, V1-pipeline, full-set, c=16] |

Note (Chenyang): output-token rates here count Qwen3-Omni's discrete talker LM
tokens (code tokens driving code2wav) and therefore run at audio frame rate. They
are not comparable to codec-token TTS models (e.g. S2-Pro in
benchmark_tts_seedtts.py); the two backends emit token streams with different
semantics and rates.

ASR speed (asr.speed) — historical baseline

| Lang | asr_latency_mean_s | asr_rtf_mean | asr_throughput_samples_per_s | Source                                      |
| ---- | ------------------ | ------------ | ---------------------------- | ------------------------------------------- |
| EN   | 0.354              | 0.1039       | 2.83                         | PR #316 [H200, from VC=F run]               |
| ZH   | 0.344              | 0.0861       | 2.90                         | PR #316 [H200, from Qwen3-Omni ZH VC=T run] |
| EN   | 0.224              | 0.0660       | 4.46                         | PR #351 [H100, from Qwen3-Omni EN VC=F run] |
| ZH   | 0.261              | 0.0652       | 3.83                         | PR #351 [H100, from Qwen3-Omni ZH VC=T run] |
| EN   | 0.280              | 0.0816       | 3.58                         | PR #426 [H100, V1-pipeline, from Qwen3-Omni EN VC=T run] |
| ZH   | 0.268              | 0.0667       | 3.74                         | PR #426 [H100, V1-pipeline, from Qwen3-Omni ZH VC=T run] |

Local v1 Pipeline Result (this workspace, 2026-05-01)

Accuracy (accuracy.wer)

| Model      | Config            | wer_corpus | wer_per_sample_mean | wer_per_sample_median | wer_per_sample_std | evaluated | skipped | Source                                                                  |
| ---------- | ----------------- | ---------- | ------------------- | --------------------- | ------------------ | --------- | ------- | ----------------------------------------------------------------------- |
| Qwen3-Omni | EN, voice_clone=T | 2.14%      | 2.18%               | 0.00%                 | 7.3%               | 1088/1088 | 0       | local v1 sweep [H200, full-set, c=16, sequential generate+transcribe]  |
| Qwen3-Omni | EN, voice_clone=F | 1.89%      | 1.98%               | 0.00%                 | 7.2%               | 1088/1088 | 0       | local v1 sweep [H200, full-set, c=16, sequential generate+transcribe]  |
| Qwen3-Omni | ZH, voice_clone=T | 1.49%      | 1.46%               | 0.00%                 | 8.5%               | 2020/2020 | 0       | local v1 sweep [H200, full-set, c=16, sequential generate+transcribe]  |
| Qwen3-Omni | ZH, voice_clone=F | 1.80%      | 1.65%               | 0.00%                 | 8.9%               | 2020/2020 | 0       | local v1 sweep [H200, full-set, c=16, sequential generate+transcribe]  |

Generation speed (generation.speed)

| Model      | Config            | latency_mean_s | latency_p95_s | rtf_mean | throughput_qps | output_tok_per_req_s | Source                                                                  |
| ---------- | ----------------- | -------------- | ------------- | -------- | -------------- | ------------------------------ | ----------------------------------------------------------------------- |
| Qwen3-Omni | EN, voice_clone=T | 7.961          | 11.789        | 2.2217   | 2.001          | 1.8                            | local v1 sweep [H200, full-set, c=16, sequential generate+transcribe]  |
| Qwen3-Omni | EN, voice_clone=F | 6.906          | 10.342        | 1.9382   | 2.307          | 2.1                            | local v1 sweep [H200, full-set, c=16, sequential generate+transcribe]  |
| Qwen3-Omni | ZH, voice_clone=T | 9.004          | 12.117        | 2.1632   | 1.774          | 1.9                            | local v1 sweep [H200, full-set, c=16, sequential generate+transcribe]  |
| Qwen3-Omni | ZH, voice_clone=F | 8.043          | 10.736        | 1.9326   | 1.983          | 2.1                            | local v1 sweep [H200, full-set, c=16, sequential generate+transcribe]  |

Standalone ASR speed was not logged separately in the local v1 sweep above.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import time
from dataclasses import asdict, dataclass, replace
from functools import partial
from typing import Protocol, TypedDict

import aiohttp

from benchmarks.benchmarker.conditions import (
    ConcurrencyAggregate,
    RepeatSpeedSummary,
    aggregate_repeats,
    collect_run_fingerprint,
    warn_if_tail_percentile_is_thin,
)
from benchmarks.benchmarker.data import RequestResult
from benchmarks.benchmarker.fingerprint import BenchmarkFingerprint
from benchmarks.benchmarker.runner import (
    BenchmarkRunner,
    RunConfig,
    SendFn,
    resolve_warmup,
)
from benchmarks.benchmarker.utils import (
    get_wav_duration,
    save_json_results,
    wait_for_service,
)
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
    ReferenceAudioField,
    TalkerSamplingParams,
    VoiceCloneOmni,
    build_base_url,
    preload_reference_audio,
    run_seedtts_similarity,
    run_seedtts_transcribe,
    run_seedtts_utmos,
    save_generated_audio_metadata,
    save_speed_results,
    talker_sampling_params,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

TEXT_PREVIEW_LENGTH = 60
DEFAULT_TTS_BENCHMARK_CONCURRENCY = int(os.getenv("TTS_BENCHMARK_CONCURRENCY", "16"))


class SampleRequestSend(Protocol):
    async def __call__(
        self, session: aiohttp.ClientSession, sample: SampleInput
    ) -> RequestResult: ...


@dataclass
class OmniSeedttsBenchmarkConfig:
    model: str
    meta: str
    base_url: str | None = None
    host: str = "localhost"
    port: int = 8000
    lang: str = "en"
    speaker: str = "Ethan"
    voice_clone: bool = False
    reference_audio_field: ReferenceAudioField = "audios"
    stream: bool = False
    output_dir: str = "results/omni_seedtts"
    max_samples: int | None = None
    max_new_tokens: int = 256
    temperature: float = 0.7
    seed: int | None = None
    talker_temperature: float | None = None
    talker_top_p: float | None = None
    talker_top_k: int | None = None
    talker_repetition_penalty: float | None = None
    warmup: int | None = None
    warmup_meta: str | None = None
    max_concurrency: int = DEFAULT_TTS_BENCHMARK_CONCURRENCY
    request_rate: float = float("inf")
    disable_tqdm: bool = False
    environment_fingerprint: BenchmarkFingerprint | None = None
    # Transcribe phase
    device: str = "cuda:0"
    asr_model_path: str = QWEN3_ASR_MODEL_PATH
    asr_concurrency: int = DEFAULT_ASR_TRANSCRIBE_CONCURRENCY
    similarity_checkpoint: str | None = None
    # Optional system prompt prepended to chat messages. Default ``None``
    # preserves the legacy Qwen3-Omni behavior (no system role). Pass a
    # strict TTS-only prompt to suppress chat-mode leakage on models that
    # were not fine-tuned to robustly interpret "please read aloud" as a
    # verbatim-TTS command (e.g. Ming-Omni).
    system_prompt: str | None = None


class OmniSeedttsResultsConfig(TypedDict):
    model: str
    base_url: str
    meta: str
    voice_clone: bool
    reference_audio_field: ReferenceAudioField
    stream: bool
    lang: str
    speaker: str
    max_samples: int | None
    max_new_tokens: int
    temperature: float
    seed: int | None
    talker_temperature: float | None
    talker_top_p: float | None
    talker_top_k: int | None
    talker_repetition_penalty: float | None
    warmup: int
    warmup_meta: str | None
    max_concurrency: int
    request_rate: float


class FingerprintedOmniSeedttsResultsConfig(OmniSeedttsResultsConfig):
    environment_fingerprint: BenchmarkFingerprint


class SweepRunConfig(OmniSeedttsResultsConfig):
    concurrencies: list[int]
    repeats: int


class SweepReport(TypedDict):
    config: SweepRunConfig
    results: list[ConcurrencyAggregate]


def _resolve_warmup(config: OmniSeedttsBenchmarkConfig) -> int:
    return resolve_warmup(config.warmup, config.max_concurrency)


def _build_results_config(
    config: OmniSeedttsBenchmarkConfig,
    *,
    base_url: str,
) -> OmniSeedttsResultsConfig | FingerprintedOmniSeedttsResultsConfig:
    results_config: OmniSeedttsResultsConfig = {
        "model": config.model,
        "base_url": base_url,
        "meta": config.meta,
        "voice_clone": config.voice_clone,
        "reference_audio_field": config.reference_audio_field,
        "stream": config.stream,
        "lang": config.lang,
        "speaker": config.speaker,
        "max_samples": config.max_samples,
        "max_new_tokens": config.max_new_tokens,
        "temperature": config.temperature,
        "seed": config.seed,
        "talker_temperature": config.talker_temperature,
        "talker_top_p": config.talker_top_p,
        "talker_top_k": config.talker_top_k,
        "talker_repetition_penalty": config.talker_repetition_penalty,
        "warmup": _resolve_warmup(config),
        "warmup_meta": config.warmup_meta,
        "max_concurrency": config.max_concurrency,
        "request_rate": config.request_rate,
    }
    if config.environment_fingerprint is None:
        return results_config
    fingerprinted: FingerprintedOmniSeedttsResultsConfig = {
        **results_config,
        "environment_fingerprint": config.environment_fingerprint,
    }
    return fingerprinted


def make_send_fn(
    model_name: str,
    api_url: str,
    *,
    lang: str,
    voice_clone: bool,
    speaker: str,
    max_tokens: int,
    temperature: float,
    seed: int | None,
    talker_params: TalkerSamplingParams,
    reference_audio_data: dict[str, str],
    stream: bool,
    save_audio_dir: str,
    system_prompt: str | None = None,
    reference_audio_field: ReferenceAudioField = "audios",
) -> SendFn:
    """Return a SendFn that calls the Omni chat API and saves WAV."""
    task = VoiceCloneOmni()

    async def send_fn(
        session: aiohttp.ClientSession, sample: SampleInput
    ) -> RequestResult:
        result = RequestResult(
            request_id=sample.sample_id,
            text=sample.target_text[:TEXT_PREVIEW_LENGTH],
        )
        chunk_times: list[float] = []
        text_first_time_holder: list[float] = []
        start_time = time.perf_counter()
        try:
            wav_bytes, _, usage = await task.generate_speech(
                session,
                api_url,
                model_name,
                sample,
                lang,
                speaker=speaker,
                max_tokens=max_tokens,
                temperature=temperature,
                seed=seed,
                talker_params=talker_params,
                voice_clone=voice_clone,
                stream=stream,
                system_prompt=system_prompt,
                reference_audio_field=reference_audio_field,
                reference_audio_data=reference_audio_data.get(sample.ref_audio),
                chunk_times_out=chunk_times if stream else None,
                text_first_time_holder=text_first_time_holder if stream else None,
            )
            result.audio_duration_s = get_wav_duration(wav_bytes)
            elapsed = time.perf_counter() - start_time
            if result.audio_duration_s > 0:
                result.is_success = True
                result.rtf = elapsed / result.audio_duration_s
            else:
                result.error = f"Invalid audio ({len(wav_bytes)} bytes)"

            if usage:
                result.prompt_tokens = usage.get("prompt_tokens", 0)
                result.completion_tokens = usage.get("completion_tokens", 0)

            # note (Chenyang): engine_time_s should be the time taken by
            # the engine. Current omni chat completions has no X-Engine-Time
            # header, so we use request elapsed time as engine_time_s proxy.
            # This shall largely affect the results at high concurrency,
            # since the wait time is included in the request elapsed time.

            result.engine_time_s = elapsed
            if result.completion_tokens > 0 and result.engine_time_s > 0:
                result.tok_per_s = result.completion_tokens / result.engine_time_s

            wav_path = os.path.join(save_audio_dir, f"{result.request_id}.wav")
            with open(wav_path, "wb") as f:
                f.write(wav_bytes)
            result.wav_path = wav_path

            if chunk_times:
                result.audio_ttfp_s = chunk_times[0] - start_time
                result.inter_chunk_s = [
                    chunk_times[i + 1] - chunk_times[i]
                    for i in range(len(chunk_times) - 1)
                ]
            if text_first_time_holder:
                result.text_ttft_s = text_first_time_holder[0] - start_time
        except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
            result.error = str(exc)
        finally:
            result.latency_s = time.perf_counter() - start_time
        return result

    return send_fn


def load_warmup_samples(
    config: OmniSeedttsBenchmarkConfig, warmup_count: int
) -> list[SampleInput]:
    if config.warmup_meta is None or warmup_count <= 0:
        return []
    warmup_samples = load_seedtts_samples(
        config.warmup_meta, warmup_count, split=config.lang
    )
    if len(warmup_samples) != warmup_count:
        raise ValueError(
            f"Requested {warmup_count} warmup samples, found {len(warmup_samples)}"
        )
    return warmup_samples


def configured_talker_params(
    config: OmniSeedttsBenchmarkConfig,
) -> TalkerSamplingParams:
    return talker_sampling_params(
        talker_temperature=config.talker_temperature,
        talker_top_p=config.talker_top_p,
        talker_top_k=config.talker_top_k,
        talker_repetition_penalty=config.talker_repetition_penalty,
    )


async def run_separate_warmup(
    config: OmniSeedttsBenchmarkConfig,
    *,
    base_url: str,
    warmup_samples: list[SampleInput],
    warmup_count: int,
    send_request: SampleRequestSend,
) -> None:
    """Run a separate warmup set and abort when any request fails."""

    async def send_warmup(
        session: aiohttp.ClientSession, sample: SampleInput
    ) -> RequestResult:
        try:
            return await send_request(session, sample)
        except (
            ValueError,
            RuntimeError,
            OSError,
            aiohttp.ClientError,
            asyncio.TimeoutError,
        ) as exc:
            # note (wilsonzheng0327): write results.json before aborting.
            logger.exception(f"Warmup request {sample.sample_id} failed")
            return RequestResult(request_id=sample.sample_id, error=str(exc))

    # note (wenyao): separate inputs warm the path without caching measured samples.
    warmup_runner = BenchmarkRunner(
        RunConfig(
            max_concurrency=config.max_concurrency,
            warmup=0,
            disable_tqdm=config.disable_tqdm,
        )
    )
    warmup_outputs = await warmup_runner.run(warmup_samples, send_warmup)
    completed_count = sum(warmup_output.is_success for warmup_output in warmup_outputs)
    warmup_dir = os.path.join(config.output_dir, "warmup")
    save_json_results(
        {
            "config": _build_results_config(config, base_url=base_url),
            "completed": completed_count,
            "requested": warmup_count,
            "wall_clock_s": warmup_runner.wall_clock_s,
            "results": [asdict(warmup_output) for warmup_output in warmup_outputs],
        },
        warmup_dir,
        "results.json",
    )
    if completed_count != warmup_count:
        raise RuntimeError(
            f"Benchmark warmup completed {completed_count}/{warmup_count}"
        )


async def run_omni_seedtts_benchmark(
    config: OmniSeedttsBenchmarkConfig,
) -> dict:
    """Generate audio and measure speed. Always saves audio for WER use.

    Returns a dict with keys: summary, per_request, config.
    """
    base_url = build_base_url(config)
    api_url = f"{base_url}/v1/chat/completions"

    samples = load_seedtts_samples(config.meta, config.max_samples, split=config.lang)
    logger.info(f"Prepared {len(samples)} requests")

    warmup_count = _resolve_warmup(config)
    warmup_samples = load_warmup_samples(config, warmup_count)

    # note (wilsonzheng0327): encode references once, outside per-request latency.
    if config.voice_clone and config.reference_audio_field == "audio.ref_audio":
        reference_audio_data = preload_reference_audio(samples + warmup_samples)
    else:
        reference_audio_data = {}

    save_audio_dir = os.path.abspath(os.path.join(config.output_dir, "audio"))
    os.makedirs(save_audio_dir, exist_ok=True)

    build_send_fn = partial(
        make_send_fn,
        config.model,
        api_url,
        lang=config.lang,
        voice_clone=config.voice_clone,
        speaker=config.speaker,
        max_tokens=config.max_new_tokens,
        temperature=config.temperature,
        seed=config.seed,
        talker_params=configured_talker_params(config),
        reference_audio_data=reference_audio_data,
        stream=config.stream,
        system_prompt=config.system_prompt,
        reference_audio_field=config.reference_audio_field,
    )

    if warmup_samples:
        warmup_audio_dir = os.path.abspath(
            os.path.join(config.output_dir, "warmup", "audio")
        )
        os.makedirs(warmup_audio_dir, exist_ok=True)
        await run_separate_warmup(
            config,
            base_url=base_url,
            warmup_samples=warmup_samples,
            warmup_count=warmup_count,
            send_request=build_send_fn(save_audio_dir=warmup_audio_dir),
        )
        measured_warmup_count = 0
    else:
        measured_warmup_count = warmup_count

    runner = BenchmarkRunner(
        RunConfig(
            max_concurrency=config.max_concurrency,
            request_rate=config.request_rate,
            warmup=measured_warmup_count,
            disable_tqdm=config.disable_tqdm,
        )
    )
    outputs = await runner.run(samples, build_send_fn(save_audio_dir=save_audio_dir))
    warn_if_tail_percentile_is_thin(len(outputs))

    metrics = compute_speed_metrics(outputs, wall_clock_s=runner.wall_clock_s)
    results_config = _build_results_config(config, base_url=base_url)
    benchmark_results = build_speed_results(outputs, metrics, results_config)
    save_speed_results(outputs, metrics, results_config, config.output_dir)
    save_generated_audio_metadata(outputs, samples, config.output_dir)
    return benchmark_results


def evaluate_generated_audio(
    config: OmniSeedttsBenchmarkConfig,
) -> dict:
    """Transcribe previously saved audio with ASR and compute WER + ASR speed.

    note (Chenyang): Stop the TTS server first; the ASR server is expected on
    ``config.port``.

    Returns a dict with keys: wer_summary, asr_speed, per_sample.
    """
    wer_config = {
        "model": config.model,
        "tts_model": config.model,
        "asr_model": config.asr_model_path,
        "speaker": config.speaker,
        "voice_clone": config.voice_clone,
        "meta": config.meta,
        "max_samples": config.max_samples,
        "asr_concurrency": config.asr_concurrency,
    }
    return run_seedtts_transcribe(
        config,
        wer_config=wer_config,
        log_per_sample=True,
        asr_router_port=config.port,
    )


def _config_from_args(args: argparse.Namespace) -> OmniSeedttsBenchmarkConfig:
    # ``--no-ref-audio`` is kept as a legacy alias so existing automation and
    # shell history keep working after the script merge.  ``--voice-clone``
    # remains the canonical flag.  If neither is passed the dataclass default
    # (``voice_clone=False``) applies.
    voice_clone = args.voice_clone and not args.no_ref_audio
    device = args.device if args.device is not None else args.asr_device
    return OmniSeedttsBenchmarkConfig(
        base_url=args.base_url,
        host=args.host,
        port=args.port,
        model=args.model,
        meta=args.meta,
        lang=args.lang,
        speaker=args.speaker,
        voice_clone=voice_clone,
        reference_audio_field=args.reference_audio_field,
        stream=args.stream,
        output_dir=args.output_dir,
        max_samples=args.max_samples,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        seed=args.seed,
        talker_temperature=args.talker_temperature,
        talker_top_p=args.talker_top_p,
        talker_top_k=args.talker_top_k,
        talker_repetition_penalty=args.talker_repetition_penalty,
        warmup=args.warmup,
        warmup_meta=args.warmup_meta,
        max_concurrency=args.max_concurrency,
        request_rate=args.request_rate,
        disable_tqdm=args.disable_tqdm,
        device=device,
        asr_model_path=args.asr_model_path,
        asr_concurrency=args.asr_concurrency,
        similarity_checkpoint=args.similarity_checkpoint,
        system_prompt=args.system_prompt,
    )


async def benchmark(config: OmniSeedttsBenchmarkConfig) -> dict:
    results = await run_omni_seedtts_benchmark(config)
    print_speed_summary(
        results["summary"], config.model, concurrency=config.max_concurrency
    )
    return results


def run_sweep(
    config: OmniSeedttsBenchmarkConfig, concurrencies: list[int], repeats: int
) -> SweepReport:
    """Run each concurrency level repeats times and write one sweep.json."""
    if repeats < 1:
        raise ValueError(f"repeats must be positive, got {repeats}")
    if not concurrencies:
        raise ValueError("concurrencies must not be empty")
    concurrency_aggregates: list[ConcurrencyAggregate] = []
    for concurrency in concurrencies:
        repeat_summaries: list[RepeatSpeedSummary] = []
        for repeat_index in range(1, repeats + 1):
            repeat_config = replace(
                config,
                max_concurrency=concurrency,
                output_dir=os.path.join(
                    config.output_dir, f"c{concurrency}_r{repeat_index}"
                ),
            )
            repeat_benchmark = asyncio.run(benchmark(repeat_config))
            repeat_summaries.append(
                {
                    "repeat": repeat_index,
                    "output_dir": repeat_config.output_dir,
                    **repeat_benchmark["summary"],
                    "warmup": _resolve_warmup(repeat_config),
                }
            )
        concurrency_aggregates.append(aggregate_repeats(concurrency, repeat_summaries))
    sweep_report: SweepReport = {
        "config": {
            **_build_results_config(config, base_url=build_base_url(config)),
            "concurrencies": concurrencies,
            "repeats": repeats,
        },
        "results": concurrency_aggregates,
    }
    save_json_results(sweep_report, config.output_dir, "sweep.json")
    return sweep_report


def parse_concurrencies(text: str) -> list[int]:
    levels = [int(part) for part in text.split(",") if part.strip()]
    if not levels or any(level <= 0 for level in levels):
        raise argparse.ArgumentTypeError(
            f"concurrencies must be positive integers, got {text!r}"
        )
    return levels


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="SeedTTS benchmark for Omni models (Qwen3-Omni, MiniCPM-o): "
        "speed and WER evaluation."
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
    parser.add_argument(
        "--meta",
        "--testset",
        dest="meta",
        type=str,
        default="zhaochenyang20/seed-tts-eval-arrow",
        help="HuggingFace Arrow/Parquet dataset repo id or local meta.lst path.",
    )
    parser.add_argument(
        "--lang",
        type=str,
        choices=["en", "zh"],
        default="en",
        help="Language for prompt construction and ASR.",
    )
    parser.add_argument(
        "--speaker",
        type=str,
        default="Ethan",
        choices=["Ethan", "Chelsie", "Aiden"],
        help="Qwen3-Omni speaker voice; MiniCPM-o ignores it and clones the reference.",
    )
    # Voice-clone toggle: ``--voice-clone`` and ``--no-ref-audio`` are
    # complementary flags.  They map to a single ``voice_clone`` bool with the
    # dataclass default ``False`` (plain TTS, no reference audio).
    voice_clone_group = parser.add_mutually_exclusive_group()
    voice_clone_group.add_argument(
        "--voice-clone",
        dest="voice_clone",
        action="store_true",
        help="Pass the sample's reference audio for voice cloning.",
    )
    voice_clone_group.add_argument(
        "--no-ref-audio",
        dest="no_ref_audio",
        action="store_true",
        help="Legacy alias: disable voice cloning (equivalent to omitting "
        "--voice-clone; kept for backward-compatible shell history).",
    )
    parser.set_defaults(voice_clone=False, no_ref_audio=False)
    parser.add_argument(
        "--reference-audio-field",
        choices=["audios", "audio.ref_audio"],
        default="audios",
        help="Reference transport: audios for Qwen3-Omni, audio.ref_audio for MiniCPM-o.",
    )
    parser.add_argument("--output-dir", type=str, default="results/omni_seedtts")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Sampler seed sent with every request so generated lengths are "
        "reproducible across runs; omit for unseeded sampling.",
    )
    parser.add_argument(
        "--talker-temperature",
        type=float,
        default=None,
        help="Talker sampling temperature; omit to keep the server default.",
    )
    parser.add_argument(
        "--talker-top-p",
        type=float,
        default=None,
        help="Talker nucleus sampling threshold; omit to keep the server default.",
    )
    parser.add_argument(
        "--talker-top-k",
        type=int,
        default=None,
        help="Talker top-k sampling cutoff; omit to keep the server default.",
    )
    parser.add_argument(
        "--talker-repetition-penalty",
        type=float,
        default=None,
        help="Talker repetition penalty; omit to keep the server default.",
    )
    parser.add_argument(
        "--concurrencies",
        type=parse_concurrencies,
        default=None,
        help="Comma-separated concurrency levels to sweep with --generate-only; each "
        "level runs --repeats times into <output-dir>/c<level>_r<repeat>; per-level "
        "aggregates and raw repeat summaries go to <output-dir>/sweep.json.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="Runs per concurrency level in a sweep.",
    )
    parser.add_argument(
        "--fingerprint",
        action="store_true",
        help="Record an environment fingerprint (git state, package versions, GPUs) "
        "and the server's /v1/models identity in the results config.",
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Use streaming chat completions and concatenate audio chunks.",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=None,
        help="Warmup requests; defaults to the configured concurrency. Without "
        "--warmup-meta the first measured sample is replayed, so its server "
        "caches are already warm when it is timed.",
    )
    parser.add_argument(
        "--warmup-meta",
        type=str,
        default=None,
        help="Separate SeedTTS metadata for warmup; use references and text outside "
        "the measured set. Must contain at least --warmup samples.",
    )
    parser.add_argument(
        "--max-concurrency",
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
        "--save-audio",
        action="store_true",
        help="Legacy flag kept for backward compatibility. The unified "
        "benchmark always saves generated WAVs so the transcribe phase can "
        "reuse them; passing this flag is a no-op.",
    )
    parser.add_argument("--disable-tqdm", action="store_true")
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device for ASR model (transcribe phase).",
    )
    parser.add_argument(
        "--asr-device",
        dest="asr_device",
        type=str,
        default="cuda:0",
        help="Legacy alias for --device (ASR transcription device).",
    )
    parser.add_argument(
        "--asr-model-path",
        type=str,
        default=QWEN3_ASR_MODEL_PATH,
        help="HuggingFace model id served by the ASR endpoint on --port. "
        f"Defaults to {QWEN3_ASR_MODEL_PATH}; openai/whisper-large-v3 "
        "can also be used.",
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
        "--system-prompt",
        type=str,
        default=None,
        help="Optional system role content prepended to every chat request. "
        "Default omits the system message (Qwen3-Omni-tuned legacy behavior). "
        "Pass a strict TTS-only prompt for models that leak chat-style "
        "preambles or refusals (e.g. Ming-Omni).",
    )
    parser.add_argument(
        "--with-similarity",
        action="store_true",
        help="Also score speaker similarity (WavLM-ECAPA-TDNN) after WER, "
        "per seed-tts-eval protocol.",
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
    config = _config_from_args(args)

    if args.save_audio:
        logger.info("--save-audio is a no-op: the unified benchmark always saves WAVs.")

    if args.similarity_only:
        run_seedtts_similarity(config, log_per_sample=True)
        return

    if args.utmos_only:
        run_seedtts_utmos(config, log_per_sample=True)
        return

    if args.transcribe_only:
        evaluate_generated_audio(config)
        return

    is_sweep = args.concurrencies is not None or args.repeats > 1
    if is_sweep and not args.generate_only:
        parser.error("--concurrencies and --repeats require --generate-only")

    base_url = build_base_url(config)
    wait_for_service(base_url, timeout=args.server_timeout)
    if args.fingerprint:
        config.environment_fingerprint = collect_run_fingerprint(base_url)
    else:
        config.environment_fingerprint = None
    if is_sweep:
        run_sweep(config, args.concurrencies or [config.max_concurrency], args.repeats)
        return
    gen_results = asyncio.run(benchmark(config))

    if args.generate_only:
        return

    accuracy_results = evaluate_generated_audio(config)
    similarity_results = None
    if args.with_similarity:
        similarity_results = run_seedtts_similarity(config, log_per_sample=False)
    combined = {
        "generation": {
            "speed": gen_results["summary"],
            "config": gen_results["config"],
            "per_request": gen_results["per_request"],
        },
        "accuracy": {
            "wer": accuracy_results["wer_summary"],
        },
        "asr": {
            "speed": accuracy_results["asr_speed"],
        },
    }
    if similarity_results is not None:
        combined["similarity"] = similarity_results.get("summary", similarity_results)
    save_json_results(combined, config.output_dir, "eval_results.json")


if __name__ == "__main__":
    main()
