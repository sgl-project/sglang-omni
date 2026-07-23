# SPDX-License-Identifier: Apache-2.0
"""TTS task utilities: voice-clone API clients, seed-tts eval stages, and HTTP send functions.

ASR transcription and WER scoring live in benchmarks.tasks.asr. This module
maps generated audio into ASR samples and reuses that layer.

Replaces tasks/tts_speed.py and tasks/voice_clone.py.
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import csv
import io
import json
import logging
import os
import time
import wave
from typing import AsyncIterator, Protocol

import aiohttp
import numpy as np
import soundfile as sf
import torch
from tqdm import tqdm

from benchmarks.benchmarker.data import RequestResult
from benchmarks.benchmarker.runner import SendFn
from benchmarks.benchmarker.utils import (
    WAV_HEADER_SIZE,
    get_wav_duration,
    parse_sse_event,
    save_json_results,
)
from benchmarks.dataset.seedtts import SampleInput, load_seedtts_samples
from benchmarks.metrics.performance import (
    build_speed_results,
    load_tts_speed_summary,
    print_saved_tts_speed_summary,
)
from benchmarks.metrics.speaker_similarity import WavLMSpeakerSimilarity
from benchmarks.metrics.speaker_similarity_assets import (
    ensure_speaker_similarity_assets,
)
from benchmarks.metrics.wer import (
    SampleOutput,
    calculate_asr_speed_metrics,
    calculate_wer_metrics,
    print_asr_speed_summary,
    print_wer_summary,
)
from benchmarks.tasks.asr import (
    ASR_WARMUP_MULTIPLIER,
    apply_wer,
    run_asr_transcription,
    transcribe_and_compute_wer,
)

logger = logging.getLogger(__name__)

TEXT_PREVIEW_LENGTH = 60
SPEAKER_SIMILARITY_BATCH_SIZE = 8
MOSS_TTS_TOKEN_COUNT_AUTO = "auto"
MOSS_TTS_ZH_TOKENS_PER_CHAR = 3.098411951313033
MOSS_TTS_EN_TOKENS_PER_CHAR = 0.8673376262755219
MOSS_TTS_MIN_AUTO_TOKEN_COUNT = 32
UTMOS_BATCH_SIZE = 8


# ---------------------------------------------------------------------------
# WER result persistence
# ---------------------------------------------------------------------------


def save_wer_results(
    outputs: list[SampleOutput], metrics: dict, config: dict, output_dir: str
) -> None:
    json_results = {
        "summary": metrics,
        "config": config,
        "per_sample": [
            {
                "id": o.sample_id,
                "target_text": o.target_text,
                "whisper_text": o.whisper_text,
                "ref_norm": o.ref_norm,
                "hyp_norm": o.hyp_norm,
                "wer": round(o.wer, 6) if o.is_success else None,
                "substitutions": o.substitutions if o.is_success else None,
                "deletions": o.deletions if o.is_success else None,
                "insertions": o.insertions if o.is_success else None,
                "hits": o.hits if o.is_success else None,
                "audio_duration_s": round(o.audio_duration_s, 4),
                "latency_s": round(o.latency_s, 4),
                "is_success": o.is_success,
                "error": o.error or None,
            }
            for o in outputs
        ],
    }
    save_json_results(json_results, output_dir, "wer_results.json")

    csv_path = os.path.join(output_dir, "wer_results.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "id",
                "target_text",
                "whisper_text",
                "wer",
                "substitutions",
                "deletions",
                "insertions",
                "hits",
                "audio_duration_s",
                "latency_s",
                "is_success",
                "error",
            ]
        )
        for o in outputs:
            writer.writerow(
                [
                    o.sample_id,
                    o.target_text,
                    o.whisper_text,
                    f"{o.wer:.6f}" if o.is_success else "",
                    o.substitutions if o.is_success else "",
                    o.deletions if o.is_success else "",
                    o.insertions if o.is_success else "",
                    o.hits if o.is_success else "",
                    f"{o.audio_duration_s:.4f}",
                    f"{o.latency_s:.4f}",
                    o.is_success,
                    o.error or "",
                ]
            )


class SeedttsSimilarityConfig(Protocol):
    """Subset of config fields the shared speaker-similarity pipeline reads.

    Both :class:`OmniSeedttsBenchmarkConfig` and
    :class:`TtsSeedttsBenchmarkConfig` satisfy this protocol via their
    dataclass fields; entry-point parsers default ``similarity_checkpoint``
    to ``None`` when the user does not pass ``--similarity-checkpoint``.
    """

    model: str
    meta: str
    lang: str
    output_dir: str
    device: str
    similarity_checkpoint: str | None


def run_seedtts_similarity(
    config: SeedttsSimilarityConfig,
    *,
    log_per_sample: bool = False,
) -> dict:
    """Compute prompt-vs-generated speaker similarity for saved SeedTTS audio."""
    output_dir = os.path.abspath(config.output_dir)
    generated_path = os.path.join(output_dir, "generated.json")
    with open(generated_path) as f:
        generated: list[dict] = json.load(f)
    if config.max_samples is not None:
        generated = generated[: config.max_samples]
    logger.info(f"Loaded {len(generated)} entries from {generated_path}")

    split = config.lang
    ref_audio_by_id = {
        sample.sample_id: sample.ref_audio
        for sample in load_seedtts_samples(config.meta, config.max_samples, split=split)
    }
    device = config.device
    if "cuda" in device:
        torch.cuda.set_device(device)
        logger.info(f"Set speaker-similarity CUDA device to {device}")

    # Partition entries up-front. Only rows that have a successful generation
    # AND a readable WAV AND a known reference audio AND a readable reference
    # WAV enter the batch scorer; everything else is recorded as skipped so
    # the per_sample table stays exhaustive and a generation failure cannot
    # crash the scorer or contaminate the cosine-similarity batch.
    scoreable: list[dict] = []
    skipped_rows: list[dict] = []
    for entry in generated:
        sample_id = entry.get("sample_id")
        wav_path = entry.get("wav_path")
        ref_audio = ref_audio_by_id.get(sample_id) if sample_id else None
        if (
            entry.get("is_success")
            and isinstance(wav_path, str)
            and wav_path
            and os.path.isfile(wav_path)
            and isinstance(ref_audio, str)
            and os.path.isfile(ref_audio)
        ):
            scoreable.append(entry)
            continue

        if not entry.get("is_success"):
            reason = entry.get("error") or "generation reported is_success=False"
        elif not (isinstance(wav_path, str) and wav_path):
            reason = "wav_path missing from generated.json entry"
        elif not os.path.isfile(wav_path):
            reason = f"wav file not on disk: {wav_path}"
        elif sample_id not in ref_audio_by_id:
            reason = f"no reference audio in meta for sample_id {sample_id!r}"
        else:
            reason = f"reference audio not on disk: {ref_audio}"

        skipped_rows.append(
            {
                "id": sample_id,
                "ref_audio": ref_audio,
                "wav_path": wav_path,
                "speaker_similarity": None,
                "is_success": False,
                "error": reason,
            }
        )

    if not scoreable:
        raise RuntimeError(
            "SeedTTS speaker similarity: no scoreable samples "
            f"({len(skipped_rows)}/{len(generated)} skipped — see per_sample "
            f"for details). Refusing to write empty similarity_results.json."
        )

    assets = ensure_speaker_similarity_assets(
        finetune_checkpoint_override=config.similarity_checkpoint,
    )
    scorer = WavLMSpeakerSimilarity(
        finetune_checkpoint=assets.finetune_checkpoint,
        wavlm_base=assets.wavlm_base,
        device=device,
    )
    scored_rows: list[dict] = []
    scores: list[float] = []
    for start in tqdm(
        range(0, len(scoreable), SPEAKER_SIMILARITY_BATCH_SIZE),
        desc="Speaker similarity",
    ):
        batch = scoreable[start : start + SPEAKER_SIMILARITY_BATCH_SIZE]
        sample_ids = [entry["sample_id"] for entry in batch]
        ref_audio_paths = [
            os.path.abspath(ref_audio_by_id[sample_id]) for sample_id in sample_ids
        ]
        wav_paths = [os.path.abspath(entry["wav_path"]) for entry in batch]
        similarities = scorer.score_batch(ref_audio_paths, wav_paths)

        for sample_id, ref_audio, wav_path, similarity in zip(
            sample_ids,
            ref_audio_paths,
            wav_paths,
            similarities,
        ):
            scores.append(similarity)
            scored_rows.append(
                {
                    "id": sample_id,
                    "ref_audio": ref_audio,
                    "wav_path": wav_path,
                    "speaker_similarity": similarity,
                    "is_success": True,
                    "error": None,
                }
            )
            if log_per_sample:
                logger.info(f"[{sample_id}] similarity={similarity:.3f}")

    similarity_mean = sum(scores) / len(scores)
    metrics = {
        "speaker_similarity_mean": similarity_mean,
        "total_samples": len(generated),
        "evaluated": len(scored_rows),
        "skipped": len(skipped_rows),
    }
    print(
        "SeedTTS speaker similarity: "
        f"{similarity_mean:.4f} ({len(scored_rows)}/{len(generated)} evaluated, "
        f"{len(skipped_rows)} skipped)"
    )
    if skipped_rows:
        logger.warning(
            "SeedTTS speaker similarity: %d samples skipped "
            "(see per_sample with is_success=False for details).",
            len(skipped_rows),
        )

    per_sample = scored_rows + skipped_rows
    save_json_results(
        {
            "summary": metrics,
            "config": {
                "model": config.model,
                "meta": config.meta,
                "device": device,
                "max_samples": config.max_samples,
                "similarity_checkpoint": str(assets.finetune_checkpoint),
            },
            "per_sample": per_sample,
        },
        config.output_dir,
        "similarity_results.json",
    )
    return {"summary": metrics, "per_sample": per_sample}


class SeedttsUTMOSConfig(Protocol):
    """Subset of config fields the shared UTMOS pipeline reads."""

    model: str
    output_dir: str
    device: str


def run_seedtts_utmos(
    config: SeedttsUTMOSConfig,
    *,
    log_per_sample: bool = False,
) -> dict:
    """Compute UTMOS MOS scores for saved SeedTTS generated audio."""
    from benchmarks.metrics.utmos import UTMOSScorer

    output_dir = os.path.abspath(config.output_dir)
    generated_path = os.path.join(output_dir, "generated.json")
    with open(generated_path) as f:
        generated: list[dict] = json.load(f)
    logger.info(f"Loaded {len(generated)} entries from {generated_path}")

    per_sample: list[dict | None] = [None] * len(generated)
    scoreable: list[dict] = []

    for idx, entry in enumerate(generated):
        sample_id = entry.get("sample_id")
        wav_path = entry.get("wav_path")
        if (
            isinstance(sample_id, str)
            and sample_id
            and entry.get("is_success")
            and isinstance(wav_path, str)
            and wav_path
            and os.path.isfile(wav_path)
        ):
            scoreable.append({**entry, "_idx": idx})
            continue

        if not isinstance(sample_id, str) or not sample_id:
            reason = "sample_id missing from generated.json entry"
        elif not entry.get("is_success"):
            reason = entry.get("error") or "generation reported is_success=False"
        elif not (isinstance(wav_path, str) and wav_path):
            reason = "wav_path missing from generated.json entry"
        else:
            reason = f"wav file not on disk: {wav_path}"

        per_sample[idx] = {
            "id": sample_id or wav_path,
            "wav_path": wav_path,
            "utmos_score": None,
            "is_success": False,
            "error": reason,
        }

    if not scoreable:
        raise RuntimeError(
            "UTMOS: no scoreable samples "
            f"({sum(1 for r in per_sample if r is not None)}/{len(generated)} skipped). "
            "Refusing to write empty utmos_results.json."
        )

    device = config.device
    if "cuda" in device:
        torch.cuda.set_device(device)
        logger.info(f"Set UTMOS CUDA device to {device}")

    scorer = UTMOSScorer(device=device)
    scores: list[float] = []

    for start in tqdm(
        range(0, len(scoreable), UTMOS_BATCH_SIZE),
        desc="UTMOS scoring",
    ):
        batch = scoreable[start : start + UTMOS_BATCH_SIZE]
        wav_paths = [entry["wav_path"] for entry in batch]
        batch_scores = scorer.score_batch(wav_paths)
        if len(batch_scores) != len(batch):
            raise RuntimeError(
                f"UTMOS scorer returned {len(batch_scores)} scores for {len(batch)} inputs"
            )

        for entry, score in zip(batch, batch_scores):
            scores.append(score)
            per_sample[entry["_idx"]] = {
                "id": entry["sample_id"],
                "wav_path": entry["wav_path"],
                "utmos_score": round(score, 4),
                "is_success": True,
                "error": None,
            }
            if log_per_sample:
                logger.info(f"[{entry['sample_id']}] utmos={score:.3f}")

    n_skipped = sum(1 for r in per_sample if r is None or not r["is_success"])
    n_scored = len(scores)

    metrics = {
        "utmos_mean": round(float(np.mean(scores)), 4),
        "utmos_median": round(float(np.median(scores)), 4),
        "utmos_p5": round(float(np.percentile(scores, 5)), 4),
        "utmos_p95": round(float(np.percentile(scores, 95)), 4),
        "total_samples": len(generated),
        "evaluated": n_scored,
        "skipped": n_skipped,
    }
    print(
        f"UTMOS: mean={metrics['utmos_mean']:.4f} "
        f"({n_scored}/{len(generated)} evaluated, {n_skipped} skipped)"
    )
    if n_skipped:
        logger.warning(
            "UTMOS: %d samples skipped (see per_sample for details).",
            n_skipped,
        )

    save_json_results(
        {
            "summary": metrics,
            "config": {
                "model": config.model,
                "device": device,
            },
            "per_sample": per_sample,
        },
        config.output_dir,
        "utmos_results.json",
    )
    return {"summary": metrics, "per_sample": per_sample}


# ---------------------------------------------------------------------------
# Shared transcribe pipeline (seed-tts-eval style)
# ---------------------------------------------------------------------------


class ServerEndpointConfig(Protocol):
    """Subset used by :func:`build_base_url` to resolve a server endpoint."""

    base_url: str | None
    host: str
    port: int


class SeedttsTranscribeConfig(Protocol):
    """Subset of config fields the shared transcribe pipeline reads.

    Kept narrow on purpose: ``run_seedtts_transcribe`` does not touch any
    server fields, so callers whose configs lack ``host``/``port`` can still
    satisfy this Protocol.
    """

    model: str
    output_dir: str
    lang: str
    device: str
    asr_model_path: str
    asr_concurrency: int


def build_base_url(config: ServerEndpointConfig) -> str:
    """Resolve the server base URL from an explicit override or host/port."""
    return config.base_url or f"http://{config.host}:{config.port}"


def _log_transcribe_result(
    *,
    idx: int,
    total: int,
    entry: dict,
    output: SampleOutput,
    log_per_sample: bool,
) -> None:
    if output.is_success:
        if log_per_sample:
            logger.info(
                f"[{idx + 1}/{total}] "
                f"WER={output.wer:.3f}  "
                f"asr={output.asr_latency_s:.3f}s  "
                f"ref={output.ref_norm[:50]}  "
                f"hyp={output.hyp_norm[:50]}",
            )
        return

    # only warn for post-generation transcription failures.
    # Generation failures are surfaced at speed-benchmark time and already logged.
    if entry.get("is_success", False):
        logger.warning(
            f"[{idx + 1}/{total}] Transcription failed: "
            f"{entry['sample_id']} -- {output.error}",
        )


def _transcribe_generated_via_runner(
    generated: list[dict],
    router_port: int,
    model_path: str,
    lang: str,
    concurrency: int,
    *,
    log_per_sample: bool = False,
) -> tuple[list[SampleOutput], float]:
    done = [e for e in generated if e.get("is_success", False)]
    samples = [
        SampleInput(
            sample_id=e["sample_id"],
            ref_text=e["target_text"],
            ref_audio=e["wav_path"],
            target_text=e["target_text"],
        )
        for e in done
    ]
    results, wall_s = asyncio.run(
        run_asr_transcription(
            samples,
            port=router_port,
            model_path=model_path,
            lang=lang,
            concurrency=concurrency,
            warmup=concurrency * ASR_WARMUP_MULTIPLIER,
        )
    )
    result_by_id = {r.request_id: r for r in results}
    outputs: list[SampleOutput] = []
    total = len(generated)
    for idx, e in enumerate(generated):
        output = SampleOutput(
            sample_id=e["sample_id"], target_text=e.get("target_text", "")
        )
        output.latency_s = e.get("latency_s", 0.0)
        output.audio_duration_s = e.get("audio_duration_s", 0.0)
        if not e.get("is_success", False):
            output.error = f"Generation failed: {e.get('error', 'unknown')}"
            outputs.append(output)
            continue
        result = result_by_id.get(e["sample_id"])
        if result is None or not result.is_success:
            output.error = (result.error if result else "") or "No transcription"
        else:
            output.asr_latency_s = result.latency_s
            output = apply_wer(output, result.text, lang)
        _log_transcribe_result(
            idx=idx,
            total=total,
            entry=e,
            output=output,
            log_per_sample=log_per_sample,
        )
        outputs.append(output)
    return outputs, wall_s


def run_seedtts_transcribe(
    config: SeedttsTranscribeConfig,
    *,
    wer_config: dict,
    generation_mode: str | None = None,
    log_per_sample: bool = False,
    asr_router_port: int | None = None,
) -> dict:
    """Transcribe saved audio, compute WER + ASR-speed metrics, and persist them.

    Shared pipeline used by both Qwen3-Omni and S2-Pro seed-tts-eval benchmarks.
    The caller-specific ``wer_config`` dict is embedded in ``wer_results.json``
    to preserve backward-compatible fields.

    Returns a dict with keys:
        - ``wer_summary``: corpus-level WER metrics (see :func:`calculate_wer_metrics`)
        - ``asr_speed``:   ASR transcription latency/throughput metrics
        - ``per_sample``:  list[SampleOutput] with per-sample details
    """
    generated_path = os.path.join(config.output_dir, "generated.json")
    with open(generated_path) as f:
        generated: list[dict] = json.load(f)
    logger.info(f"Loaded {len(generated)} entries from {generated_path}")

    asr_model_path = config.asr_model_path
    asr_concurrency = max(1, int(config.asr_concurrency))
    outputs, asr_wall_time_s = _transcribe_generated_via_runner(
        generated,
        asr_router_port,
        asr_model_path,
        config.lang,
        asr_concurrency,
        log_per_sample=log_per_sample,
    )

    wer_metrics = calculate_wer_metrics(outputs, config.lang)
    asr_metrics = calculate_asr_speed_metrics(outputs, wall_time_s=asr_wall_time_s)
    asr_metrics["asr_model"] = asr_model_path
    asr_metrics["asr_concurrency"] = asr_concurrency

    tts_speed_summary = load_tts_speed_summary(config.output_dir)
    print_saved_tts_speed_summary(
        config.output_dir,
        config.model,
        concurrency=wer_config.get("concurrency"),
        generation_mode=generation_mode,
    )
    print_wer_summary(
        wer_metrics,
        config.model,
        generation_mode,
        tts_speed_summary=tts_speed_summary,
    )
    print_asr_speed_summary(asr_metrics, asr_model_path)

    save_wer_results(outputs, wer_metrics, wer_config, config.output_dir)
    save_json_results(asr_metrics, config.output_dir, "asr_speed_results.json")

    return {
        "wer_summary": wer_metrics,
        "asr_speed": asr_metrics,
        "per_sample": outputs,
    }


# ---------------------------------------------------------------------------
# Voice-clone API clients
# ---------------------------------------------------------------------------


class VoiceCloneTTS:
    """Voice cloning via /v1/audio/speech (OAI TTS API format)."""

    async def generate_speech(
        self,
        session: aiohttp.ClientSession,
        api_url: str,
        model_name: str,
        sample: SampleInput,
        max_new_tokens: int = 2048,
        temperature: float = 0.8,
        seed: int | None = None,
    ) -> tuple[bytes, float]:
        payload: dict = {
            "model": model_name,
            "input": sample.target_text,
            "ref_audio": sample.ref_audio,
            "ref_text": sample.ref_text,
            "response_format": "wav",
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
        }
        if seed is not None:
            payload["seed"] = seed

        t0 = time.perf_counter()
        async with session.post(api_url, json=payload) as response:
            if response.status != 200:
                error_text = await response.text()
                raise RuntimeError(f"HTTP {response.status}: {error_text}")
            wav_bytes = await response.read()
        latency = time.perf_counter() - t0

        if len(wav_bytes) <= WAV_HEADER_SIZE:
            raise ValueError(
                f"Empty or invalid audio response ({len(wav_bytes)} bytes)"
            )
        return wav_bytes, latency

    async def generate_speech_streaming(
        self,
        session: aiohttp.ClientSession,
        api_url: str,
        model_name: str,
        sample: SampleInput,
        max_new_tokens: int = 2048,
        temperature: float = 0.8,
        seed: int | None = None,
    ) -> tuple[bytes, float]:
        """Generate speech via raw PCM streaming and return a WAV container."""
        payload: dict = {
            "model": model_name,
            "input": sample.target_text,
            "ref_audio": sample.ref_audio,
            "ref_text": sample.ref_text,
            "response_format": "pcm",
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "stream": True,
        }
        if seed is not None:
            payload["seed"] = seed

        t0 = time.perf_counter()
        pcm_chunks: list[bytes] = []

        async with session.post(api_url, json=payload) as response:
            if response.status != 200:
                error_text = await response.text()
                raise RuntimeError(f"HTTP {response.status}: {error_text}")

            pcm_format = _validate_raw_pcm_response_headers(response.headers)
            if pcm_format is None:
                content_type = response.headers.get("Content-Type")
                raise ValueError(
                    f"Expected audio/pcm streaming response, got {content_type!r}"
                )
            async for chunk, _ in _iter_response_http_chunks(response):
                if chunk:
                    pcm_chunks.append(chunk)

        latency = time.perf_counter() - t0

        if not pcm_chunks:
            raise ValueError("No audio chunks received from streaming response")
        pcm_bytes = b"".join(pcm_chunks)
        block_align = pcm_format[1] * pcm_format[2]
        if len(pcm_bytes) % block_align != 0:
            raise ValueError(
                "PCM response ended with a partial audio frame "
                f"(bytes={len(pcm_bytes)}, block_align={block_align})"
            )

        return _build_streaming_wav_bytes(pcm_chunks, pcm_format), latency

    async def evaluate_sample(
        self,
        session: aiohttp.ClientSession,
        api_url: str,
        model_name: str,
        asr: dict,
        sample: SampleInput,
        lang: str,
        device: str,
        audio_dir: str,
        max_new_tokens: int = 2048,
        temperature: float = 0.8,
        seed: int | None = None,
        stream: bool = False,
    ) -> SampleOutput:
        output = SampleOutput(
            sample_id=sample.sample_id,
            target_text=sample.target_text,
        )
        wav_path = os.path.join(audio_dir, f"{sample.sample_id}.wav")

        try:
            gen_fn = self.generate_speech_streaming if stream else self.generate_speech
            wav_bytes, latency = await gen_fn(
                session, api_url, model_name, sample, max_new_tokens, temperature, seed
            )
            with open(wav_path, "wb") as f:
                f.write(wav_bytes)
            output.latency_s = round(latency, 4)
            output.audio_duration_s = round(sf.info(wav_path).duration, 4)
        except Exception as exc:
            output.error = f"Generation failed: {exc}"
            logger.error(f"[{sample.sample_id}] {output.error}")
            return output

        return transcribe_and_compute_wer(output, wav_path, asr, lang, device)


class VoiceCloneOmni:
    """Voice cloning via /v1/chat/completions (Omni API format).

    Shared by Qwen3 Omni and future Omni models.
    """

    THINKER_MAX_NEW_TOKENS = 256

    async def generate_speech(
        self,
        session: aiohttp.ClientSession,
        api_url: str,
        model_name: str,
        sample: SampleInput,
        lang: str,
        speaker: str = "Ethan",
        max_tokens: int | None = None,
        temperature: float = 0.7,
        voice_clone: bool = False,
        stream: bool = False,
        system_prompt: str | None = None,
        chunk_times_out: list[float] | None = None,
        text_first_time_holder: list[float] | None = None,
    ) -> tuple[bytes, float, dict]:
        if max_tokens is None:
            max_tokens = self.THINKER_MAX_NEW_TOKENS

        if voice_clone:
            if lang == "en":
                prompt_text = (
                    f'Listen to the audio above. The speaker is reading: "{sample.ref_text}". '
                    f"Now please read the following text out loud in the same voice and style: "
                    f"{sample.target_text}"
                )
            else:
                prompt_text = (
                    f'听上面的音频，说话人正在朗读："{sample.ref_text}"。'
                    f"现在请用同样的声音和风格朗读以下文本：{sample.target_text}"
                )
        else:
            if lang == "en":
                prompt_text = (
                    f"Please read the following text out loud in English: "
                    f"{sample.target_text}"
                )
            else:
                prompt_text = f"请用中文朗读以下文本: {sample.target_text}"

        messages: list[dict] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt_text})

        payload = {
            "model": model_name,
            "messages": messages,
            "modalities": ["text", "audio"],
            "audio": {"format": "wav"},
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": stream,
        }
        if voice_clone:
            payload["audios"] = [sample.ref_audio]

        t0 = time.perf_counter()
        async with session.post(api_url, json=payload) as response:
            if response.status != 200:
                error_text = await response.text()
                raise RuntimeError(f"HTTP {response.status}: {error_text}")
            if stream:
                wav_bytes, usage = await self._read_streaming_chat_audio(
                    response,
                    chunk_times_out=chunk_times_out,
                    text_first_time_holder=text_first_time_holder,
                )
                latency = time.perf_counter() - t0
                return wav_bytes, latency, usage
            resp_json = await response.json()
        latency = time.perf_counter() - t0

        choices = resp_json.get("choices", [])
        if not choices:
            raise ValueError("No choices in response")

        message = choices[0].get("message", {})
        audio_obj = message.get("audio")
        if audio_obj is None:
            raise ValueError(
                f"No audio in response for sample '{sample.sample_id}'. "
                f"Text response: {message.get('content', 'N/A')[:100]}"
            )

        audio_b64 = audio_obj.get("data")
        if not audio_b64:
            raise ValueError("Empty audio data in response")

        wav_bytes = base64.b64decode(audio_b64)
        usage = resp_json.get("usage", {})
        return wav_bytes, latency, usage

    async def _read_streaming_chat_audio(
        self,
        response: aiohttp.ClientResponse,
        chunk_times_out: list[float] | None = None,
        text_first_time_holder: list[float] | None = None,
    ) -> tuple[bytes, dict]:
        """Read OpenAI chat SSE audio deltas and concatenate them into one WAV."""
        pcm_chunks: list[bytes] = []
        pcm_format: tuple[int, int, int] | None = None
        usage: dict = {}
        buffer = bytearray()

        async for chunk in response.content.iter_any():
            buffer.extend(chunk)
            while b"\n" in buffer:
                idx = buffer.index(b"\n")
                raw_line = bytes(buffer[:idx])
                del buffer[: idx + 1]
                pcm_format = _collect_chat_streaming_audio(
                    raw_line.decode("utf-8", errors="replace").strip(),
                    pcm_chunks,
                    pcm_format,
                    usage,
                    chunk_times_out=chunk_times_out,
                    text_first_time_holder=text_first_time_holder,
                )

        if buffer.strip():
            pcm_format = _collect_chat_streaming_audio(
                bytes(buffer).decode("utf-8", errors="replace").strip(),
                pcm_chunks,
                pcm_format,
                usage,
                chunk_times_out=chunk_times_out,
                text_first_time_holder=text_first_time_holder,
            )

        if not pcm_chunks or pcm_format is None:
            raise ValueError("No audio chunks received from streaming response")
        return _build_streaming_wav_bytes(pcm_chunks, pcm_format), usage

    async def evaluate_sample(
        self,
        session: aiohttp.ClientSession,
        api_url: str,
        model_name: str,
        asr: dict,
        sample: SampleInput,
        lang: str,
        asr_device: str,
        audio_dir: str,
        speaker: str = "Ethan",
        max_tokens: int | None = None,
        voice_clone: bool = False,
        stream: bool = False,
        system_prompt: str | None = None,
    ) -> SampleOutput:
        output = SampleOutput(
            sample_id=sample.sample_id,
            target_text=sample.target_text,
        )
        wav_path = os.path.join(audio_dir, f"{sample.sample_id}.wav")

        try:
            wav_bytes, latency, _usage = await self.generate_speech(
                session,
                api_url,
                model_name,
                sample,
                lang,
                speaker,
                max_tokens,
                voice_clone=voice_clone,
                stream=stream,
                system_prompt=system_prompt,
            )
            with open(wav_path, "wb") as f:
                f.write(wav_bytes)
            output.latency_s = round(latency, 4)
            output.audio_duration_s = round(sf.info(wav_path).duration, 4)
        except Exception as exc:
            output.error = f"Generation failed: {exc}"
            logger.error(f"[{sample.sample_id}] {output.error}")
            return output

        return transcribe_and_compute_wer(output, wav_path, asr, lang, asr_device)


# ---------------------------------------------------------------------------
# TTS HTTP send layer  (/v1/audio/speech)
# ---------------------------------------------------------------------------


def _build_tts_payload(
    sample: SampleInput,
    model_name: str,
    *,
    response_format: str = "wav",
    stream: bool = False,
    initial_codec_chunk_frames: int | None = None,
    no_ref_audio: bool = False,
    ref_format: str = "flat",
    voice: str | None = None,
    task_type: str | None = None,
    instructions: str | None = None,
    **gen_kwargs,
) -> dict:
    payload: dict = {
        "model": model_name,
        "input": sample.target_text,
        "response_format": "pcm" if stream else response_format,
    }
    if not no_ref_audio:
        if ref_format == "references":
            payload["references"] = [
                {"audio_path": sample.ref_audio, "text": sample.ref_text}
            ]
        else:
            payload["ref_audio"] = sample.ref_audio
            payload["ref_text"] = sample.ref_text
    if voice is not None:
        payload["voice"] = voice
    if task_type is not None:
        payload["task_type"] = task_type
    if instructions is not None:
        payload["instructions"] = instructions
    resolved_gen_kwargs = _resolve_tts_generation_kwargs(sample, gen_kwargs)
    for key, value in resolved_gen_kwargs.items():
        if value is not None:
            payload[key] = value
    if stream:
        payload["stream"] = True
        if initial_codec_chunk_frames is not None:
            payload["initial_codec_chunk_frames"] = initial_codec_chunk_frames
    return payload


def estimate_moss_tts_duration_tokens(text: str) -> int:
    """Estimate MOSS-TTS duration tokens using OpenMOSS app defaults."""
    normalized = text or ""
    effective_len = max(len(normalized), 1)
    zh_chars = sum(1 for ch in normalized if "\u4e00" <= ch <= "\u9fff")
    en_chars = sum(1 for ch in normalized if ("A" <= ch <= "Z") or ("a" <= ch <= "z"))
    factor = (
        MOSS_TTS_ZH_TOKENS_PER_CHAR
        if zh_chars and zh_chars >= en_chars
        else MOSS_TTS_EN_TOKENS_PER_CHAR
    )
    return max(MOSS_TTS_MIN_AUTO_TOKEN_COUNT, int(effective_len * factor))


def _resolve_tts_generation_kwargs(
    sample: SampleInput,
    gen_kwargs: dict,
) -> dict:
    token_count = gen_kwargs.get("token_count")
    if not isinstance(token_count, str):
        return gen_kwargs

    normalized = token_count.strip().lower()
    if normalized != MOSS_TTS_TOKEN_COUNT_AUTO:
        return gen_kwargs

    resolved = dict(gen_kwargs)
    resolved["token_count"] = estimate_moss_tts_duration_tokens(sample.target_text)
    return resolved


def _parse_response_headers(result: RequestResult, headers: dict) -> None:
    prompt_tok = headers.get("X-Prompt-Tokens")
    comp_tok = headers.get("X-Completion-Tokens")
    eng_time = headers.get("X-Engine-Time")
    if prompt_tok is not None:
        result.prompt_tokens = int(prompt_tok)
    if comp_tok is not None:
        result.completion_tokens = int(comp_tok)
    if eng_time is not None:
        result.engine_time_s = float(eng_time)
    if result.completion_tokens > 0 and result.engine_time_s > 0:
        result.tok_per_s = result.completion_tokens / result.engine_time_s


def _parse_pcm_response_format(
    headers: aiohttp.typedefs.LooseHeaders,
) -> tuple[int, int, int]:
    sample_rate = int(headers.get("x-sample-rate", 24000))
    num_channels = int(headers.get("x-channels", 1))
    bit_depth = int(headers.get("x-bit-depth", 16))
    if sample_rate <= 0:
        raise ValueError("x-sample-rate must be positive")
    if num_channels <= 0:
        raise ValueError("x-channels must be positive")
    if bit_depth <= 0 or bit_depth % 8 != 0:
        raise ValueError("x-bit-depth must be a positive multiple of 8")
    sample_width = bit_depth // 8
    return sample_rate, num_channels, sample_width


def _validate_raw_pcm_response_headers(
    headers: aiohttp.typedefs.LooseHeaders,
) -> tuple[int, int, int] | None:
    content_type = str(headers.get("Content-Type", "")).lower().split(";", 1)[0]
    if content_type != "audio/pcm":
        return None
    return _parse_pcm_response_format(headers)


async def _iter_response_http_chunks(
    response: aiohttp.ClientResponse,
) -> AsyncIterator[tuple[bytes, float]]:
    pending = bytearray()
    pending_start_s: float | None = None
    async for data, end_of_http_chunk in response.content.iter_chunks():
        now = time.perf_counter()
        if data:
            if not pending:
                pending_start_s = now
            pending.extend(data)
        if end_of_http_chunk and pending:
            yield bytes(pending), pending_start_s or now
            pending.clear()
            pending_start_s = None

    if pending:
        yield bytes(pending), pending_start_s or time.perf_counter()


async def _handle_raw_pcm_streaming_response(
    response: aiohttp.ClientResponse,
    result: RequestResult,
    start_time: float,
    save_audio_dir: str | None,
) -> None:
    pcm_chunks: list[bytes] = []
    chunk_times: list[float] = []
    try:
        pcm_format = _validate_raw_pcm_response_headers(response.headers)
    except ValueError as exc:
        result.error = f"Invalid PCM response headers: {exc}"
        return
    if pcm_format is None:
        content_type = response.headers.get("Content-Type")
        result.error = f"Expected audio/pcm streaming response, got {content_type!r}"
        return

    async for chunk, chunk_time in _iter_response_http_chunks(response):
        if not chunk:
            continue
        if not chunk_times:
            result.audio_ttfp_s = chunk_time - start_time
            result.first_audio_payload_bytes = len(chunk)
        chunk_times.append(chunk_time)
        pcm_chunks.append(bytes(chunk))

    if chunk_times:
        result.inter_chunk_s = [
            now - prev for prev, now in zip(chunk_times, chunk_times[1:])
        ]
    result.audio_chunk_count = len(pcm_chunks)
    pcm_bytes = b"".join(pcm_chunks)
    sample_rate, num_channels, sample_width = pcm_format
    bytes_per_second = sample_rate * num_channels * sample_width
    block_align = num_channels * sample_width
    if not pcm_bytes or bytes_per_second <= 0:
        result.error = f"Empty or invalid PCM response ({len(pcm_bytes)} bytes)"
        return
    if len(pcm_bytes) % block_align != 0:
        result.error = (
            "PCM response ended with a partial audio frame "
            f"(bytes={len(pcm_bytes)}, block_align={block_align})"
        )
        return

    result.audio_duration_s = len(pcm_bytes) / bytes_per_second
    elapsed = time.perf_counter() - start_time
    result.rtf = elapsed / result.audio_duration_s
    result.is_success = True
    if save_audio_dir:
        audio_path = os.path.join(save_audio_dir, f"{result.request_id}.wav")
        with open(audio_path, "wb") as fh:
            fh.write(_build_streaming_wav_bytes(pcm_chunks, pcm_format))
        result.wav_path = audio_path


async def _handle_non_streaming_response(
    response: aiohttp.ClientResponse,
    result: RequestResult,
    start_time: float,
    save_audio_dir: str | None,
) -> None:
    audio_bytes = await response.read()
    result.audio_duration_s = get_wav_duration(audio_bytes)
    elapsed = time.perf_counter() - start_time
    if result.audio_duration_s > 0:
        result.is_success = True
        result.rtf = elapsed / result.audio_duration_s
    else:
        result.error = f"Empty or invalid audio response ({len(audio_bytes)} bytes)"
        return
    _parse_response_headers(result, response.headers)
    if save_audio_dir and audio_bytes:
        audio_path = os.path.join(save_audio_dir, f"{result.request_id}.wav")
        with open(audio_path, "wb") as fh:
            fh.write(audio_bytes)
        result.wav_path = audio_path


def _collect_chat_streaming_audio(
    line: str,
    pcm_chunks: list[bytes],
    pcm_format: tuple[int, int, int] | None,
    usage: dict,
    chunk_times_out: list[float] | None = None,
    text_first_time_holder: list[float] | None = None,
) -> tuple[int, int, int] | None:
    event = parse_sse_event(line)
    if event is None:
        return pcm_format

    event_usage = event.get("usage")
    if isinstance(event_usage, dict):
        usage.clear()
        usage.update(event_usage)

    for choice in event.get("choices", []):
        if not isinstance(choice, dict):
            continue
        delta = choice.get("delta")
        if not isinstance(delta, dict):
            continue
        if text_first_time_holder is not None and not text_first_time_holder:
            content = delta.get("content")
            if isinstance(content, str) and content:
                text_first_time_holder.append(time.perf_counter())
        audio = delta.get("audio")
        if not isinstance(audio, dict) or not audio.get("data"):
            continue
        try:
            chunk_bytes = base64.b64decode(audio["data"])
            if len(chunk_bytes) <= WAV_HEADER_SIZE:
                continue
            with io.BytesIO(chunk_bytes) as buf:
                with wave.open(buf, "rb") as wf:
                    pcm_chunks.append(wf.readframes(wf.getnframes()))
                    if chunk_times_out is not None:
                        chunk_times_out.append(time.perf_counter())
                    if pcm_format is None:
                        pcm_format = (
                            wf.getframerate(),
                            wf.getnchannels(),
                            wf.getsampwidth(),
                        )
        except (binascii.Error, wave.Error, EOFError) as exc:
            logger.debug(f"Skipping malformed chat streaming audio chunk: {exc}")
    return pcm_format


def _build_streaming_wav_bytes(
    pcm_chunks: list[bytes],
    pcm_format: tuple[int, int, int],
) -> bytes:
    sample_rate, num_channels, sample_width = pcm_format
    wav_buffer = io.BytesIO()
    with wave.open(wav_buffer, "wb") as wf:
        wf.setframerate(sample_rate)
        wf.setnchannels(num_channels)
        wf.setsampwidth(sample_width)
        wf.writeframes(b"".join(pcm_chunks))
    return wav_buffer.getvalue()


def make_tts_send_fn(
    model_name: str,
    api_url: str,
    *,
    response_format: str = "wav",
    stream: bool = False,
    initial_codec_chunk_frames: int | None = None,
    no_ref_audio: bool = False,
    ref_format: str = "flat",
    voice: str | None = None,
    task_type: str | None = None,
    instructions: str | None = None,
    save_audio_dir: str | None = None,
    **gen_kwargs,
) -> SendFn:
    """Return a *send_fn(session, sample) -> RequestResult* for the runner."""

    async def send_fn(
        session: aiohttp.ClientSession, sample: SampleInput
    ) -> RequestResult:
        result = RequestResult(
            request_id=sample.sample_id,
            text=sample.target_text[:TEXT_PREVIEW_LENGTH],
        )
        payload = _build_tts_payload(
            sample,
            model_name,
            response_format=response_format,
            stream=stream,
            initial_codec_chunk_frames=initial_codec_chunk_frames,
            no_ref_audio=no_ref_audio,
            ref_format=ref_format,
            voice=voice,
            task_type=task_type,
            instructions=instructions,
            **gen_kwargs,
        )
        start_time = time.perf_counter()
        try:
            async with session.post(api_url, json=payload) as response:
                if response.status != 200:
                    result.error = f"HTTP {response.status}: {await response.text()}"
                elif stream:
                    await _handle_raw_pcm_streaming_response(
                        response, result, start_time, save_audio_dir
                    )
                else:
                    await _handle_non_streaming_response(
                        response, result, start_time, save_audio_dir
                    )
        except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
            result.error = str(exc)
        finally:
            result.latency_s = time.perf_counter() - start_time
        return result

    return send_fn


def save_generated_audio_metadata(
    outputs: list[RequestResult],
    samples: list[SampleInput],
    output_dir: str,
) -> None:
    sample_by_id = {sample.sample_id: sample for sample in samples}
    generated = [
        _request_result_to_generated_entry(output, sample_by_id[output.request_id])
        for output in outputs
    ]
    metadata_path = os.path.join(output_dir, "generated.json")
    with open(metadata_path, "w") as fh:
        json.dump(generated, fh, indent=2, ensure_ascii=False)
    logger.info(f"Generated audio metadata saved to {metadata_path}")


def _request_result_to_generated_entry(
    output: RequestResult,
    sample: SampleInput,
) -> dict:
    entry: dict = {
        "sample_id": output.request_id,
        "target_text": sample.target_text,
        "wav_path": output.wav_path,
        "is_success": output.is_success,
        "latency_s": round(output.latency_s, 4),
        "audio_duration_s": round(output.audio_duration_s, 4),
    }
    if output.error:
        entry["error"] = output.error
    return entry


def save_speed_results(
    outputs: list[RequestResult],
    metrics: dict,
    config: dict,
    output_dir: str,
) -> None:
    json_results = build_speed_results(outputs, metrics, config)
    save_json_results(json_results, output_dir, "speed_results.json")

    csv_path = os.path.join(output_dir, "results.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "id",
                "text",
                "latency_s",
                "audio_duration_s",
                "rtf",
                "prompt_tokens",
                "completion_tokens",
                "output_token_rate",
                "audio_ttfp_s",
                "audio_chunk_count",
                "first_audio_payload_bytes",
                "is_success",
                "error",
            ]
        )
        for o in outputs:
            writer.writerow(
                [
                    o.request_id,
                    o.text,
                    f"{o.latency_s:.4f}",
                    f"{o.audio_duration_s:.4f}",
                    f"{o.rtf:.4f}" if o.rtf < float("inf") else "",
                    o.prompt_tokens or "",
                    o.completion_tokens or "",
                    f"{o.tok_per_s:.1f}" if o.tok_per_s > 0 else "",
                    (f"{o.audio_ttfp_s:.4f}" if o.audio_ttfp_s is not None else ""),
                    o.audio_chunk_count or "",
                    o.first_audio_payload_bytes or "",
                    o.is_success,
                    o.error or "",
                ]
            )
    logger.info(f"Results saved to {output_dir}")
