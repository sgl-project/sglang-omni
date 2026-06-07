# SPDX-License-Identifier: Apache-2.0
"""TTS task utilities: voice-clone API clients, ASR evaluation, and HTTP send functions.

Replaces tasks/tts_speed.py and tasks/voice_clone.py.
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import concurrent.futures
import csv
import functools
import io
import json
import logging
import os
import string
import time
import wave
from dataclasses import dataclass
from typing import AsyncIterator, Protocol

import aiohttp
import numpy as np
import requests
import soundfile as sf
import torch
from jiwer import process_words
from tqdm import tqdm

from benchmarks.benchmarker.data import RequestResult
from benchmarks.benchmarker.runner import SendFn
from benchmarks.benchmarker.utils import (
    SSE_DATA_PREFIX,
    SSE_DONE_MARKER,
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
    calculate_asr_speed_metrics,
    calculate_wer_metrics,
    print_asr_speed_summary,
    print_wer_summary,
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
# ASR / WER utilities
# ---------------------------------------------------------------------------


@dataclass
class SampleOutput:
    sample_id: str = ""
    target_text: str = ""
    whisper_text: str = ""
    ref_norm: str = ""
    hyp_norm: str = ""
    wer: float = 0.0
    substitutions: int = 0
    deletions: int = 0
    insertions: int = 0
    hits: int = 0
    audio_duration_s: float = 0.0
    latency_s: float = 0.0
    asr_latency_s: float = 0.0
    is_success: bool = False
    error: str = ""


@functools.lru_cache(maxsize=1)
def _get_en_normalizer():
    """Lazy-load the required English WER normalizer from openai-whisper."""
    try:
        from whisper.normalizers import EnglishTextNormalizer
    except ImportError as exc:
        raise RuntimeError(
            "English WER requires openai-whisper "
            "(whisper.normalizers.EnglishTextNormalizer). "
            "Install pinned deps with `uv pip install -e .`."
        ) from exc

    return EnglishTextNormalizer()


def normalize_text(text: str, lang: str) -> str:
    if lang == "zh":
        from zhon.hanzi import punctuation as zh_punct

        all_punct = zh_punct + string.punctuation
        for ch in all_punct:
            if ch == "'":
                continue
            text = text.replace(ch, "")
        text = text.replace(" ", "").replace("\u3000", "").strip()
        text = " ".join(list(text))
        return text

    normalizer = _get_en_normalizer()
    return normalizer(text)


OMNI_WHISPER_MODEL_PATH = "openai/whisper-large-v3"
OMNI_WHISPER_REQUEST_TIMEOUT_S = 300
# Whisper encoder accepts at most ~30 s per request (nb_max_frames=3000).
# transformers pipeline uses chunk_length_s=30; mirror that for long talker audio.
OMNI_WHISPER_CHUNK_LENGTH_S = 30
OMNI_WHISPER_CHUNK_STRIDE_S = 25
OMNI_WHISPER_SAMPLE_RATE = 16000


def _load_wav_mono_16k(wav_path: str) -> torch.Tensor:
    import torchaudio

    audio, sample_rate = torchaudio.load(wav_path)
    if audio.ndim == 2 and audio.shape[0] > 1:
        audio = audio.mean(dim=0, keepdim=True)
    audio = audio.squeeze(0).to(torch.float32)
    if sample_rate != OMNI_WHISPER_SAMPLE_RATE:
        audio = torchaudio.functional.resample(
            audio, sample_rate, OMNI_WHISPER_SAMPLE_RATE
        )
    return audio


def _wav_bytes_from_mono_16k(audio: torch.Tensor) -> bytes:
    pcm16 = (audio.clamp(-1.0, 1.0) * 32767.0).to(torch.int16).cpu()
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(OMNI_WHISPER_SAMPLE_RATE)
        wav_file.writeframes(pcm16.numpy().tobytes())
    return buffer.getvalue()


def _post_omni_whisper_transcription(
    asr: dict,
    audio_bytes: bytes,
    filename: str,
    lang: str,
) -> str:
    response = requests.post(
        f"http://127.0.0.1:{asr['router_port']}/v1/audio/transcriptions",
        data={
            "model": asr["model_path"],
            "language": "en" if lang == "en" else lang,
            "temperature": "0",
        },
        files={
            "file": (
                filename,
                audio_bytes,
                "audio/wav",
            )
        },
        timeout=OMNI_WHISPER_REQUEST_TIMEOUT_S,
        proxies={"http": None, "https": None},
    )
    response.raise_for_status()
    return str(response.json()["text"])


def _transcribe_omni_whisper(asr: dict, wav_path: str, lang: str) -> str:
    audio = _load_wav_mono_16k(wav_path)
    duration_s = float(audio.shape[0]) / OMNI_WHISPER_SAMPLE_RATE
    if duration_s <= OMNI_WHISPER_CHUNK_LENGTH_S:
        with open(wav_path, "rb") as audio_file:
            return _post_omni_whisper_transcription(
                asr,
                audio_file.read(),
                os.path.basename(wav_path),
                lang,
            )

    chunk_samples = OMNI_WHISPER_CHUNK_LENGTH_S * OMNI_WHISPER_SAMPLE_RATE
    stride_samples = OMNI_WHISPER_CHUNK_STRIDE_S * OMNI_WHISPER_SAMPLE_RATE
    chunk_texts: list[str] = []
    for start in range(0, int(audio.shape[0]), stride_samples):
        chunk = audio[start : start + chunk_samples]
        if chunk.numel() == 0:
            break
        chunk_bytes = _wav_bytes_from_mono_16k(chunk)
        chunk_texts.append(
            _post_omni_whisper_transcription(
                asr,
                chunk_bytes,
                f"{os.path.basename(wav_path)}.chunk{start // stride_samples}.wav",
                lang,
            ).strip()
        )
        if start + chunk_samples >= audio.shape[0]:
            break
    return " ".join(text for text in chunk_texts if text)


def load_omni_whisper_asr(
    router_port: int,
    *,
    model_path: str = OMNI_WHISPER_MODEL_PATH,
) -> dict:
    """Return an ASR handle that transcribes via SGLang Omni Whisper router."""
    return {
        "type": "omni_whisper",
        "router_port": router_port,
        "model_path": model_path,
    }


QWEN3_ASR_MODEL_PATH = os.getenv("QWEN3_ASR_MODEL_PATH", "Qwen/Qwen3-ASR-1.7B")
QWEN3_ASR_REQUEST_TIMEOUT_S = 300
QWEN3_ASR_MAX_NEW_TOKENS = int(os.getenv("QWEN3_ASR_MAX_NEW_TOKENS", "128"))
# ASR transcription fan-out for WER, not TTS generation concurrency.
DEFAULT_ASR_TRANSCRIBE_CONCURRENCY = int(
    os.getenv("QWEN3_ASR_CONCURRENCY", os.getenv("SEEDTTS_ASR_CONCURRENCY", "32"))
)


def load_qwen3_asr(
    router_port: int,
    *,
    model_path: str = QWEN3_ASR_MODEL_PATH,
) -> dict:
    """Return an ASR handle that transcribes via a Qwen3-ASR sglang-omni router."""
    return {
        "type": "qwen3_asr",
        "router_port": router_port,
        "model_path": model_path,
    }


def _is_whisper_asr_model(model_path: str) -> bool:
    return "whisper" in model_path.lower()


def load_router_asr(
    router_port: int,
    *,
    model_path: str = QWEN3_ASR_MODEL_PATH,
) -> dict:
    """Return an ASR handle backed by a running SGLang Omni ASR server."""
    if _is_whisper_asr_model(model_path):
        return load_omni_whisper_asr(router_port, model_path=model_path)
    return load_qwen3_asr(router_port, model_path=model_path)


def _transcribe_qwen3_asr(asr: dict, wav_path: str, lang: str) -> str:
    """Transcribe one wav via the Qwen3-ASR server's /v1/audio/transcriptions.

    Note: do NOT send temperature=0 — Qwen3-ASR degenerates under pure greedy
    (the server bumps it to 0.01). ``language`` selects the forced prefix.
    """
    with open(wav_path, "rb") as audio_file:
        response = requests.post(
            f"http://127.0.0.1:{asr['router_port']}/v1/audio/transcriptions",
            data={
                "model": asr["model_path"],
                "language": "en" if lang == "en" else lang,
                "response_format": "json",
                "max_new_tokens": str(QWEN3_ASR_MAX_NEW_TOKENS),
            },
            files={"file": (os.path.basename(wav_path), audio_file, "audio/wav")},
            timeout=QWEN3_ASR_REQUEST_TIMEOUT_S,
            proxies={"http": None, "https": None},
        )
    response.raise_for_status()
    return str(response.json()["text"])


def _resolve_asr_backend(
    lang: str,
    asr_device: str,
    *,
    asr_router_port: int | None = None,
    asr_model_path: str = QWEN3_ASR_MODEL_PATH,
    generation_mode: str | None = None,
) -> dict:
    if asr_router_port is not None:
        return load_router_asr(asr_router_port, model_path=asr_model_path)
    return load_asr_model(lang, asr_device, generation_mode)


def load_asr_model(lang: str, device: str, generation_mode: str | None = None):
    """Legacy local ASR entry point.

    WER now runs through SGLang Omni's OpenAI-compatible transcription endpoint.
    Start an ASR server with ``sglang_omni.cli serve`` and pass its port instead
    of loading local ASR backends in-process.
    """
    mode_suffix = f" for {generation_mode} generation" if generation_mode else ""
    del device
    if lang not in {"en", "zh"}:
        raise ValueError(f"Unsupported language: {lang}")
    raise ValueError(
        "WER transcription requires a running SGLang Omni ASR server"
        f"{mode_suffix}. Start Qwen3-ASR (default "
        f"{QWEN3_ASR_MODEL_PATH}) or {OMNI_WHISPER_MODEL_PATH} and pass "
        "asr_router_port."
    )


def transcribe(asr: dict, wav_path: str, lang: str, device: str) -> str:
    if asr["type"] == "qwen3_asr":
        return _transcribe_qwen3_asr(asr, wav_path, lang)
    if asr["type"] == "omni_whisper":
        return _transcribe_omni_whisper(asr, wav_path, lang)
    raise ValueError(f"Unknown ASR type: {asr['type']}")


def transcribe_and_compute_wer(
    output: SampleOutput,
    wav_path: str,
    asr: dict,
    lang: str,
    device: str,
) -> SampleOutput:
    """Transcribe audio and compute per-sample WER metrics."""
    try:
        hyp_text = transcribe(asr, wav_path, lang, device)
    except Exception as exc:
        output.error = f"Transcription failed: {exc}"
        logger.error(f"[{output.sample_id}] {output.error}")
        return output

    output.whisper_text = hyp_text
    output.ref_norm = normalize_text(output.target_text, lang)
    output.hyp_norm = normalize_text(hyp_text, lang)

    if not output.ref_norm:
        output.error = "Empty reference after normalization"
        return output

    measures = process_words(output.ref_norm, output.hyp_norm)
    output.wer = measures.wer
    output.substitutions = measures.substitutions
    output.deletions = measures.deletions
    output.insertions = measures.insertions
    output.hits = measures.hits
    output.is_success = True
    return output


def compute_text_audio_consistency(
    request_results: list[RequestResult],
    lang: str,
    asr_device: str,
    *,
    asr_router_port: int | None = None,
    asr_model_path: str = QWEN3_ASR_MODEL_PATH,
    asr_concurrency: int = DEFAULT_ASR_TRANSCRIBE_CONCURRENCY,
) -> dict:
    """WER between each request's text output (ref) and ASR-transcribed audio (hyp)."""
    asr = _resolve_asr_backend(
        lang,
        asr_device,
        asr_router_port=asr_router_port,
        asr_model_path=asr_model_path,
    )

    outputs_by_idx: list[SampleOutput | None] = [None] * len(request_results)
    pending: list[tuple[int, RequestResult, SampleOutput]] = []
    for idx, result in enumerate(request_results):
        ref_text = " ".join(result.text.split())
        out = SampleOutput(
            sample_id=result.request_id,
            target_text=ref_text,
            latency_s=result.latency_s,
            audio_duration_s=result.audio_duration_s,
        )
        if not result.is_success or not result.wav_path:
            out.error = result.error or "No audio in response"
            outputs_by_idx[idx] = out
            continue
        pending.append((idx, result, out))

    def _transcribe_pending(
        result: RequestResult,
        output: SampleOutput,
    ) -> SampleOutput:
        asr_t0 = time.perf_counter()
        output = transcribe_and_compute_wer(
            output,
            result.wav_path,
            asr,
            lang,
            asr_device,
        )
        output.asr_latency_s = time.perf_counter() - asr_t0
        return output

    asr_concurrency = max(1, int(asr_concurrency))
    if asr_concurrency == 1:
        for idx, result, output in pending:
            outputs_by_idx[idx] = _transcribe_pending(result, output)
    else:
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=asr_concurrency,
        ) as executor:
            future_to_idx = {
                executor.submit(_transcribe_pending, result, output): idx
                for idx, result, output in pending
            }
            for future in concurrent.futures.as_completed(future_to_idx):
                outputs_by_idx[future_to_idx[future]] = future.result()

    outputs = [output for output in outputs_by_idx if output is not None]

    per_sample = [
        {
            "id": o.sample_id,
            "is_success": o.is_success,
            "wer": o.wer if o.is_success else None,
            "ref_text": o.target_text[:100],
            "hyp_text": o.whisper_text[:100],
            "ref_norm": o.ref_norm,
            "hyp_norm": o.hyp_norm,
            "audio_duration_s": o.audio_duration_s,
            "error": o.error,
        }
        for o in outputs
    ]
    return {"summary": calculate_wer_metrics(outputs, lang), "per_sample": per_sample}


def compute_text_audio_consistency_from_records(
    per_sample: list[dict],
    lang: str,
    asr_device: str,
    *,
    audio_dir: str | None = None,
    sample_id_key: str = "sample_id",
    text_key: str = "raw_response",
    asr_router_port: int | None = None,
    asr_model_path: str = QWEN3_ASR_MODEL_PATH,
    asr_concurrency: int = DEFAULT_ASR_TRANSCRIBE_CONCURRENCY,
) -> dict:
    """Compute WER from saved eval records after the inference server is stopped."""
    request_results: list[RequestResult] = []
    for record in per_sample:
        sample_id = record.get(sample_id_key)
        wav_path = record.get("wav_path") or ""
        if not wav_path and audio_dir and sample_id:
            wav_path = os.path.join(audio_dir, f"{sample_id}.wav")
        request_results.append(
            RequestResult(
                request_id=str(sample_id or ""),
                text=str(record.get(text_key) or ""),
                is_success=bool(record.get("is_success")),
                latency_s=float(record.get("latency_s") or 0),
                audio_duration_s=float(record.get("audio_duration_s") or 0),
                wav_path=wav_path,
                error=str(record.get("error") or ""),
            )
        )
    return compute_text_audio_consistency(
        request_results,
        lang,
        asr_device,
        asr_router_port=asr_router_port,
        asr_model_path=asr_model_path,
        asr_concurrency=asr_concurrency,
    )


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


def _transcribe_one_entry(
    entry: dict,
    asr: dict,
    lang: str,
    device: str,
) -> SampleOutput:
    """Transcribe a single ``generated.json`` entry and compute its WER."""
    output = SampleOutput(
        sample_id=entry["sample_id"],
        target_text=entry["target_text"],
    )
    if not entry.get("is_success", False):
        output.error = f"Generation failed: {entry.get('error', 'unknown')}"
        return output

    output.latency_s = entry.get("latency_s", 0.0)
    output.audio_duration_s = entry.get("audio_duration_s", 0.0)
    asr_t0 = time.perf_counter()
    output = transcribe_and_compute_wer(output, entry["wav_path"], asr, lang, device)
    output.asr_latency_s = time.perf_counter() - asr_t0
    return output


def _resolve_asr_concurrency(config: SeedttsTranscribeConfig) -> int:
    value = getattr(config, "asr_concurrency", DEFAULT_ASR_TRANSCRIBE_CONCURRENCY)
    if value is None:
        value = 1
    return max(1, int(value))


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

    # Only warn for post-generation transcription failures; generation
    # failures are surfaced at speed-benchmark time and already logged.
    if entry.get("is_success", False):
        logger.warning(
            f"[{idx + 1}/{total}] Transcription failed: "
            f"{entry['sample_id']} -- {output.error}",
        )


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

    asr_model_path = getattr(config, "asr_model_path", QWEN3_ASR_MODEL_PATH)
    asr = _resolve_asr_backend(
        config.lang,
        config.device,
        asr_router_port=asr_router_port,
        asr_model_path=asr_model_path,
        generation_mode=generation_mode,
    )

    tqdm_desc = (
        f"Transcribing ({config.lang})" if not generation_mode else "WER transcribe"
    )
    asr_concurrency = _resolve_asr_concurrency(config)
    outputs_by_idx: list[SampleOutput | None] = [None] * len(generated)
    asr_wall_start_s = time.perf_counter()
    if asr_concurrency == 1:
        for idx, entry in enumerate(tqdm(generated, desc=tqdm_desc)):
            output = _transcribe_one_entry(entry, asr, config.lang, config.device)
            outputs_by_idx[idx] = output
            _log_transcribe_result(
                idx=idx,
                total=len(generated),
                entry=entry,
                output=output,
                log_per_sample=log_per_sample,
            )
    else:
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=asr_concurrency,
        ) as executor:
            future_to_idx = {
                executor.submit(
                    _transcribe_one_entry,
                    entry,
                    asr,
                    config.lang,
                    config.device,
                ): idx
                for idx, entry in enumerate(generated)
            }
            with tqdm(total=len(generated), desc=tqdm_desc) as progress:
                for future in concurrent.futures.as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    entry = generated[idx]
                    output = future.result()
                    outputs_by_idx[idx] = output
                    _log_transcribe_result(
                        idx=idx,
                        total=len(generated),
                        entry=entry,
                        output=output,
                        log_per_sample=log_per_sample,
                    )
                    progress.update(1)

    asr_wall_time_s = time.perf_counter() - asr_wall_start_s
    outputs = [output for output in outputs_by_idx if output is not None]

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
        """Generate speech via streaming SSE, concatenate audio chunks into WAV."""
        payload: dict = {
            "model": model_name,
            "input": sample.target_text,
            "ref_audio": sample.ref_audio,
            "ref_text": sample.ref_text,
            "response_format": "wav",
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "stream": True,
        }
        if seed is not None:
            payload["seed"] = seed

        t0 = time.perf_counter()
        pcm_chunks: list[bytes] = []
        sample_rate = None
        num_channels = None
        sample_width = None

        async with session.post(api_url, json=payload) as response:
            if response.status != 200:
                error_text = await response.text()
                raise RuntimeError(f"HTTP {response.status}: {error_text}")

            buffer = bytearray()
            async for chunk in response.content.iter_any():
                buffer.extend(chunk)
                while b"\n" in buffer:
                    idx = buffer.index(b"\n")
                    raw_line = bytes(buffer[:idx])
                    del buffer[: idx + 1]
                    line = raw_line.decode("utf-8", errors="replace").strip()
                    if not line.startswith(SSE_DATA_PREFIX) or line == SSE_DONE_MARKER:
                        continue
                    try:
                        event = json.loads(line[len(SSE_DATA_PREFIX) :])
                    except json.JSONDecodeError:
                        continue
                    audio = event.get("audio")
                    if not isinstance(audio, dict) or not audio.get("data"):
                        continue
                    chunk_bytes = base64.b64decode(audio["data"])
                    if len(chunk_bytes) <= WAV_HEADER_SIZE:
                        continue
                    try:
                        with io.BytesIO(chunk_bytes) as buf:
                            with wave.open(buf, "rb") as wf:
                                sr = wf.getframerate()
                                ch = wf.getnchannels()
                                sw = wf.getsampwidth()
                                pcm = wf.readframes(wf.getnframes())
                        if sample_rate is None:
                            sample_rate, num_channels, sample_width = sr, ch, sw
                        pcm_chunks.append(pcm)
                    except Exception:
                        continue

            if buffer.strip():
                line = bytes(buffer).decode("utf-8", errors="replace").strip()
                if line.startswith(SSE_DATA_PREFIX) and line != SSE_DONE_MARKER:
                    try:
                        event = json.loads(line[len(SSE_DATA_PREFIX) :])
                        audio = event.get("audio")
                        if isinstance(audio, dict) and audio.get("data"):
                            chunk_bytes = base64.b64decode(audio["data"])
                            if len(chunk_bytes) > WAV_HEADER_SIZE:
                                with io.BytesIO(chunk_bytes) as buf:
                                    with wave.open(buf, "rb") as wf:
                                        pcm = wf.readframes(wf.getnframes())
                                        if sample_rate is None:
                                            sample_rate = wf.getframerate()
                                            num_channels = wf.getnchannels()
                                            sample_width = wf.getsampwidth()
                                pcm_chunks.append(pcm)
                    except (json.JSONDecodeError, Exception) as exc:
                        logger.debug(
                            "Failed to parse trailing SSE audio chunk: %s", exc
                        )

        latency = time.perf_counter() - t0

        if not pcm_chunks or sample_rate is None:
            raise ValueError("No audio chunks received from streaming response")

        wav_buf = io.BytesIO()
        with wave.open(wav_buf, "wb") as wf:
            wf.setnchannels(num_channels)
            wf.setsampwidth(sample_width)
            wf.setframerate(sample_rate)
            wf.writeframes(b"".join(pcm_chunks))
        return wav_buf.getvalue(), latency

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
        stream_format: tuple[int, int, int] | None = None
        usage: dict = {}
        buffer = bytearray()

        async for chunk in response.content.iter_any():
            buffer.extend(chunk)
            while b"\n" in buffer:
                idx = buffer.index(b"\n")
                raw_line = bytes(buffer[:idx])
                del buffer[: idx + 1]
                stream_format = _collect_chat_streaming_audio(
                    raw_line.decode("utf-8", errors="replace").strip(),
                    pcm_chunks,
                    stream_format,
                    usage,
                    chunk_times_out=chunk_times_out,
                    text_first_time_holder=text_first_time_holder,
                )

        if buffer.strip():
            stream_format = _collect_chat_streaming_audio(
                bytes(buffer).decode("utf-8", errors="replace").strip(),
                pcm_chunks,
                stream_format,
                usage,
                chunk_times_out=chunk_times_out,
                text_first_time_holder=text_first_time_holder,
            )

        if not pcm_chunks or stream_format is None:
            raise ValueError("No audio chunks received from streaming response")
        return _build_streaming_wav_bytes(pcm_chunks, stream_format), usage

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
    stream_format: str = "sse",
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
        "response_format": (
            "pcm" if stream and stream_format == "audio" else response_format
        ),
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
        payload["stream_format"] = stream_format
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


async def _handle_streaming_response(
    response: aiohttp.ClientResponse,
    result: RequestResult,
    start_time: float,
    save_audio_dir: str | None,
) -> None:
    total_audio_duration = 0.0
    usage_data: dict | None = None
    buffer = bytearray()
    pcm_chunks: list[bytes] = []
    chunk_times: list[float] = []
    stream_format: tuple[int, int, int] | None = None
    async for chunk in response.content.iter_any():
        buffer.extend(chunk)
        while b"\n" in buffer:
            idx = buffer.index(b"\n")
            raw_line = bytes(buffer[:idx])
            del buffer[: idx + 1]
            line = raw_line.decode("utf-8", errors="replace").strip()
            stream_format, duration_delta, usage_update = _collect_streaming_audio(
                line, pcm_chunks, stream_format, chunk_times
            )
            total_audio_duration += duration_delta
            if usage_update is not None:
                usage_data = usage_update
    if buffer.strip():
        line = bytes(buffer).decode("utf-8", errors="replace").strip()
        stream_format, duration_delta, usage_update = _collect_streaming_audio(
            line, pcm_chunks, stream_format, chunk_times
        )
        total_audio_duration += duration_delta
        if usage_update is not None:
            usage_data = usage_update
    result.audio_duration_s = total_audio_duration
    if total_audio_duration > 0:
        elapsed = time.perf_counter() - start_time
        result.rtf = elapsed / total_audio_duration
    if chunk_times:
        result.audio_ttfp_s = chunk_times[0] - start_time
        result.inter_chunk_s = [
            now - prev for prev, now in zip(chunk_times, chunk_times[1:])
        ]
    result.audio_chunk_count = len(pcm_chunks)
    if pcm_chunks:
        result.first_audio_payload_bytes = len(pcm_chunks[0])
    result.is_success = total_audio_duration > 0
    if save_audio_dir and pcm_chunks and stream_format is not None:
        audio_path = os.path.join(save_audio_dir, f"{result.request_id}.wav")
        with open(audio_path, "wb") as fh:
            fh.write(_build_streaming_wav_bytes(pcm_chunks, stream_format))
        result.wav_path = audio_path
    if usage_data:
        prompt_tok = usage_data.get("prompt_tokens")
        comp_tok = usage_data.get("completion_tokens")
        eng_time = usage_data.get("engine_time_s")
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
    sample_width = max(1, bit_depth // 8)
    return sample_rate, num_channels, sample_width


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
    stream_format = _parse_pcm_response_format(response.headers)
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
    sample_rate, num_channels, sample_width = stream_format
    bytes_per_second = sample_rate * num_channels * sample_width
    if not pcm_bytes or bytes_per_second <= 0:
        result.error = f"Empty or invalid PCM response ({len(pcm_bytes)} bytes)"
        return

    result.audio_duration_s = len(pcm_bytes) / bytes_per_second
    elapsed = time.perf_counter() - start_time
    result.rtf = elapsed / result.audio_duration_s
    result.is_success = True
    if save_audio_dir:
        audio_path = os.path.join(save_audio_dir, f"{result.request_id}.wav")
        with open(audio_path, "wb") as fh:
            fh.write(_build_streaming_wav_bytes(pcm_chunks, stream_format))
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


def _collect_streaming_audio(
    line: str,
    pcm_chunks: list[bytes],
    stream_format: tuple[int, int, int] | None,
    chunk_times_out: list[float] | None = None,
) -> tuple[tuple[int, int, int] | None, float, dict | None]:
    event = parse_sse_event(line)
    if event is None:
        return stream_format, 0.0, None

    usage = event.get("usage")
    usage_update = usage if isinstance(usage, dict) else None

    audio = event.get("audio")
    if not isinstance(audio, dict) or not audio.get("data"):
        return stream_format, 0.0, usage_update

    try:
        chunk_bytes = base64.b64decode(audio["data"])
        audio_format = str(audio.get("format") or "").lower()
        mime_type = str(audio.get("mime_type") or "").lower()
        if audio_format == "pcm" or mime_type == "audio/pcm":
            sample_rate = int(audio.get("sample_rate") or 24000)
            pcm_chunks.append(chunk_bytes)
            if chunk_times_out is not None:
                chunk_times_out.append(time.perf_counter())
            chunk_format = (sample_rate, 1, 2)
            if stream_format is None:
                stream_format = chunk_format
            elif stream_format != chunk_format:
                raise ValueError(
                    f"Streaming PCM format changed from {stream_format} "
                    f"to {chunk_format}"
                )
            duration = len(chunk_bytes) / (sample_rate * 2) if sample_rate > 0 else 0.0
            return stream_format, duration, usage_update

        if len(chunk_bytes) <= WAV_HEADER_SIZE:
            return stream_format, 0.0, usage_update
        with io.BytesIO(chunk_bytes) as buf:
            with wave.open(buf, "rb") as wf:
                frames = wf.readframes(wf.getnframes())
                pcm_chunks.append(frames)
                if chunk_times_out is not None:
                    chunk_times_out.append(time.perf_counter())
                sample_rate = wf.getframerate()
                num_channels = wf.getnchannels()
                sample_width = wf.getsampwidth()
                duration = wf.getnframes() / sample_rate if sample_rate > 0 else 0.0
                chunk_format = (sample_rate, num_channels, sample_width)
                if stream_format is None:
                    stream_format = chunk_format
                elif stream_format != chunk_format:
                    raise ValueError(
                        f"Streaming WAV format changed from {stream_format} "
                        f"to {chunk_format}"
                    )
                return stream_format, duration, usage_update
    except (binascii.Error, wave.Error, EOFError) as exc:
        logger.debug(f"Skipping malformed streaming audio chunk: {exc}")
        return stream_format, 0.0, usage_update


def _collect_chat_streaming_audio(
    line: str,
    pcm_chunks: list[bytes],
    stream_format: tuple[int, int, int] | None,
    usage: dict,
    chunk_times_out: list[float] | None = None,
    text_first_time_holder: list[float] | None = None,
) -> tuple[int, int, int] | None:
    event = parse_sse_event(line)
    if event is None:
        return stream_format

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
                    if stream_format is None:
                        stream_format = (
                            wf.getframerate(),
                            wf.getnchannels(),
                            wf.getsampwidth(),
                        )
        except (binascii.Error, wave.Error, EOFError) as exc:
            logger.debug(f"Skipping malformed chat streaming audio chunk: {exc}")
    return stream_format


def _build_streaming_wav_bytes(
    pcm_chunks: list[bytes],
    stream_format: tuple[int, int, int],
) -> bytes:
    sample_rate, num_channels, sample_width = stream_format
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
    stream_format: str = "sse",
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
            stream_format=stream_format,
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
                elif stream and stream_format == "audio":
                    await _handle_raw_pcm_streaming_response(
                        response, result, start_time, save_audio_dir
                    )
                elif stream:
                    await _handle_streaming_response(
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
