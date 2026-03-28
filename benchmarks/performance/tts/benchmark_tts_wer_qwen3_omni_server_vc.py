#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""WER evaluation for Qwen3-Omni TTS **with voice cloning** via sglang-omni server.

Same as benchmark_tts_wer_qwen3_omni_server.py, but passes reference audio
from seed-tts-eval to the server via the top-level ``audios`` field of the
/v1/chat/completions request.  The prompt asks the model to read the target
text in the same voice as the reference audio.

Usage:

    python -m benchmarks.performance.tts.benchmark_tts_wer_qwen3_omni_server_vc \
        --meta seedtts_testset/en/meta.lst \
        --output-dir results/qwen3_omni_server_vc_en_full \
        --lang en --port 8000
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import csv
import json
import logging
import os
import string
import subprocess
import time
from dataclasses import dataclass

import aiohttp
import numpy as np
import scipy.signal
import soundfile as sf
import torch
from jiwer import process_words
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

SUMMARY_LABEL_WIDTH = 30
SUMMARY_LINE_WIDTH = 60
QWEN3_OMNI_SAMPLE_RATE = 24000
THINKER_MAX_NEW_TOKENS = 256


@dataclass
class SampleInput:
    sample_id: str
    ref_text: str
    ref_audio: str
    target_text: str


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
    is_success: bool = False
    error: str = ""


def download_dataset(local_dir: str = "seedtts_testset") -> None:
    """Download the seed-tts-eval dataset from HuggingFace if not present."""
    meta_en = os.path.join(local_dir, "en", "meta.lst")
    if os.path.exists(meta_en):
        logger.info("Dataset already exists at %s, skipping download.", local_dir)
        return

    logger.info("Downloading seed-tts-eval dataset to %s ...", local_dir)
    cmd = [
        "huggingface-cli",
        "download",
        "zhaochenyang20/seed-tts-eval",
        "--repo-type",
        "dataset",
        "--local-dir",
        local_dir,
    ]
    subprocess.run(cmd, check=True)
    logger.info("Dataset downloaded to %s", local_dir)


def parse_meta_lst(
    path: str, max_samples: int | None = None, audio_base_dir: str | None = None
) -> list[SampleInput]:
    base_dir = audio_base_dir or os.path.dirname(path)
    samples = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("|")
            if len(parts) < 4:
                continue
            samples.append(
                SampleInput(
                    sample_id=parts[0],
                    ref_text=parts[1],
                    ref_audio=os.path.join(base_dir, parts[2]),
                    target_text=parts[3],
                )
            )
            if max_samples and len(samples) >= max_samples:
                break
    return samples


# ---------------------------------------------------------------------------
# Text normalization (same as other benchmark scripts)
# ---------------------------------------------------------------------------

_EN_NORMALIZER_UNLOADED = object()
_en_normalizer = _EN_NORMALIZER_UNLOADED


def _get_en_normalizer():
    global _en_normalizer
    if _en_normalizer is not _EN_NORMALIZER_UNLOADED:
        return _en_normalizer

    try:
        from whisper_normalizer.english import EnglishTextNormalizer

        _en_normalizer = EnglishTextNormalizer()
        logger.info("Using whisper_normalizer.english.EnglishTextNormalizer")
        return _en_normalizer
    except (ImportError, TypeError):
        pass

    try:
        from whisper.normalizers import EnglishTextNormalizer

        _en_normalizer = EnglishTextNormalizer()
        logger.info("Using whisper.normalizers.EnglishTextNormalizer")
        return _en_normalizer
    except (ImportError, TypeError):
        pass

    try:
        import json as _json
        from pathlib import Path

        import transformers
        from transformers.models.whisper.english_normalizer import (
            EnglishTextNormalizer,
        )

        json_path = (
            Path(transformers.__file__).parent / "models" / "whisper" / "english.json"
        )
        with open(json_path) as f:
            english_spelling_mapping = _json.load(f)

        _en_normalizer = EnglishTextNormalizer(english_spelling_mapping)
        logger.info(
            "Using transformers.models.whisper.english_normalizer.EnglishTextNormalizer"
        )
        return _en_normalizer
    except (ImportError, AttributeError, FileNotFoundError, TypeError) as exc:
        logger.debug("transformers EnglishTextNormalizer failed: %s", exc)

    _en_normalizer = None
    logger.warning(
        "EnglishTextNormalizer not found; falling back to simple normalizer."
    )
    return _en_normalizer


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
    if normalizer is not None:
        return normalizer(text)

    for ch in string.punctuation:
        if ch == "'":
            continue
        text = text.replace(ch, "")
    text = text.replace("  ", " ").strip().lower()
    return text


# ---------------------------------------------------------------------------
# ASR
# ---------------------------------------------------------------------------


def load_asr_model(lang: str, device: str):
    if lang == "en":
        from transformers import WhisperForConditionalGeneration, WhisperProcessor

        logger.info("Loading Whisper-large-v3 on %s...", device)
        t0 = time.perf_counter()
        processor = WhisperProcessor.from_pretrained("openai/whisper-large-v3")
        model = WhisperForConditionalGeneration.from_pretrained(
            "openai/whisper-large-v3"
        ).to(device)
        logger.info("Whisper loaded in %.1fs", time.perf_counter() - t0)
        return {"type": "whisper", "processor": processor, "model": model}
    elif lang == "zh":
        from funasr import AutoModel

        logger.info("Loading FunASR paraformer-zh...")
        t0 = time.perf_counter()
        model = AutoModel(model="paraformer-zh")
        logger.info("FunASR loaded in %.1fs", time.perf_counter() - t0)
        return {"type": "funasr", "model": model}
    else:
        raise ValueError(f"Unsupported language: {lang}")


def transcribe(asr: dict, wav_path: str, lang: str, device: str) -> str:
    if asr["type"] == "whisper":
        processor = asr["processor"]
        model = asr["model"]
        wav, sr = sf.read(wav_path)
        if sr != 16000:
            wav = scipy.signal.resample(wav, int(len(wav) * 16000 / sr))
        input_features = processor(
            wav, sampling_rate=16000, return_tensors="pt"
        ).input_features.to(device)
        forced_decoder_ids = processor.get_decoder_prompt_ids(
            language="english", task="transcribe"
        )
        with torch.no_grad():
            predicted_ids = model.generate(
                input_features, forced_decoder_ids=forced_decoder_ids
            )
        return processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    elif asr["type"] == "funasr":
        import zhconv

        res = asr["model"].generate(input=wav_path, batch_size_s=300)
        transcription = res[0]["text"]
        return zhconv.convert(transcription, "zh-cn")
    else:
        raise ValueError(f"Unknown ASR type: {asr['type']}")


# ---------------------------------------------------------------------------
# Server-based speech generation via /v1/chat/completions
# ---------------------------------------------------------------------------


async def generate_speech_server(
    session: aiohttp.ClientSession,
    api_url: str,
    model_name: str,
    sample: SampleInput,
    lang: str,
    speaker: str = "Ethan",
    max_tokens: int = THINKER_MAX_NEW_TOKENS,
) -> tuple[bytes, float]:
    """Generate speech via sglang-omni server chat completions API.

    Returns (wav_bytes, latency_seconds).

    Voice cloning: the reference audio is passed via the top-level
    ``audios`` field.  The preprocessor inserts an ``<audio>``
    placeholder into the last user message automatically.
    """
    if lang == "en":
        prompt_text = (
            f"Listen to the audio above. The speaker is reading: \"{sample.ref_text}\". "
            f"Now please read the following text out loud in the same voice and style: "
            f"{sample.target_text}"
        )
    else:
        prompt_text = (
            f"听上面的音频，说话人正在朗读：\"{sample.ref_text}\"。"
            f"现在请用同样的声音和风格朗读以下文本：{sample.target_text}"
        )

    payload = {
        "model": model_name,
        "messages": [
            {"role": "user", "content": prompt_text},
        ],
        "audios": [sample.ref_audio],
        "modalities": ["text", "audio"],
        "audio": {"format": "wav"},
        "max_tokens": max_tokens,
        "temperature": 0.7,
        "stream": False,
    }

    t0 = time.perf_counter()
    async with session.post(api_url, json=payload) as response:
        if response.status != 200:
            error_text = await response.text()
            raise RuntimeError(f"HTTP {response.status}: {error_text}")
        resp_json = await response.json()
    latency = time.perf_counter() - t0

    # Extract audio from response
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
    return wav_bytes, latency


async def evaluate_sample(
    session: aiohttp.ClientSession,
    api_url: str,
    model_name: str,
    asr: dict,
    sample: SampleInput,
    lang: str,
    asr_device: str,
    audio_dir: str,
    speaker: str,
    max_tokens: int,
) -> SampleOutput:
    output = SampleOutput(
        sample_id=sample.sample_id,
        target_text=sample.target_text,
    )

    wav_path = os.path.join(audio_dir, f"{sample.sample_id}.wav")

    # Generate audio via server
    try:
        wav_bytes, latency = await generate_speech_server(
            session, api_url, model_name, sample, lang, speaker, max_tokens
        )
        with open(wav_path, "wb") as f:
            f.write(wav_bytes)

        output.latency_s = round(latency, 4)
        wav_info = sf.info(wav_path)
        output.audio_duration_s = round(wav_info.duration, 4)
    except Exception as exc:
        output.error = f"Generation failed: {exc}"
        logger.error("[%s] %s", sample.sample_id, output.error)
        return output

    # Transcribe
    try:
        hyp_text = transcribe(asr, wav_path, lang, asr_device)
    except Exception as exc:
        output.error = f"Transcription failed: {exc}"
        logger.error("[%s] %s", sample.sample_id, output.error)
        return output

    output.whisper_text = hyp_text
    output.ref_norm = normalize_text(sample.target_text, lang)
    output.hyp_norm = normalize_text(hyp_text, lang)

    if not output.ref_norm:
        output.error = "Empty reference after normalization"
        return output

    # Compute WER
    measures = process_words(output.ref_norm, output.hyp_norm)
    output.wer = measures.wer
    output.substitutions = measures.substitutions
    output.deletions = measures.deletions
    output.insertions = measures.insertions
    output.hits = measures.hits
    output.is_success = True

    return output


# ---------------------------------------------------------------------------
# Micro-average WER aggregation
# ---------------------------------------------------------------------------


def calculate_metrics(outputs: list[SampleOutput], lang: str) -> dict:
    successes = [o for o in outputs if o.is_success]
    if not successes:
        return {"completed": 0, "failed": len(outputs)}

    total_errors = sum(o.substitutions + o.deletions + o.insertions for o in successes)
    total_ref_words = sum(o.substitutions + o.deletions + o.hits for o in successes)
    corpus_wer = total_errors / total_ref_words if total_ref_words > 0 else 0.0

    wer_arr = np.array([o.wer for o in successes])
    latencies = [o.latency_s for o in successes]
    audio_durations = [o.audio_duration_s for o in successes if o.audio_duration_s > 0]

    n_above_50 = int(np.sum(wer_arr > 0.5))

    ok_samples = [o for o in successes if o.wer <= 0.5]
    if ok_samples:
        ok_errors = sum(
            o.substitutions + o.deletions + o.insertions for o in ok_samples
        )
        ok_ref = sum(o.substitutions + o.deletions + o.hits for o in ok_samples)
        wer_below_50_micro = ok_errors / ok_ref if ok_ref > 0 else 0.0
    else:
        wer_below_50_micro = 0.0

    return {
        "lang": lang,
        "total_samples": len(outputs),
        "evaluated": len(successes),
        "skipped": len(outputs) - len(successes),
        "wer_corpus": float(corpus_wer),
        "wer_per_sample_mean": float(np.mean(wer_arr)),
        "wer_per_sample_median": float(np.median(wer_arr)),
        "wer_per_sample_std": float(np.std(wer_arr)),
        "wer_per_sample_p95": float(np.percentile(wer_arr, 95)),
        "wer_below_50_corpus": float(wer_below_50_micro),
        "n_above_50_pct_wer": n_above_50,
        "pct_above_50_pct_wer": n_above_50 / len(successes) * 100 if successes else 0,
        "latency_mean_s": float(np.mean(latencies)),
        "audio_duration_mean_s": (
            float(np.mean(audio_durations)) if audio_durations else 0
        ),
    }


def print_summary(metrics: dict, args: argparse.Namespace) -> None:
    lw = SUMMARY_LABEL_WIDTH
    w = SUMMARY_LINE_WIDTH
    print(f"\n{'=' * w}")
    print(f"{'Qwen3-Omni Server TTS WER (Voice Clone)':^{w}}")
    print(f"{'=' * w}")
    print(f"  {'Model:':<{lw}} {args.model}")
    print(f"  {'Speaker:':<{lw}} {args.speaker}")
    print(f"  {'Language:':<{lw}} {metrics.get('lang', 'N/A')}")
    print(
        f"  {'Evaluated / Total:':<{lw}} {metrics.get('evaluated', 0)}/{metrics.get('total_samples', 0)}"
    )
    print(f"  {'Skipped:':<{lw}} {metrics.get('skipped', 0)}")
    print(f"{'-' * w}")
    print(
        f"  {'WER (corpus, micro-avg):':<{lw}} {metrics.get('wer_corpus', 0):.4f} ({metrics.get('wer_corpus', 0)*100:.2f}%)"
    )
    print(f"{'-' * w}")
    print(
        f"  {'WER per-sample mean:':<{lw}} {metrics.get('wer_per_sample_mean', 0):.4f} ({metrics.get('wer_per_sample_mean', 0)*100:.2f}%)"
    )
    print(
        f"  {'WER per-sample median:':<{lw}} {metrics.get('wer_per_sample_median', 0):.4f}"
    )
    print(f"  {'WER per-sample std:':<{lw}} {metrics.get('wer_per_sample_std', 0):.4f}")
    print(f"  {'WER per-sample p95:':<{lw}} {metrics.get('wer_per_sample_p95', 0):.4f}")
    print(
        f"  {'WER corpus (excl >50%):':<{lw}} {metrics.get('wer_below_50_corpus', 0):.4f} ({metrics.get('wer_below_50_corpus', 0)*100:.2f}%)"
    )
    print(
        f"  {'>50% WER samples:':<{lw}} {metrics.get('n_above_50_pct_wer', 0)} ({metrics.get('pct_above_50_pct_wer', 0):.1f}%)"
    )
    print(f"{'-' * w}")
    print(f"  {'Latency mean (s):':<{lw}} {metrics.get('latency_mean_s', 'N/A')}")
    print(
        f"  {'Audio duration mean (s):':<{lw}} {metrics.get('audio_duration_mean_s', 'N/A')}"
    )
    print(f"{'=' * w}\n")


def save_results(
    outputs: list[SampleOutput], metrics: dict, args: argparse.Namespace
) -> None:
    os.makedirs(args.output_dir, exist_ok=True)

    json_results = {
        "summary": metrics,
        "config": {
            "model": args.model,
            "speaker": args.speaker,
            "server_url": f"http://{args.host}:{args.port}",
            "meta": args.meta,
            "max_samples": args.max_samples,
        },
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

    json_path = os.path.join(args.output_dir, "wer_results.json")
    with open(json_path, "w") as f:
        json.dump(json_results, f, indent=2, ensure_ascii=False)

    csv_path = os.path.join(args.output_dir, "wer_results.csv")
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


def wait_for_service(base_url: str, timeout: int = 1200) -> None:
    import requests as requests_lib

    logger.info("Waiting for service at %s ...", base_url)
    start = time.time()
    while True:
        try:
            resp = requests_lib.get(f"{base_url}/health", timeout=5)
            if resp.status_code == 200:
                logger.info("Service is ready.")
                return
        except requests_lib.exceptions.RequestException as exc:
            logger.debug("Health check failed: %s", exc)
        if time.time() - start > timeout:
            raise TimeoutError(f"Service at {base_url} not ready within {timeout}s")
        time.sleep(2)


async def main_async(args: argparse.Namespace):
    if "cuda" in args.asr_device:
        try:
            torch.cuda.set_device(args.asr_device)
            logger.info("Set ASR CUDA device to %s", args.asr_device)
        except Exception as e:
            logger.warning("Failed to set CUDA device %s: %s", args.asr_device, e)

    base_url = f"http://{args.host}:{args.port}"
    api_url = f"{base_url}/v1/chat/completions"

    # Optionally download dataset
    if args.download_dataset:
        download_dataset(args.dataset_dir)

    # Wait for server
    wait_for_service(base_url, timeout=args.server_timeout)

    # Load ASR
    asr = load_asr_model(args.lang, args.asr_device)

    # Parse samples
    samples = parse_meta_lst(args.meta, args.max_samples, args.audio_base_dir)
    logger.info("Loaded %d samples from %s", len(samples), args.meta)

    audio_dir = os.path.join(args.output_dir, "audio")
    os.makedirs(audio_dir, exist_ok=True)

    timeout = aiohttp.ClientTimeout(total=300)
    outputs: list[SampleOutput] = []
    async with aiohttp.ClientSession(timeout=timeout) as session:
        for i, sample in enumerate(
            tqdm(samples, desc=f"Evaluating WER ({args.lang})")
        ):
            result = await evaluate_sample(
                session,
                api_url,
                args.model,
                asr,
                sample,
                args.lang,
                args.asr_device,
                audio_dir,
                args.speaker,
                args.max_tokens,
            )
            outputs.append(result)

            if result.is_success:
                logger.info(
                    "[%d/%d] WER=%.3f  target=%.50s  whisper=%.50s",
                    i + 1,
                    len(samples),
                    result.wer,
                    result.ref_norm,
                    result.hyp_norm,
                )
            else:
                logger.warning(
                    "[%d/%d] FAILED: %s -- %s",
                    i + 1,
                    len(samples),
                    sample.sample_id,
                    result.error,
                )

    metrics = calculate_metrics(outputs, args.lang)
    print_summary(metrics, args)
    save_results(outputs, metrics, args)


def main():
    p = argparse.ArgumentParser(
        description="WER evaluation for Qwen3-Omni TTS via sglang-omni server"
    )
    p.add_argument("--meta", default="seedtts_testset/en/meta.lst")
    p.add_argument(
        "--output-dir",
        required=True,
        help="Directory to save generated audio and results",
    )
    p.add_argument(
        "--model",
        default="qwen3-omni",
        help="Model name for the API request",
    )
    p.add_argument(
        "--speaker",
        default="Ethan",
        choices=["Ethan", "Chelsie", "Aiden"],
        help="Speaker voice for Qwen3-Omni TTS",
    )
    p.add_argument(
        "--lang", choices=["en", "zh"], default="en", help="Language for ASR model"
    )
    p.add_argument("--host", type=str, default="localhost", help="Server host")
    p.add_argument("--port", type=int, default=8000, help="Server port")
    p.add_argument(
        "--asr-device", default="cuda:0", help="Device for ASR (Whisper) model"
    )
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument(
        "--max-tokens",
        type=int,
        default=THINKER_MAX_NEW_TOKENS,
        help="Max new tokens for thinker",
    )
    p.add_argument(
        "--server-timeout",
        type=int,
        default=1200,
        help="Timeout in seconds to wait for server readiness",
    )
    p.add_argument(
        "--download-dataset",
        action="store_true",
        help="Auto-download seed-tts-eval dataset before evaluation",
    )
    p.add_argument(
        "--dataset-dir",
        default="seedtts_testset",
        help="Local directory for the dataset",
    )
    p.add_argument(
        "--audio-base-dir",
        default=None,
        help="Base directory for resolving ref_audio paths in meta.lst. "
        "Needed when --meta points to a shard file in a different directory.",
    )
    args = p.parse_args()

    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
