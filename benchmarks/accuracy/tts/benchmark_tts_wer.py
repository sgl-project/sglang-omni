#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""WER evaluation for seed-tts-eval generated audio.

Supports both English (Whisper-large-v3) and Chinese (FunASR paraformer-zh).

Usage:
    # English
    python -m benchmarks.accuracy.tts.benchmark_tts_wer \
        --meta seedtts_testset/en/meta.lst \
        --model-path fishaudio/s2-pro \
        --output-dir results/s2pro_en \
        --lang en

    # Chinese
    python -m benchmarks.accuracy.tts.benchmark_tts_wer \
        --meta seedtts_testset/zh/hardcase.lst \
        --model-path fishaudio/s2-pro \
        --output-dir results/s2pro_zh \
        --lang zh
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import logging
import os
import string
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.signal
import soundfile as sf
import torch
from jiwer import process_words
from tqdm import tqdm

from sglang_omni.models.fishaudio_s2_pro.pipeline.stages import (
    create_preprocessing_executor,
    create_sglang_tts_engine_executor,
    create_vocoder_executor,
)
from sglang_omni.proto import OmniRequest, StagePayload

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

SUMMARY_LABEL_WIDTH = 30
SUMMARY_LINE_WIDTH = 60


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
    audio_duration_s: float = 0.0
    latency_s: float = 0.0
    is_success: bool = False
    error: str = ""


def parse_meta_lst(path: str, max_samples: int | None = None) -> list[SampleInput]:
    base_dir = os.path.dirname(path)
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


def normalize_text(text: str, lang: str) -> str:
    if lang == "zh":
        from zhon.hanzi import punctuation as zh_punct

        all_punct = zh_punct + string.punctuation
    else:
        all_punct = string.punctuation

    for ch in all_punct:
        if ch == "'":
            continue
        text = text.replace(ch, "")

    text = text.replace("  ", " ").strip()

    if lang == "zh":
        # Character-level: space between each character
        text = " ".join(list(text))
    else:
        text = text.lower()

    return text


def load_asr_model(lang: str, device: str):
    if lang == "en":
        from transformers import WhisperForConditionalGeneration, WhisperProcessor

        logger.info("Loading Whisper-large-v3...")
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


def transcribe(asr, wav_path: str, lang: str, device: str) -> str:
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


async def generate_speech_s2pro(
    prep,
    engine,
    vocoder,
    sample: SampleInput,
    max_new_tokens: int = 2048,
    temperature: float = 0.8,
) -> tuple[np.ndarray, int, float]:
    """Generate speech using SGLang-Omni pipeline executors locally."""
    references = []
    if sample.ref_audio and sample.ref_text:
        references.append({"audio_path": sample.ref_audio, "text": sample.ref_text})

    req = OmniRequest(
        inputs={"text": sample.target_text, "references": references},
        params={"max_new_tokens": max_new_tokens, "temperature": temperature},
    )
    payload = StagePayload(request_id=sample.sample_id, request=req, data={})

    t0 = time.perf_counter()
    
    # 1. Preprocessing
    await prep.add_request(payload)
    payload = await prep.get_result()
    
    # 2. Text-to-Semantic engine
    await engine.add_request(payload)
    payload = await engine.get_result()
    
    # 3. Vocoder (Semantic-to-Acoustic)
    await vocoder.add_request(payload)
    payload = await vocoder.get_result()
    
    latency = time.perf_counter() - t0

    audio_data = payload.data.get("audio_data")
    sr = payload.data.get("sample_rate", 44100)

    if not audio_data:
        raise ValueError("No audio data generated")

    return np.array(audio_data), sr, latency


async def evaluate_sample(
    prep,
    engine,
    vocoder,
    asr,
    sample: SampleInput,
    lang: str,
    device: str,
    audio_dir: str,
    max_new_tokens: int,
    temperature: float,
) -> SampleOutput:
    output = SampleOutput(
        sample_id=sample.sample_id,
        target_text=sample.target_text,
    )
    
    wav_path = os.path.join(audio_dir, f"{sample.sample_id}.wav")

    # Generate Audio
    try:
        audio_data, sr, latency = await generate_speech_s2pro(
            prep, engine, vocoder, sample, max_new_tokens, temperature
        )
        sf.write(wav_path, audio_data, sr)
        
        output.latency_s = round(latency, 4)
        output.audio_duration_s = round(len(audio_data) / sr, 4)
    except Exception as e:
        output.error = f"Generation failed: {e}"
        logger.error("[%s] %s", sample.sample_id, output.error)
        return output

    # Transcribe
    try:
        hyp_text = transcribe(asr, wav_path, lang, device)
    except Exception as e:
        output.error = f"Transcription failed: {e}"
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
    output.is_success = True

    return output


def calculate_metrics(outputs: list[SampleOutput], lang: str) -> dict:
    successes = [o for o in outputs if o.is_success]
    if not successes:
        return {"completed": 0, "failed": len(outputs)}

    wers = [o.wer for o in successes]
    wer_arr = np.array(wers)
    latencies = [o.latency_s for o in successes]
    audio_durations = [o.audio_duration_s for o in successes if o.audio_duration_s > 0]

    n_above_50 = int(np.sum(wer_arr > 0.5))
    wers_below_50 = wer_arr[wer_arr <= 0.5]

    return {
        "lang": lang,
        "total_samples": len(outputs),
        "evaluated": len(successes),
        "skipped": len(outputs) - len(successes),
        "wer_mean": float(np.mean(wer_arr)),
        "wer_median": float(np.median(wer_arr)),
        "wer_std": float(np.std(wer_arr)),
        "wer_p95": float(np.percentile(wer_arr, 95)),
        "wer_below_50_mean": float(np.mean(wers_below_50)) if len(wers_below_50) else 0,
        "n_above_50_pct_wer": n_above_50,
        "pct_above_50_pct_wer": n_above_50 / len(successes) * 100 if successes else 0,
        "latency_mean_s": float(np.mean(latencies)),
        "audio_duration_mean_s": float(np.mean(audio_durations)) if audio_durations else 0,
    }


def print_summary(metrics: dict, args: argparse.Namespace) -> None:
    lw = SUMMARY_LABEL_WIDTH
    w = SUMMARY_LINE_WIDTH
    print(f"\n{'=' * w}")
    print(f"{'TTS WER Benchmark Result':^{w}}")
    print(f"{'=' * w}")
    print(f"  {'Model:':<{lw}} {args.model_path}")
    print(f"  {'Language:':<{lw}} {metrics.get('lang', 'N/A')}")
    print(
        f"  {'Evaluated / Total:':<{lw}} {metrics.get('evaluated', 0)}/{metrics.get('total_samples', 0)}"
    )
    print(f"  {'Skipped:':<{lw}} {metrics.get('skipped', 0)}")
    print(f"{'-' * w}")
    print(f"  {'WER mean:':<{lw}} {metrics.get('wer_mean', 0):.4f} ({metrics.get('wer_mean', 0)*100:.2f}%)")
    print(f"  {'WER median:':<{lw}} {metrics.get('wer_median', 0):.4f}")
    print(f"  {'WER std:':<{lw}} {metrics.get('wer_std', 0):.4f}")
    print(f"  {'WER p95:':<{lw}} {metrics.get('wer_p95', 0):.4f}")
    print(f"  {'WER (excl >50%):':<{lw}} {metrics.get('wer_below_50_mean', 0):.4f} ({metrics.get('wer_below_50_mean', 0)*100:.2f}%)")
    print(
        f"  {'>50% WER samples:':<{lw}} {metrics.get('n_above_50_pct_wer', 0)} ({metrics.get('pct_above_50_pct_wer', 0):.1f}%)"
    )
    print(f"{'-' * w}")
    print(f"  {'Latency mean (s):':<{lw}} {metrics.get('latency_mean_s', 'N/A')}")
    print(
        f"  {'Audio duration mean (s):':<{lw}} {metrics.get('audio_duration_mean_s', 'N/A')}"
    )
    print(f"{'=' * w}\n")


def save_results(outputs: list[SampleOutput], metrics: dict, args: argparse.Namespace) -> None:
    os.makedirs(args.output_dir, exist_ok=True)
    
    json_results = {
        "summary": metrics,
        "config": {
            "model_path": args.model_path,
            "meta": args.meta,
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
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
                    f"{o.audio_duration_s:.4f}",
                    f"{o.latency_s:.4f}",
                    o.is_success,
                    o.error or "",
                ]
            )


async def main_async(args: argparse.Namespace):
    # Set the default CUDA device to avoid device mismatch issues in multi-GPU environments
    if "cuda" in args.device:
        try:
            torch.cuda.set_device(args.device)
            logger.info("Set default CUDA device to %s", args.device)
        except Exception as e:
            logger.warning("Failed to set default CUDA device to %s: %s", args.device, e)

    # Load ASR
    asr = load_asr_model(args.lang, args.device)

    # Load S2-Pro locally (offline pipeline executors)
    logger.info("Loading S2-Pro models from %s on %s...", args.model_path, args.device)
    prep = create_preprocessing_executor(args.model_path)
    engine = create_sglang_tts_engine_executor(args.model_path, device=args.device)
    vocoder = create_vocoder_executor(args.model_path, device=args.device)
    await engine.start()

    try:
        # Parse samples
        samples = parse_meta_lst(args.meta, args.max_samples)
        logger.info("Loaded %d samples from %s", len(samples), args.meta)

        audio_dir = os.path.join(args.output_dir, "audio")
        os.makedirs(audio_dir, exist_ok=True)

        outputs = []
        for i, sample in enumerate(tqdm(samples, desc=f"Evaluating WER ({args.lang})")):
            result = await evaluate_sample(
                prep,
                engine,
                vocoder,
                asr,
                sample,
                args.lang,
                args.device,
                audio_dir,
                args.max_new_tokens,
                args.temperature,
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
                    "[%d/%d] FAILED: %s — %s",
                    i + 1,
                    len(samples),
                    sample.sample_id,
                    result.error,
                )

        metrics = calculate_metrics(outputs, args.lang)
        print_summary(metrics, args)
        save_results(outputs, metrics, args)

    finally:
        await engine.stop()


def main():
    p = argparse.ArgumentParser(
        description="WER evaluation (Whisper for EN, FunASR for ZH)"
    )
    p.add_argument("--meta", default="seedtts_testset/en/meta.lst")
    p.add_argument(
        "--output-dir", required=True, help="Directory to save generated audio and results"
    )
    p.add_argument("--model-path", default="fishaudio/s2-pro", help="Path to S2-Pro model")
    p.add_argument(
        "--lang", choices=["en", "zh"], default="en", help="Language for ASR model"
    )
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--max-new-tokens", type=int, default=2048)
    p.add_argument("--temperature", type=float, default=0.8)
    args = p.parse_args()
    
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
