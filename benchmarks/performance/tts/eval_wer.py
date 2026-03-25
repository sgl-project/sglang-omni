#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""WER evaluation for seed-tts-eval generated audio.

Supports both English (Whisper-large-v3) and Chinese (FunASR paraformer-zh).

Usage:
    # English
    CUDA_VISIBLE_DEVICES=7 python benchmarks/eval_wer.py \
        --meta /tmp/seed-tts-eval/seedtts_testset/en/meta.lst \
        --audio-dir results/s2pro_compile_eager/audio \
        --lang en

    # Chinese
    CUDA_VISIBLE_DEVICES=7 python benchmarks/eval_wer.py \
        --meta /tmp/seed-tts-eval/seedtts_testset/zh/meta.lst \
        --audio-dir results/s2pro_zh/audio \
        --lang zh

    # Chinese hard cases
    CUDA_VISIBLE_DEVICES=7 python benchmarks/eval_wer.py \
        --meta /tmp/seed-tts-eval/seedtts_testset/zh/hardcase.lst \
        --audio-dir results/s2pro_zh_hard/audio \
        --lang zh
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import string
import time

import numpy as np
import scipy.signal
import soundfile as sf
import torch
from jiwer import process_words
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def parse_meta_lst(path: str) -> list[dict]:
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
                {
                    "id": parts[0],
                    "ref_text": parts[1],
                    "ref_audio": os.path.join(base_dir, parts[2]),
                    "text": parts[3],
                }
            )
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


def main(args):
    lang = args.lang
    device = args.device

    # Load ASR
    asr = load_asr_model(lang, device)

    # Parse samples
    samples = parse_meta_lst(args.meta)
    logger.info("Loaded %d samples from %s", len(samples), args.meta)

    if args.max_samples and args.max_samples < len(samples):
        samples = samples[: args.max_samples]

    results = []
    skipped = 0

    for sample in tqdm(samples, desc=f"Evaluating WER ({lang})"):
        wav_path = os.path.join(args.audio_dir, f"{sample['id']}.wav")
        if not os.path.exists(wav_path):
            skipped += 1
            continue

        # Transcribe
        transcription = transcribe(asr, wav_path, lang, device)

        # Compute WER
        ref = normalize_text(sample["text"], lang)
        hyp = normalize_text(transcription, lang)

        if not ref:
            skipped += 1
            continue

        measures = process_words(ref, hyp)
        wer = measures.wer

        results.append(
            {
                "id": sample["id"],
                "ref_text": sample["text"],
                "hyp_text": transcription,
                "ref_norm": ref,
                "hyp_norm": hyp,
                "wer": wer,
                "substitutions": measures.substitutions,
                "deletions": measures.deletions,
                "insertions": measures.insertions,
            }
        )

    # Aggregate
    wers = [r["wer"] for r in results]
    wer_arr = np.array(wers)

    n_above_50 = int(np.sum(wer_arr > 0.5))
    wers_below_50 = wer_arr[wer_arr <= 0.5]

    summary = {
        "lang": lang,
        "total_samples": len(samples),
        "evaluated": len(results),
        "skipped": skipped,
        "wer_mean": float(np.mean(wer_arr)) if len(wer_arr) else 0,
        "wer_median": float(np.median(wer_arr)) if len(wer_arr) else 0,
        "wer_std": float(np.std(wer_arr)) if len(wer_arr) else 0,
        "wer_p95": float(np.percentile(wer_arr, 95)) if len(wer_arr) else 0,
        "wer_below_50_mean": float(np.mean(wers_below_50)) if len(wers_below_50) else 0,
        "n_above_50_pct_wer": n_above_50,
        "pct_above_50_pct_wer": n_above_50 / len(results) * 100 if results else 0,
    }

    output = {"summary": summary, "per_sample": results}

    # Save
    os.makedirs(
        os.path.dirname(args.output) if os.path.dirname(args.output) else ".",
        exist_ok=True,
    )
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    # Print
    print("\n" + "=" * 52)
    print(f"  WER Evaluation Results ({lang.upper()})")
    print("=" * 52)
    print(f"  Evaluated:           {len(results)}/{len(samples)}")
    print(
        f"  WER mean:            {summary['wer_mean']:.4f} ({summary['wer_mean']*100:.2f}%)"
    )
    print(f"  WER median:          {summary['wer_median']:.4f}")
    print(
        f"  WER (excl >50%):     {summary['wer_below_50_mean']:.4f} ({summary['wer_below_50_mean']*100:.2f}%)"
    )
    print(
        f"  Samples >50% WER:    {n_above_50} ({summary['pct_above_50_pct_wer']:.1f}%)"
    )
    print(f"\n  Output: {args.output}")
    print("=" * 52 + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="WER evaluation (Whisper for EN, FunASR for ZH)"
    )
    p.add_argument("--meta", default="/tmp/seed-tts-eval/seedtts_testset/en/meta.lst")
    p.add_argument(
        "--audio-dir", required=True, help="Directory with generated {id}.wav files"
    )
    p.add_argument(
        "--output", default=None, help="Output JSON (default: {audio-dir}/../wer.json)"
    )
    p.add_argument(
        "--lang", choices=["en", "zh"], default="en", help="Language for ASR model"
    )
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--max-samples", type=int, default=None)
    args = p.parse_args()
    if args.output is None:
        args.output = os.path.join(os.path.dirname(args.audio_dir), "wer.json")
    main(args)