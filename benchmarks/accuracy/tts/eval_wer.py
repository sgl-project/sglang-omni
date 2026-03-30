#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""WER evaluation for seed-tts-eval generated audio.

Supports both English (Whisper-large-v3) and Chinese (FunASR paraformer-zh).

Reports both corpus-level WER (micro-average, primary metric) and per-sample
WER statistics (mean, median, std, p95) as secondary diagnostics.

Usage:
    # English
    CUDA_VISIBLE_DEVICES=7 python benchmarks/accuracy/tts/eval_wer.py \
        --meta /tmp/seed-tts-eval/seedtts_testset/en/meta.lst \
        --audio-dir results/s2pro_compile_eager/audio \
        --lang en

    # English (local whisper model)
    CUDA_VISIBLE_DEVICES=7 python benchmarks/accuracy/tts/eval_wer.py \
        --meta /tmp/seed-tts-eval/seedtts_testset/en/meta.lst \
        --audio-dir results/s2pro_compile_eager/audio \
        --lang en \
        --whisper-model /path/to/whisper-large-v3

    # Chinese
    CUDA_VISIBLE_DEVICES=7 python benchmarks/accuracy/tts/eval_wer.py \
        --meta /tmp/seed-tts-eval/seedtts_testset/zh/meta.lst \
        --audio-dir results/s2pro_zh/audio \
        --lang zh

    # Chinese hard cases
    CUDA_VISIBLE_DEVICES=7 python benchmarks/accuracy/tts/eval_wer.py \
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


def load_asr_model(lang: str, device: str, model_path: str | None = None):
    if lang == "en":
        from transformers import WhisperForConditionalGeneration, WhisperProcessor

        whisper_id = model_path or "openai/whisper-large-v3"
        logger.info("Loading Whisper-large-v3 from %s ...", whisper_id)
        t0 = time.perf_counter()
        processor = WhisperProcessor.from_pretrained(whisper_id)
        model = WhisperForConditionalGeneration.from_pretrained(whisper_id).to(device)
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
    else:
        raise ValueError(f"Unknown ASR type: {asr['type']}")


def main(args):
    lang = args.lang
    device = args.device

    # Load ASR
    asr = load_asr_model(lang, device, model_path=args.whisper_model)

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
                "hits": measures.hits,
            }
        )

    # Aggregate
    wers = [r["wer"] for r in results]
    wer_arr = np.array(wers)

    # Corpus WER (micro-average): total errors / total reference words
    total_sub = sum(r["substitutions"] for r in results)
    total_del = sum(r["deletions"] for r in results)
    total_ins = sum(r["insertions"] for r in results)
    total_hits = sum(r["hits"] for r in results)
    total_ref_words = total_sub + total_del + total_hits
    corpus_wer = (total_sub + total_del + total_ins) / total_ref_words if total_ref_words > 0 else 0

    n_above_50 = int(np.sum(wer_arr > 0.5))
    wers_below_50 = wer_arr[wer_arr <= 0.5]

    # Corpus WER excluding >50% samples
    results_below_50 = [r for r in results if r["wer"] <= 0.5]
    sub_b50 = sum(r["substitutions"] for r in results_below_50)
    del_b50 = sum(r["deletions"] for r in results_below_50)
    ins_b50 = sum(r["insertions"] for r in results_below_50)
    hits_b50 = sum(r["hits"] for r in results_below_50)
    ref_b50 = sub_b50 + del_b50 + hits_b50
    corpus_wer_below_50 = (sub_b50 + del_b50 + ins_b50) / ref_b50 if ref_b50 > 0 else 0

    summary = {
        "lang": lang,
        "total_samples": len(samples),
        "evaluated": len(results),
        "skipped": skipped,
        "corpus_wer": float(corpus_wer),
        "corpus_wer_below_50": float(corpus_wer_below_50),
        "wer_mean": float(np.mean(wer_arr)) if len(wer_arr) else 0,
        "wer_median": float(np.median(wer_arr)) if len(wer_arr) else 0,
        "wer_std": float(np.std(wer_arr)) if len(wer_arr) else 0,
        "wer_p95": float(np.percentile(wer_arr, 95)) if len(wer_arr) else 0,
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
        f"  Corpus WER:          {summary['corpus_wer']:.4f} ({summary['corpus_wer']*100:.2f}%)"
    )
    print(
        f"  Corpus WER (excl>50%): {summary['corpus_wer_below_50']:.4f} ({summary['corpus_wer_below_50']*100:.2f}%)"
    )
    print(f"  WER mean (per-sample): {summary['wer_mean']:.4f} ({summary['wer_mean']*100:.2f}%)")
    print(f"  WER median:          {summary['wer_median']:.4f}")
    print(f"  WER std:             {summary['wer_std']:.4f}")
    print(f"  WER p95:             {summary['wer_p95']:.4f}")
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
    p.add_argument(
        "--whisper-model", default=None,
        help="Local path or HF ID for Whisper model (default: openai/whisper-large-v3)"
    )
    args = p.parse_args()
    if args.output is None:
        args.output = os.path.join(os.path.dirname(args.audio_dir), "wer.json")
    main(args)
