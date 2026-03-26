"""
Benchmark accuracy (WER) for Qwen3-Omni speech synthesis.

Author:

- Jingwen Gu https://github.com/JingwenGu0829
- Chenyang Zhao https://github.com/zhaochenyang20

This script measures Word Error Rate (WER) of Qwen3-Omni's speech generation
by comparing Whisper transcriptions of generated audio against ground-truth
target texts from the seed-tts-eval dataset.

Inference backend: HuggingFace Transformers (offline). The SGLang speech
pipeline is currently broken on main, so this script bypasses it entirely
and loads the model directly via ``Qwen3OmniMoeForConditionalGeneration``.

Published results (Qwen3-Omni paper, Table 13, seed-tts-eval):
  test-en: WER 1.39%
  test-zh: WER 1.07%

Dataset: seed-tts-eval

    The seed-tts-eval testset (from BytedanceSpeech) contains samples from
    public speech corpora for objective TTS evaluation:

      - en/meta.lst : 1088 English samples from CommonVoice
      - zh/meta.lst : 2020 Chinese samples from DiDiSpeech-2

    Download from Hugging Face:

        # full set
        huggingface-cli download zhaochenyang20/seed-tts-eval \
            --repo-type dataset --local-dir seedtts_testset

        # mini set for CI (10 samples)
        huggingface-cli download zhaochenyang20/seed-tts-eval-mini \
            --repo-type dataset --local-dir seedtts_testset

Usage:

    # Quick test with mini set (5 EN + 5 ZH samples):
    CUDA_VISIBLE_DEVICES=1 python -m benchmarks.accuracy.omni.benchmark_omni_wer \
        --model-path Qwen/Qwen3-Omni-30B-A3B-Instruct \
        --testset seedtts_testset/en \
        --max-samples 5

    # Full EN evaluation:
    CUDA_VISIBLE_DEVICES=1 python -m benchmarks.accuracy.omni.benchmark_omni_wer \
        --model-path Qwen/Qwen3-Omni-30B-A3B-Instruct \
        --testset seedtts_testset/en

    # Full ZH evaluation:
    CUDA_VISIBLE_DEVICES=1 python -m benchmarks.accuracy.omni.benchmark_omni_wer \
        --model-path Qwen/Qwen3-Omni-30B-A3B-Instruct \
        --testset seedtts_testset/zh
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path

import jiwer
import numpy as np
import torch
import torchaudio

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

META_FIELD_COUNT = 4
SUMMARY_LABEL_WIDTH = 30
SUMMARY_LINE_WIDTH = 60
WHISPER_SAMPLE_RATE = 16000
QWEN3_OMNI_SAMPLE_RATE = 24000

PUBLISHED_WER = {"en": 1.39, "zh": 1.07}


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


def parse_meta_lst(
    testset_dir: str, max_samples: int | None = None
) -> list[SampleInput]:
    """Parse a seed-tts-eval meta.lst file.

    Format: id|ref_text|ref_audio_path|target_text
    """
    meta_path = Path(testset_dir) / "meta.lst"
    if not meta_path.is_file():
        raise FileNotFoundError(f"meta.lst not found at {meta_path}")

    samples: list[SampleInput] = []
    for line in meta_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split("|")
        if len(parts) < META_FIELD_COUNT:
            logger.warning("Skipping malformed line: %s", line[:80])
            continue
        ref_audio = str(Path(testset_dir) / parts[2])
        if not Path(ref_audio).is_file():
            logger.warning("Missing ref audio %s, skipping", ref_audio)
            continue
        samples.append(
            SampleInput(
                sample_id=parts[0],
                ref_text=parts[1],
                ref_audio=ref_audio,
                target_text=parts[3],
            )
        )
        if max_samples and len(samples) >= max_samples:
            break
    return samples


def detect_language(testset_dir: str) -> str:
    """Detect language from testset directory name."""
    name = Path(testset_dir).name.lower()
    if "zh" in name:
        return "zh"
    return "en"


def normalize_text(text: str, language: str = "en") -> str:
    """Normalize text for WER computation."""
    text = text.strip()
    if language == "zh":
        text = re.sub(r"[^\u4e00-\u9fff\w]", "", text)
        return text
    text = text.lower()
    text = re.sub(r"[^\w\s']", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_omni_model(
    model_path: str,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
) -> tuple:
    """Load Qwen3-Omni model and processor."""
    from transformers import AutoProcessor, Qwen3OmniMoeForConditionalGeneration

    logger.info("Loading Qwen3-Omni model from %s ...", model_path)
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
        model_path,
        dtype=dtype,
        device_map=device,
        trust_remote_code=True,
    )
    model.eval()
    logger.info("Model loaded on %s", device)
    return processor, model


def load_whisper(
    model_name: str,
    device: str = "cuda",
) -> tuple:
    """Load Whisper model and processor for ASR."""
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    logger.info("Loading Whisper model %s on %s ...", model_name, device)
    processor = WhisperProcessor.from_pretrained(model_name)
    model = WhisperForConditionalGeneration.from_pretrained(model_name)
    model = model.to(device).eval()
    return processor, model


@torch.no_grad()
def generate_speech(
    processor,
    model,
    target_text: str,
    language: str = "en",
    max_new_tokens: int = 2048,
    temperature: float = 0.7,
    speaker: str = "Chelsie",
) -> tuple[str, torch.Tensor | None, int, float]:
    """Generate speech for a text prompt.

    Returns (text_output, audio_tensor, sample_rate, latency_s).
    """
    if language == "zh":
        prompt = f"请朗读以下文本：{target_text}"
    else:
        prompt = f"Please read the following text out loud: {target_text}"

    messages = [{"role": "user", "content": prompt}]
    text_input = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(text=text_input, return_tensors="pt", padding=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    t0 = time.perf_counter()
    thinker_result, audio_wav = model.generate(
        **inputs,
        return_audio=True,
        speaker=speaker,
        thinker_max_new_tokens=max_new_tokens,
    )
    latency = time.perf_counter() - t0

    text_ids = (
        thinker_result
        if not hasattr(thinker_result, "sequences")
        else thinker_result.sequences
    )
    text_output = processor.batch_decode(
        text_ids[:, inputs["input_ids"].shape[1] :],
        skip_special_tokens=True,
    )[0]

    return text_output, audio_wav, QWEN3_OMNI_SAMPLE_RATE, latency


@torch.no_grad()
def transcribe(
    processor,
    model,
    audio: torch.Tensor,
    sample_rate: int,
    language: str = "en",
) -> str:
    """Transcribe audio using Whisper."""
    if sample_rate != WHISPER_SAMPLE_RATE:
        audio = torchaudio.functional.resample(audio, sample_rate, WHISPER_SAMPLE_RATE)
    if audio.ndim > 1:
        audio = audio.squeeze(0)

    inputs = processor(
        audio.cpu().float().numpy(),
        sampling_rate=WHISPER_SAMPLE_RATE,
        return_tensors="pt",
    )
    input_features = inputs.input_features.to(model.device)

    gen_kwargs = {"max_new_tokens": 444, "task": "transcribe"}
    gen_kwargs["language"] = "chinese" if language == "zh" else "english"

    predicted_ids = model.generate(input_features, **gen_kwargs)
    text = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    return text


def evaluate_sample(
    omni_processor,
    omni_model,
    whisper_processor,
    whisper_model,
    sample: SampleInput,
    language: str,
    max_new_tokens: int,
    temperature: float,
    speaker: str,
    audio_dir: str | None = None,
) -> SampleOutput:
    """Run inference + ASR + WER for one sample."""
    output = SampleOutput(
        sample_id=sample.sample_id,
        target_text=sample.target_text,
    )

    try:
        text_out, audio_wav, sr, latency = generate_speech(
            omni_processor,
            omni_model,
            sample.target_text,
            language=language,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            speaker=speaker,
        )
    except Exception as e:
        output.error = f"Generation failed: {e}"
        logger.error("[%s] %s", sample.sample_id, output.error)
        return output

    if audio_wav is None or audio_wav.numel() == 0:
        output.error = "No audio output"
        logger.warning("[%s] No audio output", sample.sample_id)
        return output

    audio_tensor = audio_wav.float()
    while audio_tensor.ndim > 2:
        audio_tensor = audio_tensor.squeeze(0)
    if audio_tensor.ndim == 1:
        audio_tensor = audio_tensor.unsqueeze(0)

    output.audio_duration_s = round(audio_tensor.shape[-1] / sr, 4)
    output.latency_s = round(latency, 4)

    if audio_dir:
        out_path = Path(audio_dir) / f"{sample.sample_id}.wav"
        torchaudio.save(str(out_path), audio_tensor.cpu(), sr)

    try:
        hyp_text = transcribe(
            whisper_processor, whisper_model, audio_tensor, sr, language
        )
    except Exception as e:
        output.error = f"Transcription failed: {e}"
        logger.error("[%s] %s", sample.sample_id, output.error)
        return output

    output.whisper_text = hyp_text
    output.ref_norm = normalize_text(sample.target_text, language)
    output.hyp_norm = normalize_text(hyp_text, language)

    if not output.ref_norm:
        output.error = "Empty reference after normalization"
        return output

    if language == "zh":
        # For Chinese, use character-level error rate (CER).
        # The Qwen3-Omni paper reports "WER" for Chinese but this is
        # effectively CER, as is standard for Chinese TTS evaluation.
        ref_chars = " ".join(list(output.ref_norm))
        hyp_chars = " ".join(list(output.hyp_norm))
        measures = jiwer.process_words(ref_chars, hyp_chars)
    else:
        measures = jiwer.process_words(output.ref_norm, output.hyp_norm)
    output.wer = measures.wer
    output.substitutions = measures.substitutions
    output.deletions = measures.deletions
    output.insertions = measures.insertions
    output.is_success = True
    return output


def calculate_metrics(
    outputs: list[SampleOutput],
    total_samples: int,
    language: str,
) -> dict:
    """Compute aggregate WER metrics from sample outputs."""
    successes = [o for o in outputs if o.is_success]
    if not successes:
        return {
            "completed_requests": 0,
            "failed_requests": len(outputs),
        }

    wers = [o.wer for o in successes]
    latencies = [o.latency_s for o in successes]
    audio_durations = [o.audio_duration_s for o in successes if o.audio_duration_s > 0]
    n_above_50 = sum(1 for w in wers if w > 0.5)

    published = PUBLISHED_WER.get(language, 0.0)

    return {
        "language": language,
        "total_samples": total_samples,
        "evaluated": len(successes),
        "skipped": total_samples - len(successes),
        "wer_mean": round(float(np.mean(wers)), 6),
        "wer_median": round(float(np.median(wers)), 6),
        "wer_std": round(float(np.std(wers)), 6),
        "wer_p95": round(float(np.percentile(wers, 95)), 6),
        "wer_mean_pct": round(float(np.mean(wers)) * 100, 2),
        "published_wer_pct": published,
        "delta_pct": round(float(np.mean(wers)) * 100 - published, 2),
        "n_above_50_pct_wer": n_above_50,
        "pct_above_50_pct_wer": round(n_above_50 / len(successes) * 100, 1),
        "latency_mean_s": round(float(np.mean(latencies)), 3),
        "latency_median_s": round(float(np.median(latencies)), 3),
        "latency_p95_s": round(float(np.percentile(latencies, 95)), 3),
        "audio_duration_mean_s": (
            round(float(np.mean(audio_durations)), 3) if audio_durations else 0
        ),
    }


def print_summary(metrics: dict, args: argparse.Namespace) -> None:
    """Print human-readable summary."""
    lw = SUMMARY_LABEL_WIDTH
    w = SUMMARY_LINE_WIDTH
    print(f"\n{'=' * w}")
    print(f"{'Qwen3-Omni WER Benchmark Result':^{w}}")
    print(f"{'=' * w}")
    print(f"  {'Model:':<{lw}} {args.model_path}")
    print(f"  {'Language:':<{lw}} {metrics.get('language', 'N/A')}")
    print(f"  {'Backend:':<{lw}} HuggingFace Transformers")
    print(
        f"  {'Evaluated / Total:':<{lw}} {metrics.get('evaluated', 0)}/{metrics.get('total_samples', 0)}"
    )
    print(f"  {'Skipped:':<{lw}} {metrics.get('skipped', 0)}")
    print(f"{'-' * w}")
    print(f"  {'WER mean:':<{lw}} {metrics.get('wer_mean_pct', 'N/A')}%")
    print(f"  {'WER median:':<{lw}} {round(metrics.get('wer_median', 0) * 100, 2)}%")
    print(f"  {'WER std:':<{lw}} {round(metrics.get('wer_std', 0) * 100, 2)}%")
    print(f"  {'WER p95:':<{lw}} {round(metrics.get('wer_p95', 0) * 100, 2)}%")
    print(
        f"  {'>50% WER samples:':<{lw}} {metrics.get('n_above_50_pct_wer', 0)} ({metrics.get('pct_above_50_pct_wer', 0)}%)"
    )
    print(f"{'-' * w}")
    print(f"  {'Published WER:':<{lw}} {metrics.get('published_wer_pct', 'N/A')}%")
    print(f"  {'Delta (ours - published):':<{lw}} {metrics.get('delta_pct', 'N/A'):+}%")
    print(f"{'-' * w}")
    print(f"  {'Latency mean (s):':<{lw}} {metrics.get('latency_mean_s', 'N/A')}")
    print(f"  {'Latency median (s):':<{lw}} {metrics.get('latency_median_s', 'N/A')}")
    print(f"  {'Latency p95 (s):':<{lw}} {metrics.get('latency_p95_s', 'N/A')}")
    print(
        f"  {'Audio duration mean (s):':<{lw}} {metrics.get('audio_duration_mean_s', 'N/A')}"
    )
    print(f"{'=' * w}")


def _save_json_results(
    outputs: list[SampleOutput],
    metrics: dict,
    args: argparse.Namespace,
) -> None:
    """Write benchmark results as JSON."""
    json_results = {
        "summary": metrics,
        "config": {
            "model_path": args.model_path,
            "testset": args.testset,
            "whisper_model": args.whisper_model,
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "speaker": args.speaker,
            "max_samples": args.max_samples,
            "backend": "huggingface_transformers",
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
    logger.info("JSON results saved to %s", json_path)


def _save_csv_results(
    outputs: list[SampleOutput],
    output_dir: str,
) -> None:
    """Write per-sample results as CSV."""
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
    logger.info("CSV results saved to %s", csv_path)


def save_results(
    outputs: list[SampleOutput],
    metrics: dict,
    args: argparse.Namespace,
) -> None:
    """Save all results to output directory."""
    os.makedirs(args.output_dir, exist_ok=True)
    _save_json_results(outputs, metrics, args)
    _save_csv_results(outputs, args.output_dir)


def benchmark(args: argparse.Namespace) -> None:
    """Run the WER benchmark."""
    samples = parse_meta_lst(args.testset, args.max_samples)
    language = args.language or detect_language(args.testset)
    logger.info(
        "Loaded %d samples from %s (language=%s)", len(samples), args.testset, language
    )

    omni_processor, omni_model = load_omni_model(
        args.model_path,
        device=args.device,
    )

    whisper_device = args.whisper_device or args.device
    whisper_processor, whisper_model = load_whisper(args.whisper_model, whisper_device)

    audio_dir = None
    if args.save_audio:
        audio_dir = os.path.join(args.output_dir, "audio")
        os.makedirs(audio_dir, exist_ok=True)

    outputs: list[SampleOutput] = []
    for i, sample in enumerate(samples):
        result = evaluate_sample(
            omni_processor,
            omni_model,
            whisper_processor,
            whisper_model,
            sample,
            language,
            args.max_new_tokens,
            args.temperature,
            args.speaker,
            audio_dir,
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

    metrics = calculate_metrics(outputs, len(samples), language)
    print_summary(metrics, args)
    save_results(outputs, metrics, args)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark WER accuracy for Qwen3-Omni speech synthesis."
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="Qwen/Qwen3-Omni-30B-A3B-Instruct",
        help="HuggingFace model ID or local path.",
    )
    parser.add_argument(
        "--testset",
        type=str,
        required=True,
        help="Path to seed-tts-eval testset directory (e.g. seedtts_testset/en).",
    )
    parser.add_argument(
        "--language",
        type=str,
        default=None,
        choices=["en", "zh"],
        help="Language (auto-detected from testset path if not set).",
    )
    parser.add_argument(
        "--whisper-model",
        type=str,
        default="openai/whisper-large-v3",
        help="HuggingFace Whisper model ID for ASR.",
    )
    parser.add_argument(
        "--whisper-device",
        type=str,
        default=None,
        help="Device for Whisper model (default: same as --device).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device for Qwen3-Omni model.",
    )
    parser.add_argument("--output-dir", type=str, default="results/qwen3_omni_wer")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument(
        "--speaker",
        type=str,
        default="Chelsie",
        help="Speaker name for Qwen3-Omni TTS.",
    )
    parser.add_argument(
        "--save-audio",
        action="store_true",
        help="Save generated WAV files.",
    )
    parser.add_argument("--disable-tqdm", action="store_true")
    args = parser.parse_args()

    benchmark(args)


if __name__ == "__main__":
    main()
