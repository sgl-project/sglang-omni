"""
Benchmark TTS accuracy (WER) for Ming-flash-omni's BailingTalker2.

Evaluates voice-cloning TTS quality using the seed-tts-eval dataset by:
1. Loading BailingTalker2 + AudioVAE directly on a single GPU (~4GB VRAM)
2. For each sample: voice-clone from ref_audio/ref_text, generate speech for
   target_text, ASR-transcribe, and compute WER
3. Reporting corpus-level micro-average WER + per-sample stats

Dataset: seed-tts-eval

    Download:
        huggingface-cli download zhaochenyang20/seed-tts-eval \
            --repo-type dataset --local-dir seedtts_testset

Usage:
    python benchmarks/accuracy/tts/benchmark_ming_tts_wer.py \
        --meta /tmp/seedtts_testset/zh/meta.lst \
        --model-path inclusionAI/Ming-flash-omni-2.0 \
        --output-dir /tmp/results_ming_zh \
        --lang zh --device cuda:0 --max-samples 50
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import time
import unicodedata
from dataclasses import dataclass

import numpy as np
import torch
import torchaudio
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

META_FIELD_COUNT = 4
SUMMARY_LINE_WIDTH = 60
SUMMARY_LABEL_WIDTH = 30


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class SampleInput:
    sample_id: str
    ref_text: str
    ref_audio: str
    target_text: str


@dataclass
class SampleOutput:
    sample_id: str
    target_text: str
    hypothesis: str = ""
    wer: float = 0.0
    latency: float = 0.0
    audio_duration: float = 0.0
    error: str = ""
    is_success: bool = False


# ---------------------------------------------------------------------------
# Parsing seed-tts-eval meta.lst
# ---------------------------------------------------------------------------


def parse_meta_lst(path: str, max_samples: int | None = None) -> list[SampleInput]:
    """Parse a seed-tts-eval meta.lst file (format: id|ref_text|ref_audio_path|text)."""
    base_dir = os.path.dirname(path)
    samples: list[SampleInput] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            fields = line.split("|")
            if len(fields) < META_FIELD_COUNT:
                continue
            samples.append(
                SampleInput(
                    sample_id=fields[0],
                    ref_text=fields[1],
                    ref_audio=os.path.join(base_dir, fields[2]),
                    target_text=fields[3],
                )
            )
            if max_samples and len(samples) >= max_samples:
                break
    return samples


# ---------------------------------------------------------------------------
# Text normalization
# ---------------------------------------------------------------------------


def normalize_text(text: str, lang: str = "en") -> str:
    """Normalize text for WER comparison."""
    # Unicode NFKC normalization
    text = unicodedata.normalize("NFKC", text)
    # Lowercase
    text = text.lower()
    # Remove punctuation (keep CJK characters and alphanumeric)
    if lang == "zh":
        # For Chinese: keep CJK chars and digits, remove everything else
        text = re.sub(r"[^\u4e00-\u9fff\u3400-\u4dbf0-9]", " ", text)
    else:
        # For English: keep alphanumeric
        text = re.sub(r"[^a-z0-9]", " ", text)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ---------------------------------------------------------------------------
# ASR: Whisper (EN) / FunASR (ZH)
# ---------------------------------------------------------------------------

_asr_model = None
_asr_lang = None


def load_asr_model(lang: str, device: str = "cpu"):
    """Load ASR model: Whisper for EN, FunASR for ZH."""
    global _asr_model, _asr_lang

    if _asr_model is not None and _asr_lang == lang:
        return _asr_model

    if lang == "zh":
        try:
            from funasr import AutoModel as FunAutoModel

            logger.info("Loading FunASR model for Chinese...")
            _asr_model = FunAutoModel(
                model="iic/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
                device=device,
                disable_update=True,
            )
            _asr_lang = lang
            logger.info("FunASR model loaded")
            return _asr_model
        except (ImportError, Exception) as e:
            logger.warning(
                "FunASR failed (%s), falling back to Whisper for Chinese.", e
            )

    # Whisper for EN or fallback
    import whisper

    model_size = "medium" if lang == "zh" else "base.en"
    logger.info("Loading Whisper model (%s) for %s...", model_size, lang)
    _asr_model = whisper.load_model(model_size, device=device)
    _asr_lang = lang
    logger.info("Whisper model loaded")
    return _asr_model


def transcribe(audio_path: str, lang: str, asr_device: str = "cpu") -> str:
    """Transcribe audio file to text using the loaded ASR model."""
    model = load_asr_model(lang, device=asr_device)

    if lang == "zh" and hasattr(model, "generate"):
        # FunASR interface
        result = model.generate(input=audio_path)
        if isinstance(result, list) and len(result) > 0:
            item = result[0]
            if isinstance(item, dict):
                return item.get("text", "")
            return str(item)
        return str(result)
    else:
        # Whisper interface
        result = model.transcribe(
            audio_path,
            language="zh" if lang == "zh" else "en",
        )
        return result.get("text", "")


# ---------------------------------------------------------------------------
# WER computation
# ---------------------------------------------------------------------------


def compute_wer(reference: str, hypothesis: str, lang: str = "en") -> float:
    """Compute Word Error Rate between reference and hypothesis."""
    ref = normalize_text(reference, lang)
    hyp = normalize_text(hypothesis, lang)

    if lang == "zh":
        # Character-level for Chinese
        ref_tokens = list(ref.replace(" ", ""))
        hyp_tokens = list(hyp.replace(" ", ""))
    else:
        ref_tokens = ref.split()
        hyp_tokens = hyp.split()

    if not ref_tokens:
        return 0.0 if not hyp_tokens else 1.0

    # Levenshtein distance via dynamic programming
    n = len(ref_tokens)
    m = len(hyp_tokens)
    d = [[0] * (m + 1) for _ in range(n + 1)]

    for i in range(n + 1):
        d[i][0] = i
    for j in range(m + 1):
        d[0][j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref_tokens[i - 1] == hyp_tokens[j - 1]:
                d[i][j] = d[i - 1][j - 1]
            else:
                d[i][j] = min(
                    d[i - 1][j] + 1,  # deletion
                    d[i][j - 1] + 1,  # insertion
                    d[i - 1][j - 1] + 1,  # substitution
                )

    return d[n][m] / n


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def _resolve_model_path(model_path: str) -> str:
    """Resolve a HF repo ID or local path to a local directory."""
    if os.path.isdir(model_path):
        return model_path
    # Treat as HF repo ID — download/resolve to local cache
    from huggingface_hub import snapshot_download

    logger.info("Resolving HF repo %s to local cache...", model_path)
    return snapshot_download(model_path)


def load_ming_talker(model_path: str, device: str = "cuda"):
    """Load BailingTalker2 + AudioVAE."""
    from sglang_omni.models.ming_omni.talker import AudioVAE, BailingTalker2

    local_path = _resolve_model_path(model_path)
    talker_path = os.path.join(local_path, "talker")

    logger.info("Loading BailingTalker2 from %s ...", talker_path)
    t0 = time.time()
    talker = BailingTalker2.from_pretrained(talker_path, dtype=torch.bfloat16)
    talker.to(device)
    talker.eval()
    logger.info("BailingTalker2 loaded in %.1fs", time.time() - t0)

    logger.info("Initializing CUDA graphs...")
    t0g = time.time()
    talker.initial_graph()
    logger.info("CUDA graphs initialized in %.1fs", time.time() - t0g)

    vae_path = os.path.join(talker_path, "vae")
    logger.info("Loading AudioVAE from %s ...", vae_path)
    t0v = time.time()
    vae = AudioVAE.from_pretrained(vae_path, dtype=torch.bfloat16)
    vae.to(device)
    vae.eval()
    logger.info("AudioVAE loaded in %.1fs", time.time() - t0v)

    return talker, vae


# ---------------------------------------------------------------------------
# Speech generation
# ---------------------------------------------------------------------------


@torch.no_grad()
def generate_speech_ming(
    talker,
    vae,
    sample: SampleInput,
) -> tuple[np.ndarray | None, int]:
    """Generate speech using BailingTalker2 with voice cloning.

    Returns:
        Tuple of (waveform as numpy float32, sample_rate).
    """
    all_wavs = []
    for tts_speech, _, _, _ in talker.omni_audio_generation(
        tts_text=sample.target_text,
        voice_name=None,
        prompt_text=sample.ref_text,
        prompt_wav_path=sample.ref_audio,
        audio_detokenizer=vae,
        stream=False,
    ):
        if tts_speech is not None:
            all_wavs.append(tts_speech)

    if not all_wavs:
        return None, 44100

    waveform = torch.cat(all_wavs, dim=-1)
    sample_rate = getattr(vae.config, "sample_rate", 44100)
    return waveform.squeeze().cpu().float().numpy(), sample_rate


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def calculate_metrics(outputs: list[SampleOutput]) -> dict:
    """Compute corpus-level micro-average WER and summary stats."""
    successes = [o for o in outputs if o.is_success]
    if not successes:
        return {
            "completed": 0,
            "failed": len(outputs),
            "corpus_wer": None,
        }

    wers = [o.wer for o in successes]
    latencies = [o.latency for o in successes]
    durations = [o.audio_duration for o in successes if o.audio_duration > 0]

    return {
        "completed": len(successes),
        "failed": len(outputs) - len(successes),
        "corpus_wer": round(float(np.mean(wers)), 4),
        "wer_median": round(float(np.median(wers)), 4),
        "wer_std": round(float(np.std(wers)), 4),
        "wer_p95": round(float(np.percentile(wers, 95)), 4),
        "latency_mean_s": round(float(np.mean(latencies)), 3),
        "latency_median_s": round(float(np.median(latencies)), 3),
        "audio_duration_mean_s": (
            round(float(np.mean(durations)), 3) if durations else 0
        ),
    }


def print_summary(metrics: dict, args: argparse.Namespace) -> None:
    lw = SUMMARY_LABEL_WIDTH
    w = SUMMARY_LINE_WIDTH
    print(f"\n{'=' * w}")
    print(f"{'Ming TTS WER Benchmark Result':^{w}}")
    print(f"{'=' * w}")
    print(f"  {'Model:':<{lw}} {args.model_path}")
    print(f"  {'Language:':<{lw}} {args.lang}")
    print(f"  {'Completed samples:':<{lw}} {metrics['completed']}")
    print(f"  {'Failed samples:':<{lw}} {metrics['failed']}")
    print(f"{'-' * w}")
    if metrics.get("corpus_wer") is not None:
        print(f"  {'Corpus WER (micro-avg):':<{lw}} {metrics['corpus_wer']:.4f}")
        print(f"  {'WER median:':<{lw}} {metrics['wer_median']:.4f}")
        print(f"  {'WER std:':<{lw}} {metrics['wer_std']:.4f}")
        print(f"  {'WER p95:':<{lw}} {metrics['wer_p95']:.4f}")
    print(f"{'-' * w}")
    if metrics.get("latency_mean_s"):
        print(f"  {'Latency mean (s):':<{lw}} {metrics['latency_mean_s']}")
        print(f"  {'Latency median (s):':<{lw}} {metrics['latency_median_s']}")
    if metrics.get("audio_duration_mean_s"):
        print(
            f"  {'Audio duration mean (s):':<{lw}} {metrics['audio_duration_mean_s']}"
        )
    print(f"{'=' * w}")


# ---------------------------------------------------------------------------
# Save results
# ---------------------------------------------------------------------------


def save_results(
    outputs: list[SampleOutput],
    metrics: dict,
    args: argparse.Namespace,
) -> None:
    os.makedirs(args.output_dir, exist_ok=True)

    results = {
        "summary": metrics,
        "config": {
            "model_path": args.model_path,
            "meta": args.meta,
            "lang": args.lang,
            "device": args.device,
            "max_samples": args.max_samples,
            "asr_device": args.asr_device,
        },
        "per_sample": [
            {
                "id": o.sample_id,
                "target_text": o.target_text,
                "hypothesis": o.hypothesis,
                "wer": round(o.wer, 4),
                "latency_s": round(o.latency, 4),
                "audio_duration_s": round(o.audio_duration, 4),
                "is_success": o.is_success,
                "error": o.error or None,
            }
            for o in outputs
        ],
    }

    json_path = os.path.join(args.output_dir, "wer_results.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info("Results saved to %s", json_path)


# ---------------------------------------------------------------------------
# Main benchmark loop
# ---------------------------------------------------------------------------


def benchmark(args: argparse.Namespace) -> None:
    if not os.path.isfile(args.meta):
        logger.error("Meta file not found: %s", args.meta)
        return

    samples = parse_meta_lst(args.meta, args.max_samples)
    logger.info("Loaded %d samples from %s", len(samples), args.meta)

    # Load TTS model
    talker, vae = load_ming_talker(args.model_path, device=args.device)

    # Pre-load ASR model
    load_asr_model(args.lang, device=args.asr_device)

    # Create audio output dir
    audio_dir = os.path.join(args.output_dir, "audio")
    os.makedirs(audio_dir, exist_ok=True)

    outputs: list[SampleOutput] = []
    for sample in tqdm(samples, desc="Generating & evaluating"):
        output = SampleOutput(
            sample_id=sample.sample_id,
            target_text=sample.target_text,
        )

        try:
            # Generate speech
            t0 = time.time()
            waveform, sample_rate = generate_speech_ming(talker, vae, sample)
            output.latency = time.time() - t0

            if waveform is None or len(waveform) == 0:
                output.error = "Empty waveform"
                outputs.append(output)
                continue

            output.audio_duration = len(waveform) / sample_rate

            # Save audio
            audio_path = os.path.join(audio_dir, f"{sample.sample_id}.wav")
            waveform_tensor = torch.from_numpy(waveform).unsqueeze(0)
            torchaudio.save(audio_path, waveform_tensor, sample_rate)

            # ASR transcribe
            hypothesis = transcribe(audio_path, args.lang, asr_device=args.asr_device)
            output.hypothesis = hypothesis

            # Compute WER
            output.wer = compute_wer(sample.target_text, hypothesis, args.lang)
            output.is_success = True

            logger.debug(
                "[%s] WER=%.4f | ref=%r | hyp=%r",
                sample.sample_id,
                output.wer,
                sample.target_text[:80],
                hypothesis[:80],
            )
        except Exception as e:
            output.error = str(e)
            logger.error("Error on sample %s: %s", sample.sample_id, e, exc_info=True)

        outputs.append(output)

    # Compute and print metrics
    metrics = calculate_metrics(outputs)
    print_summary(metrics, args)
    save_results(outputs, metrics, args)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark Ming TTS accuracy (WER) using seed-tts-eval."
    )
    parser.add_argument(
        "--meta",
        type=str,
        required=True,
        help="Path to seed-tts-eval meta.lst file.",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="inclusionAI/Ming-flash-omni-2.0",
        help="Path to Ming-flash-omni model (parent dir containing talker/).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/ming_tts_wer",
        help="Directory to save results and audio files.",
    )
    parser.add_argument(
        "--lang",
        type=str,
        default="zh",
        choices=["zh", "en"],
        help="Language for ASR and text normalization.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device for TTS model.",
    )
    parser.add_argument(
        "--asr-device",
        type=str,
        default="cpu",
        help="Device for ASR model (cpu recommended to save GPU memory).",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples to process.",
    )
    args = parser.parse_args()

    benchmark(args)


if __name__ == "__main__":
    main()
