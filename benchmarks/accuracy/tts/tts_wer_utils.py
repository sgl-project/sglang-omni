"""
Shared utilities for TTS WER (Word Error Rate) benchmarks.

Provides model-agnostic building blocks for evaluating TTS quality on the
seed-tts-eval dataset:

- Data structures: ``SampleInput``, ``SampleOutput``
- Dataset parsing: ``parse_meta_lst``
- Text normalization: ``normalize_text``
- ASR loading / transcription: ``load_asr_model``, ``transcribe``
- WER computation: ``compute_wer``
- Metrics aggregation: ``calculate_metrics``
- Result persistence: ``save_results``
- Server health check: ``wait_for_service``
"""

from __future__ import annotations

import json
import logging
import os
import re
import struct
import time
import unicodedata
from dataclasses import dataclass
from typing import Any

import numpy as np
import requests

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

META_FIELD_COUNT = 4
DEFAULT_DATASET_DIR = "seedtts_testset"
SUMMARY_LINE_WIDTH = 60
SUMMARY_LABEL_WIDTH = 30
WAV_HEADER_SIZE = 44

DEFAULT_WHISPER_EN = "base.en"
DEFAULT_WHISPER_ZH = "medium"


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

# Number-to-word mapping for English (0-99 + common large numbers)
_ONES = [
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "eleven",
    "twelve",
    "thirteen",
    "fourteen",
    "fifteen",
    "sixteen",
    "seventeen",
    "eighteen",
    "nineteen",
]
_TENS = [
    "",
    "",
    "twenty",
    "thirty",
    "forty",
    "fifty",
    "sixty",
    "seventy",
    "eighty",
    "ninety",
]


def _number_to_words(n: int) -> str:
    """Convert an integer to English words (supports 0-999,999,999)."""
    if n < 0:
        return "minus " + _number_to_words(-n)
    if n < 20:
        return _ONES[n]
    if n < 100:
        return _TENS[n // 10] + ("" if n % 10 == 0 else " " + _ONES[n % 10])
    if n < 1000:
        rest = _number_to_words(n % 100) if n % 100 != 0 else ""
        return _ONES[n // 100] + " hundred" + (" " + rest if rest else "")
    if n < 1_000_000:
        rest = _number_to_words(n % 1000) if n % 1000 != 0 else ""
        return _number_to_words(n // 1000) + " thousand" + (" " + rest if rest else "")
    if n < 1_000_000_000:
        rest = _number_to_words(n % 1_000_000) if n % 1_000_000 != 0 else ""
        return (
            _number_to_words(n // 1_000_000) + " million" + (" " + rest if rest else "")
        )
    return str(n)


def _normalize_numbers(text: str) -> str:
    """Replace digit sequences with their English word equivalents."""

    def _replace_match(m: re.Match) -> str:
        try:
            return _number_to_words(int(m.group()))
        except (ValueError, RecursionError):
            return m.group()

    return re.sub(r"\d+", _replace_match, text)


def normalize_text(text: str, lang: str = "en") -> str:
    """Normalize text for WER comparison."""
    # Unicode NFKC normalization
    text = unicodedata.normalize("NFKC", text)
    # Lowercase
    text = text.lower()
    # Normalize numbers to words (English only)
    if lang == "en":
        text = _normalize_numbers(text)
    # Remove punctuation (keep CJK characters and alphanumeric)
    if lang == "zh":
        # For Chinese: keep CJK chars and digits, remove everything else
        text = re.sub(r"[^\u4e00-\u9fff\u3400-\u4dbf0-9]", " ", text)
    else:
        # For English: keep alphabetic only (digits already converted to words)
        text = re.sub(r"[^a-z]", " ", text)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ---------------------------------------------------------------------------
# ASR: Whisper (EN) / FunASR (ZH)
# ---------------------------------------------------------------------------

_asr_model = None
_asr_lang = None


def load_asr_model(lang: str, device: str = "cpu", whisper_model: str | None = None):
    """Load ASR model: Whisper for EN, FunASR for ZH.

    For Chinese (``lang="zh"``), FunASR is attempted first. If it is not
    installed or fails to load, the function falls back to Whisper.
    """
    global _asr_model, _asr_lang

    if _asr_model is not None and _asr_lang == lang:
        return _asr_model

    if lang == "zh" and whisper_model is None:
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

    if whisper_model is None:
        whisper_model = DEFAULT_WHISPER_ZH if lang == "zh" else DEFAULT_WHISPER_EN
    logger.info("Loading Whisper model (%s) for %s...", whisper_model, lang)
    _asr_model = whisper.load_model(whisper_model, device=device)
    _asr_lang = lang
    logger.info("Whisper model loaded")
    return _asr_model


def transcribe(
    audio_path: str,
    lang: str,
    asr_device: str = "cpu",
    whisper_model: str | None = None,
) -> str:
    """Transcribe audio file to text using the loaded ASR model."""
    model = load_asr_model(lang, device=asr_device, whisper_model=whisper_model)

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
# WAV helpers
# ---------------------------------------------------------------------------


def get_wav_duration(wav_bytes: bytes) -> float:
    """Return PCM playback length in seconds for raw WAV bytes."""
    if len(wav_bytes) <= WAV_HEADER_SIZE:
        return 0.0
    sample_rate = struct.unpack_from("<I", wav_bytes, 24)[0]
    num_channels = struct.unpack_from("<H", wav_bytes, 22)[0]
    bits_per_sample = struct.unpack_from("<H", wav_bytes, 34)[0]
    if sample_rate == 0 or num_channels == 0 or bits_per_sample == 0:
        return 0.0
    bytes_per_sample = num_channels * bits_per_sample // 8
    pcm_size = len(wav_bytes) - WAV_HEADER_SIZE
    return pcm_size / (sample_rate * bytes_per_sample)


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


def print_wer_metrics(metrics: dict) -> None:
    """Print the common WER / latency / duration metrics block."""
    lw = SUMMARY_LABEL_WIDTH
    w = SUMMARY_LINE_WIDTH
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
    output_dir: str,
    config: dict[str, Any],
) -> None:
    """Persist per-sample results and summary metrics to JSON.

    Parameters
    ----------
    outputs:
        Per-sample evaluation results.
    metrics:
        Aggregated metrics from :func:`calculate_metrics`.
    output_dir:
        Directory where ``wer_results.json`` will be written.
    config:
        Model-specific configuration dict to record alongside the results.
    """
    os.makedirs(output_dir, exist_ok=True)

    results = {
        "summary": metrics,
        "config": config,
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

    json_path = os.path.join(output_dir, "wer_results.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info("Results saved to %s", json_path)


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------


def wait_for_service(base_url: str, timeout: int = 300) -> None:
    """Wait for the sgl-omni server to be ready."""
    logger.info("Waiting for service at %s ...", base_url)
    start = time.time()
    while True:
        try:
            resp = requests.get(f"{base_url}/health", timeout=2)
            if resp.status_code == 200:
                logger.info("Service is ready.")
                return
        except requests.exceptions.RequestException:
            pass
        if time.time() - start > timeout:
            raise TimeoutError(f"Service at {base_url} not ready within {timeout}s")
        time.sleep(2)
