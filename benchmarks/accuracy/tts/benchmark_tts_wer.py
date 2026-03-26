"""
Benchmark TTS accuracy with Seed-TTS Eval and compute normalized WER.

This benchmark reuses the online-serving request path from
`benchmarks.performance.tts.benchmark_tts_speed` so the generated audio and
speed metrics match the OpenAI-compatible `/v1/audio/speech` API.

Highlights:
  - Generates audio from the current server or reuses an existing audio dir
  - Transcribes generated WAV files with Whisper ASR
  - Applies Seed-TTS-style text normalization:
      - lowercase
      - unicode quote normalization
      - apostrophe collapsing (`T's` -> `ts`)
      - punctuation cleanup
      - optional digit-to-word normalization (`95` -> `ninety five`)
  - Saves per-sample transcripts and normalized WER to JSON
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import re
import unicodedata
from pathlib import Path
from typing import Any

from tqdm import tqdm

from benchmarks.performance.tts.benchmark_tts_speed import (
    _build_requests,
    _run_benchmark_requests,
    _run_warmup,
    _save_results,
    calculate_metrics,
    parse_meta_lst,
    print_summary,
    wait_for_service,
)

logger = logging.getLogger(__name__)

_ONES = {
    0: "zero",
    1: "one",
    2: "two",
    3: "three",
    4: "four",
    5: "five",
    6: "six",
    7: "seven",
    8: "eight",
    9: "nine",
    10: "ten",
    11: "eleven",
    12: "twelve",
    13: "thirteen",
    14: "fourteen",
    15: "fifteen",
    16: "sixteen",
    17: "seventeen",
    18: "eighteen",
    19: "nineteen",
}
_TENS = {
    20: "twenty",
    30: "thirty",
    40: "forty",
    50: "fifty",
    60: "sixty",
    70: "seventy",
    80: "eighty",
    90: "ninety",
}
_SCALES = (
    (1_000_000_000, "billion"),
    (1_000_000, "million"),
    (1_000, "thousand"),
    (100, "hundred"),
)
_QUOTE_TRANSLATION = str.maketrans(
    {
        "’": "'",
        "‘": "'",
        "‛": "'",
        "`": "'",
        "´": "'",
        "“": '"',
        "”": '"',
        "–": "-",
        "—": "-",
    }
)
_NUMBER_PATTERN = re.compile(r"\b\d[\d,]*\b")
_INTERNAL_APOSTROPHE_PATTERN = re.compile(r"(?<=\w)'(?=\w)")
_NON_ALNUM_PATTERN = re.compile(r"[^a-z0-9\s]")
_WHITESPACE_PATTERN = re.compile(r"\s+")


def _int_to_words(value: int) -> str:
    if value < 0:
        return f"minus {_int_to_words(-value)}"
    if value < 20:
        return _ONES[value]
    if value < 100:
        tens = (value // 10) * 10
        ones = value % 10
        if ones == 0:
            return _TENS[tens]
        return f"{_TENS[tens]} {_ONES[ones]}"

    for divisor, label in _SCALES:
        if value >= divisor:
            quotient, remainder = divmod(value, divisor)
            words = f"{_int_to_words(quotient)} {label}"
            if remainder == 0:
                return words
            return f"{words} {_int_to_words(remainder)}"

    raise ValueError(f"unsupported integer: {value}")


def _expand_number_tokens(text: str) -> str:
    def replace(match: re.Match[str]) -> str:
        raw = match.group(0).replace(",", "")
        return _int_to_words(int(raw))

    return _NUMBER_PATTERN.sub(replace, text)


def normalize_wer_text(
    text: str,
    *,
    normalize_numbers: bool = True,
) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.translate(_QUOTE_TRANSLATION).lower().strip()
    if normalize_numbers:
        text = _expand_number_tokens(text)
    text = _INTERNAL_APOSTROPHE_PATTERN.sub("", text)
    text = _NON_ALNUM_PATTERN.sub(" ", text)
    text = _WHITESPACE_PATTERN.sub(" ", text).strip()
    return text


def _word_edit_distance(reference_words: list[str], hypothesis_words: list[str]) -> int:
    if not reference_words:
        return len(hypothesis_words)
    if not hypothesis_words:
        return len(reference_words)

    previous_row = list(range(len(hypothesis_words) + 1))
    for ref_index, ref_word in enumerate(reference_words, start=1):
        current_row = [ref_index]
        for hyp_index, hyp_word in enumerate(hypothesis_words, start=1):
            substitution_cost = 0 if ref_word == hyp_word else 1
            current_row.append(
                min(
                    previous_row[hyp_index] + 1,
                    current_row[hyp_index - 1] + 1,
                    previous_row[hyp_index - 1] + substitution_cost,
                )
            )
        previous_row = current_row
    return previous_row[-1]


def compute_word_error_rate(references: list[str], hypotheses: list[str]) -> float:
    if len(references) != len(hypotheses):
        raise ValueError("references and hypotheses must have the same length")

    total_reference_words = 0
    total_edits = 0
    for reference, hypothesis in zip(references, hypotheses, strict=True):
        reference_words = reference.split()
        hypothesis_words = hypothesis.split()
        total_reference_words += len(reference_words)
        total_edits += _word_edit_distance(reference_words, hypothesis_words)

    if total_reference_words == 0:
        return 0.0
    return total_edits / total_reference_words


def _resolve_device(device: str) -> int | str:
    normalized = device.strip().lower()
    if normalized == "cpu":
        return "cpu"
    if normalized == "cuda":
        return 0
    if normalized.startswith("cuda:"):
        return int(normalized.split(":", 1)[1])
    return int(device)


def transcribe_audio_files(
    audio_paths: list[Path],
    *,
    asr_model: str,
    asr_language: str | None,
    asr_device: str,
) -> list[str]:
    from transformers import pipeline

    device = _resolve_device(asr_device)
    generate_kwargs: dict[str, Any] = {"task": "transcribe"}
    if asr_language:
        generate_kwargs["language"] = asr_language

    asr_pipeline = pipeline(
        "automatic-speech-recognition",
        model=asr_model,
        device=device,
    )

    hypotheses: list[str] = []
    for audio_path in tqdm(audio_paths, desc="ASR", disable=False):
        result = asr_pipeline(
            str(audio_path),
            return_timestamps=False,
            generate_kwargs=generate_kwargs,
        )
        hypotheses.append(str(result["text"]).strip())
    return hypotheses


def _build_pairs(
    samples: list[dict[str, Any]],
    hypotheses: list[str],
    *,
    normalize_numbers: bool,
) -> list[dict[str, Any]]:
    pairs: list[dict[str, Any]] = []
    for sample, hypothesis in zip(samples, hypotheses, strict=True):
        reference = sample["text"]
        reference_normalized = normalize_wer_text(
            reference,
            normalize_numbers=normalize_numbers,
        )
        hypothesis_normalized = normalize_wer_text(
            hypothesis,
            normalize_numbers=normalize_numbers,
        )
        pair_wer = compute_word_error_rate(
            [reference_normalized],
            [hypothesis_normalized],
        )
        pairs.append(
            {
                "id": sample["id"],
                "ref": reference,
                "hyp": hypothesis,
                "ref_norm": reference_normalized,
                "hyp_norm": hypothesis_normalized,
                "pair_wer": pair_wer,
            }
        )
    return pairs


def _save_wer_results(
    *,
    output_dir: Path,
    pairs: list[dict[str, Any]],
    wer_normalized: float,
    args: argparse.Namespace,
    speed_metrics: dict[str, Any] | None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "samples": len(pairs),
        "wer_normalized": wer_normalized,
        "config": {
            "model": args.model,
            "testset": args.testset,
            "output_dir": str(output_dir),
            "asr_model": args.asr_model,
            "asr_language": args.asr_language,
            "asr_device": args.asr_device,
            "normalize_numbers": not args.disable_number_normalization,
            "stream": args.stream,
            "no_ref_audio": args.no_ref_audio,
            "max_samples": args.max_samples,
            "skip_generation": args.skip_generation,
        },
        "speed_summary": speed_metrics,
        "pairs": pairs,
    }
    result_path = output_dir / "wer_results.json"
    result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    logger.info("WER results saved to %s", result_path)


def _collect_audio_paths(samples: list[dict[str, Any]], audio_dir: Path) -> list[Path]:
    audio_paths = [audio_dir / f"{sample['id']}.wav" for sample in samples]
    missing = [str(path) for path in audio_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "missing generated audio files:\n" + "\n".join(missing[:10])
        )
    return audio_paths


async def benchmark(args: argparse.Namespace) -> None:
    samples = parse_meta_lst(args.testset, args.max_samples)
    if not samples:
        raise FileNotFoundError(f"no samples loaded from {args.testset}")
    if args.stream and not args.skip_generation:
        raise ValueError(
            "streaming WER generation is not supported yet because the speed "
            "benchmark does not persist merged WAV files for SSE responses"
        )

    output_dir = Path(args.output_dir)
    audio_dir = output_dir / "audio"
    speed_metrics: dict[str, Any] | None = None

    if not args.skip_generation:
        base_url = args.base_url or f"http://{args.host}:{args.port}"
        api_url = f"{base_url}/v1/audio/speech"
        wait_for_service(base_url)

        requests_list = _build_requests(samples, api_url, args)
        logger.info("Prepared %d requests", len(requests_list))

        audio_dir.mkdir(parents=True, exist_ok=True)
        if args.warmup > 0:
            await _run_warmup(requests_list, args.warmup)

        logger.info(
            "Benchmarking %d requests (max_concurrency=%s)...",
            len(requests_list),
            args.max_concurrency,
        )
        outputs = await _run_benchmark_requests(
            requests_list,
            args,
            str(audio_dir),
        )
        speed_metrics = calculate_metrics(outputs)
        print_summary(speed_metrics, args)
        _save_results(outputs, speed_metrics, args, base_url)
        if args.generate_only:
            logger.info("Generation-only mode enabled; skipping ASR and WER scoring.")
            return

    audio_paths = _collect_audio_paths(samples, audio_dir)
    hypotheses = transcribe_audio_files(
        audio_paths,
        asr_model=args.asr_model,
        asr_language=args.asr_language,
        asr_device=args.asr_device,
    )

    pairs = _build_pairs(
        samples,
        hypotheses,
        normalize_numbers=not args.disable_number_normalization,
    )
    wer_normalized = compute_word_error_rate(
        [pair["ref_norm"] for pair in pairs],
        [pair["hyp_norm"] for pair in pairs],
    )
    _save_wer_results(
        output_dir=output_dir,
        pairs=pairs,
        wer_normalized=wer_normalized,
        args=args,
        speed_metrics=speed_metrics,
    )

    print(f"WER (normalized): {wer_normalized:.4%}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark TTS accuracy and compute normalized WER."
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
        default="default",
        help="Model name for the API request.",
    )
    parser.add_argument(
        "--testset",
        type=str,
        default="seed-tts-eval/en/meta.lst",
        help="Path to Seed-TTS Eval meta.lst.",
    )
    parser.add_argument(
        "--no-ref-audio",
        action="store_true",
        help="Skip ref audio/text from testset (TTS without voice cloning).",
    )
    parser.add_argument("--output-dir", type=str, default="results/tts_wer")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--repetition-penalty", type=float, default=None)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=1,
        help="Maximum concurrent requests.",
    )
    parser.add_argument(
        "--request-rate",
        type=float,
        default=float("inf"),
        help="Requests per second (inf = send all at once).",
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Send requests with stream=true (SSE audio chunks).",
    )
    parser.add_argument("--disable-tqdm", action="store_true")
    parser.add_argument(
        "--skip-generation",
        action="store_true",
        help="Reuse existing WAV files from output_dir/audio instead of calling the API.",
    )
    parser.add_argument(
        "--generate-only",
        action="store_true",
        help="Only generate audio and speed results, then exit before ASR/WER.",
    )
    parser.add_argument(
        "--asr-model",
        type=str,
        default="openai/whisper-large-v3-turbo",
        help="ASR model used to transcribe generated audio.",
    )
    parser.add_argument(
        "--asr-language",
        type=str,
        default="en",
        help="Language hint passed to Whisper generation.",
    )
    parser.add_argument(
        "--asr-device",
        type=str,
        default="cuda:0",
        help="ASR device: cpu, cuda, cuda:0, or a device index like 0.",
    )
    parser.add_argument(
        "--disable-number-normalization",
        action="store_true",
        help="Disable digit-to-word normalization before WER scoring.",
    )
    return parser


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    args = build_arg_parser().parse_args()
    asyncio.run(benchmark(args))


if __name__ == "__main__":
    main()
