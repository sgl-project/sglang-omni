# SPDX-License-Identifier: Apache-2.0
"""Speech MMLU task: send audio questions, parse answers, compute accuracy.

Supports two evaluation modes:
- text: Audio-in -> Text-out (accuracy only)
- text+audio: Audio-in -> Text+Audio-out (accuracy + audio generation metrics)
"""

from __future__ import annotations

import base64
import csv
import json
import logging
import os
import time
from dataclasses import asdict, dataclass
from typing import Any

import aiohttp

from benchmarks.benchmarker.data import RequestResult
from benchmarks.benchmarker.runner import SendFn
from benchmarks.benchmarker.utils import get_wav_duration
from benchmarks.dataset.speech_mmlu import SpeechMmluSample
from benchmarks.metrics.accuracy import (
    INDEX_TO_LETTER,
    compute_accuracy_metrics,
    extract_answer_letter,
)

logger = logging.getLogger(__name__)

DEFAULT_PROMPT = (
    "Listen to the audio carefully and answer the multiple-choice question. "
    "Reply with only the letter of the correct answer (A, B, C, or D)."
)


@dataclass
class SpeechMmluResult:
    sample_id: str = ""
    subject: str = ""
    correct_answer: int = -1
    predicted_answer: int = -1  # -1 means unparseable
    raw_response: str = ""
    is_correct: bool = False
    is_parseable: bool = False
    latency_s: float = 0.0
    # Audio output fields (text+audio mode only)
    has_audio: bool = False
    audio_duration_s: float = 0.0
    error: str = ""


def make_speech_mmlu_send_fn(
    model_name: str,
    api_url: str,
    *,
    modalities: list[str] | None = None,
    prompt: str = DEFAULT_PROMPT,
    max_tokens: int = 32,
    temperature: float = 0.0,
    save_audio_dir: str | None = None,
) -> SendFn:
    """Return a SendFn for Speech MMLU evaluation via /v1/chat/completions."""
    if modalities is None:
        modalities = ["text"]

    audio_mode = "audio" in modalities

    async def send_fn(
        session: aiohttp.ClientSession, sample: SpeechMmluSample
    ) -> RequestResult:
        result = RequestResult(request_id=sample.sample_id)
        start_time = time.perf_counter()

        payload: dict[str, Any] = {
            "model": model_name,
            "messages": [{"role": "user", "content": prompt}],
            "audios": [sample.audio_path],
            "modalities": modalities,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False,
        }
        if audio_mode:
            payload["audio"] = {"format": "wav"}

        try:
            async with session.post(
                api_url,
                json=payload,
                headers={"Content-Type": "application/json"},
            ) as resp:
                resp.raise_for_status()
                resp_json = await resp.json()

            choices = resp_json.get("choices", [])
            if not choices:
                result.error = "No choices in response"
                return result

            message = choices[0].get("message", {})
            content = message.get("content", "")
            result.text = content
            result.is_success = True

            usage = resp_json.get("usage", {})
            result.prompt_tokens = usage.get("prompt_tokens", 0)
            result.completion_tokens = usage.get("completion_tokens", 0)

            # Extract audio if present
            if audio_mode:
                audio_obj = message.get("audio")
                if audio_obj and audio_obj.get("data"):
                    wav_bytes = base64.b64decode(audio_obj["data"])
                    if len(wav_bytes) > 44:  # WAV header size
                        result.audio_duration_s = get_wav_duration(wav_bytes)
                        if save_audio_dir:
                            wav_path = os.path.join(
                                save_audio_dir, f"{sample.sample_id}.wav"
                            )
                            with open(wav_path, "wb") as f:
                                f.write(wav_bytes)
                            result.wav_path = wav_path

        except (aiohttp.ClientError, Exception) as exc:
            result.error = str(exc)
        finally:
            result.latency_s = time.perf_counter() - start_time

        return result

    return send_fn


def build_speech_mmlu_results(
    request_results: list[RequestResult],
    samples: list[SpeechMmluSample],
    modalities: list[str] | None = None,
) -> list[SpeechMmluResult]:
    """Match RequestResults back to samples and extract answer predictions."""
    if modalities is None:
        modalities = ["text"]
    audio_mode = "audio" in modalities

    sample_map = {s.sample_id: s for s in samples}
    results = []

    for rr in request_results:
        sample = sample_map.get(rr.request_id)
        if sample is None:
            continue

        smr = SpeechMmluResult(
            sample_id=rr.request_id,
            subject=sample.subject,
            correct_answer=sample.correct_answer,
            raw_response=rr.text,
            latency_s=rr.latency_s,
            error=rr.error,
        )

        if rr.is_success and rr.text:
            pred = extract_answer_letter(rr.text)
            if pred is not None:
                smr.predicted_answer = pred
                smr.is_parseable = True
                smr.is_correct = pred == sample.correct_answer

        if audio_mode:
            smr.has_audio = rr.audio_duration_s > 0
            smr.audio_duration_s = rr.audio_duration_s

        results.append(smr)

    return results


def print_speech_mmlu_summary(
    metrics: dict[str, Any],
    model_name: str,
    *,
    speed_metrics: dict[str, Any] | None = None,
) -> None:
    """Pretty-print Speech MMLU evaluation results."""
    print("\n" + "=" * 60)
    print(f"  Speech MMLU Results — {model_name}")
    print("=" * 60)
    print(f"  Total samples:    {metrics['total_samples']}")
    print(f"  Parseable:        {metrics['parseable_samples']}")
    print(f"  Unparseable:      {metrics['unparseable_samples']}")
    print(f"  Correct:          {metrics['correct']}")
    print(f"  Incorrect:        {metrics['incorrect']}")
    print(f"  Overall accuracy: {metrics['overall_accuracy']:.2%}")
    print("-" * 60)
    print(f"  {'Subject':<40} {'Acc':>6}  {'N':>5}")
    print("-" * 60)
    for subj, info in sorted(
        metrics["per_subject"].items(), key=lambda x: -x[1]["accuracy"]
    ):
        print(f"  {subj:<40} {info['accuracy']:>6.2%}  {info['total']:>5}")
    print("=" * 60)

    if speed_metrics:
        print(f"\n  Latency mean:     {speed_metrics.get('latency_mean_s', 0):.3f}s")
        print(f"  Latency p95:      {speed_metrics.get('latency_p95_s', 0):.3f}s")
        print(
            f"  Throughput:       {speed_metrics.get('throughput_rps', 0):.2f} req/s"
        )
        print("=" * 60)
    print()


def save_speech_mmlu_results(
    results: list[SpeechMmluResult],
    metrics: dict[str, Any],
    config: dict[str, Any],
    output_dir: str,
    *,
    speed_metrics: dict[str, Any] | None = None,
) -> None:
    """Save evaluation results as JSON and CSV."""
    os.makedirs(output_dir, exist_ok=True)

    # JSON
    output = {
        "summary": metrics,
        "config": config,
        "per_sample": [asdict(r) for r in results],
    }
    if speed_metrics:
        output["speed_metrics"] = speed_metrics

    json_path = os.path.join(output_dir, "speech_mmlu_results.json")
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)
    logger.info("Saved results to %s", json_path)

    # CSV
    csv_path = os.path.join(output_dir, "speech_mmlu_results.csv")
    if results:
        fieldnames = list(asdict(results[0]).keys())
        # Add human-readable answer letters
        fieldnames.extend(["correct_letter", "predicted_letter"])
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in results:
                row = asdict(r)
                row["correct_letter"] = INDEX_TO_LETTER.get(r.correct_answer, "?")
                row["predicted_letter"] = (
                    INDEX_TO_LETTER.get(r.predicted_answer, "?")
                    if r.is_parseable
                    else ""
                )
                writer.writerow(row)
        logger.info("Saved CSV to %s", csv_path)
