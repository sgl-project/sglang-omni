# SPDX-License-Identifier: Apache-2.0
"""MMMU benchmark case -- send_fn, metrics, and persistence.

Answer-parsing helpers (``parse_multi_choice_response``,
``parse_open_response``, ``eval_open``) live in
``benchmarks.metrics.accuracy``. They are re-exported from this module
for backwards compatibility with existing call sites.
"""

from __future__ import annotations

import asyncio
import base64
import logging
import os
import random
import time

import aiohttp

from benchmarks.benchmarker.data import RequestResult
from benchmarks.benchmarker.runner import SendFn
from benchmarks.dataset.mmmu import MMMUSample, image_to_data_uri
from benchmarks.metrics.accuracy import (
    eval_open,
    parse_multi_choice_response,
    parse_open_response,
)

__all__ = [
    "MULTI_CHOICE_INSTRUCTION",
    "compute_mmmu_metrics",
    "eval_open",
    "make_mmmu_send_fn",
    "parse_multi_choice_response",
    "parse_open_response",
    "print_mmmu_accuracy_summary",
]

logger = logging.getLogger(__name__)

SUMMARY_LABEL_WIDTH = 28
SUMMARY_LINE_WIDTH = 50

MULTI_CHOICE_INSTRUCTION = (
    "\nAnswer the following multiple-choice question. "
    "The last line of your response should be of the "
    "following format: 'Answer: $LETTER' (without quotes) "
    "where LETTER is one of the options. "
    "Think step by step before answering."
)


def make_mmmu_send_fn(
    model_name: str,
    api_url: str,
    *,
    max_tokens: int = 2048,
    temperature: float = 0.0,
    enable_audio: bool = False,
    audio_dir: str | None = None,
) -> SendFn:
    """Return a *send_fn* that sends an MMMUSample to /v1/chat/completions.

    Uses the sglang-omni request format with a top-level ``images`` field.
    When *enable_audio* is ``False`` (default), requests text-only output.
    When ``True``, requests ``["text", "audio"]`` modalities, decodes the
    audio response, saves it as a WAV file under *audio_dir*, and stores
    the path in ``RequestResult.wav_path``.
    """
    modalities = ["text", "audio"] if enable_audio else ["text"]
    if enable_audio:
        import soundfile as sf

    async def send_fn(
        session: aiohttp.ClientSession, sample: MMMUSample
    ) -> RequestResult:
        result = RequestResult(
            request_id=sample.sample_id,
            text=sample.prompt[:60],
        )

        payload: dict = {
            "model": model_name,
            "messages": [{"role": "user", "content": sample.prompt}],
            "images": [image_to_data_uri(img) for img in sample.images],
            "modalities": modalities,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False,
        }
        if enable_audio:
            payload["audio"] = {"format": "wav"}

        start_time = time.perf_counter()
        try:
            async with session.post(api_url, json=payload) as response:
                response.raise_for_status()
                body = await response.json()

            message = body.get("choices", [{}])[0].get("message", {})
            content = message.get("content", "")
            result.text = content or ""

            if enable_audio and audio_dir:
                audio_obj = message.get("audio")
                if audio_obj is None:
                    result.error = "No audio in response"
                    return result
                audio_b64 = audio_obj.get("data", "")
                if not audio_b64:
                    result.error = "Empty audio data in response"
                    return result
                wav_path = os.path.join(audio_dir, f"{sample.sample_id}.wav")
                with open(wav_path, "wb") as f:
                    f.write(base64.b64decode(audio_b64))
                result.wav_path = wav_path

                wav_info = sf.info(wav_path)
                result.audio_duration_s = round(wav_info.duration, 4)

            result.is_success = True

            usage = body.get("usage", {})
            if usage:
                result.prompt_tokens = usage.get("prompt_tokens", 0)
                result.completion_tokens = usage.get("completion_tokens", 0)

            elapsed = time.perf_counter() - start_time
            result.engine_time_s = elapsed
            if result.audio_duration_s > 0:
                result.rtf = elapsed / result.audio_duration_s
            if result.completion_tokens > 0 and result.engine_time_s > 0:
                result.tok_per_s = result.completion_tokens / result.engine_time_s
        except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
            result.error = str(exc)
        finally:
            result.latency_s = time.perf_counter() - start_time

        return result

    return send_fn


def compute_mmmu_metrics(
    samples: list[MMMUSample],
    results: list[RequestResult],
) -> tuple[dict, list[dict]]:
    """Parse answers, compute accuracy, and build per-sample detail list.

    Returns ``(summary_dict, per_sample_list)``.
    """
    assert len(samples) == len(
        results
    ), f"Sample/result count mismatch: {len(samples)} samples vs {len(results)} results"
    # Fix the random seed so that MC fallback choices are deterministic across
    # CI runs.  The MMMU reference eval also uses random fallback, so this
    # does not change the evaluation methodology.
    random.seed(42)

    correct = 0
    failed = 0
    mc_fallback = 0
    per_sample: list[dict] = []

    for sample, result in zip(samples, results):
        record = {
            "sample_id": sample.sample_id,
            "subject": sample.subject,
            "question_type": sample.question_type,
            "expected": sample.answer,
            "latency_s": round(result.latency_s, 4),
            "prompt_tokens": result.prompt_tokens,
            "completion_tokens": result.completion_tokens,
            "tok_per_s": (round(result.tok_per_s, 1) if result.tok_per_s > 0 else None),
        }

        if not result.is_success:
            record.update(
                predicted="",
                raw_response=result.error,
                is_correct=False,
                is_success=False,
                error=result.error,
            )
            failed += 1
        else:
            gold = sample.answer
            if (
                sample.question_type == "multiple-choice"
                and sample.all_choices
                and sample.index2ans
            ):
                predicted, is_fallback = parse_multi_choice_response(
                    result.text,
                    sample.all_choices,
                    sample.index2ans,
                )
                if is_fallback:
                    mc_fallback += 1
                    logger.debug(
                        f"MMMU multi-choice parse fallback for sample "
                        f"{sample.sample_id}"
                    )
                is_correct = gold is not None and predicted == gold
            else:
                parsed_list = parse_open_response(result.text)
                is_correct = gold is not None and eval_open(gold, parsed_list)
                predicted = ", ".join(map(str, parsed_list))

            if is_correct:
                correct += 1

            record.update(
                predicted=predicted,
                raw_response=result.text,
                is_correct=is_correct,
                is_success=True,
                error="",
            )

        per_sample.append(record)

    total = len(samples)
    accuracy = correct / total if total > 0 else 0.0

    summary = {
        "total_samples": total,
        "correct": correct,
        "accuracy": round(accuracy, 4),
        "failed": failed,
        "mc_fallback": mc_fallback,
    }
    return summary, per_sample


def print_mmmu_accuracy_summary(metrics: dict, model_name: str) -> None:
    """Print formatted MMMU accuracy summary to stdout."""
    lw = SUMMARY_LABEL_WIDTH
    print(f"\n{'=' * SUMMARY_LINE_WIDTH}")
    print(f"  MMMU Accuracy — {model_name}")
    print(f"{'=' * SUMMARY_LINE_WIDTH}")
    print(f"  {'Total samples:':<{lw}} {metrics['total_samples']}")
    print(f"  {'Correct:':<{lw}} {metrics['correct']}")
    print(
        f"  {'Accuracy:':<{lw}} {metrics['accuracy']:.4f} "
        f"({metrics['accuracy'] * 100:.1f}%)"
    )
    print(f"  {'Failed requests:':<{lw}} {metrics['failed']}")
    print(f"  {'MC parse fallback:':<{lw}} {metrics['mc_fallback']}")
    print(f"{'=' * SUMMARY_LINE_WIDTH}\n")
