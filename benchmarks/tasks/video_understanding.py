# SPDX-License-Identifier: Apache-2.0
"""Video understanding benchmark helpers."""

from __future__ import annotations

import asyncio
import base64
import binascii
import logging
import os
import random
import struct
import time
from collections import defaultdict
from typing import Any

import aiohttp

from benchmarks.benchmarker.data import RequestResult
from benchmarks.benchmarker.runner import SendFn
from benchmarks.benchmarker.utils import get_wav_duration, print_accuracy_breakdown
from benchmarks.dataset.videomme import VideoMMESample
from benchmarks.tasks.visual_understand import parse_multi_choice_response

logger = logging.getLogger(__name__)

SUMMARY_LABEL_WIDTH = 28
SUMMARY_LINE_WIDTH = 50


def make_videomme_send_fn(
    model_name: str,
    api_url: str,
    *,
    max_tokens: int = 256,
    temperature: float = 0.0,
    video_fps: float | None = None,
    video_max_frames: int | None = None,
    video_min_pixels: int | None = None,
    video_max_pixels: int | None = None,
    video_total_pixels: int | None = None,
    enable_audio: bool = False,
    audio_dir: str | None = None,
) -> SendFn:
    modalities = ["text", "audio"] if enable_audio else ["text"]

    async def send_fn(
        session: aiohttp.ClientSession,
        sample: VideoMMESample,
    ) -> RequestResult:
        result = RequestResult(
            request_id=sample.sample_id,
            text=sample.prompt[:60],
        )

        payload: dict[str, Any] = {
            "model": model_name,
            "messages": [{"role": "user", "content": sample.prompt}],
            "videos": [sample.video_path],
            "modalities": modalities,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False,
        }
        if enable_audio:
            payload["audio"] = {"format": "wav"}
        if video_fps is not None:
            payload["video_fps"] = video_fps
        if video_max_frames is not None:
            payload["video_max_frames"] = video_max_frames
        if video_min_pixels is not None:
            payload["video_min_pixels"] = video_min_pixels
        if video_max_pixels is not None:
            payload["video_max_pixels"] = video_max_pixels
        if video_total_pixels is not None:
            payload["video_total_pixels"] = video_total_pixels

        start_time = time.perf_counter()
        try:
            async with session.post(api_url, json=payload) as response:
                response.raise_for_status()
                body = await response.json()

            message = body.get("choices", [{}])[0].get("message", {})
            result.text = message.get("content", "") or ""

            if enable_audio and audio_dir:
                audio_obj = message.get("audio")
                if not isinstance(audio_obj, dict):
                    result.error = "No audio in response"
                    return result
                audio_b64 = audio_obj.get("data", "")
                if not audio_b64:
                    result.error = "Empty audio data in response"
                    return result
                try:
                    wav_bytes = base64.b64decode(audio_b64, validate=True)
                    result.audio_duration_s = round(get_wav_duration(wav_bytes), 4)
                except (binascii.Error, ValueError, struct.error) as exc:
                    result.error = f"Invalid audio data: {exc}"
                    return result

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

            if enable_audio and audio_dir and result.audio_duration_s > 0:
                try:
                    os.makedirs(audio_dir, exist_ok=True)
                    wav_path = os.path.join(audio_dir, f"{sample.sample_id}.wav")
                    with open(wav_path, "wb") as f:
                        f.write(wav_bytes)
                except OSError as exc:
                    result.error = f"Failed to save audio: {exc}"
                    return result
                result.wav_path = wav_path
            result.is_success = True
        except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
            result.error = str(exc)
        finally:
            result.latency_s = time.perf_counter() - start_time

        return result

    return send_fn


def _finalize_breakdown(
    buckets: dict[str, dict[str, int]]
) -> dict[str, dict[str, Any]]:
    return {
        key: {
            "total": value["total"],
            "correct": value["correct"],
            "accuracy": (
                round(value["correct"] / value["total"], 4)
                if value["total"] > 0
                else 0.0
            ),
        }
        for key, value in sorted(buckets.items())
    }


def compute_videomme_metrics(
    samples: list[VideoMMESample],
    results: list[RequestResult],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    assert len(samples) == len(
        results
    ), f"Sample/result count mismatch: {len(samples)} samples vs {len(results)} results"
    random.seed(42)

    correct = 0
    failed = 0
    mc_fallback = 0
    per_duration: dict[str, dict[str, int]] = defaultdict(
        lambda: {"total": 0, "correct": 0}
    )
    per_domain: dict[str, dict[str, int]] = defaultdict(
        lambda: {"total": 0, "correct": 0}
    )
    per_task_type: dict[str, dict[str, int]] = defaultdict(
        lambda: {"total": 0, "correct": 0}
    )
    per_sample: list[dict[str, Any]] = []

    for sample, result in zip(samples, results):
        record = {
            "sample_id": sample.sample_id,
            "video_path": sample.video_path,
            "url": sample.url,
            "video_id": sample.video_id,
            "question_id": sample.question_id,
            "duration": sample.duration,
            "domain": sample.domain,
            "sub_category": sample.sub_category,
            "task_type": sample.task_type,
            "expected": sample.answer,
            "latency_s": round(result.latency_s, 4),
            "prompt_tokens": result.prompt_tokens,
            "completion_tokens": result.completion_tokens,
            "tok_per_s": (round(result.tok_per_s, 1) if result.tok_per_s > 0 else None),
            "audio_duration_s": (
                round(result.audio_duration_s, 4)
                if result.audio_duration_s > 0
                else None
            ),
            "rtf": (round(result.rtf, 4) if result.rtf > 0 else None),
            "wav_path": result.wav_path or "",
        }

        per_duration[sample.duration]["total"] += 1
        per_domain[sample.domain]["total"] += 1
        per_task_type[sample.task_type]["total"] += 1

        if not result.is_success:
            record.update(
                predicted="",
                raw_response=result.error,
                is_correct=False,
                is_success=False,
                error=result.error,
            )
            failed += 1
            per_sample.append(record)
            continue

        predicted, is_fallback = parse_multi_choice_response(
            result.text,
            sample.all_choices,
            sample.index2ans,
        )
        is_correct = predicted == sample.answer
        if is_fallback:
            mc_fallback += 1
            logger.debug("Video-MME parse fallback for sample %s", sample.sample_id)
        if is_correct:
            correct += 1
            per_duration[sample.duration]["correct"] += 1
            per_domain[sample.domain]["correct"] += 1
            per_task_type[sample.task_type]["correct"] += 1

        record.update(
            predicted=predicted,
            raw_response=result.text,
            is_correct=is_correct,
            is_success=True,
            error="",
        )
        per_sample.append(record)

    total = len(samples)
    summary = {
        "total_samples": total,
        "correct": correct,
        "accuracy": round(correct / total, 4) if total > 0 else 0.0,
        "failed": failed,
        "mc_fallback": mc_fallback,
        "per_duration": _finalize_breakdown(per_duration),
        "per_domain": _finalize_breakdown(per_domain),
        "per_task_type": _finalize_breakdown(per_task_type),
    }
    return summary, per_sample


def print_videomme_accuracy_summary(metrics: dict[str, Any], model_name: str) -> None:
    lw = SUMMARY_LABEL_WIDTH
    print(f"\n{'=' * SUMMARY_LINE_WIDTH}")
    print(f"  Video-MME Accuracy — {model_name}")
    print(f"{'=' * SUMMARY_LINE_WIDTH}")
    print(f"  {'Total samples:':<{lw}} {metrics['total_samples']}")
    print(f"  {'Correct:':<{lw}} {metrics['correct']}")
    print(
        f"  {'Accuracy:':<{lw}} {metrics['accuracy']:.4f} "
        f"({metrics['accuracy'] * 100:.1f}%)"
    )
    print(f"  {'Failed requests:':<{lw}} {metrics['failed']}")
    print(f"  {'MC parse fallback:':<{lw}} {metrics['mc_fallback']}")
    print_accuracy_breakdown("By duration", metrics.get("per_duration", {}))
    print_accuracy_breakdown("By domain", metrics.get("per_domain", {}))
    print_accuracy_breakdown("By task type", metrics.get("per_task_type", {}))
    print(f"{'=' * SUMMARY_LINE_WIDTH}\n")
