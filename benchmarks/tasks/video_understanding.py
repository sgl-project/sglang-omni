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
from typing import Any, TypedDict

import aiohttp

from benchmarks.benchmarker.data import RequestResult
from benchmarks.benchmarker.runner import SendFn
from benchmarks.benchmarker.utils import get_wav_duration
from benchmarks.dataset.videomme import VideoAMMESample, VideoMMESample
from benchmarks.tasks.visual_understand import parse_multi_choice_response

logger = logging.getLogger(__name__)

VIDEOAMME_REQUEST_TEXT = (
    "Use the video and the audio question to answer. "
    "Return the final answer as Answer: $LETTER."
)


class VideoMMERecord(TypedDict):
    sample_id: str
    video_path: str
    url: str
    video_id: str
    question_id: str
    duration: str
    domain: str
    sub_category: str
    task_type: str
    expected: str
    latency_s: float
    prompt_tokens: int
    completion_tokens: int
    tok_per_s: float | None
    audio_duration_s: float | None
    rtf: float | None
    wav_path: str
    predicted: str
    raw_response: str
    is_correct: bool
    is_success: bool
    is_mc_fallback: bool
    error: str


def _apply_chat_completion_response(
    result: RequestResult,
    body: dict[str, Any],
    *,
    audio_output_dir: str | None,
    sample_id: str,
) -> bool:
    message = body.get("choices", [{}])[0].get("message", {})
    result.text = message.get("content", "") or ""
    wav_bytes = b""

    if audio_output_dir:
        audio_obj = message.get("audio")
        if not isinstance(audio_obj, dict):
            result.error = "No audio in response"
            return False
        audio_b64 = audio_obj.get("data", "")
        if not audio_b64:
            result.error = "Empty audio data in response"
            return False
        try:
            wav_bytes = base64.b64decode(audio_b64, validate=True)
            result.audio_duration_s = round(get_wav_duration(wav_bytes), 4)
        except (binascii.Error, ValueError, struct.error) as exc:
            result.error = f"Invalid audio data: {exc}"
            return False

    usage = body.get("usage", {})
    if usage:
        result.prompt_tokens = usage.get("prompt_tokens", 0)
        result.completion_tokens = usage.get("completion_tokens", 0)

    if audio_output_dir and result.audio_duration_s > 0:
        try:
            os.makedirs(audio_output_dir, exist_ok=True)
            wav_path = os.path.join(audio_output_dir, f"{sample_id}.wav")
            with open(wav_path, "wb") as f:
                f.write(wav_bytes)
        except OSError as exc:
            result.error = f"Failed to save audio: {exc}"
            return False
        result.wav_path = wav_path

    result.is_success = True
    return True


def make_video_send_fn(
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
    enable_audio_input: bool = False,
    audio_output_dir: str | None = None,
    fixed_prompt: str | None = None,
) -> SendFn:
    modalities = ["text", "audio"] if audio_output_dir else ["text"]

    async def send_fn(
        session: aiohttp.ClientSession,
        sample: VideoMMESample | VideoAMMESample,
    ) -> RequestResult:
        prompt = fixed_prompt or sample.prompt
        result = RequestResult(
            request_id=sample.sample_id,
            text=prompt[:60],
        )

        payload: dict[str, Any] = {
            "model": model_name,
            "messages": [{"role": "user", "content": prompt}],
            "videos": [sample.video_path],
            "modalities": modalities,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False,
        }
        if enable_audio_input:
            assert isinstance(sample, VideoAMMESample)
            payload["audios"] = [sample.audio_path]
        if audio_output_dir:
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

            if not _apply_chat_completion_response(
                result,
                body,
                audio_output_dir=audio_output_dir,
                sample_id=sample.sample_id,
            ):
                return result

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


def build_videomme_result_records(
    samples: list[VideoMMESample],
    results: list[RequestResult],
) -> list[VideoMMERecord]:
    """Parse responses into persisted per-sample records."""
    assert len(samples) == len(
        results
    ), f"Sample/result count mismatch: {len(samples)} samples vs {len(results)} results"
    random.seed(42)

    per_sample: list[VideoMMERecord] = []

    for sample, result in zip(samples, results):
        record: VideoMMERecord = {
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
            "predicted": "",
            "raw_response": result.error,
            "is_correct": False,
            "is_success": False,
            "is_mc_fallback": False,
            "error": result.error,
        }

        if not result.is_success:
            per_sample.append(record)
            continue

        predicted, is_fallback = parse_multi_choice_response(
            result.text,
            sample.all_choices,
            sample.index2ans,
        )
        is_correct = predicted == sample.answer
        if is_fallback:
            logger.debug("Video-MME parse fallback for sample %s", sample.sample_id)

        record.update(
            predicted=predicted,
            raw_response=result.text,
            is_correct=is_correct,
            is_success=True,
            is_mc_fallback=is_fallback,
            error="",
        )
        per_sample.append(record)

    return per_sample
