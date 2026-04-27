# SPDX-License-Identifier: Apache-2.0
"""MMSU task runner and evaluation helpers."""

from __future__ import annotations

import asyncio
import base64
import csv
import json
import os
import time
from dataclasses import asdict, dataclass
from typing import Any

import aiohttp

from benchmarks.benchmarker.data import RequestResult
from benchmarks.benchmarker.runner import SendFn
from benchmarks.benchmarker.utils import get_wav_duration, save_json_results
from benchmarks.dataset.mmsu import MmsuSample, normalize_text
from benchmarks.metrics.accuracy import INDEX_TO_LETTER, extract_answer_letter

DEFAULT_PROMPT = (
    "Listen to the audio and answer the multiple-choice question. "
    "Reply with only A, B, C, or D."
)


def _extract_message_text(message: dict[str, Any]) -> str:
    """Extract text from a chat completion response message."""
    content = message.get("content")
    if isinstance(content, str) and content.strip():
        return content.strip()
    # Fall back to audio transcript when content is empty
    audio_obj = message.get("audio", {})
    transcript = audio_obj.get("transcript", "") if isinstance(audio_obj, dict) else ""
    return transcript.strip()


def _build_question_text(sample: MmsuSample, prompt: str) -> str:
    lines = [f"{prompt}\n", f"Question: {sample.question}"]
    for i, choice in enumerate(sample.choices):
        lines.append(f"{chr(ord('A') + i)}. {choice}")
    return "\n".join(lines)


def _extract_prediction(
    raw_response: str,
    choices: list[str],
) -> tuple[int | None, str]:
    predicted_index = extract_answer_letter(raw_response)
    if predicted_index is not None and predicted_index < len(choices):
        return predicted_index, choices[predicted_index]

    normalized_response = normalize_text(raw_response)
    for index, choice in enumerate(choices):
        normalized_choice = normalize_text(choice)
        if not normalized_choice:
            continue
        if normalized_response == normalized_choice:
            return index, choice

    return None, ""


def _build_request_payload(
    sample: MmsuSample,
    *,
    model_name: str,
    prompt: str,
    modalities: list[str],
    max_tokens: int,
    temperature: float,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model_name,
        "messages": [
            {
                "role": "user",
                "content": _build_question_text(sample, prompt),
            }
        ],
        "audios": [sample.audio_path],
        "modalities": modalities,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False,
    }
    if "audio" in modalities:
        payload["audio"] = {"format": "wav"}
    return payload


def _save_response_audio(
    wav_bytes: bytes,
    sample_id: str,
    save_audio_dir: str | None,
) -> str:
    if not save_audio_dir or len(wav_bytes) <= 44:
        return ""

    wav_path = os.path.join(save_audio_dir, f"{sample_id}.wav")
    with open(wav_path, "wb") as file_obj:
        file_obj.write(wav_bytes)
    return wav_path


def _build_result_from_response(
    result: RequestResult,
    response_json: dict[str, Any],
    *,
    audio_mode: bool,
    sample_id: str,
    save_audio_dir: str | None,
) -> RequestResult:
    message = response_json.get("choices", [{}])[0].get("message", {})
    result.text = _extract_message_text(message)

    usage = response_json.get("usage", {})
    result.prompt_tokens = usage.get("prompt_tokens", 0)
    result.completion_tokens = usage.get("completion_tokens", 0)

    if not audio_mode:
        if result.text:
            result.is_success = True
        else:
            result.error = "Empty response"
        return result

    audio_obj = message.get("audio")
    if isinstance(audio_obj, dict) and audio_obj.get("data"):
        wav_bytes = base64.b64decode(audio_obj["data"])
        result.audio_duration_s = get_wav_duration(wav_bytes)
        result.wav_path = _save_response_audio(wav_bytes, sample_id, save_audio_dir)

    if result.text or result.audio_duration_s > 0:
        result.is_success = True
    else:
        result.error = "Empty response"
    return result


@dataclass
class MmsuResult:
    sample_id: str = ""
    task_name: str = ""
    category: str = ""
    sub_category: str = ""
    sub_sub_category: str = ""
    linguistics_sub_discipline: str = ""
    correct_choice: str = ""
    correct_answer: str = ""
    predicted_choice: str = ""
    predicted_answer: str = ""
    raw_response: str = ""
    is_correct: bool = False
    is_parseable: bool = False
    is_success: bool = False
    latency_s: float = 0.0
    has_audio: bool = False
    audio_duration_s: float = 0.0
    error: str = ""


def make_mmsu_send_fn(
    model_name: str,
    api_url: str,
    *,
    modalities: list[str] | None = None,
    prompt: str = DEFAULT_PROMPT,
    max_tokens: int = 32,
    temperature: float = 0.0,
    save_audio_dir: str | None = None,
) -> SendFn:
    if modalities is None:
        modalities = ["text"]

    audio_mode = "audio" in modalities

    async def send_fn(
        session: aiohttp.ClientSession,
        sample: MmsuSample,
    ) -> RequestResult:
        result = RequestResult(request_id=sample.sample_id)
        payload = _build_request_payload(
            sample,
            model_name=model_name,
            prompt=prompt,
            modalities=modalities,
            max_tokens=max_tokens,
            temperature=temperature,
        )

        start_time = time.perf_counter()
        try:
            async with session.post(
                api_url,
                json=payload,
                headers={"Content-Type": "application/json"},
            ) as response:
                response.raise_for_status()
                response_json = await response.json()

            result = _build_result_from_response(
                result,
                response_json,
                audio_mode=audio_mode,
                sample_id=sample.sample_id,
                save_audio_dir=save_audio_dir,
            )

            elapsed = time.perf_counter() - start_time
            result.engine_time_s = elapsed
            if result.audio_duration_s > 0:
                result.rtf = elapsed / result.audio_duration_s
            if result.completion_tokens > 0 and elapsed > 0:
                result.tok_per_s = result.completion_tokens / elapsed
        except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
            result.error = str(exc)
        finally:
            result.latency_s = time.perf_counter() - start_time

        return result

    return send_fn


def build_mmsu_results(
    request_results: list[RequestResult],
    samples: list[MmsuSample],
    modalities: list[str] | None = None,
) -> list[MmsuResult]:
    if modalities is None:
        modalities = ["text"]

    audio_mode = "audio" in modalities
    sample_map = {sample.sample_id: sample for sample in samples}
    results: list[MmsuResult] = []

    for request_result in request_results:
        sample = sample_map.get(request_result.request_id)
        if sample is None:
            continue

        predicted_index, predicted_answer = _extract_prediction(
            request_result.text,
            sample.choices,
        )
        correct_index = sample.answer_index

        index_match = predicted_index is not None and correct_index == predicted_index
        text_match = bool(
            predicted_answer
            and normalize_text(predicted_answer) == normalize_text(sample.answer_text)
        )

        result = MmsuResult(
            sample_id=sample.sample_id,
            task_name=sample.task_name,
            category=sample.category,
            sub_category=sample.sub_category,
            sub_sub_category=sample.sub_sub_category,
            linguistics_sub_discipline=sample.linguistics_sub_discipline,
            correct_choice=INDEX_TO_LETTER.get(correct_index, ""),
            correct_answer=sample.answer_text,
            predicted_choice=INDEX_TO_LETTER.get(predicted_index, ""),
            predicted_answer=predicted_answer,
            raw_response=request_result.text,
            is_correct=index_match or text_match,
            is_parseable=predicted_index is not None or bool(predicted_answer),
            is_success=bool(request_result.is_success),
            latency_s=request_result.latency_s,
            error=request_result.error,
        )

        if audio_mode:
            result.has_audio = request_result.audio_duration_s > 0
            result.audio_duration_s = request_result.audio_duration_s

        results.append(result)

    return results


def save_mmsu_results(
    results: list[MmsuResult],
    metrics: dict[str, Any],
    config: dict[str, Any],
    output_dir: str,
    *,
    speed_metrics: dict[str, Any] | None = None,
    wer_metrics: dict[str, Any] | None = None,
) -> None:
    summary_output = {
        "summary": metrics,
        "config": config,
        "per_sample": [asdict(result) for result in results],
    }
    if speed_metrics:
        summary_output["speed_metrics"] = speed_metrics
    if wer_metrics:
        summary_output["wer"] = wer_metrics

    save_json_results(summary_output, output_dir, "mmsu_results.json")

    os.makedirs(output_dir, exist_ok=True)
    jsonl_path = os.path.join(output_dir, "mmsu_predictions.jsonl")
    with open(jsonl_path, "w") as file_obj:
        for result in results:
            file_obj.write(json.dumps(asdict(result)) + "\n")

    csv_path = os.path.join(output_dir, "mmsu_results.csv")
    if results:
        fieldnames = list(asdict(results[0]).keys())
        with open(csv_path, "w", newline="") as file_obj:
            writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
            writer.writeheader()
            for result in results:
                writer.writerow(asdict(result))
