# SPDX-License-Identifier: Apache-2.0
"""MMMU benchmark case: answer parsing and request execution."""

from __future__ import annotations

import asyncio
import base64
import logging
import os
import random
import re
import time
from typing import TypedDict

import aiohttp

from benchmarks.benchmarker.data import RequestResult
from benchmarks.benchmarker.runner import SendFn
from benchmarks.dataset.mmmu import MMMUSample, image_to_data_uri

logger = logging.getLogger(__name__)

MULTI_CHOICE_INSTRUCTION = (
    "\nAnswer the following multiple-choice question. "
    "The last line of your response should be of the "
    "following format: 'Answer: $LETTER' (without quotes) "
    "where LETTER is one of the options. "
    "Think step by step before answering."
)


class MMMURecord(TypedDict):
    sample_id: str
    subject: str
    question_type: str
    expected: str
    latency_s: float
    prompt_tokens: int
    completion_tokens: int
    tok_per_s: float | None
    predicted: str
    raw_response: str
    is_correct: bool
    is_success: bool
    is_mc_fallback: bool
    error: str


def _check_is_number(s: str) -> bool:
    try:
        float(s.replace(",", ""))
        return True
    except ValueError:
        return False


def _normalize_str(s: str) -> list[float | str]:
    """Normalize a string for open-ended answer comparison."""
    s = s.strip()
    if _check_is_number(s):
        return [round(float(s.replace(",", "")), 2)]
    return [s.lower()] if len(s) > 1 else [" " + s, s + " "]


def _extract_numbers(s: str) -> list[str]:
    """Extract all numbers (with commas, scientific notation, etc.) from *s*."""
    pattern_commas = r"-?\b\d{1,3}(?:,\d{3})+\b"
    pattern_scientific = r"-?\d+(?:\.\d+)?[eE][+-]?\d+"
    pattern_simple = r"-?(?:\d+\.\d+|\.\d+|\d+\b)(?![eE][+-]?\d+)(?![,\d])"
    return (
        re.findall(pattern_commas, s)
        + re.findall(pattern_scientific, s)
        + re.findall(pattern_simple, s)
    )


def _parse_open_answer_tag(response: str) -> str | None:
    """Try to extract the answer from an explicit 'Answer: ...' line.

    Supports formats like ``Answer: 42``, ``Answer: MgS``,
    ``Answer: \\boxed{13.0}``.  Returns ``None`` when no match is found.
    """
    matches = re.findall(
        r"[Aa]nswer\s*:\s*\*?\*?\s*(.+)",
        response,
    )
    if not matches:
        return None
    raw = matches[-1].strip().rstrip(".")
    # Unwrap \boxed{...} if present
    boxed = re.search(r"\\boxed\{(.+?)\}", raw)
    if boxed:
        raw = boxed.group(1)
    # Strip surrounding ** (bold markdown)
    raw = raw.strip("*").strip()
    return raw if raw else None


def parse_open_response(response: str) -> list[float | str]:
    """Extract answer candidates from an open-ended model response.

    First tries to extract from an explicit ``Answer: ...`` line.
    Falls back to heuristic key-subresponse extraction.
    """
    # Fast path: explicit "Answer: ..."
    tag_answer = _parse_open_answer_tag(response)
    if tag_answer is not None:
        out: list = []
        out.extend(_normalize_str(tag_answer))
        for num in _extract_numbers(tag_answer):
            out.extend(_normalize_str(num))
        return list(dict.fromkeys(out))

    # Fallback: heuristic extraction
    def _get_key_subresponses(resp: str) -> list[str]:
        resp = resp.strip().strip(".").lower()
        subs = re.split(r"\.\s(?=[A-Z])|\n", resp)
        indicators = [
            "could be ",
            "so ",
            "is ",
            "thus ",
            "therefore ",
            "final ",
            "answer ",
            "result ",
        ]
        keys: list[str] = []
        for i, s in enumerate(subs):
            cands = indicators + ["="] if i == len(subs) - 1 else indicators
            shortest = None
            for ind in cands:
                if ind in s:
                    part = s.split(ind)[-1].strip()
                    if not shortest or len(part) < len(shortest):
                        shortest = part
            if shortest and shortest not in (":", ",", ".", "!", "?", ";", "'"):
                keys.append(shortest)
        return keys or [resp]

    key_resps = _get_key_subresponses(response)
    pred_list = key_resps.copy()
    for r in key_resps:
        pred_list.extend(_extract_numbers(r))
    out = []
    for x in pred_list:
        out.extend(_normalize_str(x))
    return list(dict.fromkeys(out))


def eval_open(gold: str | list[str], preds: list[float | str]) -> bool:
    """Check if any prediction matches the gold answer (fuzzy)."""
    if isinstance(gold, list):
        norm_answers: list = []
        for ans in gold:
            norm_answers.extend(_normalize_str(ans))
    else:
        norm_answers = _normalize_str(gold)
    for p in preds:
        if isinstance(p, str):
            for na in norm_answers:
                if isinstance(na, str) and na in p:
                    return True
        else:
            if p in norm_answers:
                return True
    return False


def parse_multi_choice_response(
    response: str,
    all_choices: list[str],
    index2ans: dict[str, str],
) -> tuple[str, bool]:
    """Extract a single answer letter from the model response.

    Priority: ``Answer: X`` → ``(A)`` bracket → ``·A·`` space-padded →
    option-text match → last-occurrence tie-break → random fallback.

    Returns ``(choice, is_fallback)``. ``is_fallback`` is ``True`` iff
    nothing could be parsed out of *response* and a random choice was
    returned — this counter is observational and doesn't change scoring
    behavior vs. the MMMU reference eval.
    """
    answer_matches = re.findall(r"[Aa]nswer\s*:\s*\*?\*?\s*\(?([A-Z])\)?", response)
    if answer_matches:
        candidate = answer_matches[-1]
        if candidate in all_choices:
            return candidate, False

    for char in (",", ".", "!", "?", ";", ":", "'"):
        response = response.strip(char)
    response = " " + response + " "

    candidates: list[str] = []
    for choice in all_choices:
        if f"({choice})" in response:
            candidates.append(choice)
    if not candidates:
        for choice in all_choices:
            if f" {choice} " in response:
                candidates.append(choice)
    if not candidates and len(response.split()) > 5:
        for idx, ans in index2ans.items():
            if ans and ans.lower() in response.lower():
                candidates.append(idx)
    if not candidates:
        return random.choice(all_choices), True
    if len(candidates) == 1:
        return candidates[0], False

    starts: list[int] = []
    for can in candidates:
        pos = response.rfind(f"({can})")
        if pos == -1:
            pos = response.rfind(f" {can} ")
        if pos == -1 and index2ans.get(can):
            pos = response.lower().rfind(index2ans[can].lower())
        starts.append(pos)
    return candidates[max(range(len(candidates)), key=starts.__getitem__)], False


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


def build_mmmu_result_records(
    samples: list[MMMUSample],
    results: list[RequestResult],
) -> list[MMMURecord]:
    """Parse responses into persisted per-sample records."""
    assert len(samples) == len(
        results
    ), f"Sample/result count mismatch: {len(samples)} samples vs {len(results)} results"
    # Fix the random seed so MC fallback choices stay deterministic across CI runs.
    random.seed(42)

    per_sample: list[MMMURecord] = []

    for sample, result in zip(samples, results):
        record: MMMURecord = {
            "sample_id": sample.sample_id,
            "subject": sample.subject,
            "question_type": sample.question_type,
            "expected": sample.answer,
            "latency_s": round(result.latency_s, 4),
            "prompt_tokens": result.prompt_tokens,
            "completion_tokens": result.completion_tokens,
            "tok_per_s": (round(result.tok_per_s, 1) if result.tok_per_s > 0 else None),
            "predicted": "",
            "raw_response": result.error,
            "is_correct": False,
            "is_success": False,
            "is_mc_fallback": False,
            "error": result.error,
        }

        if result.is_success:
            gold = sample.answer
            is_fallback = False
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
                    logger.debug(
                        f"MMMU multi-choice parse fallback for sample "
                        f"{sample.sample_id}"
                    )
                is_correct = gold is not None and predicted == gold
            else:
                parsed_list = parse_open_response(result.text)
                is_correct = gold is not None and eval_open(gold, parsed_list)
                predicted = ", ".join(map(str, parsed_list))

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
