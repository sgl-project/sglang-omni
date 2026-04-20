# SPDX-License-Identifier: Apache-2.0
"""SocialOmni benchmark task helpers.

Single-file layout aligned with existing benchmark modules. Sections below cover:
- common request helpers
- level1 prompt / parsing / scoring
- judge orchestration helpers
- level2 workflow helpers
- level2 metrics and endpoint preflight
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import aiohttp
import numpy as np

from benchmarks.benchmarker.data import RequestResult
from benchmarks.benchmarker.runner import SendFn
from benchmarks.dataset.socialomni import SocialOmniLevel1Sample, SocialOmniLevel2Sample
from benchmarks.metrics.performance import compute_speed_metrics

logger = logging.getLogger(__name__)

DEFAULT_SYSTEM_PROMPT = "You are a careful multimodal reasoning assistant."
LEVEL1_INSTRUCTION = (
    "Answer ONLY with the option letter (A, B, C, or D). Do not include any other text."
)
LEVEL2_Q1_INSTRUCTION = "Answer ONLY with the option letter (A or B)."
LEVEL2_Q2_INSTRUCTION = (
    "Provide only the natural interruption utterance with no explanation."
)
LEVEL1_USER_PROMPT = "Watch the video and answer the multiple-choice question about what happens in the relevant segment."
LEVEL2_Q1_USER_PROMPT = "Decide whether the named speaker should interrupt immediately after the timestamped prefix."
LEVEL2_Q2_USER_PROMPT = (
    "Generate the natural interruption utterance for that exact moment."
)
JUDGE_BUCKETS = (0, 25, 50, 75, 100)


@dataclass(frozen=True)
class JudgeSpec:
    base_url: str
    api_url: str
    model: str


@dataclass(frozen=True)
class JudgeOutcome:
    judge: JudgeSpec
    raw_response: str
    score: int | None
    request_result: RequestResult
    error: str = ""


def build_base_url(
    *,
    base_url: str | None = None,
    host: str = "localhost",
    port: int = 8000,
) -> str:
    """Return a normalized base URL for an OpenAI-compatible endpoint."""
    return (base_url or f"http://{host}:{port}").rstrip("/")


def build_chat_api_url(base_url: str) -> str:
    """Normalize a base URL to a /v1/chat/completions endpoint."""
    normalized = base_url.rstrip("/")
    if normalized.endswith("/chat/completions"):
        return normalized
    if normalized.endswith("/v1"):
        return f"{normalized}/chat/completions"
    return f"{normalized}/v1/chat/completions"


def maybe_openai_headers() -> dict[str, str]:
    """Return optional auth headers when OPENAI_API_KEY is available."""
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        return {"Content-Type": "application/json"}
    return {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }


def strip_option_prefix(option: str) -> str:
    """Strip a leading option prefix like 'A. ' from an option string."""
    return re.sub(r"^[A-D]\.\s*", "", option.strip(), flags=re.IGNORECASE)


def join_prompt_sections(*sections: str) -> str:
    """Join non-empty prompt sections with blank lines."""
    return "\n\n".join(
        section.strip() for section in sections if section and section.strip()
    )


def normalize_multiple_choice_prediction(
    text: str,
    *,
    letters: tuple[str, ...],
) -> str:
    """Extract the first usable option letter from a model response."""
    content = (text or "").strip()
    if not content:
        return ""
    upper = content.upper()
    tagged = re.search(
        r"(?:ANSWER|CHOICE)\s*(?:IS|:)\s*([%s])\b" % "".join(letters),
        upper,
    )
    if tagged:
        return tagged.group(1)
    if upper[0] in letters:
        return upper[0]
    match = re.search(r"\b([%s])\b" % "".join(letters), upper)
    if match:
        return match.group(1)
    return ""


def normalize_level2_yes_no_prediction(text: str) -> str:
    """Mirror the reference Level2 Q1 normalization semantics."""
    parsed = normalize_multiple_choice_prediction(text, letters=("A", "B"))
    if parsed:
        return parsed
    lowered = (text or "").lower()
    if "yes" in lowered:
        return "A"
    if "no" in lowered:
        return "B"
    return ""


def extract_message_text(message: dict[str, Any]) -> str:
    """Extract plain text content from a chat completion message."""
    content = message.get("content")
    if isinstance(content, str) and content.strip():
        return content.strip()
    if isinstance(content, list):
        text_parts = [
            part.get("text", "")
            for part in content
            if isinstance(part, dict) and part.get("type") == "text"
        ]
        merged = "\n".join(part for part in text_parts if part.strip()).strip()
        if merged:
            return merged
    return ""


async def send_chat_completion(
    session: aiohttp.ClientSession,
    *,
    api_url: str,
    payload: dict[str, Any],
    request_id: str,
    headers: dict[str, str] | None = None,
) -> RequestResult:
    """Send an OpenAI-compatible chat completion request."""
    result = RequestResult(request_id=request_id)
    start_time = time.perf_counter()
    try:
        async with session.post(
            api_url,
            json=payload,
            headers=headers or maybe_openai_headers(),
        ) as response:
            response.raise_for_status()
            body = await response.json()

        message = body.get("choices", [{}])[0].get("message", {})
        result.text = extract_message_text(message)
        usage = body.get("usage", {})
        result.prompt_tokens = usage.get("prompt_tokens", 0)
        result.completion_tokens = usage.get("completion_tokens", 0)
        result.is_success = bool(result.text)
        if not result.is_success:
            result.error = "Empty response"
    except (aiohttp.ClientError, asyncio.TimeoutError, ValueError) as exc:
        result.error = str(exc)
    finally:
        elapsed = time.perf_counter() - start_time
        result.latency_s = elapsed
        result.engine_time_s = elapsed
        if result.completion_tokens > 0 and elapsed > 0:
            result.tok_per_s = result.completion_tokens / elapsed
    return result


async def preflight_chat_completion_endpoint(
    session: aiohttp.ClientSession,
    *,
    api_url: str,
    model_name: str,
    endpoint_name: str,
) -> None:
    """Fail fast if an OpenAI-compatible endpoint is unreachable or unusable."""
    payload: dict[str, Any] = {
        "model": model_name,
        "messages": [{"role": "user", "content": "Reply with OK."}],
        "modalities": ["text"],
        "max_tokens": 4,
        "temperature": 0.0,
        "stream": False,
    }
    result = await send_chat_completion(
        session,
        api_url=api_url,
        payload=payload,
        request_id=f"preflight:{endpoint_name}",
    )
    if not result.is_success:
        raise RuntimeError(
            f"{endpoint_name} preflight failed for {api_url}: {result.error or 'empty response'}"
        )


def build_socialomni_level1_prompt(sample: SocialOmniLevel1Sample) -> str:
    """Build a level1 prompt from the dataset sample."""
    options_block = "\n".join(
        f"{chr(ord('A') + index)}. {strip_option_prefix(option)}"
        for index, option in enumerate(sample.options)
    )
    asr_block = f"ASR transcript:\n{sample.asr_content}" if sample.asr_content else ""
    question_block = f"Question: {sample.question}\n\nOptions:\n{options_block}"
    return join_prompt_sections(
        DEFAULT_SYSTEM_PROMPT,
        LEVEL1_USER_PROMPT,
        asr_block,
        question_block,
        LEVEL1_INSTRUCTION,
    )


def extract_level1_prediction(raw_response: str) -> str:
    """Extract the Level1 choice letter from raw model output."""
    return normalize_multiple_choice_prediction(
        raw_response, letters=("A", "B", "C", "D")
    )


def make_socialomni_level1_send_fn(
    model_name: str,
    api_url: str,
    *,
    max_tokens: int = 32,
    temperature: float = 0.0,
) -> SendFn:
    """Return a send_fn for SocialOmni level1 over /v1/chat/completions."""

    async def send_fn(session, sample: SocialOmniLevel1Sample):
        payload: dict[str, Any] = {
            "model": model_name,
            "messages": [
                {"role": "user", "content": build_socialomni_level1_prompt(sample)}
            ],
            "videos": [sample.video_path],
            "modalities": ["text"],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False,
        }
        return await send_chat_completion(
            session,
            api_url=api_url,
            payload=payload,
            request_id=sample.sample_id,
        )

    return send_fn


def compute_socialomni_level1_results(
    samples: list[SocialOmniLevel1Sample],
    request_results: list[RequestResult],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Build summary + per-sample rows for SocialOmni level1."""
    if len(samples) != len(request_results):
        raise ValueError(
            "samples and request_results must have the same length "
            f"({len(samples)} != {len(request_results)})"
        )

    per_sample: list[dict[str, Any]] = []
    correct = 0
    parsed = 0
    failed = 0
    by_consistency: dict[str, dict[str, int]] = {}

    for sample, result in zip(samples, request_results, strict=True):
        prediction = extract_level1_prediction(result.text) if result.is_success else ""
        is_correct = bool(prediction and prediction == sample.correct_answer)
        if prediction:
            parsed += 1
        if is_correct:
            correct += 1
        if not result.is_success:
            failed += 1

        consistency = (
            str(sample.metadata.get("consistency", "unknown")).strip() or "unknown"
        )
        bucket = by_consistency.setdefault(consistency, {"total": 0, "correct": 0})
        bucket["total"] += 1
        bucket["correct"] += int(is_correct)

        per_sample.append(
            {
                "sample_id": sample.sample_id,
                "video_path": sample.video_path,
                "prediction": prediction,
                "correct_answer": sample.correct_answer,
                "is_correct": is_correct,
                "raw_response": result.text,
                "latency_s": round(result.latency_s, 3),
                "error": result.error,
                "metadata": sample.metadata,
            }
        )

    total = len(samples)
    summary = {
        "total_samples": total,
        "parsed_predictions": parsed,
        "correct": correct,
        "failed_requests": failed,
        "accuracy": round(correct / total, 4) if total else 0.0,
        "accuracy_by_consistency": {
            key: {
                "total": value["total"],
                "correct": value["correct"],
                "accuracy": round(value["correct"] / value["total"], 4),
            }
            for key, value in sorted(by_consistency.items())
            if value["total"] > 0
        },
    }
    return summary, per_sample


def print_socialomni_level1_summary(summary: dict[str, Any]) -> None:
    """Print a compact level1 summary."""
    logger.info(
        "SocialOmni level1 accuracy: %.2f%% (%s/%s)",
        summary.get("accuracy", 0.0) * 100.0,
        summary.get("correct", 0),
        summary.get("total_samples", 0),
    )


def validate_judge_specs(
    judges: int,
    judge_base_urls: list[str] | None,
    judge_models: list[str] | None,
) -> list[JudgeSpec]:
    """Validate and normalize repeated judge CLI args."""
    base_urls = judge_base_urls or []
    models = judge_models or []
    if len(base_urls) != len(models):
        raise ValueError(
            "Judge configuration is incomplete: provide the same number of "
            "--judge-base-url and --judge-model arguments."
        )
    if len(base_urls) != judges:
        raise ValueError(
            f"--judges {judges} requires exactly {judges} judge endpoint/model pairs; "
            f"received {len(base_urls)}."
        )

    return [
        JudgeSpec(
            base_url=base_url.rstrip("/"),
            api_url=build_chat_api_url(base_url.rstrip("/")),
            model=model,
        )
        for base_url, model in zip(base_urls, models, strict=True)
    ]


def parse_judge_score(text: str) -> int:
    """Parse a judge score and snap it to {0,25,50,75,100}."""
    match = re.search(r"-?\d+(?:\.\d+)?", str(text or ""))
    if not match:
        return 0
    score = max(0.0, min(100.0, float(match.group(0))))
    return min(JUDGE_BUCKETS, key=lambda bucket: abs(bucket - score))


def aggregate_judge_scores(scores: list[int]) -> float:
    """Aggregate multiple judge scores as an arithmetic mean."""
    if not scores:
        return 0.0
    return round(sum(scores) / len(scores), 2)


async def run_socialomni_judge(
    session: aiohttp.ClientSession,
    *,
    judge: JudgeSpec,
    sample_id: str,
    video_path: str,
    reference_answer: str,
    candidate_answer: str,
) -> JudgeOutcome:
    """Execute a single OpenAI-compatible Q2 judge request."""
    if not candidate_answer.strip():
        empty_result = RequestResult(request_id=f"{sample_id}:{judge.model}")
        empty_result.error = "Empty candidate answer"
        return JudgeOutcome(
            judge=judge,
            raw_response="",
            score=0,
            request_result=empty_result,
            error=empty_result.error,
        )

    payload: dict[str, Any] = {
        "model": judge.model,
        "messages": [
            {
                "role": "user",
                "content": (
                    "You are a strict evaluator for dialog continuation.\n"
                    "Score the candidate interruption against the reference on the "
                    "exact scale {0, 25, 50, 75, 100}.\n"
                    "Consider semantic match, intent correctness, and key information completeness.\n"
                    "Output ONLY one number.\n\n"
                    f"[Reference]\n{reference_answer}\n\n"
                    f"[Candidate]\n{candidate_answer}\n"
                ),
            }
        ],
        "videos": [video_path],
        "modalities": ["text"],
        "max_tokens": 8,
        "temperature": 0.0,
        "stream": False,
    }
    request_result = await send_chat_completion(
        session,
        api_url=judge.api_url,
        payload=payload,
        request_id=f"{sample_id}:{judge.model}",
    )
    if not request_result.is_success:
        return JudgeOutcome(
            judge=judge,
            raw_response=request_result.text,
            score=None,
            request_result=request_result,
            error=request_result.error or "Judge request failed",
        )

    return JudgeOutcome(
        judge=judge,
        raw_response=request_result.text,
        score=parse_judge_score(request_result.text),
        request_result=request_result,
    )


def ensure_ffmpeg_available() -> None:
    """Fail fast when ffmpeg is unavailable for prefix cutting."""
    if shutil.which("ffmpeg") is None:
        raise RuntimeError(
            "SocialOmni level2 requires ffmpeg to cut the video prefix, but ffmpeg was not found in PATH."
        )


def parse_level2_timestamp_to_seconds(timestamp: str) -> float:
    """Mirror the reference SocialOmni timestamp parsing semantics."""
    text = str(timestamp or "").strip()
    if not text:
        return 0.0
    if text.isdigit():
        return float(text)
    if ":" in text:
        parts = text.split(":")
        if len(parts) == 3:
            try:
                return float(parts[1])
            except Exception:  # noqa: BLE001
                pass
        if len(parts) == 2:
            try:
                minutes = int(parts[0])
                seconds = float(parts[1])
                return minutes * 60 + seconds
            except Exception:  # noqa: BLE001
                pass
    return float(text)


def cut_video_prefix(input_video: str, timestamp_s: float, output_video: str) -> None:
    """Cut *input_video* from t=0 to *timestamp_s* into *output_video*."""
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        input_video,
        "-t",
        str(max(0.0, timestamp_s)),
        "-c",
        "copy",
        "-y",
        output_video,
    ]
    completed = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if completed.returncode != 0 or not Path(output_video).is_file():
        raise RuntimeError(
            f"ffmpeg failed to cut video prefix at {timestamp_s}s for {input_video}: {completed.stderr.strip()}"
        )


def build_socialomni_level2_q1_prompt(sample: SocialOmniLevel2Sample) -> str:
    """Build the Q1 prompt for a level2 sample."""
    options_block = f"A. {sample.question_1.option_a}\nB. {sample.question_1.option_b}"
    asr_block = f"ASR transcript:\n{sample.full_asr}" if sample.full_asr else ""
    question_block = (
        f"Question: {sample.question_1.question}\n\nOptions:\n{options_block}"
    )
    return join_prompt_sections(
        DEFAULT_SYSTEM_PROMPT,
        LEVEL2_Q1_USER_PROMPT,
        asr_block,
        question_block,
        LEVEL2_Q1_INSTRUCTION,
    )


def build_socialomni_level2_q2_prompt(sample: SocialOmniLevel2Sample) -> str:
    """Build the Q2 prompt for a level2 sample."""
    asr_block = f"ASR transcript:\n{sample.full_asr}" if sample.full_asr else ""
    question_block = f"Question: {sample.question_2.question}"
    return join_prompt_sections(
        DEFAULT_SYSTEM_PROMPT,
        LEVEL2_Q2_USER_PROMPT,
        asr_block,
        question_block,
        LEVEL2_Q2_INSTRUCTION,
    )


def resolve_level2_outcome(
    *,
    q1_answer: str,
    q1_prediction: str,
    judge_scores: list[int],
    judge_errors: list[str],
) -> tuple[bool, str, float | None]:
    """Apply the reference gating semantics and return (q1_correct, branch, q2_score)."""
    q1_correct = bool(q1_prediction and q1_answer and q1_prediction == q1_answer)
    if q1_answer == "A":
        if not q1_correct:
            return q1_correct, "zeroed_wrong_q1", 0.0
        if judge_errors:
            return q1_correct, "judge_failed", None
        return q1_correct, "yes_judged", aggregate_judge_scores(judge_scores)
    if q1_correct:
        return q1_correct, "no_skipped", None
    return q1_correct, "zeroed_wrong_q1", 0.0


async def _run_level2_primary_request(
    session: aiohttp.ClientSession,
    *,
    api_url: str,
    model_name: str,
    sample_id: str,
    video_path: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
) -> RequestResult:
    payload: dict[str, Any] = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "videos": [video_path],
        "modalities": ["text"],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False,
    }
    return await send_chat_completion(
        session,
        api_url=api_url,
        payload=payload,
        request_id=sample_id,
    )


async def run_socialomni_level2_sample(
    session: aiohttp.ClientSession,
    *,
    sample: SocialOmniLevel2Sample,
    api_url: str,
    model_name: str,
    judge_specs: list[JudgeSpec],
    max_tokens: int,
    temperature: float,
) -> tuple[dict[str, Any], list[RequestResult], list[RequestResult]]:
    """Run the full level2 workflow for one sample."""
    workflow_start = time.perf_counter()
    primary_requests: list[RequestResult] = []
    judge_requests: list[RequestResult] = []
    judge_scores: list[dict[str, Any]] = []
    row: dict[str, Any] = {
        "sample_id": sample.sample_id,
        "video_path": sample.video_path,
        "timestamp": sample.question_1.timestamp,
        "q1_prediction": "",
        "q1_correct": False,
        "q2_response": "",
        "judge_scores": judge_scores,
        "q2_score": None,
        "branch_status": "not_started",
        "error": "",
    }

    try:
        with tempfile.TemporaryDirectory(
            prefix=f"socialomni_{sample.sample_id}_"
        ) as tmpdir:
            cut_path = str(Path(tmpdir) / Path(sample.video_path).name)
            cut_video_prefix(
                sample.video_path,
                parse_level2_timestamp_to_seconds(sample.question_1.timestamp),
                cut_path,
            )
            q1_result = await _run_level2_primary_request(
                session,
                api_url=api_url,
                model_name=model_name,
                sample_id=f"{sample.sample_id}:q1",
                video_path=cut_path,
                prompt=build_socialomni_level2_q1_prompt(sample),
                max_tokens=max_tokens,
                temperature=temperature,
            )
            primary_requests.append(q1_result)
            row["q1_response"] = q1_result.text
            row["q1_prediction"] = normalize_level2_yes_no_prediction(q1_result.text)

            if not q1_result.is_success or not row["q1_prediction"]:
                row["branch_status"] = "q1_failed"
                row["q2_score"] = 0.0
                row["error"] = (
                    q1_result.error or "Failed to obtain a parseable Q1 answer"
                )
                return row, primary_requests, judge_requests

            if sample.question_1.correct_answer == "A" and row["q1_prediction"] == "A":
                q2_result = await _run_level2_primary_request(
                    session,
                    api_url=api_url,
                    model_name=model_name,
                    sample_id=f"{sample.sample_id}:q2",
                    video_path=cut_path,
                    prompt=build_socialomni_level2_q2_prompt(sample),
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                primary_requests.append(q2_result)
                row["q2_response"] = q2_result.text
                if not q2_result.is_success:
                    row["branch_status"] = "q2_failed"
                    row["q2_score"] = 0.0
                    row["error"] = q2_result.error or "Q2 generation failed"
                    return row, primary_requests, judge_requests

                outcomes = await asyncio.gather(
                    *[
                        run_socialomni_judge(
                            session,
                            judge=judge,
                            sample_id=sample.sample_id,
                            video_path=sample.video_path,
                            reference_answer=sample.question_2.answer,
                            candidate_answer=q2_result.text,
                        )
                        for judge in judge_specs
                    ]
                )
                judge_errors: list[str] = []
                numeric_scores: list[int] = []
                for outcome in outcomes:
                    judge_requests.append(outcome.request_result)
                    judge_scores.append(
                        {
                            "judge_model": outcome.judge.model,
                            "judge_base_url": outcome.judge.base_url,
                            "score": outcome.score,
                            "raw_response": outcome.raw_response,
                            "error": outcome.error,
                        }
                    )
                    if outcome.error:
                        judge_errors.append(f"{outcome.judge.model}: {outcome.error}")
                    elif outcome.score is not None:
                        numeric_scores.append(outcome.score)
                q1_correct, branch_status, q2_score = resolve_level2_outcome(
                    q1_answer=sample.question_1.correct_answer,
                    q1_prediction=row["q1_prediction"],
                    judge_scores=numeric_scores,
                    judge_errors=judge_errors,
                )
                row["q1_correct"] = q1_correct
                row["branch_status"] = branch_status
                row["q2_score"] = q2_score
                if judge_errors:
                    row["error"] = "; ".join(judge_errors)
            else:
                q1_correct, branch_status, q2_score = resolve_level2_outcome(
                    q1_answer=sample.question_1.correct_answer,
                    q1_prediction=row["q1_prediction"],
                    judge_scores=[],
                    judge_errors=[],
                )
                row["q1_correct"] = q1_correct
                row["branch_status"] = branch_status
                row["q2_score"] = q2_score
    except Exception as exc:  # noqa: BLE001
        row["branch_status"] = "workflow_failed"
        row["q2_score"] = 0.0
        row["error"] = str(exc)

    if not row["q1_correct"] and row["q1_prediction"]:
        row["q1_correct"] = row["q1_prediction"] == sample.question_1.correct_answer
    row["workflow_latency_s"] = round(time.perf_counter() - workflow_start, 3)
    row["primary_latency_s"] = round(sum(req.latency_s for req in primary_requests), 3)
    row["judge_latency_s"] = round(sum(req.latency_s for req in judge_requests), 3)
    return row, primary_requests, judge_requests


async def run_socialomni_level2_benchmark(
    samples: list[SocialOmniLevel2Sample],
    *,
    api_url: str,
    model_name: str,
    judge_specs: list[JudgeSpec],
    max_tokens: int,
    temperature: float,
    max_concurrency: int,
    timeout_s: int,
) -> tuple[list[dict[str, Any]], list[RequestResult], list[RequestResult], float]:
    """Run the level2 workflow benchmark across all samples."""
    ensure_ffmpeg_available()
    semaphore = asyncio.Semaphore(max(1, max_concurrency))
    timeout = aiohttp.ClientTimeout(total=timeout_s)
    wall_clock_start = time.perf_counter()

    async with aiohttp.ClientSession(timeout=timeout) as session:

        async def _limited(sample: SocialOmniLevel2Sample):
            async with semaphore:
                return await run_socialomni_level2_sample(
                    session,
                    sample=sample,
                    api_url=api_url,
                    model_name=model_name,
                    judge_specs=judge_specs,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )

        completed = await asyncio.gather(*[_limited(sample) for sample in samples])

    per_sample = [item[0] for item in completed]
    primary_requests = [request for _, requests, _ in completed for request in requests]
    judge_requests = [request for _, _, requests in completed for request in requests]
    return (
        per_sample,
        primary_requests,
        judge_requests,
        time.perf_counter() - wall_clock_start,
    )


def build_socialomni_level2_summary(per_sample: list[dict[str, Any]]) -> dict[str, Any]:
    """Build the level2 accuracy summary from per-sample rows."""
    q1_total = len(per_sample)
    q1_correct = sum(1 for row in per_sample if row.get("q1_correct"))
    q2_scores = [
        float(row["q2_score"])
        for row in per_sample
        if isinstance(row.get("q2_score"), (int, float))
    ]
    branch_counts: dict[str, int] = {}
    failed_samples = 0
    for row in per_sample:
        branch = str(row.get("branch_status", "unknown") or "unknown")
        branch_counts[branch] = branch_counts.get(branch, 0) + 1
        if row.get("error"):
            failed_samples += 1

    return {
        "q1_total": q1_total,
        "q1_correct": q1_correct,
        "q1_accuracy": round(q1_correct / q1_total, 4) if q1_total else 0.0,
        "q2_count": len(q2_scores),
        "q2_avg_score": round(sum(q2_scores) / len(q2_scores), 2) if q2_scores else 0.0,
        "failed_samples": failed_samples,
        "branch_counts": branch_counts,
    }


def build_socialomni_level2_metrics(
    per_sample: list[dict[str, Any]],
    primary_requests: list[RequestResult],
    judge_requests: list[RequestResult],
    *,
    wall_clock_s: float,
) -> dict[str, Any]:
    """Build separated metrics for primary model, judges, and overall workflow."""
    primary_metrics = compute_speed_metrics(primary_requests)
    primary_metrics["q1_requests"] = sum(
        1 for request in primary_requests if request.request_id.endswith(":q1")
    )
    primary_metrics["q2_requests"] = sum(
        1 for request in primary_requests if request.request_id.endswith(":q2")
    )

    judge_metrics = compute_speed_metrics(judge_requests)
    judge_metrics["requested_judges"] = len(judge_requests)

    workflow_latencies = [row.get("workflow_latency_s", 0.0) for row in per_sample]
    overall_metrics: dict[str, Any] = {
        "workflow_wall_clock_s": round(wall_clock_s, 3),
        "samples": len(per_sample),
        "successful_samples": sum(1 for row in per_sample if not row.get("error")),
        "failed_samples": sum(1 for row in per_sample if row.get("error")),
    }
    if workflow_latencies:
        overall_metrics.update(
            {
                "workflow_latency_mean_s": round(float(np.mean(workflow_latencies)), 3),
                "workflow_latency_p95_s": round(
                    float(np.percentile(workflow_latencies, 95)), 3
                ),
            }
        )

    return {
        "primary_model_metrics": primary_metrics,
        "judge_metrics": judge_metrics,
        "overall_metrics": overall_metrics,
    }


def print_socialomni_level2_summary(summary: dict[str, Any]) -> None:
    """Print a compact level2 summary."""
    logger.info(
        "SocialOmni level2 Q1 accuracy: %.2f%% (%s/%s), Q2 average: %.2f over %s scored samples",
        summary.get("q1_accuracy", 0.0) * 100.0,
        summary.get("q1_correct", 0),
        summary.get("q1_total", 0),
        summary.get("q2_avg_score", 0.0),
        summary.get("q2_count", 0),
    )


__all__ = [
    "JudgeOutcome",
    "JudgeSpec",
    "aggregate_judge_scores",
    "build_base_url",
    "build_chat_api_url",
    "build_socialomni_level1_prompt",
    "build_socialomni_level2_metrics",
    "build_socialomni_level2_q1_prompt",
    "build_socialomni_level2_q2_prompt",
    "build_socialomni_level2_summary",
    "compute_socialomni_level1_results",
    "cut_video_prefix",
    "ensure_ffmpeg_available",
    "extract_level1_prediction",
    "make_socialomni_level1_send_fn",
    "parse_judge_score",
    "parse_level2_timestamp_to_seconds",
    "preflight_chat_completion_endpoint",
    "print_socialomni_level1_summary",
    "print_socialomni_level2_summary",
    "resolve_level2_outcome",
    "run_socialomni_judge",
    "run_socialomni_level2_benchmark",
    "run_socialomni_level2_sample",
    "send_chat_completion",
    "validate_judge_specs",
]
