# SPDX-License-Identifier: Apache-2.0
"""HTTP protocol helpers for the SocialOmni benchmark."""

from __future__ import annotations

import asyncio
import json
import os
import re
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import aiohttp

from benchmarks.benchmarker.data import RequestResult
from benchmarks.dataset.socialomni import SocialOmniLevel1Sample, SocialOmniLevel2Sample
from benchmarks.metrics.socialomni import SOCIALOMNI_JUDGE_NAMES, SOCIALOMNI_SCORE_BUCKETS

RETRYABLE_STATUS = frozenset({408, 429})
# Reasoning-capable judges may consume hidden tokens before emitting the score.
JUDGE_MAX_TOKENS = 8192
JUDGE_PARSE_ATTEMPTS = 3
LEVEL1_MAX_TOKENS = 32
LEVEL2_WHEN_MAX_TOKENS = 8
LEVEL2_RESPONSE_MAX_TOKENS = 256


@dataclass(frozen=True)
class JudgeSpec:
    name: str
    model: str
    base_url: str
    api_key_env: str | None
    max_concurrency: int


def public_judge_record(judge: JudgeSpec) -> dict[str, Any]:
    """Serialize judge metadata without persisting endpoint paths or credentials."""
    parts = urlsplit(judge.base_url)
    host = parts.hostname or ""
    if parts.port is not None:
        host = f"{host}:{parts.port}"
    return {
        "name": judge.name,
        "model": judge.model,
        "base_url": urlunsplit((parts.scheme, host, "", "", "")),
        "api_key_env": judge.api_key_env,
        "max_concurrency": judge.max_concurrency,
    }


def validate_endpoint_url(base_url: str) -> None:
    try:
        parts = urlsplit(base_url)
        valid = (
            parts.scheme in ("http", "https")
            and bool(parts.hostname)
            and (parts.port is None or 1 <= parts.port <= 65535)
            and parts.username is None
            and parts.password is None
            and not parts.query
            and not parts.fragment
        )
    except ValueError:
        valid = False
    if not valid:
        raise ValueError(
            "base_url must be an HTTP(S) URL with a valid host and port, "
            "without userinfo, query or fragment"
        )


def chat_completions_url(base_url: str) -> str:
    base = base_url.rstrip("/")
    if base.endswith("/chat/completions"):
        return base
    return (
        f"{base}/chat/completions"
        if base.endswith("/v1")
        else f"{base}/v1/chat/completions"
    )


def load_judge_config(path: str | Path) -> list[JudgeSpec]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = payload.get("judges") if isinstance(payload, dict) else None
    if not isinstance(rows, list) or len(rows) != 3:
        raise ValueError("judge config must contain exactly three judges")
    allowed = {"name", "model", "base_url", "api_key_env", "max_concurrency"}
    judges: list[JudgeSpec] = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or set(row) - allowed:
            raise ValueError(f"judges[{index}] has invalid fields")
        for field in ("name", "model", "base_url"):
            if not isinstance(row.get(field), str) or not row[field].strip():
                raise ValueError(f"judges[{index}].{field} must be non-empty")
        concurrency = row.get("max_concurrency", 1)
        validate_endpoint_url(row["base_url"].strip())
        if type(concurrency) is not int or concurrency < 1:
            raise ValueError(f"judges[{index}].max_concurrency must be >= 1")
        api_key_env = row.get("api_key_env")
        if api_key_env is not None and (
            not isinstance(api_key_env, str)
            or not api_key_env
            or api_key_env != api_key_env.strip()
        ):
            raise ValueError(
                f"judges[{index}].api_key_env must be a non-empty string "
                "without surrounding whitespace, or null"
            )
        judges.append(
            JudgeSpec(
                name=row["name"].strip(),
                model=row["model"].strip(),
                base_url=row["base_url"].strip(),
                api_key_env=api_key_env,
                max_concurrency=concurrency,
            )
        )
    if {judge.name for judge in judges} != set(SOCIALOMNI_JUDGE_NAMES):
        raise ValueError(f"judge names must be exactly {SOCIALOMNI_JUDGE_NAMES}")
    return judges


def _headers(api_key_env: str | None) -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if api_key_env:
        token = os.environ.get(api_key_env)
        if not token:
            raise RuntimeError(
                f"API key environment variable is not set: {api_key_env}"
            )
        headers["Authorization"] = f"Bearer {token}"
    return headers


def validate_judge_credentials(judges: Sequence[JudgeSpec]) -> None:
    """Fail before model inference when a configured judge credential is absent."""
    for judge in judges:
        _headers(judge.api_key_env)


def _response_text(body: dict[str, Any]) -> str:
    if "error" in body:
        raise ValueError("response contains an error")
    choices = body.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError("choices must be a non-empty list")
    if not isinstance(choices[0], dict):
        raise ValueError("choices[0] must be an object")
    message = choices[0].get("message")
    if not isinstance(message, dict):
        raise ValueError("choices[0].message must be an object")
    content = message.get("content")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        if any(
            not isinstance(part, dict)
            or part.get("type") != "text"
            or not isinstance(part.get("text"), str)
            for part in content
        ):
            raise ValueError("message content parts must contain text strings")
        return "\n".join(part["text"].strip() for part in content).strip()
    raise ValueError("message content must be a string or a list of text parts")


async def request_chat_completion(
    session: aiohttp.ClientSession,
    *,
    api_url: str,
    payload: dict[str, Any],
    request_id: str,
    api_key_env: str | None = None,
    max_attempts: int = 3,
    attempt_records: list[dict[str, Any]] | None = None,
) -> RequestResult:
    """Send a chat completion, retrying transient errors up to max_attempts."""
    request_started = time.perf_counter()
    last = RequestResult(request_id=request_id, error="not attempted")

    def finish(result: RequestResult) -> RequestResult:
        if attempt_records is not None:
            elapsed = time.perf_counter() - attempt_started
            physical = replace(
                result,
                request_id=f"{request_id}:http:{attempt + 1}",
                latency_s=elapsed,
                engine_time_s=elapsed if result.is_success else 0.0,
                tok_per_s=result.completion_tokens / elapsed if elapsed else 0.0,
            )
            attempt_records.append(asdict(physical))
        return result

    for attempt in range(max_attempts):
        attempt_started = time.perf_counter()
        try:
            async with session.post(
                api_url, json=payload, headers=_headers(api_key_env)
            ) as response:
                raw = await response.text()
                if response.status >= 400:
                    last = RequestResult(
                        request_id=request_id,
                        latency_s=time.perf_counter() - request_started,
                        error=f"HTTP {response.status}: {raw[:2000]}",
                    )
                    retry = (
                        response.status in RETRYABLE_STATUS
                        or 500 <= response.status < 600
                    )
                else:
                    try:
                        body = json.loads(raw)
                    except json.JSONDecodeError as exc:
                        last = RequestResult(
                            request_id=request_id,
                            latency_s=time.perf_counter() - request_started,
                            error=f"invalid JSON response: {exc}: {raw[:1000]}",
                        )
                        retry = True
                    else:
                        if not isinstance(body, dict):
                            last = RequestResult(
                                request_id=request_id,
                                latency_s=time.perf_counter() - request_started,
                                error=f"invalid JSON response object: {raw[:1000]}",
                            )
                            retry = True
                        else:
                            try:
                                text = _response_text(body)
                            except ValueError as exc:
                                return finish(
                                    RequestResult(
                                        request_id=request_id,
                                        latency_s=time.perf_counter() - request_started,
                                        error=f"invalid completion response: {exc}",
                                    )
                                )
                            usage = body.get("usage", {})
                            try:
                                if not isinstance(usage, dict):
                                    raise ValueError("usage must be an object")
                                prompt_tokens = usage.get("prompt_tokens", 0)
                                completion_tokens = usage.get("completion_tokens", 0)
                                for count in (prompt_tokens, completion_tokens):
                                    if type(count) is not int or count < 0:
                                        raise ValueError(
                                            "token counts must be non-negative integers"
                                        )
                            except (TypeError, ValueError, OverflowError) as exc:
                                return finish(
                                    RequestResult(
                                        request_id=request_id,
                                        text=text,
                                        latency_s=time.perf_counter() - request_started,
                                        error=f"invalid token usage: {exc}",
                                    )
                                )
                            elapsed = time.perf_counter() - request_started
                            return finish(
                                RequestResult(
                                    request_id=request_id,
                                    text=text,
                                    is_success=True,
                                    latency_s=elapsed,
                                    engine_time_s=elapsed,
                                    tok_per_s=(
                                        completion_tokens / elapsed if elapsed else 0.0
                                    ),
                                    prompt_tokens=prompt_tokens,
                                    completion_tokens=completion_tokens,
                                )
                            )
        except RuntimeError as exc:
            return finish(
                RequestResult(
                    request_id=request_id,
                    latency_s=time.perf_counter() - request_started,
                    error=str(exc),
                )
            )
        except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
            last = RequestResult(
                request_id=request_id,
                latency_s=time.perf_counter() - request_started,
                error=f"{type(exc).__name__}: {exc}",
            )
            retry = True
        finish(last)
        if not retry or attempt + 1 == max_attempts:
            return last
        await asyncio.sleep(2**attempt)
    return last


def parse_choice(text: str, choices: Sequence[str]) -> str:
    content = (text or "").strip().upper()
    alphabet = "".join(re.escape(choice) for choice in choices)
    match = re.fullmatch(
        rf"(?:(?:ANSWER|CHOICE)\s*(?:IS|:)?\s*)?([{alphabet}])[.)]?", content
    ) or re.fullmatch(rf"\\?BOXED\s*\{{\s*([{alphabet}])\s*\}}", content)
    return match.group(1) if match else ""


def build_level1_result_records(
    samples: Sequence[SocialOmniLevel1Sample], results: Sequence[RequestResult]
) -> list[dict[str, Any]]:
    return [
        {
            "sample_id": sample.sample_id,
            "gold_answer": sample.gold_answer,
            "predicted_answer": (
                parse_choice(result.text.strip().splitlines()[-1], ("A", "B", "C", "D"))
                if result.is_success and result.text.strip()
                else ""
            ),
            "visibility": sample.visibility,
            "is_success": result.is_success,
            "raw_response": result.text,
            "request": asdict(result),
        }
        for sample, result in zip(samples, results, strict=True)
    ]


def parse_when(text: str) -> str:
    choice = parse_choice(text, ("A", "B"))
    if choice:
        return "YES" if choice == "A" else "NO"
    normalized = (text or "").strip().upper()
    return normalized if normalized in {"YES", "NO"} else ""


def parse_judge_score(text: str) -> int | None:
    values = re.findall(r"[+-]?(?:\d+(?:\.\d+)?|\.\d+)", text or "")
    if len(values) != 1 or values[0] not in {"0", "25", "50", "75", "100"}:
        return None
    return int(values[0])


def build_level1_prompt(sample: SocialOmniLevel1Sample) -> str:
    options = "\n".join(
        f"{letter}. {option}"
        for letter, option in zip(("A", "B", "C", "D"), sample.options, strict=True)
    )
    return (
        f"{sample.question}\n{options}\nUse the video and its audio. "
        "Reply on the last line as Answer: X, where X is A, B, C, or D."
    )


def build_when_prompt(sample: SocialOmniLevel2Sample) -> str:
    return (
        f"Target participant: {sample.target_participant}\n{sample.question_when}\n"
        "A. YES\nB. NO\nUse only the provided audio-video prefix. "
        "Reply exactly as Answer: A or Answer: B."
    )


def build_response_prompt(sample: SocialOmniLevel2Sample) -> str:
    return (
        f"Target participant: {sample.target_participant}\n{sample.question_how}\n"
        "Use only the provided audio-video prefix. Return only the participant's next utterance."
    )


def build_judge_prompt(sample: SocialOmniLevel2Sample, candidate: str) -> str:
    return (
        "Score the candidate continuation for contextual fit, target-role consistency, "
        "intent correctness, and key information completeness. Choose exactly one score "
        "from {0, 25, 50, 75, 100}. Output only the score.\n\n"
        f"Target participant:\n{sample.target_participant}\n\n"
        f"Reference context:\n{sample.reference_context}\n\n"
        f"Reference continuation:\n{sample.reference_response}\n\n"
        f"Candidate continuation:\n{candidate}"
    )


def model_payload(
    model: str, prompt: str, video_path: str, max_tokens: int
) -> dict[str, Any]:
    return {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "videos": [video_path],
        "use_audio_in_video": True,
        "modalities": ["text"],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": False,
    }


def judge_payload(judge: JudgeSpec, prompt: str) -> dict[str, Any]:
    return {
        "model": judge.model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": JUDGE_MAX_TOKENS,
        "temperature": 0.0,
        "stream": False,
    }


def make_level1_send_fn(model: str, base_url: str):
    async def send(
        session: aiohttp.ClientSession, sample: SocialOmniLevel1Sample
    ) -> RequestResult:
        return await request_chat_completion(
            session,
            api_url=chat_completions_url(base_url),
            payload=model_payload(
                model,
                build_level1_prompt(sample),
                sample.video_path,
                LEVEL1_MAX_TOKENS,
            ),
            request_id=sample.sample_id,
        )

    return send

