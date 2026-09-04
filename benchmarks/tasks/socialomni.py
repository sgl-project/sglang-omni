# SPDX-License-Identifier: Apache-2.0
"""Request and media helpers for SocialOmni."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
import os
import re
import shutil
import time
import uuid
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, TypeVar, cast

import aiohttp
import imageio_ffmpeg

from benchmarks.benchmarker.data import RequestResult
from benchmarks.benchmarker.runner import resolve_warmup
from benchmarks.dataset.socialomni import SocialOmniLevel1Sample, SocialOmniLevel2Sample
from benchmarks.metrics.socialomni import (
    SOCIALOMNI_JUDGE_NAMES,
    SOCIALOMNI_SCORE_BUCKETS,
)

RETRYABLE_STATUS = frozenset({408, 429})
PREFIX_ENCODING = {
    "video_codec": "libx264",
    "preset": "fast",
    "crf": "18",
    "audio_codec": "aac",
    "audio_bitrate": "192k",
}
# Reasoning-capable judges may consume hidden tokens before emitting the score.
JUDGE_MAX_TOKENS = 8192
JUDGE_PARSE_ATTEMPTS = 3
LEVEL1_MAX_TOKENS = 32
LEVEL2_WHEN_MAX_TOKENS = 8
LEVEL2_RESPONSE_MAX_TOKENS = 256

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class JudgeSpec:
    name: str
    model: str
    base_url: str
    api_key_env: str | None
    max_concurrency: int


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
        if not isinstance(concurrency, int) or concurrency < 1:
            raise ValueError(f"judges[{index}].max_concurrency must be >= 1")
        api_key_env = row.get("api_key_env")
        if api_key_env is not None and not isinstance(api_key_env, str):
            raise ValueError(f"judges[{index}].api_key_env must be a string or null")
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
        if judge.api_key_env and not os.environ.get(judge.api_key_env):
            raise RuntimeError(
                f"API key environment variable is not set: {judge.api_key_env}"
            )


def _response_text(body: dict[str, Any]) -> str:
    choices = body.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    if not isinstance(choices[0], dict):
        return ""
    message = choices[0].get("message")
    if not isinstance(message, dict):
        return ""
    content = message.get("content")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        return "\n".join(
            str(part.get("text", "")).strip()
            for part in content
            if isinstance(part, dict) and part.get("type") == "text"
        ).strip()
    return ""


async def request_chat_completion(
    session: aiohttp.ClientSession,
    *,
    api_url: str,
    payload: dict[str, Any],
    request_id: str,
    api_key_env: str | None = None,
    max_attempts: int = 3,
) -> RequestResult:
    """Send one completion with bounded retry and preserved HTTP error text."""
    request_started = time.perf_counter()
    last = RequestResult(request_id=request_id, error="not attempted")
    for attempt in range(max_attempts):
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
                            usage = body.get("usage") or {}
                            if not isinstance(usage, dict):
                                usage = {}
                            return RequestResult(
                                request_id=request_id,
                                text=_response_text(body),
                                is_success=True,
                                latency_s=time.perf_counter() - request_started,
                                prompt_tokens=int(usage.get("prompt_tokens") or 0),
                                completion_tokens=int(
                                    usage.get("completion_tokens") or 0
                                ),
                            )
        except RuntimeError as exc:
            return RequestResult(
                request_id=request_id,
                latency_s=time.perf_counter() - request_started,
                error=str(exc),
            )
        except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
            last = RequestResult(
                request_id=request_id,
                latency_s=time.perf_counter() - request_started,
                error=f"{type(exc).__name__}: {exc}",
            )
            retry = True
        if not retry or attempt + 1 == max_attempts:
            return last
        await asyncio.sleep(2**attempt)
    return last


def parse_choice(text: str, choices: Sequence[str]) -> str:
    content = (text or "").strip().upper()
    alphabet = "".join(re.escape(choice) for choice in choices)
    explicit = re.findall(
        rf"(?:ANSWER|CHOICE)\s*(?:IS|:)?\s*([{alphabet}])\b", content
    ) + re.findall(rf"\\?BOXED\s*\{{\s*([{alphabet}])\s*\}}", content)
    if explicit:
        return explicit[0] if len(set(explicit)) == 1 else ""
    plain = re.fullmatch(rf"([{alphabet}])[.)]?", content)
    return plain.group(1) if plain else ""


def parse_when(text: str) -> str:
    choice = parse_choice(text, ("A", "B"))
    if choice:
        return "YES" if choice == "A" else "NO"
    normalized = (text or "").strip().upper()
    return normalized if normalized in {"YES", "NO"} else ""


def parse_judge_score(text: str) -> int | None:
    values = re.findall(r"(?<!\d)(?:100|75|50|25|0)(?!\d)", text or "")
    return int(values[0]) if len(values) == 1 else None


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


def resolve_ffmpeg_executable() -> str | None:
    system = shutil.which("ffmpeg")
    if system:
        return system
    bundled = Path(imageio_ffmpeg.get_ffmpeg_exe())
    return str(bundled) if bundled.is_file() and os.access(bundled, os.X_OK) else None


def build_ffmpeg_prefix_command(
    ffmpeg: str, source: Path, timestamp_s: float, output: Path
) -> list[str]:
    return [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(source),
        "-t",
        f"{timestamp_s:.6f}",
        "-map",
        "0:v:0",
        "-map",
        "0:a?",
        "-c:v",
        PREFIX_ENCODING["video_codec"],
        "-preset",
        PREFIX_ENCODING["preset"],
        "-crf",
        PREFIX_ENCODING["crf"],
        "-c:a",
        PREFIX_ENCODING["audio_codec"],
        "-b:a",
        PREFIX_ENCODING["audio_bitrate"],
        "-movflags",
        "+faststart",
        "-y",
        str(output),
    ]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


async def create_video_prefix(
    input_path: str | Path, timestamp_s: float, cache_dir: str | Path
) -> Path:
    """Re-encode video and audio up to the query time into an atomic cache entry."""
    if not math.isfinite(timestamp_s) or timestamp_s <= 0:
        raise ValueError("timestamp_s must be finite and positive")
    source = Path(input_path).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    ffmpeg = resolve_ffmpeg_executable()
    if not ffmpeg:
        raise RuntimeError("ffmpeg is required for SocialOmni Level 2")
    key = hashlib.sha256(
        json.dumps(
            {
                "source_sha256": await asyncio.to_thread(_sha256, source),
                "timestamp_s": f"{timestamp_s:.6f}",
                "encoding": PREFIX_ENCODING,
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    output = cache / f"{key}.mp4"
    if output.is_file() and output.stat().st_size:
        return output
    temporary = cache / f".{key}.{uuid.uuid4().hex}.tmp.mp4"
    process = await asyncio.create_subprocess_exec(
        *build_ffmpeg_prefix_command(ffmpeg, source, timestamp_s, temporary),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    _, stderr = await process.communicate()
    if process.returncode or not temporary.is_file() or not temporary.stat().st_size:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(
            f"ffmpeg prefix generation failed for {source}: "
            f"{stderr.decode(errors='replace')[:2000]}"
        )
    temporary.replace(output)
    return output


T = TypeVar("T")
R = TypeVar("R")


async def bounded_map(
    items: Sequence[T], worker: Callable[[T], Awaitable[R]], max_concurrency: int
) -> list[R]:
    """Map with a fixed worker set while preserving input order."""
    if max_concurrency < 1:
        raise ValueError("max_concurrency must be >= 1")
    queue: asyncio.Queue[tuple[int, T] | None] = asyncio.Queue()
    for index, item in enumerate(items):
        queue.put_nowait((index, item))
    worker_count = min(max_concurrency, len(items))
    for _ in range(worker_count):
        queue.put_nowait(None)
    missing = object()
    results: list[R | object] = [missing] * len(items)

    async def consume() -> None:
        while (item := await queue.get()) is not None:
            index, value = item
            results[index] = await worker(value)

    tasks = [asyncio.create_task(consume()) for _ in range(worker_count)]
    await asyncio.gather(*tasks)
    return [cast(R, result) for result in results]


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


async def run_level2_model(
    samples: Sequence[SocialOmniLevel2Sample],
    *,
    model: str,
    base_url: str,
    prefix_cache_dir: str | Path,
    max_concurrency: int,
    timeout_s: int,
) -> tuple[list[dict[str, Any]], list[RequestResult], float]:
    """Run fixed-prefix when and forced gold-positive response requests."""
    timeout = aiohttp.ClientTimeout(total=timeout_s)
    requests: list[RequestResult] = []
    async with aiohttp.ClientSession(timeout=timeout, trust_env=True) as session:
        warmup_count = resolve_warmup(None, max_concurrency) if samples else 0
        if warmup_count:
            sample = samples[0]
            prefix = await create_video_prefix(
                sample.video_path, sample.timestamp_s, prefix_cache_dir
            )

            async def warmup_when(_index: int) -> RequestResult:
                return await request_chat_completion(
                    session,
                    api_url=chat_completions_url(base_url),
                    payload=model_payload(
                        model,
                        build_when_prompt(sample),
                        str(prefix),
                        LEVEL2_WHEN_MAX_TOKENS,
                    ),
                    request_id=f"{sample.sample_id}:warmup",
                )

            warmup_results = await bounded_map(
                range(warmup_count), warmup_when, max_concurrency
            )
            for index, result in enumerate(warmup_results, start=1):
                status = "ok" if result.is_success else result.error
                logger.info("SocialOmni warmup %d/%d: %s", index, warmup_count, status)

        started = time.perf_counter()

        async def run_one(sample: SocialOmniLevel2Sample) -> dict[str, Any]:
            try:
                prefix = await create_video_prefix(
                    sample.video_path, sample.timestamp_s, prefix_cache_dir
                )
            except (OSError, RuntimeError, ValueError) as exc:
                failure = RequestResult(
                    request_id=f"{sample.sample_id}:prefix",
                    error=f"{type(exc).__name__}: {exc}",
                )
                requests.append(failure)
                return {
                    "sample_id": sample.sample_id,
                    "gold_when": sample.gold_when,
                    "predicted_when": "",
                    "when_success": False,
                    "when_raw_response": "",
                    "gold_response": "",
                    "gold_response_success": False,
                    "gold_judge_scores": {},
                    "judge_results": {},
                    "requests": [asdict(failure)],
                }
            when = await request_chat_completion(
                session,
                api_url=chat_completions_url(base_url),
                payload=model_payload(
                    model,
                    build_when_prompt(sample),
                    str(prefix),
                    LEVEL2_WHEN_MAX_TOKENS,
                ),
                request_id=f"{sample.sample_id}:when",
            )
            requests.append(when)
            response = None
            if sample.gold_when == "YES":
                response = await request_chat_completion(
                    session,
                    api_url=chat_completions_url(base_url),
                    payload=model_payload(
                        model,
                        build_response_prompt(sample),
                        str(prefix),
                        LEVEL2_RESPONSE_MAX_TOKENS,
                    ),
                    request_id=f"{sample.sample_id}:response",
                )
                requests.append(response)
            return {
                "sample_id": sample.sample_id,
                "gold_when": sample.gold_when,
                "predicted_when": parse_when(when.text) if when.is_success else "",
                "when_success": when.is_success,
                "when_raw_response": when.text,
                "gold_response": response.text if response else "",
                "gold_response_success": response.is_success if response else None,
                "gold_judge_scores": {},
                "judge_results": {},
                "requests": [
                    asdict(result) for result in (when, response) if result is not None
                ],
            }

        records = await bounded_map(samples, run_one, max_concurrency)
        measured_wall_s = time.perf_counter() - started
    return records, requests, measured_wall_s


async def run_judges(
    samples: Sequence[SocialOmniLevel2Sample],
    records: list[dict[str, Any]],
    judges: Sequence[JudgeSpec],
    *,
    timeout_s: int,
) -> tuple[list[RequestResult], list[dict[str, str]]]:
    """Score every non-empty gold-positive response with all fixed judges."""
    by_id = {sample.sample_id: sample for sample in samples}
    jobs = [
        (record, judge)
        for record in records
        if record["gold_when"] == "YES"
        and record["gold_response_success"]
        and str(record["gold_response"]).strip()
        for judge in judges
    ]
    semaphores = {
        judge.name: asyncio.Semaphore(judge.max_concurrency) for judge in judges
    }
    timeout = aiohttp.ClientTimeout(total=timeout_s)
    async with aiohttp.ClientSession(timeout=timeout, trust_env=True) as session:

        async def run_one(job: tuple[dict[str, Any], JudgeSpec]):
            record, judge = job
            sample = by_id[str(record["sample_id"])]
            async with semaphores[judge.name]:
                total_latency = 0.0
                total_prompt_tokens = 0
                total_completion_tokens = 0
                score = None
                for attempt in range(JUDGE_PARSE_ATTEMPTS):
                    result = await request_chat_completion(
                        session,
                        api_url=chat_completions_url(judge.base_url),
                        payload=judge_payload(
                            judge,
                            build_judge_prompt(sample, str(record["gold_response"])),
                        ),
                        request_id=f"{sample.sample_id}:judge:{judge.name}",
                        api_key_env=judge.api_key_env,
                    )
                    total_latency += result.latency_s
                    total_prompt_tokens += result.prompt_tokens
                    total_completion_tokens += result.completion_tokens
                    if not result.is_success:
                        break
                    score = parse_judge_score(result.text)
                    if score in SOCIALOMNI_SCORE_BUCKETS:
                        break
                    if attempt + 1 < JUDGE_PARSE_ATTEMPTS:
                        await asyncio.sleep(2**attempt)
                result.latency_s = total_latency
                result.prompt_tokens = total_prompt_tokens
                result.completion_tokens = total_completion_tokens
            if score not in SOCIALOMNI_SCORE_BUCKETS:
                result.is_success = False
                result.error = result.error or (
                    f"invalid judge score after {JUDGE_PARSE_ATTEMPTS} attempts: "
                    f"{result.text!r}"
                )
            return record, judge.name, score, result

        outcomes = await bounded_map(
            jobs, run_one, sum(judge.max_concurrency for judge in judges)
        )
    failures: list[dict[str, str]] = []
    results: list[RequestResult] = []
    for record, judge_name, score, result in outcomes:
        results.append(result)
        record["judge_results"][judge_name] = {
            "score": score,
            "raw_response": result.text,
            "is_success": result.is_success,
            "latency_s": result.latency_s,
            "prompt_tokens": result.prompt_tokens,
            "completion_tokens": result.completion_tokens,
            "error": result.error,
        }
        if result.is_success and score is not None:
            record["gold_judge_scores"][judge_name] = score
        else:
            failures.append(
                {
                    "request_id": result.request_id,
                    "sample_id": str(record["sample_id"]),
                    "judge": judge_name,
                    "error": result.error,
                }
            )
    return results, failures
