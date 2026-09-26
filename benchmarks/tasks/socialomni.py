# SPDX-License-Identifier: Apache-2.0
"""Model and judge evaluation phases for SocialOmni."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Sequence
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import aiohttp

from benchmarks.benchmarker.data import RequestResult
from benchmarks.benchmarker.runner import BenchmarkRunner, RunConfig
from benchmarks.dataset.socialomni import (
    SocialOmniLevel1Sample,
    SocialOmniLevel2Sample,
    create_video_prefix,
)
from benchmarks.tasks.socialomni_protocol import (
    JUDGE_MAX_TOKENS,
    JUDGE_PARSE_ATTEMPTS,
    LEVEL1_MAX_TOKENS,
    LEVEL2_RESPONSE_MAX_TOKENS,
    LEVEL2_WHEN_MAX_TOKENS,
    JudgeSpec,
    build_level1_prompt,
    build_level1_result_records,
    build_judge_prompt,
    build_response_prompt,
    build_when_prompt,
    chat_completions_url,
    judge_payload,
    load_judge_config,
    make_level1_send_fn,
    model_payload,
    parse_choice,
    parse_judge_score,
    parse_when,
    request_chat_completion,
    validate_endpoint_url,
    validate_judge_credentials,
)
from benchmarks.metrics.socialomni import SOCIALOMNI_SCORE_BUCKETS

async def run_level2_model(
    samples: Sequence[SocialOmniLevel2Sample],
    *,
    model: str,
    base_url: str,
    prefix_cache_dir: str | Path,
    max_concurrency: int,
    timeout_s: int,
    request_rate: float = float("inf"),
    warmup: int | None = None,
    disable_tqdm: bool = False,
) -> tuple[list[dict[str, Any]], list[RequestResult], float]:
    """Prepare video prefixes, then run decisions and gold-positive responses."""
    records: list[dict[str, Any]] = []
    prepared: list[tuple[SocialOmniLevel2Sample, Path]] = []
    requests: list[RequestResult] = []
    for sample in samples:
        record = {
            "sample_id": sample.sample_id,
            "gold_when": sample.gold_when,
            "predicted_when": "",
            "when_success": False,
            "when_raw_response": "",
            "gold_response": "",
            "gold_response_success": False if sample.gold_when == "YES" else None,
            "gold_judge_scores": {},
            "judge_results": {},
            "requests": [],
        }
        records.append(record)
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
            record["requests"].append(asdict(failure))
        else:
            prepared.append((sample, prefix))

    by_id = {record["sample_id"]: record for record in records}
    measured_wall_s = 0.0
    for phase in ("when", "response"):
        cohort = [
            item for item in prepared if phase == "when" or item[0].gold_when == "YES"
        ]
        if not cohort:
            continue

        async def send(
            session: aiohttp.ClientSession, item: tuple[SocialOmniLevel2Sample, Path]
        ) -> RequestResult:
            sample, prefix = item
            prompt = (
                build_when_prompt(sample)
                if phase == "when"
                else build_response_prompt(sample)
            )
            max_tokens = (
                LEVEL2_WHEN_MAX_TOKENS
                if phase == "when"
                else LEVEL2_RESPONSE_MAX_TOKENS
            )
            return await request_chat_completion(
                session,
                api_url=chat_completions_url(base_url),
                payload=model_payload(model, prompt, str(prefix), max_tokens),
                request_id=f"{sample.sample_id}:{phase}",
            )

        runner = BenchmarkRunner(
            RunConfig(
                max_concurrency=max_concurrency,
                request_rate=request_rate,
                timeout_s=timeout_s,
                warmup=warmup,
                disable_tqdm=disable_tqdm,
                trust_env=True,
            )
        )
        outcomes = await runner.run(cohort, send)
        measured_wall_s += runner.wall_clock_s
        requests.extend(outcomes)
        for (sample, _prefix), result in zip(cohort, outcomes, strict=True):
            record = by_id[sample.sample_id]
            record["requests"].append(asdict(result))
            if phase == "when":
                record["when_success"] = result.is_success
                record["when_raw_response"] = result.text
                record["predicted_when"] = (
                    parse_when(result.text) if result.is_success else ""
                )
            else:
                record["gold_response"] = result.text
                record["gold_response_success"] = result.is_success
    return records, requests, measured_wall_s


async def run_judges(
    samples: Sequence[SocialOmniLevel2Sample],
    records: list[dict[str, Any]],
    judges: Sequence[JudgeSpec],
    *,
    timeout_s: int,
    request_rate: float = float("inf"),
    disable_tqdm: bool = False,
) -> tuple[list[RequestResult], list[dict[str, str]]]:
    """Return logical scores and retain each physical request in the records."""
    by_id = {sample.sample_id: sample for sample in samples}
    eligible = [
        record
        for record in records
        if record["gold_when"] == "YES"
        and record["gold_response_success"]
        and str(record["gold_response"]).strip()
    ]
    attempts_by_score: dict[tuple[str, str], list[dict[str, Any]]] = {}

    async def run_judge(judge: JudgeSpec) -> list[RequestResult]:
        async def send(
            session: aiohttp.ClientSession, record: dict[str, Any]
        ) -> RequestResult:
            sample = by_id[str(record["sample_id"])]
            started = time.perf_counter()
            total_engine_time = 0.0
            total_prompt_tokens = 0
            total_completion_tokens = 0
            score = None
            attempts = attempts_by_score[(str(record["sample_id"]), judge.name)] = []
            score_id = f"{sample.sample_id}:judge:{judge.name}"
            for attempt in range(JUDGE_PARSE_ATTEMPTS):
                attempt_start = len(attempts)
                result = await request_chat_completion(
                    session,
                    api_url=chat_completions_url(judge.base_url),
                    payload=judge_payload(
                        judge, build_judge_prompt(sample, str(record["gold_response"]))
                    ),
                    request_id=f"{score_id}:attempt:{attempt + 1}",
                    api_key_env=judge.api_key_env,
                    attempt_records=attempts,
                )
                for physical in attempts[attempt_start:]:
                    total_engine_time += physical["engine_time_s"]
                    total_prompt_tokens += physical["prompt_tokens"]
                    total_completion_tokens += physical["completion_tokens"]
                if not result.is_success:
                    break
                score = parse_judge_score(result.text)
                if score in SOCIALOMNI_SCORE_BUCKETS:
                    break
                if attempt + 1 < JUDGE_PARSE_ATTEMPTS:
                    await asyncio.sleep(2**attempt)
            result = replace(result, request_id=score_id)
            result.latency_s = time.perf_counter() - started
            result.engine_time_s = total_engine_time
            result.tok_per_s = (
                total_completion_tokens / total_engine_time
                if total_engine_time
                else 0.0
            )
            result.prompt_tokens = total_prompt_tokens
            result.completion_tokens = total_completion_tokens
            if score not in SOCIALOMNI_SCORE_BUCKETS:
                result.is_success = False
                result.error = result.error or (
                    f"invalid judge score after {JUDGE_PARSE_ATTEMPTS} attempts: "
                    f"{result.text!r}"
                )
            return result

        # Disable warmup to avoid duplicate paid judge requests.
        runner = BenchmarkRunner(
            RunConfig(
                max_concurrency=judge.max_concurrency,
                request_rate=request_rate,
                timeout_s=timeout_s,
                warmup=0,
                disable_tqdm=disable_tqdm,
                trust_env=True,
            )
        )
        return await runner.run(eligible, send)

    outcomes = await asyncio.gather(*(run_judge(judge) for judge in judges))
    failures: list[dict[str, str]] = []
    results: list[RequestResult] = []
    for judge, judge_results in zip(judges, outcomes, strict=True):
        for record, result in zip(eligible, judge_results, strict=True):
            score = parse_judge_score(result.text) if result.is_success else None
            results.append(result)
            record["judge_results"][judge.name] = {
                "request": asdict(result),
                "attempts": attempts_by_score[(str(record["sample_id"]), judge.name)],
                "score": score,
                "raw_response": result.text,
                "is_success": result.is_success,
                "latency_s": result.latency_s,
                "prompt_tokens": result.prompt_tokens,
                "completion_tokens": result.completion_tokens,
                "error": result.error,
            }
            if result.is_success and score is not None:
                record["gold_judge_scores"][judge.name] = score
            else:
                failures.append(
                    {
                        "request_id": result.request_id,
                        "sample_id": str(record["sample_id"]),
                        "judge": judge.name,
                        "phase": "level2_judge",
                        "error": result.error,
                    }
                )
    return results, failures
