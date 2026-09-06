# SPDX-License-Identifier: Apache-2.0
"""Evaluate Qwen3-Omni on the SocialOmni paper protocol."""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

from benchmarks.benchmarker.runner import BenchmarkRunner, RunConfig, resolve_warmup
from benchmarks.benchmarker.utils import save_json_results, wait_for_service
from benchmarks.dataset.socialomni import (
    SOCIALOMNI_DATASET_ID,
    SOCIALOMNI_DATASET_REVISION,
    SOCIALOMNI_PAPER_CORE_SIZE,
    inspect_socialomni_dataset,
    load_socialomni_level1_samples,
    load_socialomni_level2_samples,
)
from benchmarks.metrics.performance import compute_speed_metrics
from benchmarks.metrics.socialomni import (
    SOCIALOMNI_JUDGE_NAMES,
    SOCIALOMNI_SCORE_BUCKETS,
    compute_socialomni_level1_metrics,
    compute_socialomni_level2_metrics,
    compute_socialomni_when_metrics,
)
from benchmarks.runtime_metrics import collect_benchmark_provenance
from benchmarks.tasks.socialomni import (
    JUDGE_MAX_TOKENS,
    LEVEL1_MAX_TOKENS,
    LEVEL2_RESPONSE_MAX_TOKENS,
    LEVEL2_WHEN_MAX_TOKENS,
    load_judge_config,
    make_level1_send_fn,
    parse_choice,
    run_judges,
    run_level2_model,
    validate_judge_credentials,
)


@dataclass(frozen=True)
class SocialOmniEvalConfig:
    dataset_root: str
    model: str
    base_url: str
    level: Literal["level1", "level2", "both"]
    judge_config: str | None
    prefix_cache_dir: str
    mini: bool
    max_samples: int | None
    max_concurrency: int
    timeout_s: int
    output_dir: str


def _request_failure(result: object, phase: str) -> dict[str, str] | None:
    if getattr(result, "is_success", False):
        return None
    return {
        "phase": phase,
        "request_id": str(getattr(result, "request_id", "")),
        "error": str(getattr(result, "error", "")),
    }


def _judges_complete(records: list[dict[str, Any]], configured: bool) -> bool:
    required = set(SOCIALOMNI_JUDGE_NAMES)

    def has_all_scores(record: dict[str, Any]) -> bool:
        scores = record["gold_judge_scores"]
        return (
            isinstance(scores, dict)
            and set(scores) == required
            and all(
                type(score) is int and score in SOCIALOMNI_SCORE_BUCKETS
                for score in scores.values()
            )
        )

    return configured and all(
        has_all_scores(record)
        for record in records
        if record["gold_when"] == "YES"
        and record["gold_response_success"]
        and str(record["gold_response"]).strip()
    )


async def run_socialomni(config: SocialOmniEvalConfig) -> dict[str, Any]:
    if config.max_concurrency < 1:
        raise ValueError("max_concurrency must be >= 1")
    levels = ("level1", "level2") if config.level == "both" else (config.level,)
    judges = (
        load_judge_config(config.judge_config)
        if "level2" in levels and config.judge_config
        else []
    )
    validate_judge_credentials(judges)
    warmup = resolve_warmup(None, config.max_concurrency)
    warmup_requests = {
        level: warmup if config.max_samples != 0 else 0 for level in levels
    }
    dataset_identity = inspect_socialomni_dataset(config.dataset_root, levels)
    provenance = collect_benchmark_provenance(
        model_id=config.model,
        model_revision=None,
        dataset_id=SOCIALOMNI_DATASET_ID,
        dataset_revision=(
            SOCIALOMNI_DATASET_REVISION
            if dataset_identity["metadata_matches_expected_revision"]
            else None
        ),
        launch_command=None,
        server_config={
            "base_url": config.base_url,
            "max_concurrency": config.max_concurrency,
            "warmup_requests": warmup_requests,
            "timeout_s": config.timeout_s,
            "temperature": 0.0,
            "use_audio_in_video": True,
        },
    )
    output: dict[str, Any] = {
        "config": {
            **asdict(config),
            "dataset_root": str(Path(config.dataset_root).resolve()),
            "judge_config": None,
            "dataset_id": SOCIALOMNI_DATASET_ID,
            "expected_dataset_revision": SOCIALOMNI_DATASET_REVISION,
            "warmup_requests": warmup_requests,
            "generation": {
                "temperature": 0.0,
                "stream": False,
                "level1_max_tokens": LEVEL1_MAX_TOKENS,
                "level2_when_max_tokens": LEVEL2_WHEN_MAX_TOKENS,
                "level2_response_max_tokens": LEVEL2_RESPONSE_MAX_TOKENS,
                "judge_max_tokens": JUDGE_MAX_TOKENS,
            },
        },
        "dataset": dataset_identity,
        "provenance": provenance,
        "summary": {},
        "per_sample": {},
        "failures": [],
    }

    if "level1" in levels:
        samples = load_socialomni_level1_samples(
            config.dataset_root,
            mini=config.mini,
            max_samples=config.max_samples,
        )
        runner = BenchmarkRunner(
            RunConfig(
                max_concurrency=config.max_concurrency,
                timeout_s=config.timeout_s,
            )
        )
        request_results = await runner.run(
            samples, make_level1_send_fn(config.model, config.base_url)
        )
        records = []
        for sample, result in zip(samples, request_results, strict=True):
            prediction = (
                parse_choice(result.text, ("A", "B", "C", "D"))
                if result.is_success
                else ""
            )
            records.append(
                {
                    "sample_id": sample.sample_id,
                    "gold_answer": sample.gold_answer,
                    "predicted_answer": prediction,
                    "visibility": sample.visibility,
                    "is_success": result.is_success,
                    "raw_response": result.text,
                    "request": asdict(result),
                }
            )
            failure = _request_failure(result, "level1")
            if failure:
                output["failures"].append(failure)
            elif not prediction:
                output["failures"].append(
                    {
                        "phase": "level1_parse",
                        "request_id": result.request_id,
                        "error": f"unparseable response: {result.text!r}",
                    }
                )
        output["per_sample"]["level1"] = records
        output["summary"]["level1"] = {
            "metrics": compute_socialomni_level1_metrics(records),
            "speed": compute_speed_metrics(
                request_results, wall_clock_s=runner.wall_clock_s
            ),
        }

    if "level2" in levels:
        samples = load_socialomni_level2_samples(
            config.dataset_root,
            mini=config.mini,
            max_samples=config.max_samples,
        )
        records, model_requests, model_wall_s = await run_level2_model(
            samples,
            model=config.model,
            base_url=config.base_url,
            prefix_cache_dir=config.prefix_cache_dir,
            max_concurrency=config.max_concurrency,
            timeout_s=config.timeout_s,
        )
        for result in model_requests:
            failure = _request_failure(result, "level2_model")
            if failure:
                output["failures"].append(failure)

        output["config"]["judges"] = [asdict(judge) for judge in judges]
        judge_requests = []
        judge_failures: list[dict[str, str]] = []
        judge_wall_s = 0.0
        if judges:
            judge_started = time.perf_counter()
            judge_requests, judge_failures = await run_judges(
                samples, records, judges, timeout_s=config.timeout_s
            )
            judge_wall_s = time.perf_counter() - judge_started
            output["failures"].extend(judge_failures)
        for record in records:
            if record["when_success"] and not record["predicted_when"]:
                output["failures"].append(
                    {
                        "phase": "level2_parse",
                        "request_id": f"{record['sample_id']}:when",
                        "error": f"unparseable response: {record['when_raw_response']!r}",
                    }
                )

        required_judgments = sum(
            1
            for record in records
            if record["gold_when"] == "YES"
            and record["gold_response_success"]
            and str(record["gold_response"]).strip()
        )
        judges_complete = _judges_complete(records, bool(judges))
        complete_metrics = (
            compute_socialomni_level2_metrics(records) if judges_complete else None
        )
        selected = {
            "when": (
                complete_metrics["when"]
                if complete_metrics
                else compute_socialomni_when_metrics(records)
            ),
            "quality": None,
            "judge_status": {
                "complete": judges_complete,
                "eligible_responses": required_judgments,
                "completed_scores": sum(
                    len(record["gold_judge_scores"]) for record in records
                ),
                "required_scores": required_judgments * len(SOCIALOMNI_JUDGE_NAMES),
            },
        }
        if complete_metrics:
            selected["quality"] = complete_metrics["quality"]
            selected["bootstrap"] = complete_metrics["bootstrap"]
            selected["judge_names"] = complete_metrics["judge_names"]
        output["summary"]["level2"] = {
            "metrics": selected,
            "speed": {
                "model": compute_speed_metrics(
                    model_requests, wall_clock_s=model_wall_s
                ),
                "judges": compute_speed_metrics(
                    judge_requests, wall_clock_s=judge_wall_s
                ),
            },
        }
        if len(records) >= SOCIALOMNI_PAPER_CORE_SIZE:
            core = records[:SOCIALOMNI_PAPER_CORE_SIZE]
            core_complete = _judges_complete(core, bool(judges))
            core_metrics = (
                compute_socialomni_level2_metrics(core) if core_complete else None
            )
            core_summary: dict[str, Any] = {
                "sample_count": len(core),
                "when": (
                    core_metrics["when"]
                    if core_metrics
                    else compute_socialomni_when_metrics(core)
                ),
                "quality": None,
                "judges_complete": core_complete,
            }
            if core_metrics:
                core_summary["quality"] = core_metrics["quality"]
            output["paper_core_200"] = core_summary
        output["per_sample"]["level2"] = records

    level1_complete = True
    level2_complete = "level2" not in levels or (
        output["summary"]["level2"]["metrics"]["judge_status"]["complete"]
    )
    output["summary"]["status"] = (
        "complete" if level1_complete and level2_complete else "incomplete"
    )
    return output


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--level", choices=("level1", "level2", "both"), default="both")
    parser.add_argument("--judge-config")
    parser.add_argument("--prefix-cache-dir", default="results/socialomni-prefixes")
    parser.add_argument("--mini", action="store_true")
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--max-concurrency", type=int, default=1)
    parser.add_argument("--timeout-s", type=int, default=300)
    parser.add_argument("--output-dir", default="results/socialomni")
    return parser


def main() -> None:
    args = _parser().parse_args()
    config = SocialOmniEvalConfig(**vars(args))
    wait_for_service(config.base_url, timeout=config.timeout_s)
    output = asyncio.run(run_socialomni(config))
    commit = output["provenance"]["repository"]["commit"] or "unknown"
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    path = save_json_results(
        output, config.output_dir, f"socialomni-{commit[:12]}-{stamp}.json"
    )
    print(
        json.dumps(
            {
                "status": output["summary"]["status"],
                "result": path,
            }
        )
    )
    if output["summary"]["status"] != "complete":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
