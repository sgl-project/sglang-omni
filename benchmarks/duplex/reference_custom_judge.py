# SPDX-License-Identifier: Apache-2.0
"""Run a separately identified custom judge over immutable reference transcripts."""

from __future__ import annotations

import copy
import fcntl
import math
import time
from argparse import Namespace
from collections import Counter
from pathlib import Path
from statistics import NormalDist
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue

from benchmarks.duplex.reference_behavior import (
    JudgeTransport,
    Sleep,
    behavior_units,
    build_request,
    openai_transport,
    run_judgment,
)
from benchmarks.duplex.reference_core import (
    AUDIO_FILES,
    C_LABELS,
    REFERENCE_REVISION,
    VARIANTS,
    Engine,
    Progress,
    atomic_write_json,
    canonical_hash,
    read_json,
    record_identity,
    selected,
    sha256_file,
    utc_now,
)
from benchmarks.duplex.reference_source import ReferenceBehavior, load_official_behavior


class CustomDecoding(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)

    temperature: float = Field(ge=0, le=2)
    top_p: float = Field(gt=0, le=1)
    top_k: int = Field(ge=-1)
    min_p: float = Field(ge=0, le=1)
    repetition_penalty: float = Field(gt=0)
    max_tokens: int = Field(gt=0)


class CustomJudgeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    model_id: str = Field(min_length=1)
    model_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    tokenizer_id: str = Field(min_length=1)
    tokenizer_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    served_model: str = Field(min_length=1)
    precision: Literal["bf16"]
    enable_thinking: Literal[False]
    decoding: CustomDecoding
    seeds: list[int] = Field(min_length=1, max_length=3)
    server_launch_receipt: str = Field(min_length=1)
    server_launch_receipt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


def build_custom_request(
    official: ReferenceBehavior,
    engine: Engine,
    sid: str,
    config: CustomJudgeConfig,
    experiment_hash: str,
) -> dict[str, JsonValue]:
    request = build_request(official, engine.sample_dir(sid))
    body = request["body"]
    body.update(
        model=config.served_model,
        temperature=config.decoding.temperature,
        top_p=config.decoding.top_p,
        max_tokens=config.decoding.max_tokens,
        extra_body={
            "top_k": config.decoding.top_k,
            "min_p": config.decoding.min_p,
            "repetition_penalty": config.decoding.repetition_penalty,
            "chat_template_kwargs": {"enable_thinking": config.enable_thinking},
        },
    )
    identity = {
        "experiment_hash": experiment_hash,
        "engine": engine.name,
        "sample_id": sid,
        "body": body,
        "seeds": config.seeds,
        "transcript_sha256": request["transcript_sha256"],
        "audio_sha256": {
            name: sha256_file(engine.source_audio(sid, name)) for name in AUDIO_FILES
        },
        "asr_receipt_sha256": {
            name: sha256_file(engine.sample_dir(sid) / "receipts" / f"asr-{name}.json")
            for name in ("input", "clean_input", "output", "clean_output")
        },
    }
    return {**identity, "request_hash": canonical_hash(identity)}


def summarize_custom(
    args: Namespace,
    engines: list[Engine],
    states: dict[str, dict[str, dict[str, JsonValue]]],
    experiment: dict[str, JsonValue],
) -> dict[str, JsonValue]:
    summary = {
        "generated_at": utc_now(),
        "scope": "custom_behavior_judge_non_official",
        "experiment_hash": canonical_hash(experiment),
        "experiment": experiment,
        "proportion_denominator": "valid custom labels only",
        "confidence_interval": {
            "method": "Wilson score",
            "confidence": 0.95,
            "note": "dataset sampling uncertainty; not human agreement or repeated-generation variance",
        },
        "engines": {},
    }
    for engine in engines:
        sample_ids = selected(engine, args.only)
        groups = {"all": sample_ids}
        for sample_id in sample_ids:
            groups.setdefault(engine.samples[sample_id]["category"], []).append(
                sample_id
            )
        summary["engines"][engine.name] = {}
        for group, members in sorted(groups.items()):
            sample_states = [states[engine.name][sample_id] for sample_id in members]
            statuses = Counter(sample_state["status"] for sample_state in sample_states)
            labels = Counter(
                sample_state["label"]
                for sample_state in sample_states
                if sample_state["status"] == "valid"
            )
            valid_count = sum(labels.values())
            proportions = {}
            z_squared = NormalDist().inv_cdf(0.975) ** 2
            for label in C_LABELS:
                if valid_count:
                    proportion = labels[label] / valid_count
                    denominator = 1 + z_squared / valid_count
                    center = (proportion + z_squared / (2 * valid_count)) / denominator
                    margin = (
                        math.sqrt(
                            z_squared * proportion * (1 - proportion) / valid_count
                            + z_squared**2 / (4 * valid_count**2)
                        )
                        / denominator
                    )
                    ci95 = [max(0.0, center - margin), min(1.0, center + margin)]
                else:
                    proportion, ci95 = None, None
                proportions[label] = {
                    "count": labels[label],
                    "proportion": proportion,
                    "ci95": ci95,
                }
            reasons = Counter(
                reason
                for sample_id in members
                for variant in engine.samples[sample_id]["variants"].values()
                if not variant["eligible"]
                for reason in variant["reasons"]
            )
            summary["engines"][engine.name][group] = {
                "selected_pairs": len(members),
                "selected_sessions": len(members) * len(VARIANTS),
                "eligible_sessions": sum(
                    engine.eligible(sample_id, variant)
                    for sample_id in members
                    for variant in VARIANTS
                ),
                "eligible_pairs": sum(
                    sample_state["eligible"] for sample_state in sample_states
                ),
                "asr_ready_pairs": sum(
                    sample_state["asr_ready"] for sample_state in sample_states
                ),
                "attempted_pairs": sum(
                    sample_state["attempts"] > 0 for sample_state in sample_states
                ),
                "attempts": sum(
                    sample_state["attempts"] for sample_state in sample_states
                ),
                "status": dict(sorted(statuses.items())),
                "ineligible_variant_reasons": dict(sorted(reasons.items())),
                "valid_n": valid_count,
                "valid_label_proportions": proportions,
            }
    atomic_write_json(args.out / "summary.json", summary)
    atomic_write_json(args.out / "sample-status.json", states)
    return summary


def run_custom(
    args: Namespace,
    engines: list[Engine],
    paths: dict[str, Path],
    transport: JudgeTransport | None = None,
    sleep: Sleep = time.sleep,
) -> Counter[str]:
    for source in (args.source_scores, *(engine.tree for engine in engines)):
        if args.out.resolve().is_relative_to(
            source.resolve()
        ) or source.resolve().is_relative_to(args.out.resolve()):
            raise SystemExit(
                "custom --out must be independent of source scores and audio"
            )
        else:
            pass
    config = CustomJudgeConfig.model_validate(read_json(args.judge_config))
    launch_path = (args.judge_config.parent / config.server_launch_receipt).resolve()
    if sha256_file(launch_path) != config.server_launch_receipt_sha256:
        raise SystemExit("server launch receipt differs from the pinned SHA-256")
    else:
        pass
    official = load_official_behavior(paths["behavior"], paths["instruction"])
    custom_behavior = copy.copy(official)
    custom_behavior.model = config.served_model
    experiment = {
        "scope": "custom_behavior_judge_non_official",
        "reference_revision": REFERENCE_REVISION,
        "reference_files": {name: sha256_file(path) for name, path in paths.items()},
        "custom_judge_sha256": sha256_file(Path(__file__)),
        "source_scores": str(args.source_scores.resolve()),
        "source_receipts": {
            engine.name: read_json(engine.root / "manifest-receipt.json")
            for engine in engines
        },
        "selected_samples": {
            engine.name: selected(engine, args.only) for engine in engines
        },
        "config": config.model_dump(),
        "server_launch_receipt": read_json(launch_path),
    }
    experiment_hash = canonical_hash(experiment)
    args.out.mkdir(parents=True, exist_ok=True)
    with open(args.out / ".lock", "a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit(
                "another custom judge phase owns this output directory"
            ) from None
        identity_path = args.out / "experiment.json"
        if identity_path.exists():
            if read_json(identity_path) != experiment:
                raise SystemExit("custom judge identity changed; use a new --out")
            else:
                pass
        elif any(path.name != ".lock" for path in args.out.iterdir()):
            raise SystemExit(
                "custom judge requires a new --out or matching experiment.json"
            )
        else:
            atomic_write_json(identity_path, experiment)
        states, pending_requests = {}, []
        for engine in engines:
            ready, blocked = behavior_units(engine, args.only)
            states[engine.name] = {}
            for sample_id in selected(engine, args.only):
                folder = args.out / "engines" / engine.name / "samples" / sample_id
                result_path = folder / "result.json"
                previous_result = (
                    read_json(result_path) if result_path.exists() else None
                )
                sample_state = {
                    "eligible": all(
                        engine.eligible(sample_id, variant) for variant in VARIANTS
                    ),
                    "asr_ready": sample_id in ready,
                    "attempts": (
                        len(previous_result["attempts"]) if previous_result else 0
                    ),
                    "label": None,
                    "status": blocked.get(sample_id, "not_judged"),
                }
                states[engine.name][sample_id] = sample_state
                if sample_id in blocked:
                    continue
                else:
                    pass
                request = build_custom_request(
                    official, engine, sample_id, config, experiment_hash
                )
                request_path = folder / "request.json"
                if request_path.exists():
                    if read_json(request_path) != request:
                        sample_state["status"] = "stale_request"
                        continue
                    else:
                        pass
                else:
                    atomic_write_json(request_path, request)
                if previous_result is not None:
                    if previous_result["request_hash"] != request["request_hash"]:
                        sample_state["status"] = "result_request_mismatch"
                        continue
                    else:
                        pass
                    sample_state.update(
                        status=previous_result["status"], label=previous_result["label"]
                    )
                    if previous_result["status"] != "failed" or not args.retry_failed:
                        continue
                    else:
                        pass
                else:
                    pass
                pending_requests.append(
                    (result_path, request, previous_result, sample_state)
                )
        if args.phase == "custom-judge":
            pending_requests = pending_requests[: args.limit]
            progress = Progress(args.out, "custom-judge", len(pending_requests))
            record_identity(
                args.out,
                "custom-judge",
                {
                    "experiment_hash": experiment_hash,
                    "base_url": args.base_url,
                    "timeout_s": args.timeout_s,
                    "retry_sleep_s": args.retry_sleep_s,
                    "sdk_max_retries": 0,
                },
            )
            if pending_requests:
                transport = transport or openai_transport(
                    args.api_key_env, args.timeout_s, args.base_url
                )
            else:
                pass
            for result_path, request, previous_result, sample_state in pending_requests:
                result = run_judgment(
                    custom_behavior,
                    request["body"],
                    request["seeds"],
                    transport,
                    args.retry_sleep_s,
                    sleep,
                )
                if result["status"] in ("valid", "invalid_label"):
                    completion_choice = result["attempts"][-1]["response"]["choices"][0]
                    if completion_choice.get("finish_reason") != "stop":
                        result.update(status="invalid_finish", label=None)
                    else:
                        pass
                else:
                    pass
                result.update(
                    request_hash=request["request_hash"], finished_at=utc_now()
                )
                if previous_result:
                    result["attempts"] = (
                        previous_result["attempts"] + result["attempts"]
                    )
                else:
                    pass
                atomic_write_json(result_path, result)
                sample_state.update(
                    status=result["status"],
                    label=result["label"],
                    attempts=len(result["attempts"]),
                )
                progress.add(result["status"])
            progress.write(finished=True)
        else:
            pass
        summary = summarize_custom(args, engines, states, experiment)
        counts: Counter[str] = Counter()
        for engine in summary["engines"].values():
            counts.update(engine["all"]["status"])
        return counts
