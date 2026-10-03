# SPDX-License-Identifier: Apache-2.0
"""Prepare exact reference judge requests and retain bounded API outcomes."""

from __future__ import annotations

import json
import os
import time
from argparse import Namespace
from collections import Counter
from pathlib import Path
from typing import Protocol

from pydantic import JsonValue

from benchmarks.duplex.reference_core import (
    C_LABELS,
    JUDGE_MAX_ATTEMPTS,
    JUDGE_MODEL,
    REFERENCE_FILES,
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


class JudgeTransport(Protocol):
    def __call__(
        self, body: dict[str, JsonValue], seed: int
    ) -> dict[str, JsonValue]: ...


class Sleep(Protocol):
    def __call__(self, seconds: float) -> None: ...


def build_request(official: ReferenceBehavior, sample: Path) -> dict[str, JsonValue]:
    """Exact official payload from the four official transcript JSON files."""
    transcripts = {}
    for name in ("input_clean", "input_noisy", "output_clean", "output_noisy"):
        file_name = {
            "input_clean": "clean_input.json",
            "input_noisy": "input.json",
            "output_clean": "clean_output.json",
            "output_noisy": "output.json",
        }[name]
        with open(sample / file_name, "r") as file_handle:
            transcripts[name] = json.load(file_handle)
    user_message = official.template(
        *(
            official.json_dict_to_compact_text(transcripts[transcript_role])
            for transcript_role in (
                "input_clean",
                "input_noisy",
                "output_clean",
                "output_noisy",
            )
        )
    )
    body = {
        "model": official.model,
        "messages": [
            {"role": "system", "content": official.instruction},
            {"role": "user", "content": user_message},
        ],
    }
    seeds = [official.initial_seed + i for i in range(JUDGE_MAX_ATTEMPTS)]
    return {
        "body": body,
        "seeds": seeds,
        "transcript_sha256": {
            file_name: sha256_file(sample / file_name)
            for file_name in (
                "input.json",
                "clean_input.json",
                "output.json",
                "clean_output.json",
            )
        },
        "request_hash": canonical_hash(
            {
                "body": body,
                "seeds": seeds,
                "instruction_sha256": REFERENCE_FILES["instruction"][1],
                "eval_behavior_sha256": REFERENCE_FILES["behavior"][1],
            }
        ),
    }


def behavior_units(engine: Engine, only: list[str]) -> tuple[list[str], dict[str, str]]:
    ready, blocked = [], {}
    for sample_id in selected(engine, only):
        if not all(engine.eligible(sample_id, variant) for variant in VARIANTS):
            blocked[sample_id] = "variant_ineligible"
            continue
        else:
            pass
        receipt_directory = engine.sample_dir(sample_id) / "receipts"
        receipts = {
            stem: (
                read_json(receipt_directory / f"asr-{stem}.json")
                if (receipt_directory / f"asr-{stem}.json").exists()
                else {"status": "not_run"}
            )
            for stem in ("input", "clean_input", "output", "clean_output")
        }
        states = [receipt["status"] for receipt in receipts.values()]
        if all(status == "ok" for status in states):
            for stem, receipt in receipts.items():
                transcript = engine.sample_dir(sample_id) / f"{stem}.json"
                if sha256_file(transcript) != receipt["transcript_sha256"]:
                    blocked[sample_id] = "stale_request"
                else:
                    pass
                if (
                    sha256_file(engine.source_audio(sample_id, f"{stem}.wav"))
                    != receipt["audio_sha256"]
                ):
                    blocked[sample_id] = "asr_audio_changed"
                else:
                    pass
            if sample_id not in blocked:
                ready.append(sample_id)
            else:
                pass
        else:
            blocked[sample_id] = "asr_" + next(
                status for status in states if status != "ok"
            )
    return ready, blocked


def run_prepare_judge(
    args: Namespace, engines: list[Engine], paths: dict[str, Path]
) -> Counter[str]:
    official = load_official_behavior(paths["behavior"], paths["instruction"])
    counts: Counter[str] = Counter()
    for engine in engines:
        ready, blocked = behavior_units(engine, args.only)
        counts.update(f"blocked_{reason}" for reason in blocked.values())
        for sample_id in ready:
            path = engine.sample_dir(sample_id) / "judge" / "request.json"
            request = build_request(official, engine.sample_dir(sample_id))
            if path.exists():
                old = read_json(path)
                if old["request_hash"] != request["request_hash"]:
                    raise SystemExit(
                        f"{path}: prepared request differs from current transcripts"
                    )
                else:
                    pass
                counts["reused"] += 1
                continue
            else:
                pass
            atomic_write_json(path, {**request, "prepared_at": utc_now()})
            counts["prepared"] += 1
    record_identity(
        args.out, "prepare-judge", {"judge_model": JUDGE_MODEL, "counts": dict(counts)}
    )
    return counts


def run_judgment(
    official: ReferenceBehavior,
    body: dict[str, JsonValue],
    seeds: list[int],
    transport: JudgeTransport,
    sleep_s: float,
    sleep: Sleep = time.sleep,
) -> dict[str, JsonValue]:
    """Official eval_behavior loop (seed += 1 per exception), bounded to len(seeds) attempts."""
    attempts = []
    for i, seed in enumerate(seeds):
        attempt = {"seed": seed, "started_at": utc_now()}
        attempts.append(attempt)
        try:
            response = transport(body, seed)
            attempt["response"] = response
            prediction = response["choices"][0]["message"]["content"]
            result = official.parse_eval(prediction)
        except Exception as exc:
            attempt["error"] = f"{type(exc).__name__}: {exc}"
            if i + 1 < len(seeds):
                sleep(sleep_s)
            else:
                pass
            continue
        served_model = response.get("model")
        labels = result.get("behaviour") if isinstance(result, dict) else None
        if served_model != official.model:
            status = "model_mismatch"
        elif isinstance(labels, list) and len(labels) == 1 and labels[0] in C_LABELS:
            status = "valid"
        else:
            status = "invalid_label"
        return {
            "status": status,
            "parsed": result,
            "label": labels[0] if status == "valid" else None,
            "served_model": served_model,
            "system_fingerprint": response.get("system_fingerprint"),
            "attempts": attempts,
        }
    return {"status": "failed", "parsed": None, "label": None, "attempts": attempts}


def openai_transport(
    api_key_env: str, timeout_s: float, base_url: str | None = None
) -> JudgeTransport:
    from openai import OpenAI

    key = os.environ.get(api_key_env)
    if not key:
        raise SystemExit(f"judge needs credentials in ${api_key_env}")
    else:
        pass
    client = OpenAI(api_key=key, base_url=base_url, max_retries=0, timeout=timeout_s)
    keep = (
        "id",
        "object",
        "created",
        "model",
        "system_fingerprint",
        "service_tier",
        "choices",
        "usage",
    )

    def transport(body: dict[str, JsonValue], seed: int) -> dict[str, JsonValue]:
        try:
            response = client.chat.completions.create(**body, seed=seed).model_dump(
                mode="json"
            )
        except Exception as exc:
            status = getattr(exc, "status_code", None)
            raise RuntimeError(
                f"{type(exc).__name__} status={status}: {str(exc).replace(key, '[REDACTED]')}"
            ) from None
        return {field_name: response.get(field_name) for field_name in keep}

    return transport


def run_judge(
    args: Namespace,
    engines: list[Engine],
    paths: dict[str, Path],
    transport: JudgeTransport | None = None,
    sleep: Sleep = time.sleep,
) -> Counter[str]:
    if args.judge != JUDGE_MODEL:
        raise SystemExit(
            f"--judge must be exactly {JUDGE_MODEL}; other judges are a separate experiment"
        )
    else:
        pass
    official = load_official_behavior(paths["behavior"], paths["instruction"])
    pending_requests, counts = [], Counter()
    for engine in engines:
        _, blocked = behavior_units(engine, args.only)
        for sample_id in selected(engine, args.only):
            sample = engine.sample_dir(sample_id)
            request_path, result_path = (
                sample / "judge" / "request.json",
                sample / "judge" / "result.json",
            )
            if not request_path.exists():
                continue
            else:
                pass
            if sample_id in blocked:
                counts[blocked[sample_id]] += 1
                continue
            else:
                pass
            prepared = read_json(request_path)
            if (
                build_request(official, sample)["request_hash"]
                != prepared["request_hash"]
            ):
                counts["stale_request"] += 1
                atomic_write_json(
                    sample / "judge" / "stale.json",
                    {"at": utc_now(), "request": str(request_path)},
                )
                continue
            else:
                pass
            if result_path.exists():
                old = read_json(result_path)
                if old["request_hash"] != prepared["request_hash"]:
                    raise SystemExit(
                        f"{result_path}: result belongs to another request"
                    )
                else:
                    pass
                if old["status"] != "failed" or not args.retry_failed:
                    counts["reused"] += 1
                    if old["status"] != "valid":
                        counts[f"reused_{old['status']}"] += 1
                    else:
                        pass
                    continue
                else:
                    pass
            else:
                pass
            pending_requests.append((sample, prepared))
    pending_requests = pending_requests[
        : args.max_requests if args.max_requests is not None else args.limit
    ]
    progress = Progress(args.out, "judge", len(pending_requests))
    if pending_requests:
        record_identity(
            args.out,
            "judge",
            {
                "judge_model": JUDGE_MODEL,
                "max_attempts": JUDGE_MAX_ATTEMPTS,
                "retry_sleep_s": args.retry_sleep_s,
                "timeout_s": args.timeout_s,
                "sdk_max_retries": 0,
                "requests": len(pending_requests),
            },
        )
        transport = transport or openai_transport(
            args.api_key_env, args.timeout_s, args.base_url
        )
    else:
        pass
    for sample, prepared in pending_requests:
        result = run_judgment(
            official,
            prepared["body"],
            prepared["seeds"],
            transport,
            args.retry_sleep_s,
            sleep,
        )
        result.update(request_hash=prepared["request_hash"], finished_at=utc_now())
        atomic_write_json(sample / "judge" / "result.json", result)
        if result["parsed"] is not None:
            atomic_write_json(sample / "content_tag.json", result["parsed"])
        else:
            pass
        counts[result["status"]] += 1
        progress.add(result["status"])
    progress.write(finished=True)
    return counts
