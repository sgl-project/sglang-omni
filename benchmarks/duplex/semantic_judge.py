# SPDX-License-Identifier: Apache-2.0
"""Three-axis semantic judge (accept/fail/unresolved) over reference transcripts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Literal

from openai import OpenAI
from pydantic import BaseModel, ConfigDict, Field, JsonValue, ValidationError

from benchmarks.duplex.reference_core import utc_now
from benchmarks.duplex.report_models import SemanticSummary

ASSETS = Path(__file__).with_name("semantic")
PROMPT_PATH = ASSETS / "judge-prompt.txt"
SCHEMA_PATH = ASSETS / "judge-response.schema.json"
CONTROLS_PATH = ASSETS / "judge-controls.jsonl"
CONTROL_EXPECTED_PATH = ASSETS / "control-expected.json"
AXES = ("interaction_handling", "relevance", "grounding")
EVENT_TEXT_KEYS = {
    "user_interruption": "current_turn_text",
    "talking_to_other": "current_turn_text",
    "user_backchannel": "backchannel_text",
    "background_speech": "background_text",
}
TRANSCRIPT_ROLES = ("input", "clean_input", "output", "clean_output")
NATIVE_TEXT_EVENT = "response.output_audio_transcript.delta"
ACCEPTABLE_OUTCOMES = ["Apply the supplied rubric."]
MAX_BATCH_PAIRS = 8
MAX_ATTEMPTS = 3

AxisStatus = Literal["accept", "fail", "unresolved"]
EvidenceSource = Literal[
    "output_asr_text",
    "native_output_text",
    "input_asr_text",
    "original_task",
    "event_text",
    "clean_output_asr_text",
    "none",
]
AxisCounts = dict[str, int]
Packet = dict[str, JsonValue]


class AxisJudgment(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    status: AxisStatus
    reason: str = Field(max_length=350)
    source: EvidenceSource
    quote: str = Field(max_length=220)


class AxisOutcome(BaseModel):
    status: AxisStatus
    judgment: AxisJudgment | None = None
    conversion_reason: str | None = None


class JudgeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    model_id: str
    model_revision: str
    served_model: str
    base_url: str
    interface: str = (
        "OpenAI-compatible chat completions with JSON-schema constrained output"
    )
    enable_thinking: bool
    temperature: float
    top_p: float
    max_tokens: int
    seed: int
    batch_pairs: int = MAX_BATCH_PAIRS
    max_attempts: int = MAX_ATTEMPTS
    retry_policy: str = (
        "retry only transport errors, non-stop finishes and unparseable JSON; "
        "a parseable response is final"
    )
    prompt_sha256: str
    schema_sha256: str
    controls_sha256: str


class PopulationRow(BaseModel):
    sample_id: str
    original_sample_id: str
    category: str
    eligible: bool
    exclusion_reasons: list[str]


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> JsonValue:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, document: JsonValue) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(document, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def native_output_text(transcript_path: Path, window_end_s: float) -> str:
    if not transcript_path.is_file():
        return ""
    else:
        pass
    events = read_json(transcript_path)["events"]
    return "".join(
        event["delta"]
        for event in events
        if event["type"] == NATIVE_TEXT_EVENT and event["receipt_s"] <= window_end_s
    ).strip()


def asr_problems(sample_scores: Path, sample_audio: Path) -> list[str]:
    problems = []
    for role in TRANSCRIPT_ROLES:
        receipt_path = sample_scores / "receipts" / f"asr-{role}.json"
        if not receipt_path.is_file():
            problems.append(f"asr-{role} receipt missing")
            continue
        else:
            pass
        receipt = read_json(receipt_path)
        transcript_path = sample_scores / f"{role}.json"
        if receipt["status"] != "ok":
            problems.append(f"asr-{role} status {receipt['status']}")
        elif file_sha256(sample_audio / f"{role}.wav") != receipt["audio_sha256"]:
            problems.append(f"asr-{role} audio sha256 differs from receipt")
        elif file_sha256(transcript_path) != receipt["transcript_sha256"]:
            problems.append(f"asr-{role} transcript sha256 differs from receipt")
        else:
            pass
    return problems


def build_packet(
    sample: dict[str, JsonValue],
    opaque_id: str,
    sample_scores: Path,
    dataset_root: Path,
) -> Packet:
    category = sample["category"]
    metadata = read_json(dataset_root / sample["sample_id"] / "metadata.json")
    overlap = sample["variants"]["overlap"]
    window_end_s = overlap["window"]["T_s"]
    output = read_json(sample_scores / "output.json")
    return {
        "sample_id": opaque_id,
        "category": category,
        "original_task": metadata["context_text"],
        "event_text": metadata[EVENT_TEXT_KEYS[category]],
        "event_s": metadata["timestamps"],
        "window_s": [0, window_end_s],
        "input_asr_text": read_json(sample_scores / "input.json")["text"],
        "clean_input_asr_text": read_json(sample_scores / "clean_input.json")["text"],
        "output_asr_text": output["text"],
        "output_asr_segments": output["chunks"],
        "native_output_text": native_output_text(
            Path(overlap["source"]["directory"]) / "transcript.json", window_end_s
        ),
        "clean_output_asr_text": read_json(sample_scores / "clean_output.json")["text"],
        "boundary": overlap["boundary"],
        "available_context_and_tool_results": [],
        "acceptable_outcomes": ACCEPTABLE_OUTCOMES,
    }


def build_inputs(
    reference_audio: Path, engine_scores: Path, dataset_root: Path
) -> tuple[list[PopulationRow], list[Packet], dict[str, JsonValue]]:
    """Every manifest pair gets a population row; only assessable pairs get a packet."""
    manifest_path = reference_audio / "reference-manifest.json"
    samples = sorted(
        read_json(manifest_path)["samples"], key=lambda row: row["sample_id"]
    )
    population, packets, source_hashes = [], [], {}
    for index, sample in enumerate(samples, start=1):
        original_id = sample["sample_id"]
        opaque_id = f"p{index:04d}"
        reasons = [
            f"{variant}: {reason}"
            for variant, state in sample["variants"].items()
            for reason in state["reasons"]
        ]
        sample_scores = engine_scores / "samples" / original_id
        if not reasons:
            reasons = asr_problems(sample_scores, reference_audio / original_id)
        else:
            pass
        population.append(
            PopulationRow(
                sample_id=opaque_id,
                original_sample_id=original_id,
                category=sample["category"],
                eligible=not reasons,
                exclusion_reasons=reasons,
            )
        )
        if reasons:
            continue
        else:
            pass
        packets.append(build_packet(sample, opaque_id, sample_scores, dataset_root))
        source_hashes[original_id] = {
            **{
                f"{role}.wav": file_sha256(
                    reference_audio / original_id / f"{role}.wav"
                )
                for role in TRANSCRIPT_ROLES
            },
            **{
                f"{role}.json": file_sha256(sample_scores / f"{role}.json")
                for role in TRANSCRIPT_ROLES
            },
            **{
                f"asr-{role}.json": file_sha256(
                    sample_scores / "receipts" / f"asr-{role}.json"
                )
                for role in TRANSCRIPT_ROLES
            },
        }
    receipt = {
        "reference_manifest_sha256": file_sha256(manifest_path),
        "dataset_root": str(dataset_root),
        "sources": source_hashes,
    }
    return population, packets, receipt


def request_body(
    config: JudgeConfig, prompt: str, schema: JsonValue, packets: list[Packet]
) -> dict[str, JsonValue]:
    user_message = (
        "Response JSON schema:\n"
        + json.dumps(schema)
        + "\n\nCases, in input order:\n"
        + "\n".join(json.dumps(packet, ensure_ascii=False) for packet in packets)
    )
    return {
        "model": config.served_model,
        "messages": [
            {"role": "system", "content": prompt},
            {"role": "user", "content": user_message},
        ],
        "temperature": config.temperature,
        "top_p": config.top_p,
        "max_tokens": config.max_tokens,
        "seed": config.seed,
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "semantic_assessments", "schema": schema},
        },
        "extra_body": {
            "chat_template_kwargs": {"enable_thinking": config.enable_thinking}
        },
    }


def submit(
    client: OpenAI, body: dict[str, JsonValue], max_attempts: int
) -> dict[str, JsonValue]:
    """Return the attempt log; the last attempt holds the final content or error."""
    attempts = []
    for _ in range(max_attempts):
        attempt = {"started_at": utc_now()}
        attempts.append(attempt)
        try:
            response = client.chat.completions.create(**body).model_dump(
                mode="json", warnings=False
            )
        except Exception as exc:
            attempt["error"] = f"{type(exc).__name__}: {exc}"
            continue
        attempt["response"] = response
        choice = response["choices"][0]
        if choice["finish_reason"] != "stop":
            attempt["error"] = f"finish_reason {choice['finish_reason']}"
            continue
        else:
            pass
        try:
            content = json.loads(choice["message"]["content"])
        except (TypeError, ValueError) as exc:
            attempt["error"] = f"unparseable JSON: {exc}"
            continue
        if isinstance(content, dict) and isinstance(content.get("assessments"), list):
            return {"status": "returned", "content": content, "attempts": attempts}
        else:
            attempt["error"] = "response has no assessments array"
    return {"status": "failed", "content": None, "attempts": attempts}


def unresolved(reason: str) -> dict[str, AxisOutcome]:
    return {
        axis: AxisOutcome(status="unresolved", conversion_reason=reason)
        for axis in AXES
    }


def validate_axis(raw_axis: JsonValue, packet: Packet) -> AxisOutcome:
    try:
        judgment = AxisJudgment.model_validate(raw_axis)
    except ValidationError as exc:
        return AxisOutcome(
            status="unresolved", conversion_reason=f"schema: {exc.errors()[0]['msg']}"
        )
    if judgment.source == "none":
        if judgment.status != "unresolved" or judgment.quote:
            reason = "source none is valid only for unresolved with an empty quote"
            return AxisOutcome(
                status="unresolved", judgment=judgment, conversion_reason=reason
            )
        else:
            return AxisOutcome(status="unresolved", judgment=judgment)
    elif not judgment.quote:
        if judgment.status == "unresolved":
            return AxisOutcome(status="unresolved", judgment=judgment)
        else:
            reason = f"{judgment.status} verdict has no evidence quote"
            return AxisOutcome(
                status="unresolved", judgment=judgment, conversion_reason=reason
            )
    elif judgment.quote not in str(packet[judgment.source]):
        reason = f"quote is not an exact substring of {judgment.source}"
        return AxisOutcome(
            status="unresolved", judgment=judgment, conversion_reason=reason
        )
    else:
        return AxisOutcome(status=judgment.status, judgment=judgment)


def validate_response(
    submission: dict[str, JsonValue], packets: list[Packet]
) -> dict[str, dict[str, AxisOutcome]]:
    by_id = {packet["sample_id"]: packet for packet in packets}
    outcomes = {sample_id: unresolved("no assessment returned") for sample_id in by_id}
    if submission["status"] != "returned":
        return {sample_id: unresolved("batch response failed") for sample_id in by_id}
    else:
        pass
    returned_ids = [
        item.get("sample_id") if isinstance(item, dict) else None
        for item in submission["content"]["assessments"]
    ]
    for item, sample_id in zip(submission["content"]["assessments"], returned_ids):
        if sample_id not in by_id:
            continue
        elif returned_ids.count(sample_id) > 1:
            outcomes[sample_id] = unresolved("duplicate sample_id in response")
        elif not isinstance(item.get("axes"), dict) or set(item) != {
            "sample_id",
            "axes",
        }:
            outcomes[sample_id] = unresolved("assessment does not match the schema")
        else:
            outcomes[sample_id] = {
                axis: (
                    validate_axis(item["axes"][axis], by_id[sample_id])
                    if axis in item["axes"]
                    else AxisOutcome(
                        status="unresolved", conversion_reason="axis missing"
                    )
                )
                for axis in AXES
            }
    return outcomes


def joint_status(axes: dict[str, AxisOutcome]) -> AxisStatus:
    statuses = {outcome.status for outcome in axes.values()}
    if "fail" in statuses:
        return "fail"
    elif statuses == {"accept"}:
        return "accept"
    else:
        return "unresolved"


def run_controls(
    client: OpenAI, config: JudgeConfig, prompt: str, schema: JsonValue, out: Path
) -> bool:
    controls = [
        json.loads(line) for line in CONTROLS_PATH.read_text().splitlines() if line
    ]
    expected = {
        item["sample_id"]: {axis: item["axes"][axis]["status"] for axis in AXES}
        for item in read_json(CONTROL_EXPECTED_PATH)["assessments"]
    }
    body = request_body(config, prompt, schema, controls)
    write_json(out / "controls" / "request.json", body)
    submission = submit(client, body, config.max_attempts)
    write_json(out / "controls" / "response.json", submission)
    outcomes = validate_response(submission, controls)
    comparison = {
        sample_id: {
            axis: {"expected": expected[sample_id][axis], "observed": axes[axis].status}
            for axis in AXES
        }
        for sample_id, axes in outcomes.items()
    }
    mismatches = sum(
        cell["expected"] != cell["observed"]
        for axes in comparison.values()
        for cell in axes.values()
    )
    write_json(
        out / "controls" / "comparison.json",
        {"mismatches": mismatches, "cases": comparison},
    )
    print(
        f"controls: {len(AXES) * len(controls) - mismatches}/{len(AXES) * len(controls)} axis statuses match"
    )
    for sample_id, axes in comparison.items():
        for axis, cell in axes.items():
            if cell["expected"] != cell["observed"]:
                print(
                    f"  {sample_id} {axis}: expected {cell['expected']}, got {cell['observed']}"
                )
            else:
                pass
    return mismatches == 0


def grade_batch(
    client: OpenAI,
    config: JudgeConfig,
    prompt: str,
    schema: JsonValue,
    batch_dir: Path,
    packets: list[Packet],
) -> dict[str, dict[str, AxisOutcome]]:
    assessments_path = batch_dir / "assessments.json"
    if assessments_path.is_file():
        saved = read_json(assessments_path)
        return {
            sample_id: {
                axis: AxisOutcome.model_validate(saved[sample_id][axis])
                for axis in AXES
            }
            for sample_id in saved
        }
    else:
        pass
    body = request_body(config, prompt, schema, packets)
    write_json(batch_dir / "request.json", body)
    submission = submit(client, body, config.max_attempts)
    write_json(batch_dir / "response.json", submission)
    outcomes = validate_response(submission, packets)
    write_json(
        assessments_path,
        {
            sample_id: {
                axis: outcome.model_dump(mode="json") for axis, outcome in axes.items()
            }
            for sample_id, axes in outcomes.items()
        },
    )
    return outcomes


def axis_counts(statuses: list[AxisStatus]) -> AxisCounts:
    return {
        "selected": len(statuses),
        "accepted": statuses.count("accept"),
        "failed": statuses.count("fail"),
        "unresolved": statuses.count("unresolved"),
    }


def percentages(counts: AxisCounts) -> dict[str, JsonValue]:
    resolved = counts["accepted"] + counts["failed"]
    selected = counts["selected"]
    return {
        **counts,
        "quality_percent": 100 * counts["accepted"] / resolved if resolved else None,
        "coverage_percent": 100 * resolved / selected if selected else None,
        "unresolved_bounds_percent": (
            [
                100 * counts["accepted"] / selected,
                100 * (counts["accepted"] + counts["unresolved"]) / selected,
            ]
            if selected
            else None
        ),
    }


def summarize(
    rows: list[dict[str, JsonValue]],
    config: JudgeConfig,
    inputs_sha256: str,
    submitted: int,
    returned: int,
) -> dict[str, JsonValue]:
    groups = {"all": rows}
    for row in rows:
        groups.setdefault(row["category"], []).append(row)
    categories = {
        group: {
            **{
                axis: percentages(axis_counts([row[axis] for row in members]))
                for axis in AXES
            },
            "joint": percentages(axis_counts([row["joint"] for row in members])),
        }
        for group, members in sorted(groups.items())
    }
    eligible = sum(row["eligible"] for row in rows)
    return {
        "scope": (
            f"Custom semantic A/F/U judge {config.model_id}@{config.model_revision[:12]} "
            f"via {config.interface}; thinking={config.enable_thinking}, "
            f"temperature={config.temperature}; transcript-supported semantics only"
        ),
        "inputs_sha256": inputs_sha256,
        "overall_axes": {
            axis: {
                key: categories["all"][axis][key]
                for key in ("selected", "accepted", "failed", "unresolved")
            }
            for axis in AXES
        },
        "overall_joint": {
            key: categories["all"]["joint"][key]
            for key in ("selected", "accepted", "failed", "unresolved")
        },
        "uncertainty_note": (
            f"{len(rows)} selected, {eligible} assessable, {submitted} submitted, {returned} returned. "
            "One generation per pair; the six authored controls are a rubric check, "
            "not human calibration or judge accuracy."
        ),
        "categories": categories,
    }


def load_config(args: argparse.Namespace) -> JudgeConfig:
    return JudgeConfig(
        model_id=args.model_id,
        model_revision=args.model_revision,
        served_model=args.served_model,
        base_url=args.base_url,
        enable_thinking=args.enable_thinking,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        seed=args.seed,
        prompt_sha256=file_sha256(PROMPT_PATH),
        schema_sha256=file_sha256(SCHEMA_PATH),
        controls_sha256=file_sha256(CONTROLS_PATH),
    )


def freeze(path: Path, document: JsonValue, what: str) -> None:
    if path.is_file() and read_json(path) != document:
        raise SystemExit(
            f"{path}: {what} changed since this directory was created; use a new --out"
        )
    else:
        write_json(path, document)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-audio", type=Path, required=True)
    parser.add_argument(
        "--scores", type=Path, required=True, help="Reference scoring root"
    )
    parser.add_argument("--engine", required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--api-key-env", default="CUSTOM_JUDGE_API_KEY")
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--served-model", required=True)
    parser.add_argument(
        "--enable-thinking", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=32768)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--controls-only", action="store_true")
    args = parser.parse_args(argv)

    config = load_config(args)
    prompt = PROMPT_PATH.read_text(encoding="utf-8")
    schema = read_json(SCHEMA_PATH)
    client = OpenAI(
        api_key=os.environ.get(args.api_key_env, "EMPTY"),
        base_url=args.base_url,
        max_retries=0,
        timeout=1800,
    )
    args.out.mkdir(parents=True, exist_ok=True)
    freeze(
        args.out / "judge-config.json",
        config.model_dump(mode="json"),
        "judge configuration",
    )
    if not run_controls(client, config, prompt, schema, args.out):
        print(
            "Control check failed: this judge configuration is not accepted.",
            file=sys.stderr,
        )
        return 1
    elif args.controls_only:
        return 0
    else:
        pass

    population, packets, input_receipt = build_inputs(
        args.reference_audio, args.scores / "engines" / args.engine, args.dataset_root
    )
    freeze(
        args.out / "population.json",
        [row.model_dump() for row in population],
        "population",
    )
    freeze(args.out / "inputs.json", packets, "judge inputs")
    inputs_sha256 = file_sha256(args.out / "inputs.json")
    freeze(
        args.out / "input-receipt.json",
        {**input_receipt, "inputs_sha256": inputs_sha256},
        "input receipt",
    )

    batches = [
        packets[start : start + config.batch_pairs]
        for start in range(0, len(packets), config.batch_pairs)
    ]
    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        results = list(
            executor.map(
                lambda numbered: grade_batch(
                    client,
                    config,
                    prompt,
                    schema,
                    args.out / "batches" / f"b{numbered[0]:03d}",
                    numbered[1],
                ),
                enumerate(batches, start=1),
            )
        )
    outcomes = {
        sample_id: axes for result in results for sample_id, axes in result.items()
    }
    returned = sum(
        read_json(args.out / "batches" / f"b{index:03d}" / "response.json")["status"]
        == "returned"
        for index in range(1, len(batches) + 1)
    )
    rows = []
    for row in population:
        axes = outcomes.get(
            row.sample_id,
            unresolved("pair not assessable: " + "; ".join(row.exclusion_reasons)),
        )
        rows.append(
            {
                **row.model_dump(),
                **{axis: axes[axis].status for axis in AXES},
                "joint": joint_status(axes),
                "conversions": {
                    axis: axes[axis].conversion_reason
                    for axis in AXES
                    if axes[axis].conversion_reason
                },
            }
        )
    write_json(args.out / "quality-outcomes.json", rows)
    summary = summarize(rows, config, inputs_sha256, len(batches), returned)
    SemanticSummary.model_validate(summary)
    write_json(args.out / "summary.json", summary)
    overall = summary["categories"]["all"]
    for axis in (*AXES, "joint"):
        cell = overall[axis]
        quality = (
            "n/a"
            if cell["quality_percent"] is None
            else f"{cell['quality_percent']:.1f}%"
        )
        print(
            f"{axis}: quality {quality}, coverage {cell['coverage_percent']:.1f}%, "
            f"A/F/U {cell['accepted']}/{cell['failed']}/{cell['unresolved']}"
        )
    print(f"Wrote {args.out / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
else:
    pass
