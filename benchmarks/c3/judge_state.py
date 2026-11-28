# SPDX-License-Identifier: Apache-2.0 AND MIT
# Portions derived from https://github.com/step-out/C3 (Copyright (c) 2025-2026 C3 Benchmark Authors, MIT License; see benchmarks/c3/LICENSE).
"""LLM judge configuration, checkpoint validation, and offline state loading for C3."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    StringConstraints,
    model_validator,
)

from benchmarks.c3.containment import (
    read_bytes_nofollow,
    unlink_contained_file_nofollow,
    write_contained_json_atomic,
)
from benchmarks.c3.http_transport import (
    SAFE_ENV_VAR_NAME_PATTERN,
    compute_endpoint_identity_sha256,
    sanitize_endpoint_origin,
    validate_http_endpoint_url,
)
from benchmarks.c3.prompts import (
    build_c3_judge_messages,
    compute_c3_judge_prompt_sha256,
    parse_c3_judge_verdict,
)
from benchmarks.c3.schemas import (
    JUDGMENT_CHECKPOINT_SCHEMA_VERSION,
    C3JudgeManifestRecord,
    C3JudgmentCheckpointRecord,
    C3PredictionCheckpointRecord,
    C3PublicJudgeManifestEntry,
)
from benchmarks.dataset.c3 import C3InstanceRecord

FORBIDDEN_CREDENTIAL_KEYS = frozenset(
    {
        "api_key",
        "apikey",
        "token",
        "access_token",
        "bearer_token",
        "secret",
        "password",
        "passwd",
        "authorization",
    }
)


ApiKeyEnvName = Annotated[str, StringConstraints(pattern=SAFE_ENV_VAR_NAME_PATTERN)]


class C3JudgeEndpointSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)

    name: str = Field(min_length=1, max_length=128)
    model: str = Field(min_length=1, max_length=256)
    model_revision: str | None = None
    base_url: str
    api_key_env: ApiKeyEnvName | None = None
    max_concurrency: int = Field(default=8, ge=1, le=256)
    enable_thinking: bool | None = None
    temperature: float | None = Field(default=0.0, ge=0.0, le=2.0)
    max_tokens: int = Field(default=4096, ge=16, le=32768)

    @model_validator(mode="after")
    def validate_endpoint_url(self) -> "C3JudgeEndpointSpec":
        validate_http_endpoint_url(self.base_url)
        return self

    def compute_fingerprint_sha256(self) -> str:
        serialized_fingerprint_bytes = json.dumps(
            {
                "name": self.name,
                "model": self.model,
                "model_revision": self.model_revision,
                "endpoint_origin": sanitize_endpoint_origin(self.base_url),
                "endpoint_identity_sha256": compute_endpoint_identity_sha256(
                    self.base_url
                ),
                "enable_thinking": self.enable_thinking,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(serialized_fingerprint_bytes).hexdigest()


class C3JudgeConfigFile(BaseModel):
    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)

    judges: list[C3JudgeEndpointSpec] = Field(min_length=1)


@dataclass(frozen=True, kw_only=True)
class C3PendingJudgeWorkItem:
    instance: C3InstanceRecord
    system_prompt: str
    user_prompt: str
    prompt_sha256: str
    prediction_sha256: str


@dataclass(frozen=True, kw_only=True)
class C3JudgePhaseResult:
    authoritative_judge_names: tuple[str, ...]
    authoritative_public_judges: list[C3PublicJudgeManifestEntry]
    prompt_bundle_sha256: str
    verdicts_by_judge: dict[str, dict[str, bool]]
    failed_judgments_by_judge: dict[str, dict[str, str]]


def load_c3_judge_configs(config_path: Path) -> list[C3JudgeEndpointSpec]:
    expanded_path = config_path.expanduser()
    config_bytes = read_bytes_nofollow(expanded_path.parent, expanded_path.name)
    parsed_json = json.loads(config_bytes.decode("utf-8"))
    if not isinstance(parsed_json, dict):
        raise ValueError(
            f"{config_path} must be a JSON object with a non-empty 'judges' list"
        )
    elif set(parsed_json.keys()) & FORBIDDEN_CREDENTIAL_KEYS:
        raise ValueError(
            f"{config_path} must not embed raw credentials; pass api_key_env inside judge entries instead"
        )
    elif set(parsed_json.keys()) != {"judges"} or not isinstance(
        parsed_json.get("judges"), list
    ):
        raise ValueError(
            f"{config_path} must be a JSON object containing only a non-empty 'judges' list"
        )
    else:
        pass

    judge_entries = parsed_json["judges"]
    if not judge_entries:
        raise ValueError(f"{config_path} must declare at least one judge")
    else:
        for raw_judge in judge_entries:
            if not isinstance(raw_judge, dict):
                raise ValueError(f"{config_path} judge entries must be JSON objects")
            elif set(raw_judge.keys()) & FORBIDDEN_CREDENTIAL_KEYS:
                raise ValueError(
                    f"{config_path} must not embed raw credentials; pass api_key_env instead"
                )
            else:
                pass

    parsed_file = C3JudgeConfigFile.model_validate(parsed_json)
    seen_names: set[str] = set()
    for judge_spec in parsed_file.judges:
        if judge_spec.name in seen_names:
            raise ValueError(
                f"Duplicate judge name {judge_spec.name!r} in {config_path}"
            )
        else:
            seen_names.add(judge_spec.name)
    return list(parsed_file.judges)


def validate_single_judgment_record(
    raw_judgment_payload: JsonValue,
    *,
    instance: C3InstanceRecord,
    expected_prediction_sha256: str,
    expected_prompt_sha256: str,
    expected_judge_fingerprint_sha256: str,
) -> bool | None:
    """Return validated boolean verdict if cache-consistent, raise on tampering/mismatch, or None if stale."""
    if (
        isinstance(raw_judgment_payload, dict)
        and raw_judgment_payload.get("schema_version")
        != JUDGMENT_CHECKPOINT_SCHEMA_VERSION
    ):
        return None
    elif (
        isinstance(raw_judgment_payload, dict)
        and raw_judgment_payload.get("judge_fingerprint_sha256")
        != expected_judge_fingerprint_sha256
    ):
        raise ValueError(
            f"Judge configuration fingerprint mismatch for {instance.instance_id}"
        )
    elif isinstance(raw_judgment_payload, dict) and (
        raw_judgment_payload.get("prediction_sha256") != expected_prediction_sha256
        or raw_judgment_payload.get("prompt_sha256") != expected_prompt_sha256
    ):
        return None
    else:
        try:
            judgment_record = C3JudgmentCheckpointRecord.model_validate(
                raw_judgment_payload
            )
        except ValueError as validation_error:
            raise ValueError(
                f"Invalid cached judgment for {instance.instance_id} (non-successful finish_reason or malformed fields): {validation_error}"
            ) from None
        reparsed_verdict = parse_c3_judge_verdict(
            judgment_record.raw_verdict_text, instance.language
        )
        if reparsed_verdict is None or reparsed_verdict != judgment_record.correct:
            raise ValueError(
                f"Cached judgment boolean disagrees with raw_verdict_text for {instance.instance_id}"
            )
        else:
            return judgment_record.correct


def load_judgments_json_map(
    output_dir: Path,
) -> dict[str, dict[str, dict[str, JsonValue]]]:
    try:
        judgments_bytes = read_bytes_nofollow(output_dir, "judgments.json")
    except FileNotFoundError:
        return {}
    loaded_judgments = json.loads(judgments_bytes.decode("utf-8"))
    if not isinstance(loaded_judgments, dict):
        raise ValueError("judgments.json must be a JSON object keyed by judge name")
    else:
        parsed_by_judge: dict[str, dict[str, dict[str, JsonValue]]] = {}
        for judge_name, judge_section in loaded_judgments.items():
            if not isinstance(judge_name, str) or not isinstance(judge_section, dict):
                raise ValueError(
                    f"Malformed judge section {judge_name!r} in judgments.json"
                )
            else:
                typed_section: dict[str, dict[str, JsonValue]] = {}
                for instance_id, record_value in judge_section.items():
                    if not isinstance(record_value, dict):
                        raise ValueError(
                            f"Malformed judgment record for {judge_name}/{instance_id} in judgments.json"
                        )
                    else:
                        typed_section[instance_id] = record_value
                parsed_by_judge[judge_name] = typed_section
        return parsed_by_judge


def load_structured_judge_failures(
    output_dir: Path, relative_failures_path: str = "judge_failures.json"
) -> dict[str, dict[str, str]]:
    try:
        failure_bytes = read_bytes_nofollow(output_dir, relative_failures_path)
    except FileNotFoundError:
        return {}
    parsed_failures = json.loads(failure_bytes.decode("utf-8"))
    if not isinstance(parsed_failures, dict):
        raise ValueError(
            f"{relative_failures_path} must be a JSON object keyed by judge name"
        )
    else:
        normalized_failures_by_judge: dict[str, dict[str, str]] = {}
        for judge_name, judge_map in parsed_failures.items():
            if not isinstance(judge_name, str) or not isinstance(judge_map, dict):
                raise ValueError(
                    f"Malformed judge failure section {judge_name!r} in {relative_failures_path}"
                )
            else:
                typed_failure_reasons: dict[str, str] = {}
                for instance_id, failure_reason in judge_map.items():
                    if isinstance(instance_id, str) and isinstance(failure_reason, str):
                        typed_failure_reasons[instance_id] = failure_reason
                    else:
                        raise ValueError(
                            f"Malformed failure entry under {judge_name!r} in {relative_failures_path}"
                        )
                if typed_failure_reasons:
                    normalized_failures_by_judge[judge_name] = typed_failure_reasons
                else:
                    pass
        return normalized_failures_by_judge


def persist_structured_judge_failures(
    output_dir: Path,
    failures_by_judge: dict[str, dict[str, str]],
    *,
    relative_failures_path: str = "judge_failures.json",
) -> None:
    non_empty_failures_by_judge = {
        judge_name: dict(failure_reasons_by_instance)
        for judge_name, failure_reasons_by_instance in failures_by_judge.items()
        if failure_reasons_by_instance
    }
    if non_empty_failures_by_judge:
        write_contained_json_atomic(
            output_dir, relative_failures_path, non_empty_failures_by_judge
        )
    else:
        unlink_contained_file_nofollow(output_dir, relative_failures_path)


def load_offline_verified_judge_state(
    output_dir: Path,
    *,
    automated_instances: list[C3InstanceRecord],
    predictions_by_id: dict[str, C3PredictionCheckpointRecord],
    supplied_judge_specs: list[C3JudgeEndpointSpec],
    current_prompt_bundle_sha256: str,
) -> C3JudgePhaseResult:
    """Load and validate judge_manifest.json, judgments.json, and judge_failures.json for offline scoring."""
    if not automated_instances:
        return C3JudgePhaseResult(
            authoritative_judge_names=("human_expert",),
            authoritative_public_judges=[],
            prompt_bundle_sha256=current_prompt_bundle_sha256,
            verdicts_by_judge={"human_expert": {}},
            failed_judgments_by_judge={},
        )
    else:
        pass

    try:
        judge_manifest_bytes = read_bytes_nofollow(output_dir, "judge_manifest.json")
    except FileNotFoundError:
        raise ValueError(
            "Missing authoritative judge_manifest.json for automated C3 subtasks; run --phase judge first"
        ) from None
    saved_judge_manifest = C3JudgeManifestRecord.model_validate_json(
        judge_manifest_bytes.decode("utf-8")
    )
    if not saved_judge_manifest.judges:
        raise ValueError("judge_manifest.json declares no judges")
    elif saved_judge_manifest.prompt_bundle_sha256 != current_prompt_bundle_sha256:
        raise ValueError(
            "Current C3 prompt bundle differs from saved judge_manifest.json prompt_bundle_sha256; "
            "re-run --phase judge to refresh judgments and provenance"
        )
    else:
        pass

    saved_entries_by_name = {
        judge_entry.name: judge_entry for judge_entry in saved_judge_manifest.judges
    }
    ordered_saved_names = [
        judge_entry.name for judge_entry in saved_judge_manifest.judges
    ]
    if supplied_judge_specs:
        supplied_names = [judge_spec.name for judge_spec in supplied_judge_specs]
        if set(supplied_names) != set(ordered_saved_names):
            raise ValueError(
                f"Score-phase --judge-config roster {supplied_names} differs from authoritative "
                f"{ordered_saved_names} in judge_manifest.json; run --phase judge to update the roster"
            )
        else:
            for judge_spec in supplied_judge_specs:
                if (
                    saved_entries_by_name[judge_spec.name].judge_fingerprint_sha256
                    != judge_spec.compute_fingerprint_sha256()
                ):
                    raise ValueError(
                        f"Judge {judge_spec.name!r} in --judge-config conflicts with saved judge_manifest.json"
                    )
                else:
                    pass
    else:
        pass

    judge_names = tuple(ordered_saved_names)
    verdicts_by_judge: dict[str, dict[str, bool]] = {
        judge_name: {} for judge_name in judge_names
    }
    loaded_judgments = load_judgments_json_map(output_dir)
    for judge_name in judge_names:
        judge_section = loaded_judgments.get(judge_name, {})
        expected_fingerprint = saved_entries_by_name[
            judge_name
        ].judge_fingerprint_sha256
        for instance in automated_instances:
            if (
                instance.instance_id not in judge_section
                or instance.instance_id not in predictions_by_id
            ):
                continue
            else:
                prediction_record = predictions_by_id[instance.instance_id]
                system_prompt, user_prompt = build_c3_judge_messages(
                    instance, prediction_record.assistant_turns
                )
                expected_prompt_sha256 = compute_c3_judge_prompt_sha256(
                    system_prompt, user_prompt
                )
                validated_bool = validate_single_judgment_record(
                    judge_section[instance.instance_id],
                    instance=instance,
                    expected_prediction_sha256=prediction_record.prediction_sha256,
                    expected_prompt_sha256=expected_prompt_sha256,
                    expected_judge_fingerprint_sha256=expected_fingerprint,
                )
                if validated_bool is None:
                    raise ValueError(
                        f"Stale judgment entry for {judge_name}/{instance.instance_id} in judgments.json; re-run --phase judge"
                    )
                else:
                    verdicts_by_judge[judge_name][instance.instance_id] = validated_bool

    automated_ids = {instance.instance_id for instance in automated_instances}
    loaded_failures = load_structured_judge_failures(output_dir)
    failed_judgments_by_judge: dict[str, dict[str, str]] = {}
    for judge_name in judge_names:
        judge_failure_map = {
            instance_id: failure_reason
            for instance_id, failure_reason in loaded_failures.get(
                judge_name, {}
            ).items()
            if instance_id in automated_ids
            and instance_id not in verdicts_by_judge.get(judge_name, {})
        }
        if judge_failure_map:
            failed_judgments_by_judge[judge_name] = judge_failure_map
        else:
            pass

    return C3JudgePhaseResult(
        authoritative_judge_names=judge_names,
        authoritative_public_judges=list(saved_judge_manifest.judges),
        prompt_bundle_sha256=saved_judge_manifest.prompt_bundle_sha256,
        verdicts_by_judge=verdicts_by_judge,
        failed_judgments_by_judge=failed_judgments_by_judge,
    )
