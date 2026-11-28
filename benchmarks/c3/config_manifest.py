# SPDX-License-Identifier: Apache-2.0 AND MIT
# Portions derived from https://github.com/step-out/C3 (Copyright (c) 2025-2026 C3 Benchmark Authors, MIT License; see benchmarks/c3/LICENSE).
"""Evaluation configuration, manifest compatibility, and failure-ledger persistence for C3."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import JsonValue

from benchmarks.c3.containment import (
    read_bytes_nofollow,
    unlink_contained_file_nofollow,
    validate_non_symlink_root_directory,
    write_contained_json_atomic,
)
from benchmarks.c3.http_transport import (
    compute_endpoint_identity_sha256,
    validate_http_endpoint_url,
)
from benchmarks.c3.registry import (
    C3EvaluationScope,
    C3LanguageSelection,
    C3OutputModality,
)
from benchmarks.c3.schemas import C3PredictionCheckpointRecord, C3RunManifestRecord
from benchmarks.dataset.c3 import C3InstanceRecord
from benchmarks.eval.benchmark_duplex_mpe import sanitize_launch_command

C3PhaseSelection = Literal["all", "generate", "judge", "score"]


@dataclass(frozen=True, kw_only=True)
class C3EvalConfig:
    dataset_root: Path
    output_dir: Path
    phase: C3PhaseSelection = "all"
    scope: C3EvaluationScope = "auto"
    language: C3LanguageSelection = "all"
    subtasks: tuple[str, ...] | None = None
    mini: bool = False
    max_samples_per_subtask: int | None = None
    base_url: str | None = None
    api_key_env: str | None = None
    model: str | None = None
    model_revision: str | None = None
    launch_command: str | None = None
    modalities: tuple[C3OutputModality, ...] = ("text",)
    temperature: float = 0.0
    max_tokens: int = 1024
    seed: int | None = 1
    max_concurrency: int = 8
    warmup_requests: int = 1
    request_timeout_seconds: float = 180.0
    judge_config_path: Path | None = None
    manual_generation_annotations_path: Path | None = None
    export_generation_audio: bool = False
    allow_unverified_dataset: bool = False
    trust_env: bool = False

    def __post_init__(self) -> None:
        if self.max_concurrency < 1:
            raise ValueError("--max-concurrency must be >= 1")
        elif self.warmup_requests < 0:
            raise ValueError("--warmup must be >= 0")
        elif self.max_tokens < 1:
            raise ValueError("--max-tokens must be >= 1")
        elif self.request_timeout_seconds <= 0:
            raise ValueError("--timeout-s must be > 0")
        elif (
            self.max_samples_per_subtask is not None
            and self.max_samples_per_subtask < 1
        ):
            raise ValueError("--max-samples-per-subtask must be >= 1")
        elif not self.modalities or any(
            modality not in ("text", "audio") for modality in self.modalities
        ):
            raise ValueError("--modalities must be drawn from {'text', 'audio'}")
        elif "text" not in self.modalities:
            raise ValueError(
                "--modalities must include 'text' for C3 dialogue responses"
            )
        elif (
            self.phase in ("generate", "judge")
            and self.manual_generation_annotations_path is not None
        ):
            raise ValueError(
                f"--manual-generation-annotations is only supported for --phase all or --phase score, not --phase {self.phase}"
            )
        else:
            pass

        sanitize_launch_command(self.launch_command)
        verified_dataset_root = validate_non_symlink_root_directory(
            self.dataset_root, create_if_missing=False
        )
        verified_output_dir = validate_non_symlink_root_directory(
            self.output_dir, create_if_missing=False
        )
        if (
            verified_output_dir == verified_dataset_root
            or verified_output_dir.is_relative_to(verified_dataset_root)
            or verified_dataset_root.is_relative_to(verified_output_dir)
        ):
            raise ValueError("--output-dir and --dataset-root must not overlap")
        else:
            pass

        if self.phase in ("all", "generate"):
            if not self.base_url:
                raise ValueError(f"--base-url is required for --phase {self.phase}")
            elif not self.model:
                raise ValueError(f"--model is required for --phase {self.phase}")
            else:
                validate_http_endpoint_url(self.base_url)
        elif self.base_url is not None:
            validate_http_endpoint_url(self.base_url)
        else:
            pass


def resolve_effective_scope(
    config: C3EvalConfig, instances: list[C3InstanceRecord]
) -> C3EvaluationScope:
    has_human_instances = any(
        instance.requires_human_audio_annotation for instance in instances
    )
    if (
        config.mini
        or config.max_samples_per_subtask is not None
        or config.subtasks is not None
    ):
        return "subset"
    elif config.scope == "auto" and has_human_instances:
        return "full"
    elif config.scope == "single_turn" and has_human_instances:
        return "subset"
    else:
        return config.scope


def load_inference_failure_ledger(
    output_dir: Path, relative_ledger_path: str = "inference_failures.json"
) -> dict[str, str]:
    try:
        ledger_bytes = read_bytes_nofollow(output_dir, relative_ledger_path)
    except FileNotFoundError:
        return {}
    parsed_ledger = json.loads(ledger_bytes.decode("utf-8"))
    if not isinstance(parsed_ledger, dict):
        raise ValueError(f"{relative_ledger_path} must contain a JSON object")
    else:
        return {
            instance_id: reason
            for instance_id, reason in parsed_ledger.items()
            if isinstance(instance_id, str) and isinstance(reason, str)
        }


def persist_inference_failure_ledger(
    output_dir: Path,
    failures_by_instance: dict[str, str],
    *,
    relative_ledger_path: str = "inference_failures.json",
) -> None:
    if failures_by_instance:
        write_contained_json_atomic(
            output_dir, relative_ledger_path, dict(failures_by_instance)
        )
    else:
        unlink_contained_file_nofollow(output_dir, relative_ledger_path)


def load_authoritative_run_manifest(
    output_dir: Path,
    *,
    expected_selected_instances_sha256: str,
    config: C3EvalConfig,
) -> C3RunManifestRecord:
    try:
        manifest_bytes = read_bytes_nofollow(output_dir, "run.json")
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Missing authoritative run.json in {output_dir}; run --phase generate first"
        ) from None
    manifest = C3RunManifestRecord.model_validate_json(manifest_bytes.decode("utf-8"))
    if manifest.selected_instances_sha256 != expected_selected_instances_sha256:
        raise ValueError("Selected C3 instances do not match saved run.json")
    elif config.model is not None and config.model != manifest.model:
        raise ValueError(
            f"--model {config.model!r} conflicts with saved model {manifest.model!r} in run.json"
        )
    elif (
        config.model_revision is not None
        and config.model_revision != manifest.model_revision
    ):
        raise ValueError(
            "--model-revision conflicts with saved model_revision in run.json"
        )
    elif (
        config.launch_command is not None
        and sanitize_launch_command(config.launch_command) != manifest.launch_command
    ):
        raise ValueError(
            "--launch-command conflicts with saved launch_command in run.json"
        )
    elif (
        config.base_url is not None
        and compute_endpoint_identity_sha256(config.base_url)
        != manifest.endpoint_identity_sha256
    ):
        raise ValueError(
            "--base-url conflicts with saved endpoint_identity_sha256 in run.json"
        )
    else:
        return manifest


def serialize_predictions_map(
    predictions_by_id: dict[str, C3PredictionCheckpointRecord],
) -> dict[str, JsonValue]:
    return {
        instance_id: record.model_dump(mode="json")
        for instance_id, record in predictions_by_id.items()
    }


def parse_c3_cli_config(argv: list[str] | None = None) -> C3EvalConfig:
    parser = argparse.ArgumentParser(
        description="Evaluate Spoken Dialogue Models on the C3 bilingual benchmark (/v1/chat/completions)."
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--phase",
        choices=("all", "generate", "judge", "score"),
        default=C3EvalConfig.phase,
    )
    parser.add_argument(
        "--scope",
        choices=("auto", "single_turn", "full", "subset"),
        default=C3EvalConfig.scope,
    )
    parser.add_argument(
        "--language", choices=("all", "en", "zh"), default=C3EvalConfig.language
    )
    parser.add_argument("--subtasks", type=str, default=C3EvalConfig.subtasks)
    parser.add_argument("--mini", action="store_true", default=C3EvalConfig.mini)
    parser.add_argument(
        "--max-samples-per-subtask",
        type=int,
        default=C3EvalConfig.max_samples_per_subtask,
    )
    parser.add_argument("--base-url", type=str, default=C3EvalConfig.base_url)
    parser.add_argument("--api-key-env", type=str, default=C3EvalConfig.api_key_env)
    parser.add_argument("--model", type=str, default=C3EvalConfig.model)
    parser.add_argument(
        "--model-revision", type=str, default=C3EvalConfig.model_revision
    )
    parser.add_argument(
        "--launch-command", type=str, default=C3EvalConfig.launch_command
    )
    parser.add_argument(
        "--modalities", type=str, default=",".join(C3EvalConfig.modalities)
    )
    parser.add_argument("--temperature", type=float, default=C3EvalConfig.temperature)
    parser.add_argument("--max-tokens", type=int, default=C3EvalConfig.max_tokens)
    parser.add_argument("--seed", type=int, default=C3EvalConfig.seed)
    parser.add_argument(
        "--max-concurrency", type=int, default=C3EvalConfig.max_concurrency
    )
    parser.add_argument("--warmup", type=int, default=C3EvalConfig.warmup_requests)
    parser.add_argument(
        "--timeout-s", type=float, default=C3EvalConfig.request_timeout_seconds
    )
    parser.add_argument(
        "--judge-config", type=Path, default=C3EvalConfig.judge_config_path
    )
    parser.add_argument(
        "--manual-generation-annotations",
        type=Path,
        default=C3EvalConfig.manual_generation_annotations_path,
    )
    parser.add_argument(
        "--export-generation-audio",
        action="store_true",
        default=C3EvalConfig.export_generation_audio,
    )
    parser.add_argument(
        "--allow-unverified-dataset",
        action="store_true",
        default=C3EvalConfig.allow_unverified_dataset,
    )
    parser.add_argument(
        "--trust-env", action="store_true", default=C3EvalConfig.trust_env
    )
    parsed_arguments = parser.parse_args(argv)

    subtask_tuple = (
        tuple(
            segment.strip()
            for segment in parsed_arguments.subtasks.split(",")
            if segment.strip()
        )
        if parsed_arguments.subtasks
        else None
    )
    modalities_list: list[C3OutputModality] = []
    for raw_token in parsed_arguments.modalities.split(","):
        modality_token = raw_token.strip()
        if not modality_token:
            continue
        elif modality_token == "text":
            if "text" not in modalities_list:
                modalities_list.append("text")
            else:
                pass
        elif modality_token == "audio":
            if "audio" not in modalities_list:
                modalities_list.append("audio")
            else:
                pass
        else:
            raise ValueError(
                f"Unsupported modality {modality_token!r}; --modalities must be drawn from {'text', 'audio'}"
            )
    return C3EvalConfig(
        dataset_root=parsed_arguments.dataset_root,
        output_dir=parsed_arguments.output_dir,
        phase=parsed_arguments.phase,
        scope=parsed_arguments.scope,
        language=parsed_arguments.language,
        subtasks=subtask_tuple,
        mini=parsed_arguments.mini,
        max_samples_per_subtask=parsed_arguments.max_samples_per_subtask,
        base_url=parsed_arguments.base_url,
        api_key_env=parsed_arguments.api_key_env,
        model=parsed_arguments.model,
        model_revision=parsed_arguments.model_revision,
        launch_command=parsed_arguments.launch_command,
        modalities=tuple(modalities_list),
        temperature=parsed_arguments.temperature,
        max_tokens=parsed_arguments.max_tokens,
        seed=parsed_arguments.seed,
        max_concurrency=parsed_arguments.max_concurrency,
        warmup_requests=parsed_arguments.warmup,
        request_timeout_seconds=parsed_arguments.timeout_s,
        judge_config_path=parsed_arguments.judge_config,
        manual_generation_annotations_path=parsed_arguments.manual_generation_annotations,
        export_generation_audio=parsed_arguments.export_generation_audio,
        allow_unverified_dataset=parsed_arguments.allow_unverified_dataset,
        trust_env=parsed_arguments.trust_env,
    )
