# SPDX-License-Identifier: Apache-2.0 AND MIT
# Portions derived from https://github.com/step-out/C3 (Copyright (c) 2025-2026 C3 Benchmark Authors, MIT License; see benchmarks/c3/LICENSE).
"""CLI entry point for the C3 bilingual spoken-dialogue benchmark on /v1/chat/completions."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import sys

from pydantic import JsonValue

from benchmarks.c3.config_manifest import (
    C3EvalConfig,
    load_authoritative_run_manifest,
    load_inference_failure_ledger,
    parse_c3_cli_config,
    resolve_effective_scope,
)
from benchmarks.c3.containment import (
    read_bytes_nofollow,
    validate_non_symlink_root_directory,
    write_contained_json_atomic,
    write_text_atomic,
)
from benchmarks.c3.generation import run_c3_generate_phase
from benchmarks.c3.judge_state import (
    C3JudgeEndpointSpec,
    load_c3_judge_configs,
    load_offline_verified_judge_state,
)
from benchmarks.c3.judging import judge_c3_predictions_via_http
from benchmarks.c3.metrics import C3LanguageMetricSummary, score_c3_language_instances
from benchmarks.c3.prompts import (
    C3ManualGenerationAnnotation,
    compute_c3_prompt_bundle_sha256,
    load_manual_generation_annotations,
)
from benchmarks.c3.registry import C3_LANGUAGES, C3Language, select_c3_subtask_specs
from benchmarks.c3.reporting import build_c3_evaluation_summary_and_report
from benchmarks.c3.runner import validate_cached_predictions
from benchmarks.dataset.c3 import load_and_verify_c3_snapshot
from sglang_omni.utils.logging import configure_dependency_loggers


async def run_c3_evaluation(config: C3EvalConfig) -> dict[str, JsonValue]:
    dataset_root = validate_non_symlink_root_directory(
        config.dataset_root, create_if_missing=False
    )
    output_dir = validate_non_symlink_root_directory(
        config.output_dir, create_if_missing=True
    )

    include_human_generation = (
        config.scope == "full"
        or config.export_generation_audio
        or (
            config.scope == "subset"
            and config.subtasks is not None
            and any("generation" in subtask_token for subtask_token in config.subtasks)
        )
    )
    selected_subtask_specs = select_c3_subtask_specs(
        language=config.language,
        include_human_generation=include_human_generation,
        include_multiturn=(config.scope != "single_turn"),
        subtask_filter=config.subtasks,
    )
    loaded_snapshot = load_and_verify_c3_snapshot(
        dataset_root,
        subtask_specs=selected_subtask_specs,
        mini=config.mini,
        max_samples_per_subtask=config.max_samples_per_subtask,
        strict_counts=True,
        verify_all_audio_headers=True,
    )
    instances = loaded_snapshot.instances
    dataset_identity = loaded_snapshot.identity
    dataset_matches_pinned = dataset_identity.dataset_matches_expected_revision
    pinned_audio_sha256_by_path = loaded_snapshot.pinned_audio_sha256_by_path
    effective_scope = resolve_effective_scope(config, instances)

    if (
        not dataset_matches_pinned
        and effective_scope in ("auto", "full")
        and not config.allow_unverified_dataset
    ):
        raise ValueError(
            "Local C3 dataset files do not match the pinned Hugging Face revision SHA256SUMS manifest; "
            "re-download via python -m benchmarks.dataset.prepare --dataset c3 or pass "
            "--allow-unverified-dataset for diagnostic non-leaderboard runs"
        )
    else:
        pass

    instances_by_id = {instance.instance_id: instance for instance in instances}
    selected_instances_sha256 = hashlib.sha256(
        json.dumps([instance.instance_id for instance in instances]).encode("utf-8")
    ).hexdigest()

    automated_instances = [
        instance
        for instance in instances
        if not instance.requires_human_audio_annotation
    ]
    human_instances_by_id = {
        instance.instance_id: instance
        for instance in instances
        if instance.requires_human_audio_annotation
    }

    if (
        config.phase in ("all", "judge")
        and automated_instances
        and config.judge_config_path is None
    ):
        raise ValueError(
            f"--judge-config is required for --phase {config.phase} when automated C3 subtasks are selected"
        )
    else:
        pass

    if config.phase in ("all", "generate"):
        generation_result = await run_c3_generate_phase(
            config,
            instances,
            pinned_audio_sha256_by_path=pinned_audio_sha256_by_path,
        )
        predictions_by_id = generation_result.predictions_by_id
        failed_instances = generation_result.failed_instances
        saved_run_manifest = load_authoritative_run_manifest(
            output_dir,
            expected_selected_instances_sha256=selected_instances_sha256,
            config=config,
        )
    else:
        saved_run_manifest = load_authoritative_run_manifest(
            output_dir,
            expected_selected_instances_sha256=selected_instances_sha256,
            config=config,
        )
        try:
            predictions_bytes = read_bytes_nofollow(output_dir, "predictions.json")
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Missing predictions.json in {output_dir}; run --phase generate first"
            ) from None
        parsed_predictions_json = json.loads(predictions_bytes.decode("utf-8"))
        if not isinstance(parsed_predictions_json, dict):
            raise ValueError("predictions.json must be a JSON object")
        else:
            predictions_by_id = validate_cached_predictions(
                parsed_predictions_json,
                instances_by_id,
                dataset_root,
                output_dir,
                require_generation_audio_export=(
                    saved_run_manifest.export_generation_audio
                    or bool(human_instances_by_id)
                ),
                pinned_audio_sha256_by_path=pinned_audio_sha256_by_path,
            )
        failed_instances = {
            instance_id: failure_reason
            for instance_id, failure_reason in load_inference_failure_ledger(
                output_dir
            ).items()
            if instance_id in instances_by_id and instance_id not in predictions_by_id
        }

    completed_instances_count = sum(
        1 for instance in instances if instance.instance_id in predictions_by_id
    )
    truncated_instances_count = sum(
        1
        for instance in instances
        if instance.instance_id in predictions_by_id
        and "length" in predictions_by_id[instance.instance_id].finish_reasons
    )
    if config.phase == "generate":
        return {
            "status": (
                "complete"
                if completed_instances_count == len(instances) and not failed_instances
                else "incomplete"
            ),
            "selected_instances": len(instances),
            "completed_instances": completed_instances_count,
            "truncated_instances": truncated_instances_count,
            "failed_instances": failed_instances,
        }
    else:
        pass

    current_prompt_bundle_sha256 = compute_c3_prompt_bundle_sha256()
    judge_specs: list[C3JudgeEndpointSpec] = (
        load_c3_judge_configs(config.judge_config_path)
        if config.judge_config_path is not None
        else []
    )
    if config.phase in ("all", "judge") and automated_instances:
        judge_result = await judge_c3_predictions_via_http(
            instances,
            predictions_by_id,
            judge_specs,
            output_dir=output_dir,
            timeout_seconds=config.request_timeout_seconds,
            trust_env=config.trust_env,
        )
    else:
        judge_result = load_offline_verified_judge_state(
            output_dir,
            automated_instances=automated_instances,
            predictions_by_id=predictions_by_id,
            supplied_judge_specs=judge_specs,
            current_prompt_bundle_sha256=current_prompt_bundle_sha256,
        )

    verdicts_by_judge = judge_result.verdicts_by_judge
    failed_judgments_by_judge = judge_result.failed_judgments_by_judge
    judge_names = judge_result.authoritative_judge_names
    required_automated_judgments = (
        len(automated_instances) * len(judge_names) if automated_instances else 0
    )
    resolved_automated_judgments = sum(
        1
        for judge_name in judge_names
        for instance in automated_instances
        if instance.instance_id in verdicts_by_judge.get(judge_name, {})
    )

    if config.phase == "judge":
        total_failed_judgments = sum(
            len(failure_reasons_by_instance)
            for failure_reasons_by_instance in failed_judgments_by_judge.values()
        )
        judge_only_complete = (
            completed_instances_count == len(instances)
            and not failed_instances
            and resolved_automated_judgments == required_automated_judgments
            and total_failed_judgments == 0
            and not human_instances_by_id
        )
        return {
            "status": "complete" if judge_only_complete else "incomplete",
            "selected_instances": len(instances),
            "completed_instances": completed_instances_count,
            "truncated_instances": truncated_instances_count,
            "required_automated_judgments": required_automated_judgments,
            "resolved_automated_judgments": resolved_automated_judgments,
            "pending_human_generation_instances": len(human_instances_by_id),
            "failed_judgments": failed_judgments_by_judge,
        }
    else:
        pass

    manual_annotations: dict[str, C3ManualGenerationAnnotation] = {}
    manual_annotations_file_sha256: str | None = None
    if config.manual_generation_annotations_path is not None:
        expected_hashes = {
            instance_id: (
                prediction_record.prediction_sha256,
                prediction_record.generated_audio_sha256,
            )
            for instance_id, prediction_record in predictions_by_id.items()
        }
        manual_annotations, manual_annotations_file_sha256 = (
            load_manual_generation_annotations(
                config.manual_generation_annotations_path,
                allowed_human_instances=human_instances_by_id,
                expected_hashes_by_instance=expected_hashes,
            )
        )
    else:
        pass

    combined_verdicts_by_judge: dict[str, dict[str, bool]] = {
        judge_name: dict(verdicts_by_judge.get(judge_name, {}))
        for judge_name in judge_names
    }
    for human_id, annotation in manual_annotations.items():
        for judge_name in judge_names:
            combined_verdicts_by_judge[judge_name][human_id] = annotation.correct

    language_results: dict[C3Language, C3LanguageMetricSummary] = {}
    for language_code in C3_LANGUAGES:
        if any(instance.language == language_code for instance in instances):
            language_results[language_code] = score_c3_language_instances(
                language_code,
                instances,
                combined_verdicts_by_judge,
                judge_names,
                scope=effective_scope,
                dataset_matches_pinned_revision=dataset_matches_pinned,
            )
        else:
            pass

    summary, markdown_report = build_c3_evaluation_summary_and_report(
        saved_run_manifest=saved_run_manifest,
        effective_scope=effective_scope,
        dataset_identity=dataset_identity,
        selected_instances_sha256=selected_instances_sha256,
        selected_instances_count=len(instances),
        completed_instances_count=completed_instances_count,
        truncated_instances_count=truncated_instances_count,
        failed_instances=failed_instances,
        automated_instances_present=bool(automated_instances),
        judge_names=judge_names,
        public_judge_entries=judge_result.authoritative_public_judges,
        effective_prompt_bundle_sha256=judge_result.prompt_bundle_sha256,
        required_automated_judgments=required_automated_judgments,
        resolved_automated_judgments=resolved_automated_judgments,
        failed_judgments_by_judge=failed_judgments_by_judge,
        required_human_annotations=len(human_instances_by_id),
        manual_annotations=manual_annotations,
        manual_annotations_file_sha256=manual_annotations_file_sha256,
        language_results=language_results,
    )
    write_contained_json_atomic(output_dir, "c3_results.json", summary)
    write_text_atomic(output_dir, "RESULTS.md", markdown_report)
    print(markdown_report)
    return summary


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    configure_dependency_loggers()
    config = parse_c3_cli_config(argv)
    summary = asyncio.run(run_c3_evaluation(config))
    return 0 if summary.get("status") == "complete" else 1


if __name__ == "__main__":
    sys.exit(main())
else:
    pass
