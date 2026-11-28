# SPDX-License-Identifier: Apache-2.0 AND MIT
# Portions derived from https://github.com/step-out/C3 (Copyright (c) 2025-2026 C3 Benchmark Authors, MIT License; see benchmarks/c3/LICENSE).
"""Markdown report formatting and final evaluation summary construction for C3."""

from __future__ import annotations

import hashlib
import uuid

from pydantic import JsonValue, TypeAdapter

from benchmarks.c3.metrics import C3LanguageMetricSummary
from benchmarks.c3.prompts import C3ManualGenerationAnnotation
from benchmarks.c3.registry import (
    C3_DATASET_ID,
    C3_DATASET_REVISION,
    C3_LANGUAGES,
    C3_PHENOMENA,
    C3EvaluationScope,
    C3Language,
)
from benchmarks.c3.schemas import C3PublicJudgeManifestEntry, C3RunManifestRecord
from benchmarks.dataset.c3 import C3DatasetIdentitySummary
from benchmarks.runtime_metrics import collect_benchmark_provenance

LANGUAGE_SUMMARY_ADAPTER: TypeAdapter[C3LanguageMetricSummary] = TypeAdapter(
    C3LanguageMetricSummary
)
DATASET_IDENTITY_ADAPTER: TypeAdapter[C3DatasetIdentitySummary] = TypeAdapter(
    C3DatasetIdentitySummary
)


def format_percent(accuracy: float | None) -> str:
    if accuracy is not None:
        return f"{accuracy * 100.0:.2f}%"
    else:
        return "n/a"


def build_c3_evaluation_summary_and_report(
    *,
    saved_run_manifest: C3RunManifestRecord,
    effective_scope: C3EvaluationScope,
    dataset_identity: C3DatasetIdentitySummary,
    selected_instances_sha256: str,
    selected_instances_count: int,
    completed_instances_count: int,
    truncated_instances_count: int,
    failed_instances: dict[str, str],
    automated_instances_present: bool,
    judge_names: tuple[str, ...],
    public_judge_entries: list[C3PublicJudgeManifestEntry],
    effective_prompt_bundle_sha256: str,
    required_automated_judgments: int,
    resolved_automated_judgments: int,
    failed_judgments_by_judge: dict[str, dict[str, str]],
    required_human_annotations: int,
    manual_annotations: dict[str, C3ManualGenerationAnnotation],
    manual_annotations_file_sha256: str | None,
    language_results: dict[C3Language, C3LanguageMetricSummary],
) -> tuple[dict[str, JsonValue], str]:
    dataset_matches_pinned = dataset_identity.dataset_matches_expected_revision
    total_failed_judgments = sum(
        len(failure_reasons_by_instance)
        for failure_reasons_by_instance in failed_judgments_by_judge.values()
    )
    resolved_human_annotations = len(manual_annotations)
    is_complete = (
        completed_instances_count == selected_instances_count
        and not failed_instances
        and resolved_automated_judgments == required_automated_judgments
        and total_failed_judgments == 0
        and resolved_human_annotations == required_human_annotations
    )

    provenance = collect_benchmark_provenance(
        model_id=saved_run_manifest.model,
        model_revision=saved_run_manifest.model_revision,
        dataset_id=C3_DATASET_ID,
        dataset_revision=C3_DATASET_REVISION if dataset_matches_pinned else None,
        launch_command=saved_run_manifest.launch_command,
        evaluation_input_sha256=selected_instances_sha256,
        server_config={
            "requested_scope": saved_run_manifest.requested_scope,
            "effective_scope": effective_scope,
            "language": saved_run_manifest.language,
            "subtasks": saved_run_manifest.subtasks,
            "mini": saved_run_manifest.mini,
            "max_samples_per_subtask": saved_run_manifest.max_samples_per_subtask,
            "base_url": saved_run_manifest.base_url,
            "endpoint_identity_sha256": saved_run_manifest.endpoint_identity_sha256,
            "modalities": list(saved_run_manifest.modalities),
            "export_generation_audio": saved_run_manifest.export_generation_audio,
            "temperature": saved_run_manifest.temperature,
            "max_tokens": saved_run_manifest.max_tokens,
            "seed": saved_run_manifest.seed,
            "max_concurrency": saved_run_manifest.max_concurrency,
            "concurrency_history": list(saved_run_manifest.concurrency_history),
            "warmup_requests": saved_run_manifest.warmup_requests,
            "warmup_history": list(saved_run_manifest.warmup_history),
            "request_timeout_seconds": saved_run_manifest.request_timeout_seconds,
            "generation_prompt_bundle_sha256": saved_run_manifest.prompt_bundle_sha256,
            "prompt_bundle_sha256": effective_prompt_bundle_sha256,
            "judges": [
                judge_entry.model_dump(mode="json")
                for judge_entry in public_judge_entries
            ],
            "manual_generation_annotations_sha256": manual_annotations_file_sha256,
        },
    )

    scoring_attempt_id = uuid.uuid4().hex
    effective_judge_names = judge_names if automated_instances_present else ()
    failed_instances_count = len(failed_instances)
    missing_instances = max(
        0,
        selected_instances_count - completed_instances_count - failed_instances_count,
    )
    missing_judgments = max(
        0,
        required_automated_judgments
        - resolved_automated_judgments
        - total_failed_judgments,
    )
    missing_human_annotations = max(
        0, required_human_annotations - resolved_human_annotations
    )
    heading_suffix = "" if is_complete else " [PARTIAL / INCOMPLETE]"
    selected_languages_label = ", ".join(
        language_code
        for language_code in C3_LANGUAGES
        if language_code in language_results
    )

    if effective_scope == "full":
        scope_banner = (
            f"**Paper Full Scope (`scope=full`, languages: `{selected_languages_label}`)**: "
            f"Evaluates `{selected_instances_count}` selected instances including `{required_human_annotations}` "
            f"human-scored phonological generation items (`{resolved_human_annotations}/{required_human_annotations}` "
            "human-verified labels resolved)."
        )
    elif effective_scope == "auto":
        scope_banner = (
            f"**Automated Subset (`C3-Auto`, `scope=auto`, languages: `{selected_languages_label}`)**: "
            f"Evaluates `{selected_instances_count}` automated understanding and multi-turn instances and excludes "
            "human-listened phonological generation items; `Phonological` below reflects understanding subtasks "
            "only and must not be conflated with the full paper overall."
        )
    else:
        scope_banner = (
            f"**Subset Run (`scope={effective_scope}`, languages: `{selected_languages_label}`)**: "
            f"Evaluates `{selected_instances_count}` selected instances (`{required_human_annotations}` human-scored "
            "generation items selected); five-phenomenon macro accuracy is computed only when all five phenomena "
            "are complete."
        )

    revision_warning_lines = (
        []
        if dataset_matches_pinned
        else [
            "",
            "> **Warning (`unverified_dataset`)**: Local dataset files do not match the pinned Hugging Face "
            "`SHA256SUMS` manifest; official `c3_auto_macro_accuracy` / `c3_paper_full_macro_accuracy` claims are disabled.",
        ]
    )
    judge_label = (
        ", ".join(effective_judge_names)
        if effective_judge_names
        else "none (human-only selection)"
    )
    report_lines: list[str] = [
        f"# C3 Spoken Dialogue Evaluation ({saved_run_manifest.model}){heading_suffix}",
        "",
        f"**Scoring attempt ID**: `{scoring_attempt_id}`",
        "",
        f"**Inference coverage**: selected `{selected_instances_count}`; completed `{completed_instances_count}` (length-truncated `{truncated_instances_count}`); missing `{missing_instances}`; failed `{failed_instances_count}`.",
        "",
        f"**LLM judge coverage** (`{judge_label}`): required `{required_automated_judgments}`; resolved `{resolved_automated_judgments}`; missing/pending `{missing_judgments}`; failed `{total_failed_judgments}`.",
        "",
        f"**Human generation coverage**: required `{required_human_annotations}`; resolved `{resolved_human_annotations}`; missing/pending `{missing_human_annotations}`.",
        "",
        f"> {scope_banner}",
        *revision_warning_lines,
        "",
        "| Language | Instances | Scope | Phonological | Semantic | Omission | Coreference | Multi-turn | Macro Accuracy |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|",
    ]

    for language_code in C3_LANGUAGES:
        if language_code not in language_results:
            continue
        else:
            language_summary = language_results[language_code]
            formatted_phenomenon_cells = [
                (
                    format_percent(
                        language_summary.phenomena[phenomenon_name].mean_accuracy
                    )
                    if phenomenon_name in language_summary.phenomena
                    else "n/a"
                )
                for phenomenon_name in C3_PHENOMENA
            ]
            macro_text = format_percent(language_summary.mean_macro_accuracy)
            report_lines.append(
                f"| {language_code} | {language_summary.instances} | {effective_scope} | "
                f"{formatted_phenomenon_cells[0]} | {formatted_phenomenon_cells[1]} | "
                f"{formatted_phenomenon_cells[2]} | {formatted_phenomenon_cells[3]} | "
                f"{formatted_phenomenon_cells[4]} | **{macro_text}** |"
            )

    report_lines.extend(["", "## Subtask Breakdown", ""])
    for language_code in C3_LANGUAGES:
        if language_code not in language_results:
            continue
        else:
            language_summary = language_results[language_code]
            report_lines.extend(
                [
                    f"### Language: `{language_code}`",
                    "",
                    "| Subtask | Instances | Pending Instances | Mean Accuracy |",
                    "|---|---:|---:|---:|",
                ]
            )
            for subtask_name, subtask_summary in language_summary.subtasks.items():
                report_lines.append(
                    f"| `{subtask_name}` | {subtask_summary.instances} | "
                    f"{subtask_summary.pending_instances} | "
                    f"{format_percent(subtask_summary.mean_accuracy)} |"
                )
            report_lines.append("")

    markdown_report = "\n".join(report_lines)
    results_markdown_sha256 = hashlib.sha256(
        markdown_report.encode("utf-8")
    ).hexdigest()

    serialized_language_results: dict[str, JsonValue] = {
        language_code: LANGUAGE_SUMMARY_ADAPTER.dump_python(
            language_summary, mode="json"
        )
        for language_code, language_summary in language_results.items()
    }
    evaluation_summary: dict[str, JsonValue] = {
        "status": "complete" if is_complete else "incomplete",
        "scoring_attempt_id": scoring_attempt_id,
        "results_markdown_sha256": results_markdown_sha256,
        "benchmark": "c3_spoken",
        "scope": effective_scope,
        "paper_metric_scope_complete": (
            effective_scope == "full" and dataset_matches_pinned
        ),
        "model": saved_run_manifest.model,
        "dataset_identity": DATASET_IDENTITY_ADAPTER.dump_python(
            dataset_identity, mode="json"
        ),
        "selected_instances": selected_instances_count,
        "completed_instances": completed_instances_count,
        "truncated_instances": truncated_instances_count,
        "failed_instances": failed_instances,
        "required_judgments": required_automated_judgments,
        "resolved_judgments": resolved_automated_judgments,
        "failed_judgments": failed_judgments_by_judge,
        "required_human_annotations": required_human_annotations,
        "resolved_human_annotations": resolved_human_annotations,
        "manual_generation_annotations": {
            instance_id: annotation.model_dump(mode="json")
            for instance_id, annotation in manual_annotations.items()
        },
        "judge_names": list(effective_judge_names),
        "results": serialized_language_results,
        "provenance": provenance,
    }
    return evaluation_summary, markdown_report
