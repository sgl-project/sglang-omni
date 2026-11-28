# SPDX-License-Identifier: Apache-2.0 AND MIT
# Portions derived from https://github.com/step-out/C3 (Copyright (c) 2025-2026 C3 Benchmark Authors, MIT License; see benchmarks/c3/LICENSE).
"""C3 phenomenon-level macro aggregation and branch averaging."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from benchmarks.c3.registry import (
    C3_PHENOMENA,
    C3EvaluationScope,
    C3Language,
    C3MacroMetricName,
    C3Phenomenon,
    C3PhonologicalScopeLabel,
    C3SubtaskName,
)
from benchmarks.dataset.c3 import C3InstanceRecord

C3_EXPECTED_FULL_PHENOMENON_COUNTS: dict[C3Language, dict[C3Phenomenon, int]] = {
    "en": {
        "phonological": 29,
        "semantic": 51,
        "omission": 102,
        "coreference": 540,
        "multi-turn": 34,
    },
    "zh": {
        "phonological": 37,
        "semantic": 118,
        "omission": 70,
        "coreference": 60,
        "multi-turn": 38,
    },
}

C3_HUMAN_GENERATION_COUNTS: dict[C3Language, int] = {"en": 12, "zh": 22}


@dataclass(frozen=True, kw_only=True)
class MetricRatioCell:
    numerator: int
    denominator: int
    pending: int
    value: float | None


@dataclass(frozen=True, kw_only=True)
class C3SubtaskMetricSummary:
    instances: int
    pending_instances: int
    mean_accuracy: float | None
    by_judge: dict[str, MetricRatioCell]


@dataclass(frozen=True, kw_only=True)
class C3PhenomenonMetricSummary:
    instances: int
    pending_instances: int
    aggregation: Literal["mean_of_branches", "micro"]
    phonological_scope: C3PhonologicalScopeLabel | None
    mean_accuracy: float | None
    by_judge: dict[str, float | None]


@dataclass(frozen=True, kw_only=True)
class C3LanguageMetricSummary:
    language: C3Language
    scope: C3EvaluationScope
    paper_metric_scope_complete: bool
    dataset_matches_pinned_revision: bool
    excludes_human_phonological_generation: bool
    selected_human_generation_count: int
    excluded_human_generation_count: int | None
    instances: int
    included_phenomena: tuple[C3Phenomenon, ...]
    macro_metric_name: C3MacroMetricName
    mean_macro_accuracy: float | None
    macro_accuracy_by_judge: dict[str, float | None]
    micro_accuracy_by_judge: dict[str, MetricRatioCell]
    phenomena: dict[C3Phenomenon, C3PhenomenonMetricSummary]
    subtasks: dict[C3SubtaskName, C3SubtaskMetricSummary]


def score_c3_language_instances(
    language: C3Language,
    instances: list[C3InstanceRecord],
    verdicts_by_judge: dict[str, dict[str, bool]],
    judge_names: tuple[str, ...],
    *,
    scope: C3EvaluationScope,
    dataset_matches_pinned_revision: bool = True,
) -> C3LanguageMetricSummary:
    """Score one language split under C3 phenomenon-macro and multi-judge rules while keeping missing verdicts pending."""
    language_instances = [
        instance for instance in instances if instance.language == language
    ]
    if not language_instances:
        raise ValueError(f"No C3 instances provided for language={language}")
    else:
        pass

    instances_by_subtask: dict[C3SubtaskName, list[C3InstanceRecord]] = {}
    instances_by_phenomenon: dict[C3Phenomenon, list[C3InstanceRecord]] = {}
    for instance in language_instances:
        instances_by_subtask.setdefault(instance.subtask, []).append(instance)
        instances_by_phenomenon.setdefault(instance.phenomenon, []).append(instance)

    if scope in ("auto", "single_turn", "full"):
        expected_counts = dict(C3_EXPECTED_FULL_PHENOMENON_COUNTS[language])
        if scope != "full":
            expected_counts["phonological"] -= C3_HUMAN_GENERATION_COUNTS[language]
        else:
            pass
        if scope == "single_turn":
            expected_counts.pop("multi-turn", None)
        else:
            pass
        observed_counts = {
            phenomenon: len(subtask_instances)
            for phenomenon, subtask_instances in instances_by_phenomenon.items()
        }
        if observed_counts != expected_counts:
            raise ValueError(
                f"C3 {language} {scope} phenomenon counts {observed_counts} "
                f"do not match expected {expected_counts}; use scope='subset' for partial runs"
            )
        else:
            pass
    else:
        pass

    subtask_scores: dict[C3SubtaskName, C3SubtaskMetricSummary] = {}
    for subtask_name in sorted(instances_by_subtask):
        subtask_instances = instances_by_subtask[subtask_name]
        per_judge_subtask: dict[str, MetricRatioCell] = {}
        judge_accuracies: list[float] = []
        any_judge_pending = False
        pending_instance_ids: set[str] = set()
        for judge_name in judge_names:
            judge_verdicts = verdicts_by_judge.get(judge_name, {})
            resolved_verdicts: list[bool] = []
            for instance in subtask_instances:
                if instance.instance_id in judge_verdicts:
                    resolved_verdicts.append(judge_verdicts[instance.instance_id])
                else:
                    pending_instance_ids.add(instance.instance_id)
            pending_count = len(subtask_instances) - len(resolved_verdicts)
            correct_count = sum(1 for verdict in resolved_verdicts if verdict)
            accuracy_value = (
                correct_count / len(subtask_instances)
                if pending_count == 0 and len(subtask_instances) > 0
                else None
            )
            if accuracy_value is None:
                any_judge_pending = True
            else:
                judge_accuracies.append(accuracy_value)
            per_judge_subtask[judge_name] = MetricRatioCell(
                numerator=correct_count,
                denominator=len(subtask_instances),
                pending=pending_count,
                value=accuracy_value,
            )
        if not judge_names:
            pending_instance_ids.update(
                instance.instance_id for instance in subtask_instances
            )
            any_judge_pending = True
        else:
            pass
        mean_subtask_accuracy = (
            sum(judge_accuracies) / len(judge_accuracies)
            if not any_judge_pending and judge_accuracies
            else None
        )
        subtask_scores[subtask_name] = C3SubtaskMetricSummary(
            instances=len(subtask_instances),
            pending_instances=len(pending_instance_ids),
            mean_accuracy=mean_subtask_accuracy,
            by_judge=per_judge_subtask,
        )

    phenomenon_scores: dict[C3Phenomenon, C3PhenomenonMetricSummary] = {}
    per_judge_phenomenon_values: dict[str, list[float]] = {
        judge_name: [] for judge_name in judge_names
    }
    per_judge_complete: dict[str, bool] = {
        judge_name: True for judge_name in judge_names
    }

    present_phenomena = tuple(
        phenomenon
        for phenomenon in C3_PHENOMENA
        if phenomenon in instances_by_phenomenon
    )
    has_human_generation_items = any(
        instance.requires_human_audio_annotation for instance in language_instances
    )
    for phenomenon in present_phenomena:
        phenomenon_instances = instances_by_phenomenon[phenomenon]
        is_branched = phenomenon in ("omission", "coreference")
        per_judge_phenomenon: dict[str, float | None] = {}
        phenomenon_judge_means: list[float] = []
        phenomenon_pending_ids: set[str] = set()
        for judge_name in judge_names:
            judge_verdicts = verdicts_by_judge.get(judge_name, {})
            missing_ids = [
                instance.instance_id
                for instance in phenomenon_instances
                if instance.instance_id not in judge_verdicts
            ]
            phenomenon_pending_ids.update(missing_ids)
            if missing_ids:
                per_judge_complete[judge_name] = False
                per_judge_phenomenon[judge_name] = None
            elif is_branched:
                branch_subtasks = sorted(
                    {instance.subtask for instance in phenomenon_instances}
                )
                branch_accuracies = [
                    sum(
                        1
                        for instance in instances_by_subtask[branch_name]
                        if judge_verdicts[instance.instance_id]
                    )
                    / len(instances_by_subtask[branch_name])
                    for branch_name in branch_subtasks
                ]
                phenomenon_accuracy = sum(branch_accuracies) / len(branch_accuracies)
                per_judge_phenomenon[judge_name] = phenomenon_accuracy
                per_judge_phenomenon_values[judge_name].append(phenomenon_accuracy)
                phenomenon_judge_means.append(phenomenon_accuracy)
            else:
                correct_total = sum(
                    1
                    for instance in phenomenon_instances
                    if judge_verdicts[instance.instance_id]
                )
                phenomenon_accuracy = correct_total / len(phenomenon_instances)
                per_judge_phenomenon[judge_name] = phenomenon_accuracy
                per_judge_phenomenon_values[judge_name].append(phenomenon_accuracy)
                phenomenon_judge_means.append(phenomenon_accuracy)

        if not judge_names:
            phenomenon_pending_ids.update(
                instance.instance_id for instance in phenomenon_instances
            )
        else:
            pass

        mean_phenomenon_accuracy = (
            sum(phenomenon_judge_means) / len(phenomenon_judge_means)
            if len(phenomenon_judge_means) == len(judge_names) and judge_names
            else None
        )
        phonological_scope_label: C3PhonologicalScopeLabel | None = (
            (
                "full_with_human_generation"
                if scope == "full"
                else (
                    "includes_human_generation"
                    if has_human_generation_items
                    else "understanding_only"
                )
            )
            if phenomenon == "phonological"
            else None
        )
        phenomenon_scores[phenomenon] = C3PhenomenonMetricSummary(
            instances=len(phenomenon_instances),
            pending_instances=len(phenomenon_pending_ids),
            aggregation=("mean_of_branches" if is_branched else "micro"),
            phonological_scope=phonological_scope_label,
            mean_accuracy=mean_phenomenon_accuracy,
            by_judge=per_judge_phenomenon,
        )

    all_five_present = len(present_phenomena) == len(C3_PHENOMENA)
    macro_by_judge: dict[str, float | None] = {}
    valid_judge_macros: list[float] = []
    for judge_name in judge_names:
        if (
            all_five_present
            and per_judge_complete[judge_name]
            and len(per_judge_phenomenon_values[judge_name]) == len(C3_PHENOMENA)
        ):
            judge_macro_accuracy = sum(per_judge_phenomenon_values[judge_name]) / len(
                C3_PHENOMENA
            )
            macro_by_judge[judge_name] = judge_macro_accuracy
            valid_judge_macros.append(judge_macro_accuracy)
        else:
            macro_by_judge[judge_name] = None

    mean_macro_accuracy = (
        sum(valid_judge_macros) / len(valid_judge_macros)
        if len(valid_judge_macros) == len(judge_names) and judge_names
        else None
    )

    micro_by_judge: dict[str, MetricRatioCell] = {}
    for judge_name in judge_names:
        judge_verdicts = verdicts_by_judge.get(judge_name, {})
        resolved_flags = [
            judge_verdicts[instance.instance_id]
            for instance in language_instances
            if instance.instance_id in judge_verdicts
        ]
        pending_count = len(language_instances) - len(resolved_flags)
        correct_count = sum(1 for flag in resolved_flags if flag)
        micro_by_judge[judge_name] = MetricRatioCell(
            numerator=correct_count,
            denominator=len(language_instances),
            pending=pending_count,
            value=(
                correct_count / len(language_instances) if pending_count == 0 else None
            ),
        )

    macro_metric_name: C3MacroMetricName
    if not dataset_matches_pinned_revision:
        macro_metric_name = "c3_unverified_dataset_macro_accuracy"
    elif scope == "full":
        macro_metric_name = "c3_paper_full_macro_accuracy"
    elif scope == "auto":
        macro_metric_name = "c3_auto_macro_accuracy"
    else:
        macro_metric_name = "c3_subset_macro_accuracy"

    selected_human_count = sum(
        1 for instance in language_instances if instance.requires_human_audio_annotation
    )
    return C3LanguageMetricSummary(
        language=language,
        scope=scope,
        paper_metric_scope_complete=(
            scope == "full" and dataset_matches_pinned_revision
        ),
        dataset_matches_pinned_revision=dataset_matches_pinned_revision,
        excludes_human_phonological_generation=(selected_human_count == 0),
        selected_human_generation_count=selected_human_count,
        excluded_human_generation_count=(
            C3_HUMAN_GENERATION_COUNTS[language] - selected_human_count
            if scope in ("auto", "single_turn", "full")
            else None
        ),
        instances=len(language_instances),
        included_phenomena=present_phenomena,
        macro_metric_name=macro_metric_name,
        mean_macro_accuracy=mean_macro_accuracy,
        macro_accuracy_by_judge=macro_by_judge,
        micro_accuracy_by_judge=micro_by_judge,
        phenomena=phenomenon_scores,
        subtasks=subtask_scores,
    )
