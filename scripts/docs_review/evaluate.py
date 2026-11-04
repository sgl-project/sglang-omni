# SPDX-License-Identifier: Apache-2.0
"""Score a labeled documentation-review set without showing future evidence."""

from __future__ import annotations

import math
from datetime import datetime

from pydantic import BaseModel, Field


class EvaluationMaterial(BaseModel):
    available_at: datetime
    body: str


class EvaluationDefect(BaseModel):
    rule_id: str
    file_path: str


class EvaluationSample(BaseModel):
    sample_id: str
    review_cutoff_time: datetime
    labeled_defects: list[EvaluationDefect]
    predicted_defects: list[EvaluationDefect]
    materials: list[EvaluationMaterial]


class EvaluationThresholds(BaseModel):
    precision_minimum: float = Field(ge=0, le=1)
    recall_minimum: float = Field(ge=0, le=1)


class EvaluationReport(BaseModel):
    sample_count: int
    labeled_defect_count: int
    true_positive_count: int
    false_positive_count: int
    false_negative_count: int
    precision: float
    recall: float
    precision_low: float
    precision_high: float
    recall_low: float
    recall_high: float
    false_positives_per_pull_request: float
    meets_precision_threshold: bool
    meets_recall_threshold: bool
    preliminary_signal: bool
    limitation: str
    visible_material_by_sample: dict[str, str]


EVALUATION_LIMITATION = (
    "These figures are a preliminary signal, not a final verdict, including at 30 to 50 pull requests. "
    "The interval shows the uncertainty at this sample size. "
    "The model may have seen public pull requests during training. "
    "Only materials available at the review cutoff are visible."
)


def evaluate_samples(
    samples: list[EvaluationSample], thresholds: EvaluationThresholds
) -> EvaluationReport:
    true_positive_count = 0
    false_positive_count = 0
    false_negative_count = 0
    visible: dict[str, str] = {}
    for sample in samples:
        labeled = defect_keys(sample.labeled_defects)
        predicted = defect_keys(sample.predicted_defects)
        true_positive_count += len(labeled & predicted)
        false_positive_count += len(predicted - labeled)
        false_negative_count += len(labeled - predicted)
        visible[sample.sample_id] = visible_material(sample)
    predicted_positive_count = true_positive_count + false_positive_count
    labeled_defect_count = true_positive_count + false_negative_count
    precision = ratio(true_positive_count, predicted_positive_count)
    recall = ratio(true_positive_count, labeled_defect_count)
    precision_low, precision_high = wilson_interval(
        true_positive_count, predicted_positive_count
    )
    recall_low, recall_high = wilson_interval(true_positive_count, labeled_defect_count)
    sample_count = len(samples)
    if sample_count:
        false_positives_per_pull_request = false_positive_count / sample_count
    else:
        false_positives_per_pull_request = 0.0
    return EvaluationReport(
        sample_count=sample_count,
        labeled_defect_count=labeled_defect_count,
        true_positive_count=true_positive_count,
        false_positive_count=false_positive_count,
        false_negative_count=false_negative_count,
        precision=precision,
        recall=recall,
        precision_low=precision_low,
        precision_high=precision_high,
        recall_low=recall_low,
        recall_high=recall_high,
        false_positives_per_pull_request=false_positives_per_pull_request,
        meets_precision_threshold=sample_count > 0
        and precision >= thresholds.precision_minimum,
        meets_recall_threshold=sample_count > 0 and recall >= thresholds.recall_minimum,
        preliminary_signal=True,
        limitation=EVALUATION_LIMITATION,
        visible_material_by_sample=visible,
    )


def visible_material(sample: EvaluationSample) -> str:
    bodies = [
        material.body
        for material in sample.materials
        if material.available_at <= sample.review_cutoff_time
    ]
    return "\n".join(bodies)


def defect_keys(defects: list[EvaluationDefect]) -> set[tuple[str, str]]:
    return {(defect.rule_id, defect.file_path) for defect in defects}


def ratio(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    else:
        return numerator / denominator


def wilson_interval(success_count: int, sample_count: int) -> tuple[float, float]:
    if sample_count == 0:
        return (0.0, 0.0)
    else:
        z_score = 1.96
        proportion = success_count / sample_count
        z_squared = z_score * z_score
        denominator = 1 + z_squared / sample_count
        center = (proportion + z_squared / (2 * sample_count)) / denominator
        margin = (
            z_score
            * math.sqrt(
                (proportion * (1 - proportion) + z_squared / (4 * sample_count))
                / sample_count
            )
            / denominator
        )
        return (max(0.0, center - margin), min(1.0, center + margin))
