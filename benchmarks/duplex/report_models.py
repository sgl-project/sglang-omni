# SPDX-License-Identifier: Apache-2.0
"""Read the fields used by the offline benchmark report."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, NonNegativeInt

VariantName = Literal["overlap", "clean"]


class ProtocolDiagnostics(BaseModel):
    status: str
    protocol_verdict: str | None = None


class TraceIdentity(BaseModel):
    trace_sha256: str | None = None


class ReportVariant(BaseModel):
    eligible: bool
    reasons: list[str]
    source: TraceIdentity | None = None
    protocol_diagnostics: ProtocolDiagnostics | None = None


class ReportSample(BaseModel):
    sample_id: str
    category: str
    variants: dict[VariantName, ReportVariant]


class ReportManifest(BaseModel):
    samples: list[ReportSample]


class ManifestReceipt(BaseModel):
    engine: str
    projected_manifest_sha256: str


class SummaryPopulation(BaseModel):
    manifest_sha256: str
    sample_ids: list[str]


class MeanBootstrap(BaseModel):
    ci95: tuple[float, float] | None


class IntervalDistribution(BaseModel):
    interval_n: NonNegativeInt
    samples: NonNegativeInt
    zero_interval_samples: NonNegativeInt
    mean_s: float | None
    median_s: float | None
    bootstrap_pooled_mean: MeanBootstrap


class IntervalSummaries(BaseModel):
    stop: IntervalDistribution
    response: IntervalDistribution


class TimingSummary(BaseModel):
    population: NonNegativeInt
    status: dict[str, NonNegativeInt]
    official_all_intervals: IntervalSummaries


class LabelProportion(BaseModel):
    count: NonNegativeInt


class BehaviorSummary(BaseModel):
    population: NonNegativeInt
    status: dict[str, NonNegativeInt]
    valid_n: NonNegativeInt
    valid_label_proportions: dict[str, LabelProportion]


class ScoreGroup(BaseModel):
    asr_files: dict[str, NonNegativeInt]
    timing_official_overlap: TimingSummary
    timing_supplementary_clean: TimingSummary
    behavior: BehaviorSummary


class ReferenceSummary(BaseModel):
    reference_revision: str
    engines: dict[str, dict[str, ScoreGroup]]
    populations: dict[str, SummaryPopulation] = Field(default_factory=dict)


class ReplaySession(BaseModel):
    sample: str
    variant: VariantName
    recorded_status: str
    status: str


class ReplayReceipt(BaseModel):
    selected: NonNegativeInt
    replayed: NonNegativeInt
    samples: list[ReplaySession]


class SemanticAxis(BaseModel):
    selected: NonNegativeInt
    accepted: NonNegativeInt
    failed: NonNegativeInt
    unresolved: NonNegativeInt


class SemanticSummary(BaseModel):
    scope: str
    inputs_sha256: str
    overall_axes: dict[str, SemanticAxis]
    overall_joint: SemanticAxis
    uncertainty_note: str
