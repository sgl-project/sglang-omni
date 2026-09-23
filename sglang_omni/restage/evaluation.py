"""Joint request-level SLO evaluation for measured Restage trials."""

import math
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class Observation:
    request_id: str
    scheduled_s: float
    completed_s: float | None
    success: bool
    quality_pass: bool | None
    first_output_s: float | None = None
    audio_duration_s: float | None = None
    max_playback_underrun_s: float | None = None


@dataclass(frozen=True)
class QualityReport:
    """Per-request transcription verdicts plus the corpus-level WER gate."""

    verdicts: dict[str, bool]
    corpus_wer: float | None
    corpus_pass: bool
    max_wer: float


@dataclass(frozen=True)
class SLO:
    max_latency_s: float | None = None
    max_ttfa_s: float | None = None
    max_rtf: float | None = None
    max_underrun_s: float | None = None
    min_good_fraction: float = 0.99

    def __post_init__(self):
        if not 0 < self.min_good_fraction <= 1:
            raise ValueError("min_good_fraction must be in (0, 1]")
        for limit in (
            self.max_latency_s,
            self.max_ttfa_s,
            self.max_rtf,
            self.max_underrun_s,
        ):
            if limit is not None and (not math.isfinite(limit) or limit < 0):
                raise ValueError("SLO limits must be finite and nonnegative")


@dataclass(frozen=True)
class Evaluation:
    feasible: bool
    expected_requests: int
    successful_requests: int
    good_requests: int
    missing_requests: int
    good_fraction: float
    completed_qps: float
    goodput_qps: float
    violations: dict[str, int]
    corpus_wer: float | None = None
    corpus_quality_pass: bool | None = None


def _valid(value):
    return value is not None and math.isfinite(value) and value >= 0


def evaluate(
    observations: Sequence[Observation],
    slo: SLO,
    *,
    expected_requests: int,
    elapsed_s: float,
    corpus_wer: float | None = None,
    corpus_quality_pass: bool | None = None,
) -> Evaluation:
    """Count joint successes against the entire offered cohort.

    Timestamps share a monotonic clock and latency starts at the scheduled
    arrival, including dispatch lag. elapsed_s includes final drain. Quality
    must be evaluated by the workload's quality adapter; None is unmeasured.
    A request passes when it was transcribed; transcript accuracy is gated at
    corpus level, so a failed ``corpus_quality_pass`` makes the trial
    infeasible. This point estimate does not establish repeatability or
    confidence bounds.
    """
    if expected_requests < 1 or expected_requests < len(observations):
        raise ValueError("expected_requests must cover a nonempty offered cohort")
    if not math.isfinite(elapsed_s) or elapsed_s <= 0:
        raise ValueError("elapsed_s must be finite and positive")
    if len({item.request_id for item in observations}) != len(observations):
        raise ValueError("Duplicate request_id in measured observations")
    missing = expected_requests - len(observations)
    violations = Counter({"missing_request": missing}) if missing else Counter()
    successful = good = 0
    for item in observations:
        if not item.success:
            violations["request_failed"] += 1
            continue
        latency = (
            item.completed_s - item.scheduled_s
            if item.completed_s is not None
            else None
        )
        if not _valid(latency):
            violations["latency_missing_or_invalid"] += 1
            continue
        successful += 1
        failed = []
        if item.quality_pass is not True:
            failed.append("quality_failed_or_unmeasured")
        ttfa = (
            item.first_output_s - item.scheduled_s
            if item.first_output_s is not None
            else None
        )
        if ttfa is not None and ttfa > latency:
            ttfa = None
        duration = item.audio_duration_s
        rtf = latency / duration if _valid(duration) and duration > 0 else None
        for name, value, limit in (
            ("latency", latency, slo.max_latency_s),
            ("ttfa", ttfa, slo.max_ttfa_s),
            ("rtf", rtf, slo.max_rtf),
            ("underrun", item.max_playback_underrun_s, slo.max_underrun_s),
        ):
            if limit is None:
                continue
            if not _valid(value):
                failed.append(f"{name}_missing_or_invalid")
            elif value > limit:
                failed.append(name)
        if failed:
            violations.update(failed)
        else:
            good += 1
    fraction = good / expected_requests
    return Evaluation(
        feasible=fraction >= slo.min_good_fraction and corpus_quality_pass is not False,
        expected_requests=expected_requests,
        successful_requests=successful,
        good_requests=good,
        missing_requests=missing,
        good_fraction=fraction,
        completed_qps=successful / elapsed_s,
        goodput_qps=good / elapsed_s,
        violations=dict(violations),
        corpus_wer=corpus_wer,
        corpus_quality_pass=corpus_quality_pass,
    )
