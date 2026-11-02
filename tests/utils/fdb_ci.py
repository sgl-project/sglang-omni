# SPDX-License-Identifier: Apache-2.0
"""Full-Duplex-Bench CI: run one strategy's fixed selection, read its metrics, gate regressions."""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import statistics
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from huggingface_hub import snapshot_download
from pydantic import JsonValue

from benchmarks.duplex.fdb.common import JUDGE_MODEL_ID
from benchmarks.duplex.fdb.engines import ENGINES, EngineName
from tests.utils import MetricCheckCollector, apply_mos_slack

V15_PER_SUBSET = 5
V10_PER_SUBSET = 5
# note (jeffro): no CI metric reads icc_backchannel, which only the official
# evaluator scores.
V10_SUBSETS = (
    "synthetic_pause_handling",
    "candor_pause_handling",
    "candor_turn_taking",
    "synthetic_user_interruption",
)
V15_CATEGORIES = (
    "user_interruption",
    "user_backchannel",
    "talking_to_other",
    "background_speech",
)
JUDGE_REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
# note (jeffro): a recorder that sends an input chunk late fails its session on
# the client side; the server's protocol verdict on it still has to pass.
MAX_CLIENT_PACING_FAILURES = 1
# note (jeffro): random flips across the five-case rates mostly cancel while a
# regression moves many cases at once, so the rates share one slack; a slack of
# one case per rate failed unchanged code in most repeated H100 runs.
POOLED_RATE_TOLERANCE = 5
# note (jeffro): the pool can hide one rate losing every case, and random flips
# rarely empty a rate whose reference met at least this many cases.
RATE_FLOOR_REFERENCE_CASES = 3
# note (jeffro): a reference is the mean count over repeated runs divided by n,
# kept to four decimals, so cases on target stay fractional and are rounded only
# to drop float error.
REFERENCE_CASE_DECIMALS = 4
# note (jeffro): a pair whose model was silent when its event began never tested
# the overlap; such a run still scores well, since nothing was there to go wrong.
MIN_EXERCISED_PAIRS_PER_CATEGORY = 3
MIN_EXERCISED_PAIRS = 15
LATENCY_TOLERANCE = 1.25
OMISSION_RATE_TOLERANCE = 0.02
VOICE_SIMILARITY_TOLERANCE = 3.0

MetricKind = Literal["rate", "latency_s", "omission_rate", "utmos", "voice_similarity"]


@dataclass(frozen=True, kw_only=True)
class MetricSpec:
    kind: MetricKind
    is_higher_better: bool
    needs_semantic_judge: bool = False
    is_gated: bool = True


METRIC_SPECS: dict[str, MetricSpec] = {
    "v15_respond_on_interruption": MetricSpec(kind="rate", is_higher_better=True),
    "v15_resume_on_backchannel": MetricSpec(kind="rate", is_higher_better=True),
    "v15_resume_on_talking_to_other": MetricSpec(kind="rate", is_higher_better=True),
    "v15_resume_on_background_speech": MetricSpec(kind="rate", is_higher_better=True),
    # note (jeffro): reported, not gated, until repeated runs show the semantic
    # judge's verdicts on one recording are stable.
    "v15_overlap_handled": MetricSpec(
        kind="rate", is_higher_better=True, needs_semantic_judge=True, is_gated=False
    ),
    "v15_grounded_reply": MetricSpec(
        kind="rate", is_higher_better=True, needs_semantic_judge=True, is_gated=False
    ),
    "v15_stop_latency_mean_s": MetricSpec(kind="latency_s", is_higher_better=False),
    "v15_response_latency_mean_s": MetricSpec(kind="latency_s", is_higher_better=False),
    "v10_synthetic_pause_intrusion_rate": MetricSpec(
        kind="rate", is_higher_better=False
    ),
    "v10_candor_pause_intrusion_rate": MetricSpec(kind="rate", is_higher_better=False),
    "v10_turn_response_rate_2s": MetricSpec(kind="rate", is_higher_better=True),
    "listening_omission_rate": MetricSpec(kind="omission_rate", is_higher_better=False),
    "listening_utmos_mean": MetricSpec(kind="utmos", is_higher_better=True),
    "listening_voice_similarity_mean": MetricSpec(
        kind="voice_similarity", is_higher_better=True
    ),
}
BEHAVIOR_METRICS = {
    "v15_respond_on_interruption": ("user_interruption", "C_RESPOND"),
    "v15_resume_on_backchannel": ("user_backchannel", "C_RESUME"),
    "v15_resume_on_talking_to_other": ("talking_to_other", "C_RESUME"),
    "v15_resume_on_background_speech": ("background_speech", "C_RESUME"),
}
SEMANTIC_METRICS = {
    "v15_overlap_handled": "interaction_handling",
    "v15_grounded_reply": "grounding",
}
UNIFIED_METRICS = {
    "v10_synthetic_pause_intrusion_rate": (
        "synthetic_pause_handling",
        "pause_intrusion_rate",
    ),
    "v10_candor_pause_intrusion_rate": (
        "candor_pause_handling",
        "pause_intrusion_rate",
    ),
    "v10_turn_response_rate_2s": ("candor_turn_taking", "response_rate_2s"),
}
LISTENING_METRICS = {
    "listening_omission_rate": "omission_rate",
    "listening_utmos_mean": "utmos_mean",
    "listening_voice_similarity_mean": "voice_similarity_mean",
}


@dataclass(frozen=True, kw_only=True)
class Measurement:
    """A metric value; a rate also keeps its count over n."""

    value: float
    count: int | None = None
    n: int | None = None


@dataclass(frozen=True, kw_only=True)
class FdbCiStrategy:
    """One strategy's CI run: its engine, how its sessions spread over the GPUs
    and whether the semantic judge grades it."""

    engine_name: EngineName
    gpu_arguments: tuple[str, ...]
    num_shards: int
    should_run_semantic_judge: bool

    @property
    def run_name(self) -> str:
        return f"ci-{self.engine_name}"


def read_json(path: Path) -> dict[str, JsonValue]:
    return json.loads(path.read_text())


def runbook(env: dict[str, str], *arguments: str) -> None:
    subprocess.run(
        [sys.executable, "-m", "benchmarks.duplex.fdb", *arguments],
        env=env,
        check=True,
    )


def link_judge_model(fdb_work: Path) -> None:
    """Point the runbook's judge path at the shared Hugging Face cache, pinned,
    so setup finds it instead of downloading 27B into every CI home."""
    judge_path = fdb_work / "models" / "Qwen3.8-27B"
    judge_path.parent.mkdir(parents=True, exist_ok=True)
    if not judge_path.exists():
        judge_path.symlink_to(
            snapshot_download(JUDGE_MODEL_ID, revision=JUDGE_REVISION)
        )
    else:
        pass
    judge_path.with_name(judge_path.name + ".revision").write_text(
        f"{JUDGE_REVISION}\n"
    )


def run_fdb_ci(strategy: FdbCiStrategy) -> Path:
    """Set up, record, score and judge the strategy's selection; returns the run root."""
    engine = ENGINES[strategy.engine_name]
    fdb_work = Path(os.environ["OMNI_CI_HOME"]) / "fdb"
    env = {
        **os.environ,
        "FDB_WORK": str(fdb_work),
        "ENGINE": engine.name,
        "JUDGE": "qwen",
    }
    if engine.launch_config is None:
        env["MODEL_PATH"] = snapshot_download(
            engine.model_id, revision=engine.model_revision
        )
    else:
        pass
    link_judge_model(fdb_work)
    run_root = fdb_work / "runs" / strategy.run_name
    # note (jeffro): generate refuses a repeat that already holds a recording,
    # so a retried attempt starts from an empty run.
    if run_root.exists():
        shutil.rmtree(run_root)
    else:
        pass
    runbook(env, "setup")
    runbook(
        env,
        "generate",
        "--run-name",
        strategy.run_name,
        "--v15-per-subset",
        str(V15_PER_SUBSET),
        "--v10-per-subset",
        str(V10_PER_SUBSET),
        "--v10-subset-count",
        "icc_backchannel=0",
        "--num-shards",
        str(strategy.num_shards),
        *strategy.gpu_arguments,
        "--all-stages",
        *([] if strategy.should_run_semantic_judge else ["--skip-semantic-judge"]),
    )
    return run_root


def variant_outcomes(recording: Path) -> Counter[str]:
    """Counts of pass, not_exercised (kept the protocol, but no output overlapped
    the input), client_pacing (failed only by late client sends) and
    protocol_failure over every recorded session."""
    outcomes: Counter[str] = Counter()
    for run in recording.glob("shard-*/run.json"):
        for sample in read_json(run)["samples"]:
            for variant in sample["variants"].values():
                if variant["status"] in ("pass", "not_exercised"):
                    outcomes[variant["status"]] += 1
                elif (
                    variant["protocol_verdict"] == "pass"
                    and variant["errors"]
                    and all(
                        error.startswith("input pacing deviation")
                        for error in variant["errors"]
                    )
                ):
                    outcomes["client_pacing"] += 1
                else:
                    outcomes["protocol_failure"] += 1
    return outcomes


def is_speaking_at(segments: JsonValue, time_s: float) -> bool:
    return any(start_s <= time_s < end_s for start_s, end_s in segments)


def count_exercised_overlaps(run_root: Path, engine_name: EngineName) -> Counter[str]:
    """Per v1.5 category, the overlap pairs whose model was already speaking when
    the event began, the moment the category tests."""
    repeat = run_root / "repeat-1"
    event_starts_s: dict[str, float] = {}
    for run in (repeat / "recording").glob("shard-*/run.json"):
        for sample in read_json(run)["samples"]:
            span = sample["event_span_s"]
            assert span is not None, f"{sample['id']}: no event timestamps"
            event_starts_s[sample["id"]] = finite_measurement(sample["id"], span[0])
    assert len(event_starts_s) == V15_PER_SUBSET * len(V15_CATEGORIES), sorted(
        event_starts_s
    )
    samples = repeat / "scores" / "engines" / engine_name / "samples"
    exercised: Counter[str] = Counter({category: 0 for category in V15_CATEGORIES})
    for sample_id, event_start_s in event_starts_s.items():
        receipt = read_json(samples / sample_id / "receipts" / "timing-overlap.json")
        if is_speaking_at(receipt["model_segments"], event_start_s):
            exercised[sample_id.split("/")[0]] += 1
        else:
            pass
    return exercised


def check_overlaps_exercised(
    checks: MetricCheckCollector, exercised: Counter[str]
) -> None:
    """Record a run whose model was too often silent when the events began to
    have tested the overlaps at all."""
    for category in V15_CATEGORIES:
        checks.check(
            exercised[category] >= MIN_EXERCISED_PAIRS_PER_CATEGORY,
            f"{category}: overlap exercised in {exercised[category]}/"
            f"{V15_PER_SUBSET} pairs < {MIN_EXERCISED_PAIRS_PER_CATEGORY}",
        )
    total = sum(exercised.values())
    pairs = V15_PER_SUBSET * len(V15_CATEGORIES)
    checks.check(
        total >= MIN_EXERCISED_PAIRS,
        f"overlap exercised in {total}/{pairs} pairs < {MIN_EXERCISED_PAIRS}",
    )


def finite_measurement(name: str, value: JsonValue) -> float:
    assert (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    ), f"{name}: missing or nonfinite value {value!r}"
    return float(value)


def rate_measurement(name: str, count: JsonValue, n: JsonValue) -> Measurement:
    assert isinstance(n, int) and n > 0, f"{name}: no scored samples, n={n!r}"
    assert isinstance(count, int) and 0 <= count <= n, f"{name}: {count!r}/{n}"
    return Measurement(value=count / n, count=count, n=n)


def collect_fdb_metrics(
    run_root: Path, engine_name: EngineName, should_run_semantic_judge: bool
) -> dict[str, Measurement]:
    """Every CI metric of the run's first repeat; asserts the selection was
    recorded and scored in full before reading a value."""
    repeat = run_root / "repeat-1"
    pairs = V15_PER_SUBSET * len(V15_CATEGORIES)
    v10_samples = V10_PER_SUBSET * len(V10_SUBSETS)
    outcomes = variant_outcomes(repeat / "recording") + variant_outcomes(
        repeat / "v10" / "recording"
    )
    assert outcomes["protocol_failure"] == 0, f"protocol failures: {dict(outcomes)}"
    assert (
        outcomes["client_pacing"] <= MAX_CLIENT_PACING_FAILURES
    ), f"client pacing failures: {dict(outcomes)}"
    assert sum(outcomes.values()) == 2 * pairs + v10_samples, dict(outcomes)
    metrics: dict[str, Measurement] = {}

    behavior = read_json(repeat / "judge-qwen" / "summary.json")["engines"][engine_name]
    for name, (category, label) in BEHAVIOR_METRICS.items():
        proportion = behavior[category]["valid_label_proportions"][label]
        metrics[name] = rate_measurement(
            name, proportion["count"], behavior[category]["valid_n"]
        )

    semantic_path = repeat / "semantic-qwen" / "summary.json"
    if should_run_semantic_judge:
        axes = read_json(semantic_path)["categories"]["all"]
        for name, axis in SEMANTIC_METRICS.items():
            # note (jeffro): an unresolved case counts against the rate; left
            # out, a judge that resolves almost nothing would shrink n until
            # the one-case slack lets any result pass.
            counts = axes[axis]
            metrics[name] = rate_measurement(
                name,
                counts["accepted"],
                counts["accepted"] + counts["failed"] + counts["unresolved"],
            )
    else:
        assert not semantic_path.exists(), f"{semantic_path} exists"

    samples = repeat / "scores" / "engines" / engine_name / "samples"
    receipts = list(samples.glob("*/*/receipts/timing-overlap.json"))
    assert Counter(read_json(path)["status"] for path in receipts) == {
        "ok": pairs
    }, f"timing receipts: {len(receipts)} of {pairs}"
    intervals: dict[str, list[float]] = {
        "latency_stop_list": [],
        "latency_resp_list": [],
    }
    for path in receipts:
        interval_path = path.parent.parent / "latency_intervals.json"
        assert (
            hashlib.sha256(interval_path.read_bytes()).hexdigest()
            == read_json(path)["intervals_sha256"]
        ), f"Timing intervals changed after scoring: {interval_path}"
        document = read_json(interval_path)
        for key, values in intervals.items():
            values.extend(
                finite_measurement(key, end) - finite_measurement(key, start)
                for start, end in document[key]
            )
    for name, key in (
        ("v15_stop_latency_mean_s", "latency_stop_list"),
        ("v15_response_latency_mean_s", "latency_resp_list"),
    ):
        assert intervals[key], f"{name}: no timing intervals"
        assert all(value >= 0 for value in intervals[key]), f"{name}: negative interval"
        metrics[name] = Measurement(value=statistics.fmean(intervals[key]))

    unified = read_json(repeat / "v10" / "reference" / "unified.json")["subsets"]
    assert set(unified) == set(V10_SUBSETS), sorted(unified)
    for subset, entry in unified.items():
        assert entry["counts"]["selected"] == V10_PER_SUBSET, subset
    for name, (subset, metric) in UNIFIED_METRICS.items():
        rate = unified[subset]["metrics"][metric]
        metrics[name] = rate_measurement(name, rate["count"], rate["n"])
        assert math.isclose(
            finite_measurement(name, rate["value"]), metrics[name].value
        ), f"{name}: inconsistent rate"

    listening = read_json(repeat / "listening" / "summary.json")["datasets"]["v1.5"][
        "all"
    ]
    for name, key in LISTENING_METRICS.items():
        metrics[name] = Measurement(value=finite_measurement(name, listening[key]))
    assert (run_root / "RESULTS.md").is_file()
    return metrics


def regression_bound(name: str, reference: float) -> float:
    """The worst value the gate accepts for a continuous metric whose baseline is reference."""
    spec = METRIC_SPECS[name]
    if spec.kind == "latency_s":
        return reference * LATENCY_TOLERANCE
    elif spec.kind == "omission_rate":
        return reference + OMISSION_RATE_TOLERANCE
    elif spec.kind == "utmos":
        return apply_mos_slack(reference)
    else:
        return reference - VOICE_SIMILARITY_TOLERANCE


def cases_on_target(spec: MetricSpec, count: float, n: int) -> float:
    """Cases of a rate that went the way the metric wants: the counted ones when
    higher is better, the rest when lower is better."""
    if spec.is_higher_better:
        return round(count, REFERENCE_CASE_DECIMALS)
    else:
        return round(n - count, REFERENCE_CASE_DECIMALS)


def rate_cases(name: str) -> int:
    """The cases every run scores for a gated rate."""
    if name in BEHAVIOR_METRICS:
        return V15_PER_SUBSET
    else:
        return V10_PER_SUBSET


def check_no_regression(
    checks: MetricCheckCollector,
    metrics: dict[str, Measurement],
    references: dict[str, float],
) -> None:
    """Record every gated metric that is unmeasured or worse than its bound. The
    rates are gated by their cases on target together, and each by a floor."""
    pooled_reference_cases = 0.0
    pooled_observed_cases = pooled_n = 0
    for name, reference in references.items():
        spec = METRIC_SPECS[name]
        if not spec.is_gated:
            pass
        elif name not in metrics:
            checks.fail(f"{name}: unmeasured")
        elif spec.kind == "rate":
            measurement = metrics[name]
            assert measurement.count is not None and measurement.n is not None
            n = rate_cases(name)
            # note (jeffro): a case the judge could not score would shrink n and
            # shift the pool, so the rate fails instead.
            checks.check(
                measurement.n == n,
                f"{name}: scored {measurement.n} of {n} cases",
            )
            reference_cases = cases_on_target(spec, reference * n, n)
            observed_cases = int(cases_on_target(spec, measurement.count, n))
            checks.check(
                reference_cases < RATE_FLOOR_REFERENCE_CASES or observed_cases > 0,
                f"{name}: {measurement.count}/{n} collapsed from "
                f"{reference * n:g}/{n}",
            )
            pooled_reference_cases += reference_cases
            pooled_observed_cases += observed_cases
            pooled_n += n
        else:
            value = metrics[name].value
            bound = regression_bound(name, reference)
            if spec.is_higher_better:
                checks.check(value >= bound, f"{name}: {value:.4g} < {bound:.4g}")
            else:
                checks.check(value <= bound, f"{name}: {value:.4g} > {bound:.4g}")
    pooled_bound = max(
        0.0,
        round(pooled_reference_cases - POOLED_RATE_TOLERANCE, REFERENCE_CASE_DECIMALS),
    )
    checks.check(
        pooled_observed_cases >= pooled_bound,
        f"pooled rates: {pooled_observed_cases}/{pooled_n} cases on target "
        f"< {pooled_bound:g}/{pooled_n}",
    )
