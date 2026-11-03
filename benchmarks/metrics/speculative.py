# SPDX-License-Identifier: Apache-2.0
"""Server-log stats and the draft-against-no-draft comparison for speculative sweeps."""

from __future__ import annotations

import re
import statistics
from dataclasses import dataclass
from typing import Literal

from pydantic import AliasChoices, BaseModel, Field

Side = Literal["baseline", "draft"]
# The roadmap's k: the highest concurrency where the draft gains at least this much.
K_GAIN = 1.2
# The roadmap's never-worse gate: at every load the draft keeps at least this share of the throughput, and at most this multiple of the p95 latency.
NEVER_WORSE_THROUGHPUT_RATIO = 0.97
NEVER_WORSE_LATENCY_RATIO = 1.05
# note (wilsonzheng0327): the first allocation is the target's pool; a draft then allocates its own with the same token count.
KV_POOL_PATTERN = re.compile(r"KV Cache is allocated\..*?#tokens: (\d+)")
TOKEN_USAGE_PATTERN = re.compile(r"token usage: ([0-9.]+)")
RETRACTION_PATTERN = re.compile(r"#retracted_reqs: (\d+)")
ACCEPT_LENGTH_PATTERN = re.compile(r"accept len: ([0-9.]+)")
PREFILL_TOKENS_PATTERN = re.compile(r"#new-token: (\d+), #cached-token: (\d+)")
ROADMAP_CONCURRENCIES = (1, 4, 8, 16, 32, 64)


class MetricMean(BaseModel):
    mean: float | None


class MetricPeak(BaseModel):
    max: float


class ClientResources(BaseModel):
    gpu_memory_used_peak_mib: MetricPeak | None = None


class ClientResult(BaseModel):
    """One concurrency level as the benchmark_asr_seedtts sweep writes it."""

    concurrency: int
    total: int
    skipped: int
    # note (ruiling): Old client JSON cannot distinguish scoring skips from request failures.
    failed: int = Field(validation_alias=AliasChoices("failed", "skipped"))
    throughput_samples_per_s: MetricMean
    latency_p95_s: MetricMean
    corpus_wer: MetricMean
    resources: ClientResources


class ClientPayload(BaseModel):
    results: list[ClientResult]


@dataclass(kw_only=True)
class ServerLogStats:
    kv_pool_tokens: int | None
    token_usage_peak: float | None
    retracted_requests: int
    accept_length_mean: float | None
    cached_prompt_token_share: float | None


@dataclass(kw_only=True)
class Leg:
    name: str
    side: Side
    startup_s: float
    server: ServerLogStats
    results: list[ClientResult]


@dataclass(kw_only=True)
class SideAtConcurrency:
    """One side's legs at one concurrency, one value per leg that reported it."""

    throughputs: list[float]
    latencies_p95_s: list[float]
    wers: list[float]
    failed_requests: int


@dataclass(kw_only=True)
class ConcurrencyRow:
    concurrency: int
    baseline: SideAtConcurrency
    draft: SideAtConcurrency
    gain: float | None
    latency_ratio: float | None
    is_complete: bool


@dataclass(kw_only=True)
class Comparison:
    rows: list[ConcurrencyRow]
    k: int | None
    worse_concurrencies: list[int]
    incomplete_concurrencies: list[int]
    accept_length_mean: float | None


def mean_or_none(values: list[float]) -> float | None:
    return statistics.mean(values) if values else None


def mean_ratio(draft_values: list[float], baseline_values: list[float]) -> float | None:
    """The draft's mean over the no-draft mean."""
    draft_mean = mean_or_none(draft_values)
    baseline_mean = mean_or_none(baseline_values)
    return (
        draft_mean / baseline_mean if draft_mean is not None and baseline_mean else None
    )


def read_server_log(text: str) -> ServerLogStats:
    """The pool size, usage, retractions, acceptance and prefix-cache share a server logged."""
    pools = KV_POOL_PATTERN.findall(text)
    usages = [float(value) for value in TOKEN_USAGE_PATTERN.findall(text)]
    prefills = [
        (int(new_tokens), int(cached_tokens))
        for new_tokens, cached_tokens in PREFILL_TOKENS_PATTERN.findall(text)
    ]
    prompt_tokens = sum(
        new_tokens + cached_tokens for new_tokens, cached_tokens in prefills
    )
    return ServerLogStats(
        kv_pool_tokens=int(pools[0]) if pools else None,
        token_usage_peak=max(usages, default=None),
        retracted_requests=sum(
            int(value) for value in RETRACTION_PATTERN.findall(text)
        ),
        accept_length_mean=mean_or_none(
            [float(value) for value in ACCEPT_LENGTH_PATTERN.findall(text)]
        ),
        cached_prompt_token_share=(
            sum(cached_tokens for _, cached_tokens in prefills) / prompt_tokens
            if prompt_tokens
            else None
        ),
    )


def compare(legs: list[Leg], *, k_gain: float) -> Comparison:
    concurrencies = [result.concurrency for result in legs[0].results]
    if any(
        [result.concurrency for result in leg.results] != concurrencies for leg in legs
    ):
        raise ValueError("every leg must sweep the same concurrencies")
    else:
        pass
    sides = {
        (side, concurrency): SideAtConcurrency(
            throughputs=[], latencies_p95_s=[], wers=[], failed_requests=0
        )
        for side in ("baseline", "draft")
        for concurrency in concurrencies
    }
    for leg in legs:
        for result in leg.results:
            measured = sides[(leg.side, result.concurrency)]
            for values, value in (
                (measured.throughputs, result.throughput_samples_per_s.mean),
                (measured.latencies_p95_s, result.latency_p95_s.mean),
                (measured.wers, result.corpus_wer.mean),
            ):
                if value is not None:
                    values.append(value)
                else:
                    pass
            measured.failed_requests += result.failed
    rows = []
    for concurrency in concurrencies:
        baseline = sides[("baseline", concurrency)]
        draft = sides[("draft", concurrency)]
        gain = mean_ratio(draft.throughputs, baseline.throughputs)
        latency_ratio = mean_ratio(draft.latencies_p95_s, baseline.latencies_p95_s)
        rows.append(
            ConcurrencyRow(
                concurrency=concurrency,
                baseline=baseline,
                draft=draft,
                gain=gain,
                latency_ratio=latency_ratio,
                # note (wilsonzheng0327): throughput and p95 count only the requests that finished, so a load where any failed cannot qualify.
                is_complete=baseline.failed_requests == 0
                and draft.failed_requests == 0
                and gain is not None
                and latency_ratio is not None,
            )
        )
    complete_rows = [row for row in rows if row.is_complete]
    return Comparison(
        rows=rows,
        k=max(
            (row.concurrency for row in complete_rows if row.gain >= k_gain),
            default=None,
        ),
        worse_concurrencies=[
            row.concurrency
            for row in complete_rows
            if row.gain < NEVER_WORSE_THROUGHPUT_RATIO
            or row.latency_ratio > NEVER_WORSE_LATENCY_RATIO
        ],
        incomplete_concurrencies=[
            row.concurrency for row in rows if not row.is_complete
        ],
        accept_length_mean=mean_or_none(
            [
                leg.server.accept_length_mean
                for leg in legs
                if leg.side == "draft" and leg.server.accept_length_mean is not None
            ]
        ),
    )


def number(value: float | int | None, digits: int, suffix: str = "") -> str:
    return "–" if value is None else f"{value:.{digits}f}{suffix}"


def percent(fraction: float | None) -> str:
    return "–" if fraction is None else f"{fraction * 100:.2f}%"


def gap_percent(values: list[float]) -> float | None:
    """The spread between one side's legs, relative to their mean."""
    return (
        (max(values) - min(values)) / statistics.mean(values) * 100
        if len(values) > 1
        else None
    )


def render(comparison: Comparison, legs: list[Leg], *, title: str) -> str:
    order = ", ".join(leg.name for leg in legs)
    lines = [
        f"# {title}: draft against no draft",
        "",
        f"Legs in order: {order}. Throughput is samples/s, the mean of each side's legs; gap is the spread between a side's legs.",
        "",
        "| c | No draft | Draft | Gain | Gap, no draft | Gap, draft | p95, no draft (s) | p95, draft (s) | p95 change | WER, no draft | WER, draft | Failed, no draft | Failed, draft |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---:|---:|",
    ]
    for row in comparison.rows:
        lines.append(
            f"| {row.concurrency}{'' if row.is_complete else ' (incomplete)'} "
            f"| {number(mean_or_none(row.baseline.throughputs), 2)} "
            f"| {number(mean_or_none(row.draft.throughputs), 2)} "
            f"| {number(row.gain, 2, 'x')} "
            f"| {number(gap_percent(row.baseline.throughputs), 1, '%')} "
            f"| {number(gap_percent(row.draft.throughputs), 1, '%')} "
            f"| {number(mean_or_none(row.baseline.latencies_p95_s), 3)} "
            f"| {number(mean_or_none(row.draft.latencies_p95_s), 3)} "
            f"| {number(None if row.latency_ratio is None else (row.latency_ratio - 1) * 100, 1, '%')} "
            f"| {', '.join(percent(wer) for wer in row.baseline.wers) or '–'} "
            f"| {', '.join(percent(wer) for wer in row.draft.wers) or '–'} "
            f"| {row.baseline.failed_requests} "
            f"| {row.draft.failed_requests} |"
        )
    lines += [
        "",
        f"- k, the highest complete concurrency with at least {K_GAIN}x: {comparison.k if comparison.k is not None else 'none'}",
        f"- Worse than no draft (throughput under {NEVER_WORSE_THROUGHPUT_RATIO}x or p95 over {NEVER_WORSE_LATENCY_RATIO}x): "
        f"{', '.join(f'c{concurrency}' for concurrency in comparison.worse_concurrencies) or 'none'}",
        "- Incomplete, so left out of k and the never-worse check (failed requests or missing numbers): "
        f"{', '.join(f'c{concurrency}' for concurrency in comparison.incomplete_concurrencies) or 'none'}",
        "- Prompt tokens served from the prefix cache, which repeated prompts inflate: "
        + ", ".join(
            f"{leg.name} {percent(leg.server.cached_prompt_token_share)}"
            for leg in legs
        ),
    ]
    baseline_wers = [wer for row in comparison.rows for wer in row.baseline.wers]
    draft_wers = [wer for row in comparison.rows for wer in row.draft.wers]
    lines += [
        f"- WER across legs and concurrencies: no draft {percent(min(baseline_wers, default=None))} to {percent(max(baseline_wers, default=None))}, "
        f"draft {percent(min(draft_wers, default=None))} to {percent(max(draft_wers, default=None))}",
        "",
        "| Leg | Startup (s) | KV pool (tokens) | Peak token usage | Retractions | Accept len | Peak GPU memory (MiB) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for leg in legs:
        peaks = [
            result.resources.gpu_memory_used_peak_mib.max
            for result in leg.results
            if result.resources.gpu_memory_used_peak_mib is not None
        ]
        lines.append(
            f"| {leg.name} | {leg.startup_s:.1f} | {number(leg.server.kv_pool_tokens, 0)} "
            f"| {number(leg.server.token_usage_peak, 2)} | {leg.server.retracted_requests} "
            f"| {number(leg.server.accept_length_mean, 2)} | {number(max(peaks, default=None), 0)} |"
        )
    rows_by_concurrency = {row.concurrency: row for row in comparison.rows}
    roadmap_cells = []
    for concurrency in ROADMAP_CONCURRENCIES:
        row = rows_by_concurrency.get(concurrency)
        if row is None:
            roadmap_cells.append("–")
        elif row.is_complete:
            roadmap_cells.append(number(row.gain, 2, "x"))
        else:
            roadmap_cells.append("invalid")

    lines += [
        "",
        "Roadmap row:",
        "",
        f"| {title} | | {number(comparison.accept_length_mean, 1)} | {' | '.join(roadmap_cells)} "
        f"| WER {percent(mean_or_none(baseline_wers))} → {percent(mean_or_none(draft_wers))} | |",
    ]
    return "\n".join(lines) + "\n"
