# SPDX-License-Identifier: Apache-2.0
"""Server-log stats and the draft-against-no-draft comparison for speculative sweeps."""

from __future__ import annotations

import re
import statistics
from dataclasses import dataclass, field
from typing import Literal

Side = Literal["baseline", "draft"]
AccuracyMetric = Literal["WER", "CER", "DER", "Accuracy"]
# The roadmap's k: the highest concurrency where the draft gains at least this much.
K_GAIN = 1.2
# note (wilsonzheng0327): the roadmap's never-worse gate; at every load the draft keeps at least this share of the throughput, and at most this multiple of the p95 latency and of the p95 first-output latency.
NEVER_WORSE_THROUGHPUT_RATIO = 0.97
NEVER_WORSE_LATENCY_RATIO = 1.05
# note (daii-0818): first output also gets the roadmap's 5 ms absolute allowance.
FIRST_OUTPUT_LATENCY_SLACK_S = 0.005
# note (wilsonzheng0327): the first allocation is the target's pool; a draft then allocates its own with the same token count.
KV_POOL_PATTERN = re.compile(r"KV Cache is allocated\..*?#tokens: (\d+)")
TOKEN_USAGE_PATTERN = re.compile(r"token usage: ([0-9.]+)")
RETRACTION_PATTERN = re.compile(r"#retracted_reqs: (\d+)")
ACCEPT_LENGTH_PATTERN = re.compile(r"accept len: ([0-9.]+)")
PREFILL_TOKENS_PATTERN = re.compile(r"#new-token: (\d+), #cached-token: (\d+)")
ROADMAP_CONCURRENCIES = (1, 4, 8, 16, 32, 64)
DECODE_BATCH_PATTERN = re.compile(
    r"Decode batch, #running-req: (\d+),.*cuda graph: (True|False), gen throughput \(token/s\): ([0-9.]+)"
)
# Inclusive (lowest, highest) numbers of requests running in one decode batch; no highest means no upper limit.
RUNNING_REQUEST_RANGES: tuple[tuple[int, int | None], ...] = (
    (1, 1),
    (2, 4),
    (5, 8),
    (9, 16),
    (17, 32),
    (33, 64),
    (65, None),
)


@dataclass(kw_only=True)
class LoadMeasurement:
    """One leg at one concurrency; each number is the mean over the leg's repeats."""

    concurrency: int
    total_requests: int
    failed_requests: int
    throughput_requests_per_s: float | None
    latency_p95_s: float | None
    first_output_latency_p95_s: float | None
    accuracy: dict[AccuracyMetric, float]
    gpu_memory_peak_mib: float | None
    incomplete_reasons: list[str] = field(default_factory=list)


@dataclass(kw_only=True)
class ServerLogStats:
    kv_pool_tokens: int | None
    token_usage_peak: float | None
    retracted_requests: int
    accept_length_mean: float | None
    cached_prompt_token_share: float | None


@dataclass(kw_only=True)
class DecodeBatchLine:
    running_requests: int
    accept_length: float | None
    used_cuda_graph: bool
    gen_throughput_tokens_per_s: float


@dataclass(kw_only=True)
class Leg:
    name: str
    side: Side
    startup_s: float
    server: ServerLogStats
    loads: list[LoadMeasurement]
    decode_batches: list[DecodeBatchLine] = field(default_factory=list)


@dataclass(kw_only=True)
class DecodeRow:
    lowest_running_requests: int
    highest_running_requests: int | None
    baseline_lines: int
    draft_lines: int
    baseline_gen_throughput_tokens_per_s: float | None
    draft_gen_throughput_tokens_per_s: float | None
    decode_gain: float | None
    draft_accept_length: float | None
    baseline_cuda_graph_share: float | None
    draft_cuda_graph_share: float | None


@dataclass(kw_only=True)
class SideAtConcurrency:
    """One side's legs at one concurrency, one value per leg that reported it."""

    throughputs: list[float]
    latencies_p95_s: list[float]
    first_output_latencies_p95_s: list[float]
    accuracy: dict[AccuracyMetric, list[float]]
    failed_requests: int
    incomplete_reasons: list[str] = field(default_factory=list)


@dataclass(kw_only=True)
class ConcurrencyRow:
    concurrency: int
    baseline: SideAtConcurrency
    draft: SideAtConcurrency
    gain: float | None
    latency_ratio: float | None
    first_output_latency_ratio: float | None
    is_complete: bool


@dataclass(kw_only=True)
class Comparison:
    rows: list[ConcurrencyRow]
    k: int | None
    worse_concurrencies: list[int]
    incomplete_concurrencies: list[int]
    accept_length_mean: float | None
    accuracy_metrics: list[AccuracyMetric]
    decode_rows: list[DecodeRow]


def mean_or_none(values: list[float]) -> float | None:
    return statistics.mean(values) if values else None


def median_or_none(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


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


def read_decode_batches(text: str) -> list[DecodeBatchLine]:
    batches: list[DecodeBatchLine] = []
    for line in text.splitlines():
        decode_batch = DECODE_BATCH_PATTERN.search(line)
        if decode_batch:
            running_requests, used_cuda_graph, gen_throughput = decode_batch.groups()
            accept_length = ACCEPT_LENGTH_PATTERN.search(line)
            batches.append(
                DecodeBatchLine(
                    running_requests=int(running_requests),
                    accept_length=(
                        float(accept_length.group(1)) if accept_length else None
                    ),
                    used_cuda_graph=used_cuda_graph == "True",
                    gen_throughput_tokens_per_s=float(gen_throughput),
                )
            )
        else:
            pass
    return batches


def cuda_graph_share(batches: list[DecodeBatchLine]) -> float | None:
    return (
        sum(batch.used_cuda_graph for batch in batches) / len(batches)
        if batches
        else None
    )


def decode_rows(legs: list[Leg]) -> list[DecodeRow]:
    rows: list[DecodeRow] = []
    for lowest_running_requests, highest_running_requests in RUNNING_REQUEST_RANGES:
        batches = {
            side: [
                batch
                for leg in legs
                if leg.side == side
                for batch in leg.decode_batches
                if batch.running_requests >= lowest_running_requests
                and (
                    highest_running_requests is None
                    or batch.running_requests <= highest_running_requests
                )
            ]
            for side in ("baseline", "draft")
        }
        baseline_throughput = median_or_none(
            [batch.gen_throughput_tokens_per_s for batch in batches["baseline"]]
        )
        draft_throughput = median_or_none(
            [batch.gen_throughput_tokens_per_s for batch in batches["draft"]]
        )
        rows.append(
            DecodeRow(
                lowest_running_requests=lowest_running_requests,
                highest_running_requests=highest_running_requests,
                baseline_lines=len(batches["baseline"]),
                draft_lines=len(batches["draft"]),
                baseline_gen_throughput_tokens_per_s=baseline_throughput,
                draft_gen_throughput_tokens_per_s=draft_throughput,
                decode_gain=(
                    draft_throughput / baseline_throughput
                    if draft_throughput is not None and baseline_throughput
                    else None
                ),
                draft_accept_length=mean_or_none(
                    [
                        batch.accept_length
                        for batch in batches["draft"]
                        if batch.accept_length is not None
                    ]
                ),
                baseline_cuda_graph_share=cuda_graph_share(batches["baseline"]),
                draft_cuda_graph_share=cuda_graph_share(batches["draft"]),
            )
        )
    return rows


def compare(legs: list[Leg], *, k_gain: float) -> Comparison:
    concurrencies = [load.concurrency for load in legs[0].loads]
    if any([load.concurrency for load in leg.loads] != concurrencies for leg in legs):
        raise ValueError("every leg must sweep the same concurrencies")
    else:
        pass
    accuracy_metrics = list(
        dict.fromkeys(
            metric for leg in legs for load in leg.loads for metric in load.accuracy
        )
    )
    sides = {
        (side, concurrency): SideAtConcurrency(
            throughputs=[],
            latencies_p95_s=[],
            first_output_latencies_p95_s=[],
            accuracy={metric: [] for metric in accuracy_metrics},
            failed_requests=0,
        )
        for side in ("baseline", "draft")
        for concurrency in concurrencies
    }
    for leg in legs:
        for load in leg.loads:
            measured = sides[(leg.side, load.concurrency)]
            measured.incomplete_reasons.extend(
                f"{leg.name}: {reason}" for reason in load.incomplete_reasons
            )
            for values, value in (
                (measured.throughputs, load.throughput_requests_per_s),
                (measured.latencies_p95_s, load.latency_p95_s),
                (
                    measured.first_output_latencies_p95_s,
                    load.first_output_latency_p95_s,
                ),
            ):
                if value is not None:
                    values.append(value)
                else:
                    pass
            for metric, value in load.accuracy.items():
                measured.accuracy[metric].append(value)
            measured.failed_requests += load.failed_requests
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
                first_output_latency_ratio=mean_ratio(
                    draft.first_output_latencies_p95_s,
                    baseline.first_output_latencies_p95_s,
                ),
                # note (wilsonzheng0327): throughput and p95 count only the requests that finished, so a load where any failed cannot qualify.
                is_complete=baseline.failed_requests == 0
                and draft.failed_requests == 0
                and not baseline.incomplete_reasons
                and not draft.incomplete_reasons
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
            or (
                row.first_output_latency_ratio is not None
                and row.first_output_latency_ratio > NEVER_WORSE_LATENCY_RATIO
                and statistics.mean(row.draft.first_output_latencies_p95_s)
                > statistics.mean(row.baseline.first_output_latencies_p95_s)
                + FIRST_OUTPUT_LATENCY_SLACK_S
            )
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
        accuracy_metrics=accuracy_metrics,
        decode_rows=decode_rows(legs),
    )


def number(value: float | int | None, digits: int, suffix: str = "") -> str:
    return "–" if value is None else f"{value:.{digits}f}{suffix}"


def throughput(requests_per_s: float | None) -> str:
    """Four significant digits, so 38-minute meetings at 0.03 requests/s keep their differences."""
    return "–" if requests_per_s is None else f"{requests_per_s:.4g}"


def percent(fraction: float | None) -> str:
    return "–" if fraction is None else f"{fraction * 100:.2f}%"


def change_percent(ratio: float | None) -> str:
    return number(None if ratio is None else (ratio - 1) * 100, 1, "%")


def gap_percent(values: list[float]) -> float | None:
    """The spread between one side's legs, relative to their mean."""
    mean_value = mean_or_none(values)
    return (
        (max(values) - min(values)) / mean_value * 100
        if len(values) > 1 and mean_value
        else None
    )


def render(comparison: Comparison, legs: list[Leg], *, title: str) -> str:
    has_first_output = any(
        row.baseline.first_output_latencies_p95_s
        or row.draft.first_output_latencies_p95_s
        for row in comparison.rows
    )
    headers = [
        "c",
        "No draft",
        "Draft",
        "Gain",
        "Gap, no draft",
        "Gap, draft",
        "p95, no draft (s)",
        "p95, draft (s)",
        "p95 change",
    ]
    if has_first_output:
        headers += [
            "First output p95, no draft (s)",
            "First output p95, draft (s)",
            "First output change",
        ]
    else:
        pass
    for metric in comparison.accuracy_metrics:
        headers += [f"{metric}, no draft", f"{metric}, draft"]
    headers += ["Failed, no draft", "Failed, draft"]
    lines = [
        f"# {title}: draft against no draft",
        "",
        f"Legs in order: {', '.join(leg.name for leg in legs)}. Throughput is requests per second, the mean of each side's legs; gap is the spread between a side's legs.",
        "",
        f"| {' | '.join(headers)} |",
        f"|{'---:|' * len(headers)}",
    ]
    for row in comparison.rows:
        cells = [
            f"{row.concurrency}{'' if row.is_complete else ' (incomplete)'}",
            throughput(mean_or_none(row.baseline.throughputs)),
            throughput(mean_or_none(row.draft.throughputs)),
            number(row.gain, 2, "x"),
            number(gap_percent(row.baseline.throughputs), 1, "%"),
            number(gap_percent(row.draft.throughputs), 1, "%"),
            number(mean_or_none(row.baseline.latencies_p95_s), 3),
            number(mean_or_none(row.draft.latencies_p95_s), 3),
            change_percent(row.latency_ratio),
        ]
        if has_first_output:
            cells += [
                number(mean_or_none(row.baseline.first_output_latencies_p95_s), 3),
                number(mean_or_none(row.draft.first_output_latencies_p95_s), 3),
                change_percent(row.first_output_latency_ratio),
            ]
        else:
            pass
        for metric in comparison.accuracy_metrics:
            cells += [
                ", ".join(percent(value) for value in side.accuracy[metric]) or "–"
                for side in (row.baseline, row.draft)
            ]
        cells += [str(row.baseline.failed_requests), str(row.draft.failed_requests)]
        lines.append(f"| {' | '.join(cells)} |")
    lines += [
        "",
        f"- k, the highest complete concurrency with at least {K_GAIN}x: {comparison.k if comparison.k is not None else 'none'}",
        f"- Worse than no draft (throughput under {NEVER_WORSE_THROUGHPUT_RATIO}x, p95 over {NEVER_WORSE_LATENCY_RATIO}x, or first-output p95 over both {NEVER_WORSE_LATENCY_RATIO}x and +{FIRST_OUTPUT_LATENCY_SLACK_S * 1000:g} ms): "
        f"{', '.join(f'c{concurrency}' for concurrency in comparison.worse_concurrencies) or 'none'}",
        "- Incomplete, so left out of k, the never-worse check and accuracy summaries (failed requests, missing numbers or incomplete evaluation): "
        f"{', '.join(f'c{concurrency}' for concurrency in comparison.incomplete_concurrencies) or 'none'}",
        "- Prompt tokens served from the prefix cache, which repeated prompts inflate: "
        + ", ".join(
            f"{leg.name} {percent(leg.server.cached_prompt_token_share)}"
            for leg in legs
        ),
    ]
    for row in comparison.rows:
        for side in (row.baseline, row.draft):
            lines.extend(
                f"- c{row.concurrency}: {reason}" for reason in side.incomplete_reasons
            )
    complete_rows = [row for row in comparison.rows if row.is_complete]
    accuracy_by_side = {
        metric: {
            "baseline": [
                value
                for row in complete_rows
                for value in row.baseline.accuracy[metric]
            ],
            "draft": [
                value for row in complete_rows for value in row.draft.accuracy[metric]
            ],
        }
        for metric in comparison.accuracy_metrics
    }
    for metric, values in accuracy_by_side.items():
        lines.append(
            f"- {metric} across legs and concurrencies: no draft {percent(min(values['baseline'], default=None))} to {percent(max(values['baseline'], default=None))}, "
            f"draft {percent(min(values['draft'], default=None))} to {percent(max(values['draft'], default=None))}"
        )
    lines += [
        "",
        "| Leg | Startup (s) | KV pool (tokens) | Peak token usage | Retractions | Accept len | Peak GPU memory (MiB) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for leg in legs:
        peaks = [
            load.gpu_memory_peak_mib
            for load in leg.loads
            if load.gpu_memory_peak_mib is not None
        ]
        lines.append(
            f"| {leg.name} | {leg.startup_s:.1f} | {number(leg.server.kv_pool_tokens, 0)} "
            f"| {number(leg.server.token_usage_peak, 2)} | {leg.server.retracted_requests} "
            f"| {number(leg.server.accept_length_mean, 2)} | {number(max(peaks, default=None), 0)} |"
        )
    lines += [
        "",
        "Logged generation throughput by requests running in one batch: the median of the server's gen throughput lines, a window rate that includes prefill and idle time between log lines.",
        "",
        "| Running | Lines, no draft | Lines, draft | Logged tokens/s, no draft | Logged tokens/s, draft | Logged gain | Accept len | CUDA graph, no draft | CUDA graph, draft |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for decode_row in comparison.decode_rows:
        running_label = (
            f"{decode_row.lowest_running_requests}+"
            if decode_row.highest_running_requests is None
            else f"{decode_row.lowest_running_requests}-{decode_row.highest_running_requests}"
        )
        lines.append(
            f"| {running_label} "
            f"| {decode_row.baseline_lines} | {decode_row.draft_lines} "
            f"| {number(decode_row.baseline_gen_throughput_tokens_per_s, 1)} "
            f"| {number(decode_row.draft_gen_throughput_tokens_per_s, 1)} "
            f"| {number(decode_row.decode_gain, 2, 'x')} "
            f"| {number(decode_row.draft_accept_length, 2)} "
            f"| {percent(decode_row.baseline_cuda_graph_share)} "
            f"| {percent(decode_row.draft_cuda_graph_share)} |"
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
    accuracy_cell = ", ".join(
        f"{metric} {percent(mean_or_none(values['baseline']))} → {percent(mean_or_none(values['draft']))}"
        for metric, values in accuracy_by_side.items()
    )
    lines += [
        "",
        "Roadmap row:",
        "",
        f"| {title} | | {number(comparison.accept_length_mean, 1)} | {' | '.join(roadmap_cells)} | {accuracy_cell} | |",
    ]
    return "\n".join(lines) + "\n"
