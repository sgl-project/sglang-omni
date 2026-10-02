# SPDX-License-Identifier: Apache-2.0
"""Run synchronized realtime sessions and summarize client-side serving behavior."""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from pydantic import JsonValue

from benchmarks.duplex.client import SAMPLE_RATE, run_session
from benchmarks.duplex.profiles import DEFAULT_PROFILE, PROFILES, ProfileName
from benchmarks.duplex.serving_metrics import (
    aggregate_sessions,
    distribution,
    session_metrics,
)
from benchmarks.duplex.serving_unit_metrics import (
    aggregate_unit_sessions,
    failed_unit_session,
    unit_session_metrics,
)
from benchmarks.duplex.v15_audio import normalize_audio

START_LEAD_S = 0.2
DEFAULT_RESERVE_MS = 80.0


async def run_concurrency(
    url: str,
    pcms: list[bytes],
    *,
    concurrency: int,
    profile: ProfileName,
    output_dir: Path,
    timeout_s: float,
    reserve_s: float,
    pacing: Literal["realtime", "lockstep"] = "realtime",
    stagger: bool = False,
) -> dict[str, JsonValue]:
    if concurrency < 1:
        raise ValueError("concurrency must be positive")
    if not pcms or any(not pcm or len(pcm) % 2 for pcm in pcms):
        raise ValueError("inputs must be nonempty PCM16")
    is_continuous = PROFILES[profile].continuous_output
    output_dir.mkdir(parents=True, exist_ok=False)
    loop = asyncio.get_running_loop()
    start_gate: asyncio.Future[float] = loop.create_future()
    ready = [loop.create_future() for _ in range(concurrency)]
    trace_paths = []
    tasks = []
    for index in range(concurrency):
        session_dir = output_dir / f"session-{index:03d}"
        session_dir.mkdir()
        trace_path = session_dir / "trace.jsonl"
        trace_paths.append(trace_path)
        task = asyncio.create_task(
            run_session(
                url,
                pcms[index % len(pcms)],
                scenario="continuous",
                trace_path=trace_path,
                timeout_s=timeout_s,
                profile=profile,
                start_gate=start_gate,
                ready=ready[index],
                pacing=pacing,
                # note (Junnan Li): Spread starts over one native unit so unit boundaries do not coincide.
                start_offset_s=(
                    index / concurrency * PROFILES[profile].native_unit_ms / 1000
                    if stagger
                    else 0.0
                ),
            )
        )
        task.add_done_callback(
            lambda completed, readiness=ready[index]: (
                readiness.set_result(False) if not readiness.done() else None
            )
        )
        tasks.append(task)
    configured = await asyncio.gather(*ready)
    start_s = time.perf_counter() + START_LEAD_S
    start_gate.set_result(start_s)
    results = await asyncio.gather(*tasks, return_exceptions=True)
    sessions = []
    for index, (trace_path, result) in enumerate(zip(trace_paths, results)):
        session_id = f"session-{index:03d}"
        duration_s = len(pcms[index % len(pcms)]) / (2 * SAMPLE_RATE)
        try:
            session = (session_metrics if is_continuous else unit_session_metrics)(
                trace_path,
                session_id=session_id,
                input_duration_s=duration_s,
                profile=profile,
                reserve_s=reserve_s,
            )
        except (OSError, ValueError, KeyError, TypeError) as exc:
            if not is_continuous:
                session = failed_unit_session(
                    session_id, trace_path, duration_s, profile
                )
                session["errors"].append(f"unreadable session artifacts: {exc}")
            else:
                session = {
                    "session_id": session_id,
                    "input_duration_s": duration_s,
                    "trace_file": str(trace_path),
                    "success": False,
                    "errors": [f"unreadable session artifacts: {exc}"],
                    "ttfa_s": None,
                    "send_lateness_s": distribution([]),
                    "late_send_count": 0,
                    "late_send_rate": None,
                    "output_gap_s": distribution([]),
                    "output_gap_excess_s": distribution([]),
                    "output_drift_s": distribution([]),
                    "final_output_drift_s": None,
                    "send_lateness_values_s": [],
                    "output_gap_values_s": [],
                    "output_gap_excess_values_s": [],
                    "output_drift_values_s": [],
                    "output_samples": 0,
                    "output_duration_s": 0.0,
                    "output_coverage": 0.0,
                    "underrun_count": 1,
                    "underrun_total_s": duration_s,
                    "underrun_worst_s": duration_s,
                    "underrun_ratio": 1.0,
                }
        if isinstance(result, BaseException):
            session["success"] = False
            session["errors"].append(f"client task: {type(result).__name__}: {result}")
        sessions.append(session)
    summary: dict[str, JsonValue] = {
        "concurrency": concurrency,
        "input_duration_s": max(len(pcm) for pcm in pcms) / (2 * SAMPLE_RATE),
        "common_start_s": start_s,
        "configured_sessions": sum(configured),
        "profile": profile,
        "pacing": pacing,
        "playback_startup_reserve_s": reserve_s,
        "sessions": sessions,
        "aggregate": (aggregate_sessions if is_continuous else aggregate_unit_sessions)(
            sessions
        ),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return summary


def format_ms(value: float | None) -> str:
    return f"{value * 1000:.1f}" if value is not None else "-"


def format_percent(value: float | None) -> str:
    return f"{value * 100:.1f}%" if value is not None else "-"


def print_lockstep_summaries(summaries: list[dict[str, JsonValue]]) -> None:
    print(
        f"{'C':>3} {'sessions':>9} {'units':>6} {'speak':>6} "
        f"{'unit p50':>9} {'unit p95':>9} {'unit p99':>9} {'speak p95':>10} "
        f"{'units/s each':>13} {'units/s total':>14}"
    )
    for summary in summaries:
        aggregate = summary["aggregate"]
        sessions = (
            f"{aggregate['successful_sessions']}/{aggregate['attempted_sessions']}"
        )
        each = aggregate["session_units_per_s"]["p50"]
        print(
            f"{summary['concurrency']:>3} {sessions:>9} "
            f"{aggregate['completed_units']:>6} {aggregate['speak_units']:>6} "
            f"{format_ms(aggregate['unit_lag_s']['p50']):>9} "
            f"{format_ms(aggregate['unit_lag_s']['p95']):>9} "
            f"{format_ms(aggregate['unit_lag_s']['p99']):>9} "
            f"{format_ms(aggregate['speak_unit_lag_s']['p95']):>10} "
            f"{(f'{each:.2f}' if each is not None else '-'):>13} "
            f"{aggregate['total_units_per_s']:>14.2f}"
        )


def print_unit_summaries(summaries: list[dict[str, JsonValue]]) -> None:
    print(
        f"{'C':>3} {'sessions':>9} {'units':>6} {'speak':>6} {'miss':>7} "
        f"{'lag p50':>8} {'lag p95':>8} {'lag p99':>8} {'speak p95':>10} "
        f"{'reply p50':>10} {'reply p95':>10} {'underrun':>9} {'late sends':>11}"
    )
    for summary in summaries:
        aggregate = summary["aggregate"]
        sessions = (
            f"{aggregate['successful_sessions']}/{aggregate['attempted_sessions']}"
        )
        print(
            f"{summary['concurrency']:>3} {sessions:>9} "
            f"{aggregate['completed_units']:>6} {aggregate['speak_units']:>6} "
            f"{format_percent(aggregate['unit_miss_rate']):>7} "
            f"{format_ms(aggregate['unit_lag_s']['p50']):>8} "
            f"{format_ms(aggregate['unit_lag_s']['p95']):>8} "
            f"{format_ms(aggregate['unit_lag_s']['p99']):>8} "
            f"{format_ms(aggregate['speak_unit_lag_s']['p95']):>10} "
            f"{format_ms(aggregate['response_audio_lag_s']['p50']):>10} "
            f"{format_ms(aggregate['response_audio_lag_s']['p95']):>10} "
            f"{format_percent(aggregate['underrun_ratio']):>9} "
            f"{format_percent(aggregate['late_send_rate']):>11}"
        )


def print_summary(summary: dict[str, JsonValue]) -> None:
    aggregate = summary["aggregate"]
    print(
        f"Concurrency: {summary['concurrency']}  "
        f"Duration: {summary['input_duration_s']:.2f}s  "
        f"Sessions: {aggregate['successful_sessions']}/{aggregate['attempted_sessions']}"
    )
    print(f"{'Metric':22} {'p50':>9} {'p75':>9} {'p95':>9} {'p99':>9} {'max':>9}")
    for label, name in (
        ("TTFA (ms)", "ttfa_s"),
        ("Send lateness (ms)", "send_lateness_s"),
        ("Output gap (ms)", "output_gap_s"),
        ("Gap excess (ms)", "output_gap_excess_s"),
        ("Output drift (ms)", "output_drift_s"),
    ):
        values = aggregate[name]
        print(
            f"{label:22} {format_ms(values['p50']):>9} "
            f"{format_ms(values['p75']):>9} "
            f"{format_ms(values['p95']):>9} "
            f"{format_ms(values['p99']):>9} {format_ms(values['max']):>9}"
        )
    coverage = aggregate["output_coverage"]["p50"]
    print(f"Output coverage p50: {coverage * 100:.1f}%")
    print(
        f"Late sends (>{aggregate['late_send_threshold_s'] * 1000:.0f}ms): "
        f"{aggregate['late_send_count']}/{aggregate['send_lateness_s']['n']} "
        f"({format_percent(aggregate['late_send_rate'])})"
    )
    print(
        "Final output drift p50/p95: "
        f"{format_ms(aggregate['final_output_drift_s']['p50'])}/"
        f"{format_ms(aggregate['final_output_drift_s']['p95'])}ms"
    )
    print(
        f"Underrun sessions: {aggregate['underrun_sessions']}/"
        f"{aggregate['attempted_sessions']}  "
        f"Count: {aggregate['underrun_count']}  "
        f"Total/max: {aggregate['underrun_total_s'] * 1000:.1f}/"
        f"{aggregate['underrun_worst_s'] * 1000:.1f}ms  "
        f"Ratio: {format_percent(aggregate['underrun_ratio'])}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument(
        "--audio",
        required=True,
        type=Path,
        nargs="+",
        help="input recordings; sessions take them in turn",
    )
    parser.add_argument(
        "--profile", choices=["nemotron", *PROFILES], default="nemotron"
    )
    concurrency = parser.add_mutually_exclusive_group(required=True)
    concurrency.add_argument("--concurrency", type=int)
    concurrency.add_argument("--concurrencies")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--timeout-s", type=float, default=90.0)
    parser.add_argument("--startup-reserve-ms", type=float, default=DEFAULT_RESERVE_MS)
    parser.add_argument(
        "--pacing",
        choices=["realtime", "lockstep"],
        default="realtime",
        help="lockstep sends each unit when the previous one is done",
    )
    parser.add_argument(
        "--stagger",
        action="store_true",
        help="spread session starts evenly over one native unit",
    )
    parser.add_argument(
        "--warmup-runs",
        type=int,
        default=0,
        help="single-session runs before the first level; kept under warmup-N",
    )
    args = parser.parse_args()
    profile: ProfileName = (
        DEFAULT_PROFILE if args.profile == "nemotron" else args.profile
    )
    levels = (
        [args.concurrency]
        if args.concurrency is not None
        else [int(value) for value in args.concurrencies.split(",")]
    )
    if not levels or any(level < 1 for level in levels):
        parser.error("concurrency values must be positive")
    if args.timeout_s <= 0 or args.startup_reserve_ms < 0:
        parser.error("timeout must be positive and startup reserve nonnegative")
    if args.warmup_runs < 0:
        parser.error("warmup runs must be nonnegative")
    if args.pacing == "lockstep" and PROFILES[profile].continuous_output:
        parser.error("lockstep pacing needs a profile scored per unit")
    pcms = [normalize_audio(path)[0] for path in args.audio]
    output_dir = args.output_dir or Path("serving-results") / datetime.now(
        timezone.utc
    ).strftime("%Y%m%dT%H%M%SZ")
    output_dir.mkdir(parents=True, exist_ok=False)

    async def run_all() -> list[dict[str, JsonValue]]:
        for index in range(args.warmup_runs):
            await run_concurrency(
                args.url,
                pcms[:1],
                concurrency=1,
                profile=profile,
                output_dir=output_dir / f"warmup-{index}",
                timeout_s=args.timeout_s,
                reserve_s=args.startup_reserve_ms / 1000,
            )
        return [
            await run_concurrency(
                args.url,
                pcms,
                concurrency=level,
                profile=profile,
                output_dir=output_dir / f"c{level}",
                timeout_s=args.timeout_s,
                reserve_s=args.startup_reserve_ms / 1000,
                pacing=args.pacing,
                stagger=args.stagger,
            )
            for level in levels
        ]

    summaries = asyncio.run(run_all())
    (output_dir / "summary.json").write_text(
        json.dumps({"runs": summaries}, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    if args.pacing == "lockstep":
        print_lockstep_summaries(summaries)
    elif not PROFILES[profile].continuous_output:
        print_unit_summaries(summaries)
    elif len(summaries) == 1:
        print_summary(summaries[0])
    else:
        print(
            "C  success  TTFA p95  gap p99  coverage p50  "
            "late sends  final drift p50  underrun sessions  underrun ratio"
        )
        for summary in summaries:
            aggregate = summary["aggregate"]
            underrun_sessions = (
                f"{aggregate['underrun_sessions']}/{aggregate['attempted_sessions']}"
            )
            print(
                f"{summary['concurrency']:<2} "
                f"{aggregate['successful_sessions']}/{aggregate['attempted_sessions']:<7} "
                f"{format_ms(aggregate['ttfa_s']['p95']):>8} "
                f"{format_ms(aggregate['output_gap_s']['p99']):>8} "
                f"{aggregate['output_coverage']['p50'] * 100:>12.1f}% "
                f"{format_percent(aggregate['late_send_rate']):>11} "
                f"{format_ms(aggregate['final_output_drift_s']['p50']):>15} "
                f"{underrun_sessions:>17} "
                f"{format_percent(aggregate['underrun_ratio']):>15}"
            )
    print(f"Artifacts: {output_dir}")


if __name__ == "__main__":
    main()
