# SPDX-License-Identifier: Apache-2.0
"""Record paired overlap/clean v1.5 samples as native continuous sessions."""

from __future__ import annotations

import asyncio
import contextlib
import gc
import hashlib
import logging
import multiprocessing
import os
import pickle
import time
from concurrent.futures import Executor, ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path
from typing import Literal, Protocol

import numpy as np
import soundfile
from pydantic import JsonValue

import benchmarks.duplex.v15_dataset as v15_dataset
from benchmarks.duplex.artifacts import replay_run, source_fingerprint
from benchmarks.duplex.client import (
    LEGACY_TAIL_S,
    LEGACY_TRANSPORT,
    MAX_TIMEOUT_S,
    PACKET_MS,
    SAMPLE_RATE,
    TRANSPORT,
    TURN_BASED_TRANSPORT,
    TurnDetection,
    profile_protocol,
    run_session,
)
from benchmarks.duplex.profiles import DEFAULT_PROFILE, PROFILES, ProfileName
from benchmarks.duplex.v10_dataset import Sample as V10Sample
from benchmarks.duplex.v15_audio import (
    LEGACY_BARGE_IN_CUT,
    PACING_TOLERANCE_S,
    TURN_BASED_BARGE_IN_CUT,
    normalize_audio,
    reconstruct_output,
    write_json,
)
from benchmarks.duplex.v15_dataset import Sample as V15Sample

logger = logging.getLogger(__name__)

VARIANTS = {"overlap": "input", "clean": "clean_input"}
RUN_KIND = "full-duplex-bench-v1.5-paired"
RUNNER_FILES = (
    "benchmarks/duplex/v15_audio.py",
    "benchmarks/duplex/v15_dataset.py",
    "benchmarks/duplex/v15_runner.py",
)
VARIANT_STATUSES = ("pending", "invalid", "error", "fail", "not_exercised", "pass")
# Post-processing parses the whole trace and encodes reports in pure Python; it gets
# at most a quarter of the host so the recorder and the server keep their CPUs.
MAX_POSTPROCESS_WORKERS = 4
POSTPROCESS_NICE = 10
LOOP_LAG_INTERVAL_S = 0.01


class DuplexDataset(Protocol):
    def discover_samples(
        self,
        root: Path,
        sample_ids: list[str] | None = None,
        max_per_subset: int | None = None,
    ) -> list[V10Sample] | list[V15Sample]: ...

    def inventory(self, root: Path) -> dict[str, dict[str, int] | list[str]]: ...


def lower_priority() -> None:
    os.nice(POSTPROCESS_NICE)


def default_postprocess_workers() -> int:
    usable_cpus = (
        len(os.sched_getaffinity(0))
        if hasattr(os, "sched_getaffinity")
        else os.cpu_count() or 1
    )
    return max(1, min(MAX_POSTPROCESS_WORKERS, usable_cpus // 4))


# note (edwingao28): Replay and reconstruction hold the GIL for 100 ms and more in long
# C calls and full collections, so they run in other processes, off the pacing loop.
def postprocess_executor(workers: int) -> Executor:
    return ProcessPoolExecutor(
        max_workers=workers,
        mp_context=multiprocessing.get_context("spawn"),
        initializer=lower_priority,
    )


def postprocess_variant(
    variant_dir: Path, profile: ProfileName
) -> dict[str, JsonValue]:
    protocol_report = replay_run(variant_dir)
    write_json(variant_dir / "report.json", protocol_report)
    (case_report,) = protocol_report["cases"]
    playout_report = reconstruct_output(variant_dir, profile=profile)
    return {
        "case": {
            "status": case_report["status"],
            "violations": case_report["violations"],
            "admission_denials": case_report["metrics"]["admission_denials"],
        },
        "playout": {
            key: playout_report[key]
            for key in (
                "errors",
                "input_timing",
                "media_samples",
                "playout_samples",
                "sample_rate",
                "initial_delay_s",
            )
        }
        | {"media_pcm_sha256": playout_report["pcm_sha256"].get("output-media.wav")},
    }


def summarize(samples: list[dict[str, JsonValue]]) -> dict[str, JsonValue]:
    statuses = [
        variant["status"]
        for sample in samples
        for variant in sample["variants"].values()
    ]
    return {
        "samples_selected": len(samples),
        "samples_invalid": sum(bool(sample["errors"]) for sample in samples),
        "variants_selected": len(statuses),
        "variants": {status: statuses.count(status) for status in VARIANT_STATUSES},
        "pairs_qualified": sum(
            all(variant["qualified"] for variant in sample["variants"].values())
            for sample in samples
        ),
        "per_subset": {
            subset: sum(sample["subset"] == subset for sample in samples)
            for subset in dict.fromkeys(sample["subset"] for sample in samples)
        },
    }


def summarize_load(
    samples: list[dict[str, JsonValue]], concurrency: int
) -> dict[str, JsonValue]:
    """Latency and admission figures comparable only at the same concurrency."""
    variants = [
        variant for sample in samples for variant in sample["variants"].values()
    ]
    outputs = [variant["output"] for variant in variants if variant["output"]]
    timings = [
        variant["input_timing"] for variant in variants if variant["input_timing"]
    ]
    sessions = [variant["session"] for variant in variants if variant["session"]]
    initial_delays_s = [
        output["initial_delay_s"]
        for output in outputs
        if output["initial_delay_s"] is not None
    ]
    return {
        "concurrency": concurrency,
        "recording_wall_s": max(
            (
                session["finished_s"]
                for session in sessions
                if session["finished_s"] is not None
            ),
            default=None,
        ),
        "initial_delay_s": {
            "count": len(initial_delays_s),
            "p50": (
                float(np.percentile(initial_delays_s, 50)) if initial_delays_s else None
            ),
            "p90": (
                float(np.percentile(initial_delays_s, 90)) if initial_delays_s else None
            ),
            "max": max(initial_delays_s, default=None),
        },
        "max_input_pacing_deviation_s": max(
            (
                timing["max_abs_deviation_s"]
                for timing in timings
                if timing["max_abs_deviation_s"] is not None
            ),
            default=None,
        ),
        "input_pacing_violations": sum(
            timing["max_abs_deviation_s"] is not None and not timing["within_tolerance"]
            for timing in timings
        ),
        "variants_with_admission_denials": sum(
            bool(output["admission_denials"]) for output in outputs
        ),
        "variants_passed": sum(variant["status"] == "pass" for variant in variants),
    }


async def run_samples(
    dataset_root: Path,
    *,
    url: str,
    output: Path,
    server: dict[str, JsonValue],
    dataset_revision: str,
    timeout_s: float,
    sample_ids: list[str] | None = None,
    max_per_subset: int | None = None,
    dataset: DuplexDataset = v15_dataset,
    variants: dict[str, str] | None = None,
    kind: str = RUN_KIND,
    profile: ProfileName = DEFAULT_PROFILE,
    concurrency: int = 1,
    turn_detection: TurnDetection | None = None,
    legacy_tail_s: float = LEGACY_TAIL_S,
    postprocess: Executor | None = None,
) -> dict[str, JsonValue]:
    """Run selected sample variants, retaining failures in the selected denominator.

    turn_detection configures legacy-protocol and turn-based sessions and
    legacy_tail_s legacy-protocol ones; see benchmarks.duplex.client.run_session.
    postprocess runs replay and reconstruction; by default a pool of lower-priority
    processes, owned and shut down by this call.
    """
    if not dataset_revision:
        raise ValueError("dataset_revision must be nonempty")
    else:
        pass
    if not 0 < timeout_s <= MAX_TIMEOUT_S:
        raise ValueError(f"timeout_s must be positive and at most {MAX_TIMEOUT_S}")
    else:
        pass
    if concurrency < 1:
        raise ValueError("concurrency must be at least 1")
    else:
        pass
    protocol = profile_protocol(profile, turn_detection)
    legacy = protocol == "legacy"
    legacy_config = (
        {
            "turn_detection": turn_detection.model_dump(mode="json"),
            "tail_s": legacy_tail_s,
            "barge_in_cut": LEGACY_BARGE_IN_CUT,
            **LEGACY_TRANSPORT,
        }
        if legacy
        else None
    )
    turn_based_config = (
        {
            "turn_detection": turn_detection.model_dump(mode="json"),
            "barge_in_cut": TURN_BASED_BARGE_IN_CUT,
            **TURN_BASED_TRANSPORT,
        }
        if protocol == "turn_based"
        else None
    )
    variants = VARIANTS if variants is None else variants
    dataset_root = dataset_root.resolve()
    samples = dataset.discover_samples(dataset_root, sample_ids, max_per_subset)
    if not samples:
        raise ValueError("no samples selected")
    else:
        pass
    output.mkdir(parents=True, exist_ok=False)

    source = source_fingerprint()
    repository_root = Path(__file__).resolve().parents[2]
    source["files_sha256"].update(
        {
            path: hashlib.sha256((repository_root / path).read_bytes()).hexdigest()
            for path in RUNNER_FILES
        }
    )
    sample_records = []
    for sample in samples:
        sample_record = {**asdict(sample), "variants": {}}
        for variant, source_key in variants.items():
            sample_record["variants"][variant] = {
                "directory": str(Path("samples") / sample.directory / variant),
                "status": "invalid" if sample.errors else "pending",
                "protocol_verdict": None,
                "qualified": False,
                "errors": list(sample.errors),
                "violations": [],
                "source": {
                    "file": sample.paths.get(source_key),
                    "sha256": sample.sha256.get(source_key),
                },
                "normalization": None,
                "input": None,
                "input_timing": None,
                "output": None,
                "session": None,
                "files": {},
            }
        sample_records.append(sample_record)

    write_json(
        output / "manifest.json",
        {
            "schema_version": 1,
            "kind": kind,
            "profile": profile,
            "source": source,
            "server": server,
            "dataset": {
                "root": str(dataset_root),
                "revision": dataset_revision,
                "inventory": dataset.inventory(dataset_root),
                "selection": {
                    "sample_ids": sample_ids,
                    "max_per_subset": max_per_subset,
                },
            },
            "config": {
                "scenario": "continuous",
                "variants": variants,
                "timeout_s": timeout_s,
                "concurrency": concurrency,
                "packet_ms": PACKET_MS,
                "pacing_tolerance_s": PACING_TOLERANCE_S,
                "transport": TRANSPORT,
                "response_cancel": "never sent",
                "protocol": PROFILES[profile].protocol,
                "legacy": legacy_config,
                "turn_based": turn_based_config,
            },
            "samples": [
                {
                    "id": sample_record["id"],
                    "errors": sample_record["errors"],
                    "variants": {
                        variant_name: {
                            source_key: variant_state[source_key]
                            for source_key in ("directory", "source")
                        }
                        for variant_name, variant_state in sample_record[
                            "variants"
                        ].items()
                    },
                }
                for sample_record in sample_records
            ],
        },
    )

    client_loop: dict[str, JsonValue] = {
        "interval_s": LOOP_LAG_INTERVAL_S,
        "max_lag_s": None,
        "lags_over_tolerance": 0,
    }

    def run_state(
        status: Literal["preparing", "running", "complete"]
    ) -> dict[str, JsonValue]:
        return {
            "schema_version": 1,
            "status": status,
            "summary": summarize(sample_records),
            "load": summarize_load(sample_records, concurrency)
            | {"client_loop": client_loop},
            "samples": sample_records,
        }

    def save(
        status: Literal["preparing", "running", "complete"]
    ) -> dict[str, JsonValue]:
        result = run_state(status)
        write_json(output / "run.json", result)
        return result

    save("preparing")
    for sample, sample_record in zip(samples, sample_records):
        for variant, source_key in variants.items():
            variant_state = sample_record["variants"][variant]
            if variant_state["status"] != "pending":
                continue
            else:
                pass
            variant_dir = output / variant_state["directory"]
            try:
                pcm, normalization = normalize_audio(
                    dataset_root / sample.paths[source_key]
                )
                duration_s = len(pcm) / (SAMPLE_RATE * 2)
                variant_state["normalization"] = normalization
                variant_state["input"] = {
                    "sha256": hashlib.sha256(pcm).hexdigest(),
                    "duration_s": duration_s,
                }
                if duration_s + (legacy_tail_s if legacy else 0.0) >= timeout_s:
                    variant_state["status"] = "invalid"
                    variant_state["errors"].append(
                        f"input duration {duration_s:.3f}s"
                        + (f" plus legacy tail {legacy_tail_s}s" if legacy else "")
                        + f" is not below timeout {timeout_s}s"
                    )
                    continue
                else:
                    pass
                variant_dir.mkdir(parents=True)
                (variant_dir / "input.pcm").write_bytes(pcm)
                soundfile.write(
                    str(variant_dir / "input.wav"),
                    np.frombuffer(pcm, dtype="<i2"),
                    SAMPLE_RATE,
                    subtype="PCM_16",
                )
                write_json(
                    variant_dir / "manifest.json",
                    {
                        "schema_version": 1,
                        "profile": profile,
                        "source": source,
                        "server": server,
                        "config": {
                            "packet_ms": PACKET_MS,
                            "paced": True,
                            "timeout_s": timeout_s,
                            "input_duration_s": duration_s,
                            "transport": TRANSPORT,
                            "dataset_sample": sample.id,
                            "dataset_variant": variant,
                            "protocol": PROFILES[profile].protocol,
                            "legacy": legacy_config,
                            "turn_based": turn_based_config,
                        },
                        "input": {
                            "file": "input.pcm",
                            "sha256": variant_state["input"]["sha256"],
                            "sample_rate": SAMPLE_RATE,
                        },
                        "cases": [
                            {
                                "id": "continuous",
                                "scenario": "continuous",
                                "trace_file": "continuous.jsonl",
                            }
                        ],
                    },
                )
            except Exception as exc:
                logger.exception(
                    f"variant {sample_record['id']}/{variant} preparation failed"
                )
                variant_state["status"] = "error"
                variant_state["errors"].append(
                    f"preparation failed: {type(exc).__name__}: {exc}"
                )
                save("preparing")

    loop = asyncio.get_running_loop()
    with contextlib.ExitStack() as exit_stack:
        if postprocess is None:
            workers = default_postprocess_workers()
            postprocess = postprocess_executor(workers)
            exit_stack.callback(postprocess.shutdown, wait=True, cancel_futures=True)
            # note (edwingao28): Workers start and import before any session paces.
            await asyncio.gather(
                *(loop.run_in_executor(postprocess, os.getpid) for _ in range(workers))
            )
        else:
            pass

        # note (edwingao28): The indented encode grows with the run, so a thread writes
        # checkpoints, one at a time since every write goes through one partial file.
        checkpoint_dirty = False
        checkpoint_task: asyncio.Task[None] | None = None

        async def write_checkpoints() -> None:
            nonlocal checkpoint_dirty
            while checkpoint_dirty:
                checkpoint_dirty = False
                snapshot = pickle.loads(pickle.dumps(run_state("running")))
                await asyncio.to_thread(write_json, output / "run.json", snapshot)

        def checkpoint() -> None:
            nonlocal checkpoint_dirty, checkpoint_task
            checkpoint_dirty = True
            if checkpoint_task is None:
                checkpoint_task = asyncio.create_task(write_checkpoints())
            elif checkpoint_task.done():
                # note (edwingao28): A failed write raises here, not lost with its task.
                checkpoint_task.result()
                checkpoint_task = asyncio.create_task(write_checkpoints())
            else:
                pass

        save("running")
        pending = [
            (sample_record, variant, variant_state)
            for sample_record in sample_records
            for variant, variant_state in sample_record["variants"].items()
            if variant_state["status"] == "pending"
        ]
        session_slots = asyncio.Semaphore(concurrency)
        open_sessions = 0
        recording_started_s = time.perf_counter()

        async def record_variant(
            sample_record: dict[str, JsonValue],
            variant: str,
            variant_state: dict[str, JsonValue],
        ) -> None:
            nonlocal open_sessions
            async with session_slots:
                variant_dir = output / variant_state["directory"]
                variant_state["session"] = {
                    "started_s": time.perf_counter() - recording_started_s,
                    "finished_s": None,
                    "open_sessions_at_start": open_sessions,
                }
                open_sessions += 1
                try:
                    await run_session(
                        url,
                        await asyncio.to_thread((variant_dir / "input.pcm").read_bytes),
                        trace_path=variant_dir / "continuous.jsonl",
                        timeout_s=timeout_s,
                        profile=profile,
                        turn_detection=turn_detection,
                        tail_s=legacy_tail_s,
                    )
                    report = await loop.run_in_executor(
                        postprocess, postprocess_variant, variant_dir, profile
                    )
                except Exception as exc:
                    logger.exception(f"variant {sample_record['id']}/{variant} failed")
                    variant_state["status"] = "error"
                    variant_state["errors"].append(f"{type(exc).__name__}: {exc}")
                else:
                    case_report, playout_report = report["case"], report["playout"]
                    variant_state["protocol_verdict"] = case_report["status"]
                    variant_state["violations"] = case_report["violations"]
                    variant_state["errors"].extend(playout_report["errors"])
                    variant_state["status"] = (
                        "fail" if playout_report["errors"] else case_report["status"]
                    )
                    variant_state["qualified"] = variant_state["status"] == "pass"
                    variant_state["input_timing"] = playout_report["input_timing"]
                    variant_state["output"] = {
                        "media_pcm_sha256": playout_report["media_pcm_sha256"],
                        "media_duration_s": playout_report["media_samples"]
                        / playout_report["sample_rate"],
                        "playout_duration_s": playout_report["playout_samples"]
                        / playout_report["sample_rate"],
                        "initial_delay_s": playout_report["initial_delay_s"],
                        "admission_denials": case_report["admission_denials"],
                    }
                open_sessions -= 1
                variant_state["session"]["finished_s"] = (
                    time.perf_counter() - recording_started_s
                )
                variant_state["files"] = {
                    path.name: str(path.relative_to(output))
                    for path in sorted(variant_dir.iterdir())
                    if not path.name.startswith(".")
                }
                checkpoint()

        # note (edwingao28): Objects alive before the first session never become garbage
        # mid-run; frozen, they leave the full collections that land on the pacing loop.
        gc.collect()
        gc.freeze()
        monitor = asyncio.create_task(watch_loop(client_loop))
        try:
            await asyncio.gather(*(record_variant(*entry) for entry in pending))
        finally:
            monitor.cancel()
            await asyncio.gather(monitor, return_exceptions=True)
            gc.unfreeze()
        if checkpoint_task is not None:
            await checkpoint_task
        else:
            pass
        return save("complete")


# note (edwingao28): A pacing violation while this loop stays punctual points at the
# send path or the host, not at the recorder's loop.
async def watch_loop(client_loop: dict[str, JsonValue]) -> None:
    expected_s = time.perf_counter() + LOOP_LAG_INTERVAL_S
    while True:
        await asyncio.sleep(max(0.0, expected_s - time.perf_counter()))
        lag_s = time.perf_counter() - expected_s
        client_loop["max_lag_s"] = max(client_loop["max_lag_s"] or 0.0, lag_s)
        client_loop["lags_over_tolerance"] += lag_s > PACING_TOLERANCE_S
        expected_s = max(expected_s + LOOP_LAG_INTERVAL_S, time.perf_counter())
