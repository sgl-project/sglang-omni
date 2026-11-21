# SPDX-License-Identifier: Apache-2.0
"""Measure one offload run and write the record D1 compares.

The workload is a parameter, not a constant: point it at a prompt list and a
served speech endpoint, and it reports what the client heard and what the GPU
held while it happened.  The record it writes is `OffloadRun`, so two runs of
the same workload compare with `offload_record.compare_runs`.

    python -m benchmarks.offload.run_offload_bench \
        --url http://127.0.0.1:30000 --model qwen3-tts --prompts prompts.txt \
        --workload seedtts-en-100 --checkpoint main --concurrency 1 \
        --label main --out runs/main-c1.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import subprocess
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import httpx

from benchmarks.offload.offload_record import (
    LatencyRecord,
    MemoryRecord,
    OffloadRun,
    Provenance,
    write_run,
)
from benchmarks.runtime_metrics import ResourceMonitor

PCM16_BYTES_PER_SAMPLE = 2
DEFAULT_SAMPLE_RATE_HZ = 24000
AUDIO_HEADER = "X-Sample-Rate"


@dataclass(kw_only=True)
class RequestTiming:
    """What one request cost: when its audio started, how it arrived, how long it was."""

    first_chunk_s: float
    chunk_gaps_s: list[float] = field(default_factory=list)
    audio_s: float = 0.0
    elapsed_s: float = 0.0


def percentile(values: list[float], fraction: float) -> float:
    """The value below which `fraction` of a sorted sample falls, nearest-rank."""
    if not values:
        raise ValueError("percentile needs at least one value")
    else:
        pass
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round(fraction * (len(ordered) - 1))))
    return ordered[index]


def summarize_requests(timings: list[RequestTiming], wall_s: float) -> LatencyRecord:
    """The client-side metrics of one run, from its per-request timings."""
    if not timings:
        raise ValueError("a run needs at least one request")
    elif wall_s <= 0:
        raise ValueError("a run needs a positive wall time")
    else:
        pass
    first_chunks = [timing.first_chunk_s for timing in timings]
    gaps = [gap for timing in timings for gap in timing.chunk_gaps_s]
    real_time_factors = [
        timing.elapsed_s / timing.audio_s for timing in timings if timing.audio_s > 0
    ]
    if not real_time_factors:
        raise ValueError("a run needs at least one request that produced audio")
    else:
        pass
    return LatencyRecord(
        ttfa_p50_s=percentile(first_chunks, 0.50),
        ttfa_p95_s=percentile(first_chunks, 0.95),
        inter_chunk_p95_s=percentile(gaps, 0.95) if gaps else 0.0,
        inter_chunk_p99_s=percentile(gaps, 0.99) if gaps else 0.0,
        rtf_p95=percentile(real_time_factors, 0.95),
        throughput_qps=len(timings) / wall_s,
    )


def gpu_identity(gpu_index: int) -> tuple[str, str, float]:
    """Name, driver and total memory of one GPU, read from nvidia-smi."""
    query = "name,driver_version,memory.total"
    completed = subprocess.run(
        [
            "nvidia-smi",
            f"--id={gpu_index}",
            f"--query-gpu={query}",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "nvidia-smi could not read the GPU identity: " + completed.stderr.strip()
        )
    else:
        pass
    name, driver, memory_mib = [part.strip() for part in completed.stdout.split(",")]
    return name, driver, float(memory_mib)


def git_commit() -> str:
    """The commit the server was measured from, as the repository records it."""
    completed = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
        cwd=Path(__file__).resolve().parents[2],
    )
    return completed.stdout.strip() or "unknown"


def read_prompts(path: Path) -> list[str]:
    """One prompt per non-empty line; a JSONL file keeps its `prompt` field."""
    prompts: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text:
            continue
        elif text.startswith("{"):
            prompts.append(str(json.loads(text)["prompt"]))
        else:
            prompts.append(text)
    if not prompts:
        raise ValueError(f"{path} carried no prompt")
    else:
        pass
    return prompts


async def speak_once(
    client: httpx.AsyncClient,
    url: str,
    body: dict[str, object],
    sample_rate_hz: int,
) -> RequestTiming:
    """Stream one sentence and record when its audio arrived."""
    started = time.perf_counter()
    first_chunk_s = 0.0
    gaps: list[float] = []
    pcm_bytes = 0
    previous = started
    async with client.stream("POST", url, json=body) as response:
        response.raise_for_status()
        rate = int(response.headers.get(AUDIO_HEADER, sample_rate_hz))
        async for data in response.aiter_bytes():
            if not data:
                continue
            elif pcm_bytes == 0:
                first_chunk_s = time.perf_counter() - started
            else:
                gaps.append(time.perf_counter() - previous)
            previous = time.perf_counter()
            pcm_bytes += len(data)
    elapsed_s = time.perf_counter() - started
    audio_s = pcm_bytes / PCM16_BYTES_PER_SAMPLE / rate
    return RequestTiming(
        first_chunk_s=first_chunk_s,
        chunk_gaps_s=gaps,
        audio_s=audio_s,
        elapsed_s=elapsed_s,
    )


async def measure(args: argparse.Namespace) -> OffloadRun:
    """Run the workload and build the record."""
    prompts = read_prompts(Path(args.prompts))[: args.limit]
    gpu_name, driver, memory_mib = gpu_identity(args.gpu_index)
    monitor = ResourceMonitor(
        gpu_index=args.gpu_index, interval_s=args.memory_interval_s
    ).start()
    if monitor.error is not None:
        raise RuntimeError("GPU memory sampling failed: " + monitor.error)
    else:
        pass
    semaphore = asyncio.Semaphore(args.concurrency)

    async with httpx.AsyncClient(timeout=args.timeout_s) as client:

        async def one(prompt: str) -> RequestTiming:
            async with semaphore:
                return await speak_once(
                    client,
                    f"{args.url.rstrip('/')}/v1/audio/speech",
                    {
                        "model": args.model,
                        "input": prompt,
                        "voice": args.voice,
                        "response_format": "pcm",
                        "stream": True,
                    },
                    args.sample_rate_hz,
                )

        started = time.perf_counter()
        timings = await asyncio.gather(*(one(prompt) for prompt in prompts))
        wall_s = time.perf_counter() - started

    resources = monitor.stop()
    if resources.get("error") is not None:
        raise RuntimeError("GPU memory sampling failed: " + str(resources["error"]))
    else:
        pass
    samples = resources.get("samples") or []
    peak = max((sample["gpu_memory_used_mib"] for sample in samples), default=0.0)
    idle = samples[0]["gpu_memory_used_mib"] if samples else 0.0
    steady = (
        statistics.median([sample["gpu_memory_used_mib"] for sample in samples])
        if samples
        else 0.0
    )
    process_peaks = [
        sample["gpu_process_memory_mib"]
        for sample in samples
        if sample.get("gpu_process_memory_mib") is not None
    ]
    return OffloadRun(
        label=args.label,
        provenance=Provenance(
            git_commit=git_commit(),
            gpu_name=gpu_name,
            gpu_memory_mib=memory_mib,
            driver_version=driver,
            model=args.model,
            checkpoint=args.checkpoint,
            concurrency=args.concurrency,
            workload=args.workload,
            recorded_at=datetime.now(UTC).isoformat(timespec="seconds"),
        ),
        memory=MemoryRecord(
            peak_mib=float(peak),
            steady_mib=float(steady),
            after_idle_mib=float(idle),
            process_peak_mib=max(process_peaks) if process_peaks else None,
        ),
        latency=summarize_requests(list(timings), wall_s),
        startup_s=args.startup_s,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True, help="base URL of the served model")
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--prompts", required=True, help="one prompt per line, or JSONL"
    )
    parser.add_argument(
        "--workload", required=True, help="the workload name the pair shares"
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--label", required=True, help="main or the change under test")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--voice", default="default")
    parser.add_argument("--sample-rate-hz", type=int, default=DEFAULT_SAMPLE_RATE_HZ)
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--memory-interval-s", type=float, default=0.2)
    parser.add_argument("--timeout-s", type=float, default=600.0)
    parser.add_argument("--startup-s", type=float, default=0.0)
    parser.add_argument("--gpu-process-pid", type=int, action="append", default=[])
    arguments = parser.parse_args(argv)
    run = asyncio.run(measure(arguments))
    written = write_run(run, arguments.out)
    print(f"wrote {written}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
