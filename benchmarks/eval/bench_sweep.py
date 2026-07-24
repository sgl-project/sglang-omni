"""Open-loop request-rate sweep for MPS-DP TTS replicas.

Note (Yueying Li): open-loop Poisson arrivals instead of fixed concurrency,
because past saturation a closed-loop client absorbs the backlog into TTFC
and hides the capacity knee.

Usage:
    python -m benchmarks.eval.bench_sweep <N> <total_rates_csv> \
        <samples_per_client> <outdir>

Assumes N replicas already healthy on 127.0.0.1:8801..880N. Clients use
disjoint seed-tts-eval EN shards within a stage so shared caches cannot
inflate multi-client numbers.

Environment:
    BENCH_MODEL   model name passed to benchmark_tts_seedtts (default higgs)
    BENCH_STREAM  1 (default) records TTFC; 0 drops --stream for models that
                  generate the final waveform only (TTFC percentiles null)
    BENCH_REF     1 (default) sends reference audio; 0 text-only synthesis
    CLIENT_CORES  optional CPU list to pin client processes to

Per-stage results are written to <outdir>/sweep.json.
"""

import json
import os
import shutil
import signal
import subprocess
import sys
import time

import numpy as np

N = int(sys.argv[1])
RATES = [float(r) for r in sys.argv[2].split(",")]
SAMPLES = int(sys.argv[3])
OUT = sys.argv[4]
EN_TOTAL = 1088  # seed-tts-eval EN split size

os.makedirs(OUT, exist_ok=True)
# Note (Jiaxin Deng): each client in a stage takes a disjoint SAMPLES-length
# block so a shared server cache cannot inflate one client's numbers; N disjoint
# blocks must exist in the EN split, and stages rotate the starting block.
NUM_SLOTS = EN_TOTAL // SAMPLES if SAMPLES > 0 else 0
if not (0 < N <= NUM_SLOTS):
    sys.exit(
        f"bench_sweep: need 0 < N <= EN_TOTAL//SAMPLES ({NUM_SLOTS}) for disjoint "
        f"shards, got N={N} SAMPLES={SAMPLES}"
    )


def _spawn_client(stage_index, rate, per_client_rate, stage_dir, i):
    slot = (stage_index + i) % NUM_SLOTS
    offset = slot * SAMPLES
    cdir = os.path.join(stage_dir, f"client{i}")
    # Note (Yueying Li): remove any previous run's output for this client so a
    # client that fails before writing results cannot resurrect a stale
    # speed_results.json into this stage's aggregation.
    shutil.rmtree(cdir, ignore_errors=True)
    cmd = [
        sys.executable,
        "-m",
        "benchmarks.eval.benchmark_tts_seedtts",
        "--generate-only",
        "--use-existing-server",
        "--meta",
        "zhaochenyang20/seed-tts-eval-arrow",
        "--model",
        os.environ.get("BENCH_MODEL", "higgs"),
        "--host",
        "127.0.0.1",
        "--port",
        str(8801 + i),
        "--lang",
        "en",
        "--max-samples",
        str(SAMPLES),
        "--sample-offset",
        str(offset),
        "--concurrency",
        "0",
        "--request-rate",
        str(per_client_rate),
        "--output-dir",
        cdir,
        "--disable-tqdm",
    ]
    if os.environ.get("BENCH_STREAM", "1") != "0":
        cmd.append("--stream")
    if os.environ.get("BENCH_REF", "1") != "0":
        cmd.extend(["--ref-format", "references"])
    else:
        cmd.append("--no-ref-audio")
    logf = open(os.path.join(OUT, f"rate{rate:g}_client{i}.log"), "w")
    preexec = None
    if os.environ.get("CLIENT_CORES"):
        client_cores = {int(c) for c in os.environ["CLIENT_CORES"].split(",")}
        preexec = lambda: os.sched_setaffinity(0, client_cores)  # noqa: E731
    # Note (Yueying Li): each client gets its own session so teardown can signal
    # the whole process tree; close the log handle ourselves if Popen never
    # returns a process to own it.
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=logf,
            stderr=subprocess.STDOUT,
            preexec_fn=preexec,
            start_new_session=True,
        )
    except BaseException:
        logf.close()
        raise
    return proc, logf


def _stop_client(proc, sig):
    try:
        os.killpg(proc.pid, sig)
    except (ProcessLookupError, PermissionError):
        pass


sweep = []
for stage_index, rate in enumerate(RATES):
    per_client_rate = rate / N
    stage_dir = os.path.join(OUT, f"rate{rate:g}")
    procs = []
    # Note (Jiaxin Deng): terminate every started client and close its log even
    # if a later spawn fails, so a partial stage cannot leak traffic or handles
    # into the next stage.
    # Note (Yueying Li): teardown signals each client's whole session (SIGTERM,
    # bounded wait, SIGKILL escalation) and reaps it, so traffic is provably
    # stopped before the stage exits and no zombie survives into the next one.
    try:
        for i in range(N):
            procs.append(
                _spawn_client(stage_index, rate, per_client_rate, stage_dir, i)
            )
        t0 = time.time()
        fails = 0
        for p, logf in procs:
            fails += p.wait() != 0
        wall = time.time() - t0
    finally:
        for p, logf in procs:
            if p.poll() is None:
                _stop_client(p, signal.SIGTERM)
                try:
                    p.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    _stop_client(p, signal.SIGKILL)
                    p.wait()
            logf.close()

    ttfcs, lats, rtfs, audio_secs = [], [], [], []
    completed, failed = 0, 0
    achieved_runner = audio_sps = 0.0  # from each client's runner wall clock
    for i in range(N):
        f = os.path.join(stage_dir, f"client{i}", "speed_results.json")
        if not os.path.exists(f):
            continue
        with open(f) as fh:
            d = json.load(fh)
        for r in d["per_request"]:
            if r.get("is_success"):
                lats.append(r["latency_s"])
                # Note (Yueying Li): TTFC only exists for streaming runs; latency/RTF always do.
                if r.get("audio_ttfp_s") is not None:
                    ttfcs.append(r["audio_ttfp_s"])
                if r.get("rtf") is not None:
                    rtfs.append(r["rtf"])
                if r.get("audio_duration_s") is not None:
                    audio_secs.append(r["audio_duration_s"])
        s = d["summary"]
        completed += s["completed_requests"]
        failed += s["failed_requests"]
        achieved_runner += s.get("throughput_qps") or 0
        audio_sps += s.get("audio_throughput_s_per_s") or 0
    ttfcs.sort()
    rtfs.sort()
    lats_sorted = sorted(lats)

    # Note (Yueying Li): numpy linear-interpolated percentiles, matching
    # benchmarks.metrics.performance, so every repository benchmark shares one
    # percentile definition.
    def pct(arr, p):
        return round(float(np.percentile(arr, 100 * p)), 4) if arr else None

    stage = {
        "offered_total_rate": rate,
        # Note (Yueying Li): completed / (spawn-to-exit wall) includes client
        # startup + drain; use achieved_qps_runner for capacity claims.
        "achieved_qps": round(completed / wall, 2),
        "achieved_qps_runner": round(achieved_runner, 2),
        "audio_s_per_s": round(audio_sps, 1),
        "wall_s": round(wall, 1),
        "completed": completed,
        "failed": failed,
        "client_fails": fails,
        "ttfc_p50": pct(ttfcs, 0.50),
        "ttfc_p90": pct(ttfcs, 0.90),
        "ttfc_p95": pct(ttfcs, 0.95),
        "ttfc_p99": pct(ttfcs, 0.99),
        "ttfc_sorted": ttfcs,  # full CDF support
        "rtf_p50": pct(rtfs, 0.50),
        "rtf_p95": pct(rtfs, 0.95),
        "rtf_p99": pct(rtfs, 0.99),
        "rtf_sorted": rtfs,  # full CDF support
        "latency_mean": round(sum(lats) / len(lats), 3) if lats else None,
        "latency_p50": pct(lats_sorted, 0.50),
        "latency_p99": pct(lats_sorted, 0.99),
        "audio_duration_mean_s": (
            round(sum(audio_secs) / len(audio_secs), 3) if audio_secs else None
        ),
    }
    sweep.append(stage)

    def fmt(v, spec):
        return format(v, spec) if v is not None else "-"

    print(
        f"rate={rate:g} achieved={stage['achieved_qps_runner']} "
        f"audio_s/s={stage['audio_s_per_s']} rtf_p50={fmt(stage['rtf_p50'], '.2f')} "
        f"p50={fmt(stage['ttfc_p50'], '.3f')} p99={fmt(stage['ttfc_p99'], '.3f')} "
        f"failed={failed}",
        flush=True,
    )
    time.sleep(5)  # drain between stages

with open(os.path.join(OUT, "sweep.json"), "w") as f:
    json.dump({"n_replicas": N, "samples_per_client": SAMPLES, "stages": sweep}, f)
print("sweep written:", os.path.join(OUT, "sweep.json"))
