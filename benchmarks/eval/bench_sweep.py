#!/usr/bin/env python3
"""bench_serving-style open-loop request-rate sweep for MPS-DP TTS replicas.

Closed-loop benchmarking (fixed concurrency) conflates server capacity with
client-side queueing: past the saturation point, TTFC absorbs the whole
backlog. This driver instead sweeps *offered* rates: for each total rate it
runs one open-loop Poisson client per replica (``--concurrency 0`` disables
the closed-loop semaphore; ``--request-rate r`` gives exponential
inter-arrivals) and harvests every per-request TTFC/RTF/latency for full
CDFs, so TTFC can be read as a function of offered throughput and the
capacity knee located.

Usage:
    python -m benchmarks.eval.bench_sweep <N> <total_rates_csv> \
        <samples_per_client> <outdir>

    # e.g. 4 replicas, offered 8/16/24/32 qps total, 200 samples per client:
    python -m benchmarks.eval.bench_sweep 4 8,16,24,32 200 /tmp/sweep_dp4

Assumes N replicas already healthy on 127.0.0.1:8801..880N (e.g. launched
via examples/mps_dp/launch.sh or autodp.sh). Clients use disjoint
seed-tts-eval EN shards within a stage so shared caches cannot inflate
multi-client numbers.

Environment:
    BENCH_MODEL   model name passed to benchmark_tts_seedtts (default higgs)
    CLIENT_CORES  optional comma-separated CPU list to pin client processes
                  to (keeps load generation off the server cores); unpinned
                  when unset

Per-stage results (offered vs achieved qps, TTFC/RTF/latency percentiles,
and full sorted arrays for CDF plots) are written to <outdir>/sweep.json.
"""
import json
import os
import subprocess
import sys
import time

N = int(sys.argv[1])
RATES = [float(r) for r in sys.argv[2].split(",")]
SAMPLES = int(sys.argv[3])
OUT = sys.argv[4]
EN_TOTAL = 1088  # seed-tts-eval EN split size

os.makedirs(OUT, exist_ok=True)
sweep = []
shard_slot = 0
for rate in RATES:
    per_client_rate = rate / N
    stage_dir = os.path.join(OUT, f"rate{rate:g}")
    procs = []
    for i in range(N):
        # Disjoint shards within a stage; cycle across stages when the EN
        # split (1088) is exhausted (later stages may replay earlier texts).
        offset = (shard_slot * SAMPLES) % (EN_TOTAL - SAMPLES)
        shard_slot += 1
        cdir = os.path.join(stage_dir, f"client{i}")
        cmd = [
            sys.executable, "-m", "benchmarks.eval.benchmark_tts_seedtts",
            "--generate-only", "--use-existing-server",
            "--meta", "zhaochenyang20/seed-tts-eval-arrow",
            "--model", os.environ.get("BENCH_MODEL", "higgs"),
            "--host", "127.0.0.1", "--port", str(8801 + i),
            "--ref-format", "references", "--stream", "--lang", "en",
            "--max-samples", str(SAMPLES), "--sample-offset", str(offset),
            "--concurrency", "0", "--request-rate", str(per_client_rate),
            "--output-dir", cdir, "--disable-tqdm",
        ]
        logf = open(os.path.join(OUT, f"rate{rate:g}_client{i}.log"), "w")
        preexec = None
        if os.environ.get("CLIENT_CORES"):
            client_cores = {
                int(c) for c in os.environ["CLIENT_CORES"].split(",")
            }
            preexec = lambda: os.sched_setaffinity(0, client_cores)  # noqa: E731
        procs.append((subprocess.Popen(
            cmd, stdout=logf, stderr=subprocess.STDOUT, preexec_fn=preexec,
        ), logf))
    t0 = time.time()
    fails = 0
    for p, logf in procs:
        fails += p.wait() != 0
        logf.close()
    wall = time.time() - t0

    ttfcs, lats, rtfs, audio_secs = [], [], [], []
    completed, failed = 0, 0
    achieved_runner = audio_sps = 0.0  # from each client's runner wall clock
    for i in range(N):
        f = os.path.join(stage_dir, f"client{i}", "speed_results.json")
        if not os.path.exists(f):
            continue
        d = json.load(open(f))
        for r in d["per_request"]:
            if r.get("is_success") and r.get("audio_ttfp_s") is not None:
                ttfcs.append(r["audio_ttfp_s"])
                lats.append(r["latency_s"])
                if r.get("rtf") is not None:
                    rtfs.append(r["rtf"])
                if r.get("audio_duration_s") is not None:
                    audio_secs.append(r["audio_duration_s"])
        s = d["summary"]
        completed += s["completed_requests"]
        failed += s["failed_requests"]
        achieved_runner += s.get("throughput_qps") or 0
        audio_sps += s.get("audio_throughput_s_per_s") or 0
    ttfcs.sort(); rtfs.sort(); lats_sorted = sorted(lats)

    def pct(arr, p):
        return arr[min(int(p * len(arr)), len(arr) - 1)] if arr else None

    stage = {
        "offered_total_rate": rate,
        # completed / (spawn-to-exit wall): includes client startup + drain —
        # use achieved_qps_runner for capacity claims.
        "achieved_qps": round(completed / wall, 2),
        "achieved_qps_runner": round(achieved_runner, 2),
        "audio_s_per_s": round(audio_sps, 1),
        "wall_s": round(wall, 1),
        "completed": completed, "failed": failed, "client_fails": fails,
        "ttfc_p50": pct(ttfcs, 0.50), "ttfc_p90": pct(ttfcs, 0.90),
        "ttfc_p95": pct(ttfcs, 0.95), "ttfc_p99": pct(ttfcs, 0.99),
        "ttfc_sorted": ttfcs,  # full CDF support
        "rtf_p50": pct(rtfs, 0.50), "rtf_p95": pct(rtfs, 0.95),
        "rtf_p99": pct(rtfs, 0.99),
        "rtf_sorted": rtfs,  # full CDF support
        "latency_mean": round(sum(lats) / len(lats), 3) if lats else None,
        "latency_p50": pct(lats_sorted, 0.50), "latency_p99": pct(lats_sorted, 0.99),
        "audio_duration_mean_s": (
            round(sum(audio_secs) / len(audio_secs), 3) if audio_secs else None
        ),
    }
    sweep.append(stage)
    rtf50 = f"{stage['rtf_p50']:.2f}" if stage["rtf_p50"] is not None else "-"
    print(f"rate={rate:g} achieved={stage['achieved_qps_runner']} "
          f"audio_s/s={stage['audio_s_per_s']} rtf_p50={rtf50} "
          f"p50={stage['ttfc_p50']:.3f} p99={stage['ttfc_p99']:.3f} "
          f"failed={failed}", flush=True)
    time.sleep(5)  # drain between stages

with open(os.path.join(OUT, "sweep.json"), "w") as f:
    json.dump({"n_replicas": N, "samples_per_client": SAMPLES,
               "stages": sweep}, f)
print("sweep written:", os.path.join(OUT, "sweep.json"))
