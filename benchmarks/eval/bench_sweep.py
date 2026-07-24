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

Per-stage results are written to <outdir>/sweep.json. A stage is only
formally reported when every client exited zero and produced a result under
this run's nonce-named directory; otherwise the stage is marked invalid,
carries no metrics, and the sweep exits nonzero.
"""

import json
import os
import signal
import subprocess
import sys
import time

import numpy as np

EN_TOTAL = 1088  # seed-tts-eval EN split size


def _assert_fresh_dir(path):
    # Note (Jiaxin Deng): stale results reached sweep.json through a reused
    # deterministic directory whose cleanup errors were swallowed; refuse any
    # pre-existing path (including symlinks) instead of trying to delete it,
    # and let makedirs raise on permission problems.
    if os.path.islink(path) or os.path.lexists(path):
        sys.exit(f"bench_sweep: refusing pre-existing path {path}")
    os.makedirs(path)


def _spawn_client(out, stage_dir, per_client_rate, samples, offset, i):
    cdir = os.path.join(stage_dir, f"client{i}")
    if os.path.islink(cdir) or os.path.lexists(cdir):
        sys.exit(f"bench_sweep: refusing pre-existing client path {cdir}")
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
        str(samples),
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
    logf = open(os.path.join(stage_dir, f"client{i}.log"), "w")
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


def _collect_stage(stage_dir, n, rcs):
    """Return (per-client result dicts, invalid_reason).

    Note (Jiaxin Deng): a nonzero exit, a missing result, or an unreadable
    result invalidates the WHOLE stage; a partially failed stage must never
    surface formal numbers.
    """
    if any(rc != 0 for rc in rcs):
        return None, f"client exit codes {rcs}"
    results = []
    for i in range(n):
        f = os.path.join(stage_dir, f"client{i}", "speed_results.json")
        try:
            with open(f) as fh:
                results.append(json.load(fh))
        except (OSError, ValueError) as exc:
            return None, f"client{i} result unreadable: {exc!r}"
    return results, None


def _stage_metrics(rate, wall, results):
    ttfcs, lats, rtfs, audio_secs = [], [], [], []
    completed, failed = 0, 0
    achieved_runner = audio_sps = 0.0  # from each client's runner wall clock
    for d in results:
        for r in d["per_request"]:
            if r.get("is_success"):
                lats.append(r["latency_s"])
                # Note (Yueying Li): TTFC only exists for streaming runs;
                # latency/RTF always do.
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

    return {
        "offered_total_rate": rate,
        # Note (Yueying Li): completed / (spawn-to-exit wall) includes client
        # startup + drain; use achieved_qps_runner for capacity claims.
        "achieved_qps": round(completed / wall, 2),
        "achieved_qps_runner": round(achieved_runner, 2),
        "audio_s_per_s": round(audio_sps, 1),
        "wall_s": round(wall, 1),
        "completed": completed,
        "failed": failed,
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


def main():
    n = int(sys.argv[1])
    rates = [float(r) for r in sys.argv[2].split(",")]
    samples = int(sys.argv[3])
    out = sys.argv[4]
    # Note (Jiaxin Deng): stage directories are nonce-named so results can only
    # come from clients this run spawned; a reused OUT cannot alias them.
    nonce = f"{time.strftime('%Y%m%d-%H%M%S')}-{os.getpid()}"

    os.makedirs(out, exist_ok=True)
    # Note (Jiaxin Deng): each client in a stage takes a disjoint SAMPLES-length
    # block so a shared server cache cannot inflate one client's numbers; N
    # disjoint blocks must exist in the EN split, and stages rotate the
    # starting block.
    num_slots = EN_TOTAL // samples if samples > 0 else 0
    if not (0 < n <= num_slots):
        sys.exit(
            f"bench_sweep: need 0 < N <= EN_TOTAL//SAMPLES ({num_slots}) for "
            f"disjoint shards, got N={n} SAMPLES={samples}"
        )

    sweep = []
    invalid_stages = 0
    for stage_index, rate in enumerate(rates):
        per_client_rate = rate / n
        stage_dir = os.path.join(out, f"rate{rate:g}-{nonce}")
        _assert_fresh_dir(stage_dir)
        procs = []
        rcs = []
        # Note (Jiaxin Deng): terminate every started client and close its log
        # even if a later spawn fails, so a partial stage cannot leak traffic
        # or handles into the next stage.
        # Note (Yueying Li): teardown signals each client's whole session
        # (SIGTERM, bounded wait, SIGKILL escalation) and reaps it, so traffic
        # is provably stopped before the stage exits and no zombie survives
        # into the next one.
        try:
            for i in range(n):
                slot = (stage_index + i) % num_slots
                procs.append(
                    _spawn_client(
                        out, stage_dir, per_client_rate, samples, slot * samples, i
                    )
                )
            t0 = time.time()
            for p, _logf in procs:
                rcs.append(p.wait())
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

        results, invalid = _collect_stage(stage_dir, n, rcs)
        if invalid:
            invalid_stages += 1
            stage = {"offered_total_rate": rate, "invalid": invalid}
            print(f"rate={rate:g} INVALID: {invalid}", flush=True)
        else:
            stage = _stage_metrics(rate, wall, results)

            def fmt(v, spec):
                return format(v, spec) if v is not None else "-"

            print(
                f"rate={rate:g} achieved={stage['achieved_qps_runner']} "
                f"audio_s/s={stage['audio_s_per_s']} "
                f"rtf_p50={fmt(stage['rtf_p50'], '.2f')} "
                f"p50={fmt(stage['ttfc_p50'], '.3f')} "
                f"p99={fmt(stage['ttfc_p99'], '.3f')} "
                f"failed={stage['failed']}",
                flush=True,
            )
        sweep.append(stage)
        time.sleep(5)  # drain between stages

    with open(os.path.join(out, "sweep.json"), "w") as f:
        json.dump(
            {
                "n_replicas": n,
                "samples_per_client": samples,
                "nonce": nonce,
                "stages": sweep,
            },
            f,
        )
    print("sweep written:", os.path.join(out, "sweep.json"))
    if invalid_stages:
        sys.exit(f"bench_sweep: {invalid_stages} invalid stage(s); not formal output")


if __name__ == "__main__":
    main()
