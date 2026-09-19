# SPDX-License-Identifier: Apache-2.0
"""Continuous closed-loop seed-tts load generator for throughput / GPU-idle profiling.

Keeps exactly ``--conc`` requests in flight at all times: each of ``conc`` worker
coroutines re-issues the next request the instant its previous one completes, for
``--secs``. No inter-batch drain, so the server stays saturated for the whole
measurement window (this is what makes the device-level SM% window clean). Reuses
the in-repo ``make_send_fn`` + ``load_seedtts_samples`` so requests match the normal
benchmark path exactly.

Run one instance per replica (or several against a router) as a numactl-pinned
subprocess so N clients are N independent GILs (rules out client-GIL as the serial
bottleneck). Point it at a server port (``--port``) or any base URL (``--base-url``,
e.g. a router).

Reads throughput over the steady window [warmup_secs, secs] and a per-5s-bucket
coefficient of variation so you can confirm the load was steady.
"""
import argparse
import asyncio
import json
import os
import statistics as st
import sys
import tempfile
import time

# Make the repo importable without installing (this file lives in benchmarks/dp_throughput).
_REPO = os.environ.get("REPO") or os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import aiohttp  # noqa: E402

from benchmarks.dataset.prepare import DATASETS  # noqa: E402
from benchmarks.dataset.seedtts import load_seedtts_samples  # noqa: E402
from benchmarks.eval.benchmark_omni_seedtts import make_send_fn  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--port", type=int, default=0)
ap.add_argument(
    "--base-url", default=None, help="override http://host:port (e.g. a router)"
)
ap.add_argument("--model", required=True)
ap.add_argument("--conc", type=int, required=True)
ap.add_argument("--secs", type=int, default=90)
ap.add_argument("--warmup-secs", type=int, default=15)
ap.add_argument("--voice-clone", action="store_true")
ap.add_argument("--max-new-tokens", type=int, default=512)
ap.add_argument("--lang", default="en")
ap.add_argument("--meta", default=None, help="dataset id; default DATASETS['seedtts']")
ap.add_argument("--out", required=True)
args = ap.parse_args()

save_dir = tempfile.mkdtemp(dir="/dev/shm" if os.path.isdir("/dev/shm") else None)
meta = args.meta or DATASETS["seedtts"]
samples = load_seedtts_samples(meta, None, split=args.lang)
if not samples:
    raise SystemExit("no seed-tts samples loaded")
N = len(samples)

base_url = args.base_url or f"http://127.0.0.1:{args.port}"
api_url = f"{base_url}/v1/chat/completions"
send_fn = make_send_fn(
    args.model,
    api_url,
    lang=args.lang,
    voice_clone=args.voice_clone,
    speaker="Ethan",
    max_tokens=args.max_new_tokens,
    temperature=0.7,
    stream=False,
    save_audio_dir=save_dir,
)

comp_ts = []  # completion timestamps (rel to t0), success only
errors = [0]
state = {"stop": None, "t0": None}


async def worker(session, k):
    i = k
    while True:
        if time.perf_counter() >= state["stop"]:
            return
        s = samples[i % N]
        i += args.conc
        r = await send_fn(session, s)
        if getattr(r, "is_success", False):
            comp_ts.append(time.perf_counter() - state["t0"])
        else:
            errors[0] += 1


async def main():
    conn = aiohttp.TCPConnector(limit=0, limit_per_host=0)
    timeout = aiohttp.ClientTimeout(total=600)
    async with aiohttp.ClientSession(connector=conn, timeout=timeout) as session:
        state["t0"] = time.perf_counter()
        state["stop"] = state["t0"] + args.secs
        await asyncio.gather(
            *[asyncio.create_task(worker(session, k)) for k in range(args.conc)]
        )


asyncio.run(main())

w0, w1 = args.warmup_secs, args.secs
win = [t for t in comp_ts if w0 <= t <= w1]
qps_window = len(win) / max(1e-9, (w1 - w0))
buckets = {}
for t in comp_ts:
    b = int(t // 5) * 5
    buckets[b] = buckets.get(b, 0) + 1
bucket_qps = {b: round(c / 5.0, 2) for b, c in sorted(buckets.items())}
bvals = [v for b, v in bucket_qps.items() if w0 <= b < w1]

agg = {
    "base_url": base_url,
    "conc": args.conc,
    "secs": args.secs,
    "warmup_secs": args.warmup_secs,
    "n_completions_total": len(comp_ts),
    "n_completions_window": len(win),
    "errors": errors[0],
    "qps_window": round(qps_window, 4),
    "qps_overall": round(len(comp_ts) / max(1e-9, args.secs), 4),
    "bucket_qps_cv": (
        round(st.pstdev(bvals) / (st.mean(bvals) + 1e-9), 3) if bvals else None
    ),
    "bucket_qps": bucket_qps,
}
json.dump(agg, open(args.out, "w"), indent=2)
print(
    f"[client {base_url}] qps_window={qps_window:.3f} n_win={len(win)} "
    f"errors={errors[0]} cv={agg['bucket_qps_cv']}",
    flush=True,
)
