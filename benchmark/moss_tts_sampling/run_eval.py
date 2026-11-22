"""Drive the /v1/audio/speech endpoint over the fixed SeedTTS English order.

Fixed order, fixed public seed, fixed concurrency; warmup requests are not
recorded. Latency is client-side full-response time. Peak HBM usage is polled
from npu-smi during the timed phase.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import pathlib
import subprocess
import threading
import time

import aiohttp
import numpy as np
import soundfile as sf


def poll_memory(device_index: int, wrapper: str, npu_smi: str, stop, samples):
    cmd = [wrapper, npu_smi, "info", "-t", "usages", "-i", str(device_index)]
    while not stop.is_set():
        try:
            out = subprocess.run(
                cmd, capture_output=True, text=True, timeout=10
            ).stdout
            rate = None
            for line in out.splitlines():
                if "HBM Usage Rate(%)" in line:
                    rate = float(line.split(":")[-1].strip())
            if rate is not None:
                samples.append((time.perf_counter(), rate))
        except Exception:
            pass
        stop.wait(1.0)


def wav_duration(path: pathlib.Path) -> float | None:
    try:
        return sf.info(str(path)).duration
    except Exception:
        return None


async def fetch(session, url, payload, sem):
    async with sem:
        t0 = time.perf_counter()
        try:
            async with session.post(url, json=payload) as resp:
                body = await resp.read()
                status = resp.status
                headers = dict(resp.headers)
        except Exception as exc:  # noqa: BLE001 - recorded per item
            return {
                "status": -1,
                "body": b"",
                "latency_s": time.perf_counter() - t0,
                "error": repr(exc),
                "headers": {},
            }
        return {
            "status": status,
            "body": body,
            "latency_s": time.perf_counter() - t0,
            "error": None,
            "headers": headers,
        }


async def run(args) -> dict:
    rows = json.loads(pathlib.Path(args.manifest).read_text())
    if args.limit:
        rows = rows[: args.limit]
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    url = args.base_url.rstrip("/") + "/v1/audio/speech"
    sem = asyncio.Semaphore(args.concurrency)
    timeout = aiohttp.ClientTimeout(total=args.timeout)
    connector = aiohttp.TCPConnector(limit=args.concurrency + 4)

    stop = threading.Event()
    mem_samples: list[tuple[float, float]] = []
    mem_thread = threading.Thread(
        target=poll_memory,
        args=(args.device_index, args.wrapper, args.npu_smi, stop, mem_samples),
        daemon=True,
    )
    mem_thread.start()

    def payload_for(item):
        payload = {
            "model": args.model,
            "input": item["target_text"],
            "ref_audio": item["ref_wav"],
            "ref_text": item["ref_text"],
            "language": "English",
            "response_format": "wav",
            "seed": args.seed,
        }
        payload.update(args.extra)
        return payload

    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        warmup_results = []
        if args.warmup:
            warm = rows[: args.warmup]
            warmup_results = await asyncio.gather(
                *(fetch(session, url, payload_for(row), sem) for row in warm)
            )

        timed_start = time.perf_counter()
        tasks = [
            asyncio.ensure_future(fetch(session, url, payload_for(row), sem))
            for row in rows
        ]
        results = await asyncio.gather(*tasks)
        timed_end = time.perf_counter()

    stop.set()
    mem_thread.join(timeout=5)

    records = []
    latencies = []
    audio_seconds = 0.0
    failures = []
    for row, res in zip(rows, results):
        index = row["index"]
        wav_path = out_dir / f"{index:05d}_{row['sample_id']}.wav"
        ok = res["status"] == 200 and res["body"][:4] == b"RIFF"
        duration = None
        if ok:
            wav_path.write_bytes(res["body"])
            duration = wav_duration(wav_path)
            audio_seconds += duration or 0.0
        if res["status"] == 200:
            latencies.append(res["latency_s"])
        else:
            failures.append(
                {
                    "index": index,
                    "status": res["status"],
                    "error": res["error"],
                    "body": res["body"][:500].decode("utf-8", "replace"),
                }
            )
        records.append(
            {
                "index": index,
                "sample_id": row["sample_id"],
                "status": res["status"],
                "ok": ok,
                "latency_s": round(res["latency_s"], 4),
                "wav_s": duration,
                "bytes": len(res["body"]),
                "sample_rate": res["headers"].get("X-Sample-Rate"),
                "error": res["error"],
            }
        )

    timed_peak = [
        rate
        for stamp, rate in mem_samples
        if timed_start <= stamp <= timed_end
    ]
    lat = np.asarray(latencies) if latencies else np.asarray([0.0])
    wall = timed_end - timed_start
    summary = {
        "run_id": args.run_id,
        "started_at": args.started_at,
        "base_url": args.base_url,
        "model": args.model,
        "seed": args.seed,
        "concurrency": args.concurrency,
        "warmup": args.warmup,
        "num_requests": len(rows),
        "num_ok": int(sum(r["ok"] for r in records)),
        "num_http_200": len(latencies),
        "wall_s": round(wall, 3),
        "throughput_rps": round(len(rows) / wall, 4) if wall > 0 else None,
        "latency_s": {
            "mean": round(float(lat.mean()), 4),
            "p50": round(float(np.percentile(lat, 50)), 4),
            "p95": round(float(np.percentile(lat, 95)), 4),
            "max": round(float(lat.max()), 4),
        },
        "generated_audio_s": round(audio_seconds, 2),
        "audio_rtf": round(wall / audio_seconds, 4) if audio_seconds else None,
        "hbm_peak_percent": max(timed_peak) if timed_peak else None,
        "hbm_peak_mb": round(65536 * max(timed_peak) / 100) if timed_peak else None,
        "failures": failures,
        "records": records,
    }
    if args.warmup:
        summary["warmup_ok"] = sum(
            1 for res in warmup_results if res["status"] == 200
        )
    pathlib.Path(out_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--manifest", default="/root/work/bench/en_manifest.json")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--model", default="MOSS-TTS-v1.5")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--wrapper", default="/tmp/e31-stack/native")
    parser.add_argument("--npu-smi", default="/usr/local/bin/npu-smi")
    parser.add_argument("--run-id", default="run")
    parser.add_argument("--extra", type=json.loads, default={})
    args = parser.parse_args()
    args.started_at = time.strftime("%Y-%m-%dT%H:%M:%S")

    summary = asyncio.run(run(args))
    print(json.dumps({k: v for k, v in summary.items() if k != "records"}, indent=1))


if __name__ == "__main__":
    main()
