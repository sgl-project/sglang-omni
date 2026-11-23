"""Drive the /v1/audio/speech endpoint over the fixed SeedTTS English order.

Fixed order, fixed public seed, fixed concurrency; warmup requests are not
recorded. Latency is client-side full-response time. Throughput is reported
twice: once over attempted requests and once over requests that produced a
valid WAV, so a run that silently drops or corrupts responses cannot look
faster than it is. Peak HBM usage is polled from npu-smi during the timed
phase; MB values require npu-smi to report the device capacity, otherwise the
percentage is reported on its own.
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
    cmd = []
    if wrapper:
        cmd.append(wrapper)
    cmd.extend([npu_smi, "info", "-t", "usages", "-i", str(device_index)])
    while not stop.is_set():
        try:
            out = subprocess.run(
                cmd, capture_output=True, text=True, timeout=10
            ).stdout
            capacity_mb = None
            rate = None
            for line in out.splitlines():
                if "HBM Capacity(MB)" in line:
                    try:
                        capacity_mb = float(line.split(":")[-1].strip())
                    except ValueError:
                        pass
                elif "HBM Usage Rate(%)" in line:
                    try:
                        rate = float(line.split(":")[-1].strip())
                    except ValueError:
                        pass
            if rate is not None:
                samples.append((time.perf_counter(), rate, capacity_mb))
        except Exception:  # noqa: BLE001 - polling must never abort the run
            pass
        stop.wait(1.0)


def wav_duration(path: pathlib.Path) -> float | None:
    try:
        return sf.info(str(path)).duration
    except Exception:  # noqa: BLE001 - invalid WAV is a recorded failure
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


def latency_stats(values: list[float]) -> dict | None:
    if not values:
        return None
    lat = np.asarray(values)
    return {
        "mean": round(float(lat.mean()), 4),
        "p50": round(float(np.percentile(lat, 50)), 4),
        "p95": round(float(np.percentile(lat, 95)), 4),
        "max": round(float(lat.max()), 4),
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
    mem_samples: list[tuple[float, float, float | None]] = []
    mem_thread = threading.Thread(
        target=poll_memory,
        args=(args.device_index, args.npu_smi_wrapper, args.npu_smi, stop, mem_samples),
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
    ok_latencies = []
    attempted_latencies = []
    audio_seconds = 0.0
    failures = []
    for row, res in zip(rows, results):
        index = row["index"]
        wav_path = out_dir / f"{index:05d}_{row['sample_id']}.wav"
        http_ok = res["status"] == 200
        is_riff = res["body"][:4] == b"RIFF"
        duration = None
        reason = None
        if not http_ok:
            reason = "http_error"
        elif not is_riff:
            reason = "invalid_wav_header"
        else:
            wav_path.write_bytes(res["body"])
            duration = wav_duration(wav_path)
            if duration is None:
                reason = "unreadable_wav"
        ok = http_ok and is_riff and duration is not None
        if ok:
            audio_seconds += duration
        if http_ok:
            attempted_latencies.append(res["latency_s"])
            if ok:
                ok_latencies.append(res["latency_s"])
        if not ok:
            failures.append(
                {
                    "index": index,
                    "sample_id": row["sample_id"],
                    "status": res["status"],
                    "reason": reason,
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
                "reason": reason,
                "latency_s": round(res["latency_s"], 4),
                "wav_s": duration,
                "bytes": len(res["body"]),
                "sample_rate": res["headers"].get("X-Sample-Rate"),
                "error": res["error"],
            }
        )

    timed_mem = [
        (rate, capacity_mb)
        for stamp, rate, capacity_mb in mem_samples
        if timed_start <= stamp <= timed_end
    ]
    timed_rates = [rate for rate, _ in timed_mem]
    timed_mb = [
        capacity_mb * rate / 100
        for rate, capacity_mb in timed_mem
        if capacity_mb is not None
    ]

    num_ok = sum(1 for record in records if record["ok"])
    num_http_200 = sum(1 for record in records if record["status"] == 200)
    wall = timed_end - timed_start
    success = num_ok == len(rows)
    summary = {
        "run_id": args.run_id,
        "started_at": args.started_at,
        "base_url": args.base_url,
        "model": args.model,
        "seed": args.seed,
        "concurrency": args.concurrency,
        "warmup": args.warmup,
        "num_requests": len(rows),
        "num_ok": num_ok,
        "num_http_200": num_http_200,
        "num_failed": len(failures),
        "success": success,
        "wall_s": round(wall, 3),
        "successful_throughput_rps": round(num_ok / wall, 4) if wall > 0 else None,
        "attempted_throughput_rps": round(len(rows) / wall, 4) if wall > 0 else None,
        "latency_s": latency_stats(ok_latencies),
        "attempted_latency_s": latency_stats(attempted_latencies),
        "generated_audio_s": round(audio_seconds, 2),
        "audio_rtf": round(wall / audio_seconds, 4) if audio_seconds else None,
        "hbm_peak_percent": max(timed_rates) if timed_rates else None,
        "hbm_peak_mb": round(max(timed_mb)) if timed_mb else None,
        "failures": failures,
        "records": records,
    }
    if args.warmup:
        summary["warmup_ok"] = sum(
            1
            for res in warmup_results
            if res["status"] == 200 and res["body"][:4] == b"RIFF"
        )
    pathlib.Path(out_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--model", default="MOSS-TTS-v1.5")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--npu-smi", default="npu-smi")
    parser.add_argument(
        "--npu-smi-wrapper",
        default="",
        help="optional wrapper prefix for npu-smi (e.g. an env launcher)",
    )
    parser.add_argument("--run-id", default="run")
    parser.add_argument("--extra", type=json.loads, default={})
    args = parser.parse_args()
    args.started_at = time.strftime("%Y-%m-%dT%H:%M:%S")

    summary = asyncio.run(run(args))
    print(json.dumps({k: v for k, v in summary.items() if k != "records"}, indent=1))
    raise SystemExit(0 if summary["success"] else 1)


if __name__ == "__main__":
    main()
