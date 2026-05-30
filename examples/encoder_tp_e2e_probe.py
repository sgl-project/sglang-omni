# SPDX-License-Identifier: Apache-2.0
"""Probe a running SGLang-Omni server for Encoder TP E2E validation.

This helper does not launch the server. Use it after starting either
upstream-main or a PR-branch server so the same request payload can be reused
across the required evidence chain:

1. latest ``origin/main`` reproduction;
2. same PR commit, encoder ``tp=1`` baseline;
3. same PR commit, encoder ``tp>1`` comparison.

It writes a small artifact directory containing git state, CUDA/NVML snapshots,
global and process-level GPU sample summaries, request JSON, per-run responses,
and a summary. The output is intentionally plain JSON so it can be pasted into
``docs/developer_reference/encoder_tp_performance_report.md``.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="qwen3-omni")
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--repo", default=".")
    parser.add_argument("--video", default=None, help="Local path or URL")
    parser.add_argument("--audio", default=None, help="Local path or URL")
    parser.add_argument("--video-fps", type=float, default=None)
    parser.add_argument("--video-max-frames", type=int, default=None)
    parser.add_argument("--video-min-pixels", type=int, default=None)
    parser.add_argument("--video-max-pixels", type=int, default=None)
    parser.add_argument(
        "--audio-no-truncation",
        action="store_true",
        help="Set audio_truncation=false in the request payload.",
    )
    parser.add_argument(
        "--prompt",
        required=True,
        help="User prompt sent with the video/audio payload.",
    )
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument(
        "--gpu-sample-interval",
        type=float,
        default=0.5,
        help=(
            "Seconds between nvidia-smi samples while each request is in flight. "
            "Set to 0 to disable peak-memory/utilization sampling."
        ),
    )
    parser.add_argument(
        "--expect-substring",
        action="append",
        default=[],
        help="Case-insensitive substring that must appear in response text.",
    )
    return parser.parse_args()


def _run_cmd(cmd: list[str], *, cwd: str | None = None) -> dict[str, Any]:
    try:
        proc = subprocess.run(
            cmd,
            cwd=cwd,
            text=True,
            capture_output=True,
            timeout=30,
            check=False,
        )
        return {
            "cmd": cmd,
            "returncode": proc.returncode,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
        }
    except Exception as exc:  # noqa: BLE001
        return {"cmd": cmd, "error": repr(exc)}


def _query_gpu_sample() -> dict[str, Any]:
    fields = [
        "index",
        "name",
        "uuid",
        "memory.used",
        "memory.free",
        "utilization.gpu",
    ]
    cmd = [
        "nvidia-smi",
        f"--query-gpu={','.join(fields)}",
        "--format=csv,noheader,nounits",
    ]
    result = _run_cmd(cmd)
    if result.get("returncode") != 0:
        return {"ok": False, "result": result}

    gpus: list[dict[str, Any]] = []
    for line in str(result.get("stdout", "")).splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != len(fields):
            continue
        try:
            gpus.append(
                {
                    "index": int(parts[0]),
                    "name": parts[1],
                    "uuid": parts[2],
                    "memory_used_mib": int(parts[3]),
                    "memory_free_mib": int(parts[4]),
                    "utilization_gpu_percent": int(parts[5]),
                }
            )
        except ValueError:
            continue

    process_sample = _query_gpu_process_sample()
    sample = {
        "ok": True,
        "gpus": gpus,
        "process_query_ok": process_sample.get("ok"),
        "processes": process_sample.get("processes", []),
    }
    if not process_sample.get("ok"):
        sample["process_result"] = process_sample.get("result")
    return sample


def _query_gpu_process_sample() -> dict[str, Any]:
    fields = ["pid", "process_name", "gpu_uuid", "used_memory"]
    cmd = [
        "nvidia-smi",
        f"--query-compute-apps={','.join(fields)}",
        "--format=csv,noheader,nounits",
    ]
    result = _run_cmd(cmd)
    if result.get("returncode") != 0:
        return {"ok": False, "result": result, "processes": []}

    processes: list[dict[str, Any]] = []
    for line in str(result.get("stdout", "")).splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != len(fields):
            continue
        try:
            processes.append(
                {
                    "pid": int(parts[0]),
                    "process_name": parts[1],
                    "gpu_uuid": parts[2],
                    "used_memory_mib": int(parts[3]),
                }
            )
        except ValueError:
            continue
    return {"ok": True, "processes": processes}


class _GpuSampler:
    def __init__(self, interval_s: float) -> None:
        self._interval_s = max(float(interval_s), 0.0)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.samples: list[dict[str, Any]] = []

    def __enter__(self) -> "_GpuSampler":
        if self._interval_s <= 0:
            return self
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *args: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(self._interval_s * 2, 1.0))

    def _run(self) -> None:
        while not self._stop.is_set():
            sample = _query_gpu_sample()
            sample["time_s"] = time.time()
            self.samples.append(sample)
            self._stop.wait(self._interval_s)


def _summarize_gpu_samples(samples: list[dict[str, Any]]) -> dict[str, Any]:
    by_gpu: dict[str, dict[str, Any]] = {}
    for sample in samples:
        if not sample.get("ok"):
            continue
        for gpu in sample.get("gpus", []):
            index = str(gpu["index"])
            current = by_gpu.setdefault(
                index,
                {
                    "index": gpu["index"],
                    "name": gpu["name"],
                    "uuid": gpu.get("uuid"),
                    "initial_memory_used_mib": gpu["memory_used_mib"],
                    "initial_memory_free_mib": gpu["memory_free_mib"],
                    "final_memory_used_mib": gpu["memory_used_mib"],
                    "final_memory_free_mib": gpu["memory_free_mib"],
                    "max_memory_used_mib": gpu["memory_used_mib"],
                    "max_memory_delta_mib": 0,
                    "min_memory_free_mib": gpu["memory_free_mib"],
                    "max_utilization_gpu_percent": gpu[
                        "utilization_gpu_percent"
                    ],
                    "samples": 0,
                },
            )
            current["final_memory_used_mib"] = gpu["memory_used_mib"]
            current["final_memory_free_mib"] = gpu["memory_free_mib"]
            current["max_memory_used_mib"] = max(
                current["max_memory_used_mib"],
                gpu["memory_used_mib"],
            )
            current["max_memory_delta_mib"] = max(
                current["max_memory_delta_mib"],
                gpu["memory_used_mib"] - current["initial_memory_used_mib"],
            )
            current["min_memory_free_mib"] = min(
                current["min_memory_free_mib"],
                gpu["memory_free_mib"],
            )
            current["max_utilization_gpu_percent"] = max(
                current["max_utilization_gpu_percent"],
                gpu["utilization_gpu_percent"],
            )
            current["samples"] += 1
    return {
        "sample_count": len(samples),
        "gpus": [by_gpu[index] for index in sorted(by_gpu, key=int)],
    }


def _summarize_gpu_process_samples(samples: list[dict[str, Any]]) -> dict[str, Any]:
    by_process: dict[tuple[str, int], dict[str, Any]] = {}
    for sample in samples:
        if not sample.get("ok"):
            continue
        for process in sample.get("processes", []):
            gpu_uuid = str(process.get("gpu_uuid", ""))
            if not gpu_uuid or "pid" not in process:
                continue
            try:
                pid = int(process["pid"])
                used_memory_mib = int(process["used_memory_mib"])
            except (TypeError, ValueError):
                continue
            key = (gpu_uuid, pid)
            current = by_process.setdefault(
                key,
                {
                    "gpu_uuid": gpu_uuid,
                    "pid": pid,
                    "process_name": process.get("process_name"),
                    "initial_used_memory_mib": used_memory_mib,
                    "final_used_memory_mib": used_memory_mib,
                    "max_used_memory_mib": used_memory_mib,
                    "max_memory_delta_mib": 0,
                    "samples": 0,
                },
            )
            current["process_name"] = process.get("process_name")
            current["final_used_memory_mib"] = used_memory_mib
            current["max_used_memory_mib"] = max(
                current["max_used_memory_mib"],
                used_memory_mib,
            )
            current["max_memory_delta_mib"] = max(
                current["max_memory_delta_mib"],
                used_memory_mib - current["initial_used_memory_mib"],
            )
            current["samples"] += 1

    return {
        "sample_count": len(samples),
        "processes": [
            by_process[key]
            for key in sorted(by_process, key=lambda item: (item[0], item[1]))
        ],
    }


def _write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")


def _post_json(url: str, payload: dict[str, Any], *, timeout: float) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8", errors="replace")
            try:
                parsed = json.loads(body)
            except json.JSONDecodeError:
                parsed = {"raw_body": body}
            return {
                "ok": 200 <= resp.status < 300,
                "status": resp.status,
                "body": parsed,
            }
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        try:
            parsed = json.loads(body)
        except json.JSONDecodeError:
            parsed = {"raw_body": body}
        return {"ok": False, "status": exc.code, "body": parsed}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": repr(exc)}


def _get_health(base_url: str, *, timeout: float = 10.0) -> dict[str, Any]:
    url = base_url.rstrip("/") + "/health"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            body = resp.read().decode("utf-8", errors="replace")
            return {"ok": 200 <= resp.status < 300, "status": resp.status, "body": body}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": repr(exc)}


def build_request(args: argparse.Namespace) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": args.model,
        "messages": [{"role": "user", "content": args.prompt}],
        "modalities": ["text"],
        "max_tokens": args.max_tokens,
        "stream": False,
    }
    if args.video:
        payload["videos"] = [args.video]
        if args.video_fps is not None:
            payload["video_fps"] = args.video_fps
        if args.video_max_frames is not None:
            payload["video_max_frames"] = args.video_max_frames
        if args.video_min_pixels is not None:
            payload["video_min_pixels"] = args.video_min_pixels
        if args.video_max_pixels is not None:
            payload["video_max_pixels"] = args.video_max_pixels
    if args.audio:
        payload["audios"] = [args.audio]
        if args.audio_no_truncation:
            payload["audio_truncation"] = False
    if not args.video and not args.audio:
        raise ValueError("at least one of --video or --audio is required")
    return payload


def extract_response_text(response_body: dict[str, Any]) -> str:
    choices = response_body.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    message = choices[0].get("message") if isinstance(choices[0], dict) else None
    if not isinstance(message, dict):
        return ""
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
        return "".join(parts)
    return ""


def _record_environment(out_dir: Path, repo: str) -> None:
    _write_json(
        out_dir / "environment.json",
        {
            "python": sys.version,
            "platform": platform.platform(),
            "cwd": os.getcwd(),
            "repo": str(Path(repo).resolve()),
            "torch": _run_cmd(
                [
                    sys.executable,
                    "-c",
                    (
                        "import torch, json; "
                        "print(json.dumps({"
                        "'version': torch.__version__, "
                        "'cuda': torch.version.cuda, "
                        "'cuda_available': torch.cuda.is_available(), "
                        "'device_count': torch.cuda.device_count()}))"
                    ),
                ],
            ),
        },
    )
    (out_dir / "nvidia_smi_before.txt").write_text(
        json.dumps(_run_cmd(["nvidia-smi"]), indent=2) + "\n"
    )
    for name, cmd in {
        "git_rev_parse_head.json": ["git", "rev-parse", "HEAD"],
        "git_status_short.json": ["git", "status", "--short"],
        "git_diff_stat.json": ["git", "diff", "--stat"],
    }.items():
        _write_json(out_dir / name, _run_cmd(cmd, cwd=repo))


def _validate_text(text: str, expected_substrings: list[str]) -> list[str]:
    errors: list[str] = []
    if not text.strip():
        errors.append("empty response text")
    folded = text.lower()
    for expected in expected_substrings:
        if expected.lower() not in folded:
            errors.append(f"missing expected substring {expected!r}")
    return errors


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percentile / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[int(position)]
    lower_value = ordered[lower]
    upper_value = ordered[upper]
    return lower_value + (upper_value - lower_value) * (position - lower)


def summarize_runs(runs: list[dict[str, Any]]) -> dict[str, Any]:
    measured = [run for run in runs if not run.get("warmup")]
    latencies = [float(run["latency_s"]) for run in measured if run.get("ok")]
    successful = [run for run in measured if run.get("ok")]
    prompt_tokens = [
        int(run["usage"]["prompt_tokens"])
        for run in successful
        if isinstance(run.get("usage"), dict)
        and "prompt_tokens" in run["usage"]
    ]
    completion_tokens = [
        int(run["usage"]["completion_tokens"])
        for run in successful
        if isinstance(run.get("usage"), dict)
        and "completion_tokens" in run["usage"]
    ]
    total_tokens = [
        int(run["usage"]["total_tokens"])
        for run in successful
        if isinstance(run.get("usage"), dict)
        and "total_tokens" in run["usage"]
    ]
    completion_tokens_per_s = [
        int(run["usage"]["completion_tokens"]) / float(run["latency_s"])
        for run in successful
        if isinstance(run.get("usage"), dict)
        and "completion_tokens" in run["usage"]
        and float(run["latency_s"]) > 0
    ]
    total_tokens_per_s = [
        int(run["usage"]["total_tokens"]) / float(run["latency_s"])
        for run in successful
        if isinstance(run.get("usage"), dict)
        and "total_tokens" in run["usage"]
        and float(run["latency_s"]) > 0
    ]
    return {
        "total_runs": len(runs),
        "measured_runs": len(measured),
        "successes": sum(1 for run in measured if run.get("ok")),
        "failures": sum(1 for run in measured if not run.get("ok")),
        "latency_s_mean": statistics.mean(latencies) if latencies else None,
        "latency_s_stdev": statistics.stdev(latencies) if len(latencies) > 1 else None,
        "latency_s_p50": statistics.median(latencies) if latencies else None,
        "latency_s_p90": _percentile(latencies, 90),
        "latency_s_p95": _percentile(latencies, 95),
        "latency_s_max": max(latencies) if latencies else None,
        "prompt_tokens_mean": statistics.mean(prompt_tokens)
        if prompt_tokens
        else None,
        "prompt_tokens_max": max(prompt_tokens) if prompt_tokens else None,
        "completion_tokens_mean": statistics.mean(completion_tokens)
        if completion_tokens
        else None,
        "completion_tokens_max": max(completion_tokens)
        if completion_tokens
        else None,
        "total_tokens_mean": statistics.mean(total_tokens) if total_tokens else None,
        "total_tokens_max": max(total_tokens) if total_tokens else None,
        "completion_tokens_per_s_mean": statistics.mean(completion_tokens_per_s)
        if completion_tokens_per_s
        else None,
        "total_tokens_per_s_mean": statistics.mean(total_tokens_per_s)
        if total_tokens_per_s
        else None,
    }


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir) / args.case_id
    out_dir.mkdir(parents=True, exist_ok=True)

    request = build_request(args)
    _write_json(out_dir / "request.json", request)
    _record_environment(out_dir, args.repo)

    health_before = _get_health(args.base_url)
    _write_json(out_dir / "health_before.json", health_before)

    endpoint = args.base_url.rstrip("/") + "/v1/chat/completions"
    runs: list[dict[str, Any]] = []
    gpu_samples: list[dict[str, Any]] = []
    for idx in range(args.warmup + args.repeat):
        is_warmup = idx < args.warmup
        started = time.perf_counter()
        with _GpuSampler(args.gpu_sample_interval) as sampler:
            response = _post_json(endpoint, request, timeout=args.timeout)
        latency_s = time.perf_counter() - started
        for sample in sampler.samples:
            sample["case_id"] = args.case_id
            sample["run_index"] = idx
            sample["warmup"] = is_warmup
        gpu_samples.extend(sampler.samples)
        text = extract_response_text(response.get("body", {}))
        usage = response.get("body", {}).get("usage")
        validation_errors = _validate_text(text, args.expect_substring)
        ok = bool(response.get("ok")) and not validation_errors
        run = {
            "case_id": args.case_id,
            "index": idx,
            "warmup": is_warmup,
            "ok": ok,
            "latency_s": latency_s,
            "response_status": response.get("status"),
            "response_error": response.get("error"),
            "validation_errors": validation_errors,
            "response_text": text,
            "response_body": response.get("body"),
            "usage": usage if isinstance(usage, dict) else None,
            "gpu_peak_summary": _summarize_gpu_samples(sampler.samples),
            "gpu_process_peak_summary": _summarize_gpu_process_samples(
                sampler.samples
            ),
        }
        runs.append(run)
        with (out_dir / "runs.jsonl").open("a") as f:
            f.write(json.dumps(run, sort_keys=True) + "\n")

    health_after = _get_health(args.base_url)
    _write_json(out_dir / "health_after.json", health_after)
    (out_dir / "nvidia_smi_after.txt").write_text(
        json.dumps(_run_cmd(["nvidia-smi"]), indent=2) + "\n"
    )
    with (out_dir / "gpu_samples.jsonl").open("w") as f:
        for sample in gpu_samples:
            f.write(json.dumps(sample, sort_keys=True) + "\n")
    gpu_peak_summary = _summarize_gpu_samples(gpu_samples)
    gpu_process_peak_summary = _summarize_gpu_process_samples(gpu_samples)
    _write_json(out_dir / "gpu_peak_summary.json", gpu_peak_summary)
    _write_json(
        out_dir / "gpu_process_peak_summary.json",
        gpu_process_peak_summary,
    )

    summary = summarize_runs(runs)
    summary.update(
        {
            "case_id": args.case_id,
            "health_before_ok": health_before.get("ok"),
            "health_after_ok": health_after.get("ok"),
            "output_dir": str(out_dir),
            "gpu_peak_summary": gpu_peak_summary,
            "gpu_process_peak_summary": gpu_process_peak_summary,
        }
    )
    _write_json(out_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))

    if summary["failures"] or not summary["health_after_ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
