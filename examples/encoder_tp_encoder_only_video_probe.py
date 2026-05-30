# SPDX-License-Identifier: Apache-2.0
"""Encoder-only video memory probe for Qwen3-Omni encoder TP.

This script intentionally does not launch thinker, talker, code2wav, or an
HTTP server. It preprocesses one video once per frame cap, then runs only the
SGLang-backed image/video encoder forward in fresh subprocesses:

* TP1: one process on one GPU.
* TP2/TP4/etc.: one process per rank/GPU, rank 0 owning input IO and followers
  receiving the scheduler's metadata/tensor fan-out.

The output is an artifact directory with raw logs, NVML samples, per-rank
results, and a small linear fit over successful same-length TP points.
It is for evidence collection only; it is not a benchmark harness for E2E
generation quality or latency.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import math
import os
import pickle
import re
import signal
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VIDEO = (
    "/data/encoder_tp_evidence_20260526/media/draw_loop_2048frames.mp4"
)
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _json_default(value: Any) -> Any:
    try:
        import torch

        if torch.is_tensor(value):
            return {
                "tensor": True,
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "device": str(value.device),
            }
    except Exception:  # noqa: BLE001
        pass
    if isinstance(value, Path):
        return str(value)
    return repr(value)


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, indent=2, sort_keys=True, default=_json_default),
        encoding="utf-8",
    )


def _run_cmd(cmd: list[str], *, timeout: float = 30.0) -> dict[str, Any]:
    try:
        proc = subprocess.run(
            cmd,
            text=True,
            capture_output=True,
            timeout=timeout,
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
    gpu_fields = [
        "index",
        "uuid",
        "name",
        "memory.used",
        "memory.free",
        "utilization.gpu",
    ]
    gpu_res = _run_cmd(
        [
            "nvidia-smi",
            f"--query-gpu={','.join(gpu_fields)}",
            "--format=csv,noheader,nounits",
        ]
    )
    process_res = _run_cmd(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name,gpu_uuid,used_memory",
            "--format=csv,noheader,nounits",
        ]
    )
    sample: dict[str, Any] = {
        "time_s": time.time(),
        "ok": gpu_res.get("returncode") == 0,
        "gpu_result": gpu_res if gpu_res.get("returncode") != 0 else None,
        "process_result": (
            process_res if process_res.get("returncode") != 0 else None
        ),
        "gpus": [],
        "processes": [],
    }
    if gpu_res.get("returncode") == 0:
        for line in str(gpu_res.get("stdout", "")).splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) != len(gpu_fields):
                continue
            try:
                sample["gpus"].append({
                    "index": int(parts[0]),
                    "uuid": parts[1],
                    "name": parts[2],
                    "memory_used_mib": int(parts[3]),
                    "memory_free_mib": int(parts[4]),
                    "utilization_gpu_percent": int(parts[5]),
                })
            except ValueError:
                continue
    if process_res.get("returncode") == 0:
        for line in str(process_res.get("stdout", "")).splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) != 4:
                continue
            try:
                sample["processes"].append({
                    "pid": int(parts[0]),
                    "process_name": parts[1],
                    "gpu_uuid": parts[2],
                    "used_memory_mib": int(parts[3]),
                })
            except ValueError:
                continue
    return sample


class _GpuSampler:
    def __init__(self, path: Path, interval_s: float) -> None:
        self.path = path
        self.interval_s = max(float(interval_s), 0.0)
        self.samples: list[dict[str, Any]] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self) -> "_GpuSampler":
        if self.interval_s <= 0:
            return self
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *args: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(self.interval_s * 2.0, 1.0))

    def _run(self) -> None:
        with self.path.open("a", encoding="utf-8") as f:
            while not self._stop.is_set():
                sample = _query_gpu_sample()
                self.samples.append(sample)
                f.write(json.dumps(sample, sort_keys=True) + "\n")
                f.flush()
                self._stop.wait(self.interval_s)


def _summarize_samples(
    samples: list[dict[str, Any]],
    *,
    gpu_indices: list[int],
    pids: list[int],
) -> dict[str, Any]:
    uuid_to_index: dict[str, int] = {}
    gpu_summary: dict[str, dict[str, Any]] = {}
    proc_summary: dict[str, dict[str, Any]] = {}
    pid_set = {int(pid) for pid in pids if int(pid) > 0}
    gpu_set = {int(gpu) for gpu in gpu_indices}
    for sample in samples:
        if not sample.get("ok"):
            continue
        for gpu in sample.get("gpus", []):
            idx = int(gpu["index"])
            uuid_to_index[str(gpu["uuid"])] = idx
            if idx not in gpu_set:
                continue
            key = str(idx)
            current = gpu_summary.setdefault(
                key,
                {
                    "index": idx,
                    "name": gpu["name"],
                    "peak_used_mib": 0,
                    "min_free_mib": None,
                    "max_utilization_gpu_percent": 0,
                },
            )
            current["peak_used_mib"] = max(
                int(current["peak_used_mib"]),
                int(gpu["memory_used_mib"]),
            )
            if current["min_free_mib"] is None:
                current["min_free_mib"] = int(gpu["memory_free_mib"])
            else:
                current["min_free_mib"] = min(
                    int(current["min_free_mib"]),
                    int(gpu["memory_free_mib"]),
                )
            current["max_utilization_gpu_percent"] = max(
                int(current["max_utilization_gpu_percent"]),
                int(gpu["utilization_gpu_percent"]),
            )
        for proc in sample.get("processes", []):
            pid = int(proc["pid"])
            if pid_set and pid not in pid_set:
                continue
            gpu_uuid = str(proc["gpu_uuid"])
            key = str(pid)
            current = proc_summary.setdefault(
                key,
                {
                    "pid": pid,
                    "process_name": proc["process_name"],
                    "gpu_uuid": gpu_uuid,
                    "gpu_index": uuid_to_index.get(gpu_uuid),
                    "peak_used_mib": 0,
                },
            )
            current["gpu_index"] = uuid_to_index.get(gpu_uuid, current["gpu_index"])
            current["peak_used_mib"] = max(
                int(current["peak_used_mib"]),
                int(proc["used_memory_mib"]),
            )
    return {
        "gpus": gpu_summary,
        "processes": proc_summary,
        "sample_count": len(samples),
    }


def _pick_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _parse_int_list(value: str) -> list[int]:
    out: list[int] = []
    for raw in value.split(","):
        item = raw.strip()
        if not item:
            continue
        out.append(int(item))
    if not out:
        raise argparse.ArgumentTypeError("expected at least one integer")
    return out


def _parse_tp_specs(value: str) -> list[tuple[int, list[int]]]:
    specs: list[tuple[int, list[int]]] = []
    for raw in value.split(";"):
        item = raw.strip()
        if not item:
            continue
        if ":" not in item:
            raise argparse.ArgumentTypeError(
                "expected TP specs like '1:6;4:1,3,4,5'"
            )
        tp_text, gpu_text = item.split(":", 1)
        try:
            tp_size = int(tp_text)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"invalid TP size {tp_text!r}") from exc
        gpu_ids = _parse_int_list(gpu_text)
        if tp_size <= 0:
            raise argparse.ArgumentTypeError("TP size must be positive")
        if len(gpu_ids) != tp_size:
            raise argparse.ArgumentTypeError(
                f"tp{tp_size} requires {tp_size} GPU ids, got {gpu_ids}"
            )
        specs.append((tp_size, gpu_ids))
    if not specs:
        raise argparse.ArgumentTypeError("expected at least one TP spec")
    return specs


def _tensor_bytes(value: Any) -> int:
    try:
        import torch

        if torch.is_tensor(value):
            return int(value.numel() * value.element_size())
    except Exception:  # noqa: BLE001
        pass
    return 0


def _summarize_encoder_inputs(encoder_inputs: dict[str, Any]) -> dict[str, Any]:
    try:
        import torch
    except Exception:  # noqa: BLE001
        torch = None  # type: ignore[assignment]

    summary: dict[str, Any] = {
        "input_tensor_bytes": 0,
        "tensors": {},
        "video_grid_thw": None,
        "actual_grid_rows": None,
        "visual_tokens": None,
    }
    for key, value in encoder_inputs.items():
        if torch is not None and torch.is_tensor(value):
            bytes_ = _tensor_bytes(value)
            summary["input_tensor_bytes"] += bytes_
            summary["tensors"][key] = {
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "bytes": bytes_,
            }
    grid = encoder_inputs.get("video_grid_thw")
    if torch is not None and torch.is_tensor(grid):
        grid_long = grid.to(dtype=torch.long)
        summary["video_grid_thw"] = grid_long.tolist()
        summary["actual_grid_rows"] = int(grid_long.shape[0])
        summary["visual_tokens_pre_merge"] = int(grid_long.prod(dim=-1).sum().item())
    return summary


def _preprocess_video(
    *,
    model_path: str,
    video_path: str,
    frame_cap: int,
    video_fps: float | None,
    prompt: str,
    max_tokens: int,
    artifact_dir: Path,
) -> tuple[Path, dict[str, Any]]:
    from sglang_omni.models.qwen3_omni.components.preprocessor import (
        Qwen3OmniPreprocessor,
    )
    from sglang_omni.proto import OmniRequest, StagePayload

    artifact_dir.mkdir(parents=True, exist_ok=True)
    input_path = artifact_dir / f"preprocessed_video_{frame_cap}.pkl"
    summary_path = artifact_dir / f"preprocessed_video_{frame_cap}.json"
    if input_path.exists() and summary_path.exists():
        return input_path, json.loads(summary_path.read_text(encoding="utf-8"))
    inputs: dict[str, Any] = {
        "messages": [{"role": "user", "content": prompt}],
        "videos": [video_path],
        "video_max_frames": int(frame_cap),
    }
    if video_fps is not None:
        inputs["video_fps"] = float(video_fps)
    payload = StagePayload(
        request_id=f"video-{frame_cap}",
        request=OmniRequest(inputs=inputs, params={"max_new_tokens": max_tokens}),
        data={},
    )
    preprocessor = Qwen3OmniPreprocessor(
        model_path,
        video_fps=video_fps,
        video_max_frames=frame_cap,
    )
    started = time.perf_counter()
    payload = asyncio.run(preprocessor(payload))
    elapsed_s = time.perf_counter() - started
    encoder_inputs = payload.data["encoder_inputs"]["image_encoder"]
    with input_path.open("wb") as f:
        pickle.dump(
            {
                "payload_data": payload.data,
                "encoder_inputs": encoder_inputs,
                "request_inputs": inputs,
            },
            f,
        )
    summary = _summarize_encoder_inputs(encoder_inputs)
    summary.update({
        "frame_cap": int(frame_cap),
        "video": video_path,
        "video_fps": video_fps,
        "video_max_pixels_override": None,
        "preprocess_elapsed_s": elapsed_s,
        "input_path": str(input_path),
    })
    _write_json(artifact_dir / f"preprocessed_video_{frame_cap}.json", summary)
    return input_path, summary


def _result_paths(case_dir: Path, tp_size: int) -> list[Path]:
    return [case_dir / f"rank{rank}_result.json" for rank in range(tp_size)]


def _terminate_processes(processes: list[subprocess.Popen[str]]) -> None:
    for proc in processes:
        if proc.poll() is None:
            try:
                proc.terminate()
            except ProcessLookupError:
                pass
    deadline = time.time() + 10.0
    for proc in processes:
        while proc.poll() is None and time.time() < deadline:
            time.sleep(0.1)
    for proc in processes:
        if proc.poll() is None:
            try:
                proc.kill()
            except ProcessLookupError:
                pass


def _classify_failure(
    *,
    rank_results: list[dict[str, Any]],
    logs: str,
    timed_out: bool,
    returncodes: list[int | None],
) -> str:
    if any(str(item.get("failure_type")) == "admission_reject" for item in rank_results):
        return "admission_reject"
    text = logs.lower()
    if (
        any(str(item.get("failure_type")) == "encoder_oom" for item in rank_results)
        or "cuda out of memory" in text
        or "outofmemoryerror" in text
    ):
        return "encoder_oom"
    if timed_out:
        return "timeout_or_peer_hang"
    if any((rc is not None and rc < 0) for rc in returncodes):
        return "killed_or_signal"
    return "other_crash"


def _load_rank_results(paths: list[Path]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for path in paths:
        if not path.exists():
            continue
        try:
            out.append(json.loads(path.read_text(encoding="utf-8")))
        except Exception as exc:  # noqa: BLE001
            out.append({"path": str(path), "parse_error": repr(exc)})
    return out


def _run_case(
    *,
    args: argparse.Namespace,
    tp_size: int,
    frame_cap: int,
    gpu_ids: list[int],
    input_path: Path,
    preprocess_summary: dict[str, Any],
    artifact_dir: Path,
) -> dict[str, Any]:
    if len(gpu_ids) != tp_size:
        raise ValueError(f"tp{tp_size} requires {tp_size} GPU ids, got {gpu_ids}")

    case_dir = artifact_dir / f"tp{tp_size}_frames{frame_cap}"
    case_dir.mkdir(parents=True, exist_ok=True)
    before = _query_gpu_sample()
    _write_json(case_dir / "nvidia_smi_before.json", before)

    nccl_port = _pick_free_port()
    processes: list[subprocess.Popen[str]] = []
    rank_log_paths: list[Path] = []
    rank_result_paths = _result_paths(case_dir, tp_size)
    started = time.perf_counter()
    common_env = os.environ.copy()
    common_env.update({
        "PYTHONPATH": str(REPO_ROOT) + os.pathsep + common_env.get("PYTHONPATH", ""),
        "SGLANG_OMNI_ENCODER_TIMING_DETAIL": "1",
        "SGLANG_OMNI_ENCODER_MEMORY_DETAIL": "1",
        "SGLANG_OMNI_ENCODER_GPU_GUARD": "0",
        "SGLANG_ONE_VISIBLE_DEVICE_PER_PROCESS": "true",
        "SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK": "false",
        "PYTORCH_CUDA_ALLOC_CONF": common_env.get(
            "PYTORCH_CUDA_ALLOC_CONF",
            "expandable_segments:True",
        ),
    })

    for rank, gpu_id in enumerate(gpu_ids):
        log_path = case_dir / f"rank{rank}.log"
        rank_log_paths.append(log_path)
        env = dict(common_env)
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            "_worker",
            "--model-path",
            args.model_path,
            "--input-path",
            str(input_path),
            "--result-path",
            str(rank_result_paths[rank]),
            "--tp-size",
            str(tp_size),
            "--tp-rank",
            str(rank),
            "--physical-gpu-id",
            str(gpu_id),
            "--nccl-port",
            str(nccl_port),
            "--frame-cap",
            str(frame_cap),
            "--dtype",
            args.dtype,
        ]
        if args.tp_parity_mode:
            cmd.extend(["--tp-parity-mode", args.tp_parity_mode])
        with log_path.open("w", encoding="utf-8") as log_f:
            log_f.write(
                "COMMAND " + " ".join(cmd) + "\n"
                f"CUDA_VISIBLE_DEVICES={env['CUDA_VISIBLE_DEVICES']}\n"
            )
            log_f.flush()
        log_f = log_path.open("a", encoding="utf-8")
        proc = subprocess.Popen(
            cmd,
            cwd=str(REPO_ROOT),
            env=env,
            stdout=log_f,
            stderr=subprocess.STDOUT,
            text=True,
            preexec_fn=os.setsid,
        )
        proc._encoder_probe_log_f = log_f  # type: ignore[attr-defined]
        processes.append(proc)

    timed_out = False
    with _GpuSampler(case_dir / "gpu_samples.jsonl", args.gpu_sample_interval) as sampler:
        deadline = time.time() + float(args.timeout)
        while time.time() < deadline:
            if all(proc.poll() is not None for proc in processes):
                break
            time.sleep(0.5)
        if not all(proc.poll() is not None for proc in processes):
            timed_out = True
            for proc in processes:
                if proc.poll() is None:
                    try:
                        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                    except ProcessLookupError:
                        pass
            _terminate_processes(processes)
        elapsed_s = time.perf_counter() - started
        for proc in processes:
            log_f = getattr(proc, "_encoder_probe_log_f", None)
            if log_f is not None:
                log_f.close()

        pids = [int(proc.pid) for proc in processes]
        sample_summary = _summarize_samples(
            sampler.samples,
            gpu_indices=gpu_ids,
            pids=pids,
        )

    after = _query_gpu_sample()
    _write_json(case_dir / "nvidia_smi_after.json", after)
    rank_results = _load_rank_results(rank_result_paths)
    logs = ""
    for path in rank_log_paths:
        if path.exists():
            logs += "\n" + path.read_text(encoding="utf-8", errors="replace")
    returncodes = [proc.poll() for proc in processes]
    success = (
        not timed_out
        and len(rank_results) == tp_size
        and all(bool(item.get("success")) for item in rank_results)
        and all(rc == 0 for rc in returncodes)
    )
    failure_type = None if success else _classify_failure(
        rank_results=rank_results,
        logs=logs,
        timed_out=timed_out,
        returncodes=returncodes,
    )
    summary = {
        "tp_size": tp_size,
        "frame_cap": int(frame_cap),
        "gpu_ids": gpu_ids,
        "forced_encoder_max_batch_size": 1,
        "admission_budget": None,
        "gpu_guard": "measurement_only_allow_all",
        "success": success,
        "failure_type": failure_type,
        "elapsed_s": elapsed_s,
        "timed_out": timed_out,
        "returncodes": returncodes,
        "pids": [int(proc.pid) for proc in processes],
        "nccl_port": nccl_port,
        "case_dir": str(case_dir),
        "preprocess": preprocess_summary,
        "rank_results": rank_results,
        "sample_summary": sample_summary,
        "log_paths": [str(path) for path in rank_log_paths],
        "result_paths": [str(path) for path in rank_result_paths],
    }
    _write_json(case_dir / "summary.json", summary)
    return summary


def _peak_from_result_or_samples(
    rank_result: dict[str, Any],
    process_summary: dict[str, Any],
) -> int | None:
    pid = str(rank_result.get("pid", ""))
    candidates: list[int] = []
    final_bytes = rank_result.get("final_nvml_process_bytes")
    if isinstance(final_bytes, int):
        candidates.append(math.ceil(final_bytes / 1024**2))
    max_mark = rank_result.get("max_mark_nvml_process_bytes")
    if isinstance(max_mark, int):
        candidates.append(math.ceil(max_mark / 1024**2))
    proc = process_summary.get("processes", {}).get(pid)
    if isinstance(proc, dict) and isinstance(proc.get("peak_used_mib"), int):
        candidates.append(int(proc["peak_used_mib"]))
    if not candidates:
        return None
    return max(candidates)


def _summarize_memory(case: dict[str, Any]) -> dict[str, Any]:
    rank_rows: list[dict[str, Any]] = []
    for rank_result in case.get("rank_results", []):
        peak_mib = _peak_from_result_or_samples(
            rank_result,
            case.get("sample_summary", {}),
        )
        baseline_bytes = rank_result.get("after_load_nvml_process_bytes")
        baseline_mib = (
            math.ceil(int(baseline_bytes) / 1024**2)
            if isinstance(baseline_bytes, int)
            else None
        )
        delta_mib = (
            peak_mib - baseline_mib
            if peak_mib is not None and baseline_mib is not None
            else None
        )
        rank_rows.append({
            "rank": rank_result.get("rank"),
            "pid": rank_result.get("pid"),
            "physical_gpu_id": rank_result.get("physical_gpu_id"),
            "peak_mib": peak_mib,
            "after_load_mib": baseline_mib,
            "delta_mib": delta_mib,
            "torch_max_allocated_mib": _bytes_to_mib(
                rank_result.get("torch_max_memory_allocated_bytes")
            ),
            "torch_max_reserved_mib": _bytes_to_mib(
                rank_result.get("torch_max_memory_reserved_bytes")
            ),
            "output": rank_result.get("output_summary"),
        })
    peaks = [row["peak_mib"] for row in rank_rows if isinstance(row["peak_mib"], int)]
    return {
        "rank_rows": rank_rows,
        "max_rank_peak_mib": max(peaks) if peaks else None,
        "sum_rank_peak_mib": sum(peaks) if peaks else None,
    }


def _bytes_to_mib(value: Any) -> int | None:
    if not isinstance(value, int):
        return None
    return math.ceil(int(value) / 1024**2)


def _fit_line(points: list[tuple[float, float]]) -> dict[str, Any] | None:
    if len(points) < 2:
        return None
    n = float(len(points))
    sx = sum(x for x, _ in points)
    sy = sum(y for _, y in points)
    sxx = sum(x * x for x, _ in points)
    sxy = sum(x * y for x, y in points)
    denom = n * sxx - sx * sx
    if denom == 0:
        return None
    b = (n * sxy - sx * sy) / denom
    a = (sy - b * sx) / n
    fitted = [a + b * x for x, _ in points]
    ss_res = sum((y - yhat) ** 2 for (_, y), yhat in zip(points, fitted))
    mean_y = sy / n
    ss_tot = sum((y - mean_y) ** 2 for _, y in points)
    r2 = None if ss_tot == 0 else 1.0 - ss_res / ss_tot
    return {"A_mib": a, "b_mib_per_frame": b, "r2": r2, "n": len(points)}


def _build_final_summary(cases: list[dict[str, Any]]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for case in cases:
        mem = _summarize_memory(case)
        row = {
            "tp_size": case["tp_size"],
            "frame_cap": case["frame_cap"],
            "success": case["success"],
            "failure_type": case["failure_type"],
            "elapsed_s": case["elapsed_s"],
            "case_dir": case["case_dir"],
            "actual_grid_rows": case["preprocess"].get("actual_grid_rows"),
            "visual_tokens_pre_merge": case["preprocess"].get(
                "visual_tokens_pre_merge"
            ),
            **mem,
        }
        rows.append(row)

    by_frame: dict[int, dict[int, dict[str, Any]]] = {}
    for row in rows:
        by_frame.setdefault(int(row["frame_cap"]), {})[int(row["tp_size"])] = row

    tp_sizes = sorted({int(row["tp_size"]) for row in rows})
    fits_by_tp: dict[str, dict[str, Any]] = {}
    comparisons_to_tp1: dict[str, dict[str, Any]] = {}
    for tp_size in tp_sizes:
        max_points: list[tuple[float, float]] = []
        sum_points: list[tuple[float, float]] = []
        token_max_points: list[tuple[float, float]] = []
        for row in sorted(rows, key=lambda item: int(item["frame_cap"])):
            if int(row["tp_size"]) != tp_size or not row.get("success"):
                continue
            max_peak = row.get("max_rank_peak_mib")
            sum_peak = row.get("sum_rank_peak_mib")
            tokens = row.get("visual_tokens_pre_merge")
            if isinstance(max_peak, int):
                max_points.append((float(row["frame_cap"]), float(max_peak)))
                if isinstance(tokens, int):
                    token_max_points.append((float(tokens) / 1000.0, float(max_peak)))
            if isinstance(sum_peak, int):
                sum_points.append((float(row["frame_cap"]), float(sum_peak)))
        fits_by_tp[str(tp_size)] = {
            "max_rank_peak_by_frame": _fit_line(max_points),
            "sum_rank_peak_by_frame_reference_only": _fit_line(sum_points),
            "max_rank_peak_by_1k_premerge_tokens": _fit_line(token_max_points),
            "fit_frames": [int(x) for x, _ in max_points],
        }

    tp1_fit = fits_by_tp.get("1", {}).get("max_rank_peak_by_frame")
    if tp1_fit:
        a1 = float(tp1_fit["A_mib"])
        b1 = float(tp1_fit["b_mib_per_frame"])
        for tp_size in tp_sizes:
            if tp_size == 1:
                continue
            fit = fits_by_tp.get(str(tp_size), {}).get("max_rank_peak_by_frame")
            if not fit:
                comparisons_to_tp1[str(tp_size)] = {
                    "slope_conclusion": "insufficient_success_points",
                }
                continue
            a = float(fit["A_mib"])
            b = float(fit["b_mib_per_frame"])
            comparison = {
                "tp1_b_mib_per_frame": b1,
                f"tp{tp_size}_b_mib_per_frame": b,
                "slope_conclusion": (
                    f"b{tp_size}_lt_b1" if b < b1 else f"b{tp_size}_ge_b1"
                ),
                "crossover_frame_cap": None,
            }
            if b < b1:
                comparison["crossover_frame_cap"] = (a - a1) / (b1 - b)
            comparisons_to_tp1[str(tp_size)] = comparison

    success_by_tp: dict[str, list[int]] = {}
    for tp_size in tp_sizes:
        caps = [
            int(row["frame_cap"])
            for row in rows
            if int(row["tp_size"]) == tp_size and row["success"]
        ]
        success_by_tp[str(tp_size)] = sorted(caps)

    fit_tp1 = fits_by_tp.get("1", {}).get("max_rank_peak_by_frame")
    fit_tp2_max = fits_by_tp.get("2", {}).get("max_rank_peak_by_frame")
    fit_tp2_sum = fits_by_tp.get("2", {}).get("sum_rank_peak_by_frame_reference_only")
    tp2_comparison = comparisons_to_tp1.get("2", {})
    return {
        "rows": rows,
        "fit_frames": sorted({
            int(frame)
            for row in rows
            if row.get("success")
            for frame in [row["frame_cap"]]
        }),
        "fits_by_tp": fits_by_tp,
        "comparisons_to_tp1": comparisons_to_tp1,
        "fit": {
            "tp1_peak": fit_tp1,
            "tp2_max_rank_peak": fit_tp2_max,
            "tp2_sum_peak_reference_only": fit_tp2_sum,
            "crossover_frame_cap": tp2_comparison.get("crossover_frame_cap"),
            "slope_conclusion": tp2_comparison.get(
                "slope_conclusion",
                "insufficient_success_points",
            ),
        },
        "capacity": {
            "tp1_max_success_frame_cap": (
                max(success_by_tp.get("1", [])) if success_by_tp.get("1") else None
            ),
            "tp2_max_success_frame_cap": (
                max(success_by_tp.get("2", [])) if success_by_tp.get("2") else None
            ),
            "max_success_frame_cap_by_tp": {
                tp: max(caps) if caps else None for tp, caps in success_by_tp.items()
            },
            "success_frame_caps_by_tp": success_by_tp,
        },
    }


def _format_mark_name(line: str) -> str | None:
    match = re.search(r"encoder_memory_mark .* mark=([^ ]+)", line)
    return match.group(1) if match else None


def _parse_memory_marks_from_log(log_path: Path) -> list[dict[str, Any]]:
    if not log_path.exists():
        return []
    marks: list[dict[str, Any]] = []
    pattern = re.compile(r"([a-zA-Z_]+)=([^ ]+)")
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if "encoder_memory_mark" not in line:
            continue
        item: dict[str, Any] = {"raw": line}
        for key, value in pattern.findall(line):
            if key in {"stage", "mark"}:
                item[key] = value
            elif value == "None":
                item[key] = None
            else:
                try:
                    item[key] = int(value)
                except ValueError:
                    item[key] = value
        if "mark" not in item:
            item["mark"] = _format_mark_name(line)
        marks.append(item)
    return marks


def _parse_timing_from_log(log_path: Path) -> list[dict[str, Any]]:
    if not log_path.exists():
        return []
    timings: list[dict[str, Any]] = []
    pattern = re.compile(r"([a-zA-Z_]+)=([^ ]+)")
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if "encoder_batch_timing" not in line:
            continue
        item: dict[str, Any] = {"raw": line}
        for key, value in pattern.findall(line):
            if key == "stage":
                item[key] = value
            else:
                try:
                    if "." in value:
                        item[key] = float(value)
                    else:
                        item[key] = int(value)
                except ValueError:
                    item[key] = value
        timings.append(item)
    return timings


class _MeasurementOnlyGuard:
    def __init__(self, decision_cls: Any) -> None:
        self._decision_cls = decision_cls

    def check(self, batch_cost_bytes: int) -> Any:
        return self._decision_cls(
            allowed=True,
            reason="measurement_only_allow_all",
            batch_cost_bytes=max(int(batch_cost_bytes), 0),
        )

    def reserve(self, batch_cost_bytes: int) -> tuple[Any, None]:
        return self.check(batch_cost_bytes), None

    def release(self, reservation: Any) -> None:
        del reservation


def _move_tensors_to_cpu(value: Any) -> Any:
    import torch

    if torch.is_tensor(value):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: _move_tensors_to_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_move_tensors_to_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_move_tensors_to_cpu(item) for item in value)
    return value


def _output_summary(value: Any) -> Any:
    import torch

    if torch.is_tensor(value):
        return {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "device": str(value.device),
            "numel": int(value.numel()),
        }
    if isinstance(value, dict):
        return {key: _output_summary(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_output_summary(item) for item in value]
    return value


def _worker_run(args: argparse.Namespace) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s:%(lineno)d %(message)s",
    )
    import torch
    import torch.distributed as dist

    from sglang_omni.model_runner.sglang_encoder_runner import SGLangEncoderRunner
    from sglang_omni.models.qwen3_omni.encoder_adapters import (
        Qwen3OmniImageEncoderAdapter,
    )
    from sglang_omni.proto import StagePayload
    from sglang_omni.scheduling.encoder_scheduler import (
        EncoderScheduler,
        _GpuGuardDecision,
    )
    from sglang_omni.scheduling.messages import IncomingMessage
    from sglang_omni.utils.gpu_memory import get_process_gpu_memory_bytes

    result_path = Path(args.result_path)
    started = time.perf_counter()
    result: dict[str, Any] = {
        "success": False,
        "pid": os.getpid(),
        "rank": int(args.tp_rank),
        "tp_size": int(args.tp_size),
        "physical_gpu_id": int(args.physical_gpu_id),
        "frame_cap": int(args.frame_cap),
        "input_path": args.input_path,
    }

    def finish(extra: dict[str, Any], *, code: int) -> None:
        result.update(extra)
        result["elapsed_s"] = time.perf_counter() - started
        try:
            result["final_nvml_process_bytes"] = get_process_gpu_memory_bytes(0)
            result["torch_memory_allocated_bytes"] = int(torch.cuda.memory_allocated(0))
            result["torch_memory_reserved_bytes"] = int(torch.cuda.memory_reserved(0))
            result["torch_max_memory_allocated_bytes"] = int(
                torch.cuda.max_memory_allocated(0)
            )
            result["torch_max_memory_reserved_bytes"] = int(
                torch.cuda.max_memory_reserved(0)
            )
            free_bytes, total_bytes = torch.cuda.mem_get_info(0)
            result["torch_mem_get_info"] = {
                "free_bytes": int(free_bytes),
                "total_bytes": int(total_bytes),
            }
        except Exception as exc:  # noqa: BLE001
            result["final_memory_error"] = repr(exc)
        _write_json(result_path, result)
        raise SystemExit(code)

    try:
        with open(args.input_path, "rb") as f:
            input_bundle = pickle.load(f)
        payload_data = input_bundle["payload_data"]

        runner = SGLangEncoderRunner(
            model_path=args.model_path,
            gpu_id=int(args.physical_gpu_id),
            tp_rank=int(args.tp_rank),
            tp_size=int(args.tp_size),
            nccl_port=int(args.nccl_port),
            encoder_specs=Qwen3OmniImageEncoderAdapter.encoder_specs,
            dtype=args.dtype,
            tp_parity_mode=args.tp_parity_mode,
        )
        torch.cuda.synchronize(0)
        result["after_load_nvml_process_bytes"] = get_process_gpu_memory_bytes(0)
        result["after_load_allocated_bytes"] = int(torch.cuda.memory_allocated(0))
        result["after_load_reserved_bytes"] = int(torch.cuda.memory_reserved(0))

        adapter = Qwen3OmniImageEncoderAdapter(
            hf_config=runner.model_config.hf_config,
            dtype={"float16": torch.float16, "bfloat16": torch.bfloat16}[args.dtype],
            tp_size=int(args.tp_size),
        )
        scheduler = EncoderScheduler(
            runner,
            adapter,
            max_batch_size=1,
            max_batch_wait_ms=0,
            request_cost_fn=adapter.request_cost_fn,
            batch_cost_fn=getattr(adapter, "batch_cost_fn", None),
            max_batch_cost=None,
            max_single_request_cost=None,
            gpu_memory_guard=_MeasurementOnlyGuard(_GpuGuardDecision),
        )
        if runner.is_entry_rank:
            payload = StagePayload(
                request_id=f"video-{args.frame_cap}",
                request=input_bundle.get("request"),
                data=payload_data,
            )
            scheduler.inbox.put(
                IncomingMessage(
                    request_id=payload.request_id,
                    type="new_request",
                    data=payload,
                )
            )

        iter_started = time.perf_counter()
        scheduler._reset_memory_peak_stats()
        recv_started = time.perf_counter()
        messages, recv_err = scheduler._recv_messages()
        recv_ms = (time.perf_counter() - recv_started) * 1000.0
        if scheduler._gather_pre_forward_error(recv_err):
            scheduler._release_active_reservation()
            finish(
                {
                    "success": False,
                    "failure_type": "admission_reject",
                    "error": repr(recv_err),
                    "message_count": len(messages),
                },
                code=2,
            )
        if not messages:
            scheduler._release_active_reservation()
            finish(
                {
                    "success": False,
                    "failure_type": "no_messages",
                    "error": "scheduler received no messages",
                },
                code=3,
            )

        build_err: BaseException | None = None
        build_started = time.perf_counter()
        plan = None
        try:
            plan = adapter.build_batch(messages)
        except Exception as exc:  # noqa: BLE001
            build_err = exc
        build_ms = (time.perf_counter() - build_started) * 1000.0
        if scheduler._gather_pre_forward_error(build_err):
            scheduler._release_active_reservation()
            finish(
                {
                    "success": False,
                    "failure_type": "build_error",
                    "error": repr(build_err),
                    "message_count": len(messages),
                },
                code=4,
            )
        assert plan is not None

        scheduler._log_memory_mark("after_build_before_forward", batch_size=1)
        forward_started = time.perf_counter()
        raw = runner.encode_batch(plan)
        torch.cuda.synchronize(0)
        forward_ms = (time.perf_counter() - forward_started) * 1000.0
        scheduler._log_memory_mark("after_forward", batch_size=1)

        output = None
        slice_ms = 0.0
        if runner.is_entry_rank:
            slice_started = time.perf_counter()
            sliced = adapter.slice_results(raw, plan, messages)
            torch.cuda.synchronize(0)
            slice_ms = (time.perf_counter() - slice_started) * 1000.0
            scheduler._log_memory_mark("after_slice_output_staging", batch_size=1)
            output = sliced[0].data["encoder_outs"]["image_encoder"]

        logging.info(
            "encoder_batch_timing stage=%s tp_rank=%d/%d batch_size=1 "
            "recv_ms=%.3f inbox_admission_ms=%.3f strip_h2d_ms=%.3f "
            "metadata_broadcast_ms=%.3f follower_allocation_ms=%.3f "
            "allocation_handshake_ms=%.3f tensor_broadcast_ms=%.3f "
            "rank_wait_skew_ms=%.3f rank_arrival_skew_ms=%.3f "
            "build_ms=%.3f forward_ms=%.3f slice_ms=%.3f total_ms=%.3f",
            scheduler._stage_name,
            runner.tp_rank,
            runner.tp_size,
            recv_ms,
            scheduler._last_recv_timing.inbox_admission_ms,
            scheduler._last_recv_timing.strip_h2d_ms,
            scheduler._last_recv_timing.metadata_broadcast_ms,
            scheduler._last_recv_timing.follower_allocation_ms,
            scheduler._last_recv_timing.allocation_handshake_ms,
            scheduler._last_recv_timing.tensor_broadcast_ms,
            scheduler._last_recv_timing.rank_wait_skew_ms,
            scheduler._last_recv_timing.rank_arrival_skew_ms,
            build_ms,
            forward_ms,
            slice_ms,
            (time.perf_counter() - iter_started) * 1000.0,
        )
        scheduler._release_active_reservation()
        scheduler._log_memory_mark(
            "after_cleanup_synchronize",
            batch_size=1,
            synchronize=True,
        )
        log_path = Path(args.result_path).with_name(f"rank{args.tp_rank}.log")
        marks = _parse_memory_marks_from_log(log_path)
        timings = _parse_timing_from_log(log_path)
        max_mark_nvml = max(
            (
                int(mark["nvml_process_bytes"])
                for mark in marks
                if isinstance(mark.get("nvml_process_bytes"), int)
            ),
            default=None,
        )
        finish(
            {
                "success": True,
                "message_count": len(messages),
                "output_summary": _output_summary(output) if output is not None else None,
                "timing": timings,
                "memory_marks": marks,
                "max_mark_nvml_process_bytes": max_mark_nvml,
            },
            code=0,
        )
    except RuntimeError as exc:
        text = repr(exc)
        failure_type = (
            "encoder_oom"
            if "out of memory" in text.lower() or "OutOfMemoryError" in text
            else "runtime_error"
        )
        finish({"success": False, "failure_type": failure_type, "error": text}, code=10)
    except Exception as exc:  # noqa: BLE001
        finish({"success": False, "failure_type": "other_crash", "error": repr(exc)}, code=11)
    finally:
        try:
            if dist.is_available() and dist.is_initialized():
                dist.destroy_process_group()
        except Exception:  # noqa: BLE001
            pass


def _worker_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("_worker")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--input-path", required=True)
    parser.add_argument("--result-path", required=True)
    parser.add_argument("--tp-size", type=int, required=True)
    parser.add_argument("--tp-rank", type=int, required=True)
    parser.add_argument("--physical-gpu-id", type=int, required=True)
    parser.add_argument("--nccl-port", type=int, required=True)
    parser.add_argument("--frame-cap", type=int, required=True)
    parser.add_argument("--dtype", choices=["float16", "bfloat16"], default="float16")
    parser.add_argument("--tp-parity-mode", default="default")
    return parser


def _main_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model-path", default="/data/qwen3omni")
    parser.add_argument("--video", default=DEFAULT_VIDEO)
    parser.add_argument(
        "--output-dir",
        default="/data/encoder_tp_evidence_20260526/"
        "h100_encoder_only_video_memory_20260529",
    )
    parser.add_argument("--frame-caps", type=_parse_int_list, default=[128, 256, 512, 768, 1024, 1536, 2048])
    parser.add_argument("--tp1-gpu", type=int, default=6)
    parser.add_argument("--tp2-gpus", type=_parse_int_list, default=[6, 7])
    parser.add_argument(
        "--tp-specs",
        type=_parse_tp_specs,
        default=None,
        help=(
            "Optional semicolon-separated TP/GPU specs, e.g. "
            "'1:6;4:1,3,4,5'. When omitted, preserves the legacy "
            "--tp1-gpu/--tp2-gpus behavior."
        ),
    )
    parser.add_argument("--video-fps", type=float, default=30.0)
    parser.add_argument("--prompt", default="Briefly describe the video.")
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--dtype", choices=["float16", "bfloat16"], default="float16")
    parser.add_argument("--tp-parity-mode", default="default")
    parser.add_argument("--timeout", type=float, default=1200.0)
    parser.add_argument("--gpu-sample-interval", type=float, default=0.25)
    parser.add_argument(
        "--stop-after-first-failure",
        action="store_true",
        help="Stop testing larger caps for a TP size after its first failure.",
    )
    return parser


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "_worker":
        _worker_run(_worker_parser().parse_args())
        return

    args = _main_parser().parse_args()
    artifact_dir = Path(args.output_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    _write_json(artifact_dir / "command.json", {
        "argv": sys.argv,
        "cwd": os.getcwd(),
        "env": {
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "PYTORCH_CUDA_ALLOC_CONF": os.environ.get("PYTORCH_CUDA_ALLOC_CONF"),
        },
    })
    _write_json(artifact_dir / "git_status.json", {
        "status": _run_cmd(["git", "status", "--short"], timeout=30),
        "head": _run_cmd(["git", "rev-parse", "HEAD"], timeout=30),
        "diff_stat": _run_cmd(["git", "diff", "--stat"], timeout=30),
    })
    _write_json(artifact_dir / "nvidia_smi_initial.json", _query_gpu_sample())

    tp_specs = (
        args.tp_specs
        if args.tp_specs is not None
        else [
            (1, [int(args.tp1_gpu)]),
            (2, [int(gpu) for gpu in args.tp2_gpus]),
        ]
    )
    cases: list[dict[str, Any]] = []
    failed_tp: set[int] = set()
    for frame_cap in args.frame_caps:
        cap_dir = artifact_dir / "preprocess"
        try:
            input_path, preprocess_summary = _preprocess_video(
                model_path=args.model_path,
                video_path=args.video,
                frame_cap=int(frame_cap),
                video_fps=args.video_fps,
                prompt=args.prompt,
                max_tokens=args.max_tokens,
                artifact_dir=cap_dir,
            )
        except Exception as exc:  # noqa: BLE001
            _write_json(
                artifact_dir / f"preprocess_failed_{frame_cap}.json",
                {"frame_cap": int(frame_cap), "error": repr(exc)},
            )
            break
        for tp_size, gpu_ids in tp_specs:
            if args.stop_after_first_failure and tp_size in failed_tp:
                continue
            case = _run_case(
                args=args,
                tp_size=tp_size,
                frame_cap=int(frame_cap),
                gpu_ids=gpu_ids,
                input_path=input_path,
                preprocess_summary=preprocess_summary,
                artifact_dir=artifact_dir,
            )
            cases.append(case)
            with (artifact_dir / "results.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps(case, sort_keys=True, default=_json_default) + "\n")
            if not case.get("success"):
                failed_tp.add(tp_size)

    final = _build_final_summary(cases)
    final.update({
        "artifact_dir": str(artifact_dir),
        "model_path": args.model_path,
        "video": args.video,
        "video_fps": args.video_fps,
        "frame_caps_requested": [int(x) for x in args.frame_caps],
        "tp1_gpu": int(args.tp1_gpu),
        "tp2_gpus": [int(x) for x in args.tp2_gpus],
        "tp_specs": [
            {"tp_size": int(tp_size), "gpu_ids": [int(gpu) for gpu in gpu_ids]}
            for tp_size, gpu_ids in tp_specs
        ],
        "notes": [
            "encoder-only image/video forward; no thinker/talker/generation",
            "fresh subprocesses per TP/frame point",
            "no video_max_pixels override set by this probe",
            "activation budget disabled; measurement-only GPU guard allowed all candidates",
            "encoder_max_batch_size forced to 1",
        ],
    })
    _write_json(artifact_dir / "summary.json", final)
    print(json.dumps(final["fit"], indent=2, sort_keys=True))
    print(json.dumps(final["comparisons_to_tp1"], indent=2, sort_keys=True))
    print(json.dumps(final["capacity"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
