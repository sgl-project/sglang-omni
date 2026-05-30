# SPDX-License-Identifier: Apache-2.0
"""Colocated video memory-slope probe for Qwen3-Omni encoder TP.

This launches the normal OpenAI-compatible pipeline with
``examples/qwen3_omni_encoder_tp.py --layout colocated-2gpu`` and sends one
video request per fresh server. It is intentionally small and evidence-focused:

* TP1 uses image/audio TP1 on two visible GPUs.
* TP2 uses image/audio TP2 on the same two visible GPUs.
* Each frame cap starts a fresh server so allocator cache from prior requests
  does not become the fitted slope.
* The fit reports both image-encoder process NVML peaks from
  ``encoder_memory_mark`` and sampled whole-GPU peaks.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VIDEO = (
    "/data/encoder_tp_evidence_20260526/media/draw_loop_2048frames.mp4"
)
DEFAULT_OUT = (
    "/data/encoder_tp_evidence_20260526/"
    "h100_colocated_video_memory_slope_20260529"
)


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _run_cmd(cmd: list[str], *, timeout: float = 60.0) -> dict[str, Any]:
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(REPO_ROOT),
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


def _parse_int_list(value: str) -> list[int]:
    items = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not items:
        raise argparse.ArgumentTypeError("expected comma-separated integers")
    return items


def _health_ok(base_url: str, *, timeout: float = 5.0) -> bool:
    try:
        with urllib.request.urlopen(base_url.rstrip("/") + "/health", timeout=timeout) as resp:
            return 200 <= resp.status < 300
    except Exception:  # noqa: BLE001
        return False


def _wait_health(base_url: str, *, timeout_s: float) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if _health_ok(base_url):
            return True
        time.sleep(2.0)
    return False


def _terminate(proc: subprocess.Popen[str]) -> None:
    if proc.poll() is not None:
        return
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.time() + 20.0
    while proc.poll() is None and time.time() < deadline:
        time.sleep(0.5)
    if proc.poll() is None:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except ProcessLookupError:
            pass


_MEMORY_MARK_RE = re.compile(r"encoder_memory_mark (?P<body>.*)")
_KEY_VALUE_RE = re.compile(r"([a-zA-Z_]+)=([^ ]+)")
_TIMING_RE = re.compile(r"encoder_batch_timing (?P<body>.*)")
_ADMISSION_RE = re.compile(r"encoder_admission (?P<body>.*)")


def _parse_key_values(body: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in _KEY_VALUE_RE.findall(body):
        if value == "None":
            out[key] = None
            continue
        try:
            if "." in value:
                out[key] = float(value)
            else:
                out[key] = int(value)
        except ValueError:
            out[key] = value
    return out


def _parse_server_log(path: Path) -> dict[str, Any]:
    marks: list[dict[str, Any]] = []
    timings: list[dict[str, Any]] = []
    admissions: list[dict[str, Any]] = []
    if not path.exists():
        return {"memory_marks": marks, "timings": timings, "admissions": admissions}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = _MEMORY_MARK_RE.search(line)
        if match:
            marks.append({"raw": line, **_parse_key_values(match.group("body"))})
            continue
        match = _TIMING_RE.search(line)
        if match:
            timings.append({"raw": line, **_parse_key_values(match.group("body"))})
            continue
        match = _ADMISSION_RE.search(line)
        if match:
            admissions.append({"raw": line, **_parse_key_values(match.group("body"))})
    image_marks = [
        mark for mark in marks if mark.get("stage") == "image_encoder"
    ]
    by_rank: dict[str, dict[str, Any]] = {}
    for mark in image_marks:
        rank = str(mark.get("tp_rank"))
        current = by_rank.setdefault(
            rank,
            {
                "tp_rank": rank,
                "peak_nvml_mib": None,
                "peak_mark": None,
                "peak_max_reserved_mib": None,
            },
        )
        nvml = mark.get("nvml_process_bytes")
        if isinstance(nvml, int):
            mib = (nvml + 1024**2 - 1) // 1024**2
            if current["peak_nvml_mib"] is None or mib > current["peak_nvml_mib"]:
                current["peak_nvml_mib"] = mib
                current["peak_mark"] = mark.get("mark")
        reserved = mark.get("max_reserved_bytes")
        if isinstance(reserved, int):
            mib = (reserved + 1024**2 - 1) // 1024**2
            if (
                current["peak_max_reserved_mib"] is None
                or mib > current["peak_max_reserved_mib"]
            ):
                current["peak_max_reserved_mib"] = mib
    rank_peaks = list(by_rank.values())
    max_rank_peak = max(
        (
            int(item["peak_nvml_mib"])
            for item in rank_peaks
            if item.get("peak_nvml_mib") is not None
        ),
        default=None,
    )
    return {
        "memory_marks": marks,
        "timings": timings,
        "admissions": admissions,
        "image_rank_peaks": rank_peaks,
        "image_max_rank_peak_mib": max_rank_peak,
    }


def _read_probe_summary(case_dir: Path, case_id: str) -> dict[str, Any] | None:
    path = case_dir / case_id / "summary.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _gpu_peak_for_visible(summary: dict[str, Any] | None, visible_gpus: list[int]) -> dict[str, Any]:
    if summary is None:
        return {}
    rows = summary.get("gpu_peak_summary", {}).get("gpus", [])
    out: dict[str, Any] = {}
    for row in rows:
        idx = row.get("index")
        if idx in visible_gpus:
            out[str(idx)] = {
                "max_memory_used_mib": row.get("max_memory_used_mib"),
                "max_memory_delta_mib": row.get("max_memory_delta_mib"),
                "min_memory_free_mib": row.get("min_memory_free_mib"),
            }
    return out


def _run_case(
    args: argparse.Namespace,
    *,
    tp_size: int,
    frame_cap: int,
    port: int,
) -> dict[str, Any]:
    case_id = f"tp{tp_size}_frames{frame_cap}"
    case_dir = Path(args.output_dir) / case_id
    case_dir.mkdir(parents=True, exist_ok=True)
    server_log = case_dir / "server.log"
    visible_gpus = _parse_int_list(args.cuda_visible_devices)
    env = os.environ.copy()
    env.update({
        "CUDA_VISIBLE_DEVICES": args.cuda_visible_devices,
        "SGLANG_OMNI_ENCODER_TIMING_DETAIL": "1",
        "SGLANG_OMNI_ENCODER_MEMORY_DETAIL": "1",
        "SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN": "1",
        "PYTORCH_CUDA_ALLOC_CONF": env.get(
            "PYTORCH_CUDA_ALLOC_CONF",
            "expandable_segments:True",
        ),
        "PYTHONPATH": str(REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", ""),
    })
    server_cmd = [
        sys.executable,
        str(REPO_ROOT / "examples" / "qwen3_omni_encoder_tp.py"),
        "--model",
        args.model_path,
        "--layout",
        "colocated-2gpu",
        "--image-tp",
        str(tp_size),
        "--audio-tp",
        str(tp_size),
        "--image-encoder-activation-budget-gib",
        str(args.image_budget_gib),
        "--audio-encoder-activation-budget-gib",
        str(args.audio_budget_gib),
        "--encoder-total-gpu-memory-fraction",
        str(args.encoder_total_gpu_memory_fraction),
        "--thinker-mem-fraction-static",
        str(args.thinker_mem_fraction_static),
        "--talker-mem-fraction-static",
        str(args.talker_mem_fraction_static),
        "--thinker-max-seq-len",
        str(args.thinker_max_seq_len),
        "--encoder-max-batch-size",
        "1",
        "--port",
        str(port),
    ]
    _write_json(
        case_dir / "case_config.json",
        {
            "case_id": case_id,
            "server_cmd": server_cmd,
            "env": {key: env.get(key) for key in (
                "CUDA_VISIBLE_DEVICES",
                "SGLANG_OMNI_ENCODER_TIMING_DETAIL",
                "SGLANG_OMNI_ENCODER_MEMORY_DETAIL",
                "SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN",
                "PYTORCH_CUDA_ALLOC_CONF",
            )},
        },
    )
    _write_json(case_dir / "nvidia_smi_before.json", _run_cmd(["nvidia-smi"]))
    started = time.perf_counter()
    with server_log.open("w", encoding="utf-8") as f:
        proc = subprocess.Popen(
            server_cmd,
            cwd=str(REPO_ROOT),
            env=env,
            stdout=f,
            stderr=subprocess.STDOUT,
            text=True,
            preexec_fn=os.setsid,
        )
    base_url = f"http://127.0.0.1:{port}"
    server_ready = _wait_health(base_url, timeout_s=args.server_timeout)
    probe_rc: int | None = None
    probe_summary = None
    if server_ready:
        probe_root = case_dir / "probe"
        probe_cmd = [
            sys.executable,
            str(REPO_ROOT / "examples" / "encoder_tp_e2e_probe.py"),
            "--base-url",
            base_url,
            "--case-id",
            case_id,
            "--output-dir",
            str(probe_root),
            "--repo",
            str(REPO_ROOT),
            "--video",
            args.video,
            "--video-fps",
            str(args.video_fps),
            "--video-max-frames",
            str(frame_cap),
            "--prompt",
            args.prompt,
            "--max-tokens",
            str(args.max_tokens),
            "--timeout",
            str(args.request_timeout),
            "--repeat",
            "1",
            "--gpu-sample-interval",
            str(args.gpu_sample_interval),
        ]
        probe = subprocess.run(
            probe_cmd,
            cwd=str(REPO_ROOT),
            env=env,
            text=True,
            capture_output=True,
            check=False,
            timeout=args.request_timeout + 120,
        )
        probe_rc = probe.returncode
        _write_json(
            case_dir / "probe_process.json",
            {
                "cmd": probe_cmd,
                "returncode": probe.returncode,
                "stdout": probe.stdout,
                "stderr": probe.stderr,
            },
        )
        probe_summary = _read_probe_summary(probe_root, case_id)
    _terminate(proc)
    elapsed_s = time.perf_counter() - started
    time.sleep(2.0)
    _write_json(case_dir / "nvidia_smi_after.json", _run_cmd(["nvidia-smi"]))
    log_summary = _parse_server_log(server_log)
    success = bool(
        server_ready
        and probe_summary is not None
        and probe_summary.get("failures") == 0
        and probe_summary.get("health_after_ok")
    )
    result = {
        "case_id": case_id,
        "tp_size": tp_size,
        "frame_cap": frame_cap,
        "port": port,
        "success": success,
        "server_ready": server_ready,
        "server_returncode": proc.poll(),
        "probe_returncode": probe_rc,
        "elapsed_s": elapsed_s,
        "case_dir": str(case_dir),
        "server_log": str(server_log),
        "probe_summary": probe_summary,
        "gpu_peaks_visible": _gpu_peak_for_visible(probe_summary, visible_gpus),
        "image_rank_peaks": log_summary["image_rank_peaks"],
        "image_max_rank_peak_mib": log_summary["image_max_rank_peak_mib"],
        "timing": log_summary["timings"],
        "admissions": log_summary["admissions"],
    }
    _write_json(case_dir / "summary.json", result)
    return result


def _fit(points: list[tuple[float, float]]) -> dict[str, Any] | None:
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
    mean_y = sy / n
    ss_res = sum((y - (a + b * x)) ** 2 for x, y in points)
    ss_tot = sum((y - mean_y) ** 2 for _, y in points)
    return {
        "A_mib": a,
        "b_mib_per_frame": b,
        "r2": None if ss_tot == 0 else 1.0 - ss_res / ss_tot,
        "n": len(points),
    }


def _final_summary(results: list[dict[str, Any]], visible_gpus: list[int]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for result in results:
        visible_peaks = result.get("gpu_peaks_visible", {})
        whole_max = max(
            (
                int(item["max_memory_used_mib"])
                for item in visible_peaks.values()
                if item.get("max_memory_used_mib") is not None
            ),
            default=None,
        )
        rows.append({
            "tp_size": result["tp_size"],
            "frame_cap": result["frame_cap"],
            "success": result["success"],
            "image_max_rank_peak_mib": result.get("image_max_rank_peak_mib"),
            "visible_whole_gpu_max_mib": whole_max,
            "gpu_peaks_visible": visible_peaks,
            "case_dir": result["case_dir"],
            "probe_failures": (
                None
                if result.get("probe_summary") is None
                else result["probe_summary"].get("failures")
            ),
            "prompt_tokens": (
                None
                if result.get("probe_summary") is None
                else result["probe_summary"].get("prompt_tokens_max")
            ),
        })
    fits: dict[str, Any] = {}
    for tp_size in sorted({int(row["tp_size"]) for row in rows}):
        tp_rows = [
            row for row in rows
            if int(row["tp_size"]) == tp_size
            and row["success"]
            and row.get("image_max_rank_peak_mib") is not None
        ]
        fits[f"tp{tp_size}_image_max_rank"] = _fit([
            (float(row["frame_cap"]), float(row["image_max_rank_peak_mib"]))
            for row in sorted(tp_rows, key=lambda item: item["frame_cap"])
        ])
        gpu_rows = [
            row for row in rows
            if int(row["tp_size"]) == tp_size
            and row["success"]
            and row.get("visible_whole_gpu_max_mib") is not None
        ]
        fits[f"tp{tp_size}_visible_whole_gpu_max"] = _fit([
            (float(row["frame_cap"]), float(row["visible_whole_gpu_max_mib"]))
            for row in sorted(gpu_rows, key=lambda item: item["frame_cap"])
        ])
    return {
        "rows": sorted(rows, key=lambda item: (item["tp_size"], item["frame_cap"])),
        "fits": fits,
        "visible_gpus": visible_gpus,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="/data/qwen3omni")
    parser.add_argument("--video", default=DEFAULT_VIDEO)
    parser.add_argument("--output-dir", default=DEFAULT_OUT)
    parser.add_argument("--frame-caps", type=_parse_int_list, default=[64, 96, 128, 160])
    parser.add_argument("--cuda-visible-devices", default="1,3")
    parser.add_argument("--base-port", type=int, default=8320)
    parser.add_argument("--video-fps", type=float, default=30.0)
    parser.add_argument("--prompt", default="Briefly describe the video.")
    parser.add_argument("--max-tokens", type=int, default=8)
    parser.add_argument("--thinker-mem-fraction-static", type=float, default=0.78)
    parser.add_argument("--talker-mem-fraction-static", type=float, default=0.12)
    parser.add_argument("--thinker-max-seq-len", type=int, default=131072)
    parser.add_argument("--encoder-total-gpu-memory-fraction", type=float, default=0.01)
    parser.add_argument("--image-budget-gib", type=float, default=64.0)
    parser.add_argument("--audio-budget-gib", type=float, default=1.0)
    parser.add_argument("--server-timeout", type=float, default=420.0)
    parser.add_argument("--request-timeout", type=float, default=600.0)
    parser.add_argument("--gpu-sample-interval", type=float, default=0.5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    visible_gpus = _parse_int_list(args.cuda_visible_devices)
    _write_json(
        out_dir / "experiment_config.json",
        {
            "argv": sys.argv,
            "args": vars(args),
            "git_head": _run_cmd(["git", "rev-parse", "HEAD"]),
            "git_status": _run_cmd(["git", "status", "--short"]),
            "nvidia_smi_initial": _run_cmd(["nvidia-smi"]),
        },
    )
    results: list[dict[str, Any]] = []
    port = int(args.base_port)
    for frame_cap in args.frame_caps:
        for tp_size in (1, 2):
            result = _run_case(args, tp_size=tp_size, frame_cap=frame_cap, port=port)
            results.append(result)
            with (out_dir / "results.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps(result, sort_keys=True) + "\n")
            port += 1
    final = _final_summary(results, visible_gpus)
    _write_json(out_dir / "summary.json", final)
    print(json.dumps(final, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
