# SPDX-License-Identifier: Apache-2.0
"""Assemble independent Oracle and SGLang Nemotron benchmark artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from benchmarks.nemotron_metrics import (
    NOT_REPORTED,
    aggregate_performance,
    aggregate_quality,
    new_report,
)


def _backend_section(raw: dict[str, Any]) -> dict[str, Any]:
    backend = raw["backend"]
    payload = raw[backend]
    rows = payload["samples"]
    quality = aggregate_quality([row["quality"] for row in rows])
    if quality["language_accuracy"] == NOT_REPORTED:
        quality["language_accuracy_reason"] = (
            "auto-mode outputs contained no detected locale tag"
        )
    performance = aggregate_performance(
        rows,
        wall_time_s=float(payload["wall_time_s"]),
        batch_sizes=[size for row in rows for size in row.get("batch_sizes", [])],
        cache_reuse_count=sum(int(row.get("cache_reuse_count", 0)) for row in rows),
        cache_observations=sum(int(row.get("cache_observations", 0)) for row in rows),
        nvml=payload.get("nvml"),
        pytorch_peak_allocated_bytes=payload.get("pytorch_peak_allocated_bytes"),
        pytorch_peak_reserved_bytes=payload.get("pytorch_peak_reserved_bytes"),
    )
    if backend == "oracle":
        performance["actual_batch_distribution"] = {
            "status": "not applicable",
            "reason": "explicit Oracle is a single-request loop",
        }
    return {"quality": quality, "performance": performance}


def assemble(oracle_path: Path, sglang_path: Path, output_path: Path) -> dict[str, Any]:
    oracle = json.loads(oracle_path.read_text(encoding="utf-8"))
    sglang = json.loads(sglang_path.read_text(encoding="utf-8"))
    report = new_report(
        config={
            **oracle["config"],
            "dataset": oracle["dataset"],
            "oracle_artifact": str(oracle_path),
            "sglang_artifact": str(sglang_path),
        }
    )
    report["oracle"] = _backend_section(oracle)
    report["sglang"] = _backend_section(sglang)
    oracle_rows = oracle["oracle"]["samples"]
    sglang_rows = sglang["sglang"]["samples"]
    report["cross_check"] = {
        "oracle_sglang_clean_text_equal": [r["text"] for r in oracle_rows]
        == [r["text"] for r in sglang_rows],
        "oracle_sglang_raw_text_equal": [r["raw_text"] for r in oracle_rows]
        == [r["raw_text"] for r in sglang_rows],
        "status": "smoke_only",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def write_markdown(report: dict[str, Any], path: Path) -> None:
    fields = [
        ("WER/CER", "quality.wer_cer"),
        ("language accuracy", "quality.language_accuracy"),
        ("req/s", "performance.req_per_s"),
        ("audio-seconds/s", "performance.audio_seconds_per_s"),
        ("RTF", "performance.rtf"),
        ("RTFx", "performance.rtfx"),
        ("E2E P50 (s)", "performance.e2e_p50_s"),
        ("E2E P99 (s)", "performance.e2e_p99_s"),
        ("TTFT P50 (s)", "performance.ttft_p50_s"),
        ("TTFT P99 (s)", "performance.ttft_p99_s"),
        ("chunk P50 (ms)", "performance.chunk_p50_ms"),
        ("chunk P99 (ms)", "performance.chunk_p99_ms"),
        ("finalization P50 (s)", "performance.finalization_latency_p50_s"),
        ("finalization P99 (s)", "performance.finalization_latency_p99_s"),
        ("actual batch distribution", "performance.actual_batch_distribution"),
        ("cache reuse", "performance.cache_reuse"),
        ("NVML peak used (MiB)", "performance.nvml_peak_used_mib"),
        ("PyTorch peak allocated (bytes)", "performance.pytorch_peak_allocated_bytes"),
        ("PyTorch peak reserved (bytes)", "performance.pytorch_peak_reserved_bytes"),
    ]

    def value(section: dict[str, Any], path: str) -> str:
        current: Any = section
        for key in path.split("."):
            current = current.get(key, NOT_REPORTED) if isinstance(current, dict) else NOT_REPORTED
        return json.dumps(current, ensure_ascii=False, sort_keys=True) if isinstance(current, (dict, list)) else str(current)

    lines = [
        "# Nemotron streaming comparison",
        "",
        "Model-card/H100 results are intentionally excluded. This is an RTX 4090 local smoke only.",
        "",
        "| metric | Transformers Oracle | SGLang native streaming |",
        "|---|---:|---:|",
    ]
    for label, path_key in fields:
        lines.append(f"| {label} | {value(report['oracle'], path_key)} | {value(report['sglang'], path_key)} |")
    lines.extend([
        "",
        f"Dataset: `{report['config']['dataset']['id']}` / `{report['config']['dataset']['config']}` / `{report['config']['dataset']['split']}` @ `{report['config']['dataset']['revision']}`.",
        f"Cross-check: `{json.dumps(report['cross_check'], ensure_ascii=False, sort_keys=True)}`.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--oracle", type=Path, required=True)
    parser.add_argument("--sglang", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown", type=Path)
    args = parser.parse_args()
    report = assemble(args.oracle, args.sglang, args.output)
    if args.markdown:
        write_markdown(report, args.markdown)
    print(json.dumps({"output": str(args.output), "cross_check": report["cross_check"]}))


if __name__ == "__main__":
    main()
