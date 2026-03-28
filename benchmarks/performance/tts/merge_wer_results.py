#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Merge multiple WER benchmark result JSON files into a single result.

Used to combine sharded parallel evaluation runs into one unified report.

Usage:
    python merge_wer_results.py \
        --parts part_0/wer_results.json part_1/wer_results.json part_2/wer_results.json \
        --output merged/wer_results.json
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np


def merge_results(part_files: list[str]) -> dict:
    all_samples = []
    configs = []

    for path in part_files:
        with open(path) as f:
            data = json.load(f)
        all_samples.extend(data["per_sample"])
        configs.append(data.get("config", {}))

    # Recompute metrics from merged per-sample data
    successes = [s for s in all_samples if s["is_success"]]
    total = len(all_samples)
    evaluated = len(successes)
    skipped = total - evaluated

    if not successes:
        summary = {"completed": 0, "failed": total}
    else:
        total_errors = sum(
            s["substitutions"] + s["deletions"] + s["insertions"] for s in successes
        )
        total_ref_words = sum(
            s["substitutions"] + s["deletions"] + s["hits"] for s in successes
        )
        corpus_wer = total_errors / total_ref_words if total_ref_words > 0 else 0.0

        wer_arr = np.array([s["wer"] for s in successes])
        latencies = [s["latency_s"] for s in successes]
        audio_durations = [
            s["audio_duration_s"] for s in successes if s["audio_duration_s"] > 0
        ]

        n_above_50 = int(np.sum(wer_arr > 0.5))

        ok_samples = [s for s in successes if s["wer"] <= 0.5]
        if ok_samples:
            ok_errors = sum(
                s["substitutions"] + s["deletions"] + s["insertions"]
                for s in ok_samples
            )
            ok_ref = sum(
                s["substitutions"] + s["deletions"] + s["hits"] for s in ok_samples
            )
            wer_below_50_micro = ok_errors / ok_ref if ok_ref > 0 else 0.0
        else:
            wer_below_50_micro = 0.0

        summary = {
            "lang": "en",
            "total_samples": total,
            "evaluated": evaluated,
            "skipped": skipped,
            "wer_corpus": float(corpus_wer),
            "wer_per_sample_mean": float(np.mean(wer_arr)),
            "wer_per_sample_median": float(np.median(wer_arr)),
            "wer_per_sample_std": float(np.std(wer_arr)),
            "wer_per_sample_p95": float(np.percentile(wer_arr, 95)),
            "wer_below_50_corpus": float(wer_below_50_micro),
            "n_above_50_pct_wer": n_above_50,
            "pct_above_50_pct_wer": (
                n_above_50 / evaluated * 100 if evaluated else 0
            ),
            "latency_mean_s": float(np.mean(latencies)),
            "audio_duration_mean_s": (
                float(np.mean(audio_durations)) if audio_durations else 0
            ),
            "num_parts": len(part_files),
        }

    # Print summary
    w = 60
    lw = 30
    print(f"\n{'=' * w}")
    print(f"{'Merged Qwen3-Omni WER Result':^{w}}")
    print(f"{'=' * w}")
    print(f"  {'Parts merged:':<{lw}} {len(part_files)}")
    print(
        f"  {'Evaluated / Total:':<{lw}} {summary.get('evaluated', 0)}/{summary.get('total_samples', 0)}"
    )
    print(f"  {'Skipped:':<{lw}} {summary.get('skipped', 0)}")
    print(f"{'-' * w}")
    print(
        f"  {'WER (corpus, micro-avg):':<{lw}} {summary.get('wer_corpus', 0):.4f} ({summary.get('wer_corpus', 0)*100:.2f}%)"
    )
    print(f"{'-' * w}")
    print(
        f"  {'WER per-sample mean:':<{lw}} {summary.get('wer_per_sample_mean', 0):.4f} ({summary.get('wer_per_sample_mean', 0)*100:.2f}%)"
    )
    print(
        f"  {'WER per-sample median:':<{lw}} {summary.get('wer_per_sample_median', 0):.4f}"
    )
    print(
        f"  {'WER per-sample std:':<{lw}} {summary.get('wer_per_sample_std', 0):.4f}"
    )
    print(
        f"  {'WER per-sample p95:':<{lw}} {summary.get('wer_per_sample_p95', 0):.4f}"
    )
    print(
        f"  {'WER corpus (excl >50%):':<{lw}} {summary.get('wer_below_50_corpus', 0):.4f} ({summary.get('wer_below_50_corpus', 0)*100:.2f}%)"
    )
    print(
        f"  {'>50% WER samples:':<{lw}} {summary.get('n_above_50_pct_wer', 0)} ({summary.get('pct_above_50_pct_wer', 0):.1f}%)"
    )
    print(f"{'-' * w}")
    print(
        f"  {'Latency mean (s):':<{lw}} {summary.get('latency_mean_s', 'N/A')}"
    )
    print(
        f"  {'Audio duration mean (s):':<{lw}} {summary.get('audio_duration_mean_s', 'N/A')}"
    )
    print(f"{'=' * w}\n")

    return {
        "summary": summary,
        "config": {
            "merged_from": part_files,
            "part_configs": configs,
        },
        "per_sample": all_samples,
    }


def main():
    p = argparse.ArgumentParser(description="Merge sharded WER results")
    p.add_argument(
        "--parts", nargs="+", required=True, help="Paths to per-shard wer_results.json"
    )
    p.add_argument("--output", required=True, help="Output merged JSON path")
    args = p.parse_args()

    for path in args.parts:
        try:
            with open(path) as f:
                json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"ERROR: Cannot read {path}: {e}", file=sys.stderr)
            sys.exit(1)

    merged = merge_results(args.parts)

    import os

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(merged, f, indent=2, ensure_ascii=False)

    print(f"Merged results saved to: {args.output}")


if __name__ == "__main__":
    main()
