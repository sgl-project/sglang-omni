# SPDX-License-Identifier: Apache-2.0
"""WER and ASR-speed metric computation and presentation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from benchmarks.metrics._format import SPEED_LABEL_WIDTH, SPEED_LINE_WIDTH

if TYPE_CHECKING:
    from benchmarks.tasks.tts import SampleOutput


def calculate_wer_metrics(outputs: list["SampleOutput"], lang: str) -> dict:
    """Compute corpus-level WER metrics from per-sample outputs."""
    successes = [o for o in outputs if o.is_success]
    if not successes:
        return {
            "lang": lang,
            "total_samples": len(outputs),
            "evaluated": 0,
            "skipped": len(outputs),
            "wer_corpus": 0.0,
            "wer_per_sample_mean": 0.0,
            "wer_per_sample_median": 0.0,
            "wer_per_sample_std": 0.0,
            "wer_per_sample_p95": 0.0,
            "wer_per_sample_max": 0.0,
            "wer_below_50_corpus": 0.0,
            "n_above_50_pct_wer": 0,
            "pct_above_50_pct_wer": 0.0,
            "latency_mean_s": 0.0,
            "audio_duration_mean_s": 0.0,
        }

    total_errors = sum(o.substitutions + o.deletions + o.insertions for o in successes)
    total_ref_words = sum(o.substitutions + o.deletions + o.hits for o in successes)
    corpus_wer = total_errors / total_ref_words if total_ref_words > 0 else 0.0

    wer_arr = np.array([o.wer for o in successes])
    latencies = [o.latency_s for o in successes]
    audio_durations = [o.audio_duration_s for o in successes if o.audio_duration_s > 0]

    n_above_50 = int(np.sum(wer_arr > 0.5))
    ok_samples = [o for o in successes if o.wer <= 0.5]
    if ok_samples:
        ok_errors = sum(
            o.substitutions + o.deletions + o.insertions for o in ok_samples
        )
        ok_ref = sum(o.substitutions + o.deletions + o.hits for o in ok_samples)
        wer_below_50_micro = ok_errors / ok_ref if ok_ref > 0 else 0.0
    else:
        wer_below_50_micro = 0.0

    return {
        "lang": lang,
        "total_samples": len(outputs),
        "evaluated": len(successes),
        "skipped": len(outputs) - len(successes),
        "wer_corpus": float(corpus_wer),
        "wer_per_sample_mean": float(np.mean(wer_arr)),
        "wer_per_sample_median": float(np.median(wer_arr)),
        "wer_per_sample_std": float(np.std(wer_arr)),
        "wer_per_sample_p95": float(np.percentile(wer_arr, 95)),
        "wer_per_sample_max": float(np.max(wer_arr)),
        "wer_below_50_corpus": float(wer_below_50_micro),
        "n_above_50_pct_wer": n_above_50,
        "pct_above_50_pct_wer": (n_above_50 / len(successes) * 100 if successes else 0),
        "latency_mean_s": float(np.mean(latencies)),
        "audio_duration_mean_s": (
            float(np.mean(audio_durations)) if audio_durations else 0
        ),
    }


def print_wer_summary(
    metrics: dict, model_name: str, generation_mode: str | None = None
) -> None:
    lw = SPEED_LABEL_WIDTH
    w = SPEED_LINE_WIDTH
    title = "TTS WER Benchmark Result"
    if generation_mode:
        title = f"TTS WER Benchmark Result ({generation_mode})"
    print(f"\n{'=' * w}")
    print(f"{title:^{w}}")
    print(f"{'=' * w}")
    print(f"  {'Model:':<{lw}} {model_name}")
    if generation_mode:
        print(f"  {'Generation mode:':<{lw}} {generation_mode}")
    print(f"  {'Language:':<{lw}} {metrics.get('lang', 'N/A')}")
    print(
        f"  {'Evaluated / Total:':<{lw}} "
        f"{metrics.get('evaluated', 0)}/{metrics.get('total_samples', 0)}"
    )
    print(f"  {'Skipped:':<{lw}} {metrics.get('skipped', 0)}")
    print(f"{'-' * w}")
    print(
        f"  {'WER (corpus, micro-avg):':<{lw}} "
        f"{metrics.get('wer_corpus', 0):.4f} "
        f"({metrics.get('wer_corpus', 0) * 100:.2f}%)"
    )
    print(f"{'-' * w}")
    print(
        f"  {'WER per-sample mean:':<{lw}} "
        f"{metrics.get('wer_per_sample_mean', 0):.4f} "
        f"({metrics.get('wer_per_sample_mean', 0) * 100:.2f}%)"
    )
    print(
        f"  {'WER per-sample median:':<{lw}} "
        f"{metrics.get('wer_per_sample_median', 0):.4f}"
    )
    print(
        f"  {'WER per-sample std:':<{lw}} "
        f"{metrics.get('wer_per_sample_std', 0):.4f}"
    )
    print(
        f"  {'WER per-sample p95:':<{lw}} "
        f"{metrics.get('wer_per_sample_p95', 0):.4f}"
    )
    print(
        f"  {'WER per-sample max:':<{lw}} "
        f"{metrics.get('wer_per_sample_max', 0):.4f} "
        f"({metrics.get('wer_per_sample_max', 0) * 100:.2f}%)"
    )
    print(
        f"  {'WER corpus (excl >50%):':<{lw}} "
        f"{metrics.get('wer_below_50_corpus', 0):.4f} "
        f"({metrics.get('wer_below_50_corpus', 0) * 100:.2f}%)"
    )
    print(
        f"  {'>50% WER samples:':<{lw}} "
        f"{metrics.get('n_above_50_pct_wer', 0)} "
        f"({metrics.get('pct_above_50_pct_wer', 0):.1f}%)"
    )
    print(f"{'-' * w}")
    print(f"  {'Latency mean (s):':<{lw}} {metrics.get('latency_mean_s', 'N/A')}")
    print(
        f"  {'Audio duration mean (s):':<{lw}} "
        f"{metrics.get('audio_duration_mean_s', 'N/A')}"
    )
    print(f"{'=' * w}\n")


def calculate_asr_speed_metrics(outputs: list["SampleOutput"]) -> dict:
    """Compute speed metrics for the ASR transcription phase."""
    successes = [o for o in outputs if o.is_success and o.asr_latency_s > 0]
    if not successes:
        return {
            "total_samples": len(outputs),
            "evaluated": 0,
            "skipped": len(outputs),
            "asr_latency_mean_s": 0.0,
            "asr_latency_median_s": 0.0,
            "asr_latency_p95_s": 0.0,
            "asr_latency_p99_s": 0.0,
            "asr_total_time_s": 0.0,
            "asr_throughput_samples_per_s": 0.0,
            "asr_rtf_mean": 0.0,
            "asr_rtf_median": 0.0,
            "asr_audio_processed_s": 0.0,
        }

    latencies = np.array([o.asr_latency_s for o in successes])
    total_asr_time = float(np.sum(latencies))

    audio_durations = [o.audio_duration_s for o in successes if o.audio_duration_s > 0]
    rtfs = np.array(
        [
            o.asr_latency_s / o.audio_duration_s
            for o in successes
            if o.audio_duration_s > 0
        ]
    )

    return {
        "total_samples": len(outputs),
        "evaluated": len(successes),
        "skipped": len(outputs) - len(successes),
        "asr_latency_mean_s": float(np.mean(latencies)),
        "asr_latency_median_s": float(np.median(latencies)),
        "asr_latency_p95_s": float(np.percentile(latencies, 95)),
        "asr_latency_p99_s": float(np.percentile(latencies, 99)),
        "asr_total_time_s": total_asr_time,
        "asr_throughput_samples_per_s": (
            float(len(successes) / total_asr_time) if total_asr_time > 0 else 0.0
        ),
        "asr_rtf_mean": float(np.mean(rtfs)) if len(rtfs) > 0 else 0.0,
        "asr_rtf_median": float(np.median(rtfs)) if len(rtfs) > 0 else 0.0,
        "asr_audio_processed_s": (
            float(sum(audio_durations)) if audio_durations else 0.0
        ),
    }


def print_asr_speed_summary(metrics: dict, model_name: str) -> None:
    """Print ASR speed metrics summary table."""
    lw = SPEED_LABEL_WIDTH
    w = SPEED_LINE_WIDTH
    print(f"\n{'=' * w}")
    print(f"{'ASR Speed Benchmark Result':^{w}}")
    print(f"{'=' * w}")
    print(f"  {'Model:':<{lw}} {model_name}")
    print(
        f"  {'Evaluated / Total:':<{lw}} "
        f"{metrics.get('evaluated', 0)}/{metrics.get('total_samples', 0)}"
    )
    print(f"  {'Skipped:':<{lw}} {metrics.get('skipped', 0)}")
    print(f"{'-' * w}")
    print(
        f"  {'ASR latency mean (s):':<{lw}} "
        f"{metrics.get('asr_latency_mean_s', 'N/A')}"
    )
    print(
        f"  {'ASR latency median (s):':<{lw}} "
        f"{metrics.get('asr_latency_median_s', 'N/A')}"
    )
    print(
        f"  {'ASR latency p95 (s):':<{lw}} "
        f"{metrics.get('asr_latency_p95_s', 'N/A')}"
    )
    print(
        f"  {'ASR latency p99 (s):':<{lw}} "
        f"{metrics.get('asr_latency_p99_s', 'N/A')}"
    )
    print(f"  {'ASR RTF mean:':<{lw}} {metrics.get('asr_rtf_mean', 'N/A')}")
    print(f"  {'ASR RTF median:':<{lw}} {metrics.get('asr_rtf_median', 'N/A')}")
    print(
        f"  {'ASR total time (s):':<{lw}} " f"{metrics.get('asr_total_time_s', 'N/A')}"
    )
    print(
        f"  {'ASR throughput (samples/s):':<{lw}} "
        f"{metrics.get('asr_throughput_samples_per_s', 'N/A')}"
    )
    if metrics.get("asr_audio_processed_s"):
        print(
            f"  {'Audio processed (s):':<{lw}} " f"{metrics['asr_audio_processed_s']}"
        )
    print(f"{'=' * w}")
