# SPDX-License-Identifier: Apache-2.0
"""WER and ASR-speed metric computation and presentation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from benchmarks.metrics._format import (
    SPEED_LABEL_WIDTH,
    SPEED_LINE_WIDTH,
    print_speed_metric_line,
)


@dataclass
class SampleOutput:
    sample_id: str = ""
    target_text: str = ""
    whisper_text: str = ""
    ref_norm: str = ""
    hyp_norm: str = ""
    wer: float = 0.0
    substitutions: int = 0
    deletions: int = 0
    insertions: int = 0
    hits: int = 0
    audio_duration_s: float = 0.0
    latency_s: float = 0.0
    asr_latency_s: float = 0.0
    is_success: bool = False
    error: str = ""


def calculate_wer_metrics(outputs: list[SampleOutput], lang: str) -> dict:
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
            "latency_median_s": 0.0,
            "latency_p95_s": 0.0,
            "rtf_mean": 0.0,
            "audio_duration_mean_s": 0.0,
        }

    total_errors = sum(o.substitutions + o.deletions + o.insertions for o in successes)
    total_ref_words = sum(o.substitutions + o.deletions + o.hits for o in successes)
    corpus_wer = total_errors / total_ref_words if total_ref_words > 0 else 0.0

    wer_arr = np.array([o.wer for o in successes])
    latencies = [o.latency_s for o in successes]
    audio_durations = [o.audio_duration_s for o in successes if o.audio_duration_s > 0]
    latency_arr = np.array(latencies) if latencies else np.array([0.0])
    rtf_values = [
        o.latency_s / o.audio_duration_s
        for o in successes
        if o.audio_duration_s > 0 and o.latency_s > 0
    ]

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
        "latency_mean_s": float(np.mean(latency_arr)),
        "latency_median_s": float(np.median(latency_arr)),
        "latency_p95_s": float(np.percentile(latency_arr, 95)),
        "rtf_mean": float(np.mean(rtf_values)) if rtf_values else 0.0,
        "audio_duration_mean_s": (
            float(np.mean(audio_durations)) if audio_durations else 0
        ),
    }


def _metric_value(metrics: dict, *keys: str, default: float = 0.0) -> float:
    for key in keys:
        value = metrics.get(key)
        if value is not None:
            return value
    return default


def _print_wer_summary_table(
    metrics: dict,
    model_name: str,
    *,
    title: str,
    model_label: str,
    generation_mode: str | None = None,
    tts_speed_summary: dict | None = None,
) -> None:
    lw = SPEED_LABEL_WIDTH
    w = SPEED_LINE_WIDTH
    if generation_mode:
        title = f"{title} ({generation_mode})"
    wer_corpus = _metric_value(metrics, "wer_corpus", "corpus_wer")
    wer_per_sample_mean = _metric_value(metrics, "wer_per_sample_mean")
    wer_per_sample_max = _metric_value(metrics, "wer_per_sample_max")
    print(f"\n{'=' * w}")
    print(f"{title:^{w}}")
    print(f"{'=' * w}")
    print(f"  {model_label:<{lw}} {model_name}")
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
        f"{wer_corpus:.4f} "
        f"({wer_corpus * 100:.2f}%)"
    )
    print(f"{'-' * w}")
    print(
        f"  {'WER per-sample mean:':<{lw}} "
        f"{wer_per_sample_mean:.4f} "
        f"({wer_per_sample_mean * 100:.2f}%)"
    )
    print(
        f"  {'WER per-sample median:':<{lw}} "
        f"{_metric_value(metrics, 'wer_per_sample_median'):.4f}"
    )
    print(
        f"  {'WER per-sample std:':<{lw}} "
        f"{_metric_value(metrics, 'wer_per_sample_std'):.4f}"
    )
    print(
        f"  {'WER per-sample p95:':<{lw}} "
        f"{_metric_value(metrics, 'wer_per_sample_p95'):.4f}"
    )
    print(
        f"  {'WER per-sample max:':<{lw}} "
        f"{wer_per_sample_max:.4f} "
        f"({wer_per_sample_max * 100:.2f}%)"
    )
    print(
        f"  {'WER corpus (excl >50%):':<{lw}} "
        f"{_metric_value(metrics, 'wer_below_50_corpus'):.4f} "
        f"({_metric_value(metrics, 'wer_below_50_corpus') * 100:.2f}%)"
    )
    print(
        f"  {'>50% WER samples:':<{lw}} "
        f"{metrics.get('n_above_50_pct_wer', 0)} "
        f"({metrics.get('pct_above_50_pct_wer', 0):.1f}%)"
    )
    print(f"{'-' * w}")
    print_speed_metric_line(lw, "Latency mean (s):", metrics, "latency_mean_s")
    print_speed_metric_line(lw, "Latency p95 (s):", metrics, "latency_p95_s")
    print_speed_metric_line(lw, "RTF mean:", metrics, "rtf_mean")
    if tts_speed_summary is not None:
        print_speed_metric_line(
            lw, "TTFC mean (s):", tts_speed_summary, "audio_ttfp_mean_s"
        )
        print_speed_metric_line(
            lw, "Throughput (req/s):", tts_speed_summary, "throughput_qps"
        )
    print_speed_metric_line(
        lw, "Audio duration mean (s):", metrics, "audio_duration_mean_s"
    )
    print(f"{'=' * w}\n")


def print_wer_summary(
    metrics: dict,
    model_name: str,
    generation_mode: str | None = None,
    *,
    tts_speed_summary: dict | None = None,
) -> None:
    _print_wer_summary_table(
        metrics,
        model_name,
        title="TTS WER Benchmark Result",
        model_label="TTS model:",
        generation_mode=generation_mode,
        tts_speed_summary=tts_speed_summary,
    )


def print_asr_wer_summary(metrics: dict, model_name: str) -> None:
    _print_wer_summary_table(
        metrics,
        model_name,
        title="ASR WER Benchmark Result",
        model_label="ASR model:",
    )


def calculate_asr_speed_metrics(
    outputs: list[SampleOutput],
    *,
    wall_time_s: float | None = None,
) -> dict:
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
            "asr_latency_sum_s": 0.0,
            "asr_throughput_samples_per_s": 0.0,
            "asr_rtf_mean": 0.0,
            "asr_rtf_median": 0.0,
            "asr_audio_processed_s": 0.0,
        }

    latencies = np.array([o.asr_latency_s for o in successes])
    latency_sum_s = float(np.sum(latencies))
    total_asr_time = (
        float(wall_time_s)
        if wall_time_s is not None and wall_time_s > 0
        else latency_sum_s
    )

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
        "asr_latency_sum_s": latency_sum_s,
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
    print(f"  {'ASR model:':<{lw}} {model_name}")
    print(
        f"  {'Evaluated / Total:':<{lw}} "
        f"{metrics.get('evaluated', 0)}/{metrics.get('total_samples', 0)}"
    )
    if metrics.get("asr_concurrency"):
        print(f"  {'ASR concurrency:':<{lw}} {metrics['asr_concurrency']}")
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
    if metrics.get("asr_rtf_p95") is not None:
        print(f"  {'ASR RTF p95:':<{lw}} {metrics['asr_rtf_p95']}")
    print(
        f"  {'ASR total time (s):':<{lw}} " f"{metrics.get('asr_total_time_s', 'N/A')}"
    )
    if metrics.get("asr_latency_sum_s") and (
        metrics.get("asr_latency_sum_s") != metrics.get("asr_total_time_s")
    ):
        print(
            f"  {'ASR latency sum (s):':<{lw}} "
            f"{metrics.get('asr_latency_sum_s', 'N/A')}"
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
