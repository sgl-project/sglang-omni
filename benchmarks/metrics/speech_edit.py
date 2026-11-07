# SPDX-License-Identifier: Apache-2.0
"""Signal-level metrics for speech time-stretch and volume editing."""

from __future__ import annotations

import statistics
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf


def score_signal_edit(
    source_audio: str | Path,
    generated_audio: str | Path,
    *,
    task: str,
    scale: float,
) -> dict[str, float]:
    """Score one generated file using the benchmark's official formulas."""
    if scale <= 0:
        raise ValueError("scale must be positive")
    source, source_rate = sf.read(source_audio, dtype="float32", always_2d=False)
    generated, generated_rate = sf.read(
        generated_audio, dtype="float32", always_2d=False
    )
    if source.size == 0 or generated.size == 0:
        raise ValueError("source and generated audio must be non-empty")

    if task == "time_stretch":
        source_duration = _duration_seconds(source, source_rate)
        generated_duration = _duration_seconds(generated, generated_rate)
        target_duration = source_duration / scale
        absolute_error = abs(generated_duration - target_duration)
        return {
            "source_duration_s": source_duration,
            "target_duration_s": target_duration,
            "generated_duration_s": generated_duration,
            "absolute_duration_error_s": absolute_error,
            "relative_duration_error": absolute_error / source_duration,
        }

    if task == "volume":
        source_amplitude = float(np.abs(source).mean())
        if source_amplitude == 0:
            raise ValueError(
                "relative amplitude error is undefined for silent source audio"
            )
        generated_amplitude = float(np.abs(generated).mean())
        target_amplitude = source_amplitude * scale
        absolute_error = abs(generated_amplitude - target_amplitude)
        return {
            "source_mean_absolute_amplitude": source_amplitude,
            "target_mean_absolute_amplitude": target_amplitude,
            "generated_mean_absolute_amplitude": generated_amplitude,
            "absolute_amplitude_error": absolute_error,
            "relative_amplitude_error": absolute_error / source_amplitude,
        }

    raise ValueError("signal metrics are available only for time_stretch and volume")


def aggregate_signal_edit_scores(
    scores: Iterable[dict[str, float]],
) -> dict[str, Any]:
    """Average the official error fields while retaining the sample count."""
    rows = list(scores)
    if not rows:
        return {"evaluated": 0}
    metric_names = (
        "absolute_duration_error_s",
        "relative_duration_error",
        "absolute_amplitude_error",
        "relative_amplitude_error",
    )
    summary: dict[str, Any] = {"evaluated": len(rows)}
    for name in metric_names:
        values = [row[name] for row in rows if name in row]
        if values:
            summary[f"{name}_mean"] = statistics.fmean(values)
    return summary


def _duration_seconds(audio: np.ndarray, sample_rate: int) -> float:
    if sample_rate <= 0:
        raise ValueError("sample rate must be positive")
    return float(audio.shape[0]) / sample_rate
