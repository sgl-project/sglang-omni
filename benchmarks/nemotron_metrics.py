# SPDX-License-Identifier: Apache-2.0
"""Metrics and report schema for the Nemotron streaming comparison.

The report deliberately has two independent backend sections (``oracle`` and
``sglang``).  It is not a model-card reproduction: model-card H100 numbers
must never be copied into either local RTX 4090 result.
"""

from __future__ import annotations

import math
import re
import subprocess
import threading
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

NOT_REPORTED = "not reported"
NOT_APPLICABLE = "not applicable"

_LANGUAGE_TAG = re.compile(r"<(?P<locale>[A-Za-z]{2,3}-[A-Za-z]{2,3})>")


def percentile(values: Iterable[float], fraction: float) -> float | None:
    """Return the nearest-rank percentile, or ``None`` for an empty series."""
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return None
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("fraction must be between 0 and 1")
    index = max(math.ceil(fraction * len(ordered)) - 1, 0)
    return ordered[index]


def levenshtein(left: str, right: str) -> int:
    """Character edit distance, kept public for CER and unit tests."""
    return _sequence_levenshtein(list(left), list(right))


def _sequence_levenshtein(left: list[Any], right: list[Any]) -> int:
    previous = list(range(len(right) + 1))
    for left_index, left_char in enumerate(left, 1):
        current = [left_index]
        for right_index, right_char in enumerate(right, 1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[right_index] + 1,
                    previous[right_index - 1] + (left_char != right_char),
                )
            )
        previous = current
    return previous[-1]


def normalize_transcript(text: str, language: str | None) -> str:
    """Conservative normalization matching the benchmark's declared scope."""
    value = _LANGUAGE_TAG.sub(" ", text or "")
    value = re.sub(r"[^\w\s']+", " ", value, flags=re.UNICODE)
    value = re.sub(r"\s+", " ", value).strip().casefold()
    if (language or "").casefold().split("-", 1)[0] in {"zh", "ja", "ko"}:
        return "".join(value.split())
    return value


def score_transcript(
    reference: str,
    hypothesis: str,
    *,
    language: str | None,
    detected_language: str | None = None,
    auto_language: bool = False,
) -> dict[str, Any]:
    """Compute per-sample WER/CER and optional auto-language correctness."""
    reference_norm = normalize_transcript(reference, language)
    hypothesis_norm = normalize_transcript(hypothesis, language)
    chars = (language or "").casefold().split("-", 1)[0] in {"zh", "ja", "ko"}
    reference_units = list(reference_norm) if chars else reference_norm.split()
    hypothesis_units = list(hypothesis_norm) if chars else hypothesis_norm.split()
    errors = _sequence_levenshtein(reference_units, hypothesis_units)
    denominator = max(len(reference_units), 1)
    result: dict[str, Any] = {
        "reference": reference,
        "hypothesis": hypothesis,
        "reference_normalized": reference_norm,
        "hypothesis_normalized": hypothesis_norm,
        "metric": "CER" if chars else "WER",
        "error_rate": errors / denominator,
        "reference_units": len(reference_units),
        "edit_errors": errors,
    }
    if auto_language:
        expected = (language or "").casefold()
        detected = (detected_language or "").casefold()
        result["language_correct"] = detected == expected if detected else None
    else:
        result["language_correct"] = None
    return result


def aggregate_quality(samples: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    rows = list(samples)
    valid = [row for row in rows if row.get("error_rate") is not None]
    total_units = sum(int(row.get("reference_units", 0)) for row in valid)
    total_errors = sum(int(row.get("edit_errors", 0)) for row in valid)
    language_rows = [row for row in valid if row.get("language_correct") is not None]
    return {
        "samples": len(rows),
        "scored_samples": len(valid),
        "wer_cer": (
            total_errors / total_units if total_units else NOT_REPORTED
        ),
        "language_accuracy": (
            sum(bool(row["language_correct"]) for row in language_rows)
            / len(language_rows)
            if language_rows
            else NOT_REPORTED
        ),
        "per_sample": rows,
    }


def batch_distribution(batch_sizes: Iterable[int]) -> dict[str, Any]:
    counts = Counter(int(size) for size in batch_sizes)
    total = sum(counts.values())
    return {
        "counts": {str(size): count for size, count in sorted(counts.items())},
        "fractions": {
            str(size): count / total for size, count in sorted(counts.items())
        }
        if total
        else {},
        "observations": total,
    }


@dataclass
class NvidiaSmiMonitor:
    """Best-effort process/GPU memory sampler without a Python NVML package."""

    gpu_index: int = 0
    interval_s: float = 0.1
    samples_mib: list[float] = field(default_factory=list)
    _stop: threading.Event = field(default_factory=threading.Event, init=False)
    _thread: threading.Thread | None = field(default=None, init=False)

    def start(self) -> None:
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> dict[str, Any]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, self.interval_s * 5))
        return {
            "available": bool(self.samples_mib),
            "peak_used_mib": max(self.samples_mib) if self.samples_mib else NOT_REPORTED,
            "samples": len(self.samples_mib),
            "sample_interval_s": self.interval_s,
        }

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                output = subprocess.check_output(
                    [
                        "nvidia-smi",
                        "--id",
                        str(self.gpu_index),
                        "--query-gpu=memory.used",
                        "--format=csv,noheader,nounits",
                    ],
                    text=True,
                    stderr=subprocess.DEVNULL,
                    timeout=2,
                )
                self.samples_mib.append(float(output.strip().splitlines()[0]))
            except (OSError, subprocess.SubprocessError, ValueError, IndexError):
                pass
            self._stop.wait(self.interval_s)


def empty_performance() -> dict[str, Any]:
    return {
        "req_per_s": NOT_REPORTED,
        "audio_seconds_per_s": NOT_REPORTED,
        "rtf": NOT_REPORTED,
        "rtfx": NOT_REPORTED,
        "e2e_p50_s": NOT_REPORTED,
        "e2e_p99_s": NOT_REPORTED,
        "ttft_p50_s": NOT_REPORTED,
        "ttft_p99_s": NOT_REPORTED,
        "chunk_p50_ms": NOT_REPORTED,
        "chunk_p99_ms": NOT_REPORTED,
        "finalization_latency_p50_s": NOT_REPORTED,
        "finalization_latency_p99_s": NOT_REPORTED,
        "actual_batch_distribution": batch_distribution([]),
        "cache_reuse": {"count": NOT_REPORTED, "rate": NOT_REPORTED},
        "nvml_peak_used_mib": NOT_REPORTED,
        "pytorch_peak_allocated_bytes": NOT_REPORTED,
        "pytorch_peak_reserved_bytes": NOT_REPORTED,
    }


def aggregate_performance(
    samples: Iterable[Mapping[str, Any]],
    *,
    wall_time_s: float,
    batch_sizes: Iterable[int],
    cache_reuse_count: int,
    cache_observations: int,
    nvml: Mapping[str, Any] | None = None,
    pytorch_peak_allocated_bytes: int | None = None,
    pytorch_peak_reserved_bytes: int | None = None,
) -> dict[str, Any]:
    rows = list(samples)
    audio_s = sum(float(row.get("audio_seconds", 0.0)) for row in rows)
    e2e = [float(row["e2e_s"]) for row in rows if row.get("e2e_s") is not None]
    ttft = [float(row["ttft_s"]) for row in rows if row.get("ttft_s") is not None]
    finalization = [
        float(row["finalization_latency_s"])
        for row in rows
        if row.get("finalization_latency_s") is not None
    ]
    chunk = [
        float(value)
        for row in rows
        for value in row.get("chunk_latency_ms", [])
    ]
    compute_s = sum(float(row.get("model_compute_s", 0.0)) for row in rows)
    completed = len(rows)

    def present(value: float | None) -> float | str:
        return value if value is not None else NOT_REPORTED

    return {
        "req_per_s": completed / wall_time_s if wall_time_s > 0 else NOT_REPORTED,
        "audio_seconds_per_s": audio_s / wall_time_s if wall_time_s > 0 else NOT_REPORTED,
        "rtf": compute_s / audio_s if compute_s > 0 and audio_s > 0 else NOT_REPORTED,
        "rtfx": audio_s / compute_s if compute_s > 0 else NOT_REPORTED,
        "e2e_p50_s": present(percentile(e2e, 0.50)),
        "e2e_p99_s": present(percentile(e2e, 0.99)),
        "ttft_p50_s": present(percentile(ttft, 0.50)),
        "ttft_p99_s": present(percentile(ttft, 0.99)),
        "chunk_p50_ms": present(percentile(chunk, 0.50)),
        "chunk_p99_ms": present(percentile(chunk, 0.99)),
        "finalization_latency_p50_s": present(percentile(finalization, 0.50)),
        "finalization_latency_p99_s": present(percentile(finalization, 0.99)),
        "actual_batch_distribution": batch_distribution(batch_sizes),
        "cache_reuse": {
            "count": cache_reuse_count,
            "rate": cache_reuse_count / cache_observations if cache_observations else NOT_REPORTED,
        },
        "nvml_peak_used_mib": (nvml or {}).get("peak_used_mib", NOT_REPORTED),
        "pytorch_peak_allocated_bytes": (
            pytorch_peak_allocated_bytes
            if pytorch_peak_allocated_bytes is not None
            else NOT_REPORTED
        ),
        "pytorch_peak_reserved_bytes": (
            pytorch_peak_reserved_bytes
            if pytorch_peak_reserved_bytes is not None
            else NOT_REPORTED
        ),
    }


def new_report(*, config: Mapping[str, Any]) -> dict[str, Any]:
    """Create the stable top-level shape used by JSON and Markdown reports."""
    return {
        "schema_version": 1,
        "comparison_scope": "local_rtx4090_only",
        "model_card": {"included": False, "reason": "H100 conditions not comparable"},
        "config": dict(config),
        "oracle": {"quality": aggregate_quality([]), "performance": empty_performance()},
        "sglang": {"quality": aggregate_quality([]), "performance": empty_performance()},
    }


__all__ = [
    "NOT_APPLICABLE",
    "NOT_REPORTED",
    "NvidiaSmiMonitor",
    "aggregate_performance",
    "aggregate_quality",
    "batch_distribution",
    "empty_performance",
    "levenshtein",
    "new_report",
    "normalize_transcript",
    "percentile",
    "score_transcript",
]
