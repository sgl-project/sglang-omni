# SPDX-License-Identifier: Apache-2.0
"""Waveform tolerance metrics for audio-touching changes.

Bitwise equality only holds on deterministic test models. On the real bf16
Code2Wav vocoder, batched decode diverges benignly from per-request decode
(#1126: peak ~9e-2 on ~0.2% of samples; ~2e-3 in fp32 with no sample over
1e-2, i.e. cuDNN batch-kernel numerics, not logic). These metrics replace
bitwise equality as the "audio not degraded" gate: SNR of the difference
signal, peak absolute difference, and the fraction of samples whose absolute
difference exceeds a per-sample threshold.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class WaveformComparison:
    num_samples: int
    length_mismatch: bool
    snr_db: float
    max_abs_diff: float
    exceed_fraction: float
    diff_threshold: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "num_samples": self.num_samples,
            "length_mismatch": self.length_mismatch,
            "snr_db": self.snr_db,
            "max_abs_diff": self.max_abs_diff,
            "exceed_fraction": self.exceed_fraction,
            "diff_threshold": self.diff_threshold,
        }


def compare_waveforms(
    reference: Any,
    candidate: Any,
    *,
    diff_threshold: float = 1e-2,
) -> WaveformComparison:
    """Compare a candidate waveform against a reference.

    On length mismatch the metrics are computed over the common prefix for
    diagnostics, and ``length_mismatch`` marks the comparison failed.
    """
    ref = np.asarray(reference, dtype=np.float64).reshape(-1)
    cand = np.asarray(candidate, dtype=np.float64).reshape(-1)
    length_mismatch = ref.shape[0] != cand.shape[0]
    n = min(ref.shape[0], cand.shape[0])
    ref = ref[:n]
    cand = cand[:n]
    if n == 0:
        return WaveformComparison(
            num_samples=0,
            length_mismatch=length_mismatch,
            snr_db=math.inf,
            max_abs_diff=0.0,
            exceed_fraction=0.0,
            diff_threshold=diff_threshold,
        )
    diff = cand - ref
    signal_energy = float(np.sum(ref * ref))
    noise_energy = float(np.sum(diff * diff))
    if noise_energy == 0.0:
        snr_db = math.inf
    elif signal_energy == 0.0:
        snr_db = -math.inf
    else:
        snr_db = 10.0 * math.log10(signal_energy / noise_energy)
    abs_diff = np.abs(diff)
    return WaveformComparison(
        num_samples=n,
        length_mismatch=length_mismatch,
        snr_db=snr_db,
        max_abs_diff=float(abs_diff.max()),
        exceed_fraction=float(np.mean(abs_diff > diff_threshold)),
        diff_threshold=diff_threshold,
    )


def tolerance_failures(
    comparison: WaveformComparison,
    *,
    min_snr_db: float,
    max_peak_diff: float,
    max_exceed_fraction: float,
) -> list[str]:
    """Return failure reasons for a comparison; an empty list means pass."""
    failures: list[str] = []
    if comparison.length_mismatch:
        failures.append("waveform lengths differ")
    if comparison.snr_db < min_snr_db:
        failures.append(f"snr {comparison.snr_db:.2f} dB < {min_snr_db:.2f} dB")
    if comparison.max_abs_diff > max_peak_diff:
        failures.append(
            f"peak diff {comparison.max_abs_diff:.3e} > {max_peak_diff:.3e}"
        )
    if comparison.exceed_fraction > max_exceed_fraction:
        failures.append(
            f"{comparison.exceed_fraction:.4%} of samples over "
            f"{comparison.diff_threshold:.3e} (limit {max_exceed_fraction:.4%})"
        )
    return failures
