# SPDX-License-Identifier: Apache-2.0
"""WER computation tools.

Micro-average WER = sum(S+D+I) / sum(S+D+C), consistent with HuggingFace evaluate.
"""

from __future__ import annotations

import numpy as np
from jiwer import process_words


def compute_wer(ref_norm: str, hyp_norm: str) -> dict:
    if not ref_norm:
        return {"wer": 0.0, "error": "Empty reference"}

    measures = process_words(ref_norm, hyp_norm)
    return {
        "wer": measures.wer,
        "substitutions": measures.substitutions,
        "deletions": measures.deletions,
        "insertions": measures.insertions,
        "hits": measures.hits,
    }


def aggregate_wer(
    per_sample_metrics: list[dict], bad_case_threshold: float = 0.5
) -> dict:
    wer_samples = [m for m in per_sample_metrics if "hits" in m]
    if not wer_samples:
        return {"micro_wer": None}

    total_errors = sum(
        m["substitutions"] + m["deletions"] + m["insertions"] for m in wer_samples
    )
    total_ref = sum(
        m["substitutions"] + m["deletions"] + m["hits"] for m in wer_samples
    )
    micro_wer = total_errors / total_ref if total_ref > 0 else 0.0

    wer_arr = np.array([m["wer"] for m in wer_samples])
    n_bad = int(np.sum(wer_arr > bad_case_threshold))

    ok = [m for m in wer_samples if m["wer"] <= bad_case_threshold]
    if ok:
        ok_err = sum(m["substitutions"] + m["deletions"] + m["insertions"] for m in ok)
        ok_ref = sum(m["substitutions"] + m["deletions"] + m["hits"] for m in ok)
        micro_wer_excl_bad = ok_err / ok_ref if ok_ref > 0 else 0.0
    else:
        micro_wer_excl_bad = 0.0

    return {
        "micro_wer": float(micro_wer),
        "wer_per_sample_mean": float(np.mean(wer_arr)),
        "wer_per_sample_median": float(np.median(wer_arr)),
        "wer_per_sample_std": float(np.std(wer_arr)),
        "wer_per_sample_p95": float(np.percentile(wer_arr, 95)),
        "micro_wer_excl_bad": float(micro_wer_excl_bad),
        "bad_case_count": n_bad,
        "bad_case_threshold": bad_case_threshold,
        "bad_case_pct": n_bad / len(wer_samples) * 100,
    }
