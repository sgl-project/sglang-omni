# SPDX-License-Identifier: Apache-2.0
"""Paired speaker-similarity margin utilities for voice-copy benchmarks."""

from __future__ import annotations

import json
import os

VOICE_COPY_MARGIN_RESULTS_FILENAME = "voice_copy_margin_results.json"


def _index_similarity_rows(
    results: dict, label: str
) -> tuple[dict[str, dict], list[str]]:
    rows = results.get("per_sample")
    if not isinstance(rows, list):
        raise ValueError(f"{label} similarity results must contain a per_sample list")

    indexed: dict[str, dict] = {}
    ordered_ids: list[str] = []
    for idx, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"{label} per_sample[{idx}] must be an object")
        sample_id = row.get("id") or row.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id:
            raise ValueError(f"{label} per_sample[{idx}] is missing a sample id")
        if sample_id in indexed:
            raise ValueError(
                f"{label} similarity results contain duplicate id {sample_id}"
            )
        indexed[sample_id] = row
        ordered_ids.append(sample_id)
    return indexed, ordered_ids


def _speaker_similarity_score(row: dict | None, label: str) -> tuple[float | None, str]:
    if row is None:
        return None, f"{label} row missing"
    score = row.get("speaker_similarity")
    if row.get("is_success") is not True:
        return None, str(row.get("error") or f"{label} similarity row failed")
    if isinstance(score, bool) or not isinstance(score, (int, float)):
        return None, f"{label} speaker_similarity missing"
    return float(score), ""


def _save_json_results(results: dict, output_dir: str, filename: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    with open(path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    return path


def compute_seedtts_voice_copy_margin(
    copy_similarity_results: dict,
    no_copy_similarity_results: dict,
    *,
    output_dir: str | None = None,
) -> dict:
    """Compute paired SeedTTS voice-copy similarity margin by sample id."""
    copy_by_id, ordered_ids = _index_similarity_rows(
        copy_similarity_results, "voice-copy"
    )
    no_copy_by_id, no_copy_ordered_ids = _index_similarity_rows(
        no_copy_similarity_results, "no-copy"
    )
    for sample_id in no_copy_ordered_ids:
        if sample_id not in copy_by_id:
            ordered_ids.append(sample_id)

    margins: list[float] = []
    copy_scores: list[float] = []
    no_copy_scores: list[float] = []
    per_sample: list[dict] = []

    for sample_id in ordered_ids:
        copy_row = copy_by_id.get(sample_id)
        no_copy_row = no_copy_by_id.get(sample_id)
        copy_score, copy_error = _speaker_similarity_score(copy_row, "voice-copy")
        no_copy_score, no_copy_error = _speaker_similarity_score(no_copy_row, "no-copy")

        base_row = {
            "id": sample_id,
            "ref_audio": (copy_row or no_copy_row or {}).get("ref_audio"),
            "copy_wav_path": (copy_row or {}).get("wav_path"),
            "no_copy_wav_path": (no_copy_row or {}).get("wav_path"),
            "copy_similarity": copy_score,
            "no_copy_similarity": no_copy_score,
        }
        if copy_score is None or no_copy_score is None:
            errors = [error for error in (copy_error, no_copy_error) if error]
            per_sample.append(
                {
                    **base_row,
                    "speaker_similarity_margin": None,
                    "is_success": False,
                    "error": "; ".join(errors),
                }
            )
            continue

        margin = copy_score - no_copy_score
        margins.append(margin)
        copy_scores.append(copy_score)
        no_copy_scores.append(no_copy_score)
        per_sample.append(
            {
                **base_row,
                "speaker_similarity_margin": margin,
                "is_success": True,
                "error": None,
            }
        )

    if not margins:
        raise RuntimeError(
            "SeedTTS voice-copy margin: no paired scoreable samples "
            f"({len(per_sample)} skipped)."
        )

    summary = {
        "speaker_similarity_copy_mean": sum(copy_scores) / len(copy_scores),
        "speaker_similarity_no_copy_mean": sum(no_copy_scores) / len(no_copy_scores),
        "speaker_similarity_margin_mean": sum(margins) / len(margins),
        "speaker_similarity_margin_min": min(margins),
        "speaker_similarity_margin_max": max(margins),
        "total_samples": len(per_sample),
        "evaluated": len(margins),
        "skipped": len(per_sample) - len(margins),
    }
    results = {
        "summary": summary,
        "config": {
            "copy_total_samples": copy_similarity_results.get("summary", {}).get(
                "total_samples"
            ),
            "no_copy_total_samples": no_copy_similarity_results.get("summary", {}).get(
                "total_samples"
            ),
        },
        "per_sample": per_sample,
    }
    if output_dir is not None:
        _save_json_results(results, output_dir, VOICE_COPY_MARGIN_RESULTS_FILENAME)
    return results
