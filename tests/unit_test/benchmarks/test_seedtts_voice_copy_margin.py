from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest

from benchmarks.metrics.voice_copy_margin import compute_seedtts_voice_copy_margin


def _similarity_results(rows: list[dict]) -> dict:
    evaluated = sum(1 for row in rows if row.get("is_success"))
    return {
        "summary": {
            "total_samples": len(rows),
            "evaluated": evaluated,
            "skipped": len(rows) - evaluated,
        },
        "per_sample": rows,
    }


def _row(sample_id: str, score: float | None, *, success: bool = True) -> dict:
    return {
        "id": sample_id,
        "ref_audio": f"/ref/{sample_id}.wav",
        "wav_path": f"/gen/{sample_id}.wav",
        "speaker_similarity": score,
        "is_success": success,
        "error": None if success else "failed",
    }


def test_seedtts_voice_copy_margin_pairs_by_sample_id(tmp_path: Path) -> None:
    copy_results = _similarity_results([_row("b", 0.62), _row("a", 0.81)])
    no_copy_results = _similarity_results([_row("a", 0.51), _row("b", 0.40)])

    results = compute_seedtts_voice_copy_margin(
        copy_results,
        no_copy_results,
        output_dir=str(tmp_path),
    )

    summary = results["summary"]
    assert summary["speaker_similarity_copy_mean"] == pytest.approx(0.715)
    assert summary["speaker_similarity_no_copy_mean"] == pytest.approx(0.455)
    assert summary["speaker_similarity_margin_mean"] == pytest.approx(0.26)
    assert summary["evaluated"] == 2
    assert summary["skipped"] == 0
    assert [row["id"] for row in results["per_sample"]] == ["b", "a"]

    saved = tmp_path / "voice_copy_margin_results.json"
    assert saved.exists()
    assert json.loads(saved.read_text())["summary"] == summary


def test_seedtts_voice_copy_margin_records_missing_and_failed_rows() -> None:
    copy_results = _similarity_results(
        [
            _row("a", 0.80),
            _row("b", None, success=False),
            _row("c", 0.75),
        ]
    )
    no_copy_results = _similarity_results([_row("a", 0.50), _row("b", 0.40)])

    results = compute_seedtts_voice_copy_margin(copy_results, no_copy_results)

    summary = results["summary"]
    assert summary["speaker_similarity_margin_mean"] == pytest.approx(0.30)
    assert summary["evaluated"] == 1
    assert summary["skipped"] == 2
    skipped = [row for row in results["per_sample"] if not row["is_success"]]
    assert {row["id"] for row in skipped} == {"b", "c"}
    assert any("no-copy row missing" in row["error"] for row in skipped)


def _load_tune_module() -> ModuleType:
    path = (
        Path(__file__).resolve().parents[3]
        / ".claude/skills/tune-ci-thresholds/tune.py"
    )
    spec = importlib.util.spec_from_file_location("tune_ci_thresholds", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_tune_ci_thresholds_discovers_similarity_margin_before_mean() -> None:
    tune = _load_tune_module()

    assert tune.match_metric("VC_SIMILARITY_MARGIN_MIN", None) == "similarity_margin"
    assert tune.match_metric("VC_SIMILARITY_MEAN_MIN", None) == "similarity_mean"
    assert tune.METRIC_SPECS["similarity_margin"]["worst"] == "min"
