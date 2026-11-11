# SPDX-License-Identifier: Apache-2.0
"""Exercise offline result tables without hiding missing evaluation evidence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import pytest
from pydantic import JsonValue

from benchmarks.duplex.reference_core import canonical_hash
from benchmarks.eval import benchmark_duplex_reference


def write_json(path: Path, contents: JsonValue) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(contents), encoding="utf-8")


@pytest.fixture
def report_artifacts(tmp_path: Path) -> tuple[Path, Path]:
    scores = tmp_path / "scores"
    engine = scores / "engines" / "test-model"
    samples = []
    replay_samples = []
    for sample_id in ("user_interruption/1", "background_speech/2"):
        variants = {}
        for variant in ("overlap", "clean"):
            eligible = (sample_id, variant) != ("background_speech/2", "clean")
            variants[variant] = {
                "eligible": eligible,
                "reasons": (
                    []
                    if eligible
                    else [
                        "append 106 send completed after v2 deadline",
                        "input pacing deviation 0.1077183615975077 exceeds 0.08s",
                    ]
                ),
                "source": {"trace_sha256": canonical_hash([sample_id, variant])},
                "protocol_diagnostics": {
                    "status": "pass" if eligible else "fail",
                    "protocol_verdict": "pass",
                },
            }
            replay_samples.append(
                {
                    "sample": sample_id,
                    "variant": variant,
                    "recorded_status": "pass" if eligible else "fail",
                    "status": "match",
                }
            )
        samples.append(
            {
                "sample_id": sample_id,
                "category": sample_id.split("/")[0],
                "variants": variants,
            }
        )
    manifest = {"samples": samples}
    write_json(engine / "projected-manifest.json", manifest)
    write_json(
        engine / "manifest-receipt.json",
        {
            "engine": "test-model",
            "samples": 2,
            "projected_manifest_sha256": canonical_hash(manifest),
        },
    )
    timing = {}
    for name, successful, ineligible in (
        ("timing_official_overlap", 2, 0),
        ("timing_supplementary_clean", 1, 1),
    ):
        timing[name] = {
            "population": 2,
            "status": {
                "ok": successful,
                "ineligible": ineligible,
                "label_model_speech_at_output_end": 1,
            },
            "official_all_intervals": {
                "stop": {
                    "interval_n": 0,
                    "samples": successful,
                    "zero_interval_samples": successful,
                    "mean_s": None,
                    "median_s": None,
                    "bootstrap_pooled_mean": {"ci95": None},
                },
                "response": {
                    "interval_n": 2,
                    "samples": successful,
                    "zero_interval_samples": 0,
                    "mean_s": 0.4,
                    "median_s": 0.4,
                    "bootstrap_pooled_mean": {"ci95": [0.2, 0.6]},
                },
            },
        }
    write_json(
        scores / "summary.json",
        {
            "reference_revision": "reference-test-revision",
            "populations": {
                "test-model": {
                    "manifest_sha256": canonical_hash(manifest),
                    "sample_ids": [sample["sample_id"] for sample in samples],
                }
            },
            "engines": {
                "test-model": {
                    "all": {
                        **timing,
                        "asr_files": {
                            "input.wav:ok": 2,
                            "output.wav:ok": 2,
                            "output.wav:ok_empty_transcript": 1,
                            "clean_input.wav:ok": 1,
                            "clean_input.wav:ineligible": 1,
                            "clean_output.wav:ok": 1,
                            "clean_output.wav:ineligible": 1,
                        },
                        "behavior": {
                            "population": 2,
                            "status": {"not_judged": 1, "variant_ineligible": 1},
                            "valid_n": 0,
                            "valid_label_proportions": {
                                label: {"count": 0, "proportion": None}
                                for label in (
                                    "C_RESPOND",
                                    "C_RESUME",
                                    "C_UNCERTAIN_HANDLING",
                                    "C_UNKNOWN",
                                )
                            },
                        },
                    }
                }
            },
        },
    )
    replay = tmp_path / "offline-replay.json"
    write_json(replay, {"selected": 4, "replayed": 4, "samples": replay_samples})
    return scores, replay


def test_report_separates_protocol_eligibility_and_score_coverage_without_writes(
    report_artifacts: tuple[Path, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    scores, replay = report_artifacts
    before = {
        path: (path.read_bytes(), path.stat().st_mtime_ns)
        for path in scores.parent.rglob("*")
        if path.is_file()
    }
    assert (
        benchmark_duplex_reference.main(
            [
                "report",
                "--scores",
                str(scores),
                "--engine",
                "test-model",
                "--replay",
                str(replay),
            ]
        )
        == 0
    )
    output = "\n".join(
        " ".join(line.split()) for line in capsys.readouterr().out.splitlines()
    )
    assert "Selected pairs 2" in output
    assert "Selected / recorded session traces 4 / 4" in output
    assert "Native protocol checks 4 / 4 passed" in output
    assert "Offline replay agreement 4 / 4 matched" in output
    assert "Reference-eligible sessions 3 / 4" in output
    assert "Complete eligible overlap-clean pairs 1 / 2" in output
    assert "Technical exclusions 1" in output
    assert "Successful ASR file roles 6 / 6 eligible / 8 selected" in output
    assert "Successful timing results 3 / 3 eligible / 4 selected" in output
    assert "Valid official behavior labels 0" in output
    assert "1 not_judged" in output and "1 variant_ineligible" in output
    assert "C_RESPOND 0 / 0; n/a" in output
    after = {
        path: (path.read_bytes(), path.stat().st_mtime_ns)
        for path in scores.parent.rglob("*")
        if path.is_file()
    }
    assert after == before


def test_report_preserves_missing_evidence_and_empty_intervals(
    report_artifacts: tuple[Path, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    scores, _ = report_artifacts
    summary_path = scores / "summary.json"
    summary = json.loads(summary_path.read_text())
    overall = summary["engines"]["test-model"]["all"]
    overall["asr_files"]["output.wav:ok"] = 1
    overall["asr_files"]["output.wav:not_run"] = 1
    overall["asr_files"]["extra.wav:ok"] = 100
    overall["timing_supplementary_clean"]["status"] = {"not_run": 1, "ineligible": 1}
    write_json(summary_path, summary)
    assert (
        benchmark_duplex_reference.main(
            ["report", "--scores", str(scores), "--engine", "test-model"]
        )
        == 0
    )
    output = "\n".join(
        " ".join(line.split()) for line in capsys.readouterr().out.splitlines()
    )
    assert "Successful ASR file roles 5 / 6 eligible / 8 selected" in output
    assert "Successful timing results 2 / 3 eligible / 4 selected" in output
    assert "not_run" in output
    assert "Offline replay agreement not supplied" in output
    assert "Reply WER not available in supplied artifacts" in output
    assert "Concurrent-session capacity not available in supplied artifacts" in output
    assert "all / overlap / stop: n/a, n/a, n/a; 0, 2, 2" in output


def test_report_keeps_custom_quality_separate_from_official_behavior(
    report_artifacts: tuple[Path, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    scores, _ = report_artifacts
    semantic_summary = scores.parent / "semantic-summary.json"
    resolved_axis = {
        "selected": 2,
        "accepted": 1,
        "failed": 1,
        "unresolved": 0,
        "quality_percent": 50.0,
        "coverage_percent": 100.0,
        "sampling_ci95_percent": [0.0, 100.0],
    }
    unresolved_axis = {
        "selected": 2,
        "accepted": 0,
        "failed": 0,
        "unresolved": 2,
        "quality_percent": None,
        "coverage_percent": 0.0,
        "sampling_ci95_percent": None,
    }
    write_json(
        semantic_summary,
        {
            "scope": "Custom transcript judgments; not official FDB behavior labels.",
            "inputs_sha256": "a" * 64,
            "uncertainty_note": "One fixed generation; no judge error calibration.",
            "selected_pairs": 2,
            "eligible_pairs": 1,
            "overall_axes": {
                "interaction_handling": resolved_axis,
                "relevance": resolved_axis,
                "grounding": unresolved_axis,
            },
            "overall_joint": unresolved_axis,
        },
    )
    assert (
        benchmark_duplex_reference.main(
            [
                "report",
                "--scores",
                str(scores),
                "--engine",
                "test-model",
                "--semantic-summary",
                str(semantic_summary),
            ]
        )
        == 0
    )
    output = "\n".join(
        " ".join(line.split()) for line in capsys.readouterr().out.splitlines()
    )
    assert "Valid official behavior labels 0" in output
    assert "Custom semantic quality" in output
    assert "interaction handling quality 50.0%; coverage 100.0%; A/F/U 1/1/0" in output
    assert "grounding quality n/a; coverage 0.0%; A/F/U 0/0/2" in output
    assert "Separate supplied cohort" in output


@pytest.mark.parametrize("mismatch", ["manifest", "subset", "different_population"])
def test_report_rejects_mixed_manifest_and_summary_populations(
    report_artifacts: tuple[Path, Path],
    mismatch: Literal["manifest", "subset", "different_population"],
) -> None:
    scores, _ = report_artifacts
    if mismatch == "manifest":
        manifest_path = scores / "engines" / "test-model" / "projected-manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["samples"][0]["variants"]["clean"]["source"]["trace_sha256"] = "0" * 64
        write_json(manifest_path, manifest)
    else:
        summary_path = scores / "summary.json"
        summary = json.loads(summary_path.read_text())
        if mismatch == "subset":
            summary["engines"]["test-model"]["all"]["behavior"]["population"] = 1
        else:
            summary["populations"]["test-model"]["sample_ids"][
                0
            ] = "user_interruption/999"
        write_json(summary_path, summary)
    with pytest.raises(SystemExit) as error:
        benchmark_duplex_reference.main(
            ["report", "--scores", str(scores), "--engine", "test-model"]
        )
    assert error.value.code == 2


@pytest.mark.parametrize("mismatch", ["foreign", "duplicate"])
def test_report_rejects_replay_records_outside_selected_sessions(
    report_artifacts: tuple[Path, Path], mismatch: Literal["foreign", "duplicate"]
) -> None:
    scores, replay = report_artifacts
    receipt = json.loads(replay.read_text())
    if mismatch == "foreign":
        receipt["samples"][0]["sample"] = "user_interruption/999"
    else:
        receipt["samples"][1] = receipt["samples"][0]
    write_json(replay, receipt)
    with pytest.raises(SystemExit) as error:
        benchmark_duplex_reference.main(
            [
                "report",
                "--scores",
                str(scores),
                "--engine",
                "test-model",
                "--replay",
                str(replay),
            ]
        )
    assert error.value.code == 2


@pytest.mark.parametrize(
    "invalid_count", ["negative", "excess_successes", "behavior_mismatch"]
)
def test_report_rejects_impossible_scoring_counts(
    report_artifacts: tuple[Path, Path],
    invalid_count: Literal["negative", "excess_successes", "behavior_mismatch"],
) -> None:
    scores, _ = report_artifacts
    summary_path = scores / "summary.json"
    summary = json.loads(summary_path.read_text())
    overall = summary["engines"]["test-model"]["all"]
    if invalid_count == "negative":
        overall["asr_files"]["input.wav:ok"] = -1
    elif invalid_count == "excess_successes":
        overall["asr_files"]["clean_output.wav:ok"] = 2
    else:
        overall["behavior"]["valid_n"] = 1
    write_json(summary_path, summary)
    with pytest.raises(SystemExit) as error:
        benchmark_duplex_reference.main(
            ["report", "--scores", str(scores), "--engine", "test-model"]
        )
    assert error.value.code == 2
