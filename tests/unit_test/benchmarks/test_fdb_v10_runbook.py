# SPDX-License-Identifier: Apache-2.0
"""Full-Duplex-Bench v1.0 as the runbook's second dataset: selection, repeat
consistency and aggregation, without a GPU or a network."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmarks.duplex.fdb_v15.__main__ import build_parser, sample_selection
from benchmarks.duplex.fdb_v15.aggregate import render
from benchmarks.duplex.fdb_v15.asr import v10_asr
from benchmarks.duplex.fdb_v15.common import load_settings
from benchmarks.duplex.fdb_v15.judge import pending_v10_subsets
from benchmarks.duplex.fdb_v15.selection import (
    SampleSelection,
    check_matches_other_repeats,
    describe,
    ids_text,
    select_v10_samples,
)
from benchmarks.duplex.v10_dataset import SUBSETS as V10_SUBSETS

SAMPLE_NAMES = ("1", "2", "10", "3")
V15_IDS = ["user_interruption/1", "background_speech/7"]
V10_IDS = ["synthetic_pause_handling/1", "icc_backchannel/1"]
C_LABELS = ("C_RESPOND", "C_RESUME", "C_UNCERTAIN_HANDLING", "C_UNKNOWN")


def write_v10_dataset(root: Path) -> Path:
    for subset in V10_SUBSETS:
        for name in SAMPLE_NAMES:
            (root / subset / name).mkdir(parents=True)
    return root


def test_v10_selection_takes_the_first_samples_per_subset_in_numeric_order(
    tmp_path: Path,
) -> None:
    dataset = write_v10_dataset(tmp_path / "v1.0")
    selection = SampleSelection(
        per_subset=2, subset_counts={"icc_backchannel": 0, "candor_turn_taking": 3}
    )
    sample_ids = select_v10_samples(dataset, selection)
    assert sample_ids == [
        "synthetic_pause_handling/1",
        "synthetic_pause_handling/2",
        "candor_pause_handling/1",
        "candor_pause_handling/2",
        "candor_turn_taking/1",
        "candor_turn_taking/2",
        "candor_turn_taking/3",
        "synthetic_user_interruption/1",
        "synthetic_user_interruption/2",
    ]
    assert describe(sample_ids, V10_SUBSETS) == (
        "synthetic_pause_handling 2, candor_pause_handling 2, "
        "candor_turn_taking 3, synthetic_user_interruption 2"
    )
    only_backchannel = SampleSelection(
        per_subset=None,
        subset_counts={
            subset: 0 for subset in V10_SUBSETS if subset != "icc_backchannel"
        },
    )
    assert select_v10_samples(dataset, only_backchannel) == [
        "icc_backchannel/1",
        "icc_backchannel/2",
        "icc_backchannel/3",
        "icc_backchannel/10",
    ]
    # note (luojiaxuan): every count at 0 leaves v1.0 out without reading its dataset.
    assert select_v10_samples(tmp_path / "missing", SampleSelection(per_subset=0)) == []


def parse_selection(*arguments: str) -> tuple[SampleSelection, SampleSelection]:
    parser = build_parser()
    return sample_selection(parser, parser.parse_args(["generate", *arguments]))


def test_v10_count_follows_per_subset_unless_set_or_pairs_are_explicit() -> None:
    v15, v10 = parse_selection("--per-subset", "3")
    assert (v15.per_subset, v10.per_subset, v10.subset_counts) == (3, 3, {})
    _, v10 = parse_selection("--per-subset", "all")
    assert v10.per_subset is None
    _, v10 = parse_selection("--sample-id", "user_interruption/1")
    assert v10.per_subset == 0
    _, v10 = parse_selection(
        "--sample-id", "user_interruption/1", "--v10-per-subset", "4"
    )
    assert v10.per_subset == 4
    _, v10 = parse_selection("--v10-per-subset", "0")
    assert v10.per_subset == 0
    v15, v10 = parse_selection(
        "--v10-subset-count",
        "icc_backchannel=all",
        "--v10-subset-count",
        "candor_pause_handling=0",
    )
    assert v15.subset_counts == {}
    assert v10.subset_counts == {"icc_backchannel": None, "candor_pause_handling": 0}
    with pytest.raises(SystemExit):
        parse_selection("--v10-subset-count", "user_interruption=3")


def write_repeat(run_root: Path, repeat: int, v10_ids: list[str] | None) -> Path:
    repeat_dir = run_root / f"repeat-{repeat}"
    repeat_dir.mkdir(parents=True)
    (repeat_dir / "sample-ids.txt").write_text(ids_text(V15_IDS))
    if v10_ids is not None:
        (repeat_dir / "v10").mkdir()
        (repeat_dir / "v10" / "sample-ids.txt").write_text(ids_text(v10_ids))
    return repeat_dir


def test_repeats_must_select_the_same_v10_samples(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    write_repeat(run_root, 1, V10_IDS)
    repeat_2 = run_root / "repeat-2"
    check_matches_other_repeats(repeat_2, V15_IDS, V10_IDS)
    with pytest.raises(SystemExit, match="different v1.0 samples"):
        check_matches_other_repeats(repeat_2, V15_IDS, V10_IDS[:1])
    with pytest.raises(SystemExit, match="different v1.0 samples"):
        check_matches_other_repeats(repeat_2, V15_IDS, [])
    with pytest.raises(SystemExit, match="different pairs"):
        check_matches_other_repeats(repeat_2, V15_IDS[:1], V10_IDS)
    # note (luojiaxuan): a repeat without v1.0 only matches repeats without v1.0.
    v15_only = tmp_path / "v15-only"
    write_repeat(v15_only, 1, None)
    check_matches_other_repeats(v15_only / "repeat-2", V15_IDS, [])
    with pytest.raises(SystemExit, match="different v1.0 samples"):
        check_matches_other_repeats(v15_only / "repeat-2", V15_IDS, V10_IDS)


def write_v15_scores(repeat_dir: Path, stop_s: float) -> None:
    timing = {
        "population": 2,
        "status": {"ok": 2},
        "official_all_intervals": {
            "stop": {"mean_s": stop_s},
            "response": {"mean_s": 2.5},
        },
    }
    (repeat_dir / "scores").mkdir(parents=True)
    (repeat_dir / "scores" / "summary.json").write_text(
        json.dumps(
            {"engines": {"minicpmo": {"all": {"timing_official_overlap": timing}}}}
        )
    )
    behavior = {
        "valid_n": 2,
        "valid_label_proportions": {label: {"proportion": 0.25} for label in C_LABELS},
    }
    (repeat_dir / "judge-qwen").mkdir()
    (repeat_dir / "judge-qwen" / "summary.json").write_text(
        json.dumps({"engines": {"minicpmo": {"all": behavior}}})
    )


def write_v10_summary(repeat_dir: Path, takeover: float, latency_s: float) -> None:
    tree = repeat_dir / "v10" / "reference"
    tree.mkdir(parents=True)
    subsets = {
        "candor_turn_taking": {
            "task": "smooth_turn_taking",
            "selected": 3,
            "evaluated": 3,
            "result": {"Average take turn": takeover, "Average latency": latency_s},
        },
        "icc_backchannel": {
            "task": "backchannel",
            "selected": 2,
            "evaluated": 2,
            "result": {
                "JSD mean": 0.7,
                "JSD std": 0.1,
                "TOR mean": 0.5,
                "TOR std": 0.5,
                "Frequency mean": 0.07,
                "Frequency std": 0.02,
                "Number of samples": 2.0,
            },
        },
    }
    (tree / "summary.json").write_text(json.dumps({"subsets": subsets}))


def test_results_append_a_separate_v10_table_only_when_a_run_has_v10(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "run"
    for repeat, stop_s in ((1, 1.0), (2, 2.0)):
        write_v15_scores(run_root / f"repeat-{repeat}", stop_s)
    v15_only = render(run_root, "minicpmo", "qwen")
    assert "v1.0" not in v15_only
    assert "| all | 2 | 2 | 1.500 ± 0.707 | 2.500 ± 0.000 |" in v15_only

    write_v10_summary(run_root / "repeat-1", takeover=0.5, latency_s=1.0)
    write_v10_summary(run_root / "repeat-2", takeover=1.0, latency_s=2.0)
    results = render(run_root, "minicpmo", "qwen")
    assert results.startswith(v15_only)
    lines = results.splitlines()
    assert "## FDB v1.0 results" in lines
    assert (
        "| candor_turn_taking | 3 | 3 | 75.0 ± 35.4% | 1.500 ± 0.707 | n/a | n/a | n/a |"
        in lines
    )
    assert (
        "| icc_backchannel | 2 | 2 | 50.0 ± 0.0% | n/a | n/a | 0.070 ± 0.000 | 0.700 ± 0.000 |"
        in lines
    )
    assert not any(line.startswith("| synthetic_pause_handling") for line in lines)


def test_pending_v10_subsets_skip_unexported_and_evaluated_ones(tmp_path: Path) -> None:
    tree = tmp_path / "reference"
    tree.mkdir()
    counts = {subset: {"selected": 1, "eligible": 1} for subset in V10_SUBSETS}
    counts["candor_turn_taking"] = {"selected": 1, "eligible": 0}
    counts["icc_backchannel"] = {"selected": 0, "eligible": 0}
    (tree / "manifest.json").write_text(json.dumps({"counts": counts}))
    assert pending_v10_subsets(tree) == [
        "synthetic_pause_handling",
        "candor_pause_handling",
        "synthetic_user_interruption",
    ]
    (tree / "summary.json").write_text(
        json.dumps({"subsets": {"synthetic_pause_handling": {}}})
    )
    assert pending_v10_subsets(tree) == [
        "candor_pause_handling",
        "synthetic_user_interruption",
    ]


def test_v10_asr_runs_once_per_export(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for name in ("CUDA_VISIBLE_DEVICES", "GPU", "SERVER_PORT", "JUDGE_PORT"):
        monkeypatch.delenv(name, raising=False)
    tree = tmp_path / "reference"
    tree.mkdir()
    (tree / "asr.json").write_text("{}")
    assert v10_asr(load_settings("v10"), tree) is True
