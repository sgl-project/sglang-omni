# SPDX-License-Identifier: Apache-2.0
"""Full-Duplex-Bench v1.0 as the runbook's second dataset: selection, gates,
command composition, repeat consistency and aggregation, without a GPU or a
network."""

from __future__ import annotations

import contextlib
import json
import shutil
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest

from benchmarks.duplex.fdb_v15 import asr as asr_module
from benchmarks.duplex.fdb_v15 import generate as generate_module
from benchmarks.duplex.fdb_v15 import judge as judge_module
from benchmarks.duplex.fdb_v15.__main__ import build_parser, sample_selection
from benchmarks.duplex.fdb_v15.aggregate import render
from benchmarks.duplex.fdb_v15.common import (
    PARAKEET_SHA256,
    RECORD_PROFILE,
    Settings,
    load_settings,
)
from benchmarks.duplex.fdb_v15.selection import (
    SampleSelection,
    check_matches_other_repeats,
    describe,
    ids_text,
    select_v10_samples,
)
from benchmarks.duplex.v10_dataset import SUBSETS as V10_SUBSETS
from benchmarks.duplex.v15_dataset import SUBSETS as V15_SUBSETS
from benchmarks.eval import benchmark_duplex_v10

SAMPLE_NAMES = ("1", "2", "10", "3")
V15_IDS = ["user_interruption/1", "background_speech/7"]
V10_IDS = ["synthetic_pause_handling/1", "icc_backchannel/1"]
C_LABELS = ("C_RESPOND", "C_RESUME", "C_UNCERTAIN_HANDLING", "C_UNKNOWN")
V10_REVISION = "sha256:fixture"


def write_v10_dataset(root: Path) -> Path:
    for subset in V10_SUBSETS:
        for name in SAMPLE_NAMES:
            (root / subset / name).mkdir(parents=True)
    return root


@pytest.fixture
def settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Settings:
    for name in ("CUDA_VISIBLE_DEVICES", "GPU", "SERVER_PORT", "JUDGE_PORT"):
        monkeypatch.delenv(name, raising=False)
    for name in ("SESSION_TIMEOUT_S", "V10_SESSION_TIMEOUT_S"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("FDB_WORK", str(tmp_path / "fdb"))
    return load_settings("v10")


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
    explicit = SampleSelection(
        per_subset=12, sample_ids=["icc_backchannel/10", "candor_turn_taking/2"]
    )
    assert select_v10_samples(dataset, explicit) == [
        "candor_turn_taking/2",
        "icc_backchannel/10",
    ]
    # note (luojiaxuan): every count at 0 leaves v1.0 out without reading its dataset.
    assert select_v10_samples(tmp_path / "missing", SampleSelection(per_subset=0)) == []


def test_v10_selection_refuses_unstaged_subsets(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="setup"):
        select_v10_samples(tmp_path / "missing", SampleSelection(per_subset=12))
    dataset = write_v10_dataset(tmp_path / "v1.0")
    shutil.rmtree(dataset / "candor_turn_taking")
    with pytest.raises(SystemExit, match="candor_turn_taking"):
        select_v10_samples(dataset, SampleSelection(per_subset=1))
    assert select_v10_samples(
        dataset, SampleSelection(per_subset=1, subset_counts={"candor_turn_taking": 0})
    ) == [f"{subset}/1" for subset in V10_SUBSETS if subset != "candor_turn_taking"]


def parse_selection(*arguments: str) -> tuple[SampleSelection, SampleSelection]:
    parser = build_parser()
    return sample_selection(parser, parser.parse_args(["generate", *arguments]))


def test_v10_count_follows_per_subset_unless_set_or_pairs_are_explicit(
    tmp_path: Path,
) -> None:
    v15, v10 = parse_selection("--per-subset", "3")
    assert (v15.per_subset, v10.per_subset, v10.subset_counts) == (3, 3, {})
    assert v10.sample_ids is None
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
    _, v10 = parse_selection("--v10-sample-id", "icc_backchannel/3")
    assert v10.sample_ids == ["icc_backchannel/3"]
    ids_file = tmp_path / "v10-ids.txt"
    ids_file.write_text(ids_text(V10_IDS))
    _, v10 = parse_selection("--v10-sample-ids-file", str(ids_file))
    assert v10.sample_ids == V10_IDS
    with pytest.raises(SystemExit):
        parse_selection("--v10-subset-count", "user_interruption=3")
    with pytest.raises(SystemExit):
        parse_selection("--v10-sample-id", "icc_backchannel/3", "--v10-per-subset", "1")
    with pytest.raises(SystemExit):
        parse_selection(
            "--v10-sample-id",
            "icc_backchannel/3",
            "--v10-subset-count",
            "icc_backchannel=2",
        )


def test_session_timeouts_stay_within_the_recorder_ceiling(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert (settings.session_timeout_s, settings.v10_session_timeout_s) == ("90", "230")
    monkeypatch.setenv("V10_SESSION_TIMEOUT_S", "300")
    with pytest.raises(SystemExit, match="ceiling"):
        load_settings("v10")
    monkeypatch.setenv("V10_SESSION_TIMEOUT_S", "120")
    monkeypatch.setenv("SESSION_TIMEOUT_S", "soon")
    with pytest.raises(SystemExit, match="SESSION_TIMEOUT_S"):
        load_settings("v10")


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
    # note (luojiaxuan): a repeat without v10/ is a v1.5-only repeat; the message
    # names the option that continues such a run.
    v15_only = tmp_path / "v15-only"
    write_repeat(v15_only, 1, None)
    check_matches_other_repeats(v15_only / "repeat-2", V15_IDS, [])
    with pytest.raises(SystemExit, match="v1.5-only repeat.*--v10-per-subset 0"):
        check_matches_other_repeats(v15_only / "repeat-2", V15_IDS, V10_IDS)


def stage_v15_dataset(settings: Settings) -> None:
    for subset in V15_SUBSETS:
        (settings.dataset / subset / "1").mkdir(parents=True)
    settings.dataset_revision_file.write_text("sha256:v15\n")


@contextlib.contextmanager
def no_server(settings: Settings, log_file: Path) -> Iterator[None]:
    yield


def test_generate_without_v10_writes_no_v10_directory_and_fails_before_the_server(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    stage_v15_dataset(settings)
    write_v10_dataset(settings.dataset_v10)
    servers = []
    monkeypatch.setattr(
        generate_module,
        "model_server",
        lambda *arguments: servers.append(arguments) or no_server(*arguments),
    )
    monkeypatch.setattr(generate_module, "record_shards", lambda *_: [])
    monkeypatch.setattr(generate_module, "export_reference_audio", lambda *_: None)
    v15_only = SampleSelection(per_subset=1)
    generate_module.generate(settings, 1, v15_only, SampleSelection(per_subset=0), 1)
    repeat_dir = settings.repeat_dir(1)
    assert (repeat_dir / "sample-ids.txt").read_text() == ids_text(
        [f"{subset}/1" for subset in V15_SUBSETS]
    )
    assert not (repeat_dir / "v10").exists()
    assert len(servers) == 1

    # note (luojiaxuan): a selected v1.0 without its pinned revision stops before
    # any session is recorded.
    other = load_settings("other")
    with pytest.raises(SystemExit, match="v1.0.revision.*setup"):
        generate_module.generate(other, 1, v15_only, SampleSelection(per_subset=1), 1)
    assert len(servers) == 1
    assert not (other.repeat_dir(1) / "recording").exists()


def run_v10_cli(argv: list[str], handler: str, monkeypatch: pytest.MonkeyPatch):
    """Parse a composed v1.0 command with the real CLI; the handler only captures."""
    captured = []
    monkeypatch.setattr(
        benchmark_duplex_v10, handler, lambda args: captured.append(args) or 0
    )
    assert argv[1:3] == ["-m", "benchmarks.eval.benchmark_duplex_v10"]
    assert benchmark_duplex_v10.main(argv[3:]) == 0
    return captured[0]


def test_v10_record_and_export_commands_parse_with_the_v10_cli(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings.dataset_v10_revision_file.parent.mkdir(parents=True)
    settings.dataset_v10_revision_file.write_text(f"{V10_REVISION}\n")
    record = generate_module.v10_record_command(settings)
    assert record[0] == sys.executable
    # note (luojiaxuan): record_shards appends the output and the shard's samples.
    shard = [
        *record,
        "--output",
        str(settings.repeat_dir(1)),
        "--sample-id",
        V10_IDS[0],
    ]
    args = run_v10_cli(shard, "record", monkeypatch)
    assert args.sample_id == [V10_IDS[0]]
    assert args.profile == RECORD_PROFILE
    assert args.dataset_root == settings.dataset_v10
    assert args.dataset_revision == V10_REVISION
    assert args.url == settings.realtime_url
    assert args.model_revision == settings.model_revision
    assert args.timeout == 230.0

    commands = []
    monkeypatch.setattr(
        generate_module, "run_command", lambda command: commands.append(command) or True
    )
    v10_dir = settings.repeat_dir(1) / "v10"
    shards = [v10_dir / "recording" / "shard-0", v10_dir / "recording" / "shard-1"]
    generate_module.export_v10_reference(settings, v10_dir, shards)
    args = run_v10_cli(commands[0], "reference_export", monkeypatch)
    assert args.run == shards
    assert args.dataset_root == settings.dataset_v10
    assert args.out == v10_dir / "reference"


def test_v10_asr_composes_reference_asr_once_per_export(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = []
    monkeypatch.setattr(
        asr_module,
        "run_command",
        lambda command, visible_gpus=None: calls.append((command, visible_gpus))
        or True,
    )
    tree = settings.repeat_dir(1) / "v10" / "reference"
    tree.mkdir(parents=True)
    assert asr_module.v10_asr(settings, tree) is True
    command, visible_gpus = calls[0]
    assert visible_gpus == settings.gpu
    assert command[0] == str(settings.scoring_python)
    args = run_v10_cli(command, "reference_asr", monkeypatch)
    assert args.tree == tree
    assert args.reference_source == settings.fdb_source
    assert args.nemo == settings.parakeet_nemo
    assert args.nemo_sha256 == PARAKEET_SHA256
    assert args.device == "cuda"

    (tree / "asr.json").write_text("{}")
    assert asr_module.v10_asr(settings, tree) is True
    assert len(calls) == 1


def test_evaluate_v10_runs_each_subset_on_its_own_and_reports_failures(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = []

    def run(command: list[str], extra_env: dict[str, str] | None = None) -> bool:
        calls.append((command, extra_env))
        return "candor_turn_taking" not in command

    monkeypatch.setattr(judge_module, "run_command", run)
    tree = settings.repeat_dir(1) / "v10" / "reference"
    judge_arguments = [
        "--api-key-env",
        "CUSTOM_JUDGE_API_KEY",
        "--base-url",
        settings.judge_url,
        "--served-model",
        "qwen3.8-27b",
    ]
    failed = judge_module.evaluate_v10(
        settings,
        tree,
        ["candor_turn_taking", "synthetic_user_interruption", "icc_backchannel"],
        judge_arguments,
        {"CUSTOM_JUDGE_API_KEY": "EMPTY"},
    )
    assert failed == ["candor_turn_taking"]
    assert len(calls) == 3
    command, extra_env = calls[1]
    assert extra_env == {"CUSTOM_JUDGE_API_KEY": "EMPTY"}
    assert command[0] == str(settings.scoring_python)
    args = run_v10_cli(command, "reference_evaluate", monkeypatch)
    assert args.subset == ["synthetic_user_interruption"]
    assert args.tree == tree
    assert args.reference_source == settings.fdb_source
    assert (args.api_key_env, args.base_url, args.served_model) == (
        "CUSTOM_JUDGE_API_KEY",
        settings.judge_url,
        "qwen3.8-27b",
    )
    assert judge_module.evaluate_v10(settings, tree, [], judge_arguments) == []
    assert len(calls) == 3


def test_asr_and_judge_refuse_a_selected_v10_without_its_artifacts(
    settings: Settings,
) -> None:
    repeat_dir = settings.repeat_dir(1)
    (repeat_dir / "reference-audio").mkdir(parents=True)
    (repeat_dir / "reference-audio" / "reference-manifest.json").write_text("{}")
    receipt = repeat_dir / "scores" / "engines" / "minicpmo" / "manifest-receipt.json"
    receipt.parent.mkdir(parents=True)
    receipt.write_text("{}")
    (repeat_dir / "v10").mkdir()
    (repeat_dir / "v10" / "sample-ids.txt").write_text(ids_text(V10_IDS))
    with pytest.raises(SystemExit, match="did not finish the v1.0 export"):
        asr_module.asr(settings, 1, False)
    with pytest.raises(SystemExit, match="did not finish the v1.0 export"):
        judge_module.judge(settings, 1, False)
    tree = repeat_dir / "v10" / "reference"
    tree.mkdir()
    (tree / "manifest.json").write_text("{}")
    with pytest.raises(SystemExit, match="asr.*asr.json is missing"):
        judge_module.judge(settings, 1, False)


def test_pending_v10_subsets_skip_unexported_and_evaluated_ones(tmp_path: Path) -> None:
    tree = tmp_path / "reference"
    tree.mkdir()
    counts = {subset: {"selected": 1, "eligible": 1} for subset in V10_SUBSETS}
    counts["candor_turn_taking"] = {"selected": 1, "eligible": 0}
    counts["icc_backchannel"] = {"selected": 0, "eligible": 0}
    (tree / "manifest.json").write_text(json.dumps({"counts": counts}))
    assert judge_module.pending_v10_subsets(tree) == [
        "synthetic_pause_handling",
        "candor_pause_handling",
        "synthetic_user_interruption",
    ]
    (tree / "summary.json").write_text(
        json.dumps({"subsets": {"synthetic_pause_handling": {}}})
    )
    assert judge_module.pending_v10_subsets(tree) == [
        "candor_pause_handling",
        "synthetic_user_interruption",
    ]


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


V10_COUNTS = {
    "synthetic_pause_handling": {"selected": 2, "eligible": 2},
    "candor_pause_handling": {"selected": 0, "eligible": 0},
    "candor_turn_taking": {"selected": 3, "eligible": 3},
    "synthetic_user_interruption": {"selected": 2, "eligible": 2},
    "icc_backchannel": {"selected": 2, "eligible": 0},
}


def write_v10_tree(
    repeat_dir: Path, takeover: float, latency_s: float, retries: int
) -> None:
    tree = repeat_dir / "v10" / "reference"
    tree.mkdir(parents=True)
    (tree / "manifest.json").write_text(json.dumps({"counts": V10_COUNTS}))
    subsets = {
        "synthetic_pause_handling": {
            "task": "pause_handling",
            "selected": 2,
            "evaluated": 2,
            "result": {"Average take turn": 0.5},
            "judge": None,
        },
        "candor_turn_taking": {
            "task": "smooth_turn_taking",
            "selected": 3,
            "evaluated": 3,
            "result": {"Average take turn": takeover, "Average latency": latency_s},
            "judge": None,
        },
        "synthetic_user_interruption": {
            "task": "user_interruption",
            "selected": 2,
            "evaluated": 2,
            "result": {
                "Average rating": 4.5,
                "Average take turn": 1.0,
                "Average latency": 0.2,
            },
            "judge": {"official": False, "retries": retries},
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

    write_v10_tree(run_root / "repeat-1", takeover=0.5, latency_s=1.0, retries=0)
    write_v10_tree(run_root / "repeat-2", takeover=1.0, latency_s=2.0, retries=1)
    results = render(run_root, "minicpmo", "qwen")
    assert results.startswith(v15_only)
    lines = results.splitlines()
    assert "## FDB v1.0 results" in lines
    assert not any("v1.0 cells come from" in line for line in lines)
    assert (
        "| synthetic_pause_handling | 2 | 2 | 2 | 50.0 ± 0.0% | n/a | n/a | n/a | n/a | n/a |"
        in lines
    )
    assert (
        "| candor_turn_taking | 3 | 3 | 3 | 75.0 ± 35.4% | 1.500 ± 0.707 | n/a | n/a | n/a | n/a |"
        in lines
    )
    assert (
        "| synthetic_user_interruption | 2 | 2 | 2 | 100.0 ± 0.0% | 0.200 ± 0.000 | "
        "4.500 ± 0.000 | 0-1 | n/a | n/a |" in lines
    )
    # note (luojiaxuan): a subset whose sessions all failed keeps its row.
    assert (
        "| icc_backchannel | 2 | 0 | 0 | n/a | n/a | n/a | n/a | n/a | n/a |" in lines
    )
    assert not any(line.startswith("| candor_pause_handling") for line in lines)


def test_results_name_the_repeats_that_carry_v10(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    for repeat in (1, 2):
        write_v15_scores(run_root / f"repeat-{repeat}", 1.0)
    write_v10_tree(run_root / "repeat-2", takeover=1.0, latency_s=2.0, retries=0)
    results = render(run_root, "minicpmo", "qwen")
    assert "v1.0 cells come from repeat-2 only" in results
    assert (
        "| candor_turn_taking | 3 | 3 | 3 | 100.0% | 2.000 | n/a | n/a | n/a | n/a |"
        in (results.splitlines())
    )
