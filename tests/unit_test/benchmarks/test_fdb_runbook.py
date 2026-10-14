# SPDX-License-Identifier: Apache-2.0
"""Full-Duplex-Bench v1.0 as the runbook's second dataset: selection, gates,
command composition, phase orchestration, repeat consistency and aggregation,
without a GPU or a network."""

from __future__ import annotations

import contextlib
import json
import os
import shutil
import subprocess
import sys
import types
from collections.abc import Iterator
from pathlib import Path

import pytest

from benchmarks.duplex.fdb import asr as asr_module
from benchmarks.duplex.fdb import generate as generate_module
from benchmarks.duplex.fdb import judge as judge_module
from benchmarks.duplex.fdb.__main__ import build_parser, sample_selection
from benchmarks.duplex.fdb.aggregate import render
from benchmarks.duplex.fdb.common import (
    PARAKEET_SHA256,
    RECORD_PROFILE,
    Settings,
    load_settings,
    require_v10_export,
)
from benchmarks.duplex.fdb.selection import (
    SampleSelection,
    check_matches_other_repeats,
    describe,
    ids_text,
    leave_out_hint,
    select_v10_samples,
)
from benchmarks.duplex.fdb.servers import RunningServer
from benchmarks.duplex.reference_v10 import MANIFEST_KIND
from benchmarks.duplex.run_artifacts import file_sha256
from benchmarks.duplex.v10_dataset import SUBSETS as V10_SUBSETS
from benchmarks.duplex.v15_dataset import SUBSETS as V15_SUBSETS
from benchmarks.eval import benchmark_duplex_v10

SAMPLE_NAMES = ("1", "2", "10", "3")
V15_IDS = ["user_interruption/1", "background_speech/7"]
V10_IDS = ["synthetic_pause_handling/1", "icc_backchannel/1"]
C_LABELS = ("C_RESPOND", "C_RESUME", "C_UNCERTAIN_HANDLING", "C_UNKNOWN")
V10_REVISION = "sha256:fixture"
COUNT_HINT = "pass --v10-per-subset 0"
V10_CLI = "benchmarks.eval.benchmark_duplex_v10"
LIVE_SERVER = types.SimpleNamespace(ensure_alive=lambda when: None)
SERVER_LAST_WORDS = "Scheduler hit an exception"


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


def test_v10_selection_refuses_unstaged_subsets_with_a_hint_for_its_mode(
    tmp_path: Path,
) -> None:
    with pytest.raises(SystemExit, match=f"setup.*{COUNT_HINT}"):
        select_v10_samples(tmp_path / "missing", SampleSelection(per_subset=12))
    dataset = write_v10_dataset(tmp_path / "v1.0")
    shutil.rmtree(dataset / "candor_turn_taking")
    with pytest.raises(SystemExit, match=f"candor_turn_taking.*{COUNT_HINT}"):
        select_v10_samples(dataset, SampleSelection(per_subset=1))
    with pytest.raises(SystemExit, match="drop the --v10-subset-count options"):
        select_v10_samples(
            dataset,
            SampleSelection(per_subset=0, subset_counts={"candor_turn_taking": 2}),
        )
    with pytest.raises(
        SystemExit, match="drop the --v10-sample-id or --v10-sample-ids"
    ):
        select_v10_samples(
            dataset, SampleSelection(per_subset=12, sample_ids=["candor_turn_taking/1"])
        )
    assert select_v10_samples(
        dataset, SampleSelection(per_subset=1, subset_counts={"candor_turn_taking": 0})
    ) == [f"{subset}/1" for subset in V10_SUBSETS if subset != "candor_turn_taking"]


def parse_selection(*arguments: str) -> tuple[SampleSelection, SampleSelection]:
    parser = build_parser()
    return sample_selection(parser, parser.parse_args(["generate", *arguments]))


def test_each_dataset_has_its_own_selection_options(tmp_path: Path) -> None:
    v15, v10 = parse_selection()
    assert (v15.per_subset, v10.per_subset) == (12, 12)
    assert v15.sample_ids is None and v10.sample_ids is None
    v15, v10 = parse_selection("--v15-per-subset", "3")
    assert (v15.per_subset, v10.per_subset, v10.subset_counts) == (3, 12, {})
    v15, v10 = parse_selection("--v15-per-subset", "all", "--v10-per-subset", "0")
    assert (v15.per_subset, v10.per_subset) == (None, 0)
    v15, v10 = parse_selection("--v15-sample-id", "user_interruption/1")
    assert (v15.sample_ids, v10.per_subset) == (["user_interruption/1"], 12)
    v15, v10 = parse_selection(
        "--v15-subset-count",
        "user_interruption=20",
        "--v10-subset-count",
        "icc_backchannel=all",
        "--v10-subset-count",
        "candor_pause_handling=0",
    )
    assert v15.subset_counts == {"user_interruption": 20}
    assert v10.subset_counts == {"icc_backchannel": None, "candor_pause_handling": 0}
    _, v10 = parse_selection("--v10-sample-id", "icc_backchannel/3")
    assert v10.sample_ids == ["icc_backchannel/3"]
    ids_file = tmp_path / "v10-ids.txt"
    ids_file.write_text(ids_text(V10_IDS))
    v15, v10 = parse_selection("--v10-sample-ids-file", str(ids_file))
    assert (v15.per_subset, v10.sample_ids) == (12, V10_IDS)
    # note (luojiaxuan): a subset name belongs to one dataset, and explicit IDs
    # exclude the count options of the same dataset.
    for arguments in (
        ("--v10-subset-count", "user_interruption=3"),
        ("--v15-subset-count", "icc_backchannel=3"),
        ("--v10-sample-id", "icc_backchannel/3", "--v10-per-subset", "1"),
        (
            "--v10-sample-id",
            "icc_backchannel/3",
            "--v10-subset-count",
            "icc_backchannel=2",
        ),
        ("--v15-sample-id", "user_interruption/1", "--v15-per-subset", "1"),
        (
            "--v15-sample-id",
            "user_interruption/1",
            "--v15-subset-count",
            "user_interruption=2",
        ),
    ):
        with pytest.raises(SystemExit):
            parse_selection(*arguments)


def test_leave_out_hint_names_an_option_change_the_parser_accepts() -> None:
    # note (luojiaxuan): the count-mode hint must parse next to a count selection;
    # the other two modes tell the user which option to drop.
    _, v10 = parse_selection("--v15-per-subset", "12")
    assert leave_out_hint(v10) == COUNT_HINT
    _, v10 = parse_selection("--v15-per-subset", "12", *COUNT_HINT.split()[1:])
    assert v10.per_subset == 0
    _, v10 = parse_selection(
        "--v10-per-subset", "0", "--v10-subset-count", "icc_backchannel=all"
    )
    assert leave_out_hint(v10) == "drop the --v10-subset-count options"
    _, v10 = parse_selection("--v10-sample-id", "icc_backchannel/1")
    assert (
        leave_out_hint(v10)
        == "drop the --v10-sample-id or --v10-sample-ids-file option"
    )


def test_env_exports_the_scoring_interpreter(settings: Settings) -> None:
    # note (luojiaxuan): the manual scoring commands run with this interpreter.
    assert settings.shell_exports()["SCORING_PYTHON"] == str(settings.scoring_python)


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
    check_matches_other_repeats(repeat_2, V15_IDS, V10_IDS, COUNT_HINT)
    with pytest.raises(SystemExit, match="different v1.0 samples"):
        check_matches_other_repeats(repeat_2, V15_IDS, V10_IDS[:1], COUNT_HINT)
    with pytest.raises(SystemExit, match="different v1.0 samples"):
        check_matches_other_repeats(repeat_2, V15_IDS, [], COUNT_HINT)
    with pytest.raises(SystemExit, match="different pairs"):
        check_matches_other_repeats(repeat_2, V15_IDS[:1], V10_IDS, COUNT_HINT)
    # note (luojiaxuan): a repeat without v10/ is a v1.5-only repeat; the message
    # carries the hint for the selection mode in use.
    v15_only = tmp_path / "v15-only"
    write_repeat(v15_only, 1, None)
    check_matches_other_repeats(v15_only / "repeat-2", V15_IDS, [], COUNT_HINT)
    with pytest.raises(SystemExit, match=f"v1.5-only repeat.*{COUNT_HINT}"):
        check_matches_other_repeats(v15_only / "repeat-2", V15_IDS, V10_IDS, COUNT_HINT)
    with pytest.raises(SystemExit, match="v1.5-only repeat.*drop the --v10-sample-id"):
        check_matches_other_repeats(
            v15_only / "repeat-2",
            V15_IDS,
            V10_IDS,
            "drop the --v10-sample-id or --v10-sample-ids-file option",
        )


def stage_datasets(settings: Settings) -> None:
    for subset in V15_SUBSETS:
        (settings.dataset / subset / "1").mkdir(parents=True)
    settings.dataset_revision_file.write_text("sha256:v15\n")
    write_v10_dataset(settings.dataset_v10)
    settings.dataset_v10_revision_file.write_text(f"{V10_REVISION}\n")


@contextlib.contextmanager
def no_server(settings: Settings, log_file: Path) -> Iterator[types.SimpleNamespace]:
    yield LIVE_SERVER


def stub_generate_phases(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[list[tuple[Path, Path]], list[tuple], list[tuple]]:
    """Replace the server and the recorder; the shard list a recording returns
    is one shard directory under the given root."""
    servers: list[tuple[Path, Path]] = []
    recordings: list[tuple] = []
    exports: list[tuple] = []
    monkeypatch.setattr(
        generate_module,
        "model_server",
        lambda *arguments: servers.append(arguments) or no_server(*arguments),
    )

    def record_shards(*arguments: object) -> list[Path]:
        recordings.append(arguments)
        return [arguments[1] / "recording" / "shard-0"]

    monkeypatch.setattr(generate_module, "record_shards", record_shards)
    monkeypatch.setattr(generate_module, "print_variant_status", lambda *_: None)
    monkeypatch.setattr(generate_module, "export_reference_audio", lambda *_: None)
    monkeypatch.setattr(
        generate_module,
        "export_v10_reference",
        lambda *arguments: exports.append(arguments),
    )
    return servers, recordings, exports


def test_generate_records_v10_after_v15_on_the_same_server(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    stage_datasets(settings)
    servers, recordings, exports = stub_generate_phases(monkeypatch)
    v15_ids = [f"{subset}/1" for subset in V15_SUBSETS]
    v10_ids = [f"{subset}/1" for subset in V10_SUBSETS]
    repeat_dir = settings.repeat_dir(1)
    v10_dir = repeat_dir / "v10"

    generate_module.generate(
        settings, 1, SampleSelection(per_subset=1), SampleSelection(per_subset=1), 2
    )

    assert len(servers) == 1
    assert (repeat_dir / "sample-ids.txt").read_text() == ids_text(v15_ids)
    assert (v10_dir / "sample-ids.txt").read_text() == ids_text(v10_ids)
    assert (v10_dir / "logs").is_dir()
    assert recordings == [
        (generate_module.v15_record_command(settings), repeat_dir, v15_ids, 2, 2),
        (generate_module.v10_record_command(settings), v10_dir, v10_ids, 1, 2),
    ]
    assert exports == [(settings, v10_dir, [v10_dir / "recording" / "shard-0"])]


def test_generate_without_v10_writes_no_v10_directory_and_fails_before_the_server(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    stage_datasets(settings)
    settings.dataset_v10_revision_file.unlink()
    servers, recordings, exports = stub_generate_phases(monkeypatch)
    v15_only = SampleSelection(per_subset=1)
    generate_module.generate(settings, 1, v15_only, SampleSelection(per_subset=0), 1)
    repeat_dir = settings.repeat_dir(1)
    assert (repeat_dir / "sample-ids.txt").read_text() == ids_text(
        [f"{subset}/1" for subset in V15_SUBSETS]
    )
    assert not (repeat_dir / "v10").exists()
    assert len(servers) == 1 and len(recordings) == 1 and exports == []

    # note (luojiaxuan): a selected v1.0 without its pinned revision stops before
    # any session is recorded.
    other = load_settings("other")
    with pytest.raises(SystemExit, match=f"v1.0.revision.*setup.*{COUNT_HINT}"):
        generate_module.generate(other, 1, v15_only, SampleSelection(per_subset=1), 1)
    assert len(servers) == 1
    assert not (other.repeat_dir(1) / "recording").exists()


def test_generate_with_v15_left_out_records_only_v10(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    stage_datasets(settings)
    servers, recordings, exports = stub_generate_phases(monkeypatch)
    v10_ids = [f"{subset}/1" for subset in V10_SUBSETS]
    repeat_dir = settings.repeat_dir(1)
    generate_module.generate(
        settings, 1, SampleSelection(per_subset=0), SampleSelection(per_subset=1), 1
    )
    assert (repeat_dir / "sample-ids.txt").read_text() == ""
    assert not (repeat_dir / "recording").exists()
    assert recordings == [
        (
            generate_module.v10_record_command(settings),
            repeat_dir / "v10",
            v10_ids,
            1,
            1,
        )
    ]
    assert len(servers) == 1 and len(exports) == 1
    # note (luojiaxuan): the v1.0 recording alone marks the repeat as generated.
    with pytest.raises(SystemExit, match="already holds a recording"):
        generate_module.generate(
            settings, 1, SampleSelection(per_subset=0), SampleSelection(per_subset=1), 1
        )
    with pytest.raises(SystemExit, match="both datasets are left out"):
        generate_module.generate(
            load_settings("empty"),
            1,
            SampleSelection(per_subset=0),
            SampleSelection(per_subset=0),
            1,
        )


def test_generate_discards_the_v10_marker_of_an_attempt_that_never_recorded(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    stage_datasets(settings)
    servers, _, _ = stub_generate_phases(monkeypatch)

    @contextlib.contextmanager
    def port_taken(*arguments: object) -> Iterator[None]:
        raise SystemExit("ERROR: something already serves the model port")
        yield

    monkeypatch.setattr(generate_module, "model_server", port_taken)
    v15_only = SampleSelection(per_subset=1)
    with pytest.raises(SystemExit, match="already serves"):
        generate_module.generate(
            settings, 1, v15_only, SampleSelection(per_subset=1), 1
        )
    repeat_dir = settings.repeat_dir(1)
    assert (repeat_dir / "v10" / "sample-ids.txt").is_file()
    assert not (repeat_dir / "recording").exists()

    monkeypatch.setattr(
        generate_module,
        "model_server",
        lambda *arguments: servers.append(arguments) or no_server(*arguments),
    )
    generate_module.generate(settings, 1, v15_only, SampleSelection(per_subset=0), 1)
    assert (repeat_dir / "recording").is_dir()
    assert not (repeat_dir / "v10").exists()
    assert require_v10_export(settings, 1) is None
    check_matches_other_repeats(
        settings.repeat_dir(2),
        [f"{subset}/1" for subset in V15_SUBSETS],
        [],
        COUNT_HINT,
    )


def test_generate_stops_with_the_server_log_when_the_model_server_exited(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    stage_datasets(settings)
    _, recordings, exports = stub_generate_phases(monkeypatch)

    @contextlib.contextmanager
    def model_server(settings: Settings, log_file: Path) -> Iterator[RunningServer]:
        with log_file.open("w") as log_handle:
            process = subprocess.Popen(
                [
                    sys.executable,
                    "-c",
                    f"print({SERVER_LAST_WORDS!r}); raise SystemExit(3)",
                ],
                stdout=log_handle,
            )
        process.wait()
        yield RunningServer("model server", process, log_file)

    monkeypatch.setattr(generate_module, "model_server", model_server)
    with pytest.raises(
        SystemExit,
        match=f"model server exited with code 3 during the recordings; "
        f"last lines of .*model-server.log:\n{SERVER_LAST_WORDS}",
    ):
        generate_module.generate(
            settings, 1, SampleSelection(per_subset=1), SampleSelection(per_subset=1), 1
        )
    assert len(recordings) == 2 and exports == []


def run_v10_cli(argv: list[str], handler: str, monkeypatch: pytest.MonkeyPatch):
    """Parse a composed v1.0 command with the real CLI; the handler only captures."""
    captured = []
    monkeypatch.setattr(
        benchmark_duplex_v10, handler, lambda args: captured.append(args) or 0
    )
    assert argv[1:3] == ["-m", V10_CLI]
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


def stage_scored_repeat(
    settings: Settings, v10_ids: list[str] | None, v15_ids: list[str] = V15_IDS
) -> Path:
    """A repeat after generate and asr: the v1.5 markers when v1.5 was selected,
    and the v1.0 export when v1.0 was."""
    repeat_dir = settings.repeat_dir(1)
    repeat_dir.mkdir(parents=True)
    (repeat_dir / "sample-ids.txt").write_text(ids_text(v15_ids))
    if v15_ids:
        (repeat_dir / "reference-audio").mkdir()
        (repeat_dir / "reference-audio" / "reference-manifest.json").write_text("{}")
        receipt = (
            repeat_dir / "scores" / "engines" / "minicpmo" / "manifest-receipt.json"
        )
        receipt.parent.mkdir(parents=True)
        receipt.write_text("{}")
    else:
        pass
    settings.judge_revision_file.parent.mkdir(parents=True, exist_ok=True)
    settings.judge_revision_file.write_text("0" * 40 + "\n")
    if v10_ids is not None:
        tree = repeat_dir / "v10" / "reference"
        tree.mkdir(parents=True)
        (repeat_dir / "v10" / "sample-ids.txt").write_text(ids_text(v10_ids))
        counts = {subset: {"selected": 0, "eligible": 0} for subset in V10_SUBSETS}
        for sample_id in v10_ids:
            counts[sample_id.split("/")[0]] = {"selected": 1, "eligible": 1}
        (tree / "manifest.json").write_text(json.dumps({"counts": counts}))
    else:
        pass
    return repeat_dir


def test_asr_runs_the_v10_pass_after_the_v15_phases_and_names_its_log(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = []
    failing = []

    def run(command: list[str], visible_gpus: str | None = None) -> bool:
        calls.append((command, visible_gpus))
        return command[3] not in failing

    monkeypatch.setattr(asr_module, "run_command", run)
    stage_scored_repeat(settings, V10_IDS)
    asr_module.asr(settings, 1, False)
    assert [command[3] for command, _ in calls] == [
        "asr",
        "timing",
        "reference-asr",
        "unified-score",
    ]
    assert calls[2][0][1:3] == ["-m", V10_CLI]
    assert calls[2][1] == settings.gpu
    assert calls[3][1] == ""
    assert calls[3][0][-2:] == [
        "--tree",
        str(settings.repeat_dir(1) / "v10" / "reference"),
    ]

    (settings.repeat_dir(1) / "v10" / "reference" / "asr.json").write_text("{}")
    calls.clear()
    asr_module.asr(settings, 1, False)
    assert [command[3] for command, _ in calls] == ["asr", "timing", "unified-score"]

    (settings.repeat_dir(1) / "v10" / "reference" / "asr.json").unlink()
    calls.clear()
    failing.append("reference-asr")
    with pytest.raises(
        SystemExit, match="the v1.0 ASR failed.*v10/reference/logs/asr.log"
    ):
        asr_module.asr(settings, 1, False)
    assert "unified-score" not in [command[3] for command, _ in calls]

    failing.clear()
    failing.append("unified-score")
    with pytest.raises(SystemExit, match="v1.0 unified scoring failed.*backfill"):
        asr_module.asr(settings, 1, False)
    assert [command[3] for command, _ in calls][-2:] == [
        "reference-asr",
        "unified-score",
    ]

    # note (luojiaxuan): a repeat generated with --v10-per-subset 0 has no v10/
    # and no v1.0 phase.
    calls.clear()
    v15_only = load_settings("v15-only")
    stage_scored_repeat(v15_only, None)
    asr_module.asr(v15_only, 1, False)
    assert [command[3] for command, _ in calls] == ["asr", "timing"]
    assert not any(V10_CLI in command for command, _ in calls)


def stub_judge_phases(
    monkeypatch: pytest.MonkeyPatch, failing_subset: str | None
) -> list:
    """Replace the judge server, the scoring subprocesses and the report writer."""
    calls = []
    monkeypatch.setattr(
        judge_module, "judge_server", lambda *_: contextlib.nullcontext(LIVE_SERVER)
    )

    def run(command: list[str], **_: object) -> bool:
        calls.append(command)
        return not (failing_subset is not None and failing_subset in command)

    monkeypatch.setattr(judge_module, "run_command", run)
    monkeypatch.setattr(
        judge_module.subprocess,
        "run",
        lambda *_, **__: types.SimpleNamespace(returncode=0),
    )
    return calls


def evaluated_subsets(calls: list[list[str]]) -> list[str]:
    return [
        command[command.index("--subset") + 1]
        for command in calls
        if "reference-evaluate" in command
    ]


def test_judge_evaluates_every_pending_subset_and_names_the_failed_ones(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    pending = ["candor_turn_taking/1", "icc_backchannel/1"]
    repeat_dir = stage_scored_repeat(settings, pending)
    (repeat_dir / "v10" / "reference" / "asr.json").write_text("{}")
    calls = stub_judge_phases(monkeypatch, failing_subset="candor_turn_taking")
    with pytest.raises(
        SystemExit, match="v1.0 evaluation failed for candor_turn_taking"
    ):
        judge_module.judge(settings, 1, False)
    assert evaluated_subsets(calls) == ["candor_turn_taking", "icc_backchannel"]
    assert (repeat_dir / "report.txt").is_file()


def test_judge_hands_the_pending_subsets_to_the_judge_phase(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    repeat_dir = stage_scored_repeat(
        settings, [*V10_IDS, "synthetic_user_interruption/1"]
    )
    tree = repeat_dir / "v10" / "reference"
    (tree / "asr.json").write_text("{}")
    (tree / "summary.json").write_text(
        json.dumps({"subsets": {"synthetic_pause_handling": {}}})
    )
    calls = stub_judge_phases(monkeypatch, failing_subset=None)
    handed = []
    monkeypatch.setattr(
        judge_module,
        "judge_with_qwen",
        lambda *arguments: handed.append(arguments[-2:]) or (True, []),
    )
    monkeypatch.setattr(judge_module, "write_report", lambda *_: True)
    judge_module.judge(settings, 1, False)
    # note (luojiaxuan): the judge-free subset is scored before the judge phase,
    # which gets only the interruption subset.
    assert evaluated_subsets(calls) == ["icc_backchannel"]
    assert handed == [(tree, ["synthetic_user_interruption"])]


def test_judge_runs_no_v10_phase_for_a_v15_only_repeat(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    stage_scored_repeat(settings, None)
    calls = stub_judge_phases(monkeypatch, failing_subset=None)
    judge_module.judge(settings, 1, False)
    assert calls
    assert not any(V10_CLI in command for command in calls)


@pytest.mark.parametrize(
    ("v10_ids", "exits_during", "stopped_at"),
    [
        (
            V10_IDS,
            "benchmarks.duplex.semantic_judge",
            "during the v1.5 semantic A/F/U judge",
        ),
        (
            None,
            "benchmarks.duplex.semantic_judge",
            "during the v1.5 semantic A/F/U judge",
        ),
        (V10_IDS, "custom-summarize", "before the v1.5 semantic A/F/U judge"),
        (
            [*V10_IDS, "synthetic_user_interruption/1"],
            "synthetic_user_interruption",
            "during the v1.0 evaluation of synthetic_user_interruption",
        ),
    ],
)
def test_judge_stops_with_the_server_log_once_the_judge_server_exits(
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
    v10_ids: list[str] | None,
    exits_during: str,
    stopped_at: str,
) -> None:
    repeat_dir = stage_scored_repeat(settings, v10_ids)
    if v10_ids is not None:
        (repeat_dir / "v10" / "reference" / "asr.json").write_text("{}")
    else:
        pass
    stub_judge_phases(monkeypatch, failing_subset=None)
    processes = []

    @contextlib.contextmanager
    def judge_server(settings: Settings, log_file: Path) -> Iterator[RunningServer]:
        # note (luojiaxuan): a real process stands in for the server; closing
        # its stdin makes it log a last line and exit with code 1.
        log_file.parent.mkdir(parents=True, exist_ok=True)
        with log_file.open("w") as log_handle:
            process = subprocess.Popen(
                [
                    sys.executable,
                    "-c",
                    f"import sys; sys.stdin.read(); print({SERVER_LAST_WORDS!r}); "
                    "sys.exit(1)",
                ],
                stdin=subprocess.PIPE,
                stdout=log_handle,
            )
        processes.append(process)
        try:
            yield RunningServer("judge server", process, log_file)
        finally:
            process.stdin.close()
            process.wait()

    calls = []

    def run(command: list[str], **_: object) -> bool:
        calls.append(command)
        if exits_during in command:
            processes[0].stdin.close()
            processes[0].wait()
        else:
            pass
        # note (luojiaxuan): against an exited server the v1.0 evaluation fails,
        # and the semantic judge exits 0 with its requests unresolved.
        return "reference-evaluate" not in command

    monkeypatch.setattr(judge_module, "judge_server", judge_server)
    monkeypatch.setattr(judge_module, "run_command", run)
    with pytest.raises(SystemExit) as stopped:
        judge_module.judge(settings, 1, False)
    log_file = repeat_dir / "logs" / "judge-server.log"
    assert str(stopped.value) == (
        f"ERROR: the judge server exited with code 1 {stopped_at}; "
        f"last lines of {log_file}:\n{SERVER_LAST_WORDS}"
    )
    assert exits_during in calls[-1]


def test_asr_and_judge_run_only_the_v10_phases_for_a_v10_only_repeat(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    repeat_dir = stage_scored_repeat(settings, V10_IDS, v15_ids=[])
    asr_calls = []
    monkeypatch.setattr(
        asr_module,
        "run_command",
        lambda command, **_: asr_calls.append(command) or True,
    )
    asr_module.asr(settings, 1, False)
    assert [command[3] for command in asr_calls] == ["reference-asr", "unified-score"]
    (repeat_dir / "v10" / "reference" / "asr.json").write_text("{}")
    judge_calls = stub_judge_phases(monkeypatch, failing_subset=None)
    judge_module.judge(settings, 1, False)
    assert [command[3] for command in judge_calls] == ["reference-evaluate"] * 2
    assert not (repeat_dir / "report.txt").exists()
    # note (luojiaxuan): a repeat that was never generated has no sample-ids.txt.
    other = load_settings("other")
    with pytest.raises(SystemExit, match="generate --run-name other"):
        asr_module.asr(other, 1, False)


@pytest.mark.parametrize("judge_name", ["qwen", "gpt"])
def test_judge_starts_no_judge_when_no_llm_request_is_pending(
    judge_name: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("JUDGE", judge_name)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("FDB_WORK", str(tmp_path / "fdb"))
    for name in ("CUDA_VISIBLE_DEVICES", "GPU", "SERVER_PORT", "JUDGE_PORT"):
        monkeypatch.delenv(name, raising=False)
    settings = load_settings("v10")
    no_llm = [
        "synthetic_pause_handling/1",
        "candor_turn_taking/1",
        "icc_backchannel/1",
    ]
    repeat_dir = stage_scored_repeat(settings, no_llm, v15_ids=[])
    (repeat_dir / "v10" / "reference" / "asr.json").write_text("{}")
    calls = stub_judge_phases(monkeypatch, failing_subset=None)
    monkeypatch.setattr(
        judge_module,
        "judge_server",
        lambda *_: pytest.fail("the judge server started with no LLM request"),
    )
    judge_module.judge(settings, 1, False)
    assert evaluated_subsets(calls) == [
        "synthetic_pause_handling",
        "candor_turn_taking",
        "icc_backchannel",
    ]
    assert all("--base-url" not in command for command in calls)

    # note (luojiaxuan): the interruption subset asks the judge, so the judge is
    # needed again: a server with qwen, a key with gpt.
    interruption = stage_scored_repeat(
        load_settings("v10-interruption"),
        ["synthetic_user_interruption/1"],
        v15_ids=[],
    )
    (interruption / "v10" / "reference" / "asr.json").write_text("{}")
    if judge_name == "gpt":
        with pytest.raises(SystemExit, match="OPENAI_API_KEY"):
            judge_module.judge(load_settings("v10-interruption"), 1, False)
    else:
        with pytest.raises(pytest.fail.Exception, match="judge server started"):
            judge_module.judge(load_settings("v10-interruption"), 1, False)


def test_asr_and_judge_refuse_a_selected_v10_without_its_artifacts(
    settings: Settings,
) -> None:
    repeat_dir = stage_scored_repeat(settings, V10_IDS)
    shutil.rmtree(repeat_dir / "v10" / "reference")
    with pytest.raises(SystemExit, match="did not finish the v1.0 export"):
        asr_module.asr(settings, 1, False)
    with pytest.raises(SystemExit, match="did not finish the v1.0 export"):
        judge_module.judge(settings, 1, False)
    tree = repeat_dir / "v10" / "reference"
    tree.mkdir()
    (tree / "manifest.json").write_text("{}")
    with pytest.raises(SystemExit, match="asr.*asr.json is missing"):
        judge_module.judge(settings, 1, False)


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
        LIVE_SERVER,
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
    assert judge_module.evaluate_v10(settings, tree, [], judge_arguments, None) == []
    assert len(calls) == 3


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
    "candor_pause_handling": {"selected": 2, "eligible": 0},
    "candor_turn_taking": {"selected": 3, "eligible": 3},
    "synthetic_user_interruption": {"selected": 2, "eligible": 2},
    "icc_backchannel": {"selected": 2, "eligible": 2},
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
            "judge": None,
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
    assert "## FDB v1.0 official reference results" in lines
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
    assert (
        "| icc_backchannel | 2 | 2 | 2 | 50.0 ± 0.0% | n/a | n/a | n/a | "
        "0.070 ± 0.000 | 0.700 ± 0.000 |" in lines
    )
    # note (luojiaxuan): a subset whose sessions all failed keeps its row.
    assert (
        "| candor_pause_handling | 2 | 0 | 0 | n/a | n/a | n/a | n/a | n/a | n/a |"
        in lines
    )


def test_results_list_unevaluated_v10_subsets_as_pending_or_failed(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "run"
    repeat_dir = run_root / "repeat-1"
    write_v15_scores(repeat_dir, 1.0)
    write_v10_tree(repeat_dir, takeover=1.0, latency_s=2.0, retries=0)
    tree = repeat_dir / "v10" / "reference"
    summary = json.loads((tree / "summary.json").read_text())
    for subset in ("candor_turn_taking", "icc_backchannel"):
        del summary["subsets"][subset]
    (tree / "summary.json").write_text(json.dumps(summary))
    (tree / "logs").mkdir(exist_ok=True)
    (tree / "logs" / "evaluate-icc_backchannel.log").write_text("Traceback\n")
    lines = render(run_root, "minicpmo", "qwen").splitlines()
    assert (
        "| candor_turn_taking | 3 | 3 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |"
        in lines
    )
    assert "- repeat-1 `candor_turn_taking`: not evaluated yet; run `judge`" in lines
    assert (
        "- repeat-1 `icc_backchannel`: evaluation failed; see "
        "`v10/reference/logs/evaluate-icc_backchannel.log`" in lines
    )
    # note (luojiaxuan): a subset with no eligible sample is not listed.
    assert not any("candor_pause_handling`" in line for line in lines)


def test_results_of_a_v10_only_run_have_no_v15_section(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    (run_root / "repeat-1").mkdir(parents=True)
    write_v10_tree(run_root / "repeat-1", takeover=1.0, latency_s=2.0, retries=0)
    results = render(run_root, "minicpmo", "qwen")
    assert results.startswith("# FDB results: run\n")
    assert "v1.5" not in results
    assert "## FDB v1.0 official reference results" in results


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


def write_unified_reference(repeat_dir: Path) -> Path:
    tree = repeat_dir / "v10" / "reference"
    tree.mkdir(parents=True)
    sample_ids = [
        "candor_pause_handling/1",
        "candor_turn_taking/1",
        "icc_backchannel/1",
    ]
    (repeat_dir / "v10" / "sample-ids.txt").write_text(ids_text(sample_ids))
    turn_dir = tree / "candor_turn_taking" / "1"
    turn_dir.mkdir(parents=True)
    (turn_dir / "turn_taking.json").write_text(
        json.dumps([{"text": "[TURN-TAKING]", "timestamp": [2.0, 2.5]}])
    )
    (turn_dir / "output.json").write_text(json.dumps({"text": "", "chunks": []}))
    rows = [
        {
            "sample_id": sample_ids[0],
            "subset": "candor_pause_handling",
            "eligible": False,
            "reasons": ["recording failed"],
            "files": {},
        },
        {
            "sample_id": sample_ids[1],
            "subset": "candor_turn_taking",
            "eligible": True,
            "reasons": [],
            "window": {"T_s": 4.0},
            "files": {"turn_taking.json": file_sha256(turn_dir / "turn_taking.json")},
        },
        {
            "sample_id": sample_ids[2],
            "subset": "icc_backchannel",
            "eligible": True,
            "reasons": [],
            "files": {},
        },
    ]
    (tree / "manifest.json").write_text(
        json.dumps(
            {
                "kind": MANIFEST_KIND,
                "samples": rows,
                "counts": {
                    "candor_pause_handling": {"selected": 1, "eligible": 0},
                    "candor_turn_taking": {"selected": 1, "eligible": 1},
                    "icc_backchannel": {"selected": 1, "eligible": 1},
                },
            }
        )
    )
    (tree / "asr.json").write_text(
        json.dumps(
            {
                "subsets": {
                    "candor_turn_taking": {
                        "transcripts": {"1": file_sha256(turn_dir / "output.json")}
                    }
                }
            }
        )
    )
    return tree


def test_unified_scoring_and_aggregate_cli_keep_per_repeat_coverage_and_missing_states(
    settings: Settings,
) -> None:
    repeat_dir = settings.repeat_dir(1)
    tree = write_unified_reference(repeat_dir)
    write_v15_scores(repeat_dir, 1.0)
    (repeat_dir / "judge-qwen" / "summary.json").unlink()
    environment = dict(os.environ, FDB_WORK=str(settings.fdb_work))
    scorer = subprocess.run(
        [
            sys.executable,
            "-m",
            V10_CLI,
            "unified-score",
            "--tree",
            str(tree),
        ],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert scorer.returncode == 0, scorer.stderr
    aggregate = subprocess.run(
        [
            sys.executable,
            "-m",
            "benchmarks.duplex.fdb",
            "aggregate",
            "--run-name",
            settings.run_name,
        ],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert aggregate.returncode == 0, aggregate.stderr
    results = (settings.run_root / "RESULTS.md").read_text()
    assert "## FDB v1.5 results" in results
    assert "| all | 2 | 2 | 1.000 | 2.500 | n/a | n/a | n/a | n/a | n/a |" in results
    assert "Qwen behavior judge pending or failed for repeat-1" in results
    assert results.index("## FDB v1.0 unified metrics") < results.index(
        "## FDB v1.0 official reference results"
    )
    assert "| `candor_pause_handling` |" in results
    assert (
        "| `candor_pause_handling` | 216 | 1 | 0 | 1 | 0.000 | recording failed (1) | n/a |"
        in results
    )
    assert "| `candor_pause_handling` | `pause_intrusion_rate` | n/a" in results
    assert "median not reached" in results
    assert "0/1 responded" in results
    assert "Minimum 2.000 s after reference" in results
    assert "Backchannel is scored by the official reference evaluator only." in results
    assert f"[unified.json](repeat-1/v10/reference/unified.json)" in results

    repeat_2 = settings.repeat_dir(2)
    (repeat_2 / "v10").mkdir(parents=True)
    (repeat_2 / "v10" / "sample-ids.txt").write_text(ids_text(["candor_turn_taking/1"]))
    partial = render(settings.run_root, "minicpmo", "qwen")
    assert "### repeat-2" in partial
    assert "export is incomplete" in partial
    assert "unified-score --tree" in partial
    assert f"export FDB_WORK={settings.fdb_work}" in partial
    assert "RUN_NAME" not in partial

    unified_path = tree / "unified.json"
    saved_unified = unified_path.read_text()
    missing_subset = json.loads(saved_unified)
    missing_subset["subsets"].pop("candor_turn_taking")
    unified_path.write_text(json.dumps(missing_subset))
    incomplete_subsets = render(settings.run_root, "minicpmo", "qwen")
    assert "does not cover every selected scored subset" in incomplete_subsets
    assert "median not reached" not in incomplete_subsets

    incompatible = json.loads(saved_unified)
    incompatible["schema_version"] = 2
    unified_path.write_text(json.dumps(incompatible))
    unsupported = render(settings.run_root, "minicpmo", "qwen")
    assert "unsupported kind, schema, or definition" in unsupported
    assert "median not reached" not in unsupported
    unified_path.write_text(saved_unified)

    (tree / "manifest.json").write_text((tree / "manifest.json").read_text() + " ")
    stale = render(settings.run_root, "minicpmo", "qwen")
    assert "stale for the current manifest or ASR" in stale
    assert "median not reached" not in stale

    (tree / "unified-samples.jsonl").unlink()
    incomplete = render(settings.run_root, "minicpmo", "qwen")
    assert "unified output pair is missing or incomplete" in incomplete
    assert "median not reached" not in incomplete
