# SPDX-License-Identifier: Apache-2.0
"""Step 1: record overlap and clean sessions (v1.5) and one session per sample
(v1.0) against the model server, then export fixed-window scoring audio."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from contextlib import ExitStack
from pathlib import Path

from benchmarks.duplex.fdb_v15.common import (
    ENGINE_LABEL,
    MODEL_ID,
    RECORD_PROFILE,
    REPO_ROOT,
    V10_DIR,
    Settings,
    log,
    log_tail,
    run_command,
)
from benchmarks.duplex.fdb_v15.selection import (
    SampleSelection,
    check_matches_other_repeats,
    describe,
    ids_text,
    select_samples,
    select_v10_samples,
)
from benchmarks.duplex.fdb_v15.servers import model_server
from benchmarks.duplex.run_artifacts import accounting, load_run
from benchmarks.duplex.v10_dataset import SUBSETS as V10_SUBSETS

PROGRESS_INTERVAL_S = 60


def record_command(
    settings: Settings, module: str, dataset: Path, revision_file: Path, timeout_s: str
) -> list[str]:
    """One recorder command shared by every shard, without output or samples."""
    server_revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return [
        sys.executable,
        "-m",
        module,
        "record",
        "--profile",
        RECORD_PROFILE,
        "--dataset-root",
        str(dataset),
        "--dataset-revision",
        revision_file.read_text().strip(),
        "--url",
        settings.realtime_url,
        "--model",
        MODEL_ID,
        "--model-revision",
        settings.model_revision,
        "--server-revision",
        server_revision,
        "--timeout",
        timeout_s,
    ]


def v15_record_command(settings: Settings) -> list[str]:
    return record_command(
        settings,
        "benchmarks.eval.benchmark_duplex_v15",
        settings.dataset,
        settings.dataset_revision_file,
        settings.session_timeout_s,
    )


def v10_record_command(settings: Settings) -> list[str]:
    return record_command(
        settings,
        "benchmarks.eval.benchmark_duplex_v10",
        settings.dataset_v10,
        settings.dataset_v10_revision_file,
        settings.v10_session_timeout_s,
    )


def record_shards(
    base_command: list[str],
    root: Path,
    sample_ids: list[str],
    sessions_per_sample: int,
    num_shards: int,
) -> list[Path]:
    """Run num_shards recorders in parallel under root/recording; returns the shard directories."""
    session_count = sessions_per_sample * len(sample_ids)
    shards = [
        (
            root / "recording" / f"shard-{index}",
            sample_ids[index::num_shards],
        )
        for index in range(num_shards)
    ]
    shards = [(shard_dir, shard_ids) for shard_dir, shard_ids in shards if shard_ids]
    with ExitStack() as stack:
        processes = []
        for shard_dir, shard_ids in shards:
            shard_log = stack.enter_context(
                (root / "logs" / f"record-{shard_dir.name}.log").open("w")
            )
            sample_arguments = [
                argument
                for sample_id in shard_ids
                for argument in ("--sample-id", sample_id)
            ]
            processes.append(
                subprocess.Popen(
                    [*base_command, "--output", str(shard_dir), *sample_arguments],
                    cwd=REPO_ROOT,
                    stdout=shard_log,
                    stderr=subprocess.STDOUT,
                )
            )
        pending = list(processes)
        while pending:
            try:
                pending[0].wait(timeout=PROGRESS_INTERVAL_S)
            except subprocess.TimeoutExpired:
                finished = len(list((root / "recording").rglob("report.json")))
                log(
                    f"   {time.strftime('%H:%M:%S')} finished sessions: {finished} / {session_count}"
                )
            pending = [process for process in pending if process.poll() is None]
    failed_shards = sum(process.returncode != 0 for process in processes)
    if failed_shards:
        log(
            f"WARNING: {failed_shards} shard(s) had non-passing sessions; they stay in the denominator."
        )
    else:
        pass
    return [shard_dir for shard_dir, _ in shards]


def print_variant_status(root: Path, shard_dir: Path) -> None:
    log(f"-- {shard_dir.name}: variant status")
    if not (shard_dir / "run.json").is_file():
        log(log_tail(root / "logs" / f"record-{shard_dir.name}.log"))
        return
    else:
        pass
    manifest, run, _ = load_run(shard_dir)
    log(json.dumps(accounting(manifest, run)["variant_status"]))


def export_reference_audio(
    settings: Settings, repeat_dir: Path, shard_dirs: list[Path], sample_ids: list[str]
) -> None:
    log("== Exporting fixed-window scoring audio")
    command = [
        sys.executable,
        "-m",
        "benchmarks.eval.benchmark_duplex_reference",
        "export",
        "--engine",
        ENGINE_LABEL,
        "--trace-format",
        "realtime-pcm16-v1",
        "--dataset-root",
        str(settings.dataset),
        "--out",
        str(repeat_dir / "reference-audio"),
    ]
    for shard_dir in shard_dirs:
        command += ["--run", str(shard_dir)]
    for sample_id in sample_ids:
        command += ["--only", sample_id]
    if not run_command(command):
        raise SystemExit("ERROR: export failed; see the message above.")
    else:
        pass


def export_v10_reference(
    settings: Settings, v10_dir: Path, shard_dirs: list[Path]
) -> None:
    log("== Exporting v1.0 fixed-window scoring audio")
    command = [
        sys.executable,
        "-m",
        "benchmarks.eval.benchmark_duplex_v10",
        "reference-export",
        "--dataset-root",
        str(settings.dataset_v10),
        "--out",
        str(v10_dir / "reference"),
    ]
    for shard_dir in shard_dirs:
        command += ["--run", str(shard_dir)]
    if not run_command(command):
        raise SystemExit("ERROR: v1.0 export failed; see the message above.")
    else:
        pass


def generate(
    settings: Settings,
    repeat: int,
    selection: SampleSelection,
    v10_selection: SampleSelection,
    num_shards: int,
) -> None:
    repeat_dir = settings.repeat_dir(repeat)
    if (repeat_dir / "recording").exists():
        raise SystemExit(
            f"ERROR: {repeat_dir / 'recording'} exists. "
            f"Use a new --repeat, or delete {repeat_dir} to redo it."
        )
    else:
        pass
    sample_ids = select_samples(settings.dataset, selection)
    v10_sample_ids = select_v10_samples(settings.dataset_v10, v10_selection)
    check_matches_other_repeats(repeat_dir, sample_ids, v10_sample_ids)
    if v10_sample_ids and not settings.dataset_v10_revision_file.is_file():
        raise SystemExit(
            f"ERROR: {settings.dataset_v10_revision_file} is missing. Run "
            "`python -m benchmarks.duplex.fdb_v15 setup` again, or pass "
            "--v10-per-subset 0."
        )
    else:
        pass
    # note (luojiaxuan): both recorder commands read pinned inputs; resolving
    # them here fails before a single session is recorded.
    record = v15_record_command(settings)
    v10_record = v10_record_command(settings) if v10_sample_ids else []
    (repeat_dir / "logs").mkdir(parents=True, exist_ok=True)
    (repeat_dir / "sample-ids.txt").write_text(ids_text(sample_ids))
    log(f"== Selected {len(sample_ids)} pairs: {describe(sample_ids)}")
    v10_dir = repeat_dir / V10_DIR
    if v10_sample_ids:
        (v10_dir / "logs").mkdir(parents=True, exist_ok=True)
        (v10_dir / "sample-ids.txt").write_text(ids_text(v10_sample_ids))
        log(
            f"== Selected {len(v10_sample_ids)} v1.0 samples: "
            f"{describe(v10_sample_ids, V10_SUBSETS)}"
        )
    else:
        pass
    with model_server(settings, repeat_dir / "logs" / "model-server.log"):
        (repeat_dir / "recording").mkdir()
        log(
            f"== Recording {len(sample_ids)} pairs ({2 * len(sample_ids)} sessions) "
            f"in {num_shards} shard(s) -> {repeat_dir}"
        )
        shard_dirs = record_shards(record, repeat_dir, sample_ids, 2, num_shards)
        v10_shard_dirs = []
        if v10_sample_ids:
            (v10_dir / "recording").mkdir()
            log(
                f"== Recording {len(v10_sample_ids)} v1.0 samples (one session each) "
                f"in {num_shards} shard(s) -> {v10_dir}"
            )
            v10_shard_dirs = record_shards(
                v10_record, v10_dir, v10_sample_ids, 1, num_shards
            )
        else:
            pass
    for shard_dir in shard_dirs:
        print_variant_status(repeat_dir, shard_dir)
    export_reference_audio(settings, repeat_dir, shard_dirs, sample_ids)
    for shard_dir in v10_shard_dirs:
        print_variant_status(v10_dir, shard_dir)
    if v10_shard_dirs:
        export_v10_reference(settings, v10_dir, v10_shard_dirs)
    else:
        pass
    log(f"Step 1 done: {repeat_dir / 'reference-audio'}")
