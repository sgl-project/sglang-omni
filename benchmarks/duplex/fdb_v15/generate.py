# SPDX-License-Identifier: Apache-2.0
"""Step 1: record overlap and clean sessions against the model server, then export
fixed-window scoring audio."""

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
    Settings,
    log,
    log_tail,
    run_command,
)
from benchmarks.duplex.fdb_v15.selection import (
    SampleSelection,
    check_matches_other_repeats,
    describe,
    select_samples,
)
from benchmarks.duplex.fdb_v15.servers import model_server
from benchmarks.duplex.run_artifacts import accounting, load_run
from sglang_omni.models.minicpm_o.native_config import DEFAULT_MAX_SESSIONS

PROGRESS_INTERVAL_S = 60


def record_command(settings: Settings) -> list[str]:
    """The recorder command shared by every shard, without output or samples."""
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
        "benchmarks.eval.benchmark_duplex_v15",
        "record",
        "--profile",
        RECORD_PROFILE,
        "--dataset-root",
        str(settings.dataset),
        "--dataset-revision",
        settings.dataset_revision_file.read_text().strip(),
        "--url",
        settings.realtime_url,
        "--model",
        MODEL_ID,
        "--model-revision",
        settings.model_revision,
        "--server-revision",
        server_revision,
        "--timeout",
        settings.session_timeout_s,
    ]


def record_shards(
    settings: Settings, repeat_dir: Path, sample_ids: list[str], num_shards: int
) -> list[Path]:
    """Run num_shards recorders in parallel; returns the shard directories."""
    session_count = 2 * len(sample_ids)
    shards = [
        (
            repeat_dir / "recording" / f"shard-{index}",
            sample_ids[index::num_shards],
        )
        for index in range(num_shards)
    ]
    shards = [(shard_dir, shard_ids) for shard_dir, shard_ids in shards if shard_ids]
    base_command = record_command(settings)
    with ExitStack() as stack:
        processes = []
        for shard_dir, shard_ids in shards:
            shard_log = stack.enter_context(
                (repeat_dir / "logs" / f"record-{shard_dir.name}.log").open("w")
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
                finished = len(list((repeat_dir / "recording").rglob("report.json")))
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


def print_variant_status(repeat_dir: Path, shard_dir: Path) -> None:
    log(f"-- {shard_dir.name}: variant status")
    if not (shard_dir / "run.json").is_file():
        log(log_tail(repeat_dir / "logs" / f"record-{shard_dir.name}.log"))
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


def generate(
    settings: Settings, repeat: int, selection: SampleSelection, num_shards: int
) -> None:
    repeat_dir = settings.repeat_dir(repeat)
    if (repeat_dir / "recording").exists():
        raise SystemExit(
            f"ERROR: {repeat_dir / 'recording'} exists. "
            f"Use a new --repeat, or delete {repeat_dir} to redo it."
        )
    else:
        pass
    # note (wenyao): model_server_command passes no --max_sessions, so the server
    # admits the code default.
    if num_shards > DEFAULT_MAX_SESSIONS:
        raise SystemExit(
            f"ERROR: --num-shards {num_shards} exceeds the model server's "
            f"max_sessions {DEFAULT_MAX_SESSIONS}; it rejects extra connections "
            "with HTTP 503."
        )
    else:
        pass
    sample_ids = select_samples(settings.dataset, selection)
    check_matches_other_repeats(repeat_dir, sample_ids)
    (repeat_dir / "logs").mkdir(parents=True, exist_ok=True)
    (repeat_dir / "sample-ids.txt").write_text("\n".join(sample_ids) + "\n")
    log(f"== Selected {len(sample_ids)} pairs: {describe(sample_ids)}")
    with model_server(settings, repeat_dir / "logs" / "model-server.log"):
        (repeat_dir / "recording").mkdir()
        log(
            f"== Recording {len(sample_ids)} pairs ({2 * len(sample_ids)} sessions) "
            f"in {num_shards} shard(s) -> {repeat_dir}"
        )
        shard_dirs = record_shards(settings, repeat_dir, sample_ids, num_shards)
    for shard_dir in shard_dirs:
        print_variant_status(repeat_dir, shard_dir)
    export_reference_audio(settings, repeat_dir, shard_dirs, sample_ids)
    log(f"Step 1 done: {repeat_dir / 'reference-audio'}")
