# SPDX-License-Identifier: Apache-2.0
"""Synchronized containment-race and accepted-snapshot continuity unit tests for C3."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import httpx
import pytest

from benchmarks.c3.config_manifest import persist_inference_failure_ledger
from benchmarks.c3.containment import (
    read_bytes_nofollow,
    write_contained_json_atomic,
    write_text_atomic,
)
from benchmarks.c3.judge_state import persist_structured_judge_failures
from benchmarks.c3.registry import C3SubtaskSpec, select_c3_subtask_specs
from benchmarks.c3.runner import encode_audio_file_to_wav_base64_sync
from benchmarks.dataset.c3 import C3LoadedDatasetSnapshot, load_and_verify_c3_snapshot
from benchmarks.eval.benchmark_omni_c3 import main as run_c3_cli
from tests.unit_test.fixtures.c3_fixtures import (
    build_pinned_synthetic_c3_dataset,
    write_synthetic_audio_clip,
)


def test_c3_accepted_snapshot_continuity_and_race_guards(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify synchronized ancestor symlink races, metadata single-read hashing, and cached-prediction forwarding."""
    real_os_open = os.open

    for case_index, case_kind in enumerate(
        ("write_report", "clear_inference_ledger", "clear_judge_ledger", "read_audio")
    ):
        case_root = tmp_path / f"race_{case_index}"
        work_dir = case_root / "work"
        victim_dir = case_root / "victim"
        (work_dir / "run").mkdir(parents=True)
        (victim_dir / "run").mkdir(parents=True)
        write_synthetic_audio_clip(work_dir / "run" / "clip.mp3")
        write_synthetic_audio_clip(victim_dir / "run" / "clip.mp3")
        victim_report = victim_dir / "run" / "RESULTS.md"
        victim_report.write_text("VICTIM_UNTOUCHED", encoding="utf-8")
        victim_inference_ledger = victim_dir / "run" / "inference_failures.json"
        victim_inference_ledger.write_text('{"x": "keep"}', encoding="utf-8")
        victim_judge_ledger = victim_dir / "run" / "judge_failures.json"
        victim_judge_ledger.write_text('{"j": {"x": "keep"}}', encoding="utf-8")
        write_contained_json_atomic(
            work_dir / "run", "inference_failures.json", {"x": "stale"}
        )
        write_contained_json_atomic(
            work_dir / "run", "judge_failures.json", {"j": {"x": "stale"}}
        )

        swap_state = {"armed": True, "triggered": False}

        def racing_open(
            path_argument: str | bytes,
            open_flags: int,
            open_mode: int = 0o777,
            *,
            dir_fd: int | None = None,
        ) -> int:
            if swap_state["armed"] and path_argument == "work" and dir_fd is not None:
                swap_state["armed"] = False
                swap_state["triggered"] = True
                displaced = work_dir.parent / "work_displaced"
                work_dir.rename(displaced)
                work_dir.symlink_to(victim_dir)
            else:
                pass
            if dir_fd is not None:
                return real_os_open(path_argument, open_flags, open_mode, dir_fd=dir_fd)
            else:
                return real_os_open(path_argument, open_flags, open_mode)

        monkeypatch.setattr("benchmarks.c3.containment.os.open", racing_open)
        with pytest.raises(ValueError, match="Symlink"):
            if case_kind == "write_report":
                write_text_atomic(work_dir / "run", "RESULTS.md", "ATTACKER")
            elif case_kind == "clear_inference_ledger":
                persist_inference_failure_ledger(work_dir / "run", {})
            elif case_kind == "clear_judge_ledger":
                persist_structured_judge_failures(work_dir / "run", {})
            else:
                encode_audio_file_to_wav_base64_sync(work_dir / "run", "clip.mp3")
        monkeypatch.setattr("benchmarks.c3.containment.os.open", real_os_open)

        assert swap_state["triggered"] is True
        assert victim_report.read_text(encoding="utf-8") == "VICTIM_UNTOUCHED"
        assert victim_inference_ledger.is_file()
        assert victim_judge_ledger.is_file()

    pinned_dataset, pinned_audio_map = build_pinned_synthetic_c3_dataset(
        tmp_path / "pinned_dataset", monkeypatch, instances_with_audio_per_subtask=1
    )
    pause_specs = select_c3_subtask_specs(subtask_filter=("en/phonological_pause",))

    pause_json_relative_path = "english/text/ambiguity/phonological/pause.json"
    pause_json_path = pinned_dataset / pause_json_relative_path
    original_pause_json_bytes = pause_json_path.read_bytes()
    altered_rows = json.loads(original_pause_json_bytes.decode("utf-8"))
    altered_rows[0]["notation"] = "TAMPERED_REFERENCE"
    pause_json_path.write_text(json.dumps(altered_rows), encoding="utf-8")
    real_read_bytes_nofollow = read_bytes_nofollow
    metadata_read_count = {"pause_json_reads": 0}

    def transient_tampered_metadata_read(
        root_directory: Path, relative_posix_path: str
    ) -> bytes:
        acquired_bytes = real_read_bytes_nofollow(root_directory, relative_posix_path)
        if relative_posix_path == pause_json_relative_path:
            metadata_read_count["pause_json_reads"] += 1
            if metadata_read_count["pause_json_reads"] == 1:
                # note (machengqian): restore official bytes immediately after the first read so any separate re-open would falsely pass.
                pause_json_path.write_bytes(original_pause_json_bytes)
            else:
                pass
        else:
            pass
        return acquired_bytes

    monkeypatch.setattr(
        "benchmarks.dataset.c3.read_bytes_nofollow", transient_tampered_metadata_read
    )
    tampered_snapshot = load_and_verify_c3_snapshot(
        pinned_dataset, subtask_specs=pause_specs, max_samples_per_subtask=1
    )
    monkeypatch.setattr(
        "benchmarks.dataset.c3.read_bytes_nofollow", real_read_bytes_nofollow
    )
    assert metadata_read_count["pause_json_reads"] == 1
    assert pause_json_path.read_bytes() == original_pause_json_bytes
    assert (
        "TAMPERED_REFERENCE"
        in tampered_snapshot.instances[0].reference_record.model_dump_json()
    )
    assert tampered_snapshot.identity.metadata_matches_expected_revision is False
    assert tampered_snapshot.identity.dataset_matches_expected_revision is False

    real_load_snapshot = load_and_verify_c3_snapshot
    target_clip_relative_path = "english/speech/ambiguity/phonological/pause/0.mp3"
    target_clip_path = pinned_dataset / target_clip_relative_path
    original_clip_bytes = target_clip_path.read_bytes()
    original_sums_bytes = (pinned_dataset / "SHA256SUMS").read_bytes()
    original_async_client = httpx.AsyncClient

    def simple_reply_handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {"role": "assistant", "content": "Reply"},
                        "finish_reason": "stop",
                    }
                ]
            },
        )

    def build_mock_client(
        *, timeout: float = 180.0, trust_env: bool = False
    ) -> httpx.AsyncClient:
        return original_async_client(
            transport=httpx.MockTransport(simple_reply_handler),
            timeout=timeout,
            trust_env=trust_env,
        )

    monkeypatch.setattr(httpx, "AsyncClient", build_mock_client)

    clean_out_dir = tmp_path / "clean_baseline_out"
    common_flags = [
        "--dataset-root",
        str(pinned_dataset),
        "--subtasks",
        "en/phonological_pause",
        "--max-samples-per-subtask",
        "1",
    ]
    assert (
        run_c3_cli(
            [
                *common_flags,
                "--output-dir",
                str(clean_out_dir),
                "--phase",
                "generate",
                "--model",
                "m1",
                "--base-url",
                "http://127.0.0.1:8000",
                "--warmup",
                "0",
            ]
        )
        == 0
    )

    def corrupt_sums_and_clip_after_snapshot(
        root_path: Path,
        *,
        subtask_specs: list[C3SubtaskSpec] | None = None,
        mini: bool = False,
        max_samples_per_subtask: int | None = None,
        strict_counts: bool = True,
        verify_all_audio_headers: bool = True,
    ) -> C3LoadedDatasetSnapshot:
        loaded_snapshot = real_load_snapshot(
            root_path,
            subtask_specs=subtask_specs,
            mini=mini,
            max_samples_per_subtask=max_samples_per_subtask,
            strict_counts=strict_counts,
            verify_all_audio_headers=verify_all_audio_headers,
        )
        (root_path / "SHA256SUMS").write_text("corrupted\n", encoding="utf-8")
        write_synthetic_audio_clip(target_clip_path, use_real_mp3=True)
        return loaded_snapshot

    monkeypatch.setattr(
        "benchmarks.eval.benchmark_omni_c3.load_and_verify_c3_snapshot",
        corrupt_sums_and_clip_after_snapshot,
    )
    swap_out_dir = tmp_path / "post_snapshot_swap_out"
    assert (
        run_c3_cli(
            [
                *common_flags,
                "--output-dir",
                str(swap_out_dir),
                "--phase",
                "generate",
                "--model",
                "m1",
                "--base-url",
                "http://127.0.0.1:8000",
                "--warmup",
                "0",
            ]
        )
        == 1
    )
    swap_failures = json.loads(
        (swap_out_dir / "inference_failures.json").read_text(encoding="utf-8")
    )
    assert "failed pinned SHA-256 check" in swap_failures["en/phonological_pause/0"]

    unverified_clip_sha256 = hashlib.sha256(target_clip_path.read_bytes()).hexdigest()
    assert unverified_clip_sha256 != pinned_audio_map[target_clip_relative_path]

    saved_predictions = json.loads(
        (clean_out_dir / "predictions.json").read_text(encoding="utf-8")
    )
    saved_predictions["en/phonological_pause/0"]["source_audio_sha256"] = [
        unverified_clip_sha256
    ]
    write_contained_json_atomic(clean_out_dir, "predictions.json", saved_predictions)

    target_clip_path.write_bytes(original_clip_bytes)
    (pinned_dataset / "SHA256SUMS").write_bytes(original_sums_bytes)
    with pytest.raises(ValueError, match="pinned SHA256SUMS audio digests"):
        run_c3_cli(
            [
                *common_flags,
                "--output-dir",
                str(clean_out_dir),
                "--phase",
                "generate",
                "--model",
                "m1",
                "--base-url",
                "http://127.0.0.1:8000",
                "--warmup",
                "0",
            ]
        )

    target_clip_path.write_bytes(original_clip_bytes)
    (pinned_dataset / "SHA256SUMS").write_bytes(original_sums_bytes)
    with pytest.raises(ValueError, match="pinned SHA256SUMS audio digests"):
        run_c3_cli(
            [
                *common_flags,
                "--output-dir",
                str(clean_out_dir),
                "--phase",
                "score",
            ]
        )

    target_clip_path.write_bytes(original_clip_bytes)
    (pinned_dataset / "SHA256SUMS").write_bytes(original_sums_bytes)

    outside_file = tmp_path / "outside_secret.txt"
    outside_file.write_text("KEEP_SAFE", encoding="utf-8")
    (pinned_dataset / "english" / "symlink_escape.mp3").symlink_to(outside_file)
    for unsafe_relative_path in (
        "../outside_secret.txt",
        "/etc/passwd",
        "english\\speech\\0.mp3",
        "english/symlink_escape.mp3",
    ):
        with pytest.raises(ValueError):
            read_bytes_nofollow(pinned_dataset, unsafe_relative_path)

    outside_dir = tmp_path / "outside_dir"
    outside_dir.mkdir()
    trapped_output = tmp_path / "trapped_output"
    trapped_output.mkdir()
    (trapped_output / "RESULTS.md").symlink_to(outside_file)
    with pytest.raises(ValueError, match="symlink"):
        write_text_atomic(trapped_output, "RESULTS.md", "OVERWRITTEN")
    assert outside_file.read_text(encoding="utf-8") == "KEEP_SAFE"

    fake_home = tmp_path / "fake_home"
    fake_home.mkdir()
    (fake_home / "c3-output").symlink_to(outside_dir)
    monkeypatch.setenv("HOME", str(fake_home))
    with pytest.raises(ValueError, match="Symlink"):
        run_c3_cli(
            [
                *common_flags,
                "--output-dir",
                "~/c3-output",
                "--phase",
                "generate",
                "--model",
                "m1",
                "--base-url",
                "http://127.0.0.1:8000",
            ]
        )

    writerless_fifo_path = pinned_dataset / "writerless_fifo"
    os.mkfifo(writerless_fifo_path)
    with pytest.raises(ValueError, match="Expected regular file"):
        read_bytes_nofollow(pinned_dataset, "writerless_fifo")
    writerless_fifo_path.unlink()

    os.mkfifo(pinned_dataset / "SHA256SUMS_fifo")
    (pinned_dataset / "SHA256SUMS_fifo").replace(pinned_dataset / "SHA256SUMS")
    fifo_snapshot = real_load_snapshot(
        pinned_dataset, subtask_specs=pause_specs, max_samples_per_subtask=1
    )
    assert fifo_snapshot.identity.sha256sums_verified is False
    assert fifo_snapshot.identity.dataset_matches_expected_revision is False
    (pinned_dataset / "SHA256SUMS").unlink()
    (pinned_dataset / "SHA256SUMS").write_bytes(original_sums_bytes)
