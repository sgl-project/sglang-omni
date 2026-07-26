# SPDX-License-Identifier: Apache-2.0
"""Stage integrity for bench_sweep: no stale, partial, or aliased results."""

from __future__ import annotations

import json
import os

import pytest

from benchmarks.eval.bench_sweep import (
    _assert_fresh_dir,
    _collect_stage,
    _stage_metrics,
)


def _write_result(stage_dir, i, qps=1.0):
    cdir = stage_dir / f"client{i}"
    cdir.mkdir(parents=True)
    (cdir / "speed_results.json").write_text(
        json.dumps(
            {
                "per_request": [
                    {
                        "is_success": True,
                        "latency_s": 1.0,
                        "audio_ttfp_s": 0.2,
                        "rtf": 0.5,
                        "audio_duration_s": 3.0,
                    }
                ],
                "summary": {
                    "completed_requests": 1,
                    "failed_requests": 0,
                    "throughput_qps": qps,
                    "audio_throughput_s_per_s": 2.0,
                },
            }
        ),
        encoding="utf-8",
    )


def test_fresh_dir_refuses_existing_directory(tmp_path):
    stale = tmp_path / "rate8-nonce"
    stale.mkdir()
    with pytest.raises(SystemExit, match="pre-existing"):
        _assert_fresh_dir(str(stale))


@pytest.mark.skipif(os.name != "posix", reason="symlinks need POSIX")
def test_fresh_dir_refuses_symlink(tmp_path):
    target = tmp_path / "elsewhere"
    target.mkdir()
    link = tmp_path / "rate8-nonce"
    link.symlink_to(target)
    with pytest.raises(SystemExit, match="pre-existing"):
        _assert_fresh_dir(str(link))


@pytest.mark.skipif(
    os.name != "posix" or getattr(os, "geteuid", lambda: 1)() == 0,
    reason="chmod semantics need POSIX and a non-root user",
)
def test_fresh_dir_fails_loudly_on_readonly_parent(tmp_path):
    parent = tmp_path / "out"
    parent.mkdir()
    parent.chmod(0o500)
    try:
        with pytest.raises(PermissionError):
            _assert_fresh_dir(str(parent / "rate8-nonce"))
    finally:
        parent.chmod(0o700)


def test_nonzero_exit_invalidates_the_whole_stage(tmp_path):
    _write_result(tmp_path, 0)
    _write_result(tmp_path, 1)
    results, invalid = _collect_stage(str(tmp_path), 2, [0, 3])
    assert results is None
    assert "exit codes" in invalid


def test_missing_result_invalidates_the_whole_stage(tmp_path):
    _write_result(tmp_path, 0)
    results, invalid = _collect_stage(str(tmp_path), 2, [0, 0])
    assert results is None
    assert "client1 result unreadable" in invalid


def test_corrupt_result_invalidates_the_whole_stage(tmp_path):
    _write_result(tmp_path, 0)
    cdir = tmp_path / "client1"
    cdir.mkdir()
    (cdir / "speed_results.json").write_text("{truncated", encoding="utf-8")
    results, invalid = _collect_stage(str(tmp_path), 2, [0, 0])
    assert results is None
    assert "client1 result unreadable" in invalid


def test_complete_stage_collects_and_aggregates(tmp_path):
    _write_result(tmp_path, 0, qps=1.5)
    _write_result(tmp_path, 1, qps=2.5)
    results, invalid = _collect_stage(str(tmp_path), 2, [0, 0])
    assert invalid is None
    stage = _stage_metrics(8.0, 10.0, results)
    assert stage["completed"] == 2
    assert stage["achieved_qps_runner"] == 4.0
    assert stage["ttfc_p50"] == 0.2
