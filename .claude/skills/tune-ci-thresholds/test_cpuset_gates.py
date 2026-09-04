#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for calibration cpuset binding and precheck gates."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

SKILL_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SKILL_DIR))

import tune  # noqa: E402


HOST = {
    "gpu_group_cpusets": {
        "0,1": "0-15,64-79",
        "2,3": "16-31,80-95",
        "4,5": "48-63,112-127",
        "6,7": "32-47,96-111",
    }
}


def test_parse_cpuset_spec_ranges():
    assert tune.parse_cpuset_spec("0-15,64-79") == set(range(0, 16)) | set(
        range(64, 80)
    )


def test_cpuset_for_gpus_exact_and_subset(monkeypatch):
    monkeypatch.delenv("OMNI_CI_CPUSET", raising=False)
    assert tune.cpuset_for_gpus(HOST, [0, 1]) == "0-15,64-79"
    assert tune.cpuset_for_gpus(HOST, [1]) == "0-15,64-79"
    assert tune.cpuset_for_gpus(HOST, [6, 7]) == "32-47,96-111"
    assert tune.cpuset_for_gpus(HOST, [8, 9]) is None


def test_cpuset_for_gpus_env_override(monkeypatch):
    monkeypatch.setenv("OMNI_CI_CPUSET", "48-63,112-127")
    assert tune.cpuset_for_gpus(HOST, [0, 1]) == "48-63,112-127"


def test_require_cpuset_for_gpus_raises(monkeypatch):
    monkeypatch.delenv("OMNI_CI_CPUSET", raising=False)
    with pytest.raises(RuntimeError, match="no cpuset"):
        tune.require_cpuset_for_gpus({"gpu_group_cpusets": {}}, [0, 1])


def test_precheck_cpuset_gate_missing_lane(monkeypatch):
    monkeypatch.delenv("TUNE_GPU_INCLUDE", raising=False)
    monkeypatch.delenv("OMNI_CI_CPUSET", raising=False)
    errs, _warns, detail = tune.precheck_cpuset_gate(HOST)
    assert errs
    assert detail["status"] == "missing_gpu_group"


def test_precheck_cpuset_gate_idle(monkeypatch):
    monkeypatch.setenv("TUNE_GPU_INCLUDE", "0,1")
    monkeypatch.delenv("OMNI_CI_CPUSET", raising=False)
    monkeypatch.setattr(tune, "cpuset_external_busy", lambda *_a, **_k: 0.01)
    errs, warns, detail = tune.precheck_cpuset_gate(HOST)
    assert errs == []
    assert detail["status"] == "idle"
    assert detail["cpuset"] == "0-15,64-79"
    assert not any("busy" in w for w in warns)


def test_precheck_cpuset_gate_busy_refuses(monkeypatch):
    monkeypatch.setenv("TUNE_GPU_INCLUDE", "2,3")
    monkeypatch.delenv("OMNI_CI_CPUSET", raising=False)
    monkeypatch.setattr(tune, "cpuset_external_busy", lambda *_a, **_k: 0.55)
    errs, _warns, detail = tune.precheck_cpuset_gate(HOST)
    assert any("refuse this calibration" in e for e in errs)
    assert detail["status"] == "busy"
    assert detail["busy_fraction"] == 0.55


def test_contention_peak_from_log():
    text = (
        "[cpuset-contention] cpuset=0-1 windows=3 "
        "foreign-cores mean=0.40 max=2.50 errors=0\n"
    )
    assert tune.contention_peak_from_log(text) == 2.5


def _sampler_with(samples):
    ContentionSampler = tune._import_contention_sampler()
    sampler = ContentionSampler({0})
    sampler._samples = list(samples)
    return sampler


def test_contention_abort_holds_through_a_single_spike():
    over = tune._CONTENTION_FAIL_CORES + 0.01
    assert tune.contention_abort_reason(_sampler_with([0.1, over, 0.1])) is None


def test_contention_abort_holds_before_enough_windows_exist():
    over = tune._CONTENTION_FAIL_CORES + 0.5
    short = [over] * (tune._CONTENTION_FAIL_WINDOWS - 1)
    assert tune.contention_abort_reason(_sampler_with(short)) is None


def test_contention_abort_fires_on_sustained_intrusion():
    over = tune._CONTENTION_FAIL_CORES + 0.5
    held = [0.1] + [over] * tune._CONTENTION_FAIL_WINDOWS
    reason = tune.contention_abort_reason(_sampler_with(held))
    assert reason is not None
    assert "consecutive windows" in reason


def test_contention_abort_releases_once_the_lane_recovers():
    over = tune._CONTENTION_FAIL_CORES + 0.5
    recovered = [over] * tune._CONTENTION_FAIL_WINDOWS + [0.1]
    assert tune.contention_abort_reason(_sampler_with(recovered)) is None


def test_contention_reading_describes_peak_and_mean():
    seen = tune.ContentionReading(peak=2.4, mean=0.6, windows=78)
    assert seen.describe() == "peak 2.40 cores, mean 0.60 over 78 windows"


def test_classify_cpuset_contention_rc():
    assert tune._classify("", tune._PYTEST_RC_CPUSET_CONTENTION) == (
        "cpuset_contention"
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
