# SPDX-License-Identifier: Apache-2.0
"""Fail closed when an Apple CI run cannot exercise its selected tests."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def runner():
    path = Path(__file__).resolve().parents[3] / ".github/scripts/run_apple_ut.py"
    spec = importlib.util.spec_from_file_location("run_apple_ut", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("backend", ["mps", "mlx"])
def test_rejects_non_apple_runner(runner, monkeypatch, backend):
    monkeypatch.setattr(runner.platform, "system", lambda: "Linux")
    with pytest.raises(RuntimeError, match="arm64 macOS"):
        runner.check_device(backend)


def test_rejects_unavailable_mps(runner, monkeypatch):
    monkeypatch.setattr(runner.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(runner.platform, "machine", lambda: "arm64")
    torch = SimpleNamespace(
        backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: False))
    )
    monkeypatch.setitem(sys.modules, "torch", torch)
    with pytest.raises(RuntimeError, match="Torch MPS is unavailable"):
        runner.check_device("mps")


@pytest.mark.parametrize(
    ("xml", "error"),
    [
        ("<testsuites><testsuite/></testsuites>", "no test cases"),
        (
            '<testsuites><testsuite><testcase name="metal"><skipped/></testcase>'
            "</testsuite></testsuites>",
            "must execute every selected case",
        ),
    ],
)
def test_rejects_empty_or_skipped_reports(runner, tmp_path, xml, error):
    report = tmp_path / "report.xml"
    report.write_text(xml)
    with pytest.raises(RuntimeError, match=error):
        runner.verify_report(report)


def test_accepts_executed_report(runner, tmp_path):
    report = tmp_path / "report.xml"
    report.write_text(
        '<testsuites><testsuite><testcase name="metal"/></testsuite></testsuites>'
    )
    runner.verify_report(report)


@pytest.mark.parametrize("exit_code", [1, 2, 5])
def test_propagates_pytest_failure(runner, monkeypatch, tmp_path, exit_code):
    monkeypatch.setattr(sys, "argv", ["run_apple_ut.py", "mps"])
    monkeypatch.setenv("APPLE_UT_REPORT_DIR", str(tmp_path))
    for name in (
        "SGLANG_USE_MLX",
        "PYTORCH_ENABLE_MPS_FALLBACK",
        "HF_HUB_OFFLINE",
        "HF_DATASETS_OFFLINE",
    ):
        monkeypatch.setenv(name, "")
    monkeypatch.setattr(runner, "check_device", lambda backend: {})
    monkeypatch.setattr(runner.subprocess, "check_output", lambda *a, **kw: "sha")
    monkeypatch.setattr(
        runner.subprocess,
        "run",
        lambda *a, **kw: SimpleNamespace(returncode=exit_code),
    )
    # No XML is produced: failure must return before report validation.
    assert runner.main() == exit_code
