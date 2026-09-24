# SPDX-License-Identifier: Apache-2.0
"""Contract tests for the if/else lint hook."""

from __future__ import annotations

import subprocess
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
CHECKER_PATH = REPO_ROOT / "scripts" / "check_if_else.py"
PROBE_PACKAGE = "lint_if_else_probe"


def run_checker(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(CHECKER_PATH), *args],
        check=False,
        capture_output=True,
        text=True,
    )


@contextmanager
def probe_model_file(source: str) -> Iterator[Path]:
    probe_dir = REPO_ROOT / "sglang_omni" / "models" / PROBE_PACKAGE
    probe = probe_dir / "runner.py"
    probe_dir.mkdir(parents=True)
    probe.write_text(source, encoding="utf-8")
    try:
        yield probe
    finally:
        probe.unlink(missing_ok=True)
        if probe_dir.exists():
            probe_dir.rmdir()
        else:
            pass


def test_current_sglang_omni_tree_is_clean() -> None:
    result = run_checker()
    assert result.returncode == 0, result.stderr


def test_if_elif_else_and_expressions_are_clean() -> None:
    source = """
def choose(flag, items):
    if flag:
        return 1
    elif items:
        return 2
    else:
        return 3

value = 1 if flag else 0
kept = [item for item in items if item]
"""
    with probe_model_file(source) as probe:
        result = run_checker(str(probe))
        assert result.returncode == 0, result.stderr


def test_noqa_comment_does_not_exempt_a_bare_if() -> None:
    source = "def close(session):\n    if session is not None:  # noqa: if-else\n        session.close()\n"
    with probe_model_file(source) as probe:
        result = run_checker(str(probe))
        assert result.returncode == 1
        assert "if without else" in result.stderr


def test_fix_fills_nested_one_line_and_elif() -> None:
    source = (
        "def run(flag, nested):\n"
        "    if flag:\n"
        "        if nested:\n"
        "            return 1\n"
        "    elif nested:\n"
        "        return 2\n"
        "    if flag: return 3\n"
    )
    expected = (
        "def run(flag, nested):\n"
        "    if flag:\n"
        "        if nested:\n"
        "            return 1\n"
        "        else:\n"
        "            pass\n"
        "    elif nested:\n"
        "        return 2\n"
        "    else:\n"
        "        pass\n"
        "    if flag: return 3\n"
        "    else:\n"
        "        pass\n"
    )
    with probe_model_file(source) as probe:
        result = run_checker("--fix", str(probe))
        assert result.returncode == 0, result.stderr
        assert probe.read_text(encoding="utf-8") == expected
        again = run_checker(str(probe))
        assert again.returncode == 0, again.stderr


def test_new_bare_if_fails_the_default_scan() -> None:
    with probe_model_file("def load():\n    if True:\n        return 1\n") as probe:
        result = run_checker()
        assert result.returncode == 1
        assert f"{PROBE_PACKAGE}/runner.py" in result.stderr
        assert probe.is_file()
