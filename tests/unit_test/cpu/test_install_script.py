# SPDX-License-Identifier: Apache-2.0
"""install_cpu.sh must never destroy the user's pyproject.toml.

The script swaps pyproject_cpu.toml over pyproject.toml and restores it from a
fixed backup path on exit. INT/TERM are trapped, but a SIGKILL, an OOM kill, or a
failed restore leaves the swap in place with the backup holding the only copy of
the original. Re-running then used to copy the already-swapped CPU manifest over
that backup, losing the original for good.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

_SCRIPT = Path("scripts/cpu/install_cpu.sh")
_ORIGINAL_MARKER = "# ORIGINAL-CPU-MANIFEST"


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """A throwaway repo root holding just what the script touches."""
    root = tmp_path / "repo"
    (root / "scripts" / "cpu").mkdir(parents=True)
    shutil.copy(_SCRIPT, root / "scripts" / "cpu" / "install_cpu.sh")
    (root / "pyproject.toml").write_text(f'{_ORIGINAL_MARKER}\n[project]\nname = "x"\n')
    (root / "pyproject_cpu.toml").write_text('[project]\nname = "x-cpu"\n')
    return root


def _write_preflight_python(repo: Path) -> Path:
    """Return a fake interpreter that makes installer preflights deterministic."""
    python = repo / "preflight-python"
    python.write_text(
        """#!/usr/bin/env bash
if [[ "${1:-}" == "-c" ]]; then
  echo "$0"
fi
if [[ "${1:-}" == "-" ]]; then
  cat >/dev/null
fi
exit 0
"""
    )
    python.chmod(0o755)
    return python


def _run(repo: Path) -> subprocess.CompletedProcess[str]:
    python = _write_preflight_python(repo)
    return subprocess.run(
        ["bash", "scripts/cpu/install_cpu.sh", "--check"],
        cwd=repo,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,  # the refusal path exits non-zero on purpose
        env={**os.environ, "PYTHON": str(python)},
    )


def _write_signal_python(repo: Path) -> Path:
    """Return a fake interpreter that terminates its installer parent."""
    python = repo / "signal-python"
    python.write_text(
        """#!/usr/bin/env bash
if [[ "${1:-}" == "-c" ]]; then
  echo "$0"
  exit 0
fi
if [[ "${1:-}" == "-" ]]; then
  cat >/dev/null
  exit 0
fi
if [[ "${1:-}" == "-m" && "${2:-}" == "pip" ]]; then
  kill -TERM "$PPID"
  sleep 1
  exit 0
fi
exit 0
"""
    )
    python.chmod(0o755)
    return python


def test_rerun_after_an_interrupted_swap_preserves_the_original(repo: Path) -> None:
    """Reproduce the kill: swapped manifest in place, original only in the backup."""
    backup = repo / ".pyproject.cpu.bak"
    shutil.copy(repo / "pyproject.toml", backup)  # what the script does first
    shutil.copy(repo / "pyproject_cpu.toml", repo / "pyproject.toml")  # then swaps
    # <-- process killed here, before restore()

    result = _run(repo)

    # The only copy of the user's manifest must survive the re-run untouched.
    assert _ORIGINAL_MARKER in backup.read_text()
    assert result.returncode != 0
    assert "leftover backup" in result.stderr
    # And the message must say how to get back.
    assert "cp" in result.stderr and ".pyproject.cpu.bak" in result.stderr
    # git checkout would restore the committed manifest and discard any uncommitted
    # changes the backup still holds, so it must not be offered as an alternative.
    assert "git checkout" not in result.stderr


def test_a_clean_tree_still_runs(repo: Path) -> None:
    """No backup present: the guard must not block the normal path."""
    result = _run(repo)

    assert "leftover backup" not in result.stderr
    assert (repo / "pyproject.toml").read_text().startswith(_ORIGINAL_MARKER)


def test_a_partial_staging_backup_is_discarded(repo: Path) -> None:
    """A kill during the staging copy leaves no authoritative backup to restore."""
    staging = repo / ".pyproject.cpu.bak.tmp"
    staging.write_text("partial backup")

    result = _run(repo)

    assert result.returncode == 0
    assert not staging.exists()
    assert _ORIGINAL_MARKER in (repo / "pyproject.toml").read_text()


def test_a_second_run_refuses_while_the_lock_is_held(repo: Path) -> None:
    """The leftover-backup check alone is a TOCTOU guard: two runs could both pass it
    and the second would overwrite the first's backup after the swap, losing the
    original manifest. The lock must serialize the whole section.
    """
    lock = repo / ".pyproject.cpu.lock"
    lock.touch()
    holder = subprocess.Popen(["flock", str(lock), "sleep", "10"])
    try:
        # give flock time to acquire before the script tries
        import time

        time.sleep(1)
        result = _run(repo)
    finally:
        holder.kill()
        holder.wait()

    assert result.returncode != 0
    assert "holds" in result.stderr
    # The original manifest is untouched, and no backup was created.
    assert _ORIGINAL_MARKER in (repo / "pyproject.toml").read_text()
    assert not (repo / ".pyproject.cpu.bak").exists()


def test_term_restores_the_manifest_and_exits(repo: Path) -> None:
    """TERM during installation must restore once and stop with signal status."""
    python = _write_signal_python(repo)

    result = subprocess.run(
        ["bash", "scripts/cpu/install_cpu.sh"],
        cwd=repo,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
        env={**os.environ, "PYTHON": str(python)},
    )

    assert result.returncode == 143
    assert _ORIGINAL_MARKER in (repo / "pyproject.toml").read_text()
    assert not (repo / ".pyproject.cpu.bak").exists()
    assert not (repo / ".pyproject.cpu.bak.tmp").exists()
