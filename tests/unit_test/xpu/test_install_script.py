# SPDX-License-Identifier: Apache-2.0
"""install_xpu.sh must never destroy the user's pyproject.toml.

The script swaps pyproject_xpu.toml over pyproject.toml and restores it from a
fixed backup path on exit. INT/TERM are trapped, but a SIGKILL, an OOM kill, or a
failed restore leaves the swap in place with the backup holding the only copy of
the original. Re-running then used to copy the already-swapped XPU manifest over
that backup, losing the original for good.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

_SCRIPT = Path("scripts/xpu/install_xpu.sh")
_ORIGINAL_MARKER = "# ORIGINAL-CUDA-MANIFEST"


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """A throwaway repo root holding just what the script touches."""
    root = tmp_path / "repo"
    (root / "scripts" / "xpu").mkdir(parents=True)
    shutil.copy(_SCRIPT, root / "scripts" / "xpu" / "install_xpu.sh")
    (root / "pyproject.toml").write_text(f'{_ORIGINAL_MARKER}\n[project]\nname = "x"\n')
    (root / "pyproject_xpu.toml").write_text('[project]\nname = "x-xpu"\n')
    return root


def _run(repo: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", "scripts/xpu/install_xpu.sh", "--check"],
        cwd=repo,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,  # the refusal path exits non-zero on purpose
    )


def test_rerun_after_an_interrupted_swap_preserves_the_original(repo: Path) -> None:
    """Reproduce the kill: swapped manifest in place, original only in the backup."""
    backup = repo / ".pyproject.cuda.bak"
    shutil.copy(repo / "pyproject.toml", backup)  # what the script does first
    shutil.copy(repo / "pyproject_xpu.toml", repo / "pyproject.toml")  # then swaps
    # <-- process killed here, before restore()

    result = _run(repo)

    # The only copy of the user's manifest must survive the re-run untouched.
    assert _ORIGINAL_MARKER in backup.read_text()
    assert result.returncode != 0
    assert "leftover backup" in result.stderr
    # And the message must say how to get back.
    assert "cp" in result.stderr and ".pyproject.cuda.bak" in result.stderr
    # git checkout would restore the committed manifest and discard any uncommitted
    # changes the backup still holds, so it must not be offered as an alternative.
    assert "git checkout" not in result.stderr


def test_a_clean_tree_still_runs(repo: Path) -> None:
    """No backup present: the guard must not block the normal path."""
    result = _run(repo)

    assert "leftover backup" not in result.stderr
    assert (repo / "pyproject.toml").read_text().startswith(_ORIGINAL_MARKER)


def test_a_second_run_refuses_while_the_lock_is_held(repo: Path) -> None:
    """The leftover-backup check alone is a TOCTOU guard: two runs could both pass it
    and the second would overwrite the first's backup after the swap, losing the
    original manifest. The lock must serialize the whole section.
    """
    lock = repo / ".pyproject.xpu.lock"
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
    assert not (repo / ".pyproject.cuda.bak").exists()
