# SPDX-License-Identifier: Apache-2.0
"""Safety and argument tests for the Ascend NPU install helper."""

from __future__ import annotations

import os
import shutil
import subprocess
import time
from pathlib import Path

import pytest

_SCRIPT = Path("scripts/npu/install_npu.sh")
_ORIGINAL_MARKER = "# ORIGINAL-CUDA-MANIFEST"


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    (root / "scripts" / "npu").mkdir(parents=True)
    shutil.copy(_SCRIPT, root / "scripts" / "npu" / "install_npu.sh")
    (root / "pyproject.toml").write_text(f'{_ORIGINAL_MARKER}\n[project]\nname = "x"\n')
    (root / "pyproject_npu.toml").write_text('[project]\nname = "x-npu"\n')

    # A space in the executable path catches accidental shell word splitting.
    fake_python = root / "fake python"
    fake_python.write_text(
        "#!/usr/bin/env bash\n"
        'if [[ "$1" == "-c" ]]; then printf \'%s\\n\' "$0"; fi\n'
        "exit 0\n"
    )
    fake_python.chmod(0o755)
    return root


def _run(repo: Path, *args: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["PYTHON"] = str(repo / "fake python")
    return subprocess.run(
        ["bash", "scripts/npu/install_npu.sh", *args],
        cwd=repo,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )


def test_rerun_after_interrupted_swap_preserves_original(repo: Path) -> None:
    backup = repo / ".pyproject.cuda.bak"
    shutil.copy(repo / "pyproject.toml", backup)
    shutil.copy(repo / "pyproject_npu.toml", repo / "pyproject.toml")

    result = _run(repo, "--check")

    assert result.returncode != 0
    assert _ORIGINAL_MARKER in backup.read_text()
    assert "leftover backup" in result.stderr
    assert "git checkout" not in result.stderr


def test_clean_dry_run_does_not_modify_manifest(repo: Path) -> None:
    result = _run(repo, "--check")

    assert result.returncode == 0
    assert "would run" in result.stdout
    assert (repo / "pyproject.toml").read_text().startswith(_ORIGINAL_MARKER)
    assert not (repo / ".pyproject.cuda.bak").exists()


@pytest.mark.parametrize(
    "extra", ["eval", "all", "fun-cosyvoice3", "eval,fun-cosyvoice3"]
)
def test_supported_extras_are_preserved_as_one_argument(repo: Path, extra: str) -> None:
    result = _run(repo, "--check", "--extras", extra)

    assert result.returncode == 0
    assert f".\\[{extra}\\]" in result.stdout


def test_unknown_extra_is_rejected(repo: Path) -> None:
    result = _run(repo, "--check", "--extras", "eval] --index-url bad [")

    assert result.returncode == 2
    assert "unsupported extra" in result.stderr


def test_missing_extra_value_is_rejected(repo: Path) -> None:
    result = _run(repo, "--extras")

    assert result.returncode == 2
    assert "requires a value" in result.stderr


def test_skip_device_check_flag_is_accepted(repo: Path) -> None:
    result = _run(repo, "--check", "--skip-device-check")

    assert result.returncode == 0


def test_second_run_refuses_while_lock_is_held(repo: Path) -> None:
    lock = repo / ".pyproject.npu.lock"
    lock.touch()
    holder = subprocess.Popen(["flock", str(lock), "sleep", "10"])
    try:
        time.sleep(0.2)
        result = _run(repo, "--check")
    finally:
        holder.kill()
        holder.wait()

    assert result.returncode != 0
    assert "holds" in result.stderr
    assert _ORIGINAL_MARKER in (repo / "pyproject.toml").read_text()
    assert not (repo / ".pyproject.cuda.bak").exists()
