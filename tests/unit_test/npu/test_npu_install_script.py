# SPDX-License-Identifier: Apache-2.0
"""Test safety and argument handling for the Ascend NPU install helper."""

from __future__ import annotations

import builtins
import importlib.metadata
import os
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest

from scripts.npu.config import read_config

SCRIPT = Path("scripts/npu/install_npu.sh")
ORIGINAL_MARKER = "# ORIGINAL-CUDA-MANIFEST"
VERSION = read_config()[0]["sglang-version"]


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    (root / "scripts" / "npu").mkdir(parents=True)
    shutil.copy(SCRIPT, root / "scripts" / "npu" / "install_npu.sh")
    shutil.copy(SCRIPT.with_name("config.py"), root / "scripts" / "npu" / "config.py")
    (root / "pyproject.toml").write_text(f'{ORIGINAL_MARKER}\n[project]\nname = "x"\n')
    shutil.copy("pyproject_npu.toml", root / "pyproject_npu.toml")

    # A space in the executable path catches accidental shell word splitting.
    fake_python = root / "fake python"
    fake_python.write_text(
        "#!/usr/bin/env bash\n"
        'if [[ "$1" == */config.py || ( "$1" == "-" && $# -ge 2 ) ]]; then\n'
        f'  exec {shlex.quote(sys.executable)} "$@"\n'
        'elif [[ "$1" == "-c" && "$2" == *\'version("sglang")\'* ]]; then\n'
        '  [[ "${FAKE_SGLANG_INSTALLED:-1}" == "1" ]] || exit 1\n'
        f"  printf '%s\\n' \"${{FAKE_SGLANG_VERSION:-{VERSION}}}\"\n"
        'elif [[ "$1" == "-c" ]]; then\n'
        "  printf '%s\\n' \"$0\"\n"
        "fi\n"
        "exit 0\n"
    )
    fake_python.chmod(0o755)
    return root


def run(
    repo: Path, *args: str, env_overrides: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["PYTHON"] = str(repo / "fake python")
    env.update(env_overrides or {})
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

    result = run(repo, "--check")

    assert result.returncode != 0
    assert ORIGINAL_MARKER in backup.read_text()
    assert "leftover backup" in result.stderr
    assert "git checkout" not in result.stderr


def test_clean_dry_run_does_not_modify_manifest(repo: Path) -> None:
    result = run(repo, "--check")

    assert result.returncode == 0
    assert "would run" in result.stdout
    assert (repo / "pyproject.toml").read_text().startswith(ORIGINAL_MARKER)
    assert not (repo / ".pyproject.cuda.bak").exists()


def test_default_install_resolves_project_dependencies(repo: Path) -> None:
    result = run(repo, "--check")

    assert result.returncode == 0
    assert "--no-build-isolation" not in result.stdout
    assert "--no-deps" not in result.stdout
    assert "apt-get" not in result.stdout


def test_non_editable_install_restores_manifest(repo: Path) -> None:
    result = run(
        repo,
        "--no-editable",
        "--skip-device-check",
    )

    assert result.returncode == 0, result.stderr
    command = next(
        line for line in result.stdout.splitlines() if line.startswith(">>> ")
    )
    assert shlex.split(command[4:]) == [
        str(repo / "fake python"),
        "-m",
        "pip",
        "install",
        ".",
    ]
    assert (repo / "pyproject.toml").read_text().startswith(ORIGINAL_MARKER)
    assert not (repo / ".pyproject.cuda.bak").exists()


def test_docker_dry_run_lists_dependencies_without_installing(repo: Path) -> None:
    result = run(
        repo,
        "--install-system-deps",
        "--with-qwen-tts",
        "--no-editable",
        "--skip-device-check",
        "--check",
    )

    assert result.returncode == 0, result.stderr
    assert "apt-get install -y --no-install-recommends ffmpeg=" in result.stdout
    assert "libsndfile1=" in result.stdout
    assert "sox=" in result.stdout
    assert "-m pip install --no-deps qwen-tts==0.1.1" in result.stdout
    assert "editable:    no" in result.stdout
    assert (repo / "pyproject.toml").read_text().startswith(ORIGINAL_MARKER)


def test_extras_resolve_project_dependencies(repo: Path) -> None:
    result = run(repo, "--extras", "eval", "--check")

    assert result.returncode == 0, result.stderr
    assert "--constraint" not in result.stdout
    assert "--no-deps" not in result.stdout
    assert ".[eval]" in result.stdout.replace("\\", "")


@pytest.mark.parametrize("failure", [None, "missing", "mismatch"])
def test_device_free_precheck_uses_metadata(monkeypatch, failure) -> None:
    source = SCRIPT.read_text().split("\"${PYBIN}\" - <<'PY'\n", 1)[1]
    source = source.split("\nPY\n", 1)[0]
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        assert name not in {"torch", "torch_npu", "triton", "sgl_kernel_npu"}
        return original_import(name, *args, **kwargs)

    def version(package):
        if package == "torch_npu":
            if failure == "missing":
                raise importlib.metadata.PackageNotFoundError(package)
            if failure == "mismatch":
                return "2.9.0"
        return "2.10.0"

    monkeypatch.setenv("SGLANG_OMNI_SKIP_NPU_DEVICE_CHECK", "1")
    monkeypatch.setattr(importlib.metadata, "version", version)
    monkeypatch.setattr(builtins, "__import__", guarded_import)
    with pytest.raises(SystemExit) as result:
        exec(compile(source, str(SCRIPT), "exec"), {})
    assert result.value.code == (0 if failure is None else 1)


@pytest.mark.parametrize(
    "installed",
    [
        f"{VERSION}.dev7+gec43c1f20",
        f"{VERSION}rc1",
        VERSION,
        f"{VERSION}+ascend",
        f"{VERSION}.post1",
        f"{VERSION}.1",
    ],
)
def test_matching_sglang_version_is_accepted(repo: Path, installed: str) -> None:
    result = run(repo, "--check", env_overrides={"FAKE_SGLANG_VERSION": installed})

    assert result.returncode == 0
    assert f"sglang:      {installed}" in result.stdout


@pytest.mark.parametrize(
    "installed",
    [
        "0.0.1",
        "0.0.1.post1",
        "99.0.0",
        "99.0.0.dev1",
    ],
)
def test_mismatched_sglang_version_is_rejected(repo: Path, installed: str) -> None:
    result = run(repo, "--check", env_overrides={"FAKE_SGLANG_VERSION": installed})

    assert result.returncode != 0
    assert f"supported: {VERSION} release line" in result.stderr
    assert f"installed: {installed}" in result.stderr
    assert "would run" not in result.stdout


def test_missing_sglang_is_rejected(repo: Path) -> None:
    result = run(repo, "--check", env_overrides={"FAKE_SGLANG_INSTALLED": "0"})

    assert result.returncode != 0
    assert f"supported: {VERSION} release line" in result.stderr
    assert "installed: not installed" in result.stderr
    assert "would run" not in result.stdout


def test_supported_version_is_read_from_manifest(repo: Path) -> None:
    manifest = repo / "pyproject_npu.toml"
    manifest.write_text(manifest.read_text().replace(VERSION, "99.0.0"))
    result = run(repo, "--check", env_overrides={"FAKE_SGLANG_VERSION": "99.0.0"})
    assert result.returncode == 0, result.stderr
    assert "would run" in result.stdout


@pytest.mark.parametrize(
    "extra", ["eval", "all", "fun-cosyvoice3", "eval,fun-cosyvoice3"]
)
def test_supported_extras_are_preserved_as_one_argument(repo: Path, extra: str) -> None:
    result = run(repo, "--check", "--extras", extra)

    assert result.returncode == 0
    # Bash versions differ on whether printf %q escapes commas.
    assert f".\\[{extra}\\]" in result.stdout.replace("\\,", ",")


def test_unknown_extra_is_rejected(repo: Path) -> None:
    result = run(repo, "--check", "--extras", "eval] --index-url bad [")

    assert result.returncode == 2
    assert "unsupported extra" in result.stderr


def test_missing_extra_value_is_rejected(repo: Path) -> None:
    result = run(repo, "--extras")

    assert result.returncode == 2
    assert "requires a value" in result.stderr


def test_skip_device_check_flag_is_accepted(repo: Path) -> None:
    result = run(repo, "--check", "--skip-device-check")

    assert result.returncode == 0


def test_second_run_refuses_while_lock_is_held(repo: Path) -> None:
    lock = repo / ".pyproject.npu.lock"
    lock.touch()
    holder = subprocess.Popen(["flock", str(lock), "sleep", "10"])
    try:
        time.sleep(0.2)
        result = run(repo, "--check")
    finally:
        holder.kill()
        holder.wait()

    assert result.returncode != 0
    assert "holds" in result.stderr
    assert ORIGINAL_MARKER in (repo / "pyproject.toml").read_text()
    assert not (repo / ".pyproject.cuda.bak").exists()
