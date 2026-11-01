# SPDX-License-Identifier: Apache-2.0

"""CPU subprocess coverage for signal-free GPU memory handoff."""

import json
import os
import shutil
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[3] / ".github/scripts/delete_gpu_process.sh"


def cleanup_environment(
    tmp_path: Path,
    memory_samples: list[str],
    child_pid: int,
    *,
    memory_status: int = 0,
    index_status: int = 0,
    index_output: str = "0",
) -> dict[str, str]:
    probe = tmp_path / "nvidia-smi"
    probe.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "from pathlib import Path\n"
        "arguments = sys.argv[1:]\n"
        "if 'CUDA_VISIBLE_DEVICES' in os.environ:\n"
        "    sys.exit(99)\n"
        "if arguments == ['--query-compute-apps=pid', '--format=csv,noheader', '--id=0']:\n"
        "    print(os.environ['OWNED_TEST_PID'])\n"
        "elif arguments == ['--query-gpu=index', '--format=csv,noheader,nounits']:\n"
        "    print(os.environ['TEST_GPU_INDEX_OUTPUT'])\n"
        "    sys.exit(int(os.environ['TEST_GPU_INDEX_STATUS']))\n"
        "elif arguments == ['--query-gpu=memory.used', '--format=csv,noheader,nounits', '--id=0']:\n"
        "    counter = Path(__file__).with_suffix('.count')\n"
        "    count = int(counter.read_text()) if counter.exists() else 0\n"
        "    counter.write_text(str(count + 1))\n"
        "    samples = json.loads(os.environ['TEST_GPU_MEMORY_SAMPLES'])\n"
        "    print(samples[min(count, len(samples) - 1)])\n"
        "    sys.exit(int(os.environ['TEST_GPU_MEMORY_STATUS']))\n"
        "else:\n"
        "    sys.exit(99)\n"
    )
    probe.chmod(0o755)
    return os.environ | {
        "PATH": f"{tmp_path}:{os.environ['PATH']}",
        "CUDA_VISIBLE_DEVICES": "0",
        "OMNI_CI_GPU_MEMORY_CLEAN_THRESHOLD_MB": "1024",
        "OMNI_CI_GPU_CLEAN_WAIT_SECONDS": "0",
        "OMNI_CI_GPU_CLEAN_POLL_SECONDS": "1",
        "OMNI_CI_ALLOW_UNSCOPED_GPU_CLEAN": "0",
        "GITHUB_ACTIONS": "false",
        "OWNED_TEST_PID": str(child_pid),
        "TEST_GPU_MEMORY_SAMPLES": json.dumps(memory_samples),
        "TEST_GPU_MEMORY_STATUS": str(memory_status),
        "TEST_GPU_INDEX_STATUS": str(index_status),
        "TEST_GPU_INDEX_OUTPUT": index_output,
    }


@pytest.fixture
def owned_child() -> Iterator[subprocess.Popen[str]]:
    child = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import time; print('ready', flush=True); time.sleep(60)",
        ],
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert child.stdout is not None
        assert child.stdout.readline() == "ready\n"
        yield child
    finally:
        child.terminate()
        child.wait(timeout=5)
        if child.stdout is not None:
            child.stdout.close()
        else:
            pass


@pytest.mark.parametrize(
    "flags",
    [
        ["--wait-only"],
        ["--wait-only", "--kill-orphans"],
        ["--kill-orphans", "--wait-only"],
    ],
)
def test_wait_only_never_terminates_reported_process(
    tmp_path: Path, owned_child: subprocess.Popen[str], flags: list[str]
) -> None:
    result = subprocess.run(
        ["bash", str(SCRIPT), *flags],
        env=cleanup_environment(tmp_path, ["0"], owned_child.pid),
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stderr == ""
    assert "Wait-only mode: no processes will be terminated." in result.stdout
    assert "GPU memory cleanup complete: max memory.used=0 MiB." in result.stdout
    assert owned_child.poll() is None


def test_wait_only_polls_until_strictly_below_threshold(
    tmp_path: Path, owned_child: subprocess.Popen[str]
) -> None:
    environment = cleanup_environment(
        tmp_path, [" 4096 ", "1024", "1023"], owned_child.pid
    ) | {"OMNI_CI_GPU_CLEAN_WAIT_SECONDS": "5"}
    result = subprocess.run(
        ["bash", str(SCRIPT), "--wait-only"],
        env=environment,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stderr == ""
    assert "  GPU 0: 4096 MiB used\n" in result.stdout
    assert "  GPU 0: 1024 MiB used\n" in result.stdout
    assert "  GPU 0: 1023 MiB used\n" in result.stdout
    assert result.stdout.count("  GPU 0:") == 3
    assert "GPU memory cleanup complete: max memory.used=1023 MiB." in result.stdout
    assert owned_child.poll() is None


@pytest.mark.parametrize("memory", ["", "N/A", "-1", "0\nN/A", "0\n4096"])
def test_wait_only_rejects_invalid_memory(
    tmp_path: Path, owned_child: subprocess.Popen[str], memory: str
) -> None:
    result = subprocess.run(
        ["bash", str(SCRIPT), "--wait-only"],
        env=cleanup_environment(tmp_path, [memory], owned_child.pid),
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 2, result.stdout + result.stderr
    assert "::error::Invalid GPU 0 memory reading:" in result.stderr
    assert "GPU memory cleanup complete:" not in result.stdout
    assert owned_child.poll() is None


@pytest.mark.parametrize("memory", ["1024", "4096"])
def test_wait_only_reports_busy_memory_timeout(
    tmp_path: Path, owned_child: subprocess.Popen[str], memory: str
) -> None:
    environment = cleanup_environment(tmp_path, [memory], owned_child.pid) | {
        "OMNI_CI_GPU_CLEAN_WAIT_SECONDS": "1"
    }
    result = subprocess.run(
        ["bash", str(SCRIPT), "--wait-only"],
        env=environment,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 1, result.stdout + result.stderr
    assert (
        f"::error::Timed out waiting for GPU memory.used < 1024 MiB; max memory.used={memory} MiB."
        in result.stdout
    )
    assert "GPU memory cleanup complete:" not in result.stdout
    assert result.stderr == ""
    assert owned_child.poll() is None


def test_wait_only_rejects_failed_memory_query_with_numeric_output(
    tmp_path: Path, owned_child: subprocess.Popen[str]
) -> None:
    result = subprocess.run(
        ["bash", str(SCRIPT), "--wait-only"],
        env=cleanup_environment(tmp_path, ["0"], owned_child.pid, memory_status=15),
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 2, result.stdout + result.stderr
    assert "::error::Unable to query memory for GPU 0." in result.stderr
    assert "GPU memory cleanup complete:" not in result.stdout
    assert owned_child.poll() is None


def test_wait_only_rejects_failed_gpu_discovery_with_partial_output(
    tmp_path: Path, owned_child: subprocess.Popen[str]
) -> None:
    environment = cleanup_environment(
        tmp_path, ["0"], owned_child.pid, index_status=9
    ) | {"CUDA_VISIBLE_DEVICES": "all"}
    result = subprocess.run(
        ["bash", str(SCRIPT), "--wait-only", "--kill-orphans"],
        env=environment,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 2, result.stdout + result.stderr
    assert "::error::Unable to select GPUs for cleanup." in result.stderr
    assert "GPU memory cleanup complete:" not in result.stdout
    assert owned_child.poll() is None


def test_wait_only_requires_nvidia_smi(
    tmp_path: Path, owned_child: subprocess.Popen[str]
) -> None:
    environment = cleanup_environment(tmp_path, ["0"], owned_child.pid)
    (tmp_path / "nvidia-smi").unlink()
    environment["PATH"] = str(tmp_path)
    bash_path = shutil.which("bash")
    assert bash_path is not None
    result = subprocess.run(
        [bash_path, str(SCRIPT), "--wait-only"],
        env=environment,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 2
    assert result.stderr == (
        "::error::nvidia-smi is required to verify GPU memory release.\n"
    )
    assert result.stdout == ""
    assert owned_child.poll() is None
