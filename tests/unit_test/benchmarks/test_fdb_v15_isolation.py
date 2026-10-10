# SPDX-License-Identifier: Apache-2.0
"""GPU and port isolation for concurrent Full-Duplex-Bench jobs."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from benchmarks.duplex.fdb_v15.common import (
    DEFAULT_JUDGE_PORT,
    DEFAULT_SERVER_PORT,
    JOB_PORT_STRIDE,
    NCCL_PORT_END,
    load_settings,
)
from sglang_omni.utils.port_claim import claim_tcp_port, release_tcp_port

CLAIM_BASE_PORT = 25100
CLAIM_SPAN = 20
_DEAD_PID_SCAN_START = 1_000_000
_DEAD_PID_SCAN_COUNT = 200


def dead_pid() -> int:
    for pid in range(_DEAD_PID_SCAN_START, _DEAD_PID_SCAN_START + _DEAD_PID_SCAN_COUNT):
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return pid
        except PermissionError:
            continue
    raise AssertionError("no unused pid in the scan range")


@pytest.fixture
def clean_job_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("CUDA_VISIBLE_DEVICES", "GPU", "SERVER_PORT", "JUDGE_PORT"):
        monkeypatch.delenv(name, raising=False)


def test_default_job_uses_gpu_zero(clean_job_env: None) -> None:
    settings = load_settings("isolation")
    assert settings.gpu == "0"
    assert settings.server_port == DEFAULT_SERVER_PORT
    assert settings.judge_port == DEFAULT_JUDGE_PORT
    assert settings.judge_nccl_port == DEFAULT_JUDGE_PORT + 1
    assert settings.judge_dir.name == f"port-{DEFAULT_JUDGE_PORT}"


def test_visible_device_derives_a_disjoint_port_block(
    clean_job_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    settings = load_settings("isolation")
    assert settings.gpu == "1"
    assert settings.server_port == DEFAULT_SERVER_PORT + JOB_PORT_STRIDE
    assert settings.judge_port == DEFAULT_JUDGE_PORT + JOB_PORT_STRIDE
    assert settings.judge_dir.name == f"port-{settings.judge_port}"


def test_explicit_ports_override_the_gpu_offset(
    clean_job_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    monkeypatch.setenv("SERVER_PORT", "8097")
    monkeypatch.setenv("JUDGE_PORT", "30000")
    settings = load_settings("isolation")
    assert settings.server_port == DEFAULT_SERVER_PORT
    assert settings.judge_port == DEFAULT_JUDGE_PORT


def test_two_visible_devices_are_rejected(
    clean_job_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    with pytest.raises(SystemExit, match="exactly one GPU"):
        load_settings("isolation")


def test_gpu_and_visible_device_must_agree(
    clean_job_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    monkeypatch.setenv("GPU", "0")
    with pytest.raises(SystemExit, match="disagree"):
        load_settings("isolation")


def test_gpu_without_visible_devices_still_offsets_ports(
    clean_job_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("GPU", "1")
    settings = load_settings("isolation")
    assert settings.gpu == "1"
    assert settings.server_port == DEFAULT_SERVER_PORT + JOB_PORT_STRIDE


def test_judge_ports_stay_above_the_nccl_claim_range() -> None:
    for gpu_index in range(8):
        judge_port = DEFAULT_JUDGE_PORT + gpu_index * JOB_PORT_STRIDE
        assert judge_port >= NCCL_PORT_END
        assert judge_port + 1 >= NCCL_PORT_END


def test_port_inside_nccl_range_is_rejected(
    clean_job_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("JUDGE_PORT", "29600")
    with pytest.raises(SystemExit, match="NCCL"):
        load_settings("isolation")


def test_concurrent_claims_take_different_ports(tmp_path: Path) -> None:
    script = (
        "import os\n"
        "import time\n"
        "from pathlib import Path\n"
        "from sglang_omni.utils.port_claim import claim_tcp_port\n"
        f"port = claim_tcp_port({CLAIM_BASE_PORT}, {CLAIM_SPAN})\n"
        "marker = Path(os.environ['SGLANG_OMNI_PORT_CLAIM_DIR']) "
        "/ f'held-{os.getpid()}'\n"
        "marker.write_text(str(port))\n"
        "deadline = time.time() + 10\n"
        "while time.time() < deadline:\n"
        "    held = list(Path(os.environ['SGLANG_OMNI_PORT_CLAIM_DIR']).glob('held-*'))\n"
        "    if len(held) >= 2:\n"
        "        break\n"
        "    time.sleep(0.05)\n"
        "print(port)\n"
    )
    child_env = os.environ.copy()
    child_env["SGLANG_OMNI_PORT_CLAIM_DIR"] = str(tmp_path)
    processes = [
        subprocess.Popen(
            [sys.executable, "-c", script],
            cwd=Path(__file__).resolve().parents[3],
            env=child_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        for _ in range(2)
    ]
    claimed_ports = []
    for process in processes:
        stdout, stderr = process.communicate(timeout=30)
        assert process.returncode == 0, stderr
        claimed_ports.append(int(stdout.strip()))
    assert claimed_ports[0] != claimed_ports[1]
    assert all(
        CLAIM_BASE_PORT <= port < CLAIM_BASE_PORT + CLAIM_SPAN for port in claimed_ports
    )


def test_live_claim_is_skipped_and_stale_claim_is_reused(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("SGLANG_OMNI_PORT_CLAIM_DIR", str(tmp_path))
    live_port = CLAIM_BASE_PORT
    (tmp_path / str(live_port)).write_text(f"{os.getpid()}\n")
    claimed = claim_tcp_port(CLAIM_BASE_PORT, 2)
    assert claimed == live_port + 1
    release_tcp_port(claimed)

    stale_port = CLAIM_BASE_PORT + 2
    (tmp_path / str(stale_port)).write_text(f"{dead_pid()}\n")
    reclaimed = claim_tcp_port(stale_port, 1)
    assert reclaimed == stale_port
    release_tcp_port(reclaimed)
