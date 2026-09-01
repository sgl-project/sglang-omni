# SPDX-License-Identifier: Apache-2.0
"""Native MPS lifecycle smoke on one GPU.

Author: Jiaxin Deng

Codifies the lifecycle contract of ``--mps`` end to end on real hardware:
auto activation on a colocated pipeline, attach verification, a real request,
recovery after a hard kill, the shared daemon across two serve commands, and
zero residue after teardown. Quality and throughput are out of scope here;
this file exists so a lifecycle regression fails a machine, not a user.

Not wired into a workflow yet; run manually on a GPU host:
    pytest tests/test_ci/test_mps_native.py -v -s -x
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import pytest

MODEL = os.environ.get("MPS_NATIVE_CI_MODEL", "bosonai/higgs-tts-3-4b")
STATE_ROOT = Path(os.environ.get("MPS_NATIVE_CI_STATE_ROOT", "/tmp/mps-native-ci"))
HEALTH_TRIES = 150
HEALTH_INTERVAL = 5
# Dedicated CI cards start empty; a shared dev box can raise this to tolerate
# other tenants' resident memory while still catching our own leaks.
DRAIN_BASE_MIB = int(os.environ.get("MPS_NATIVE_CI_DRAIN_BASE_MIB", "2000"))

# Marked like tests/test_ci/test_tts_mps_dp2.py: both markers are needed to stay
# out of the default PR command, which runs "not benchmark and not accelerator"
# and then "accelerator and not benchmark".
pytestmark = [
    pytest.mark.benchmark,
    pytest.mark.accelerator,
    pytest.mark.skipif(
        shutil.which("nvidia-cuda-mps-control") is None
        or shutil.which("nvidia-smi") is None,
        reason="requires an NVIDIA host with MPS tooling",
    ),
]


def _serve(port: int, mps: str) -> subprocess.Popen:
    env = {**os.environ, "SGLANG_OMNI_MPS_STATE_ROOT": str(STATE_ROOT)}
    log = open(f"/tmp/mps-native-ci-{port}.log", "ab")
    return subprocess.Popen(
        [
            sys.executable,
            "-m",
            "sglang_omni.cli",
            "serve",
            "--model-path",
            MODEL,
            "--mps",
            mps,
            "--mem-fraction-static",
            "0.45",
            # DP replicas budget KV explicitly; the second serve profiles a
            # shrunken free pool (see the KV sizing note in mps_dp.md).
            "--tts_engine.engine.max_total_tokens",
            "20000",
            "--port",
            str(port),
        ],
        env=env,
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )


def _wait_healthy(port: int, proc: subprocess.Popen) -> None:
    for _ in range(HEALTH_TRIES):
        if proc.poll() is not None:
            raise AssertionError(f"serve on port {port} died during startup")
        try:
            with urllib.request.urlopen(
                f"http://localhost:{port}/v1/models", timeout=3
            ):
                return
        except OSError:
            time.sleep(HEALTH_INTERVAL)
    tail = Path(f"/tmp/mps-native-ci-{port}.log").read_text(errors="replace")[-2000:]
    raise AssertionError(
        f"serve on port {port} never became healthy; log tail:\n{tail}"
    )


def _request_ok(port: int) -> None:
    body = json.dumps(
        {"model": MODEL, "input": "MPS native CI check.", "response_format": "wav"}
    ).encode()
    req = urllib.request.Request(
        f"http://localhost:{port}/v1/audio/speech",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=300) as resp:
        assert resp.status == 200
        assert len(resp.read()) > 1000


def _terminate(proc: subprocess.Popen, timeout: float = 120.0) -> None:
    if proc.poll() is None:
        proc.send_signal(signal.SIGTERM)
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=30)
        raise AssertionError("serve did not exit on SIGTERM")


def _assert_no_residue() -> None:
    leftovers = (
        [
            entry.name
            for entry in STATE_ROOT.iterdir()
            if not entry.name.startswith(".lock")
        ]
        if STATE_ROOT.exists()
        else []
    )
    assert leftovers == [], f"MPS state residue: {leftovers}"


def _daemon_pids() -> set[int]:
    # The daemon's cmdline has no path in it (the pipe dir travels via env),
    # so match /proc environ instead of pgrep -f, which would also match this
    # test's own wrapper shell.
    pids: set[int] = set()
    for environ_file in Path("/proc").glob("[0-9]*/environ"):
        try:
            env = environ_file.read_bytes()
            cmd = (environ_file.parent / "cmdline").read_bytes()
        except OSError:
            continue
        if str(STATE_ROOT).encode() in env and b"cuda-mps-control" in cmd:
            pids.add(int(environ_file.parent.name))
    return pids


@dataclass(frozen=True)
class _ProcessIdentity:
    pid: int
    start_time: int


def _process_identity(pid: int) -> tuple[_ProcessIdentity, str] | None:
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        return _ProcessIdentity(pid, int(fields[19])), fields[0]
    except (FileNotFoundError, IndexError, ValueError):
        return None


def _mps_process_identities() -> set[_ProcessIdentity]:
    """Capture exact MPS processes while their scoped environment is readable."""

    identities: set[_ProcessIdentity] = set()
    for environ_file in Path("/proc").glob("[0-9]*/environ"):
        try:
            env = environ_file.read_bytes()
            cmd = (environ_file.parent / "cmdline").read_bytes()
        except OSError:
            continue
        if str(STATE_ROOT).encode() not in env or b"nvidia-cuda-mps" not in cmd:
            continue
        identity_and_state = _process_identity(int(environ_file.parent.name))
        if identity_and_state is not None:
            identities.add(identity_and_state[0])
    return identities


def _assert_process_identities_gone(
    identities: set[_ProcessIdentity],
    timeout_s: float = 10.0,
) -> None:
    deadline = time.monotonic() + timeout_s
    while True:
        leftovers: dict[_ProcessIdentity, str] = {}
        for identity in identities:
            current = _process_identity(identity.pid)
            if current is not None and current[0] == identity:
                leftovers[identity] = current[1]
        if not leftovers:
            return
        if time.monotonic() >= deadline:
            raise AssertionError(
                f"MPS process residue (state is /proc status): {leftovers}. "
                "If these are zombies reparented to PID 1, run the container "
                "with a child-reaping init (for Docker, --init)."
            )
        time.sleep(0.1)


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _wait_gpu_drained(timeout_s: int = 180) -> None:
    # CUDA memory release lags SIGTERM, so back-to-back tests must wait for
    # the previous serve's footprint to drain before profiling memory.
    device = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]
    used = "unknown"
    for _ in range(timeout_s // 5):
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
                "-i",
                device,
            ],
            capture_output=True,
            text=True,
        )
        used = result.stdout.strip()
        try:
            if int(used) < DRAIN_BASE_MIB:
                return
        except ValueError:
            return
        time.sleep(5)
    raise AssertionError(f"GPU {device} still holds {used} MiB before test start")


@pytest.fixture(autouse=True)
def clean_state_root(monkeypatch):
    # Never erase the only control path or diagnostics from an interrupted run.
    # A clean root may still contain stale per-GPU lock files, which are safe to
    # discard only after both state and daemon checks pass.
    _assert_no_residue()
    stale_daemons = _daemon_pids()
    assert not stale_daemons, (
        f"stale MPS daemon(s) from an earlier run: {stale_daemons}; "
        f"preserving {STATE_ROOT} for scoped operator cleanup"
    )
    shutil.rmtree(STATE_ROOT, ignore_errors=True)
    _wait_gpu_drained()
    processes: list[subprocess.Popen] = []
    session_ids: set[int] = set()
    mps_processes: set[_ProcessIdentity] = set()
    serve = _serve
    daemon_pids = _daemon_pids

    def tracked_daemon_pids() -> set[int]:
        mps_processes.update(_mps_process_identities())
        return daemon_pids()

    def tracked_serve(port: int, mps: str) -> subprocess.Popen:
        proc = serve(port, mps)
        processes.append(proc)
        session_ids.add(proc.pid)
        return proc

    monkeypatch.setattr(sys.modules[__name__], "_serve", tracked_serve)
    monkeypatch.setattr(sys.modules[__name__], "_daemon_pids", tracked_daemon_pids)
    yield

    mps_processes.update(_mps_process_identities())
    for proc in processes:
        if proc.poll() is None:
            try:
                _terminate(proc, timeout=10)
            except AssertionError:
                pass
    if list(STATE_ROOT.glob("*/pipe")):
        _operator_cleanup(session_ids)
    else:
        _signal_test_sessions(session_ids, signal.SIGKILL)
        _assert_process_identities_gone(mps_processes)
        shutil.rmtree(STATE_ROOT, ignore_errors=True)
    _assert_process_identities_gone(mps_processes)


def _signal_test_sessions(session_ids: Iterable[int], sig: int) -> set[int]:
    """Signal only process groups created by this test and return matches."""

    signalled: set[int] = set()
    for session_id in set(session_ids):
        try:
            os.killpg(session_id, sig)
        except ProcessLookupError:
            continue
        signalled.add(session_id)
    return signalled


def _operator_cleanup(session_ids: int | Iterable[int]) -> None:
    """Clean only this test's process groups, preserving MPS signal order."""

    from sglang_omni.mps.control import SubprocessMpsControlClient

    control = SubprocessMpsControlClient()
    pipe_dirs = sorted(STATE_ROOT.glob("*/pipe"))
    sessions = {session_ids} if isinstance(session_ids, int) else set(session_ids)
    mps_processes = _mps_process_identities()

    # Every test serve owns a dedicated process group. Freezing them closes the
    # attach window while keeping every signal scoped to this test.
    live_sessions = _signal_test_sessions(sessions, signal.SIGSTOP)

    if not pipe_dirs or not _daemon_pids():
        _signal_test_sessions(live_sessions, signal.SIGKILL)
        _assert_process_identities_gone(mps_processes)
        shutil.rmtree(STATE_ROOT, ignore_errors=True)
        return

    daemon_pids: dict[Path, int] = {}
    try:
        for pipe_dir in pipe_dirs:
            daemon_pids[pipe_dir] = control.read_daemon_identity(pipe_dir)
            control.snapshot(pipe_dir)
    finally:
        _signal_test_sessions(live_sessions, signal.SIGKILL)

    for pipe_dir in daemon_pids:
        deadline = time.monotonic() + 10
        while remaining := control.snapshot(pipe_dir):
            if time.monotonic() >= deadline:
                raise AssertionError(
                    f"MPS clients remain after owned process-group kill: {remaining}"
                )
            time.sleep(0.1)

    for pipe_dir, daemon_pid in daemon_pids.items():
        control.quit_daemon(pipe_dir)
        deadline = time.monotonic() + 10
        while control.daemon_process_alive(daemon_pid):
            if time.monotonic() >= deadline:
                raise AssertionError(f"MPS daemon {daemon_pid} survived quit")
            time.sleep(0.1)
    _assert_process_identities_gone(mps_processes)
    shutil.rmtree(STATE_ROOT)


def test_hard_kill_fails_fast_until_operator_cleans():
    port = _free_port()
    proc = _serve(port, "auto")
    try:
        _wait_healthy(port, proc)
        assert _daemon_pids(), "no MPS daemon started for the colocated pipeline"
        _request_ok(port)

        dirty_session = proc.pid
        proc.kill()
        proc.wait(timeout=30)
        time.sleep(10)

        # Dirty state must fail the next start with the full picture instead
        # of being silently repaired.
        port = _free_port()
        proc = _serve(port, "auto")
        assert proc.wait(timeout=180) != 0, "serve started over dirty MPS state"
        log = Path(f"/tmp/mps-native-ci-{port}.log").read_text(errors="replace")
        assert "dirty state" in log and "rm -rf" in log

        # After the documented operator cleanup, startup succeeds again.
        _operator_cleanup(dirty_session)
        port = _free_port()
        proc = _serve(port, "auto")
        _wait_healthy(port, proc)
        _request_ok(port)
    finally:
        _terminate(proc)
    _assert_no_residue()
    assert not _daemon_pids(), "daemon outlived the last serve"


def test_two_serves_share_one_daemon():
    port_a, port_b = _free_port(), _free_port()
    proc_a = _serve(port_a, "on")
    proc_b: subprocess.Popen | None = None
    try:
        _wait_healthy(port_a, proc_a)
        proc_b = _serve(port_b, "on")
        _wait_healthy(port_b, proc_b)

        daemons = _daemon_pids()
        assert len(daemons) == 1, f"expected one shared daemon, saw {daemons}"
        _request_ok(port_a)
        _request_ok(port_b)

        # First leaver must not take the daemon down under the survivor.
        _terminate(proc_a)
        assert _daemon_pids() == daemons, "daemon died when one owner left"
        _request_ok(port_b)
    finally:
        if proc_b is not None:
            _terminate(proc_b)
        if proc_a.poll() is None:
            _terminate(proc_a)
    _assert_no_residue()
    assert not _daemon_pids(), "daemon outlived the last owner"
