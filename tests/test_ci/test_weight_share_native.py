# SPDX-License-Identifier: Apache-2.0
"""Native weight-share lifecycle smoke on one GPU.

Author: Jiaxin Deng

Codifies the ``--weight-share on`` contract end to end on real hardware. Two
same-GPU replicas of one pipeline are booted twice, unshared and shared, and the
card must end up holding materially less with sharing on: that is the whole
point of the feature, and it is measured rather than inferred from a log line.
The same run pins the spawn order that keeps a follower from deadlocking against
its leader, and a second case pins the failure mode operators will actually
meet, a leader that dies while followers hold its mappings, which must fail the
pipeline rather than serve aliased memory.

Quality and throughput are out of scope; this file exists so a lifecycle
regression fails a machine, not a user.

Not wired into a workflow yet; run manually on a GPU host:
    pytest tests/test_ci/test_weight_share_native.py -v -s -x
"""

from __future__ import annotations

import json
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from collections.abc import Iterable
from pathlib import Path

import pytest

from tests.test_ci.test_mps_native import _signal_test_sessions

# Note (Jiaxin Deng): process replicas need an engine factory that declares
# gpu_id, which rules out Higgs and Whisper today; MOSS TTS local is both
# replica capable and on the weight-share allowlist, and its separate vocoder
# process also exercises the reduction-compat flag.
MODEL = os.environ.get(
    "WEIGHT_SHARE_CI_MODEL", "OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5"
)
GPU_ID = int(os.environ.get("WEIGHT_SHARE_CI_GPU", "0"))
MAX_TOTAL_TOKENS = int(os.environ.get("WEIGHT_SHARE_CI_MAX_TOTAL_TOKENS", "30000"))
HEALTH_TRIES = 150
HEALTH_INTERVAL = 5
LOG = Path("/tmp/weight-share-ci.log")

pytestmark = [
    pytest.mark.benchmark,
    pytest.mark.accelerator,
    pytest.mark.skipif(
        shutil.which("nvidia-smi") is None,
        reason="requires an NVIDIA host",
    ),
]


def _config(tmp_path: Path) -> Path:
    # The MOSS local engine shares the "pipeline" process with preprocessing,
    # so that whole process is what gets replicated; the vocoder is a separate
    # process and stays single, which also proves a non-sharing process still
    # receives the CUDA reduction compat flag. Two replicas plus the vocoder
    # must fit the placement budget: 2 x (0.05 + 0.35) + 0.15.
    path = tmp_path / "moss_local_dp2_same_gpu.yaml"
    path.write_text(
        f"""config_cls: MossTTSLocalPipelineConfig
name: mossl
model_path: {MODEL}

stages:
  preprocessing:
    gpu: {GPU_ID}
    gpu_memory_fraction: 0.05
  tts_engine:
    gpu: {GPU_ID}
    gpu_memory_fraction: 0.35
    engine:
      mem_fraction_static: 0.30
      max_total_tokens: {MAX_TOTAL_TOKENS}
  vocoder:
    gpu: {GPU_ID}
    gpu_memory_fraction: 0.15

processes:
  pipeline:
    num_replicas: 2
    replica_devices: [{GPU_ID}, {GPU_ID}]
"""
    )
    return path


def _serve(config: Path, port: int, *, weight_share: str) -> subprocess.Popen:
    log = LOG.open("ab")
    return subprocess.Popen(
        [
            sys.executable,
            "-m",
            "sglang_omni.cli",
            "serve",
            "--config",
            str(config),
            "--mps",
            "auto",
            "--weight-share",
            weight_share,
            "--port",
            str(port),
        ],
        env={**os.environ},
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )


def _wait_healthy(port: int, proc: subprocess.Popen) -> None:
    for _ in range(HEALTH_TRIES):
        if proc.poll() is not None:
            raise AssertionError(_with_log(f"serve died during startup (port {port})"))
        try:
            with urllib.request.urlopen(
                f"http://localhost:{port}/v1/models", timeout=3
            ):
                return
        except OSError:
            time.sleep(HEALTH_INTERVAL)
    raise AssertionError(_with_log(f"serve never became healthy (port {port})"))


def _with_log(message: str) -> str:
    tail = LOG.read_text(errors="replace")[-3000:] if LOG.exists() else "(no log)"
    return f"{message}; log tail:\n{tail}"


def _speech_ok(port: int) -> None:
    body = json.dumps(
        {"model": MODEL, "input": "Weight share CI check.", "response_format": "wav"}
    ).encode()
    request = urllib.request.Request(
        f"http://localhost:{port}/v1/audio/speech",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=300) as response:
        assert response.status == 200
        assert len(response.read()) > 1000


def _spawned_pids() -> dict[str, int]:
    """Read each group's pid from the runtime's own spawn log line.

    # Note (Jiaxin Deng): /proc comm truncates to 15 characters, which cuts
    # "process-pipeline@r0" short, so the log the runner already publishes is
    # the only exact mapping available from outside the serve process.
    """
    pids: dict[str, int] = {}
    if not LOG.exists():
        return pids
    for match in re.finditer(
        r"StageGroup (\S+): spawned \d+ process\(es\) \(pids=\[([0-9, ]+)\]\)",
        LOG.read_text(errors="replace"),
    ):
        pids[match.group(1)] = int(match.group(2).split(",")[0])
    return pids


def _alive(pid: int) -> bool:
    return Path(f"/proc/{pid}").exists()


def _gpu_uuid_from_log() -> str:
    """Read the physical GPU this run landed on from the runtime's lock path.

    # Note (Jiaxin Deng): nvidia-smi reports host pids, which do not match this
    # container's namespace, so per-process attribution is unavailable; the card
    # is identified by UUID instead and measured as a whole.
    """
    match = re.search(
        r"startup lock for stage \S+: \S*sglang_omni_gpu_(GPU-[0-9a-fA-F-]+)_startup",
        LOG.read_text(errors="replace"),
    )
    assert match is not None, _with_log("no GPU startup lock line in the log")
    return match.group(1)


def _gpu_used_mib(gpu_uuid: str) -> int:
    output = subprocess.run(
        ["nvidia-smi", "--query-gpu=uuid,memory.used", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    for line in output.splitlines():
        if not line.strip():
            continue
        uuid, used = (part.strip() for part in line.split(","))
        if uuid == gpu_uuid:
            return int(used)
    raise AssertionError(f"GPU {gpu_uuid} not listed by nvidia-smi")


def _terminate(proc: subprocess.Popen, timeout: float = 180.0) -> None:
    if proc.poll() is None:
        proc.send_signal(signal.SIGTERM)
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        # The serve is a session leader, so its pgid reaches every stage worker;
        # killing only the parent would leave them holding GPU memory.
        _signal_test_sessions({proc.pid}, signal.SIGKILL)
        proc.wait(timeout=30)
        raise AssertionError("serve did not exit on SIGTERM")


def _session_members(session_ids: Iterable[int]) -> dict[int, str]:
    """Pids still holding resources in one of *session_ids*, from /proc.

    # Note (Jiaxin Deng): a zombie is an unreaped exit status, not a worker; it
    # holds no GPU memory and no MPS client, and this container runs without a
    # reaping init, so counting it would fail every clean run.
    """
    wanted = set(session_ids)
    members: dict[int, str] = {}
    for stat in Path("/proc").glob("[0-9]*/stat"):
        try:
            fields = stat.read_text().rsplit(")", 1)[1].split()
            state, pgrp = fields[0], int(fields[2])
        except (OSError, IndexError, ValueError):
            continue
        if pgrp in wanted and state != "Z":
            members[int(stat.parent.name)] = state
    return members


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


@pytest.fixture(autouse=True)
def _scoped_serves(monkeypatch):
    """Track every serve this test starts and reap its whole session afterwards.

    A failed lifecycle assertion must not leave stage workers holding GPU
    memory and MPS clients for the next test; the serve parent may already be
    gone by then, so cleanup keys on the session id, not the parent pid.
    """
    LOG.unlink(missing_ok=True)
    session_ids: set[int] = set()
    serve = _serve

    def tracked_serve(*args, **kwargs) -> subprocess.Popen:
        proc = serve(*args, **kwargs)
        session_ids.add(proc.pid)
        return proc

    monkeypatch.setattr(sys.modules[__name__], "_serve", tracked_serve)
    yield
    _signal_test_sessions(session_ids, signal.SIGTERM)
    deadline = time.monotonic() + 60
    while _session_members(session_ids) and time.monotonic() < deadline:
        time.sleep(1)
    leftovers = _session_members(session_ids)
    if leftovers:
        _signal_test_sessions(session_ids, signal.SIGKILL)
        time.sleep(2)
    assert not _session_members(
        session_ids
    ), f"stage workers outlived their serve and had to be killed: {leftovers}"


def _boot_and_measure(tmp_path, *, weight_share: str) -> tuple[int, str]:
    proc = _serve(_config(tmp_path), (port := _free_port()), weight_share=weight_share)
    try:
        _wait_healthy(port, proc)
        for _ in range(4):
            _speech_ok(port)
        gpu_uuid = _gpu_uuid_from_log()
        return _gpu_used_mib(gpu_uuid), LOG.read_text(errors="replace")
    finally:
        _terminate(proc)


def test_sharing_frees_a_full_weight_copy_and_orders_the_waves(tmp_path):
    unshared, _ = _boot_and_measure(tmp_path, weight_share="off")
    LOG.unlink(missing_ok=True)
    shared, log = _boot_and_measure(tmp_path, weight_share="on")

    # MOSS local shares an 8.44 GiB backbone, so a follower that quietly loaded
    # its own copy would leave the card at the unshared footprint.
    assert shared < unshared - 4000, (unshared, shared)

    assert "Weight sharing on GPU" in log
    assert "[weight-share] follower attached" in log
    # A follower must not be spawned before its leader has exported.
    assert (
        log.index("StageGroup pipeline@r0: spawned")
        < log.index("[weight-share] leader exported")
        < log.index("StageGroup pipeline@r1: spawned")
    )
    assert not any(_alive(pid) for pid in _spawned_pids().values())


def test_a_dead_leader_fails_the_pipeline(tmp_path):
    free_port = _free_port()
    proc = _serve(_config(tmp_path), free_port, weight_share="on")
    try:
        _wait_healthy(free_port, proc)
        pids = _spawned_pids()
        leader_pid = pids["pipeline@r0"]

        os.kill(leader_pid, signal.SIGKILL)

        deadline = time.monotonic() + 120
        while time.monotonic() < deadline and proc.poll() is None:
            time.sleep(1)
        assert proc.poll() is not None, _with_log(
            "serve kept running after its weight-share leader was killed"
        )
    finally:
        if proc.poll() is None:
            _terminate(proc)

    assert not any(_alive(pid) for pid in pids.values())
