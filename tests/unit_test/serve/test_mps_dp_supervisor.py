# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import contextlib
import importlib.util
import json
import os
import signal
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
SUPERVISOR_PATH = REPO_ROOT / "examples" / "mps_dp" / "supervisor.py"

_spec = importlib.util.spec_from_file_location("mps_dp_supervisor", SUPERVISOR_PATH)
assert _spec is not None and _spec.loader is not None
mps_dp_supervisor = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = mps_dp_supervisor
_spec.loader.exec_module(mps_dp_supervisor)


def _supervisor(tmp_path: Path):
    state = tmp_path / "gpu-0" / "run-test"
    state.mkdir(parents=True, exist_ok=True)
    supervisor = mps_dp_supervisor.ReplicaSupervisor(
        state=state,
        launch_script=REPO_ROOT / "examples" / "mps_dp" / "launch.sh",
        router_url="http://127.0.0.1:8799",
        interval_secs=0.01,
        health_failure_threshold=2,
        health_timeout_secs=0.01,
        health_tries=2,
        health_interval_secs=0.01,
    )
    return supervisor


def test_launch_manifest_records_driver_and_host_timezone() -> None:
    launch_text = (REPO_ROOT / "examples" / "mps_dp" / "launch.sh").read_text(
        encoding="utf-8"
    )

    assert "--query-gpu=driver_version" in launch_text
    assert 'echo "driver_version=$driver_version"' in launch_text
    assert 'echo "host_timezone=$host_timezone"' in launch_text
    assert 'echo "started_at=$started_at"' in launch_text


def test_restart_gate_orders_router_kv_attach_and_registration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    supervisor = _supervisor(tmp_path)
    spec = mps_dp_supervisor.ReplicaSpec(
        index=1,
        port=8802,
        log=str(tmp_path / "replica_1.log"),
        expected_tokens=4096,
        command=["python", "-m", "fake_server"],
        env={"CUDA_MPS_PIPE_DIRECTORY": "/tmp/private-mps"},
    )
    old = mps_dp_supervisor.ReplicaRecord(
        index=1,
        pid=101,
        pgid=101,
        port=8802,
        log=spec.log,
        leader_start="old-start",
    )
    new = mps_dp_supervisor.ReplicaRecord(
        index=1,
        pid=202,
        pgid=202,
        port=8802,
        log=spec.log,
        leader_start="new-start",
    )
    events: list[str] = []

    monkeypatch.setattr(
        supervisor,
        "_set_router_disabled",
        lambda _spec, disabled: events.append(
            "router_disable" if disabled else "router_enable"
        ),
    )
    monkeypatch.setattr(
        supervisor,
        "_terminate_replica",
        lambda _record: events.append("terminate"),
    )
    monkeypatch.setattr(
        supervisor,
        "_launch_replica",
        lambda _spec, _pending: events.append("launch") or new,
    )
    monkeypatch.setattr(
        supervisor,
        "_prepare_pending_restart",
        lambda _spec: events.append("pending_prepare") or object(),
        raising=False,
    )
    monkeypatch.setattr(
        supervisor,
        "_clear_pending_restart",
        lambda _pending: events.append("pending_clear"),
        raising=False,
    )
    monkeypatch.setattr(
        supervisor,
        "_wait_for_port_release",
        lambda _port: events.append("port_release"),
    )
    monkeypatch.setattr(
        supervisor,
        "_replace_replica_record",
        lambda _record: events.append("record"),
    )
    monkeypatch.setattr(
        supervisor,
        "_wait_for_health",
        lambda _spec, _record: events.append("health"),
    )
    monkeypatch.setattr(
        supervisor,
        "_validate_kv_capacity",
        lambda _spec, **_kwargs: events.append("kv"),
    )
    monkeypatch.setattr(
        supervisor,
        "_verify_mps_attach",
        lambda _spec: events.append("mps_attach"),
    )

    supervisor.restart_replica(spec, old)

    assert events == [
        "router_disable",
        "terminate",
        "port_release",
        "pending_prepare",
        "launch",
        "record",
        "pending_clear",
        "health",
        "kv",
        "mps_attach",
        "router_enable",
    ]


def test_failed_restart_keeps_worker_disabled(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    supervisor = _supervisor(tmp_path)
    spec = mps_dp_supervisor.ReplicaSpec(
        index=0,
        port=8801,
        log=str(tmp_path / "replica_0.log"),
        expected_tokens=4096,
        command=["false"],
        env={},
    )
    old = mps_dp_supervisor.ReplicaRecord(
        index=0,
        pid=101,
        pgid=101,
        port=8801,
        log=spec.log,
        leader_start="old-start",
    )
    disabled: list[bool] = []

    monkeypatch.setattr(
        supervisor,
        "_set_router_disabled",
        lambda _spec, value: disabled.append(value),
    )
    monkeypatch.setattr(supervisor, "_terminate_replica", lambda _record: None)
    monkeypatch.setattr(supervisor, "_wait_for_port_release", lambda _port: None)
    monkeypatch.setattr(
        supervisor,
        "_prepare_pending_restart",
        lambda _spec: object(),
        raising=False,
    )
    monkeypatch.setattr(
        supervisor,
        "_launch_replica",
        lambda _spec, _pending: (_ for _ in ()).throw(RuntimeError("launch failed")),
    )

    with pytest.raises(RuntimeError, match="launch failed"):
        supervisor.restart_replica(spec, old)

    assert disabled == [True]


def test_router_disable_failure_is_classified_as_restart_failed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    supervisor = _supervisor(tmp_path)
    spec = mps_dp_supervisor.ReplicaSpec(
        index=0,
        port=8801,
        log=str(tmp_path / "replica_0.log"),
        expected_tokens=None,
        command=["false"],
        env={},
    )
    old = mps_dp_supervisor.ReplicaRecord(0, 101, 101, 8801, spec.log, "old")
    recorded: list[tuple[str, dict[str, object]]] = []

    monkeypatch.setattr(
        supervisor,
        "_set_router_disabled",
        lambda _spec, _disabled: (_ for _ in ()).throw(
            RuntimeError("router unavailable")
        ),
    )
    monkeypatch.setattr(
        supervisor,
        "_record_event",
        lambda event, **fields: recorded.append((event, fields)),
    )

    with pytest.raises(RuntimeError, match="router unavailable"):
        supervisor.restart_replica(spec, old)

    assert [event for event, _fields in recorded] == [
        "restart_started",
        "restart_failed",
    ]


@pytest.mark.skipif(os.name != "posix", reason="process groups require POSIX")
def test_cleanup_pending_restart_terminates_uncommitted_replacement(
    tmp_path: Path,
) -> None:
    state = tmp_path / "gpu-0" / "run-test"
    pending_dir = state / "pending-restarts"
    pending_dir.mkdir(parents=True)
    process = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(300)"],
        start_new_session=True,
    )
    start = mps_dp_supervisor._wait_for_start_time(process.pid)
    pending = {
        "schema_version": 1,
        "index": 0,
        "port": 8801,
        "log": str(state / "logs" / "replica_0.log"),
        "token": "test-pending-token",
        "created_unix_s": time.time(),
        "pid": process.pid,
        "pgid": process.pid,
        "leader_start": start,
    }
    pending_path = pending_dir / "replica_0.json"
    pending_path.write_text(json.dumps(pending))

    try:
        cleaned = mps_dp_supervisor.cleanup_pending_restarts(state)
        process.wait(timeout=5)
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=5)

    assert cleaned == [0]
    assert not pending_path.exists()


def test_cleanup_pending_restart_removes_stale_placeholder(tmp_path: Path) -> None:
    state = tmp_path / "gpu-0" / "run-test"
    pending_dir = state / "pending-restarts"
    pending_dir.mkdir(parents=True)
    pending_path = pending_dir / "replica_1.json"
    pending_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "index": 1,
                "port": 8802,
                "log": str(state / "logs" / "replica_1.log"),
                "token": "stale-token",
                "created_unix_s": time.time() - 60,
                "pid": None,
                "pgid": None,
                "leader_start": None,
            }
        )
    )

    assert mps_dp_supervisor.cleanup_pending_restarts(state) == [1]
    assert not pending_path.exists()


@pytest.mark.skipif(os.name != "posix", reason="/proc port ownership requires Linux")
def test_cleanup_pending_restart_refuses_unrelated_port_owner(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    state = tmp_path / "gpu-0" / "run-test"
    pending_dir = state / "pending-restarts"
    pending_dir.mkdir(parents=True)
    process = subprocess.Popen(
        [
            sys.executable,
            "-u",
            "-c",
            (
                "import socket, time; "
                "s=socket.socket(); s.bind(('127.0.0.1', 0)); s.listen(); "
                "print(s.getsockname()[1], flush=True); time.sleep(300)"
            ),
        ],
        stdout=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    assert process.stdout is not None
    port = int(process.stdout.readline())
    (state / "replica_specs.json").write_text(
        json.dumps(
            [
                {
                    "index": 0,
                    "port": port,
                    "log": str(state / "logs" / "replica_0.log"),
                    "expected_tokens": None,
                    "command": ["python", "-m", "expected_server"],
                    "env": {},
                    "cwd": None,
                }
            ]
        )
    )
    pending_path = pending_dir / "replica_0.json"
    pending_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "index": 0,
                "port": port,
                "log": str(state / "logs" / "replica_0.log"),
                "token": "token-not-present-in-process-command",
                "created_unix_s": time.time() - 60,
                "pid": None,
                "pgid": None,
                "leader_start": None,
            }
        )
    )

    try:
        cleaned = mps_dp_supervisor.cleanup_pending_restarts(state)
        survived_cleanup = process.poll() is None
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=5)

    assert cleaned == [0]
    assert survived_cleanup
    assert not pending_path.exists()
    events = [
        json.loads(line)
        for line in (state / "supervisor-events.jsonl").read_text().splitlines()
    ]
    refused = [
        event
        for event in events
        if event["event"] == "pending_restart_identity_refused"
    ]
    assert len(refused) == 1
    assert refused[0]["candidate_pid"] == process.pid
    assert refused[0]["replica"] == 0
    assert f"refusing to terminate PID {process.pid}" in capsys.readouterr().err


@pytest.mark.skipif(os.name != "posix", reason="/proc port ownership requires Linux")
def test_cleanup_pending_restart_kills_matching_port_owner(tmp_path: Path) -> None:
    state = tmp_path / "gpu-0" / "run-test"
    pending_dir = state / "pending-restarts"
    pending_dir.mkdir(parents=True)
    server_code = (
        "import socket, time; "
        "s=socket.socket(); s.bind(('127.0.0.1', 0)); s.listen(); "
        "print(s.getsockname()[1], flush=True); time.sleep(300)"
    )
    command = [sys.executable, "-u", "-c", server_code]
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    assert process.stdout is not None
    port = int(process.stdout.readline())
    (state / "replica_specs.json").write_text(
        json.dumps(
            [
                {
                    "index": 0,
                    "port": port,
                    "log": str(state / "logs" / "replica_0.log"),
                    "expected_tokens": None,
                    "command": command,
                    "env": {},
                    "cwd": None,
                }
            ]
        )
    )
    pending_path = pending_dir / "replica_0.json"
    pending_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "index": 0,
                "port": port,
                "log": str(state / "logs" / "replica_0.log"),
                "token": "token-not-present-in-process-command",
                "created_unix_s": time.time() - 60,
                "pid": None,
                "pgid": None,
                "leader_start": None,
            }
        )
    )

    try:
        cleaned = mps_dp_supervisor.cleanup_pending_restarts(state)
        process.wait(timeout=5)
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=5)

    assert cleaned == [0]
    assert not pending_path.exists()


@pytest.mark.skipif(os.name != "posix", reason="fault injection requires fork")
def test_crash_mid_restart_leaves_teardown_owned_replacement(tmp_path: Path) -> None:
    supervisor = _supervisor(tmp_path)
    state = supervisor.state
    log = state / "logs" / "replica_0.log"
    old = mps_dp_supervisor.ReplicaRecord(
        0,
        999999,
        999999,
        8801,
        str(log),
        "old",
    )
    (state / "replicas.tsv").write_text(old.to_tsv())
    spec = mps_dp_supervisor.ReplicaSpec(
        index=0,
        port=8801,
        log=str(log),
        expected_tokens=None,
        command=[sys.executable, "-c", "import time; time.sleep(300)"],
        env={},
    )

    supervisor_pid = os.fork()
    if supervisor_pid == 0:
        try:
            child_supervisor = _supervisor(tmp_path)
            child_supervisor._set_router_disabled = lambda _spec, _disabled: None
            child_supervisor._terminate_replica = lambda _record: None
            child_supervisor._wait_for_port_release = lambda _port: None
            child_supervisor._replace_replica_record = lambda _record: time.sleep(300)
            child_supervisor.restart_replica(spec, old)
        finally:
            os._exit(1)

    pending_path = state / "pending-restarts" / "replica_0.json"
    replacement_pid: int | None = None
    try:
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            if pending_path.exists():
                pending = json.loads(pending_path.read_text())
                if pending.get("pid") is not None:
                    replacement_pid = int(pending["pid"])
                    break
            time.sleep(0.02)
        assert replacement_pid is not None
        assert mps_dp_supervisor._pid_is_live(replacement_pid)
        assert (state / "replicas.tsv").read_text() == old.to_tsv()

        os.kill(supervisor_pid, signal.SIGKILL)
        os.waitpid(supervisor_pid, 0)
        supervisor_pid = -1

        assert mps_dp_supervisor._pid_is_live(replacement_pid)
        assert mps_dp_supervisor.cleanup_pending_restarts(state) == [0]
        assert not mps_dp_supervisor._pid_is_live(replacement_pid)
        assert not pending_path.exists()
    finally:
        if supervisor_pid > 0:
            with contextlib.suppress(ProcessLookupError):
                os.kill(supervisor_pid, signal.SIGKILL)
            os.waitpid(supervisor_pid, 0)
        if replacement_pid is not None and mps_dp_supervisor._pid_is_live(
            replacement_pid
        ):
            with contextlib.suppress(ProcessLookupError):
                os.killpg(replacement_pid, signal.SIGKILL)


def test_prepare_pending_restart_refuses_unreconciled_record(tmp_path: Path) -> None:
    supervisor = _supervisor(tmp_path)
    pending_path = supervisor.state / "pending-restarts" / "replica_0.json"
    pending_path.parent.mkdir(parents=True)
    pending_path.write_text("{}")
    spec = mps_dp_supervisor.ReplicaSpec(
        index=0,
        port=8801,
        log=str(supervisor.state / "logs" / "replica_0.log"),
        expected_tokens=None,
        command=["false"],
        env={},
    )

    with pytest.raises(RuntimeError, match="unreconciled pending restart"):
        supervisor._prepare_pending_restart(spec)


def test_supervisor_reconciles_pending_before_first_check(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    supervisor = _supervisor(tmp_path)
    events: list[str] = []
    monkeypatch.setattr(
        mps_dp_supervisor,
        "cleanup_pending_restarts",
        lambda state: events.append(f"cleanup:{state.name}") or [],
    )
    monkeypatch.setattr(
        supervisor,
        "_record_event",
        lambda event, **_fields: events.append(event),
    )
    monkeypatch.setattr(
        supervisor,
        "check_once",
        lambda: (_ for _ in ()).throw(KeyboardInterrupt),
    )

    with pytest.raises(KeyboardInterrupt):
        supervisor.run()

    assert events == ["cleanup:run-test", "supervisor_started"]


@pytest.mark.skipif(os.name != "posix", reason="launch.sh needs a POSIX shell")
def test_down_consumes_pending_restart_ownership(tmp_path: Path) -> None:
    state_root = tmp_path / "state"
    state = state_root / "gpu-0" / "run-test"
    pending_dir = state / "pending-restarts"
    pending_dir.mkdir(parents=True)
    (state / "replicas.tsv").write_text("")
    (state / "services.tsv").write_text("")
    process = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(300)"],
        start_new_session=True,
    )
    start = mps_dp_supervisor._wait_for_start_time(process.pid)
    (pending_dir / "replica_0.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "index": 0,
                "port": 8801,
                "log": str(state / "logs" / "replica_0.log"),
                "token": "down-pending-token",
                "created_unix_s": time.time(),
                "pid": process.pid,
                "pgid": process.pid,
                "leader_start": start,
            }
        )
    )
    environment = os.environ.copy()
    environment.update(
        {
            "STATE_ROOT": str(state_root),
            "PYTHON_BIN": sys.executable,
            "DRAIN_TRIES": "1",
            "DRAIN_INTERVAL": "0.01",
        }
    )

    try:
        result = subprocess.run(
            [
                "bash",
                str(REPO_ROOT / "examples" / "mps_dp" / "launch.sh"),
                "down",
                "run-test",
            ],
            env=environment,
            capture_output=True,
            text=True,
            timeout=15,
        )
        survived_down = process.poll() is None
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=5)

    assert result.returncode == 0, result.stdout + result.stderr
    assert not survived_down
    assert not state.exists()


def test_failed_post_launch_gate_stops_replacement_and_keeps_worker_disabled(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    supervisor = _supervisor(tmp_path)
    spec = mps_dp_supervisor.ReplicaSpec(
        index=0,
        port=8801,
        log=str(tmp_path / "replica_0.log"),
        expected_tokens=4096,
        command=["python", "-m", "fake_server"],
        env={},
    )
    old = mps_dp_supervisor.ReplicaRecord(0, 101, 101, 8801, spec.log, "old")
    replacement = mps_dp_supervisor.ReplicaRecord(0, 202, 202, 8801, spec.log, "new")
    disabled: list[bool] = []
    terminated: list[int] = []

    monkeypatch.setattr(
        supervisor,
        "_set_router_disabled",
        lambda _spec, value: disabled.append(value),
    )
    monkeypatch.setattr(
        supervisor,
        "_terminate_replica",
        lambda record: terminated.append(record.pid),
    )
    monkeypatch.setattr(
        supervisor,
        "_prepare_pending_restart",
        lambda _spec: object(),
        raising=False,
    )
    monkeypatch.setattr(
        supervisor,
        "_clear_pending_restart",
        lambda _pending: None,
        raising=False,
    )
    monkeypatch.setattr(
        supervisor,
        "_launch_replica",
        lambda _spec, _pending: replacement,
    )
    monkeypatch.setattr(supervisor, "_wait_for_port_release", lambda _port: None)
    monkeypatch.setattr(supervisor, "_replace_replica_record", lambda _record: None)
    monkeypatch.setattr(supervisor, "_wait_for_health", lambda _spec, _record: None)
    monkeypatch.setattr(
        supervisor,
        "_validate_kv_capacity",
        lambda _spec, **_kwargs: (_ for _ in ()).throw(RuntimeError("KV mismatch")),
    )

    with pytest.raises(RuntimeError, match="KV mismatch"):
        supervisor.restart_replica(spec, old)

    assert disabled == [True]
    assert terminated == [101, 202]


def test_kv_validation_uses_replacement_generation_latest_log_entry(
    tmp_path: Path,
) -> None:
    supervisor = _supervisor(tmp_path)
    log = tmp_path / "replica_0.log"
    log.write_text("old boot #tokens: 4096,\nnew boot #tokens: 2048,\n")
    spec = mps_dp_supervisor.ReplicaSpec(
        index=0,
        port=8801,
        log=str(log),
        expected_tokens=4096,
        command=["true"],
        env={},
    )

    with pytest.raises(RuntimeError, match="resolved 2048 KV tokens"):
        supervisor._validate_kv_capacity(spec)


def test_kv_validation_does_not_reuse_previous_generation(
    tmp_path: Path,
) -> None:
    supervisor = _supervisor(tmp_path)
    log = tmp_path / "replica_0.log"
    old_log = "old boot #tokens: 4096,\n"
    log.write_text(old_log + "replacement reached health without KV marker\n")
    spec = mps_dp_supervisor.ReplicaSpec(
        index=0,
        port=8801,
        log=str(log),
        expected_tokens=4096,
        command=["true"],
        env={},
    )

    with pytest.raises(RuntimeError, match="resolved None KV tokens"):
        supervisor._validate_kv_capacity(
            spec,
            log_offset=len(old_log.encode()),
        )


def test_health_failures_restart_only_after_threshold(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    supervisor = _supervisor(tmp_path)
    spec = mps_dp_supervisor.ReplicaSpec(
        index=0,
        port=8801,
        log=str(tmp_path / "replica_0.log"),
        expected_tokens=None,
        command=["python", "-m", "fake_server"],
        env={},
    )
    record = mps_dp_supervisor.ReplicaRecord(
        index=0,
        pid=101,
        pgid=101,
        port=8801,
        log=spec.log,
        leader_start="old-start",
    )
    restarts: list[int] = []
    monkeypatch.setattr(supervisor, "_load_specs", lambda: [spec])
    monkeypatch.setattr(supervisor, "_load_replica_records", lambda: {0: record})
    monkeypatch.setattr(supervisor, "_identity_matches", lambda _record: True)
    monkeypatch.setattr(supervisor, "_health_ok", lambda _port: False)
    monkeypatch.setattr(
        supervisor,
        "restart_replica",
        lambda target, _record: restarts.append(target.index),
    )

    supervisor.check_once()
    assert restarts == []

    supervisor.check_once()
    assert restarts == [0]


def _replica_pair(tmp_path: Path):
    specs = [
        mps_dp_supervisor.ReplicaSpec(
            index=index,
            port=8801 + index,
            log=str(tmp_path / f"replica_{index}.log"),
            expected_tokens=None,
            command=["python", "-m", "fake_server"],
            env={},
        )
        for index in (0, 1)
    ]
    records = {
        spec.index: mps_dp_supervisor.ReplicaRecord(
            index=spec.index,
            pid=101 + spec.index,
            pgid=101 + spec.index,
            port=spec.port,
            log=spec.log,
            leader_start="old-start",
        )
        for spec in specs
    }
    return specs, records


def test_double_fault_restarts_both_replicas(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Round 4 livelock scenario: both replicas die at once, so every attach
    # verification runs while a peer is dead. Both replicas must recover.
    supervisor = _supervisor(tmp_path)
    attached = tmp_path / "attached.txt"
    attached.write_text("")
    # Emulates launch.sh verify: a scoped replica with no attached MPS client
    # fails the gate. Replicas outside the scope cannot fail it.
    script = tmp_path / "verify.sh"
    script.write_text(
        "#!/bin/bash\n"
        f'attached=" $(cat {attached}) "\n'
        'for idx in $(echo "$3" | tr "," " "); do\n'
        '  case "$attached" in\n'
        '    *" $idx "*) ;;\n'
        '    *) echo "replica $idx has no attached MPS client" >&2; exit 1;;\n'
        "  esac\ndone\nexit 0\n"
    )
    script.chmod(0o755)
    supervisor.launch_script = script
    specs, records = _replica_pair(tmp_path)
    healthy: set[int] = set()
    attempts: list[int] = []

    def fake_restart(spec, _record) -> None:
        # The replacement process is up and attached; the peer may still be dead.
        attempts.append(spec.index)
        live = sorted(healthy | {spec.index})
        attached.write_text(" ".join(str(index) for index in live))
        supervisor._verify_mps_attach(spec)
        healthy.add(spec.index)

    monkeypatch.setattr(supervisor, "_load_specs", lambda: specs)
    monkeypatch.setattr(supervisor, "_load_replica_records", lambda: dict(records))
    monkeypatch.setattr(supervisor, "_identity_matches", lambda _record: True)
    monkeypatch.setattr(
        supervisor,
        "_health_ok",
        lambda port: (port - 8801) in healthy,
    )
    monkeypatch.setattr(supervisor, "restart_replica", fake_restart)

    for _ in range(12):
        with contextlib.suppress(RuntimeError):
            supervisor.check_once()
        if healthy == {0, 1}:
            break

    assert healthy == {0, 1}
    assert set(attempts) == {0, 1}


def test_sweep_rotates_after_a_failed_restart(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    supervisor = _supervisor(tmp_path)
    specs, records = _replica_pair(tmp_path)
    attempts: list[int] = []

    def always_fails(spec, _record) -> None:
        attempts.append(spec.index)
        raise RuntimeError("MPS attach verification failed")

    monkeypatch.setattr(supervisor, "_load_specs", lambda: specs)
    monkeypatch.setattr(supervisor, "_load_replica_records", lambda: dict(records))
    monkeypatch.setattr(supervisor, "_identity_matches", lambda _record: True)
    monkeypatch.setattr(supervisor, "_health_ok", lambda _port: False)
    monkeypatch.setattr(supervisor, "restart_replica", always_fails)

    # First sweep only accrues a health failure; each later sweep attempts one
    # replica and aborts, so the attempts must alternate rather than pin to 0.
    for _ in range(5):
        with contextlib.suppress(RuntimeError):
            supervisor.check_once()

    assert attempts == [0, 1, 0, 1]


def _scope_probe_script(tmp_path: Path, detached_index: int) -> Path:
    script = tmp_path / "verify.sh"
    script.write_text(
        "#!/bin/bash\n"
        'case " $(echo "$3" | tr "," " ") " in\n'
        f'  *" {detached_index} "*)\n'
        f'    echo "replica {detached_index} has no process in the MPS client list"'
        " >&2; exit 1;;\nesac\nexit 0\n"
    )
    script.chmod(0o755)
    return script


def test_attach_gate_still_detects_a_detached_healthy_peer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    supervisor = _supervisor(tmp_path)
    supervisor.launch_script = _scope_probe_script(tmp_path, 1)
    specs, records = _replica_pair(tmp_path)
    monkeypatch.setattr(supervisor, "_load_specs", lambda: specs)
    monkeypatch.setattr(supervisor, "_load_replica_records", lambda: dict(records))
    monkeypatch.setattr(supervisor, "_identity_matches", lambda _record: True)
    monkeypatch.setattr(supervisor, "_health_ok", lambda _port: True)

    assert supervisor._attach_scope(specs[0]) == [0, 1]
    with pytest.raises(RuntimeError, match="MPS attach verification failed"):
        supervisor._verify_mps_attach(specs[0])


def test_attach_gate_scope_excludes_a_dead_peer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    supervisor = _supervisor(tmp_path)
    supervisor.launch_script = _scope_probe_script(tmp_path, 1)
    specs, records = _replica_pair(tmp_path)
    monkeypatch.setattr(supervisor, "_load_specs", lambda: specs)
    monkeypatch.setattr(supervisor, "_load_replica_records", lambda: dict(records))
    monkeypatch.setattr(
        supervisor,
        "_identity_matches",
        lambda record: record.index == 0,
    )
    monkeypatch.setattr(supervisor, "_health_ok", lambda port: port == 8801)

    assert supervisor._attach_scope(specs[0]) == [0]
    supervisor._verify_mps_attach(specs[0])


def test_singleton_lock_rejects_second_supervisor(tmp_path: Path) -> None:
    first = _supervisor(tmp_path)
    second = _supervisor(tmp_path)

    with first.singleton_lock():
        with pytest.raises(RuntimeError, match="already active"):
            with second.singleton_lock():
                pass


@pytest.mark.skipif(os.name != "posix", reason="process groups require POSIX")
def test_restart_fault_injection_runs_full_gate_chain(tmp_path: Path) -> None:
    router_updates: list[dict[str, bool]] = []

    class RouterHandler(BaseHTTPRequestHandler):
        def do_PUT(self) -> None:
            length = int(self.headers["Content-Length"])
            router_updates.append(json.loads(self.rfile.read(length)))
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b'{"status":"ok"}')

        def log_message(self, _format: str, *args: object) -> None:
            pass

    router = ThreadingHTTPServer(("127.0.0.1", 0), RouterHandler)
    router_thread = threading.Thread(target=router.serve_forever, daemon=True)
    router_thread.start()

    state = tmp_path / "gpu-0" / "run-test"
    state.mkdir(parents=True)
    log = state / "logs" / "replica_0.log"
    old = mps_dp_supervisor.ReplicaRecord(0, 999999, 999999, 0, str(log), "old")
    (state / "replicas.tsv").write_text(old.to_tsv())
    verify_script = tmp_path / "verify.sh"
    verify_script.write_text("#!/bin/bash\nexit 0\n")

    worker_code = """
from http.server import BaseHTTPRequestHandler, HTTPServer
class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200 if self.path == "/health" else 404)
        self.end_headers()
    def log_message(self, format, *args):
        pass
print("#tokens: 4096", flush=True)
HTTPServer(("127.0.0.1", 0), Handler).serve_forever()
"""
    # Replace the ephemeral port in the command after reserving it.
    probe = ThreadingHTTPServer(("127.0.0.1", 0), RouterHandler)
    worker_port = probe.server_port
    probe.server_close()
    worker_code = worker_code.replace(
        'HTTPServer(("127.0.0.1", 0)',
        f'HTTPServer(("127.0.0.1", {worker_port})',
    )
    spec = mps_dp_supervisor.ReplicaSpec(
        index=0,
        port=worker_port,
        log=str(log),
        expected_tokens=4096,
        command=[sys.executable, "-u", "-c", worker_code],
        env={},
    )
    supervisor = mps_dp_supervisor.ReplicaSupervisor(
        state=state,
        launch_script=verify_script,
        router_url=f"http://127.0.0.1:{router.server_port}",
        interval_secs=0.01,
        health_failure_threshold=1,
        health_timeout_secs=1,
        health_tries=50,
        health_interval_secs=0.02,
    )

    replacement = None
    try:
        supervisor.restart_replica(spec, old)
        replacement = supervisor._load_replica_records()[0]
        assert supervisor._identity_matches(replacement)
        assert supervisor._health_ok(worker_port)
        assert not (state / "pending-restarts" / "replica_0.json").exists()
        lifecycle_events = [
            json.loads(line)["event"]
            for line in (state / "supervisor-events.jsonl").read_text().splitlines()
        ]
        assert lifecycle_events == [
            "restart_started",
            "pending_restart_prepared",
            "pending_restart_identified",
            "pending_restart_committed",
            "restart_completed",
        ]
        assert router_updates == [
            {"disabled": True, "is_dead": True},
            {"disabled": False, "is_dead": False},
        ]
    finally:
        if replacement is not None and supervisor._identity_matches(replacement):
            os.killpg(replacement.pgid, signal.SIGTERM)
        router.shutdown()
        router.server_close()


def _spawn_group_with_stubborn_child(
    ready_marker: Path,
    exit_after_secs: float,
) -> tuple[subprocess.Popen[str], int]:
    # Leader exits on SIGTERM; its child in the same process group ignores
    # SIGTERM and only leaves on its own schedule (or on SIGKILL). This is the
    # topology that a leader-only termination wait cannot clean up. The leader
    # publishes the child PID only after the child's handler is installed, so a
    # signal can never win a race against an interpreter that is still starting.
    child_code = (
        "import signal, sys, time; "
        "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
        "open(sys.argv[1], 'w').write('ready'); "
        "time.sleep(float(sys.argv[2]))"
    )
    leader_code = f"""
import os, subprocess, sys, time
child = subprocess.Popen([
    sys.executable, "-u", "-c", {child_code!r},
    {str(ready_marker)!r}, {str(exit_after_secs)!r},
])
while not os.path.exists({str(ready_marker)!r}):
    time.sleep(0.01)
print(child.pid, flush=True)
time.sleep(300)
"""
    leader = subprocess.Popen(
        [sys.executable, "-u", "-c", leader_code],
        stdout=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    assert leader.stdout is not None
    child_pid = int(leader.stdout.readline())
    return leader, child_pid


@pytest.mark.skipif(os.name != "posix", reason="process groups require POSIX")
def test_terminate_waits_for_group_member_after_leader_exits(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(mps_dp_supervisor, "TERMINATE_TERM_GRACE_SECS", 10)
    supervisor = _supervisor(tmp_path)
    leader, child_pid = _spawn_group_with_stubborn_child(tmp_path / "ready", 1.5)
    leader_start = mps_dp_supervisor._wait_for_start_time(leader.pid)
    old = mps_dp_supervisor.ReplicaRecord(
        index=0,
        pid=leader.pid,
        pgid=leader.pid,
        port=8801,
        log=str(tmp_path / "replica_0.log"),
        leader_start=leader_start,
    )
    spec = mps_dp_supervisor.ReplicaSpec(
        index=0,
        port=8801,
        log=old.log,
        expected_tokens=None,
        command=["python", "-m", "fake_server"],
        env={},
    )
    new = mps_dp_supervisor.ReplicaRecord(0, 202, 202, 8801, old.log, "new-start")
    child_live_at_gate: list[bool] = []

    def _launch(_spec: object, _pending: object) -> object:
        child_live_at_gate.append(mps_dp_supervisor._pid_is_live(child_pid))
        return new

    for name, replacement in (
        ("_set_router_disabled", lambda _spec, _disabled: None),
        ("_launch_replica", _launch),
        ("_prepare_pending_restart", lambda _spec: object()),
        ("_clear_pending_restart", lambda _pending: None),
        ("_wait_for_port_release", lambda _port: None),
        ("_replace_replica_record", lambda _record: None),
        ("_wait_for_health", lambda _spec, _record: None),
        ("_validate_kv_capacity", lambda _spec, **_kwargs: None),
        ("_verify_mps_attach", lambda _spec: None),
    ):
        monkeypatch.setattr(supervisor, name, replacement, raising=False)

    try:
        supervisor.restart_replica(spec, old)
    finally:
        with contextlib.suppress(ProcessLookupError):
            os.kill(child_pid, signal.SIGKILL)
        with contextlib.suppress(ProcessLookupError):
            os.killpg(leader.pid, signal.SIGKILL)
        leader.wait(timeout=5)

    assert not mps_dp_supervisor._pid_is_live(leader.pid)
    assert not mps_dp_supervisor._pid_is_live(child_pid)
    # The replacement must not reach any gate while a member of the group it
    # replaces is still alive holding that replica's MPS client and memory.
    assert child_live_at_gate == [False]


@pytest.mark.skipif(os.name != "posix", reason="process groups require POSIX")
def test_terminate_kills_surviving_member_but_spares_a_reused_pid(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(mps_dp_supervisor, "TERMINATE_TERM_GRACE_SECS", 1)
    monkeypatch.setattr(mps_dp_supervisor, "TERMINATE_KILL_GRACE_SECS", 5)
    leader, child_pid = _spawn_group_with_stubborn_child(tmp_path / "ready", 300)
    leader_start = mps_dp_supervisor._wait_for_start_time(leader.pid)
    # Stands in for a PID recycled after the freeze: the number is in the frozen
    # member list but the process behind it is no longer the one that was frozen.
    innocent = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); "
            "time.sleep(300)",
        ],
        start_new_session=True,
    )
    innocent_start = mps_dp_supervisor._wait_for_start_time(innocent.pid)
    real_freeze = mps_dp_supervisor._freeze_group_members
    monkeypatch.setattr(
        mps_dp_supervisor,
        "_freeze_group_members",
        lambda record: [
            *real_freeze(record),
            (innocent.pid, "Thu Jan  1 00:00:00 1970"),
        ],
    )
    record = mps_dp_supervisor.ReplicaRecord(
        index=0,
        pid=leader.pid,
        pgid=leader.pid,
        port=8801,
        log=str(tmp_path / "replica_0.log"),
        leader_start=leader_start,
    )

    try:
        mps_dp_supervisor._terminate_record(record)
        child_survived = mps_dp_supervisor._pid_is_live(child_pid)
        innocent_survived = mps_dp_supervisor._pid_start_time(innocent.pid) == (
            innocent_start
        )
    finally:
        for pid in (child_pid, innocent.pid, leader.pid):
            with contextlib.suppress(ProcessLookupError):
                os.kill(pid, signal.SIGKILL)
        innocent.wait(timeout=5)
        leader.wait(timeout=5)

    assert not child_survived
    assert innocent_survived


@pytest.mark.skipif(os.name != "posix", reason="process groups require POSIX")
def test_terminate_cleans_group_whose_leader_is_an_unreaped_zombie(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A leader that exits first must not hide its live siblings.

    This is the shape a SIGTERM to the leader PID produces on a real replica:
    by the time the supervisor reaches termination the leader is already gone,
    and in an init-less container it is an unreapable zombie in front of stage
    processes that still hold their MPS clients and device memory.
    """
    monkeypatch.setattr(mps_dp_supervisor, "TERMINATE_TERM_GRACE_SECS", 1)
    leader, child_pid = _spawn_group_with_stubborn_child(tmp_path / "ready", 300)
    leader_start = mps_dp_supervisor._wait_for_start_time(leader.pid)
    record = mps_dp_supervisor.ReplicaRecord(
        index=0,
        pid=leader.pid,
        pgid=leader.pid,
        port=8801,
        log=str(tmp_path / "replica_0.log"),
        leader_start=leader_start,
    )
    # Kill only the leader and never reap it, so it stays a zombie exactly as
    # it would under a container PID 1 that does not wait().
    os.kill(leader.pid, signal.SIGKILL)
    for _ in range(200):
        if not mps_dp_supervisor._pid_is_live(leader.pid):
            break
        time.sleep(0.02)

    try:
        leader_is_zombie = (
            Path(f"/proc/{leader.pid}/stat").read_text().rpartition(")")[2].split()[0]
            == "Z"
        )
        assert not mps_dp_supervisor._record_identity_matches(record)
        assert mps_dp_supervisor._record_group_is_owned(record)
        mps_dp_supervisor._terminate_record(record)
        child_survived = mps_dp_supervisor._pid_is_live(child_pid)
    finally:
        with contextlib.suppress(ProcessLookupError):
            os.kill(child_pid, signal.SIGKILL)
        leader.wait(timeout=5)

    assert leader_is_zombie
    assert not child_survived


def _spawn_reapable_group(
    tmp_path: Path,
    *,
    second_child: bool = False,
    env_token: str | None = None,
) -> tuple[subprocess.Popen[str], list[int]]:
    """A process group whose leader this process can reap, like a replacement.

    The supervisor is the parent of every replica it launches, so it reaps that
    replica's leader itself within ~2s of the leader exiting — which is always
    before the health threshold brings termination around. This helper
    reproduces exactly that: the leader is a direct child here, so
    ``leader.wait()`` removes its PID from the process table entirely, leaving
    a group with live members and no leader entry at all.

    The optional second child is spawned only after the caller creates a marker
    file, which is how a member that appears *between* two membership refreshes
    is modelled.
    """
    child_code = (
        "import signal, sys, time; "
        "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
        "open(sys.argv[1], 'w').write('ready'); "
        "time.sleep(float(sys.argv[2]))"
    )
    leader_code = f"""
import os, subprocess, sys, time
def spawn(marker):
    child = subprocess.Popen([sys.executable, "-u", "-c", {child_code!r}, marker, "300"])
    while not os.path.exists(marker):
        time.sleep(0.01)
    print(child.pid, flush=True)
spawn({str(tmp_path / "ready-a")!r})
if {second_child!r}:
    while not os.path.exists({str(tmp_path / "spawn-b")!r}):
        time.sleep(0.01)
    spawn({str(tmp_path / "ready-b")!r})
time.sleep(300)
"""
    environment = os.environ.copy()
    if env_token is not None:
        environment[mps_dp_supervisor.RUN_TOKEN_ENV] = env_token
    leader = subprocess.Popen(
        [sys.executable, "-u", "-c", leader_code],
        stdout=subprocess.PIPE,
        text=True,
        env=environment,
        start_new_session=True,
    )
    assert leader.stdout is not None
    return leader, [int(leader.stdout.readline())]


def _reap_leader(leader: subprocess.Popen[str]) -> None:
    """Kill the leader and reap it, so its PID leaves the process table."""
    leader.terminate()
    leader.wait(timeout=10)
    for _ in range(500):
        if mps_dp_supervisor._pid_start_time(leader.pid) is None:
            return
        time.sleep(0.02)
    raise AssertionError(f"leader {leader.pid} was never reaped")


def _record_for(leader: subprocess.Popen[str], tmp_path: Path, start: str):
    return mps_dp_supervisor.ReplicaRecord(
        index=0,
        pid=leader.pid,
        pgid=leader.pid,
        port=8801,
        log=str(tmp_path / "replica_0.log"),
        leader_start=start,
    )


def _cleanup(pids: list[int]) -> None:
    for pid in pids:
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.kill(pid, signal.SIGKILL)


@pytest.mark.skipif(os.name != "posix", reason="process groups require POSIX")
def test_terminate_cleans_group_whose_leader_was_already_reaped(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The generation-2 case: the supervisor reaped its own replacement.

    Ownership must be provable from the frozen member list alone. Before the
    de-leadering fix this returned without sending a single signal, and the
    surviving members kept their MPS client and device memory forever.
    """
    monkeypatch.setattr(mps_dp_supervisor, "TERMINATE_TERM_GRACE_SECS", 1)
    state = tmp_path / "gpu-0" / "run-test"
    state.mkdir(parents=True)
    leader, children = _spawn_reapable_group(tmp_path)
    record = _record_for(
        leader, tmp_path, mps_dp_supervisor._wait_for_start_time(leader.pid)
    )
    frozen = mps_dp_supervisor.refresh_group_members(state, record)
    _reap_leader(leader)

    try:
        # The leader-only proof is gone; the member list still carries it.
        leader_proof = mps_dp_supervisor._record_group_is_owned(record)
        member_proof = mps_dp_supervisor._group_is_owned(state, record)
        mps_dp_supervisor._terminate_record(record, state=state)
        survived = [pid for pid in children if mps_dp_supervisor._pid_is_live(pid)]
    finally:
        _cleanup(children)

    assert sorted(pid for pid, _ in frozen) == sorted([leader.pid, *children])
    assert leader_proof is False
    assert member_proof is True
    assert survived == []


@pytest.mark.skipif(os.name != "posix", reason="process groups require POSIX")
def test_terminate_reaches_a_member_that_appeared_since_the_last_refresh(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A kill landing between two polls must not leave the newest member behind.

    The persisted list is at most one refresh interval old, so a member forked
    after that refresh is not in it. It is still reachable: one listed member
    that is alive and still in the group proves the PGID cannot have been
    recycled, which is what makes the group-wide signal safe.
    """
    monkeypatch.setattr(mps_dp_supervisor, "TERMINATE_TERM_GRACE_SECS", 1)
    state = tmp_path / "gpu-0" / "run-test"
    state.mkdir(parents=True)
    leader, children = _spawn_reapable_group(tmp_path, second_child=True)
    record = _record_for(
        leader, tmp_path, mps_dp_supervisor._wait_for_start_time(leader.pid)
    )
    frozen = mps_dp_supervisor.refresh_group_members(state, record)
    # ... and only now, after the refresh, does the second member appear.
    (tmp_path / "spawn-b").write_text("go")
    assert leader.stdout is not None
    late_child = int(leader.stdout.readline())
    children.append(late_child)
    _reap_leader(leader)

    try:
        listed = [pid for pid, _ in mps_dp_supervisor._load_group_members(state, record)]
        mps_dp_supervisor._terminate_record(record, state=state)
        survived = [pid for pid in children if mps_dp_supervisor._pid_is_live(pid)]
    finally:
        _cleanup(children)

    assert late_child not in listed
    assert sorted(pid for pid, _ in frozen) == sorted([leader.pid, children[0]])
    assert survived == []


@pytest.mark.skipif(os.name != "posix", reason="process groups require POSIX")
def test_reaped_leader_with_only_stale_members_proves_nothing_and_signals_nothing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """PID reuse must not be mistaken for a surviving member.

    Two ways a persisted row can be about a different process than the one that
    was frozen, both of which must leave the innocent process untouched:
    a recycled PID (start time no longer matches) and a process that is alive
    with a matching identity but is no longer in the recorded group.
    """
    monkeypatch.setattr(mps_dp_supervisor, "TERMINATE_TERM_GRACE_SECS", 1)
    state = tmp_path / "gpu-0" / "run-test"
    state.mkdir(parents=True)
    leader, children = _spawn_reapable_group(tmp_path)
    record = _record_for(
        leader, tmp_path, mps_dp_supervisor._wait_for_start_time(leader.pid)
    )
    innocent = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); "
            "time.sleep(300)",
        ],
        start_new_session=True,
    )
    innocent_start = mps_dp_supervisor._wait_for_start_time(innocent.pid)
    _reap_leader(leader)
    _cleanup(children)
    for _ in range(500):
        if not any(mps_dp_supervisor._pid_is_live(pid) for pid in children):
            break
        time.sleep(0.02)

    try:
        # (a) right PID, wrong process: the start time no longer matches.
        mps_dp_supervisor._persist_group_members(
            state, record, [(innocent.pid, "Thu Jan  1 00:00:00 1970")]
        )
        recycled_pid_proof = mps_dp_supervisor._group_is_owned(state, record)
        mps_dp_supervisor._terminate_record(record, state=state)
        # (b) right process, wrong group: identity matches but it left the group
        # (here: never joined it), so the PGID could have been recycled.
        mps_dp_supervisor._persist_group_members(
            state, record, [(innocent.pid, innocent_start)]
        )
        foreign_group_proof = mps_dp_supervisor._group_is_owned(state, record)
        mps_dp_supervisor._terminate_record(record, state=state)
        innocent_survived = (
            mps_dp_supervisor._pid_start_time(innocent.pid) == innocent_start
        )
    finally:
        _cleanup([innocent.pid, *children])
        innocent.wait(timeout=5)

    assert recycled_pid_proof is False
    assert foreign_group_proof is False
    assert innocent_survived


@pytest.mark.skipif(os.name != "posix", reason="process groups require POSIX")
def test_terminate_is_a_no_op_once_every_frozen_member_is_gone(
    tmp_path: Path,
) -> None:
    """The third state: not 'ownership unprovable' but 'the group is gone'."""
    state = tmp_path / "gpu-0" / "run-test"
    state.mkdir(parents=True)
    leader, children = _spawn_reapable_group(tmp_path)
    record = _record_for(
        leader, tmp_path, mps_dp_supervisor._wait_for_start_time(leader.pid)
    )
    mps_dp_supervisor.refresh_group_members(state, record)
    _reap_leader(leader)
    _cleanup(children)
    for _ in range(500):
        if not any(mps_dp_supervisor._pid_is_live(pid) for pid in children):
            break
        time.sleep(0.02)

    assert mps_dp_supervisor._group_is_owned(state, record) is False
    # Must return quietly rather than raise: nothing survived anything.
    mps_dp_supervisor._terminate_record(record, state=state)


@pytest.mark.skipif(os.name != "posix", reason="process groups require POSIX")
def test_launch_token_proves_ownership_without_leader_or_member_list(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Second factor: the run token carried in the member's own environment.

    This is the check a human had to perform by hand on the orphaned groups of
    the previous round, made mechanical: no leader entry, no usable frozen
    list, and the group is still provably this run's.
    """
    monkeypatch.setattr(mps_dp_supervisor, "TERMINATE_TERM_GRACE_SECS", 1)
    state = tmp_path / "gpu-0" / "run-test"
    state.mkdir(parents=True)
    token = "round10-token"
    (state / "replica_specs.json").write_text(
        json.dumps(
            [{"index": 0, "port": 8801, "env": {mps_dp_supervisor.RUN_TOKEN_ENV: token}}]
        )
    )
    leader, children = _spawn_reapable_group(tmp_path, env_token=token)
    record = _record_for(
        leader, tmp_path, mps_dp_supervisor._wait_for_start_time(leader.pid)
    )
    _reap_leader(leader)

    try:
        # No member list was ever persisted, so only the token can prove this.
        listed = mps_dp_supervisor._load_group_members(state, record)
        owned = mps_dp_supervisor._group_is_owned(state, record)
        mps_dp_supervisor._terminate_record(record, state=state)
        survived = [pid for pid in children if mps_dp_supervisor._pid_is_live(pid)]
    finally:
        _cleanup(children)

    assert listed == []
    assert owned is True
    assert survived == []


LAUNCH_SCRIPT = REPO_ROOT / "examples" / "mps_dp" / "launch.sh"


def _launch_sh(state: Path, script: str) -> subprocess.CompletedProcess[str]:
    """Run a snippet against launch.sh's functions.

    Sourcing with the `list` verb runs only find_runs, so the helper functions
    become available without starting anything. `set +e` undoes the script's
    `set -e` so a predicate returning non-zero is an answer, not an abort.
    """
    return subprocess.run(
        [
            "bash",
            "-c",
            f'source "{LAUNCH_SCRIPT}" list >/dev/null\nset +e\n{script}',
        ],
        env={**os.environ, "STATE_ROOT": str(state.parents[1])},
        capture_output=True,
        text=True,
    )


@pytest.mark.skipif(os.name != "posix", reason="process groups require POSIX")
def test_launch_sh_teardown_owns_a_group_whose_leader_was_reaped(
    tmp_path: Path,
) -> None:
    """The teardown-side mirror: `down` must be able to clean an orphaned group.

    launch.sh carried the same leader-gated guard as the supervisor, which is
    why teardown could detect leaked groups and still not signal them. The
    persisted member list has to give bash the same leaderless proof.
    """
    state = tmp_path / "gpu-0" / "run-test"
    state.mkdir(parents=True)
    leader, children = _spawn_reapable_group(tmp_path)
    members_file = state / "group-members" / "replica_0.tsv"
    frozen = _launch_sh(
        state,
        f'persist_group_members "{members_file}" {leader.pid}; echo rc=$?',
    )
    start = mps_dp_supervisor._wait_for_start_time(leader.pid)
    _reap_leader(leader)

    try:
        probe = _launch_sh(
            state,
            f'group_is_owned {leader.pid} "{start}" {leader.pid}; echo leader=$?\n'
            f'group_ownership_proven "{state}" "{members_file}" '
            f'{leader.pid} "{start}" {leader.pid}; echo proven=$?\n'
            f'persisted_group_members "{members_file}" {leader.pid} | wc -l',
        )
    finally:
        _cleanup(children)

    assert "rc=0" in frozen.stdout, frozen.stderr
    # Leader-only proof fails, the member list still carries ownership.
    assert "leader=1" in probe.stdout, probe.stdout + probe.stderr
    assert "proven=0" in probe.stdout, probe.stdout + probe.stderr
    assert probe.stdout.strip().splitlines()[-1].strip() == "2"


@pytest.mark.skipif(os.name != "posix", reason="process groups require POSIX")
def test_launch_sh_run_token_proves_a_group_with_no_member_list(
    tmp_path: Path,
) -> None:
    state = tmp_path / "gpu-0" / "run-test"
    state.mkdir(parents=True)
    token = "round10-launch-token"
    (state / "run_token").write_text(token + "\n")
    leader, children = _spawn_reapable_group(tmp_path, env_token=token)
    start = mps_dp_supervisor._wait_for_start_time(leader.pid)
    _reap_leader(leader)

    try:
        probe = _launch_sh(
            state,
            f'group_ownership_proven "{state}" "{state}/group-members/replica_0.tsv" '
            f'{leader.pid} "{start}" {leader.pid}; echo proven=$?',
        )
        wrong = _launch_sh(
            state,
            f'group_token_matches {leader.pid} not-this-run; echo other=$?',
        )
    finally:
        _cleanup(children)

    assert "proven=0" in probe.stdout, probe.stdout + probe.stderr
    assert "other=1" in wrong.stdout, wrong.stdout + wrong.stderr
