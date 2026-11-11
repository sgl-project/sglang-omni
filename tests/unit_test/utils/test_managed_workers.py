# SPDX-License-Identifier: Apache-2.0
"""GPU placement, startup failure, and shutdown contracts for managed workers."""

from __future__ import annotations

import signal
import subprocess
import sys
import threading
from concurrent.futures import Future
from typing import Literal
from unittest.mock import Mock

import pytest

import tests.utils.managed_workers as managed_workers
from tests.utils.managed_workers import (
    ManagedWorkerProcess,
    build_gpu_assignments,
    start_workers,
    wait_for_workers,
)


class FakeProcess:
    def __init__(
        self,
        pid: int,
        command: list[str],
        environment: dict[str, str],
    ) -> None:
        self.pid = pid
        self.command = command
        self.environment = environment

    def poll(self) -> None:
        return None


@pytest.fixture
def created_workers(
    monkeypatch: pytest.MonkeyPatch,
) -> list[FakeProcess]:
    created: list[FakeProcess] = []

    def fake_popen(
        command: list[str],
        *,
        env: dict[str, str],
        start_new_session: bool,
    ) -> FakeProcess:
        assert start_new_session
        process = FakeProcess(12345 + len(created), command, env)
        created.append(process)
        return process

    monkeypatch.setattr(managed_workers.shutil, "which", lambda command: command)
    monkeypatch.setattr(managed_workers, "port_is_available", lambda port, host: True)
    monkeypatch.setattr(managed_workers.subprocess, "Popen", fake_popen)
    return created


@pytest.mark.parametrize("placement", ["explicit", "grouped", "inherited"])
def test_worker_command_preserves_model_arguments_and_gpu_placement(
    monkeypatch: pytest.MonkeyPatch,
    created_workers: list[FakeProcess],
    placement: Literal["explicit", "grouped", "inherited"],
) -> None:
    """Worker children must receive the requested model and GPU groups."""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4,5,6,7")
    if placement == "explicit":
        worker_gpu_ids = ["8,9", "10,11"]
        expected_visibility = worker_gpu_ids
    elif placement == "grouped":
        worker_gpu_ids = build_gpu_assignments(["4", "5", "6", "7"], 2, 2)
        expected_visibility = ["4,5", "6,7"]
    else:
        worker_gpu_ids = build_gpu_assignments([], 2, 2)
        expected_visibility = ["4,5,6,7", "4,5,6,7"]
    workers = start_workers(
        model_path="Qwen/Qwen3-Omni-30B-A3B-Instruct",
        model_name="qwen3-omni",
        worker_host="127.0.0.1",
        worker_ports=[8011, 8012],
        worker_gpu_ids=worker_gpu_ids,
        worker_extra_args="--mem-fraction-static 0.6 --thinker-tp-size 2 --thinker-gpus '0,1'",
    )
    command = created_workers[0].command
    assert command[:8] == [
        "sgl-omni",
        "serve",
        "--model-path",
        "Qwen/Qwen3-Omni-30B-A3B-Instruct",
        "--host",
        "127.0.0.1",
        "--port",
        "8011",
    ]
    assert command[command.index("--model-name") + 1] == "qwen3-omni"
    assert command[command.index("--mem-fraction-static") + 1] == "0.6"
    assert command[command.index("--thinker-tp-size") + 1] == "2"
    assert command[command.index("--thinker-gpus") + 1] == "0,1"
    assert [
        process.environment["CUDA_VISIBLE_DEVICES"] for process in created_workers
    ] == expected_visibility
    assert [worker.process_group_id for worker in workers] == [
        process.pid for process in created_workers
    ]


@pytest.mark.parametrize("failure_phase", ["health", "wait", "submission"])
def test_startup_failure_cleans_up_workers(
    monkeypatch: pytest.MonkeyPatch,
    created_workers: list[FakeProcess],
    failure_phase: Literal["health", "wait", "submission"],
) -> None:
    """Failed or interrupted health setup must release every started worker."""
    terminated: list[ManagedWorkerProcess] = []

    def fail_health(worker: ManagedWorkerProcess, deadline: float) -> None:
        raise TimeoutError("worker did not become healthy")

    def interrupt_wait(futures: set[Future[None]], *, return_when: str) -> None:
        raise KeyboardInterrupt()

    monkeypatch.setattr(managed_workers, "stop_managed_workers", terminated.extend)
    monkeypatch.setattr(managed_workers, "wait_for_worker_health", fail_health)
    if failure_phase == "wait":
        monkeypatch.setattr(managed_workers, "wait", interrupt_wait)
        expected_error = KeyboardInterrupt
    elif failure_phase == "submission":
        monkeypatch.setattr(
            managed_workers.ThreadPoolExecutor,
            "submit",
            Mock(side_effect=KeyboardInterrupt),
        )
        expected_error = KeyboardInterrupt
    else:
        expected_error = TimeoutError
    workers = start_workers(
        model_path="model",
        model_name="model",
        worker_host="127.0.0.1",
        worker_ports=[8011],
        worker_gpu_ids=["7"],
    )
    with pytest.raises(expected_error):
        wait_for_workers(workers, 1)
    assert [worker.process for worker in terminated] == created_workers
    assert workers == []


def test_workers_wait_in_parallel(
    monkeypatch: pytest.MonkeyPatch,
    created_workers: list[FakeProcess],
) -> None:
    """A worker's health wait must not prevent other workers from starting theirs."""
    terminated: list[ManagedWorkerProcess] = []
    waiting_urls: list[str] = []
    waiting_lock = threading.Lock()
    both_waiting = threading.Event()

    def wait_health(worker: ManagedWorkerProcess, deadline: float) -> None:
        with waiting_lock:
            waiting_urls.append(worker.url)
            if len(waiting_urls) == 2:
                both_waiting.set()
            else:
                pass
        assert both_waiting.wait(timeout=2)
        if worker.url == "http://127.0.0.1:8012":
            raise RuntimeError("worker 2 failed")
        else:
            pass

    monkeypatch.setattr(managed_workers, "wait_for_worker_health", wait_health)
    monkeypatch.setattr(managed_workers, "stop_managed_workers", terminated.extend)
    workers = start_workers(
        model_path="model",
        model_name="model",
        worker_host="127.0.0.1",
        worker_ports=[8011, 8012],
        worker_gpu_ids=["0", "1"],
    )
    with pytest.raises(RuntimeError, match="worker 2 failed"):
        wait_for_workers(workers, 10)
    assert set(waiting_urls) == {"http://127.0.0.1:8011", "http://127.0.0.1:8012"}
    assert [worker.process for worker in terminated] == created_workers
    assert workers == []


def test_managed_shutdown_signals_only_worker_parents(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Worker parents must retain control of graceful stage shutdown."""
    events: list[tuple[str, int, int]] = []

    class FakeProcess:
        def __init__(self, pid: int) -> None:
            self.pid = pid

        def send_signal(self, sig: int) -> None:
            events.append(("parent", self.pid, sig))

        def wait(self, timeout: float) -> int:
            assert timeout >= 0
            events.append(("wait", self.pid, 0))
            return 0

    def check_group(process_group_id: int, sig: int) -> None:
        events.append(("group", process_group_id, sig))
        raise ProcessLookupError()

    monkeypatch.setattr(managed_workers.os, "killpg", check_group)
    workers = [
        managed_workers.ManagedWorkerProcess(
            url=f"http://127.0.0.1:{port}",
            process=FakeProcess(port),
            process_group_id=port,
        )
        for port in (8011, 8012)
    ]

    managed_workers.stop_managed_workers(workers)

    assert events == [
        ("parent", 8011, signal.SIGINT),
        ("parent", 8012, signal.SIGINT),
        ("wait", 8011, 0),
        ("group", 8011, 0),
        ("wait", 8012, 0),
        ("group", 8012, 0),
    ]


@pytest.mark.parametrize("parent_timed_out", [False, True])
def test_managed_shutdown_kills_remaining_group(
    monkeypatch: pytest.MonkeyPatch, parent_timed_out: bool
) -> None:
    """Children that survive graceful shutdown must be killed and reaped."""
    group_signals: list[int] = []

    class FakeProcess:
        def __init__(self) -> None:
            self.wait_count = 0
            self.signals: list[int] = []

        def send_signal(self, sig: int) -> None:
            self.signals.append(sig)

        def wait(self, timeout: float) -> int:
            self.wait_count += 1
            if parent_timed_out and self.wait_count == 1:
                raise subprocess.TimeoutExpired("worker", timeout)
            return 0

    def signal_group(process_group_id: int, sig: int) -> None:
        assert process_group_id == 8011
        group_signals.append(sig)

    def group_exited(process_group_id: int) -> bool:
        assert process_group_id == 8011
        return False

    monkeypatch.setattr(managed_workers.os, "killpg", signal_group)
    monkeypatch.setattr(managed_workers, "process_group_has_live_members", group_exited)
    process = FakeProcess()
    worker = managed_workers.ManagedWorkerProcess(
        url="http://127.0.0.1:8011",
        process=process,
        process_group_id=8011,
    )

    managed_workers.stop_managed_workers([worker])

    assert process.signals == [signal.SIGINT]
    assert process.wait_count == 2
    assert group_signals == (
        [signal.SIGKILL] if parent_timed_out else [0, signal.SIGKILL]
    )


def test_managed_shutdown_waits_for_orphaned_worker_child() -> None:
    """Shutdown must not return while an exited worker's child is still alive."""
    parent = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import subprocess, sys; "
            "subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)']); "
            "print('ready', flush=True)",
        ],
        stdout=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        assert parent.stdout is not None
        assert parent.stdout.readline().strip() == "ready"
        parent.wait(timeout=5)
        assert managed_workers.process_group_has_live_members(parent.pid)

        worker = managed_workers.ManagedWorkerProcess(
            url="http://127.0.0.1:8011",
            process=parent,
            process_group_id=parent.pid,
        )
        managed_workers.stop_managed_workers([worker])

        assert not managed_workers.process_group_has_live_members(parent.pid)
    finally:
        try:
            managed_workers.os.killpg(parent.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        parent.wait(timeout=5)
        if parent.stdout is not None:
            parent.stdout.close()
