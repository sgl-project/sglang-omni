# SPDX-License-Identifier: Apache-2.0
"""Shared managed-router helpers for Omni model CI tests."""

from __future__ import annotations

import os
import signal
import socket
import subprocess
import sys
import time
from collections import deque
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import pytest
import requests
import yaml

from sglang_omni.utils import find_available_port
from tests.utils import (
    disable_proxy,
    server_log_file,
    start_server_from_cmd,
    stop_server,
)

REQUEST_TIMEOUT = 20
LOG_TAIL_LINES = 120
ROUTER_POLICY = "least_request"
ROUTER_CLEANUP_MANIFEST_ENV = "SGLANG_OMNI_ROUTER_CLEANUP_MANIFEST"


@dataclass
class ManagedRouterHandle:
    """Running managed-router topology exposed to benchmark clients."""

    proc: subprocess.Popen
    port: int
    worker_ports: list[int]
    log_file: Path | None
    launcher_config: Path | None = None
    cleanup_manifest: Path | None = None
    is_router: bool = True
    stopped: bool = False

    def stop(self) -> None:
        if self.stopped:
            return
        try:
            stop_server(self.proc)
        finally:
            if self.cleanup_manifest is not None:
                cleanup_process_groups_from_manifest(self.cleanup_manifest)
            self.stopped = True


@dataclass
class RouterWorkerTrafficGuard:
    """Router worker counter snapshot for one benchmark run."""

    handle: ManagedRouterHandle
    label: str
    before_snapshot: dict

    def assert_served(
        self,
        *,
        min_total_requests: int | None = None,
        min_worker_share: float = 0.10,
    ) -> None:
        if not self.handle.is_router:
            return
        try:
            assert_workers_served_requests_since(
                port=self.handle.port,
                before_snapshot=self.before_snapshot,
                label=self.label,
                min_total_requests=min_total_requests,
                min_worker_share=min_worker_share,
            )
        except Exception:
            print_router_diagnostics(self.handle)
            raise


@contextmanager
def launch_managed_router(
    *,
    tmp_path_factory: pytest.TempPathFactory,
    model_path: str,
    model_name: str,
    worker_extra_args: str,
    num_workers: int = 2,
    num_gpus_per_worker: int = 1,
    wait_timeout: int = 900,
    startup_timeout: int | None = None,
    log_prefix: str = "omni_router_logs",
) -> Iterator[ManagedRouterHandle]:
    worker_base_port = _find_available_port_range(num_workers)
    worker_ports = [worker_base_port + offset for offset in range(num_workers)]
    router_port = _find_available_port_excluding(worker_ports)
    cleanup_manifest = (
        tmp_path_factory.mktemp("omni_router_cleanup") / "router_pgids.txt"
    )
    launcher_config = _write_launcher_config(
        tmp_path_factory,
        model_path=model_path,
        model_name=model_name,
        num_workers=num_workers,
        num_gpus_per_worker=num_gpus_per_worker,
        worker_base_port=worker_base_port,
        worker_extra_args=worker_extra_args,
        wait_timeout=wait_timeout,
    )
    router_log = server_log_file(tmp_path_factory, log_prefix)
    router_proc: subprocess.Popen | None = None
    handle: ManagedRouterHandle | None = None

    try:
        router_cmd = [
            sys.executable,
            "-m",
            "sglang_omni_router.serve",
            "--host",
            "0.0.0.0",
            "--port",
            str(router_port),
            "--launcher-config",
            str(launcher_config),
            "--policy",
            ROUTER_POLICY,
            "--health-success-threshold",
            "1",
            "--health-failure-threshold",
            "2",
            "--health-check-interval-secs",
            "2",
            "--log-level",
            "info",
        ]
        router_proc = start_server_from_cmd(
            router_cmd,
            router_log,
            router_port,
            timeout=startup_timeout or wait_timeout + 60,
            env={ROUTER_CLEANUP_MANIFEST_ENV: str(cleanup_manifest)},
        )
        _record_process_group(cleanup_manifest, os.getpgid(router_proc.pid))
        wait_for_all_router_workers(
            router_port,
            expected_workers=num_workers,
            timeout=wait_timeout,
        )
        print(
            "[Omni Router CI] topology "
            f"router_port={router_port} worker_ports={worker_ports} "
            f"launcher_config={launcher_config} policy={ROUTER_POLICY}"
        )
        handle = ManagedRouterHandle(
            proc=router_proc,
            port=router_port,
            worker_ports=worker_ports,
            log_file=router_log,
            launcher_config=launcher_config,
            cleanup_manifest=cleanup_manifest,
        )
        yield handle
    finally:
        if handle is not None:
            handle.stop()
        elif router_proc is not None:
            stop_server(router_proc)
        cleanup_process_groups_from_manifest(cleanup_manifest)


@contextmanager
def router_worker_traffic_guard(
    handle: ManagedRouterHandle,
    *,
    label: str,
) -> Iterator[RouterWorkerTrafficGuard]:
    before_snapshot = (
        router_get_json(handle.port, "/workers") if handle.is_router else {}
    )
    guard = RouterWorkerTrafficGuard(
        handle=handle,
        label=label,
        before_snapshot=before_snapshot,
    )
    try:
        yield guard
    except Exception:
        print_router_diagnostics(handle)
        raise


def router_get_json(port: int, path: str) -> dict:
    with disable_proxy():
        response = requests.get(
            f"http://127.0.0.1:{port}{path}",
            timeout=REQUEST_TIMEOUT,
        )
    response.raise_for_status()
    return response.json()


def wait_for_all_router_workers(
    port: int,
    *,
    expected_workers: int,
    timeout: int = 120,
) -> None:
    deadline = time.monotonic() + timeout
    last_payload: dict | None = None
    while time.monotonic() < deadline:
        last_payload = router_get_json(port, "/workers")
        if (
            last_payload["total_workers"] == expected_workers
            and last_payload["healthy_workers"] == expected_workers
            and last_payload["routable_workers"] == expected_workers
        ):
            return
        time.sleep(1)
    raise TimeoutError(f"router workers did not become fully routable: {last_payload}")


def print_worker_snapshot(label: str, snapshot: dict) -> None:
    worker_states = [
        (
            worker["display_id"],
            worker["health_state"],
            worker["active_requests"],
            worker.get("routed_requests", 0),
            worker.get("successful_requests", 0),
            worker.get("failed_requests", 0),
            worker["routable"],
        )
        for worker in snapshot["workers"]
    ]
    print(
        f"[Omni Router CI] {label} "
        f"healthy={snapshot['healthy_workers']} "
        f"routable={snapshot['routable_workers']} "
        f"workers=(id, state, active, routed, successful, failed, routable) "
        f"{worker_states}"
    )


def print_log_tail(label: str, log_file: Path | None) -> None:
    if log_file is None:
        print(f"[Omni Router CI] {label} log is streamed to terminal outside CI")
        return
    if not log_file.exists():
        print(f"[Omni Router CI] {label} log missing: {log_file}")
        return
    with log_file.open("r", encoding="utf-8", errors="replace") as log_handle:
        lines = deque(log_handle, maxlen=LOG_TAIL_LINES)
    print(f"\n[Omni Router CI] {label} log tail ({log_file})")
    for line in lines:
        print(line.rstrip())


def print_router_diagnostics(handle: ManagedRouterHandle) -> None:
    try:
        print_worker_snapshot(
            "failure /workers snapshot",
            router_get_json(handle.port, "/workers"),
        )
    except Exception as exc:  # pragma: no cover - diagnostic path
        print(f"[Omni Router CI] failed to fetch /workers during diagnostics: {exc}")
    print_log_tail("router", handle.log_file)


def assert_workers_served_requests(
    snapshot: dict,
    *,
    expected_workers: int = 2,
    min_total_requests: int | None = None,
    min_worker_share: float = 0.10,
) -> None:
    workers = snapshot["workers"]
    routed_counts = [int(worker.get("routed_requests", 0)) for worker in workers]
    successful_counts = [
        int(worker.get("successful_requests", 0)) for worker in workers
    ]
    failed_counts = [int(worker.get("failed_requests", 0)) for worker in workers]
    total_routed = sum(routed_counts)
    min_expected = max(1, int(total_routed * min_worker_share))

    assert len(workers) == expected_workers
    if min_total_requests is not None:
        assert total_routed >= min_total_requests, (
            f"Expected at least {min_total_requests} routed requests, "
            f"got {total_routed}: {routed_counts}"
        )
    assert all(count >= min_expected for count in routed_counts), (
        f"All router workers must serve traffic. routed={routed_counts}, "
        f"minimum_per_worker={min_expected}"
    )
    assert sum(successful_counts) == total_routed, (
        f"All routed requests should succeed. successful={successful_counts}, "
        f"routed={routed_counts}"
    )
    assert sum(failed_counts) == 0, f"Router recorded request failures: {failed_counts}"


def assert_workers_served_requests_since(
    *,
    port: int,
    before_snapshot: dict,
    label: str,
    min_total_requests: int | None = None,
    min_worker_share: float = 0.10,
) -> None:
    delta_snapshot = _worker_request_delta(
        before_snapshot,
        router_get_json(port, "/workers"),
    )
    print_worker_snapshot(f"{label} /workers delta", delta_snapshot)
    assert_workers_served_requests(
        delta_snapshot,
        min_total_requests=min_total_requests,
        min_worker_share=min_worker_share,
    )


def cleanup_process_groups_from_manifest(manifest: Path) -> None:
    if not manifest.exists():
        return
    process_group_ids: set[int] = set()
    for line in manifest.read_text().splitlines():
        try:
            process_group_ids.add(int(line.strip()))
        except ValueError:
            continue
    for sig, wait_seconds in ((signal.SIGTERM, 5), (signal.SIGKILL, 1)):
        remaining: set[int] = set()
        for process_group_id in process_group_ids:
            try:
                os.killpg(process_group_id, sig)
                remaining.add(process_group_id)
            except ProcessLookupError:
                continue
        if not remaining:
            return
        time.sleep(wait_seconds)
        process_group_ids = {
            process_group_id
            for process_group_id in remaining
            if _process_group_exists(process_group_id)
        }


def _write_launcher_config(
    tmp_path_factory: pytest.TempPathFactory,
    *,
    model_path: str,
    model_name: str,
    num_workers: int,
    num_gpus_per_worker: int,
    worker_base_port: int,
    worker_extra_args: str,
    wait_timeout: int,
) -> Path:
    config_path = tmp_path_factory.mktemp("omni_router_launcher") / "launcher.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "launcher": {
                    "backend": "local",
                    "model_path": model_path,
                    "model_name": model_name,
                    "num_workers": num_workers,
                    "num_gpus_per_worker": num_gpus_per_worker,
                    "worker_host": "127.0.0.1",
                    "worker_base_port": worker_base_port,
                    "worker_extra_args": worker_extra_args,
                    "wait_timeout": wait_timeout,
                }
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return config_path


def _record_process_group(manifest: Path, process_group_id: int) -> None:
    with manifest.open("a", encoding="utf-8") as handle:
        handle.write(f"{process_group_id}\n")


def _process_group_exists(process_group_id: int) -> bool:
    try:
        os.killpg(process_group_id, 0)
        return True
    except ProcessLookupError:
        return False


def _find_available_port_excluding(excluded: list[int]) -> int:
    excluded_ports = set(excluded)
    while True:
        port = find_available_port()
        if port not in excluded_ports:
            return port


def _find_available_port_range(count: int) -> int:
    for _ in range(100):
        base_port = find_available_port()
        candidates = [base_port + offset for offset in range(count)]
        if all(_port_is_available(port) for port in candidates):
            return base_port
    raise RuntimeError(f"failed to find {count} consecutive available ports")


def _port_is_available(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(("127.0.0.1", port))
        except OSError:
            return False
    return True


def _worker_request_delta(before: dict, after: dict) -> dict:
    before_workers = {worker["display_id"]: worker for worker in before["workers"]}
    delta_workers = []
    for worker in after["workers"]:
        previous = before_workers.get(worker["display_id"], {})
        delta_worker = dict(worker)
        for field in ("routed_requests", "successful_requests", "failed_requests"):
            delta_worker[field] = int(worker.get(field, 0)) - int(
                previous.get(field, 0)
            )
        delta_workers.append(delta_worker)
    return {**after, "workers": delta_workers}
