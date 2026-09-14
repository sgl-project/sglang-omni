# SPDX-License-Identifier: Apache-2.0
"""Shared managed Rust-router helpers for Omni model CI tests."""

from __future__ import annotations

import os
import signal
import socket
import subprocess
import time
from collections import deque
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

import pytest
import requests

from sglang_omni.utils import find_available_port
from sglang_omni_router.python.launcher.config import LocalLauncherConfig
from sglang_omni_router.python.launcher.local import LocalLauncher
from tests.test_model.rust_router_config import CiRouterTopology, render_router_config
from tests.utils import (
    disable_proxy,
    server_log_file,
    start_server_from_cmd,
    stop_server,
)

RUST_ROUTER_BINARY_ENV = "SGLANG_OMNI_ROUTER_BIN"
REQUEST_TIMEOUT = 20
LOG_TAIL_LINES = 120


@dataclass
class ManagedRouterHandle:
    """Running router topology exposed to benchmark clients."""

    proc: subprocess.Popen
    port: int
    worker_ports: list[int]
    log_file: Path | None
    router_config: Path | None = None
    cleanup_manifest: Path | None = None
    worker_launcher: LocalLauncher | None = None
    is_router: bool = True
    router_ready_s: float | None = None
    stopped: bool = False

    def stop(self) -> None:
        if self.stopped:
            return
        try:
            stop_server(self.proc)
        finally:
            try:
                if self.worker_launcher is not None:
                    self.worker_launcher.shutdown()
            finally:
                if self.cleanup_manifest is not None:
                    cleanup_process_groups_from_manifest(self.cleanup_manifest)
                self.stopped = True


@dataclass
class RouterWorkerTrafficGuard:
    """Router worker dispatch snapshot for one benchmark run."""

    handle: ManagedRouterHandle
    label: str
    before_snapshot: dict | None

    def assert_served(
        self,
        *,
        min_total_requests: int | None = None,
        min_worker_share: float = 0.10,
    ) -> None:
        if not self.handle.is_router:
            return
        assert self.before_snapshot is not None
        try:
            assert_workers_served_requests_since(
                handle=self.handle,
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
    router_topology: CiRouterTopology,
    num_workers: int = 2,
    num_gpus_per_worker: int = 1,
    wait_timeout: int = 900,
    startup_timeout: int | None = None,
    log_prefix: str = "omni_router_logs",
    force_log: bool = False,
    external_worker_urls: list[str] | None = None,
    worker_env: dict[str, str] | None = None,
) -> Iterator[ManagedRouterHandle]:
    """Launch a Rust router over local or externally owned workers."""
    router_binary = _rust_router_binary()
    cleanup_manifest = (
        tmp_path_factory.mktemp("omni_router_cleanup") / "router_pgids.txt"
    )
    worker_launcher: LocalLauncher | None = None

    if external_worker_urls is None:
        worker_base_port = _find_available_port_range(num_workers)
        worker_ports = [worker_base_port + offset for offset in range(num_workers)]
        worker_urls = [f"http://127.0.0.1:{port}" for port in worker_ports]
        worker_launcher = LocalLauncher(
            LocalLauncherConfig(
                model_path=model_path,
                model_name=model_name,
                num_workers=num_workers,
                num_gpus_per_worker=num_gpus_per_worker,
                worker_host="127.0.0.1",
                worker_base_port=worker_base_port,
                worker_extra_args=worker_extra_args,
                wait_timeout=wait_timeout,
            ),
            worker_env=worker_env,
        )
    else:
        if len(external_worker_urls) != num_workers:
            raise ValueError(
                f"expected {num_workers} external workers, got "
                f"{len(external_worker_urls)}"
            )
        worker_urls = list(external_worker_urls)
        worker_ports = [_worker_port(url) for url in worker_urls]

    router_port = _find_available_port_excluding(worker_ports)
    router_config = _write_router_config(
        tmp_path_factory,
        topology=router_topology,
        router_port=router_port,
        worker_urls=worker_urls,
        model_name=model_name,
    )
    router_log = (
        tmp_path_factory.mktemp(log_prefix) / "server.log"
        if force_log
        else server_log_file(tmp_path_factory, log_prefix)
    )
    router_proc: subprocess.Popen | None = None
    handle: ManagedRouterHandle | None = None

    try:
        startup_t0 = time.perf_counter()
        router_proc = start_server_from_cmd(
            [str(router_binary), "--config", str(router_config)],
            router_log,
            router_port,
            timeout=startup_timeout or wait_timeout,
            tee=force_log,
            strip_proxy=True,
            health_path="/live",
            health_body_contains=None,
        )
        _record_process_group(cleanup_manifest, os.getpgid(router_proc.pid))

        if worker_launcher is not None:
            worker_launcher.launch()
            for worker in worker_launcher.workers:
                _record_process_group(cleanup_manifest, worker.process_group_id)
            worker_launcher.wait_ready()

        wait_for_all_router_workers(
            router_port,
            expected_workers=num_workers,
            timeout=wait_timeout,
        )
        router_ready_s = time.perf_counter() - startup_t0
        print(
            "[Omni Router CI] topology "
            f"router_port={router_port} worker_ports={worker_ports} "
            f"topology={router_topology.value} binary={router_binary}"
        )
        handle = ManagedRouterHandle(
            proc=router_proc,
            port=router_port,
            worker_ports=worker_ports,
            log_file=router_log,
            router_config=router_config,
            cleanup_manifest=cleanup_manifest,
            worker_launcher=worker_launcher,
            router_ready_s=router_ready_s,
        )
        yield handle
    finally:
        if handle is not None:
            handle.stop()
        else:
            if router_proc is not None:
                stop_server(router_proc)
            if worker_launcher is not None:
                worker_launcher.shutdown()
            cleanup_process_groups_from_manifest(cleanup_manifest)


@contextmanager
def router_worker_traffic_guard(
    handle: ManagedRouterHandle,
    *,
    label: str,
) -> Iterator[RouterWorkerTrafficGuard]:
    before_snapshot = (
        router_get_json(handle.port, "/diagnostics") if handle.is_router else None
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
    payload = response.json()
    if not isinstance(payload, dict):
        raise TypeError(f"expected JSON object from router {path}")
    return payload


def wait_for_all_router_workers(
    port: int,
    *,
    expected_workers: int,
    timeout: int = 120,
) -> None:
    deadline = time.monotonic() + timeout
    last_payload: dict | None = None
    while time.monotonic() < deadline:
        try:
            last_payload = router_get_json(port, "/diagnostics")
        except (requests.RequestException, ValueError, TypeError):
            time.sleep(1)
            continue
        workers = last_payload.get("workers", [])
        if (
            last_payload.get("lifecycle") == "serving"
            and last_payload.get("ready") is True
            and len(workers) == expected_workers
            and all(worker.get("routable") is True for worker in workers)
        ):
            return
        time.sleep(1)
    raise TimeoutError(f"router workers did not become fully routable: {last_payload}")


def assert_router_healthy(handle: ManagedRouterHandle) -> dict:
    return wait_for_router_quiescence(
        handle.port,
        expected_workers=len(handle.worker_ports),
    )


def wait_for_router_quiescence(
    port: int,
    *,
    expected_workers: int,
    timeout: int = REQUEST_TIMEOUT,
) -> dict:
    deadline = time.monotonic() + timeout
    diagnostics: dict | None = None
    while time.monotonic() < deadline:
        diagnostics = router_get_json(port, "/diagnostics")
        if _router_is_quiescent(diagnostics, expected_workers=expected_workers):
            return diagnostics
        time.sleep(0.1)
    raise TimeoutError(f"router did not become healthy and quiescent: {diagnostics}")


def _router_is_quiescent(diagnostics: dict, *, expected_workers: int) -> bool:
    workers = diagnostics.get("workers", [])
    resources = diagnostics.get("resources", {})
    return (
        diagnostics.get("lifecycle") == "serving"
        and diagnostics.get("ready") is True
        and len(workers) == expected_workers
        and all(worker.get("routable") is True for worker in workers)
        and all(worker.get("active_requests") == 0 for worker in workers)
        and all(
            capacity.get("in_flight") == 0
            for worker in workers
            for capacity in worker.get("capacity", [])
        )
        and all(
            admission.get("in_flight") == 0
            for admission in diagnostics.get("admission", [])
        )
        and resources.get("buffered_request_bytes", {}).get("in_use") == 0
        and resources.get("classification_slots", {}).get("in_use") == 0
        and resources.get("websocket_sessions_registered") == 0
    )


def print_router_snapshot(label: str, snapshot: dict) -> None:
    worker_states = [
        (
            worker.get("worker_id"),
            worker.get("health"),
            worker.get("active_requests"),
            sum(_dispatch_counts(worker).values()),
            worker.get("routable"),
            worker.get("voice_owner"),
        )
        for worker in snapshot.get("workers", [])
    ]
    print(
        f"[Omni Router CI] {label} lifecycle={snapshot.get('lifecycle')} "
        f"ready={snapshot.get('ready')} "
        "workers=(id, health, active, dispatches, routable, voice_owner) "
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
    if handle.is_router:
        try:
            print_router_snapshot(
                "failure /diagnostics snapshot",
                router_get_json(handle.port, "/diagnostics"),
            )
        except Exception as exc:  # pragma: no cover - diagnostic path
            print(f"[Omni Router CI] failed to fetch /diagnostics: {exc}")
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


def assert_workers_served_requests_since(
    *,
    handle: ManagedRouterHandle,
    before_snapshot: dict,
    label: str,
    min_total_requests: int | None = None,
    min_worker_share: float = 0.10,
) -> dict:
    after_snapshot = assert_router_healthy(handle)
    delta_snapshot = worker_request_delta(before_snapshot, after_snapshot)
    print_router_snapshot(f"{label} /diagnostics delta", delta_snapshot)
    assert_workers_served_requests(
        delta_snapshot,
        min_total_requests=min_total_requests,
        min_worker_share=min_worker_share,
    )
    return delta_snapshot


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


def _rust_router_binary() -> Path:
    configured = os.environ.get(RUST_ROUTER_BINARY_ENV, "").strip()
    if not configured:
        raise RuntimeError(
            f"{RUST_ROUTER_BINARY_ENV} must name the prepared Rust router executable"
        )
    resolved = Path(configured).expanduser().resolve()
    if not resolved.is_file() or not os.access(resolved, os.X_OK):
        raise RuntimeError(f"Rust router binary is not executable: {resolved}")
    return resolved


def _write_router_config(
    tmp_path_factory: pytest.TempPathFactory,
    *,
    topology: CiRouterTopology,
    router_port: int,
    worker_urls: list[str],
    model_name: str,
) -> Path:
    config_path = tmp_path_factory.mktemp("omni_router_config") / "router.toml"
    config_path.write_text(
        render_router_config(
            topology=topology,
            router_port=router_port,
            worker_urls=worker_urls,
            model_name=model_name,
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


def _worker_port(url: str) -> int:
    port = urlsplit(url).port
    if port is None:
        raise ValueError(f"worker URL has no explicit port: {url}")
    return port


def _dispatch_counts(worker: dict) -> dict[str, int]:
    return {
        str(entry["class"]): int(entry["requests"])
        for entry in worker.get("dispatches", [])
    }


def worker_request_delta(before: dict, after: dict) -> dict:
    before_workers = {worker["worker_id"]: worker for worker in before["workers"]}
    delta_workers = []
    for worker in after["workers"]:
        previous_counts = _dispatch_counts(before_workers.get(worker["worker_id"], {}))
        class_counts = {
            service_class: count - previous_counts.get(service_class, 0)
            for service_class, count in _dispatch_counts(worker).items()
        }
        delta_workers.append(
            {
                **worker,
                "dispatches": [
                    {"class": service_class, "requests": count}
                    for service_class, count in class_counts.items()
                ],
                "routed_requests": sum(class_counts.values()),
                "routed_requests_by_class": class_counts,
            }
        )
    return {**after, "workers": delta_workers}
