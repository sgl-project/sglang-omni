# SPDX-License-Identifier: Apache-2.0
"""Process and placement helpers for managed local Omni router launches."""

from __future__ import annotations

import importlib.util
import os
import signal
import socket
import subprocess
import time
from collections.abc import Sequence

import httpx

from sglang_omni_router.launcher.config import LocalLauncherConfig


def worker_connect_host(host: str) -> str:
    if host in {"0.0.0.0", "::"}:
        return "127.0.0.1"
    return host


def build_worker_url(host: str, port: int) -> str:
    connect_host = worker_connect_host(host)
    if ":" in connect_host and not connect_host.startswith("["):
        connect_host = f"[{connect_host}]"
    return f"http://{connect_host}:{port}"


def port_is_available(host: str, port: int) -> bool:
    bind_host = host.strip("[]")
    family = socket.AF_INET6 if ":" in bind_host else socket.AF_INET
    with socket.socket(family, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((bind_host, port))
        except OSError:
            return False
    return True


def reserve_worker_ports(config: LocalLauncherConfig) -> list[int]:
    ports = [config.worker_base_port + index for index in range(config.num_workers)]
    unavailable = [
        str(port) for port in ports if not port_is_available(config.worker_host, port)
    ]
    if unavailable:
        raise RuntimeError(f"worker ports are already in use: {', '.join(unavailable)}")
    return ports


def split_cuda_visible_devices(value: str | None) -> list[str]:
    if value is None:
        return []
    return [device.strip() for device in value.split(",") if device.strip()]


def infer_available_cuda_devices() -> list[str]:
    visible_devices = split_cuda_visible_devices(os.environ.get("CUDA_VISIBLE_DEVICES"))
    if visible_devices:
        return visible_devices

    if importlib.util.find_spec("torch") is None:
        return []

    import torch

    return [str(index) for index in range(torch.cuda.device_count())]


def build_gpu_assignments(config: LocalLauncherConfig) -> list[str] | None:
    if config.worker_gpu_ids is not None:
        return config.worker_gpu_ids

    devices = infer_available_cuda_devices()
    if not devices:
        return None
    required = config.num_workers * config.num_gpus_per_worker
    if len(devices) < required:
        raise RuntimeError(
            "not enough CUDA devices for managed workers: "
            f"required={required}, available={len(devices)}"
        )

    assignments: list[str] = []
    for worker_index in range(config.num_workers):
        start = worker_index * config.num_gpus_per_worker
        stop = start + config.num_gpus_per_worker
        assignments.append(",".join(devices[start:stop]))
    return assignments


def wait_for_worker_health(
    *,
    worker_url: str,
    health_endpoint: str,
    process: subprocess.Popen,
    timeout: float,
) -> None:
    deadline = time.monotonic() + timeout
    last_error: str | None = None
    url = f"{worker_url}{health_endpoint}"

    while time.monotonic() < deadline:
        exit_code = process.poll()
        if exit_code is not None:
            raise RuntimeError(
                f"managed worker {worker_url} exited before becoming healthy "
                f"(exit_code={exit_code})"
            )
        try:
            response = httpx.get(url, timeout=2.0)
            if response.status_code == 200:
                return
            last_error = f"status={response.status_code}"
        except httpx.HTTPError as exc:
            last_error = str(exc)
        time.sleep(1)

    detail = f": {last_error}" if last_error else ""
    raise TimeoutError(f"managed worker {worker_url} did not become healthy{detail}")


def terminate_processes(processes: Sequence[subprocess.Popen]) -> None:
    for process in processes:
        if process.poll() is not None:
            continue
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGINT)
        except (ProcessLookupError, ChildProcessError):
            continue

    for process in processes:
        if process.poll() is not None:
            continue
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(process.pid), signal.SIGKILL)
            except (ProcessLookupError, ChildProcessError):
                continue
            process.wait(timeout=10)
