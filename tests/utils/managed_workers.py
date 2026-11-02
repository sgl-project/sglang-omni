# SPDX-License-Identifier: Apache-2.0
"""Start, wait for, and stop complete local Omni worker replicas."""

from __future__ import annotations

import logging
import os
import shlex
import shutil
import signal
import socket
import subprocess
import time
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass

import httpx

logger = logging.getLogger(__name__)
GROUP_EXIT_POLL_INTERVAL_S = 0.05


@dataclass(kw_only=True)
class ManagedWorkerProcess:
    url: str
    process: subprocess.Popen[bytes]
    process_group_id: int


def port_is_available(port: int, host: str = "127.0.0.1") -> bool:
    bind_host = host.strip("[]")
    family = socket.AF_INET6 if ":" in bind_host else socket.AF_INET
    with socket.socket(family, socket.SOCK_STREAM) as connection:
        connection.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            connection.bind((bind_host, port))
        except OSError:
            return False
    return True


def build_gpu_assignments(
    devices: list[str],
    num_workers: int,
    num_gpus_per_worker: int,
) -> list[str] | None:
    if not devices:
        return None
    else:
        required = num_workers * num_gpus_per_worker
        if len(devices) < required:
            raise RuntimeError(
                "not enough CUDA devices for managed workers: "
                f"required={required}, available={len(devices)}"
            )
        else:
            return [
                ",".join(devices[start : start + num_gpus_per_worker])
                for start in range(0, required, num_gpus_per_worker)
            ]


def start_workers(
    *,
    model_path: str,
    model_name: str,
    worker_host: str,
    worker_ports: list[int],
    worker_gpu_ids: list[str] | None,
    worker_extra_args: str = "",
    worker_env: dict[str, str] | None = None,
) -> list[ManagedWorkerProcess]:
    if shutil.which("sgl-omni") is None:
        raise RuntimeError("'sgl-omni' was not found on PATH")
    else:
        unavailable_ports = [
            str(port)
            for port in worker_ports
            if not port_is_available(port, worker_host)
        ]
        if unavailable_ports:
            raise RuntimeError(
                f"worker ports are already in use: {', '.join(unavailable_ports)}"
            )
        else:
            pass

    workers: list[ManagedWorkerProcess] = []
    try:
        for worker_index, port in enumerate(worker_ports):
            cuda_visible_devices = (
                worker_gpu_ids[worker_index] if worker_gpu_ids is not None else None
            )
            worker_url = f"http://{worker_host}:{port}"
            worker_command = [
                "sgl-omni",
                "serve",
                "--model-path",
                model_path,
                "--host",
                worker_host,
                "--port",
                str(port),
                "--model-name",
                model_name,
                *shlex.split(worker_extra_args),
            ]
            environment = os.environ.copy()
            environment.update(worker_env or {})
            placement = "default CUDA visibility"
            if cuda_visible_devices is not None:
                environment["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
                placement = f"CUDA_VISIBLE_DEVICES={cuda_visible_devices}"
            else:
                pass
            logger.info(
                f"Starting managed Omni worker {worker_index} on {worker_url} "
                f"with {placement}: {shlex.join(worker_command)}"
            )
            process = subprocess.Popen(
                worker_command,
                env=environment,
                start_new_session=True,
            )
            workers.append(
                ManagedWorkerProcess(
                    url=worker_url,
                    process=process,
                    process_group_id=process.pid,
                )
            )
    except BaseException:
        stop_managed_workers(workers)
        raise
    return workers


def wait_for_workers(
    workers: list[ManagedWorkerProcess], timeout_seconds: float
) -> None:
    if not workers:
        return
    else:
        deadline = time.monotonic() + timeout_seconds
        executor = ThreadPoolExecutor(max_workers=len(workers))
        try:
            futures: set[Future[None]] = {
                executor.submit(wait_for_worker_health, worker, deadline)
                for worker in workers
            }
            while futures:
                done, futures = wait(futures, return_when=FIRST_COMPLETED)
                for future in done:
                    future.result()
        except BaseException:
            stop_managed_workers(workers)
            workers.clear()
            raise
        finally:
            executor.shutdown(wait=True, cancel_futures=True)


def wait_for_worker_health(worker: ManagedWorkerProcess, deadline: float) -> None:
    last_error: str | None = None
    while time.monotonic() < deadline:
        exit_code = worker.process.poll()
        if exit_code is not None:
            raise RuntimeError(
                f"managed worker {worker.url} exited before becoming healthy "
                f"(exit_code={exit_code})"
            )
        else:
            try:
                response = httpx.get(f"{worker.url}/health", timeout=2.0)
                if response.status_code == 200:
                    logger.info(f"Managed Omni worker is healthy: {worker.url}")
                    return
                else:
                    last_error = f"status={response.status_code}"
            except httpx.HTTPError as error:
                last_error = str(error)
        time.sleep(1)

    detail = f": {last_error}" if last_error else ""
    raise TimeoutError(f"managed worker {worker.url} did not become healthy{detail}")


def stop_managed_workers(workers: list[ManagedWorkerProcess]) -> None:
    for worker in workers:
        worker.process.send_signal(signal.SIGINT)

    deadline = time.monotonic() + 30
    remaining: list[ManagedWorkerProcess] = []
    for worker in workers:
        timeout = max(0.0, deadline - time.monotonic())
        try:
            worker.process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            remaining.append(worker)
        else:
            try:
                os.killpg(worker.process_group_id, 0)
            except ProcessLookupError:
                pass
            else:
                remaining.append(worker)

    for worker in remaining:
        logger.warning(
            f"Managed Omni worker group {worker.process_group_id} "
            "remained after shutdown; killing"
        )
        try:
            os.killpg(worker.process_group_id, signal.SIGKILL)
        except ProcessLookupError:
            pass

    for worker in remaining:
        worker.process.wait(timeout=10)
        deadline = time.monotonic() + 10
        while process_group_has_live_members(worker.process_group_id):
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"Managed Omni worker group {worker.process_group_id} did not exit"
                )
            else:
                time.sleep(GROUP_EXIT_POLL_INTERVAL_S)


def process_group_has_live_members(process_group_id: int) -> bool:
    result = subprocess.run(
        ["ps", "-A", "-o", "pgid=,stat="],
        capture_output=True,
        text=True,
        check=True,
    )
    for line in result.stdout.splitlines():
        group_id, status = line.split()
        if group_id == str(process_group_id) and not status.startswith(("Z", "X")):
            return True
        else:
            pass
    return False
