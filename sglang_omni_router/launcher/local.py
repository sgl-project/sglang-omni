# SPDX-License-Identifier: Apache-2.0
"""Local subprocess launcher for complete Omni V1 worker replicas."""

from __future__ import annotations

import logging
import os
import shlex
import shutil
import subprocess
import time
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass

from sglang_omni_router.launcher.config import LocalLauncherConfig
from sglang_omni_router.launcher.utils import (
    build_gpu_assignments,
    build_worker_url,
    reserve_worker_ports,
    terminate_processes,
    wait_for_worker_health,
)

logger = logging.getLogger("sglang_omni_router.launcher")


@dataclass
class ManagedWorkerProcess:
    url: str
    port: int
    cuda_visible_devices: str | None
    process: subprocess.Popen


class LocalLauncher:
    def __init__(
        self,
        config: LocalLauncherConfig,
        *,
        command: str = "sgl-omni",
        health_endpoint: str = "/health",
    ) -> None:
        self.config = config
        self.command = command
        self.health_endpoint = health_endpoint
        self.workers: list[ManagedWorkerProcess] = []

    @property
    def worker_urls(self) -> list[str]:
        return [worker.url for worker in self.workers]

    def build_worker_command(self, port: int) -> list[str]:
        command = [
            self.command,
            "serve",
            "--version",
            "v1",
            "--model-path",
            self.config.model_path,
            "--host",
            self.config.worker_host,
            "--port",
            str(port),
        ]
        if self.config.model_name is not None:
            command.extend(["--model-name", self.config.model_name])
        if self.config.worker_extra_args:
            command.extend(shlex.split(self.config.worker_extra_args))
        return command

    def launch(self) -> list[str]:
        if self.workers:
            raise RuntimeError("managed workers have already been launched")
        if shutil.which(self.command) is None:
            raise RuntimeError(f"{self.command!r} was not found on PATH")

        ports = reserve_worker_ports(self.config)
        gpu_assignments = build_gpu_assignments(self.config)

        try:
            for worker_index, port in enumerate(ports):
                cuda_visible_devices = (
                    gpu_assignments[worker_index]
                    if gpu_assignments is not None
                    else None
                )
                worker_url = build_worker_url(self.config.worker_host, port)
                command = self.build_worker_command(port)
                env = os.environ.copy()
                placement = "default CUDA visibility"
                if cuda_visible_devices is not None:
                    env["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
                    placement = f"CUDA_VISIBLE_DEVICES={cuda_visible_devices}"
                logger.info(
                    f"Starting managed Omni V1 worker {worker_index} on {worker_url} "
                    f"with {placement}: {' '.join(shlex.quote(arg) for arg in command)}"
                )
                process = subprocess.Popen(
                    command,
                    env=env,
                    start_new_session=True,
                )
                self.workers.append(
                    ManagedWorkerProcess(
                        url=worker_url,
                        port=port,
                        cuda_visible_devices=cuda_visible_devices,
                        process=process,
                    )
                )
        except BaseException:
            self.shutdown()
            raise
        return self.worker_urls

    def wait_ready(self) -> None:
        if not self.workers:
            return

        deadline = time.monotonic() + self.config.wait_timeout
        executor = ThreadPoolExecutor(max_workers=len(self.workers))
        futures: set[Future[None]] = {
            executor.submit(self._wait_for_worker_ready, worker, deadline)
            for worker in self.workers
        }
        try:
            while futures:
                done, futures = wait(futures, return_when=FIRST_COMPLETED)
                for future in done:
                    future.result()
        except BaseException:
            self.shutdown()
            raise
        finally:
            executor.shutdown(wait=True, cancel_futures=True)

    def _wait_for_worker_ready(
        self, worker: ManagedWorkerProcess, deadline: float
    ) -> None:
        timeout = deadline - time.monotonic()
        if timeout <= 0:
            raise TimeoutError("managed workers did not become healthy before timeout")
        wait_for_worker_health(
            worker_url=worker.url,
            health_endpoint=self.health_endpoint,
            process=worker.process,
            timeout=timeout,
        )
        logger.info(f"Managed Omni V1 worker is healthy: {worker.url}")

    def launch_and_wait(self) -> list[str]:
        self.launch()
        self.wait_ready()
        return self.worker_urls

    def shutdown(self) -> None:
        if not self.workers:
            return
        terminate_processes([worker.process for worker in self.workers])
        self.workers.clear()
