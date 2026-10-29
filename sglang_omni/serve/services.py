# SPDX-License-Identifier: Apache-2.0
"""Start the services a pipeline depends on and stop them with it."""

import asyncio
import logging
import os
import signal
import subprocess
import time
import urllib.error
import urllib.request

from sglang_omni.config.schema import ServiceConfig

logger = logging.getLogger(__name__)

READY_PATH = "/v1/models"
READY_TIMEOUT_S = 600.0
POLL_INTERVAL_S = 1.0
STOP_GRACE_S = 15.0


def is_serving(port: int) -> bool:
    try:
        with urllib.request.urlopen(
            f"http://127.0.0.1:{port}{READY_PATH}", timeout=2
        ) as response:
            return response.status == 200
    except (urllib.error.URLError, TimeoutError):
        return False


class ServiceProcesses:
    """A pipeline's services, started in order and stopped together."""

    def __init__(self, services: list[ServiceConfig]) -> None:
        self.services = services
        self.started: list[tuple[ServiceConfig, subprocess.Popen[bytes]]] = []

    async def __aenter__(self) -> "ServiceProcesses":
        try:
            for service in self.services:
                await self.start(service)
        except BaseException:
            await self.stop()
            raise
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.stop()

    async def start(self, service: ServiceConfig) -> None:
        if await asyncio.to_thread(is_serving, service.port):
            logger.info("service %r already serving on :%d", service.name, service.port)
            return
        else:
            pass
        logger.info("starting service %r: %s", service.name, " ".join(service.command))
        process = subprocess.Popen(
            service.command,
            env={**os.environ, **service.env},
            start_new_session=True,
        )
        self.started.append((service, process))
        deadline = time.monotonic() + READY_TIMEOUT_S
        while not await asyncio.to_thread(is_serving, service.port):
            if process.poll() is not None:
                raise RuntimeError(
                    f"service {service.name!r} exited with code {process.returncode} before :{service.port} answered"
                )
            elif time.monotonic() >= deadline:
                raise TimeoutError(
                    f"service {service.name!r} did not answer on :{service.port} within {READY_TIMEOUT_S:g} s"
                )
            else:
                await asyncio.sleep(POLL_INTERVAL_S)
        logger.info("service %r serving on :%d", service.name, service.port)

    async def watch(self) -> None:
        while True:
            for service, process in self.started:
                if process.poll() is not None:
                    raise RuntimeError(
                        f"service {service.name!r} exited with code {process.returncode}"
                    )
                else:
                    pass
            await asyncio.sleep(POLL_INTERVAL_S)

    async def stop(self) -> None:
        for service, process in self.started:
            if process.poll() is None:
                logger.info("stopping service %r", service.name)
                os.killpg(process.pid, signal.SIGTERM)
            else:
                pass
        deadline = time.monotonic() + STOP_GRACE_S
        for service, process in self.started:
            remaining = max(0.0, deadline - time.monotonic())
            try:
                await asyncio.to_thread(process.wait, remaining)
            except subprocess.TimeoutExpired:
                logger.warning(
                    "service %r did not stop in %g s; killing",
                    service.name,
                    STOP_GRACE_S,
                )
                os.killpg(process.pid, signal.SIGKILL)
                await asyncio.to_thread(process.wait)
        self.started = []
