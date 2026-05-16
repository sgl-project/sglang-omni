# SPDX-License-Identifier: Apache-2.0
"""Background worker health checks for the Omni router."""

from __future__ import annotations

import asyncio
import logging
import random

import httpx

from sglang_omni_router.config import RouterConfig
from sglang_omni_router.worker import Worker

logger = logging.getLogger(__name__)


class HealthChecker:
    def __init__(
        self,
        *,
        workers: list[Worker],
        config: RouterConfig,
        client: httpx.AsyncClient,
    ) -> None:
        self._workers = workers
        self._config = config
        self._client = client
        self._task: asyncio.Task[None] | None = None

    async def check_all_workers_health(self) -> None:
        workers = tuple(self._workers)
        results = await asyncio.gather(
            *(self._check_worker_health(worker) for worker in workers),
            return_exceptions=True,
        )
        for worker, result in zip(workers, results, strict=True):
            if isinstance(result, Exception):
                logger.error(
                    f"Unexpected error while checking worker {worker.display_id} "
                    f"health: {type(result).__name__}: {result}",
                    exc_info=(type(result), result, result.__traceback__),
                )

    async def check_worker_health(self, worker: Worker) -> None:
        await self._check_worker_health(worker)

    async def start(self) -> None:
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._run_loop())

    async def stop(self) -> None:
        if self._task is None:
            return
        self._task.cancel()
        try:
            await self._task
        except asyncio.CancelledError:
            pass
        self._task = None

    async def _run_loop(self) -> None:
        while True:
            try:
                await self.check_all_workers_health()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("unexpected error in router health loop")
            interval = self._config.health_check_interval_secs
            jitter = random.uniform(0.8, 1.2)
            await asyncio.sleep(interval * jitter)

    async def _check_worker_health(self, worker: Worker) -> None:
        if worker.is_dead:
            return

        url = f"{worker.url}{self._config.health_check_endpoint}"
        try:
            response = await self._client.get(
                url,
                timeout=self._config.health_check_timeout_secs,
            )
        except httpx.HTTPError as exc:
            logger.debug(
                f"Worker {worker.display_id} health check failed: "
                f"{type(exc).__name__}: {exc}",
            )
            worker.record_health_result(
                ok=False,
                status_code=None,
                error=str(exc),
                failure_threshold=self._config.health_failure_threshold,
                success_threshold=self._config.health_success_threshold,
            )
            return

        ok = 200 <= response.status_code < 300
        error = None
        if not ok:
            error = response.text[:512] or f"status={response.status_code}"
            logger.debug(
                f"Worker {worker.display_id} health check returned "
                f"status_code={response.status_code}",
            )
        worker.record_health_result(
            ok=ok,
            status_code=response.status_code,
            error=error,
            failure_threshold=self._config.health_failure_threshold,
            success_threshold=self._config.health_success_threshold,
        )
