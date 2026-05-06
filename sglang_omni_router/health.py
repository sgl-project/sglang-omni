from __future__ import annotations

import asyncio
import logging
import random
from collections.abc import Sequence

import httpx

from sglang_omni_router.config import RouterConfig
from sglang_omni_router.worker import Worker

logger = logging.getLogger(__name__)


class HealthChecker:
    def __init__(
        self,
        *,
        workers: Sequence[Worker],
        config: RouterConfig,
        client: httpx.AsyncClient,
    ) -> None:
        self._workers = list(workers)
        self._config = config
        self._client = client
        self._task: asyncio.Task[None] | None = None

    async def check_once(self) -> None:
        await asyncio.gather(
            *(self._check_worker(worker) for worker in self._workers),
            return_exceptions=False,
        )

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
            interval = self._config.health_check_interval_secs
            jitter = random.uniform(0.8, 1.2)
            await asyncio.sleep(interval * jitter)
            try:
                await self.check_once()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("unexpected error in router health loop")

    async def _check_worker(self, worker: Worker) -> None:
        url = f"{worker.url}{self._config.health_check_endpoint}"
        try:
            response = await self._client.get(
                url,
                timeout=self._config.health_check_timeout_secs,
            )
            ok = 200 <= response.status_code < 300
            worker.record_health_result(
                ok=ok,
                status_code=response.status_code,
                error=None if ok else response.text[:512],
                failure_threshold=self._config.health_failure_threshold,
                success_threshold=self._config.health_success_threshold,
            )
        except Exception as exc:
            worker.record_health_result(
                ok=False,
                status_code=None,
                error=str(exc),
                failure_threshold=self._config.health_failure_threshold,
                success_threshold=self._config.health_success_threshold,
            )
