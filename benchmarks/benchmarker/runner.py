# SPDX-License-Identifier: Apache-2.0
"""BenchmarkRunner: warmup + concurrent dispatch with semaphore and rate limiting."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Coroutine

import aiohttp
import numpy as np
from tqdm.asyncio import tqdm

from benchmarks.benchmarker.data import RequestResult

logger = logging.getLogger(__name__)

SendFn = Callable[
    [aiohttp.ClientSession, Any], Coroutine[Any, Any, RequestResult]
]


@dataclass
class RunConfig:
    """Configuration for a benchmark run."""

    max_concurrency: int = 1
    request_rate: float = float("inf")
    warmup: int = 1
    disable_tqdm: bool = False
    timeout_s: int = 300


class BenchmarkRunner:
    """Orchestrates warmup and concurrent request dispatch.

    The caller provides a *send_fn(session, sample) -> RequestResult* coroutine
    that encapsulates all HTTP/response handling.  The runner only manages
    concurrency, warmup, rate-limiting, and progress display.
    """

    def __init__(self, config: RunConfig) -> None:
        self.config = config

    async def run(
        self, samples: list, send_fn: SendFn
    ) -> list[RequestResult]:
        """Run the full benchmark: warmup then timed dispatch."""
        timeout = aiohttp.ClientTimeout(total=self.config.timeout_s)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            if self.config.warmup > 0:
                await self._warmup(session, samples, send_fn)

            logger.info(
                "Benchmarking %d requests (max_concurrency=%s)...",
                len(samples),
                self.config.max_concurrency,
            )
            results = await self._dispatch(session, samples, send_fn)
        return results

    async def _warmup(
        self,
        session: aiohttp.ClientSession,
        samples: list,
        send_fn: SendFn,
    ) -> None:
        count = min(self.config.warmup, len(samples))
        logger.info("Warmup (%d requests)...", count)
        for i in range(count):
            result = await send_fn(session, samples[i])
            status = "ok" if result.is_success else result.error
            logger.info("  warmup %d/%d: %s", i + 1, count, status)

    async def _dispatch(
        self,
        session: aiohttp.ClientSession,
        samples: list,
        send_fn: SendFn,
    ) -> list[RequestResult]:
        semaphore = (
            asyncio.Semaphore(self.config.max_concurrency)
            if self.config.max_concurrency
            else None
        )
        pbar = tqdm(total=len(samples), disable=self.config.disable_tqdm)

        async def _limited(sample: Any) -> RequestResult:
            if semaphore:
                async with semaphore:
                    result = await send_fn(session, sample)
            else:
                result = await send_fn(session, sample)
            pbar.update(1)
            return result

        tasks: list[asyncio.Task] = []
        for sample in samples:
            if self.config.request_rate != float("inf"):
                interval = np.random.exponential(1.0 / self.config.request_rate)
                await asyncio.sleep(interval)
            tasks.append(asyncio.create_task(_limited(sample)))

        results: list[RequestResult] = list(await asyncio.gather(*tasks))
        pbar.close()
        return results
