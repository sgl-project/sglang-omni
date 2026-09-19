# SPDX-License-Identifier: Apache-2.0
"""BenchmarkRunner: warmup + concurrent dispatch with semaphore and rate limiting."""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Any, Callable, Coroutine

import aiohttp
import numpy as np
from tqdm.asyncio import tqdm

from benchmarks.benchmarker.data import RequestResult

logger = logging.getLogger(__name__)

SendFn = Callable[[aiohttp.ClientSession, Any], Coroutine[Any, Any, RequestResult]]


def resolve_warmup(warmup: int | None, max_concurrency: int) -> int:
    # note (luojiaxuan): warmup=None means match the configured concurrency, so
    # the timed cohort is not the first to absorb concurrency-shaped cold work.
    # An explicit count always wins, including 0 to disable warmup entirely.
    if warmup is not None:
        return warmup
    return max_concurrency if max_concurrency > 0 else 1


@dataclass
class RunConfig:
    max_concurrency: int = 1
    request_rate: float = float("inf")
    warmup: int | None = None
    disable_tqdm: bool = False
    timeout_s: int = 300
    arrival_seed: int | None = None

    @property
    def effective_warmup(self) -> int:
        return resolve_warmup(self.warmup, self.max_concurrency)


class BenchmarkRunner:
    """Support concurrent requests sending in a single benchmark run.

    Note (chenyang):
    max_concurrency is default to 1, thus all the requests are runs sequentially.

    TODO (chenyang):
    Current concurrency implementation of models are not fully supported.
    https://github.com/sgl-project/sglang-omni/issues/229
    https://github.com/sgl-project/sglang-omni/issues/228
    """

    def __init__(
        self,
        config: RunConfig,
        *,
        on_result: Callable[[RequestResult], None] | None = None,
    ) -> None:
        self.config = config
        self.on_result = on_result
        self.wall_clock_s: float = 0.0

    async def run(
        self, samples: list, send_fn: SendFn, *, warmup_sample: Any | None = None
    ) -> list[RequestResult]:
        timeout = aiohttp.ClientTimeout(total=self.config.timeout_s)
        # note (guozhihao): Closed-loop runs are bounded by max_concurrency.
        # Open-loop (max_concurrency=0) must not inherit aiohttp's default
        # 100-conn cap, or sustained overshoot silently queues on the client.
        connector = (
            aiohttp.TCPConnector(limit=0) if not self.config.max_concurrency else None
        )
        async with aiohttp.ClientSession(
            timeout=timeout, connector=connector
        ) as session:
            if self.config.effective_warmup > 0:
                # Note (Jiaxin Deng): a separate input avoids warming a timed
                # request's audio/prefix cache while retaining the warmup count.
                await self._warmup(
                    session,
                    samples if warmup_sample is None else [warmup_sample],
                    send_fn,
                )

            logger.info(
                "Benchmarking %d requests (max_concurrency=%s)...",
                len(samples),
                self.config.max_concurrency,
            )
            t0 = time.perf_counter()
            results = await self._dispatch(session, samples, send_fn)
            self.wall_clock_s = time.perf_counter() - t0
        return results

    async def _warmup(
        self,
        session: aiohttp.ClientSession,
        samples: list,
        send_fn: SendFn,
    ) -> None:
        count = self.config.effective_warmup if samples else 0
        logger.info("Warmup (%d requests)...", count)
        semaphore = (
            asyncio.Semaphore(self.config.max_concurrency)
            if self.config.max_concurrency
            else None
        )

        async def _limited(sample: Any) -> RequestResult:
            if semaphore is None:
                return await send_fn(session, sample)
            async with semaphore:
                return await send_fn(session, sample)

        # note (luojiaxuan): By default the measured cohort reuses this sample list,
        # so warming distinct samples would pre-fill per-sample server caches,
        # such as the MOSS-TTS reference-audio cache, for requests that are
        # about to be timed. Repeat one sample to get the concurrency shape
        # without widening that bias as concurrency grows.
        results = await asyncio.gather(*(_limited(samples[0]) for _ in range(count)))
        for i, result in enumerate(results):
            status = "ok" if result.is_success else result.error
            logger.info("  warmup %d/%d: %s", i + 1, count, status)
            if not result.is_success:
                raise ValueError(
                    "Warmup failed - Please make sure benchmark arguments are "
                    f"correctly specified. Error: {result.error}"
                )

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
        exponential = (
            np.random.exponential
            if self.config.arrival_seed is None
            else np.random.default_rng(self.config.arrival_seed).exponential
        )

        async def _limited(sample: Any, scheduled_s: float) -> RequestResult:
            if semaphore:
                async with semaphore:
                    dispatched_s = time.perf_counter()
                    result = await send_fn(session, sample)
                    completed_s = time.perf_counter()
            else:
                dispatched_s = time.perf_counter()
                result = await send_fn(session, sample)
                completed_s = time.perf_counter()
            result.scheduled_s = scheduled_s
            result.dispatched_s = dispatched_s
            result.completed_s = completed_s
            if self.on_result is not None:
                self.on_result(result)
            pbar.update(1)
            return result

        try:
            tasks: list[asyncio.Task] = []
            scheduled_s = time.perf_counter()
            for sample in samples:
                if self.config.request_rate != float("inf"):
                    interval = exponential(1.0 / self.config.request_rate)
                    # Note (Jiaxin Deng): preserve the offered arrival schedule
                    # when dispatch lags; relative sleeps hide client overload.
                    scheduled_s += interval
                    while (remaining := scheduled_s - time.perf_counter()) > 0:
                        await asyncio.sleep(remaining)
                else:
                    scheduled_s = time.perf_counter()
                tasks.append(asyncio.create_task(_limited(sample, scheduled_s)))

            results: list[RequestResult] = list(await asyncio.gather(*tasks))
        finally:
            # Note (Jiaxin Deng): a failed sender must not leave requests using
            # the session after the caller starts shutting down its server.
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            pbar.close()
        return results
