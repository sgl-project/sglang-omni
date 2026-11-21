# SPDX-License-Identifier: Apache-2.0
"""Check what D1 asks besides memory: the same output, and a server that survives cancels.

An offload change may move weights, but it must not move the answer.  Two
checks run against a served endpoint:

- correctness: the same request list produces byte-identical audio on the
  baseline and on the change;
- safety: cancelling part of a concurrent burst leaves the finished requests
  correct and the server still serving.

Both are transport-level, so the same code runs against a mock server today
and against a real deployment when one is available.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Literal

import httpx

AUDIO_PATH = "/v1/audio/speech"
HEALTH_PATH = "/health"


@dataclass(frozen=True, kw_only=True)
class SafetyPlan:
    """How hard to push one server: how many requests, how many at once, how many to cancel."""

    requests: int
    concurrency: int
    cancelled_requests: int = 0
    cancel_after_s: float = 0.0

    def __post_init__(self) -> None:
        if self.requests < 1:
            raise ValueError("a plan needs at least one request")
        elif self.concurrency < 1 or self.concurrency > self.requests:
            raise ValueError("concurrency must be between 1 and the request count")
        elif not 0 <= self.cancelled_requests <= self.requests:
            raise ValueError("cancelled_requests must be within the request count")
        elif self.cancelled_requests and self.cancel_after_s <= 0:
            raise ValueError("a cancel needs a positive delay")
        else:
            pass


@dataclass(kw_only=True)
class SafetyReport:
    """What the burst did: who finished, who was cancelled, and whether the server still answers."""

    planned: int
    completed: int
    cancelled: int
    server_alive: bool
    mismatched: list[str] = field(default_factory=list)

    @property
    def is_safe(self) -> bool:
        """Every planned request is accounted for, nothing mismatched, and the server survived."""
        if self.completed + self.cancelled != self.planned:
            return False
        elif self.mismatched:
            return False
        else:
            return self.server_alive


def output_mismatches(
    baseline: dict[str, bytes], candidate: dict[str, bytes]
) -> list[str]:
    """Request ids whose audio differs, or that only one side produced."""
    names = sorted(set(baseline) | set(candidate))
    mismatched: list[str] = []
    for name in names:
        if baseline.get(name) != candidate.get(name):
            mismatched.append(name)
        else:
            pass
    return mismatched


async def speak(
    client: httpx.AsyncClient,
    url: str,
    body: dict[str, object],
    *,
    cancel_after_s: float | None,
) -> bytes | Literal["cancelled"]:
    """One streaming request; with a cancel delay it stops reading part-way through."""
    pcm = bytearray()
    try:
        async with client.stream("POST", url, json=body) as response:
            response.raise_for_status()
            if cancel_after_s is None:
                async for data in response.aiter_bytes():
                    pcm.extend(data)
            else:
                iterator = response.aiter_bytes()
                deadline = asyncio.get_running_loop().time() + cancel_after_s
                while True:
                    remaining = deadline - asyncio.get_running_loop().time()
                    if remaining <= 0:
                        return "cancelled"
                    else:
                        pass
                    try:
                        data = await asyncio.wait_for(
                            anext(iterator), timeout=remaining
                        )
                    except (TimeoutError, StopAsyncIteration):
                        return "cancelled"
                    pcm.extend(data)
    except httpx.HTTPError:
        if cancel_after_s is None:
            raise
        else:
            return "cancelled"
    return bytes(pcm)


async def run_burst(
    client: httpx.AsyncClient,
    url: str,
    bodies: list[dict[str, object]],
    plan: SafetyPlan,
) -> dict[str, bytes | Literal["cancelled"]]:
    """Send every request, cancelling the tail of them after the plan's delay."""
    semaphore = asyncio.Semaphore(plan.concurrency)
    cancelled_from = plan.requests - plan.cancelled_requests

    async def one(index: int) -> tuple[int, bytes | Literal["cancelled"]]:
        async with semaphore:
            delay = plan.cancel_after_s if index >= cancelled_from else None
            return index, await speak(client, url, bodies[index], cancel_after_s=delay)

    outcomes = await asyncio.gather(*(one(index) for index in range(plan.requests)))
    return {f"req-{index}": outcome for index, outcome in outcomes}


async def check_server(
    client: httpx.AsyncClient,
    url: str,
    bodies: list[dict[str, object]],
    plan: SafetyPlan,
    *,
    baseline: dict[str, bytes] | None = None,
) -> SafetyReport:
    """Run the burst and answer D1's safety question for one deployment."""
    outcomes = await run_burst(client, url, bodies, plan)
    completed = {
        name: value for name, value in outcomes.items() if isinstance(value, bytes)
    }
    cancelled = sum(1 for value in outcomes.values() if value == "cancelled")
    health = await client.get(url.removesuffix(AUDIO_PATH) + HEALTH_PATH)
    mismatched = output_mismatches(baseline, completed) if baseline is not None else []
    return SafetyReport(
        planned=len(bodies),
        completed=len(completed),
        cancelled=cancelled,
        server_alive=health.status_code == 200,
        mismatched=mismatched,
    )
