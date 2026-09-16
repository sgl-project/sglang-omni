# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import asyncio
import gc

from sglang_omni.serve import launcher
from sglang_omni.utils.gc_control import (
    FREEZE_GC_AFTER_REQUESTS_ENV,
    FREEZE_GC_AFTER_STARTUP_ENV,
)


class _FakeClient:
    def __init__(self) -> None:
        self.calls: list[float] = []

    async def freeze_gc(self, *, timeout_s: float = 30.0) -> dict:
        self.calls.append(timeout_s)
        return {"success": True}


class _FakeCoordinator:
    completed_requests = 0


def test_two_phase_freeze_waits_for_completed_requests(monkeypatch) -> None:
    monkeypatch.delenv(FREEZE_GC_AFTER_STARTUP_ENV, raising=False)
    monkeypatch.setenv(FREEZE_GC_AFTER_REQUESTS_ENV, "3")
    monkeypatch.setattr(launcher, "_WARMUP_FREEZE_POLL_S", 0.01)

    async def _run() -> None:
        client, coordinator = _FakeClient(), _FakeCoordinator()
        task = await launcher._freeze_gc_after_startup(client, coordinator)
        assert len(client.calls) == 1  # startup freeze is immediate
        assert task is not None and not task.done()
        await asyncio.sleep(0.05)
        assert len(client.calls) == 1  # still waiting for requests
        coordinator.completed_requests = 3
        await asyncio.wait_for(task, timeout=2.0)
        assert len(client.calls) == 2

    try:
        asyncio.run(_run())
    finally:
        gc.unfreeze()


def test_second_freeze_disabled_by_zero(monkeypatch) -> None:
    monkeypatch.delenv(FREEZE_GC_AFTER_STARTUP_ENV, raising=False)
    monkeypatch.setenv(FREEZE_GC_AFTER_REQUESTS_ENV, "0")

    async def _run() -> None:
        client = _FakeClient()
        task = await launcher._freeze_gc_after_startup(client, _FakeCoordinator())
        assert task is None
        assert len(client.calls) == 1

    try:
        asyncio.run(_run())
    finally:
        gc.unfreeze()


def test_startup_freeze_opt_out(monkeypatch) -> None:
    monkeypatch.setenv(FREEZE_GC_AFTER_STARTUP_ENV, "0")

    async def _run() -> None:
        client = _FakeClient()
        assert (
            await launcher._freeze_gc_after_startup(client, _FakeCoordinator()) is None
        )
        assert client.calls == []

    asyncio.run(_run())
