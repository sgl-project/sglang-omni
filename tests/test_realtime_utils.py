# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

from sglang_omni.realtime.utils import throttle


class _Recorder:
    def __init__(self) -> None:
        self.calls: list[tuple[str, float]] = []
        self._throttle_state: dict[str, float] = {}

    @throttle(0.5, timestamp_kw="timestamp")
    async def record(self, label: str, *, timestamp: float) -> None:
        self.calls.append((label, timestamp))


@pytest.mark.asyncio
async def test_throttle_decorator_suppresses_calls_within_interval():
    recorder = _Recorder()

    await recorder.record("first", timestamp=1.0)
    await recorder.record("suppressed", timestamp=1.2)
    await recorder.record("second", timestamp=1.6)

    assert recorder.calls == [("first", 1.0), ("second", 1.6)]


@pytest.mark.asyncio
async def test_throttle_decorator_is_per_instance():
    left = _Recorder()
    right = _Recorder()

    await left.record("left", timestamp=1.0)
    await right.record("right", timestamp=1.1)

    assert left.calls == [("left", 1.0)]
    assert right.calls == [("right", 1.1)]
