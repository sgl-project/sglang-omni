# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import asyncio

from sglang_omni.pipeline.coordinator import Coordinator
from tests.unit_test.fixtures.pipeline_fakes import RecordingCoordinatorControlPlane


def _make_coordinator(admission_min_gap_ms: float) -> Coordinator:
    coordinator = Coordinator(
        "inproc://complete",
        "inproc://abort",
        entry_stage="preprocess",
        terminal_stages=["decode"],
        admission_min_gap_ms=admission_min_gap_ms,
    )
    coordinator.control_plane = RecordingCoordinatorControlPlane()
    coordinator.register_stage("preprocess", "inproc://preprocess")
    return coordinator


def test_admission_gate_default_off_admits_immediately() -> None:
    async def _run() -> None:
        coordinator = _make_coordinator(0.0)
        loop = asyncio.get_running_loop()
        start = loop.time()
        await asyncio.gather(
            *(coordinator._submit_request(f"req-{i}", {"text": "hi"}) for i in range(8))
        )
        assert loop.time() - start < 0.05
        assert len(coordinator.control_plane.submitted) == 8

    asyncio.run(_run())


def test_admission_gate_reserves_min_gap_slots() -> None:
    async def _run() -> None:
        coordinator = _make_coordinator(admission_min_gap_ms=50.0)
        loop = asyncio.get_running_loop()
        start = loop.time()
        await asyncio.gather(
            *(coordinator._submit_request(f"req-{i}", {"text": "hi"}) for i in range(4))
        )
        elapsed = loop.time() - start
        assert elapsed >= 0.15
        assert len(coordinator.control_plane.submitted) == 4
        assert coordinator._next_admission_at >= start + 4 * 0.05

    asyncio.run(_run())


def test_admission_gate_no_wait_for_spaced_arrivals() -> None:
    async def _run() -> None:
        coordinator = _make_coordinator(admission_min_gap_ms=10.0)
        loop = asyncio.get_running_loop()
        await coordinator._submit_request("req-0", {"text": "hi"})
        await asyncio.sleep(0.03)
        start = loop.time()
        await coordinator._submit_request("req-1", {"text": "hi"})
        assert loop.time() - start < 0.01

    asyncio.run(_run())
