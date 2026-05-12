# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio

import pytest

from sglang_omni_v1.pipeline.coordinator import Coordinator
from sglang_omni_v1.proto import CompleteMessage
from tests.unit_test.fixtures.pipeline_fakes import RecordingCoordinatorControlPlane


def test_coordinator_multi_terminal_completion_and_abort_contracts() -> None:
    """Preserves multi-terminal completion and abort cancellation semantics."""

    async def _run() -> None:
        coordinator = Coordinator(
            "inproc://complete",
            "inproc://abort",
            entry_stage="preprocess",
            terminal_stages=["decode", "code2wav"],
        )
        control_plane = RecordingCoordinatorControlPlane()
        coordinator.control_plane = control_plane
        coordinator.register_stage("preprocess", "inproc://preprocess")

        await coordinator._submit_request("req-1", {"text": "hello"})
        await coordinator._handle_completion(
            CompleteMessage("req-1", "decode", True, result={"text": "hi"})
        )
        assert not coordinator._completion_futures["req-1"].done()
        await coordinator._handle_completion(
            CompleteMessage("req-1", "code2wav", True, result={"audio": "ok"})
        )
        assert coordinator._completion_futures["req-1"].result() == {
            "decode": {"text": "hi"},
            "code2wav": {"audio": "ok"},
        }

        await coordinator._submit_request("req-2", "hello")
        future = coordinator._completion_futures["req-2"]
        assert await coordinator.abort("req-2") is True
        assert control_plane.aborts[0].request_id == "req-2"
        with pytest.raises(asyncio.CancelledError):
            await future

    asyncio.run(_run())


def test_coordinator_failure_completion_fails_fast_and_cleans_state() -> None:
    """Preserves fail-fast behavior and cleanup after any terminal failure."""

    async def _run() -> None:
        coordinator = Coordinator(
            "inproc://complete",
            "inproc://abort",
            entry_stage="preprocess",
            terminal_stages=["decode", "code2wav"],
        )
        coordinator.control_plane = RecordingCoordinatorControlPlane()
        coordinator.register_stage("preprocess", "inproc://preprocess")

        await coordinator._submit_request("req-1", "hello")
        future = coordinator._completion_futures["req-1"]
        await coordinator._handle_completion(
            CompleteMessage("req-1", "decode", True, result={"text": "hi"})
        )
        assert coordinator._partial_results["req-1"] == {"decode": {"text": "hi"}}

        await coordinator._handle_completion(
            CompleteMessage("req-1", "code2wav", False, error="boom")
        )

        with pytest.raises(RuntimeError, match="boom"):
            await future
        assert "req-1" not in coordinator._requests
        assert "req-1" not in coordinator._partial_results

    asyncio.run(_run())
