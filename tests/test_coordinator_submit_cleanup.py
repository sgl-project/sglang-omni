# SPDX-License-Identifier: Apache-2.0
"""Tests for coordinator submit cleanup behavior."""

from __future__ import annotations

import pytest

from sglang_omni.pipeline.coordinator import Coordinator
from sglang_omni.proto import StageInfo


class _FailingControlPlane:
    async def submit_to_stage(self, *_args, **_kwargs) -> None:
        raise RuntimeError("submit failed")


@pytest.mark.asyncio
async def test_submit_cleans_completion_future_on_submit_error() -> None:
    coordinator = Coordinator(
        completion_endpoint="inproc://completion",
        abort_endpoint="inproc://abort",
        entry_stage="entry",
    )
    coordinator._stages["entry"] = StageInfo(
        name="entry",
        control_endpoint="inproc://entry",
    )
    coordinator.control_plane = _FailingControlPlane()

    with pytest.raises(RuntimeError, match="submit failed"):
        await coordinator.submit("req-1", {"text": "hello"})

    assert "req-1" not in coordinator._completion_futures


@pytest.mark.asyncio
async def test_submit_cleans_request_info_on_submit_error() -> None:
    coordinator = Coordinator(
        completion_endpoint="inproc://completion",
        abort_endpoint="inproc://abort",
        entry_stage="entry",
    )
    coordinator._stages["entry"] = StageInfo(
        name="entry",
        control_endpoint="inproc://entry",
    )
    coordinator.control_plane = _FailingControlPlane()

    with pytest.raises(RuntimeError, match="submit failed"):
        await coordinator.submit("req-2", {"text": "hello"})

    assert "req-2" not in coordinator._requests
