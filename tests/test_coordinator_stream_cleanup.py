# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio

import pytest

from sglang_omni.pipeline.coordinator import Coordinator
from sglang_omni.proto import CompleteMessage, OmniRequest, StreamMessage


class _FakeControlPlane:
    def __init__(self) -> None:
        self.submissions: list[tuple[str, str, object]] = []
        self.abort_messages: list[object] = []

    async def submit_to_stage(self, stage_name: str, endpoint: str, msg: object) -> None:
        self.submissions.append((stage_name, endpoint, msg))

    async def broadcast_abort(self, msg: object) -> None:
        self.abort_messages.append(msg)


async def _wait_for_stream_registration(coordinator: Coordinator, request_id: str) -> None:
    for _ in range(100):
        if request_id in coordinator._stream_queues:
            return
        await asyncio.sleep(0)
    raise AssertionError(f"stream queue for {request_id} was not registered")


async def _wait_for_abort(control_plane: _FakeControlPlane, request_id: str) -> None:
    for _ in range(100):
        if [msg.request_id for msg in control_plane.abort_messages] == [request_id]:
            return
        await asyncio.sleep(0)
    raise AssertionError(f"abort for {request_id} was not observed")


def _make_coordinator() -> tuple[Coordinator, _FakeControlPlane]:
    coordinator = Coordinator(
        completion_endpoint="inproc://completion",
        abort_endpoint="inproc://abort",
        entry_stage="entry",
    )
    control_plane = _FakeControlPlane()
    coordinator.control_plane = control_plane
    coordinator.register_stage("entry", "inproc://entry")
    return coordinator, control_plane


@pytest.mark.asyncio
async def test_stream_abort_on_early_consumer_exit() -> None:
    coordinator, control_plane = _make_coordinator()
    request_id = "req-early-exit"
    received: list[object] = []

    async def _consume_one() -> None:
        async for msg in coordinator.stream(request_id, OmniRequest(inputs={"text": "hi"})):
            received.append(msg)
            break

    task = asyncio.create_task(_consume_one())
    await _wait_for_stream_registration(coordinator, request_id)
    await coordinator._handle_stream(
        StreamMessage(
            request_id=request_id,
            from_stage="decode",
            chunk={"text": "hello"},
            modality="text",
        )
    )
    await task
    await _wait_for_abort(control_plane, request_id)

    assert len(received) == 1
    assert [msg.request_id for msg in control_plane.abort_messages] == [request_id]
    assert request_id not in coordinator._requests
    assert request_id not in coordinator._stream_queues
    assert request_id not in coordinator._completion_futures


@pytest.mark.asyncio
async def test_stream_does_not_abort_after_normal_completion() -> None:
    coordinator, control_plane = _make_coordinator()
    request_id = "req-complete"

    async def _consume_all() -> list[object]:
        return [
            msg
            async for msg in coordinator.stream(
                request_id,
                OmniRequest(inputs={"text": "hi"}),
            )
        ]

    task = asyncio.create_task(_consume_all())
    await _wait_for_stream_registration(coordinator, request_id)
    await coordinator._handle_stream(
        StreamMessage(
            request_id=request_id,
            from_stage="decode",
            chunk={"text": "hello"},
            modality="text",
        )
    )
    await coordinator._handle_completion(
        CompleteMessage(
            request_id=request_id,
            from_stage="decode",
            success=True,
            result={"text": "hello"},
        )
    )

    received = await task

    assert len(received) == 2
    assert control_plane.abort_messages == []
    assert request_id not in coordinator._requests
    assert request_id not in coordinator._stream_queues
    assert request_id not in coordinator._completion_futures