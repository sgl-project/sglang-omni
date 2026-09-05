# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
from typing import Any

from sglang_omni.client import Client, GenerateRequest
from sglang_omni.client.realtime import open_realtime
from sglang_omni.pipeline.realtime_coordinator import RealtimeCoordinator
from sglang_omni.proto import CompleteMessage, InputUpdateMessage
from tests.unit_test.fixtures.pipeline_fakes import FakeScheduler
from tests.unit_test.pipeline.helpers import make_stage


class _InProcessCoordinatorControlPlane:
    def __init__(self, stage: Any) -> None:
        self.stage = stage
        self.submitted: list[tuple[str, str, Any]] = []
        self.aborts: list[Any] = []

    async def submit_to_stage(self, stage: str, endpoint: str, message: Any) -> None:
        self.submitted.append((stage, endpoint, message))
        await self.stage._handle_message(message)

    async def broadcast_abort(self, message: Any) -> None:
        self.aborts.append(message)


def test_client_realtime_update_reaches_stage_scheduler_inbox() -> None:
    async def _run() -> None:
        scheduler = FakeScheduler()
        stage = make_stage(
            name="tts_engine",
            scheduler=scheduler,
            can_accept_stream_before_payload=True,
        )
        coordinator = RealtimeCoordinator(
            completion_endpoint="inproc://complete",
            abort_endpoint="inproc://abort",
            entry_stage="tts_engine",
            terminal_stages=["tts_engine"],
        )
        control_plane = _InProcessCoordinatorControlPlane(stage)
        coordinator.control_plane = control_plane
        coordinator.register_stage("tts_engine", "inproc://tts_engine")
        client = Client(coordinator)

        handle = await open_realtime(
            client,
            GenerateRequest(prompt={"text": ""}, stream=True),
            request_id="request-1",
            session_id="session-1",
            turn_id="turn-1",
            input_stage="tts_engine",
        )

        submitted = scheduler.inbox.get_nowait()
        assert submitted.request_id == "request-1"
        assert submitted.type == "new_request"

        update = InputUpdateMessage(
            request_id="request-1",
            session_id="session-1",
            turn_id="turn-1",
            seq_no=0,
            token_ids=(7, 8),
            byte_count=4,
        )
        await handle.send_input(update)

        routed = scheduler.inbox.get_nowait()
        assert routed.request_id == "request-1"
        assert routed.type == "input_update"
        assert routed.data is update

        await coordinator._handle_completion(
            CompleteMessage(
                request_id="request-1",
                from_stage="tts_engine",
                success=True,
                result={"ok": True},
            )
        )
        chunks = [chunk async for chunk in handle]
        assert len(chunks) == 1
        assert chunks[0].request_id == "request-1"
        assert coordinator._realtime_requests == {}

    asyncio.run(_run())
