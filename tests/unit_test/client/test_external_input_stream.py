# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import asyncio
from typing import AsyncGenerator, Literal

import pytest
import torch

from sglang_omni.client import Client
from sglang_omni.client.client import ExternalInputStream
from sglang_omni.client.types import GenerateRequest
from sglang_omni.proto import CompleteMessage, OmniRequest, StreamMessage


class FakeCoordinator:
    def __init__(self) -> None:
        self.started: list[tuple[str, OmniRequest]] = []
        self.chunks: list[tuple[str, torch.Tensor, dict[str, object] | None]] = []
        self.finished: list[str] = []
        self.closed: list[str] = []
        self.finalized_events: list[str] = []

    async def start_input_stream(
        self, request_id: str, request: OmniRequest
    ) -> AsyncGenerator[CompleteMessage | StreamMessage, None]:
        self.started.append((request_id, request))

        async def _events() -> AsyncGenerator[CompleteMessage | StreamMessage, None]:
            try:
                yield StreamMessage(
                    request_id=request_id,
                    from_stage="asr",
                    chunk={"text": "hel", "modality": "text"},
                    modality="text",
                )
                yield CompleteMessage(
                    request_id=request_id,
                    from_stage="asr",
                    success=True,
                    result={"text": "hello", "finish_reason": "stop"},
                )
            finally:
                self.finalized_events.append(request_id)

        return _events()

    async def send_input_chunk(
        self,
        request_id: str,
        data: torch.Tensor,
        *,
        metadata: dict[str, object] | None = None,
    ) -> int:
        self.chunks.append((request_id, data, metadata))
        return len(self.chunks) - 1

    async def finish_input_stream(self, request_id: str) -> None:
        self.finished.append(request_id)

    async def close_input_stream(self, request_id: str) -> bool:
        self.closed.append(request_id)
        return True


def test_client_external_input_stream_handle_lifecycle() -> None:
    async def _run() -> None:
        coordinator = FakeCoordinator()
        client = Client(coordinator)
        stream = await client.start_input_stream(
            GenerateRequest(prompt="", stream=True), request_id="req"
        )
        assert isinstance(stream, ExternalInputStream)
        assert (
            await stream.send(
                torch.tensor([1, 2], dtype=torch.int16),
                metadata={"modality": "audio"},
            )
            == 0
        )
        await stream.finish()
        with pytest.raises(RuntimeError, match="already done"):
            await stream.send(torch.tensor([3], dtype=torch.int16))

        chunks = [chunk async for chunk in stream]
        assert chunks[0].text == "hel"
        assert chunks[-1].text == "hello"
        assert chunks[-1].finish_reason == "stop"
        assert coordinator.finished == ["req"]
        assert coordinator.closed == []

    asyncio.run(_run())


@pytest.mark.parametrize("close_method", ["aclose", "abort"])
def test_client_closing_stream_releases_request_and_events_once(
    close_method: Literal["aclose", "abort"],
) -> None:
    async def _run() -> None:
        coordinator = FakeCoordinator()
        stream = await Client(coordinator).start_input_stream(
            GenerateRequest(prompt="", stream=True), request_id="req"
        )
        await anext(stream)
        if close_method == "abort":
            await stream.abort()
        else:
            await stream.aclose()
        await stream.aclose()
        assert coordinator.closed == ["req"]
        assert coordinator.finalized_events == ["req"]
        with pytest.raises(RuntimeError, match="closed"):
            await stream.send(torch.tensor([1], dtype=torch.int16))

    asyncio.run(_run())


def test_client_context_manager_aborts_on_exception() -> None:
    async def _run() -> None:
        coordinator = FakeCoordinator()
        stream = await Client(coordinator).start_input_stream(
            GenerateRequest(prompt="", stream=True), request_id="req"
        )
        with pytest.raises(RuntimeError, match="boom"):
            async with stream:
                raise RuntimeError("boom")
        assert coordinator.closed == ["req"]

    asyncio.run(_run())
