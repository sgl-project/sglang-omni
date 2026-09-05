# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import asyncio

import pytest
import torch

from sglang_omni.client import Client, ExternalInputStream
from sglang_omni.client.types import GenerateRequest
from sglang_omni.proto import CompleteMessage, StreamMessage


class _FakeCoordinator:
    def __init__(self) -> None:
        self.started = []
        self.chunks = []
        self.finished = []
        self.closed = []

    async def start_input_stream(self, request_id, request):
        self.started.append((request_id, request))

        async def events():
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

        return events()

    async def send_input_chunk(self, request_id, data, *, metadata=None):
        self.chunks.append((request_id, data, metadata))
        return len(self.chunks) - 1

    async def finish_input_stream(self, request_id):
        self.finished.append(request_id)

    async def close_input_stream(self, request_id):
        self.closed.append(request_id)
        return True


def test_client_external_input_stream_handle_lifecycle() -> None:
    async def run() -> None:
        coordinator = _FakeCoordinator()
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

    asyncio.run(run())


def test_client_iterator_aclose_aborts_unfinished_request_once() -> None:
    async def run() -> None:
        coordinator = _FakeCoordinator()
        stream = await Client(coordinator).start_input_stream(
            GenerateRequest(prompt="", stream=True), request_id="req"
        )
        await stream.aclose()
        await stream.aclose()
        assert coordinator.closed == ["req"]
        with pytest.raises(RuntimeError, match="closed"):
            await stream.send(torch.tensor([1], dtype=torch.int16))

    asyncio.run(run())


def test_client_context_manager_aborts_on_exception() -> None:
    async def run() -> None:
        coordinator = _FakeCoordinator()
        stream = await Client(coordinator).start_input_stream(
            GenerateRequest(prompt="", stream=True), request_id="req"
        )
        with pytest.raises(RuntimeError, match="boom"):
            async with stream:
                raise RuntimeError("boom")
        assert coordinator.closed == ["req"]

    asyncio.run(run())
