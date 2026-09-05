# SPDX-License-Identifier: Apache-2.0
"""Client adapter for bidirectional realtime coordinator requests."""

from __future__ import annotations

import uuid
from typing import Any, AsyncIterator

from sglang_omni.client.client import Client
from sglang_omni.client.types import GenerateChunk, GenerateRequest
from sglang_omni.pipeline.realtime_coordinator import RealtimeStream
from sglang_omni.proto import StreamMessage


class RealtimeHandle(AsyncIterator[GenerateChunk]):
    def __init__(self, client: Client, stream: RealtimeStream) -> None:
        self._client = client
        self._stream = stream

    @property
    def request_id(self) -> str:
        return self._stream.request_id

    @property
    def session_id(self) -> str:
        return self._stream.session_id

    @property
    def turn_id(self) -> str:
        return self._stream.turn_id

    @property
    def input_stage(self) -> str:
        return self._stream.input_stage

    def __aiter__(self) -> "RealtimeHandle":
        return self

    async def __anext__(self) -> GenerateChunk:
        message = await self._stream.__anext__()
        if isinstance(message, StreamMessage):
            return self._client._stream_builder(self.request_id, message)
        return self._client._result_builder(self.request_id, message.result)

    async def send_input(self, message: Any) -> None:
        await self._stream.send_input(message)

    async def aclose(self) -> None:
        await self._stream.aclose()


async def open_realtime(
    client: Client,
    request: GenerateRequest,
    request_id: str | None = None,
    *,
    session_id: str,
    turn_id: str,
    input_stage: str,
) -> RealtimeHandle:
    if not request.stream:
        raise ValueError("open_realtime requires request.stream=True")
    direct_open = getattr(client, "open_realtime", None)
    if callable(direct_open):
        return await direct_open(
            request,
            request_id=request_id,
            session_id=session_id,
            turn_id=turn_id,
            input_stage=input_stage,
        )

    coordinator = client._coordinator
    open_request = getattr(coordinator, "open_realtime", None)
    if not callable(open_request):
        raise TypeError("client coordinator does not support realtime requests")

    stream = await open_request(
        request_id or str(uuid.uuid4()),
        client._build_omni_request(request),
        session_id=session_id,
        turn_id=turn_id,
        input_stage=input_stage,
    )
    return RealtimeHandle(client, stream)
