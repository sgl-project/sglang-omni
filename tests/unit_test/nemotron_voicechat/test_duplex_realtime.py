# SPDX-License-Identifier: Apache-2.0
"""Exercise model-specific framing/events through the mounted shared WebSocket."""

import asyncio
from dataclasses import replace

import pytest
import websockets

from sglang_omni.models.nemotron_voicechat.realtime import deployment
from sglang_omni.proto.session import OutputChunk, SessionRef
from tests.unit_test.fixtures.realtime_websocket import (
    append,
    endpoint,
    recv,
    send,
    until,
)


class Client:
    def __init__(self):
        self.chunks = []
        self.closed = 0
        self.queue = asyncio.Queue()
        self.ref = None

    async def open_session(self, request, *, stages, limits, session_id):
        assert stages == ["perception", "thinker", "talker", "code2wav"]
        self.ref = SessionRef(session_id)
        return self.ref

    async def append_session(self, ref, chunk):
        assert ref == self.ref
        self.chunks.append(chunk)
        data = {
            "pcm": b"\0\0" * (1764 if chunk.payload else 256),
            "text": "hello",
            "eos": chunk.eos,
        }
        out = OutputChunk(
            ref,
            chunk.seq,
            chunk.seq,
            "audio",
            chunk.t_start_ms,
            chunk.duration_ms,
            data,
            "voicechat",
            chunk.eos,
        )
        await self.queue.put(out)
        await self.queue.put(replace(out, payload=None, kind="input_done"))

    async def session_outputs(self, ref):
        while True:
            out = await self.queue.get()
            if out is None:
                break
            yield out

    async def abort_session(self, ref):
        self.ref = replace(ref, epoch=ref.epoch + 1)
        return self.ref

    async def close_session(self, ref):
        self.closed += 1
        await self.queue.put(None)


@pytest.mark.asyncio
async def test_native_websocket_partial_tail_cancel_continuation_and_close():
    client = Client()
    async with endpoint(deployment=deployment(client)) as (_, url, _, _app):
        async with websockets.connect(url) as ws:
            await recv(ws)
            await send(ws, "session.update", session={"output_modalities": ["audio"]})
            update, _ = await until(ws, "session.updated")
            assert update["session"]["audio"]["output"]["format"]["rate"] == 22050
            await append(ws, 0, 640)
            await until(ws, "sglang.input_audio.accepted")
            assert client.chunks == []
            await append(ws, 1, 640, start=40)
            _, first = await until(ws, "sglang.unit.done")
            assert len(client.chunks) == 1 and len(client.chunks[0].payload) == 2560
            assert any(e["type"] == "response.output_audio.delta" for e in first)
            await send(ws, "response.cancel")
            _, cancelled = await until(ws, "sglang.response.cancelled")
            assert any(
                e["type"] == "response.done" and e["response"]["status"] == "cancelled"
                for e in cancelled
            )
            await append(ws, 2, 100, start=80)
            await send(ws, "sglang.input_audio.end")
            drained, events = await until(ws, "sglang.input_audio.drained")
            assert drained["consumed_ms"] == 86.25
            assert drained["padding_ms"] == 73.75
            assert client.chunks[-1].eos and len(client.chunks[-1].payload) == 2560
            assert client.chunks[-1].duration_ms == 6.25
            assert any(e["type"] == "response.created" for e in events)
            await send(ws, "session.close")
            await until(ws, "session.closed")
        assert client.closed == 1


@pytest.mark.asyncio
async def test_native_websocket_disconnect_releases_session():
    client = Client()
    async with endpoint(deployment=deployment(client)) as (_, url, _, _app):
        async with websockets.connect(url) as ws:
            await recv(ws)
            await send(ws, "session.update", session={})
            await until(ws, "session.updated")
        async with asyncio.timeout(5):
            while client.closed == 0:
                await asyncio.sleep(0.01)
        assert client.closed == 1
