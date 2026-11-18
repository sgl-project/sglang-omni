# SPDX-License-Identifier: Apache-2.0
"""The Session protocol over a WebSocket on /v1/session."""

from __future__ import annotations

import asyncio
from collections.abc import Coroutine

import msgpack
import msgspec
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from sglang_omni.admission import QueueFullError
from sglang_omni.client.client import Client
from sglang_omni.proto.request import OmniRequest
from sglang_omni.proto.session import SessionIdentity, SessionLimits, TimedChunk

SESSION_WIRE_PATH = "/v1/session"
INTERNAL_ERROR_CLOSE_CODE = 1011


def pack(message: dict[str, object]) -> bytes:
    return msgpack.packb(message, use_bin_type=True)


def unpack(frame: bytes) -> dict[str, object]:
    message = msgpack.unpackb(frame, raw=False)
    if not isinstance(message, dict):
        raise ValueError("every session frame must be a msgpack map")
    else:
        return message


class SessionWire:
    """One WebSocket serving one Session through the Client session API, with the deployment's `limits`."""

    def __init__(
        self, client: Client, websocket: WebSocket, limits: SessionLimits
    ) -> None:
        self.client = client
        self.websocket = websocket
        self.limits = limits
        self.session_identity: SessionIdentity | None = None
        self.send_lock = asyncio.Lock()

    async def send(self, message: dict[str, object]) -> None:
        async with self.send_lock:
            await self.websocket.send_bytes(pack(message))

    async def receive(self) -> dict[str, object]:
        return unpack(await self.websocket.receive_bytes())

    async def run(self) -> None:
        try:
            await self.open(await self.receive())
            await self.serve()
        except WebSocketDisconnect:
            pass
        except Exception as exc:
            await self.send({"op": "error", "message": str(exc)})
            await self.websocket.close(code=INTERNAL_ERROR_CLOSE_CODE)
        finally:
            await asyncio.shield(self.close_session())

    async def open(self, message: dict[str, object]) -> None:
        if message.get("op") != "open":
            raise ValueError("the first session frame must be open")
        else:
            pass
        if "limits" in message:
            raise ValueError(
                "the open frame takes params only; session limits are the deployment's"
            )
        else:
            pass
        self.session_identity = await self.client.open_session(
            OmniRequest(None, params=dict(message.get("params", {}))),
            limits=self.limits,
        )
        await self.send({"op": "opened", "session": self.session_identity.to_dict()})

    async def serve(self) -> None:
        output_task = asyncio.create_task(self.forward_outputs())
        receive_task = asyncio.create_task(self.receive_operations())
        try:
            done, _ = await asyncio.wait(
                {output_task, receive_task}, return_when=asyncio.FIRST_COMPLETED
            )
            for task in done:
                task.result()
            if receive_task in done:
                await self.close_session()
                await output_task
                await self.send({"op": "closed"})
            else:
                raise RuntimeError("session output stream closed")
        finally:
            output_task.cancel()
            receive_task.cancel()
            await asyncio.gather(output_task, receive_task, return_exceptions=True)

    async def receive_operations(self) -> None:
        assert self.session_identity is not None
        while True:
            message = await self.receive()
            operation = message.get("op")
            if operation == "append":
                chunk = TimedChunk.from_dict(message.get("chunk"))
                await self.report(
                    self.client.append_session(self.session_identity, chunk),
                    chunk.seq,
                    "accepted",
                )
            elif operation == "control":
                event = TimedChunk.from_dict(message.get("chunk"))
                await self.report(
                    self.client.control_session(
                        self.session_identity,
                        event,
                        stages=msgspec.convert(
                            message.get("stages"), type=list[str] | None
                        ),
                        should_preempt=msgspec.convert(
                            message.get("should_preempt", False), type=bool
                        ),
                    ),
                    event.seq,
                    "controlled",
                )
            elif operation == "close":
                return
            elif operation == "open":
                raise ValueError("the session is already open")
            else:
                raise ValueError(f"unknown session operation {operation!r}")

    async def report(
        self, operation: Coroutine[None, None, object], seq: int, done: str
    ) -> None:
        """Run an append or a control and answer it with `done`; a full queue rejects it and keeps the session open."""
        try:
            await operation
        except Exception as exc:
            if QueueFullError.matches(exc):
                await self.send({"op": "rejected", "seq": seq, "reason": str(exc)})
                return
            else:
                raise
        await self.send({"op": done, "seq": seq})

    async def forward_outputs(self) -> None:
        assert self.session_identity is not None
        async for output in self.client.session_outputs(self.session_identity):
            await self.send({"op": "output", "chunk": output.to_dict()})

    async def close_session(self) -> None:
        session_identity = self.session_identity
        self.session_identity = None
        if session_identity is not None:
            await self.client.close_session(session_identity)
        else:
            pass


def register_session_wire(
    app: FastAPI, limits: SessionLimits = SessionLimits()
) -> None:
    """Mount the Session protocol WebSocket on /v1/session; every Session opens with the deployment's `limits`."""

    @app.websocket(SESSION_WIRE_PATH)
    async def session(websocket: WebSocket) -> None:
        await websocket.accept()
        await SessionWire(app.state.client, websocket, limits).run()
