# SPDX-License-Identifier: Apache-2.0
"""Live PCM diarization over a bounded WebSocket session."""

from __future__ import annotations

import asyncio
import json
import logging
import uuid

import msgspec
from fastapi import WebSocket, WebSocketDisconnect

from sglang_omni.client import Client, GenerateRequest

logger = logging.getLogger(__name__)
_MAX_AUDIO_BYTES = 32000
_IDLE_SECONDS = 60


class DiarizationSession:
    def __init__(self, websocket: WebSocket, *, client: Client, model_name: str):
        self.websocket = websocket
        self.client = client
        self.model_name = model_name
        self.session_id = f"diarization-live-{uuid.uuid4()}"
        self.messages: asyncio.Queue[bytes | str] = asyncio.Queue(maxsize=8)
        self.end = 0.0

    async def run(self):
        receiver = asyncio.create_task(self.receive())
        worker = asyncio.create_task(self.process())
        try:
            done, _ = await asyncio.wait(
                {receiver, worker}, return_when=asyncio.FIRST_COMPLETED
            )
            for task in done:
                task.result()
        except WebSocketDisconnect:
            pass
        except Exception as exc:
            logger.warning("Live diarization %s failed: %s", self.session_id, exc)
            try:
                await self.websocket.send_json({"type": "error", "message": str(exc)})
            except (WebSocketDisconnect, RuntimeError):
                pass
        finally:
            receiver.cancel()
            worker.cancel()
            await asyncio.gather(receiver, worker, return_exceptions=True)
            try:
                # A separate close request releases persistent state even if the
                # active chunk was canceled while physical inference continued.
                await self.request("close")
            except Exception:
                logger.exception(
                    "Failed to close diarization session %s", self.session_id
                )
            try:
                await self.websocket.close()
            except (WebSocketDisconnect, RuntimeError):
                pass

    async def receive(self):
        while True:
            try:
                message = await asyncio.wait_for(
                    self.websocket.receive(), _IDLE_SECONDS
                )
            except TimeoutError:
                raise ValueError(
                    "Diarization session timed out waiting for audio"
                ) from None
            if message["type"] == "websocket.disconnect":
                raise WebSocketDisconnect()
            audio = message.get("bytes")
            if audio is not None:
                if not audio or len(audio) % 2 or len(audio) > _MAX_AUDIO_BYTES:
                    raise ValueError(
                        "Send 1–16000 samples of mono 16 kHz PCM16 per message"
                    )
                item = audio
            else:
                event = json.loads(message.get("text") or "null")
                if not isinstance(event, dict) or set(event) != {"type"}:
                    raise ValueError("Expected audio.end or session.reset")
                item = event["type"]
                if item not in ("audio.end", "session.reset"):
                    raise ValueError("Expected audio.end or session.reset")
            try:
                self.messages.put_nowait(item)
            except asyncio.QueueFull:
                raise ValueError(
                    "Audio queue is full; wait for audio.ack before sending more"
                ) from None

    async def request(self, operation: str, pcm: bytes = b""):
        request_id = f"{self.session_id}-{uuid.uuid4()}"
        generation = GenerateRequest(
            model=self.model_name,
            prompt={"session_id": self.session_id, "operation": operation, "pcm": pcm},
            stream=False,
            metadata={"task": "diarization_stream"},
        )
        completion = asyncio.create_task(
            self.client.completion(generation, request_id=request_id)
        )
        try:
            result = (
                await asyncio.shield(completion)
                if operation == "open"
                else await completion
            )
            if result.diarization is None:
                raise RuntimeError("Model returned no diarization result")
            return result.diarization
        except asyncio.CancelledError:
            if operation == "open":
                # Finish admission before close, so a canceled open cannot
                # create orphaned state after the close operation ran.
                await completion
            else:
                await self.client.abort(request_id)
            raise

    async def ready(self):
        await self.websocket.send_json(
            {
                "type": "session.ready",
                "session_id": self.session_id,
                "sample_rate": 16000,
                "format": "pcm16",
                "channels": 1,
                "max_audio_bytes": _MAX_AUDIO_BYTES,
            }
        )

    async def process(self):
        await self.request("open")
        await self.ready()
        while True:
            item = await self.messages.get()
            if item == "session.reset":
                await self.request("close")
                self.session_id = f"diarization-live-{uuid.uuid4()}"
                await self.request("open")
                self.end = 0.0
                await self.ready()
                continue
            final = item == "audio.end"
            result = await self.request(
                "finish" if final else "append", b"" if final else item
            )
            if result.duration > self.end:
                await self.websocket.send_json(
                    {
                        "type": "diarization.update",
                        "start": self.end,
                        "end": result.duration,
                        "segments": msgspec.to_builtins(result.segments),
                    }
                )
                self.end = result.duration
            if final:
                await self.websocket.send_json(
                    {"type": "diarization.done", "duration": self.end}
                )
                return
            await self.websocket.send_json(
                {"type": "audio.ack", "processed_until": self.end}
            )
