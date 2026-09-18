# SPDX-License-Identifier: Apache-2.0
"""Local protocol fixture for the native Swift WebSocket client tests."""

import asyncio
import base64
import json
import sys
from pathlib import Path

from websockets.asyncio.server import serve
from websockets.exceptions import ConnectionClosed


async def handle(socket):
    try:
        await socket.send(json.dumps({"type": "session.created"}))
        update = json.loads(await socket.recv())
        assert update["session"]["turn_detection"] is None
        language = update["session"]["language"]
        if language == "stall":
            await socket.wait_closed()
            return
        await socket.send(json.dumps({"type": "session.updated"}))
        audio = bytearray()
        async for message in socket:
            event = json.loads(message)
            if event["type"] == "input_audio_buffer.append":
                if language == "fail":
                    await socket.send(json.dumps({"type": "error"}))
                    continue
                audio.extend(base64.b64decode(event["audio"], validate=True))
                if len(audio) == 3200:
                    for index, text in enumerate(["hello word", "hello world"]):
                        await socket.send(
                            json.dumps(
                                {
                                    "type": "transcription.segment",
                                    "event_index": index,
                                    "segment_id": 0,
                                    "text": text,
                                    "is_final": False,
                                }
                            )
                        )
            elif event["type"] == "transcription.done":
                assert (
                    audio == bytes([1]) * 3200 + bytes([2]) * 3200
                ), "Audio was dropped or reordered"
                await socket.send(
                    json.dumps(
                        {
                            "type": "transcription.completed",
                            "event_index": 2,
                            "text": "Hello world. Final text.",
                        }
                    )
                )
    except ConnectionClosed:
        # Note (Jiaxin Deng): The client under test disconnects mid-stream on purpose.
        pass


async def main():
    async with serve(handle, "127.0.0.1", 0) as server:
        Path(sys.argv[1]).write_text(str(server.sockets[0].getsockname()[1]))
        await asyncio.Future()


asyncio.run(main())
