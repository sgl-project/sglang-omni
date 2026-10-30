# SPDX-License-Identifier: Apache-2.0
"""The gesture benchmark sends contiguous units accepted by the session wire."""

import msgpack
import numpy as np
import pytest
from websockets.asyncio.server import ServerConnection, serve

from benchmarks.eval import miburi_gesture_session
from sglang_omni.models.personaplex.architecture import SAMPLES_PER_FRAME
from sglang_omni.pipeline.sessions import check_timing
from sglang_omni.proto.session import TimedChunk


@pytest.mark.asyncio
async def test_caller_unit_timestamps_remain_contiguous() -> None:
    """The seconds-to-milliseconds conversion used to overlap at unit 36."""
    received: list[TimedChunk] = []

    async def receive(socket: ServerConnection) -> None:
        modality_end_ms: dict[str, float] = {}
        async for frame in socket:
            message = msgpack.unpackb(frame, raw=False)
            if message["op"] == "open":
                await socket.send(msgpack.packb({"op": "opened"}))
            elif message["op"] == "append":
                chunk = TimedChunk.from_dict(message["chunk"])
                check_timing(chunk, modality_end_ms)
                modality_end_ms[chunk.modality] = chunk.t_start_ms + chunk.duration_ms
                received.append(chunk)
                if chunk.eos:
                    await socket.send(
                        msgpack.packb(
                            {"op": "output", "chunk": {"eos": True, "format": "test"}}
                        )
                    )
                else:
                    pass
            else:
                assert message["op"] == "close"
                await socket.send(msgpack.packb({"op": "closed"}))
                return

    async with serve(receive, "127.0.0.1", 0) as server:
        port = server.sockets[0].getsockname()[1]
        await miburi_gesture_session.run(
            f"ws://127.0.0.1:{port}",
            np.zeros(40 * SAMPLES_PER_FRAME, dtype=np.float32),
            {},
            "test",
        )

    assert len(received) == 41
    assert received[-1].eos
    assert received[-1].t_start_ms == 3200
