# SPDX-License-Identifier: Apache-2.0
"""Exercise Haru's CUDA stage through the public realtime endpoint."""

import asyncio
import base64
import json
import os
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest
from websockets.asyncio.client import ClientConnection, connect

from sglang_omni.utils import find_available_port
from tests.utils import REPO_ROOT, server_log_file, start_server_from_cmd, stop_server

FRAME_BYTES = 3200 * 2
REFERENCE_WAV = REPO_ROOT / "examples/streaming_cascade/voices/vivian.wav"


@pytest.fixture(scope="module")
def haru_server(tmp_path_factory: pytest.TempPathFactory) -> Iterator[str]:
    model_path = os.environ.get("HARU_MODEL_PATH")
    if model_path is None:
        pytest.skip("set HARU_MODEL_PATH to a local Haru-4B checkpoint")
    pytest.importorskip("haru")
    assert Path(model_path).is_dir()
    port = find_available_port()
    process = start_server_from_cmd(
        [
            sys.executable,
            "-m",
            "sglang_omni.cli",
            "serve",
            "--model-path",
            model_path,
            "--enable-realtime",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
        ],
        server_log_file(tmp_path_factory, "haru_realtime"),
        port,
    )
    try:
        yield f"ws://127.0.0.1:{port}/v1/realtime"
    finally:
        stop_server(process)


async def read_through(
    websocket: ClientConnection, final_type: str
) -> list[dict[str, object]]:
    events = []
    async with asyncio.timeout(45):
        while True:
            event = json.loads(await websocket.recv())
            events.append(event)
            if event["type"] == final_type:
                return events
            assert event["type"] not in ("error", "session.closed"), event


async def run_session(url: str) -> bytes:
    async with connect(url, proxy=None) as websocket:
        await read_through(websocket, "session.created")
        await websocket.send(
            json.dumps(
                {
                    "type": "session.update",
                    "event_id": "configure",
                    "session": {
                        "type": "realtime",
                        "instructions": "Say hello.",
                        "output_modalities": ["audio"],
                        "sglang": {
                            "sampling": {"seed": 47, "speak_first": True},
                            "reference_audio": {
                                "media_type": "audio/wav",
                                "data": base64.b64encode(
                                    REFERENCE_WAV.read_bytes()
                                ).decode(),
                            },
                        },
                    },
                }
            )
        )
        await read_through(websocket, "session.updated")
        events = await read_through(websocket, "response.output_audio.delta")
        await websocket.send(
            json.dumps(
                {
                    "type": "input_audio_buffer.append",
                    "event_id": "audio",
                    "sglang": {"seq": 0},
                    "audio": base64.b64encode(bytes(FRAME_BYTES * 3 // 2)).decode(),
                }
            )
        )
        # Hold EOS until a single processed frame arrives, while the 400 ms unit is incomplete.
        first_input = await read_through(websocket, "response.output_audio.delta")
        assert all(event["type"] != "sglang.unit.done" for event in first_input)
        events.extend(first_input)
        await websocket.send(
            json.dumps({"type": "sglang.input_audio.end", "event_id": "end"})
        )
        events.extend(await read_through(websocket, "sglang.input_audio.drained"))
        assert sum(event["type"] == "sglang.unit.done" for event in events) == 1
        assert sum(event["type"] == "response.done" for event in events) == 1
        assert events[-1]["consumed_ms"] == events[-1]["padding_ms"] == 200
        frames = [
            base64.b64decode(event["delta"])
            for event in events
            if event["type"] == "response.output_audio.delta"
        ]
        assert [len(frame) for frame in frames] == [FRAME_BYTES] * 4
        await websocket.send(json.dumps({"type": "session.close", "event_id": "close"}))
        await read_through(websocket, "session.closed")
        await websocket.wait_closed()
        return b"".join(frames)


@pytest.mark.accelerator
@pytest.mark.asyncio
async def test_haru_duplex_reference_replays_and_pads_tail(haru_server: str) -> None:
    audio = await run_session(haru_server)
    assert any(audio)
    assert await run_session(haru_server) == audio
