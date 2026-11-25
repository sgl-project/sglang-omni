# SPDX-License-Identifier: Apache-2.0
"""Run the Qwen D2 model and its fixed-capacity graph cache in the release environment."""

import asyncio
import base64
import json
import logging
import sys
from multiprocessing.connection import Connection

from d2_qwen.inference.runtime import Runtime
from d2_qwen.inference.session import INPUT_FRAME_SAMPLES, D2LiveSession

logger = logging.getLogger(__name__)


async def serve(connection: Connection, runtime: Runtime) -> None:
    session: D2LiveSession | None = None
    remaining_input_bytes = 0
    try:
        await runtime.load()
        connection.send_bytes(b"{}")
        while True:
            try:
                command = connection.recv_bytes()
            except EOFError:
                break
            try:
                if command == b"open":
                    session = await runtime.create_session()
                    remaining_input_bytes = (
                        runtime.max_context_frames * INPUT_FRAME_SAMPLES * 2
                    )
                    connection.send_bytes(b"{}")
                elif session is None:
                    raise RuntimeError("D2 session is not open")
                elif command == b"close":
                    try:
                        await session.close()
                    finally:
                        session = None
                    connection.send_bytes(b"{}")
                elif command.startswith(b"pcm:"):
                    pcm16 = command[4:]
                    if len(pcm16) > remaining_input_bytes:
                        raise ValueError(
                            "context_exhausted: D2 session reached its fixed context limit"
                        )
                    else:
                        batch = await session.push_pcm16(pcm16)
                        remaining_input_bytes -= len(pcm16)
                    frames = [
                        {
                            "pcm16": base64.b64encode(playback_pcm16).decode("ascii"),
                            "text_delta": event.get("text_delta", ""),
                            "sequence": event["sequence"],
                            "token_id": event["token_id"],
                            "is_interruption": event["token_id"]
                            == runtime.model.interrupt_id,
                            "response_open": event["metadata"]["response_open"],
                            "emit_audio": event["metadata"]["emit_audio"],
                        }
                        for playback_pcm16, event in zip(
                            batch.pcm_frames, batch.frame_events, strict=True
                        )
                    ]
                    connection.send_bytes(json.dumps({"frames": frames}).encode())
                else:
                    raise ValueError("Unknown D2 worker command")
            except Exception as error:
                logger.exception("D2 session operation failed")
                connection.send_bytes(json.dumps({"error": str(error)}).encode())
    finally:
        try:
            if session is not None:
                await session.close()
            else:
                pass
        finally:
            await runtime.close()


if __name__ == "__main__":
    _, socket_descriptor, model_path, device, kv_budget, seed = sys.argv
    with Connection(int(socket_descriptor)) as stage_connection:
        asyncio.run(
            serve(
                stage_connection,
                Runtime(
                    model_path, device=device, kv_budget=int(kv_budget), seed=int(seed)
                ),
            )
        )
else:
    pass
