# SPDX-License-Identifier: Apache-2.0
"""Send a laptop microphone or PCM16 WAV to the live diarization endpoint.

Install the client dependencies with pip install websockets sounddevice.
"""

import argparse
import asyncio
import json
import queue
import time
import wave
from typing import Literal, Protocol

import numpy as np
import sounddevice
from websockets import ClientConnection, connect

SAMPLE_RATE = 16000
BLOCK_FRAMES = 1600
# note (Ryan Stewart): Ride out brief network or first-inference stalls at bounded latency.
MAX_QUEUED_BLOCKS = 50
# note (Ryan Stewart): Catch up after a slow ack in one message, up to the server's 1 second limit.
MAX_BLOCKS_PER_MESSAGE = 10
MAX_EVENT_BYTES = 1024 * 1024
IDLE_POLL_S = 0.01


class StreamTime(Protocol):
    """PortAudio timestamps that sounddevice passes to an input callback."""

    inputBufferAdcTime: float  # noqa: N815 - PortAudio spelling
    currentTime: float  # noqa: N815 - PortAudio spelling


async def receive_until(
    websocket: ClientConnection, expected: Literal["audio.ack", "diarization.done"]
) -> float:
    """Print speaker updates until the expected event, then return its time in seconds."""
    while True:
        event = json.loads(await websocket.recv())
        if event["type"] == "error":
            raise RuntimeError(event["message"])
        elif event["type"] == "diarization.update":
            for segment in event["segments"]:
                print(
                    f'{segment["start"]:8.2f} to {segment["end"]:8.2f} s '
                    f'{segment["speaker"]}',
                    flush=True,
                )
        elif event["type"] == expected:
            return float(
                event["processed_until" if expected == "audio.ack" else "duration"]
            )
        else:
            pass


async def replay(websocket: ClientConnection, path: str) -> None:
    with wave.open(path, "rb") as source:
        if (source.getnchannels(), source.getframerate(), source.getsampwidth()) != (
            1,
            SAMPLE_RATE,
            2,
        ):
            raise ValueError("Use a mono, 16 kHz PCM16 WAV file")
        else:
            pass
        started = time.monotonic()
        sent_samples = 0
        while audio := source.readframes(BLOCK_FRAMES):
            await websocket.send(audio)
            await receive_until(websocket, "audio.ack")
            sent_samples += len(audio) // 2
            await asyncio.sleep(
                max(0, started + sent_samples / SAMPLE_RATE - time.monotonic())
            )


async def microphone(websocket: ClientConnection, duration_s: float) -> None:
    audio_queue: queue.Queue[bytes] = queue.Queue(maxsize=MAX_QUEUED_BLOCKS)
    failures: list[str] = []

    async def send_buffered() -> bool:
        audio = bytearray()
        for _ in range(MAX_BLOCKS_PER_MESSAGE):
            try:
                audio.extend(audio_queue.get_nowait())
            except queue.Empty:
                break
        if not audio:
            return False
        else:
            await websocket.send(bytes(audio))
            await receive_until(websocket, "audio.ack")
            return True

    def capture(
        input_audio: np.ndarray,
        frame_count: int,
        stream_time: StreamTime,
        status: sounddevice.CallbackFlags,
    ) -> None:
        """Queue one block. sounddevice also passes the frame count and timestamps."""
        if status:
            failures.append(str(status))
            raise sounddevice.CallbackAbort
        else:
            pass
        try:
            audio_queue.put_nowait(input_audio.tobytes())
        except queue.Full:
            failures.append(
                "Connection or server cannot keep up, the microphone buffer "
                "reached 5 seconds"
            )
            raise sounddevice.CallbackAbort from None

    print(
        f"Recording for {duration_s:g} seconds. Speak near your microphone.",
        flush=True,
    )
    stop_at = time.monotonic() + duration_s
    with sounddevice.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="int16",
        blocksize=BLOCK_FRAMES,
        callback=capture,
    ):
        while time.monotonic() < stop_at:
            if failures:
                raise RuntimeError(failures[0])
            elif not await send_buffered():
                await asyncio.sleep(IDLE_POLL_S)
            else:
                pass
    if failures:
        raise RuntimeError(failures[0])
    else:
        pass
    while not audio_queue.empty():
        await send_buffered()


async def main(arguments: argparse.Namespace) -> None:
    async with connect(arguments.url, max_size=MAX_EVENT_BYTES) as websocket:
        event = json.loads(await websocket.recv())
        if event["type"] != "session.ready":
            raise RuntimeError(event)
        elif arguments.file:
            await replay(websocket, arguments.file)
        else:
            await microphone(websocket, arguments.seconds)
        await websocket.send(json.dumps({"type": "audio.end"}))
        duration_s = await receive_until(websocket, "diarization.done")
        print(f"Finished: {duration_s:.2f} seconds", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--url", default="ws://localhost:8000/v1/audio/diarizations/stream"
    )
    parser.add_argument(
        "--file", help="Replay a mono 16 kHz PCM16 WAV at recording speed"
    )
    parser.add_argument(
        "--seconds", type=float, default=30, help="Microphone recording duration"
    )
    arguments = parser.parse_args()
    if arguments.seconds <= 0:
        parser.error("--seconds must be positive")
    else:
        pass
    asyncio.run(main(arguments))
