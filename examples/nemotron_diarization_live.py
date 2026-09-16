# SPDX-License-Identifier: Apache-2.0
"""Send a laptop microphone or PCM16 WAV to the live diarization endpoint.

Microphone: pip install websockets sounddevice
File replay: pip install websockets
"""

import argparse
import asyncio
import json
import queue
import time
import wave

from websockets import connect


async def receive_until_ack(websocket):
    while True:
        event = json.loads(await websocket.recv())
        if event["type"] == "error":
            raise RuntimeError(event["message"])
        if event["type"] == "diarization.update":
            for segment in event["segments"]:
                print(
                    f'{segment["start"]:8.2f}–{segment["end"]:8.2f}s '
                    f'{segment["speaker"]}',
                    flush=True,
                )
        if event["type"] in {"audio.ack", "diarization.done"}:
            return event


async def replay(websocket, path):
    with wave.open(path, "rb") as source:
        if (source.getnchannels(), source.getframerate(), source.getsampwidth()) != (
            1,
            16000,
            2,
        ):
            raise ValueError("Use a mono, 16 kHz PCM16 WAV file")
        start = time.monotonic()
        sent = 0
        while audio := source.readframes(1600):
            await websocket.send(audio)
            await receive_until_ack(websocket)
            sent += len(audio) // 2
            await asyncio.sleep(max(0, start + sent / 16000 - time.monotonic()))


async def microphone(websocket, seconds):
    import sounddevice as sd

    # Allow brief network or first-inference stalls, while bounding latency.
    audio_queue = queue.Queue(maxsize=50)
    failure = []

    async def send_buffered():
        audio = bytearray()
        # Catch up after a slow acknowledgment without paying another network
        # round trip for every 100 ms block. The server accepts at most 1 second.
        for _ in range(10):
            try:
                audio.extend(audio_queue.get_nowait())
            except queue.Empty:
                break
        if not audio:
            return False
        await websocket.send(bytes(audio))
        await receive_until_ack(websocket)
        return True

    def capture(data, frames, timing, status):
        del frames, timing
        if status:
            failure.append(str(status))
            raise sd.CallbackAbort
        try:
            audio_queue.put_nowait(bytes(data))
        except queue.Full:
            failure.append(
                "Connection or server cannot keep up; microphone buffer reached 5 seconds"
            )
            raise sd.CallbackAbort

    print(f"Recording for {seconds:g} seconds. Speak near your microphone.", flush=True)
    stop_at = time.monotonic() + seconds
    with sd.RawInputStream(
        samplerate=16000, channels=1, dtype="int16", blocksize=1600, callback=capture
    ):
        while time.monotonic() < stop_at:
            if failure:
                raise RuntimeError(failure[0])
            if not await send_buffered():
                await asyncio.sleep(0.01)
    if failure:
        raise RuntimeError(failure[0])
    while not audio_queue.empty():
        await send_buffered()


async def main(args):
    async with connect(args.url, max_size=1024 * 1024) as websocket:
        event = json.loads(await websocket.recv())
        if event["type"] != "session.ready":
            raise RuntimeError(event)
        if args.file:
            await replay(websocket, args.file)
        else:
            await microphone(websocket, args.seconds)
        await websocket.send(json.dumps({"type": "audio.end"}))
        done = await receive_until_ack(websocket)
        print(f'Finished: {done["duration"]:.2f} seconds', flush=True)


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
    args = parser.parse_args()
    if args.seconds <= 0:
        parser.error("--seconds must be positive")
    asyncio.run(main(args))
