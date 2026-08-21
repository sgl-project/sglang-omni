#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Microphone client for the Nemotron VoiceChat realtime endpoint."""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import math
import sys
import time
import wave
from array import array
from pathlib import Path

import websockets

INPUT_RATE = 16_000
OUTPUT_RATE = 22_050
FRAME_SAMPLES = 1_280
FRAME_BYTES = FRAME_SAMPLES * 2


def resample_pcm16(pcm: bytes, source_rate: int, target_rate: int) -> bytes:
    """Linearly resample mono little-endian PCM16 without extra dependencies."""
    if source_rate == target_rate or not pcm:
        return pcm
    samples = array("h")
    samples.frombytes(pcm)
    if sys.byteorder != "little":
        samples.byteswap()
    output_count = round(len(samples) * target_rate / source_rate)
    output = array("h")
    for index in range(output_count):
        position = index * source_rate / target_rate
        left = min(int(position), len(samples) - 1)
        right = min(left + 1, len(samples) - 1)
        fraction = position - left
        output.append(
            round(samples[left] * (1.0 - fraction) + samples[right] * fraction)
        )
    if sys.byteorder != "little":
        output.byteswap()
    return output.tobytes()


def _load_pyaudio():
    try:
        import pyaudio
    except ImportError as error:
        raise RuntimeError(
            "Microphone capture and playback require PyAudio. Install PortAudio, "
            "then run `python -m pip install pyaudio`."
        ) from error
    return pyaudio


def list_audio_devices() -> None:
    pyaudio = _load_pyaudio()
    audio = pyaudio.PyAudio()
    try:
        for index in range(audio.get_device_count()):
            info = audio.get_device_info_by_index(index)
            print(
                f"{index}: {info['name']} "
                f"(inputs={info['maxInputChannels']}, "
                f"outputs={info['maxOutputChannels']}, "
                f"default_rate={info['defaultSampleRate']})"
            )
    finally:
        audio.terminate()


class AudioDevices:
    def __init__(self, args: argparse.Namespace) -> None:
        self.pyaudio = _load_pyaudio()
        self.audio = self.pyaudio.PyAudio()
        self.input_stream = None
        self.output_stream = None
        self.input_frames_read = 0
        try:
            input_info = self._device_info(args.input_device_index, capture=True)
            self.input_rate = round(input_info["defaultSampleRate"])
            self.input_frame_samples = round(self.input_rate * 0.08)

            input_kwargs = {
                "format": self.pyaudio.paInt16,
                "channels": 1,
                "rate": self.input_rate,
                "input": True,
                "frames_per_buffer": self.input_frame_samples,
            }
            if args.input_device_index is not None:
                input_kwargs["input_device_index"] = args.input_device_index
            self.input_stream = self.audio.open(**input_kwargs)
            print(
                f"input device: {input_info['name']} at {self.input_rate} Hz "
                f"(resampled to {INPUT_RATE} Hz)"
            )

            if not args.no_playback:
                output_info = self._device_info(args.output_device_index, capture=False)
                self.output_rate = round(output_info["defaultSampleRate"])
                output_kwargs = {
                    "format": self.pyaudio.paInt16,
                    "channels": 1,
                    "rate": self.output_rate,
                    "output": True,
                    "frames_per_buffer": round(self.output_rate * 0.08),
                }
                if args.output_device_index is not None:
                    output_kwargs["output_device_index"] = args.output_device_index
                self.output_stream = self.audio.open(**output_kwargs)
                print(
                    f"output device: {output_info['name']} at {self.output_rate} Hz "
                    f"(resampled from {OUTPUT_RATE} Hz)"
                )
        except Exception:
            self.close()
            raise

    def _device_info(self, index: int | None, *, capture: bool):
        if index is not None:
            return self.audio.get_device_info_by_index(index)
        if capture:
            return self.audio.get_default_input_device_info()
        return self.audio.get_default_output_device_info()

    def read(self) -> bytes:
        pcm = self.input_stream.read(
            self.input_frame_samples, exception_on_overflow=False
        )
        samples = array("h")
        samples.frombytes(pcm)
        peak = max((abs(sample) for sample in samples), default=0)
        rms = math.sqrt(
            sum(sample * sample for sample in samples) / max(1, len(samples))
        )
        self.input_frames_read += 1
        if self.input_frames_read == 1 or self.input_frames_read % 12 == 0:
            print(f"microphone level: rms={rms:.0f} peak={peak}", flush=True)
        return resample_pcm16(pcm, self.input_rate, INPUT_RATE)

    def write(self, pcm: bytes) -> None:
        if self.output_stream is not None:
            self.output_stream.write(resample_pcm16(pcm, OUTPUT_RATE, self.output_rate))

    def close(self) -> None:
        for stream in (self.input_stream, self.output_stream):
            if stream is not None:
                try:
                    stream.stop_stream()
                finally:
                    stream.close()
        self.input_stream = None
        self.output_stream = None
        if self.audio is not None:
            self.audio.terminate()
            self.audio = None


def validate_session_created(event: dict) -> None:
    if event.get("type") != "session.created":
        raise RuntimeError(f"Expected session.created, got {event.get('type')!r}")
    session = event.get("session", {})
    expected = {
        "input_audio_format": "pcm16",
        "input_sample_rate": INPUT_RATE,
        "output_audio_format": "pcm16",
        "output_sample_rate": OUTPUT_RATE,
        "frame_samples": FRAME_SAMPLES,
    }
    mismatches = {
        key: (session.get(key), value)
        for key, value in expected.items()
        if session.get(key) != value
    }
    if mismatches:
        details = ", ".join(
            f"{key}={actual!r} (expected {expected_value!r})"
            for key, (actual, expected_value) in mismatches.items()
        )
        raise RuntimeError(f"Incompatible realtime session: {details}")


async def _send_frame(socket, frame: bytes) -> None:
    await socket.send(
        json.dumps(
            {
                "type": "input_audio_buffer.append",
                "audio": base64.b64encode(frame).decode("ascii"),
            }
        )
    )


async def _send_microphone(socket, receiver, devices, duration: float | None) -> None:
    print("Microphone streaming started.")
    stop_waiter = None
    deadline = None
    if duration is None:
        stop_waiter = asyncio.create_task(
            asyncio.to_thread(input, "Press Enter to stop the session.\n")
        )
    else:
        deadline = time.monotonic() + duration

    try:
        while True:
            if stop_waiter is not None and stop_waiter.done():
                await stop_waiter
                break
            if deadline is not None and time.monotonic() >= deadline:
                break
            frame = await asyncio.to_thread(devices.read)
            await _send_frame(socket, frame)
            if receiver.done():
                await receiver
    finally:
        if stop_waiter is not None and not stop_waiter.done():
            stop_waiter.cancel()


async def _send_trailing_silence(socket, receiver, seconds: float) -> None:
    silence = bytes(FRAME_BYTES)
    next_frame_at = asyncio.get_running_loop().time()
    for _ in range(math.ceil(seconds * INPUT_RATE / FRAME_SAMPLES)):
        await _send_frame(socket, silence)
        if receiver.done():
            await receiver
        next_frame_at += 0.08
        await asyncio.sleep(max(0, next_frame_at - asyncio.get_running_loop().time()))


async def run(args: argparse.Namespace) -> None:
    devices = AudioDevices(args)
    output = bytearray()
    try:
        async with websockets.connect(
            args.url, max_size=8 * 1024 * 1024, ping_timeout=60
        ) as socket:
            created = json.loads(await socket.recv())
            validate_session_created(created)
            print(json.dumps(created, indent=2))
            await socket.send(
                json.dumps(
                    {
                        "type": "session.update",
                        "session": {"instructions": args.instructions},
                    }
                )
            )
            print(await socket.recv())

            committed = asyncio.Event()

            async def receive_output() -> None:
                async for raw_event in socket:
                    event = json.loads(raw_event)
                    event_type = event.get("type")
                    if event_type == "error":
                        raise RuntimeError(event["error"]["message"])
                    if event_type == "input_audio_buffer.committed":
                        committed.set()
                    elif event_type == "response.text.delta":
                        print(event.get("delta", ""), end="", flush=True)
                    elif event_type == "response.audio.delta":
                        pcm = base64.b64decode(event["delta"], validate=True)
                        output.extend(pcm)
                        await asyncio.to_thread(devices.write, pcm)

            receiver = asyncio.create_task(receive_output())
            await _send_microphone(socket, receiver, devices, args.microphone_seconds)
            await _send_trailing_silence(socket, receiver, args.trailing_silence)
            await socket.send(json.dumps({"type": "input_audio_buffer.commit"}))

            commit_waiter = asyncio.create_task(committed.wait())
            done, _ = await asyncio.wait(
                {commit_waiter, receiver},
                timeout=args.drain_timeout,
                return_when=asyncio.FIRST_COMPLETED,
            )
            if not done:
                commit_waiter.cancel()
                raise TimeoutError("Timed out while draining queued audio frames.")
            if receiver in done:
                await receiver
                if not commit_waiter.done():
                    commit_waiter.cancel()
                    raise RuntimeError(
                        "Realtime session ended before queued audio was committed."
                    )
            await commit_waiter
            await socket.send(json.dumps({"type": "session.close"}))
            await receiver
            print()
    finally:
        devices.close()

    args.output_wav.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(args.output_wav), "wb") as destination:
        destination.setnchannels(1)
        destination.setsampwidth(2)
        destination.setframerate(OUTPUT_RATE)
        destination.writeframes(output)
    print(f"wrote {len(output) // 2} samples to {args.output_wav}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="ws://127.0.0.1:18080/v1/realtime")
    parser.add_argument("--output-wav", type=Path, default=Path("response.wav"))
    parser.add_argument(
        "--instructions", default="You are a helpful, concise voice assistant."
    )
    parser.add_argument("--input-device-index", type=int)
    parser.add_argument("--output-device-index", type=int)
    parser.add_argument("--list-devices", action="store_true")
    parser.add_argument("--no-playback", action="store_true")
    parser.add_argument("--trailing-silence", type=float, default=2.0)
    parser.add_argument(
        "--microphone-seconds",
        type=float,
        help="Stop microphone capture automatically after this many seconds.",
    )
    parser.add_argument("--drain-timeout", type=float, default=600.0)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.list_devices:
        list_audio_devices()
        return
    if args.trailing_silence < 0:
        parser.error("--trailing-silence cannot be negative")
    if args.microphone_seconds is not None and args.microphone_seconds <= 0:
        parser.error("--microphone-seconds must be positive")
    try:
        asyncio.run(run(args))
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)


if __name__ == "__main__":
    main()
