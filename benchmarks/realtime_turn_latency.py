# SPDX-License-Identifier: Apache-2.0
"""Measure turn-based latency over the native /v1/realtime protocol with server or semantic VAD.

Streams one 16 kHz mono PCM16 WAV at real time, N turns in one session, and
reports per turn, from the moment the client sent the last speech sample:
speech_stopped, first text delta, first audio delta, input transcription
completed and response.done, plus reply audio seconds (24 kHz PCM16), the
user transcript and the reply text. Output is JSON lines.
"""

import argparse
import asyncio
import base64
import json
import time
import uuid
import wave
from typing import Literal

import websockets
from websockets.asyncio.client import ClientConnection

from sglang_omni.audio.semantic_vad import SemanticVADConfig

INPUT_SAMPLE_RATE = 16000
OUTPUT_BYTES_PER_SECOND = 24000 * 2
PACKET_MS = 100
PACKET_BYTES = INPUT_SAMPLE_RATE * PACKET_MS // 1000 * 2
RECV_TIMEOUT_S = 180
MAX_FRAME_BYTES = 16 * 1024 * 1024
IGNORED_EVENTS = ("sglang.unit.done", "sglang.input_audio.accepted")
# Note (evanderfan): the last speech frame can end past the WAV, so the detector's wait alone left the turn open.
SILENCE_MARGIN_MS = 2 * PACKET_MS

TurnDetectionType = Literal["server_vad", "semantic_vad"]
Eagerness = Literal["low", "medium", "high", "auto"]


class RealtimeClient:
    """Native /v1/realtime client; audio seq continues across turns."""

    def __init__(self, websocket: ClientConnection) -> None:
        self.websocket = websocket
        self.next_seq = 0
        self.sent_bytes = 0
        self.sent_at: list[tuple[float, float]] = []

    async def send(self, event_type: str, **fields: object) -> None:
        await self.websocket.send(
            json.dumps({"type": event_type, "event_id": uuid.uuid4().hex, **fields})
        )

    async def recv(self) -> dict:
        event = json.loads(
            await asyncio.wait_for(self.websocket.recv(), RECV_TIMEOUT_S)
        )
        if event["type"] == "error":
            raise RuntimeError(event)
        else:
            return event

    async def wait_for(self, event_type: str) -> dict:
        deadline = time.monotonic() + RECV_TIMEOUT_S
        while time.monotonic() < deadline:
            event = await self.recv()
            if event["type"] == event_type:
                return event
            else:
                pass
        raise TimeoutError(f"{event_type} did not arrive")

    async def configure(
        self,
        instructions: str,
        silence_duration_ms: int,
        turn_detection_type: TurnDetectionType = "server_vad",
        eagerness: Eagerness = "medium",
    ) -> dict:
        """Ask for text and audio with the chosen turn detection; return the session.updated event."""
        turn_detection = (
            {"type": "semantic_vad", "eagerness": eagerness}
            if turn_detection_type == "semantic_vad"
            else {"type": "server_vad", "silence_duration_ms": silence_duration_ms}
        )
        await self.send(
            "session.update",
            session={
                "instructions": instructions,
                "output_modalities": ["text", "audio"],
                "audio": {"input": {"turn_detection": turn_detection}},
            },
        )
        return await self.wait_for("session.updated")

    async def stream_audio(self, pcm: bytes) -> None:
        start = time.perf_counter()
        for index, offset in enumerate(range(0, len(pcm), PACKET_BYTES)):
            await asyncio.sleep(
                max(0.0, start + index * PACKET_MS / 1000 - time.perf_counter())
            )
            piece = pcm[offset : offset + PACKET_BYTES]
            start_ms = self.sent_bytes / 2 / INPUT_SAMPLE_RATE * 1000
            self.sent_at.append((start_ms, time.perf_counter()))
            await self.send(
                "input_audio_buffer.append",
                audio=base64.b64encode(piece).decode("ascii"),
                sglang={"seq": self.next_seq, "t_start_ms": start_ms},
            )
            self.next_seq += 1
            self.sent_bytes += len(piece)

    def sent_time_of(self, audio_ms: float) -> float:
        """Wall-clock time when the packet holding audio_ms was sent."""
        return next(
            sent for start_ms, sent in reversed(self.sent_at) if start_ms <= audio_ms
        )


def required_silence_ms(
    turn_detection_type: TurnDetectionType, silence_ms: int, eagerness: Eagerness
) -> int:
    """The trailing silence that ends a turn whatever its end-of-turn score; auto takes the longest preset."""
    if turn_detection_type == "server_vad":
        wait_ms = silence_ms
    elif eagerness == "auto":
        wait_ms = max(
            SemanticVADConfig.from_eagerness(preset).max_pause_ms
            for preset in ("low", "medium", "high")
        )
    else:
        wait_ms = SemanticVADConfig.from_eagerness(eagerness).max_pause_ms
    return wait_ms + SILENCE_MARGIN_MS


def read_pcm16(path: str, silence_s: float) -> bytes:
    """A 16 kHz mono PCM16 WAV followed by silence_s of silence."""
    with wave.open(path) as wav:
        if (wav.getnchannels(), wav.getframerate(), wav.getsampwidth()) != (
            1,
            INPUT_SAMPLE_RATE,
            2,
        ):
            raise ValueError(
                f"{path} must be mono 16-bit PCM at {INPUT_SAMPLE_RATE} Hz"
            )
        else:
            pcm = wav.readframes(wav.getnframes())
    return pcm + bytes(int(INPUT_SAMPLE_RATE * silence_s) * 2)


async def measure_turn(client: RealtimeClient, pcm: bytes, turn: int) -> dict:
    marks: dict[str, float] = {}
    audio_bytes = 0
    transcript = ""
    text = ""
    sender = asyncio.create_task(client.stream_audio(pcm))
    try:
        while "done" not in marks or "transcript" not in marks:
            event = await client.recv()
            now = time.perf_counter()
            kind = event["type"]
            if kind in IGNORED_EVENTS:
                continue
            elif kind == "input_audio_buffer.speech_stopped":
                marks["speech_stopped"] = now
                marks["speech_end"] = client.sent_time_of(event["audio_end_ms"])
            elif kind == "response.output_text.delta":
                marks.setdefault("first_text", now)
            elif kind == "response.output_audio.delta":
                marks.setdefault("first_audio", now)
                audio_bytes += len(base64.b64decode(event["delta"]))
            elif kind == "conversation.item.input_audio_transcription.completed":
                transcript = event["transcript"]
                marks["transcript"] = now
            elif kind == "response.done":
                marks["done"] = now
                text = next(
                    (
                        part["text"]
                        for part in event["response"]["output"][0]["content"]
                        if part["type"] == "output_text"
                    ),
                    "",
                )
            else:
                pass
        await sender
    finally:
        sender.cancel()
    base = marks["speech_end"]
    since_speech_end_ms = {
        name: round((marks[name] - base) * 1000) if name in marks else None
        for name in (
            "speech_stopped",
            "first_text",
            "first_audio",
            "transcript",
            "done",
        )
    }
    return {
        "turn": turn,
        "speech_end_to_stopped_ms": since_speech_end_ms["speech_stopped"],
        "speech_end_to_first_text_ms": since_speech_end_ms["first_text"],
        "speech_end_to_first_audio_ms": since_speech_end_ms["first_audio"],
        "speech_end_to_transcript_ms": since_speech_end_ms["transcript"],
        "speech_end_to_done_ms": since_speech_end_ms["done"],
        "audio_s": round(audio_bytes / OUTPUT_BYTES_PER_SECOND, 2),
        "transcript": transcript,
        "text": text,
    }


async def run(
    url: str,
    audio: str,
    turns: int,
    instructions: str,
    silence_s: float,
    silence_ms: int,
    turn_detection_type: TurnDetectionType = "server_vad",
    eagerness: Eagerness = "medium",
) -> None:
    pcm = read_pcm16(audio, silence_s)
    async with websockets.connect(url, max_size=MAX_FRAME_BYTES) as websocket:
        client = RealtimeClient(websocket)
        created = await client.recv()
        if created["type"] != "session.created":
            raise RuntimeError(f"expected session.created, got {created['type']}")
        else:
            pass
        updated = await client.configure(
            instructions, silence_ms, turn_detection_type, eagerness
        )
        granted = updated["session"]["sglang"]["granted"]
        if granted.get("interaction") != "turn_based":
            raise RuntimeError(f"server did not grant turn_based: {granted}")
        else:
            pass
        for turn in range(turns):
            print(
                json.dumps(await measure_turn(client, pcm, turn), ensure_ascii=False),
                flush=True,
            )
        await client.send("session.close")
        await client.wait_for("session.closed")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="ws://127.0.0.1:8000/v1/realtime")
    parser.add_argument("--audio", required=True)
    parser.add_argument("--turns", type=int, default=3)
    parser.add_argument(
        "--silence-s",
        type=float,
        default=None,
        help="Trailing silence appended to the WAV in each turn; at least the selected turn detection's longest wait.",
    )
    parser.add_argument(
        "--turn-detection",
        choices=("server_vad", "semantic_vad"),
        default="server_vad",
        help="Realtime turn_detection.type.",
    )
    parser.add_argument(
        "--silence-ms",
        type=int,
        default=800,
        help="Server VAD turn_detection.silence_duration_ms; ignored with semantic_vad.",
    )
    parser.add_argument(
        "--eagerness",
        choices=("low", "medium", "high", "auto"),
        default="medium",
        help="Semantic VAD turn_detection.eagerness; ignored with server_vad.",
    )
    parser.add_argument(
        "--instructions",
        default="Answer in one spoken English sentence of at most 25 words.",
    )
    arguments = parser.parse_args()
    required_silence_s = (
        required_silence_ms(
            arguments.turn_detection, arguments.silence_ms, arguments.eagerness
        )
        / 1000
    )
    if arguments.silence_s is None:
        arguments.silence_s = required_silence_s
    elif arguments.silence_s < required_silence_s:
        parser.error(f"--silence-s must be at least {required_silence_s} s")
    else:
        pass
    asyncio.run(
        run(
            arguments.url,
            arguments.audio,
            arguments.turns,
            arguments.instructions,
            arguments.silence_s,
            arguments.silence_ms,
            arguments.turn_detection,
            arguments.eagerness,
        )
    )


if __name__ == "__main__":
    main()
