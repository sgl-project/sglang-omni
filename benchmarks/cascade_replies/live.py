# SPDX-License-Identifier: Apache-2.0
"""Talk to a running cascade over /v1/realtime, one session per conversation, one spoken turn at a time.

The microphone sends silence between turns, and the next turn starts only after the whole reply would have played plus a pause, so the dialog records every reply as heard. Per turn it keeps the transcript, the reply text and audio, and the time from the end of the user's speech to speech_stopped, the first text, the first audio and the end of the reply.
"""

from __future__ import annotations

import asyncio
import base64
import json
import time
import uuid
import wave
from pathlib import Path
from typing import Literal

import websockets
from pydantic import BaseModel, JsonValue

from benchmarks.cascade_replies.replies import (
    CONVERSATIONS_PATH,
    Conversation,
    ConversationSet,
    Language,
)

INPUT_SAMPLE_RATE = 16000
OUTPUT_SAMPLE_RATE = 24000
PCM16_SAMPLE_BYTES = 2
PACKET_MS = 100
PACKET_BYTES = INPUT_SAMPLE_RATE * PACKET_MS // 1000 * PCM16_SAMPLE_BYTES
OUTPUT_BYTES_PER_SECOND = OUTPUT_SAMPLE_RATE * PCM16_SAMPLE_BYTES
EVENT_TIMEOUT_S = 240
MAX_FRAME_BYTES = 64 * 1024 * 1024
# note (Haoling Pu): four conversations per language that together cover every category.
LIVE_CONVERSATIONS = (
    "en_02",
    "en_03",
    "en_08",
    "en_09",
    "zh_02",
    "zh_03",
    "zh_08",
    "zh_09",
)


TurnMark = Literal["speech_stopped", "speech_end", "first_text", "first_audio", "done"]


class ServerEvent(BaseModel):
    """The type of a realtime server event; the events read further have their own models below."""

    type: str


class SpeechStopped(BaseModel):
    audio_end_ms: int


class TranscriptionCompleted(BaseModel):
    transcript: str


class OutputDelta(BaseModel):
    delta: str


class ResponseStatusDetails(BaseModel):
    reason: str


class ResponseState(BaseModel):
    status: str
    status_details: ResponseStatusDetails


class ResponseDone(BaseModel):
    response: ResponseState


class SpokenTurn(BaseModel):
    conversation: str
    language: Language
    turn: int
    category: str
    said: str
    transcript: str
    reply: str
    status: str
    reason: str
    audio_s: float
    speech_stopped_ms: int | None
    first_text_ms: int | None
    first_audio_ms: int | None
    done_ms: int | None


async def talk(
    url: str,
    conversation: Conversation,
    audio_dir: Path,
    output_dir: Path,
    pause_s: float,
    session: dict[str, JsonValue],
) -> list[SpokenTurn]:
    microphone_queue: list[bytes] = []
    sent_at: list[tuple[int, float]] = []
    events: asyncio.Queue[tuple[float, str | bytes]] = asyncio.Queue()
    spoken_turns = []
    async with websockets.connect(url, max_size=MAX_FRAME_BYTES) as websocket:

        async def send(event_type: str, fields: dict[str, JsonValue]) -> None:
            await websocket.send(
                json.dumps({"type": event_type, "event_id": uuid.uuid4().hex, **fields})
            )

        async def receive() -> None:
            async for message in websocket:
                await events.put((time.perf_counter(), message))

        async def next_event() -> tuple[float, str, str | bytes]:
            received_at, message = await asyncio.wait_for(events.get(), EVENT_TIMEOUT_S)
            event_type = ServerEvent.model_validate_json(message).type
            if event_type == "error":
                raise RuntimeError(f"server error in {conversation.id}")
            else:
                return received_at, event_type, message

        async def microphone() -> None:
            sequence = 0
            started_at = time.perf_counter()
            while True:
                await asyncio.sleep(
                    max(
                        0.0,
                        started_at + sequence * PACKET_MS / 1000 - time.perf_counter(),
                    )
                )
                packet = (
                    microphone_queue.pop(0) if microphone_queue else bytes(PACKET_BYTES)
                )
                sent_at.append((sequence * PACKET_MS, time.perf_counter()))
                await send(
                    "input_audio_buffer.append",
                    {
                        "audio": base64.b64encode(packet).decode("ascii"),
                        "sglang": {"seq": sequence, "t_start_ms": sequence * PACKET_MS},
                    },
                )
                sequence += 1

        receiver = asyncio.create_task(receive())
        _, event_type, _ = await next_event()
        if event_type != "session.created":
            raise RuntimeError(f"expected session.created, got {event_type}")
        else:
            pass
        await send("session.update", {"session": session})
        while event_type != "session.updated":
            _, event_type, _ = await next_event()
        sender = asyncio.create_task(microphone())
        for turn, conversation_turn in enumerate(conversation.turns):
            with wave.open(str(audio_dir / f"{conversation.id}_{turn}.wav")) as source:
                pcm = source.readframes(source.getnframes())
            pcm += bytes((-len(pcm)) % PACKET_BYTES)
            microphone_queue.extend(
                pcm[offset : offset + PACKET_BYTES]
                for offset in range(0, len(pcm), PACKET_BYTES)
            )
            marks: dict[TurnMark, float] = {}
            reply = ""
            transcript = ""
            audio = bytearray()
            response: ResponseState | None = None
            while response is None:
                received_at, event_type, message = await next_event()
                if event_type == "input_audio_buffer.speech_stopped":
                    audio_end_ms = SpeechStopped.model_validate_json(
                        message
                    ).audio_end_ms
                    marks["speech_stopped"] = received_at
                    marks["speech_end"] = next(
                        sent
                        for start_ms, sent in reversed(sent_at)
                        if start_ms <= audio_end_ms
                    )
                elif (
                    event_type
                    == "conversation.item.input_audio_transcription.completed"
                ):
                    transcript = TranscriptionCompleted.model_validate_json(
                        message
                    ).transcript
                elif event_type == "response.output_text.delta":
                    marks.setdefault("first_text", received_at)
                    reply += OutputDelta.model_validate_json(message).delta
                elif event_type == "response.output_audio.delta":
                    marks.setdefault("first_audio", received_at)
                    audio += base64.b64decode(
                        OutputDelta.model_validate_json(message).delta
                    )
                elif event_type == "response.done":
                    marks["done"] = received_at
                    response = ResponseDone.model_validate_json(message).response
                else:
                    pass
            with wave.open(
                str(output_dir / f"{conversation.id}_{turn}.wav"), "wb"
            ) as sink:
                sink.setnchannels(1)
                sink.setsampwidth(PCM16_SAMPLE_BYTES)
                sink.setframerate(OUTPUT_SAMPLE_RATE)
                sink.writeframes(bytes(audio))
            audio_s = len(audio) / OUTPUT_BYTES_PER_SECOND
            speech_end = marks.get("speech_end")
            since_speech_end_ms: dict[TurnMark, int | None] = {
                mark: (
                    round((marks[mark] - speech_end) * 1000)
                    if speech_end is not None and mark in marks
                    else None
                )
                for mark in ("speech_stopped", "first_text", "first_audio", "done")
            }
            spoken_turn = SpokenTurn(
                conversation=conversation.id,
                language=conversation.language,
                turn=turn,
                category=conversation_turn.category,
                said=conversation_turn.user,
                transcript=transcript,
                reply=reply,
                status=response.status,
                reason=response.status_details.reason,
                audio_s=round(audio_s, 2),
                speech_stopped_ms=since_speech_end_ms["speech_stopped"],
                first_text_ms=since_speech_end_ms["first_text"],
                first_audio_ms=since_speech_end_ms["first_audio"],
                done_ms=since_speech_end_ms["done"],
            )
            spoken_turns.append(spoken_turn)
            print(spoken_turn.model_dump_json(), flush=True)
            heard_at = marks.get("first_audio", marks["done"]) + audio_s
            await asyncio.sleep(max(0.0, heard_at - time.perf_counter()) + pause_s)
        sender.cancel()
        await send("session.close", {})
        receiver.cancel()
    return spoken_turns


async def run_live(
    url: str,
    audio_dir: Path,
    output_dir: Path,
    pause_s: float,
    server_vad_silence_ms: int | None,
) -> None:
    conversations = {
        conversation.id: conversation
        for conversation in ConversationSet.model_validate_json(
            CONVERSATIONS_PATH.read_text()
        ).conversations
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    session: dict[str, JsonValue] = {"output_modalities": ["text", "audio"]}
    if server_vad_silence_ms is not None:
        session["audio"] = {
            "input": {
                "turn_detection": {
                    "type": "server_vad",
                    "silence_duration_ms": server_vad_silence_ms,
                }
            }
        }
    else:
        pass
    spoken_turns = []
    for conversation_id in LIVE_CONVERSATIONS:
        spoken_turns += await talk(
            url, conversations[conversation_id], audio_dir, output_dir, pause_s, session
        )
    (output_dir / "turns.jsonl").write_text(
        "".join(spoken_turn.model_dump_json() + "\n" for spoken_turn in spoken_turns)
    )
