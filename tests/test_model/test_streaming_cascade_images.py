# SPDX-License-Identifier: Apache-2.0
"""Real ASR, VLM and TTS acceptance checks against a running image cascade."""

import asyncio
import base64
import io
import json
import os
import wave
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path

import pytest
from PIL import Image, ImageDraw
from websockets.asyncio.client import ClientConnection, connect

from sglang_omni.serve.realtime.schema import JsonObject, JsonValue

PORT = os.environ.get("SGLANG_OMNI_TEST_REALTIME_PORT")
AUDIO_PATH = os.environ.get("SGLANG_OMNI_CASCADE_AUDIO")
FOLLOWUP_PATH = os.environ.get("SGLANG_OMNI_CASCADE_FOLLOWUP_AUDIO")
RESULTS_PATH = os.environ.get("SGLANG_OMNI_IMAGE_RESULTS")
pytestmark = [
    pytest.mark.accelerator,
    pytest.mark.skipif(
        not PORT or not AUDIO_PATH or not FOLLOWUP_PATH,
        reason="A running image cascade, English color question and handle follow-up WAV are required",
    ),
]


def cup_image(color: str) -> bytes:
    image = Image.new("RGB", (384, 384), "white")
    drawing = ImageDraw.Draw(image)
    drawing.ellipse((242, 123, 343, 249), fill=color)
    drawing.ellipse((259, 139, 326, 233), fill="white")
    drawing.rounded_rectangle((84, 111, 275, 302), radius=25, fill=color)
    drawing.ellipse((84, 94, 275, 140), fill=color, outline="black", width=3)
    drawing.ellipse((99, 105, 259, 130), fill="#f4f4f4")
    encoded = io.BytesIO()
    image.save(encoded, format="PNG")
    return encoded.getvalue()


def read_question(path: str) -> bytes:
    with wave.open(path, "rb") as source:
        assert (
            source.getnchannels(),
            source.getsampwidth(),
            source.getframerate(),
        ) == (1, 2, 16000)
        return source.readframes(source.getnframes())


@dataclass
class Answer:
    text: str
    transcript: str
    audio: bytes


class ImageConversation:
    def __init__(self, websocket: ClientConnection) -> None:
        self.websocket = websocket
        self.sequence = 0
        self.samples = 0
        self.event_count = 0
        self.events: list[JsonObject] = []

    async def send(self, event_type: str, **fields: JsonValue) -> None:
        self.event_count += 1
        await self.websocket.send(
            json.dumps(
                {"type": event_type, "event_id": f"image-{self.event_count}", **fields}
            )
        )

    async def receive(self) -> JsonObject:
        event = json.loads(await asyncio.wait_for(self.websocket.recv(), timeout=180))
        self.events.append(event)
        return event

    async def until(self, event_type: str) -> JsonObject:
        while True:
            event = await self.receive()
            if event["type"] == "error":
                raise RuntimeError(str(event))
            elif event["type"] == event_type:
                return event
            else:
                pass

    async def open(self, *, manual: bool = True, instructions: str = "") -> None:
        await self.until("session.created")
        settings: JsonObject = {"output_modalities": ["audio"]}
        if manual:
            settings["audio"] = {"input": {"turn_detection": None}}
        else:
            pass
        if instructions:
            settings["instructions"] = instructions
        else:
            pass
        await self.send("session.update", session=settings)
        event = await self.until("session.updated")
        assert "image" in event["session"]["sglang"]["granted"]["input_modalities"]

    async def image(self, image_bytes: bytes, acknowledge: bool = True) -> None:
        await self.send(
            "sglang.input_image.append",
            image=base64.b64encode(image_bytes).decode(),
            sglang={"t_ms": self.samples * 1000 / 16000},
        )
        if acknowledge:
            await self.until("sglang.input_image.accepted")
        else:
            pass

    async def append_audio(self, pcm: bytes, *, paced: bool = False) -> None:
        for offset in range(0, len(pcm), 2560):
            packet = pcm[offset : offset + 2560]
            await self.send(
                "input_audio_buffer.append",
                audio=base64.b64encode(packet).decode(),
                sglang={
                    "seq": self.sequence,
                    "t_start_ms": self.samples * 1000 / 16000,
                },
            )
            self.sequence += 1
            self.samples += len(packet) // 2
            if paced:
                await asyncio.sleep(len(packet) / 32000)
            else:
                pass

    async def question(self, pcm: bytes, *, manual: bool = True) -> None:
        await self.append_audio(pcm, paced=not manual)
        if manual:
            await self.send("input_audio_buffer.commit")
        else:
            await self.append_audio(bytes(32000), paced=True)

    async def answer(self) -> Answer:
        text = ""
        transcript = ""
        audio = bytearray()
        while True:
            event = await self.receive()
            event_type = event["type"]
            if event_type == "error":
                raise RuntimeError(str(event))
            elif event_type in (
                "response.output_text.delta",
                "response.output_audio_transcript.delta",
            ):
                text += event["delta"]
            elif event_type == "response.output_audio.delta":
                audio.extend(base64.b64decode(event["delta"]))
            elif event_type == "conversation.item.input_audio_transcription.completed":
                transcript = event["transcript"]
            elif event_type == "response.done":
                assert event["response"]["status"] == "completed", event
                assert text.strip() and transcript.strip() and len(audio) > 0
                assert any(audio)
                return Answer(text, transcript, bytes(audio))
            else:
                pass

    def record(self, name: str, answer: Answer) -> None:
        if RESULTS_PATH:
            directory = Path(RESULTS_PATH)
            directory.mkdir(parents=True, exist_ok=True)
            (directory / f"{name}.json").write_text(
                json.dumps(
                    {
                        "text": answer.text,
                        "transcript": answer.transcript,
                        "audio_bytes": len(answer.audio),
                        "events": self.events,
                    },
                    indent=2,
                )
            )
            with wave.open(str(directory / f"{name}.wav"), "wb") as output:
                output.setparams((1, 2, 24000, 0, "NONE", "not compressed"))
                output.writeframes(answer.audio)
        else:
            pass


@asynccontextmanager
async def conversation_on_server(
    *, manual: bool = True, instructions: str = ""
) -> AsyncIterator[ImageConversation]:
    async with connect(
        f"ws://127.0.0.1:{PORT}/v1/realtime", max_size=16 * 1024 * 1024
    ) as websocket:
        conversation = ImageConversation(websocket)
        await conversation.open(manual=manual, instructions=instructions)
        try:
            yield conversation
        finally:
            await conversation.send("session.close")
            await conversation.until("session.closed")


@pytest.mark.asyncio
@pytest.mark.parametrize("color", ["red", "blue"])
async def test_same_spoken_question_depends_on_image(color: str) -> None:
    async with conversation_on_server() as conversation:
        await conversation.image(cup_image(color))
        await conversation.question(read_question(AUDIO_PATH))
        answer = await conversation.answer()
        assert color in answer.text.lower()
        conversation.record(color, answer)


@pytest.mark.asyncio
async def test_next_image_does_not_change_the_committed_question_and_followup_retains_it() -> (
    None
):
    async with conversation_on_server() as conversation:
        await conversation.image(cup_image("red"))
        await conversation.question(read_question(AUDIO_PATH))
        await conversation.image(cup_image("blue"), acknowledge=False)
        first = await conversation.answer()
        assert "red" in first.text.lower()
        conversation.record("queued-red", first)
        await conversation.question(read_question(AUDIO_PATH))
        second = await conversation.answer()
        assert "blue" in second.text.lower()
        await conversation.question(read_question(FOLLOWUP_PATH))
        followup = await conversation.answer()
        assert "handle" in followup.text.lower() and "yes" in followup.text.lower()
        conversation.record("followup", followup)


@pytest.mark.asyncio
async def test_two_real_sessions_do_not_mix_images() -> None:
    async def ask(color: str) -> Answer:
        async with conversation_on_server() as conversation:
            await conversation.image(cup_image(color))
            await conversation.question(read_question(AUDIO_PATH))
            answer = await conversation.answer()
            conversation.record(f"concurrent-{color}", answer)
            return answer

    red, blue = await asyncio.gather(ask("red"), ask("blue"))
    assert "red" in red.text.lower() and "blue" not in red.text.lower()
    assert "blue" in blue.text.lower() and "red" not in blue.text.lower()


@pytest.mark.asyncio
async def test_missing_invalid_and_oversized_images_do_not_break_the_next_question() -> (
    None
):
    async with conversation_on_server() as conversation:
        await conversation.question(read_question(AUDIO_PATH))
        missing = await conversation.answer()
        assert "red" not in missing.text.lower() and "blue" not in missing.text.lower()
        conversation.record("missing", missing)
        for image_bytes in (b"\xff\xd8invalid", bytes(524289)):
            await conversation.image(image_bytes, acknowledge=False)
            while True:
                event = await conversation.receive()
                if event["type"] == "error":
                    assert event["sglang"]["fatal"] is False
                    break
                else:
                    pass
        await conversation.image(cup_image("red"))
        await conversation.question(read_question(AUDIO_PATH))
        recovered = await conversation.answer()
        assert "red" in recovered.text.lower()
        conversation.record("recovered", recovered)


@pytest.mark.asyncio
async def test_live_image_replacement_during_speech_applies_to_the_next_question() -> (
    None
):
    async with conversation_on_server(manual=False) as conversation:
        await conversation.append_audio(bytes(32000), paced=True)
        await conversation.image(cup_image("red"))
        sender = asyncio.create_task(
            conversation.question(read_question(AUDIO_PATH), manual=False)
        )
        try:
            await conversation.until("input_audio_buffer.speech_started")
            await conversation.image(cup_image("blue"))
            await sender
        finally:
            sender.cancel()
            await asyncio.gather(sender, return_exceptions=True)
        red = await conversation.answer()
        conversation.record("live-red", red)
        assert "red" in red.text.lower()
        await conversation.question(read_question(AUDIO_PATH), manual=False)
        blue = await conversation.answer()
        assert "blue" in blue.text.lower()
        conversation.record("live-blue", blue)
        await conversation.question(read_question(FOLLOWUP_PATH), manual=False)
        followup = await conversation.answer()
        assert "handle" in followup.text.lower() and "yes" in followup.text.lower()
        conversation.record("live-followup", followup)


@pytest.mark.asyncio
async def test_live_speech_interrupts_the_image_reply_and_next_image_is_answered() -> (
    None
):
    instructions = (
        "Answer in English. First state the cup's color in one short sentence. "
        "Then describe its shape and handle in at least four more sentences."
    )
    async with conversation_on_server(
        manual=False, instructions=instructions
    ) as conversation:
        await conversation.image(cup_image("red"))
        await conversation.question(read_question(AUDIO_PATH), manual=False)
        created = await conversation.until("response.created")
        interrupted_id = created["response"]["id"]
        await conversation.until("response.output_audio.delta")
        await conversation.image(cup_image("blue"))
        await conversation.question(read_question(AUDIO_PATH), manual=False)
        cancelled: JsonObject | None = None
        answer: Answer | None = None
        text = ""
        transcript = ""
        audio = bytearray()
        while cancelled is None or answer is None:
            event = await conversation.receive()
            if event["type"] == "error":
                raise RuntimeError(str(event))
            elif (
                event["type"] == "response.done"
                and event["response"]["id"] == interrupted_id
            ):
                cancelled = event["response"]
            elif event["type"] == "response.created":
                text = ""
                audio.clear()
            elif (
                event["type"] == "response.output_audio_transcript.delta"
                and event["response_id"] != interrupted_id
            ):
                text += event["delta"]
            elif (
                event["type"] == "response.output_audio.delta"
                and event["response_id"] != interrupted_id
            ):
                audio.extend(base64.b64decode(event["delta"]))
            elif (
                event["type"] == "conversation.item.input_audio_transcription.completed"
            ):
                transcript = event["transcript"]
            elif (
                event["type"] == "response.done"
                and event["response"]["id"] != interrupted_id
            ):
                assert event["response"]["status"] == "completed"
                answer = Answer(text, transcript, bytes(audio))
            else:
                pass
        assert cancelled["status"] == "cancelled"
        assert cancelled["status_details"]["reason"] == "interrupted"
        assert "blue" in answer.text.lower() and answer.transcript and any(answer.audio)
        conversation.record("live-barge-in", answer)


ALL_PICTURES_PATH = os.environ.get("SGLANG_OMNI_CASCADE_ALL_PICTURES_AUDIO")
FIRST_PICTURE_PATH = os.environ.get("SGLANG_OMNI_CASCADE_FIRST_PICTURE_AUDIO")


@pytest.mark.asyncio
@pytest.mark.skipif(
    not ALL_PICTURES_PATH,
    reason="A question asking for the cup colors in all pictures is required",
)
async def test_images_uploaded_together_reach_one_question() -> None:
    async with conversation_on_server() as conversation:
        await conversation.image(cup_image("red"))
        await conversation.image(cup_image("blue"))
        await conversation.question(read_question(ALL_PICTURES_PATH))
        answer = await conversation.answer()
        assert "red" in answer.text.lower() and "blue" in answer.text.lower()
        conversation.record("together", answer)


@pytest.mark.asyncio
@pytest.mark.skipif(
    not FIRST_PICTURE_PATH,
    reason="A question asking for the cup color in the first picture is required",
)
@pytest.mark.parametrize("colors", [("red", "blue"), ("blue", "red")])
async def test_earlier_picture_stays_visible_after_a_new_upload(
    colors: tuple[str, str],
) -> None:
    async with conversation_on_server() as conversation:
        spoken = ""
        for color in colors:
            await conversation.image(cup_image(color))
            await conversation.question(read_question(FOLLOWUP_PATH))
            spoken += (await conversation.answer()).text.lower()
        assert "red" not in spoken and "blue" not in spoken
        await conversation.question(read_question(FIRST_PICTURE_PATH))
        answer = await conversation.answer()
        assert colors[0] in answer.text.lower()
        assert colors[1] not in answer.text.lower()
        conversation.record(f"remembered-{colors[0]}", answer)
