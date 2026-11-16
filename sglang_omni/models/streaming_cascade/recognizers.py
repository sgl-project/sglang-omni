# SPDX-License-Identifier: Apache-2.0
"""Recognizer kinds: utterance transcripts from the session audio."""

import asyncio
import base64
import json
import re
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from typing import ClassVar, Literal
from urllib.parse import urlencode

import httpx
from pydantic import BaseModel, Field
from websockets.asyncio.client import connect
from websockets.exceptions import ConnectionClosed

from sglang_omni.audio.segmentation import (
    EMPTY_UTTERANCE,
    FINISH_UTTERANCE,
    OPEN_UTTERANCE,
    UtteranceSegmenter,
)
from sglang_omni.audio.wav import pcm16_wav
from sglang_omni.models.streaming_cascade.base import (
    ComponentConfig,
    StreamComponent,
    audio_bytes,
    bearer_headers,
    iterate,
    probe_models,
    stop_running,
    transcript_chunk,
    with_language,
)
from sglang_omni.proto.session import TimedChunk
from sglang_omni.proto.streams import SESSION_INPUT_SAMPLE_RATE_HZ
from sglang_omni.scheduling.session import ChunkEmitter


class OpenUtterance:
    """One utterance's audio flowing through a recognizer's `utterance()`."""

    def __init__(
        self,
        transcribe: Callable[[AsyncIterator[TimedChunk]], AsyncIterator[TimedChunk]],
        emit: ChunkEmitter,
    ) -> None:
        self.audio: asyncio.Queue[TimedChunk | None] = asyncio.Queue()
        self.task = asyncio.ensure_future(self.pump(transcribe, emit))

    async def pump(
        self,
        transcribe: Callable[[AsyncIterator[TimedChunk]], AsyncIterator[TimedChunk]],
        emit: ChunkEmitter,
    ) -> None:
        async for transcript in transcribe(iterate(self.audio)):
            emit(transcript)

    def feed(self, chunk: TimedChunk) -> None:
        if self.task.done():
            self.task.result()
            raise RuntimeError("the recognizer ended before its utterance")
        else:
            self.audio.put_nowait(chunk)

    async def finish(self) -> None:
        self.audio.put_nowait(None)
        await self.task

    async def cancel(self) -> None:
        self.task.cancel()
        await asyncio.wait({self.task})


class UtteranceRecognizer(StreamComponent):
    """Runs `utterance()` once per utterance of the session's audio.

    `utterance()` receives the audio of one utterance and yields transcripts,
    the last one final. A commit with no audio yields an empty final
    transcript, so the dialog answers it with turn.empty.
    """

    def __init__(self) -> None:
        self.segmenter = UtteranceSegmenter()

    def utterance(self, audio: AsyncIterator[TimedChunk]) -> AsyncIterator[TimedChunk]:
        raise NotImplementedError

    async def process(
        self, inputs: AsyncIterator[TimedChunk], emit: ChunkEmitter
    ) -> None:
        current: OpenUtterance | None = None
        try:
            async for chunk in inputs:
                for action in self.segmenter.step(chunk):
                    current = await self.act(action, current, emit)
            for action in self.segmenter.finish():
                current = await self.act(action, current, emit)
        finally:
            if current is not None:
                await current.cancel()
            else:
                pass

    async def act(
        self,
        action: TimedChunk | str,
        current: OpenUtterance | None,
        emit: ChunkEmitter,
    ) -> OpenUtterance | None:
        if action == OPEN_UTTERANCE:
            return OpenUtterance(self.utterance, emit)
        elif action == EMPTY_UTTERANCE:
            emit(transcript_chunk("", True))
            return current
        elif current is None:
            raise RuntimeError(f"recognizer received {action!r} outside an utterance")
        elif action == FINISH_UTTERANCE:
            await current.finish()
            return None
        else:
            current.feed(action)
            return current

    def adopt(self, previous: StreamComponent) -> None:
        if isinstance(previous, UtteranceRecognizer):
            self.segmenter = previous.segmenter
        else:
            pass


class OpenAITranscriptionRecognizerConfig(ComponentConfig):
    """Whole-utterance `/v1/audio/transcriptions`; one final transcript per utterance.

    With `verbose_json` the transcript carries the language the service
    reports for the utterance; `json` suits a service without it.
    """

    kind: Literal["openai_transcription"] = "openai_transcription"
    base_url: str = Field(pattern=r"^https?://[^/]+")
    model: str
    language: str = ""
    response_format: Literal["json", "verbose_json"] = "verbose_json"
    api_key_env: str = "OPENAI_API_KEY"


class TranscriptionResult(BaseModel):
    text: str
    language: str | None = None


async def request_transcription(
    client: httpx.AsyncClient,
    base_url: str,
    model: str,
    pcm16: bytes,
    sample_rate_hz: int,
    *,
    language: str | None,
    response_format: Literal["json", "verbose_json"],
) -> httpx.Response:
    """POST signed 16-bit PCM as a WAV file to the `/audio/transcriptions` endpoint of the OpenAI-compatible service at `base_url`."""
    return await client.post(
        f"{base_url.rstrip('/')}/audio/transcriptions",
        data=with_language(
            {"model": model, "response_format": response_format}, language
        ),
        files={"file": ("audio.wav", pcm16_wav(pcm16, sample_rate_hz), "audio/wav")},
    )


class RealtimeTranscriptionRecognizerConfig(ComponentConfig):
    """Streaming transcription over an sglang-omni native `/v1/realtime` deployment."""

    kind: Literal["realtime_transcription"] = "realtime_transcription"
    base_url: str = Field(pattern=r"^wss?://[^/]+")
    model: str
    api_key_env: str = "OPENAI_API_KEY"


class StreamingTranscriptionRecognizerConfig(ComponentConfig):
    """Streaming transcription over an sglang-omni /v1/realtime?intent=transcription deployment.

    The node already cuts utterances, so each session runs without server VAD. decode_interval_ms is the audio between partial transcripts; None keeps the service default, and the service enforces its own minimum.
    """

    kind: Literal["streaming_transcription"] = "streaming_transcription"
    base_url: str = Field(pattern=r"^wss?://[^/]+")
    model: str
    language: str = ""
    decode_interval_ms: int | None = Field(default=None, gt=0)
    api_key_env: str = "OPENAI_API_KEY"


async def probe_realtime_service(
    config: (
        RealtimeTranscriptionRecognizerConfig | StreamingTranscriptionRecognizerConfig
    ),
) -> None:
    http_url = re.sub(r"^ws", "http", config.base_url.rstrip("/"))
    await probe_models(f"{http_url}/v1/models", config.timeout_s, config.api_key_env)


class OpenAITranscriptionRecognizer(UtteranceRecognizer):
    """Buffers each utterance and transcribes it once the utterance ends."""

    def __init__(self, config: OpenAITranscriptionRecognizerConfig) -> None:
        super().__init__()
        self.config = config
        self.client = httpx.AsyncClient(
            timeout=config.timeout_s, headers=bearer_headers(config.api_key_env)
        )

    async def utterance(
        self, audio: AsyncIterator[TimedChunk]
    ) -> AsyncIterator[TimedChunk]:
        pcm = bytearray()
        async for chunk in audio:
            pcm.extend(audio_bytes(chunk))
        if not pcm:
            yield transcript_chunk("", True)
            return
        else:
            pass
        response = await request_transcription(
            self.client,
            self.config.base_url,
            self.config.model,
            bytes(pcm),
            SESSION_INPUT_SAMPLE_RATE_HZ,
            language=self.config.language or None,
            response_format=self.config.response_format,
        )
        response.raise_for_status()
        result = TranscriptionResult.model_validate(response.json())
        yield transcript_chunk(result.text.strip(), True, result.language or None)

    async def close(self) -> None:
        await self.client.aclose()


class RealtimeTextDelta(BaseModel):
    type: Literal["response.output_text.delta"]
    delta: str


class RealtimeTextDone(BaseModel):
    type: Literal["response.output_text.done"]
    text: str


class RealtimeErrorDetail(BaseModel):
    message: str


class RealtimeError(BaseModel):
    type: Literal["error"]
    error: RealtimeErrorDetail


class TranscriptionSegmentEvent(BaseModel):
    type: Literal["transcription.segment"]
    text: str
    is_final: bool


class TranscriptionCompletedEvent(BaseModel):
    type: Literal["transcription.completed"]
    text: str
    language: str | None = None


@dataclass
class RealtimeExchange:
    """One utterance's WebSocket: the pending receive, the audio sender once configured, and the transcript so far."""

    receive: asyncio.Task[bytes | str]
    sender: asyncio.Task[None] | None = None
    is_configured: bool = False
    text: str = ""
    final_text: str | None = None
    is_finished: bool = False


class RealtimeWebSocketRecognizer(UtteranceRecognizer):
    """Base class for recognizers"""

    intent_query: ClassVar[dict[str, str]]
    audio_end_event: ClassVar[str]

    def __init__(
        self,
        config: (
            RealtimeTranscriptionRecognizerConfig
            | StreamingTranscriptionRecognizerConfig
        ),
    ) -> None:
        super().__init__()
        self.config = config

    async def utterance(
        self, audio: AsyncIterator[TimedChunk]
    ) -> AsyncIterator[TimedChunk]:
        deadline = asyncio.get_running_loop().time() + self.config.timeout_s
        query = urlencode({**self.intent_query, "model": self.config.model})
        uri = f"{self.config.base_url.rstrip('/')}/v1/realtime?{query}"
        async with connect(
            uri,
            additional_headers=bearer_headers(self.config.api_key_env),
            open_timeout=self.config.timeout_s,
            close_timeout=2,
            max_size=4 * 1024 * 1024,
        ) as websocket:
            exchange = RealtimeExchange(asyncio.ensure_future(websocket.recv()))
            try:
                while not exchange.is_finished:
                    event = await self.next_event(websocket, exchange, deadline)
                    for transcript in await self.handle_event(
                        websocket, exchange, event, audio
                    ):
                        yield transcript
            finally:
                await stop_running((exchange.receive, exchange.sender))

    async def next_event(
        self, websocket: object, exchange: RealtimeExchange, deadline: float
    ) -> dict[str, object]:
        """The next server event while the audio sender runs; a close after the final transcript finishes the exchange."""
        loop = asyncio.get_running_loop()
        while not exchange.receive.done():
            waiting = {exchange.receive}
            if exchange.sender is not None:
                waiting.add(exchange.sender)
            else:
                pass
            await asyncio.wait(
                waiting,
                timeout=max(deadline - loop.time(), 0),
                return_when=asyncio.FIRST_COMPLETED,
            )
            if exchange.sender is not None and exchange.sender.done():
                exchange.sender.result()
                exchange.sender = None
            elif not exchange.receive.done() and loop.time() >= deadline:
                raise TimeoutError("realtime transcription timed out")
            else:
                pass
        try:
            encoded = exchange.receive.result()
        except ConnectionClosed:
            if exchange.final_text is None:
                raise RuntimeError("closed before the final transcript")
            else:
                return {"type": "session.closed"}
        exchange.receive = asyncio.ensure_future(websocket.recv())
        event = json.loads(encoded)
        if not isinstance(event, dict) or not isinstance(event.get("type"), str):
            raise ValueError("realtime events must be typed JSON objects")
        else:
            return event

    async def handle_event(
        self,
        websocket: object,
        exchange: RealtimeExchange,
        event: dict[str, object],
        audio: AsyncIterator[TimedChunk],
    ) -> list[TimedChunk]:
        """Act on one server event; the transcripts it carries."""
        raise NotImplementedError

    async def send_event(
        self, websocket: object, event_id: str, event_type: str, **fields: object
    ) -> None:
        await websocket.send(
            json.dumps({"event_id": event_id, "type": event_type, **fields})
        )

    async def send_audio(
        self, websocket: object, chunks: AsyncIterator[TimedChunk]
    ) -> None:
        seq = 0
        async for chunk in chunks:
            await self.send_event(
                websocket,
                f"audio-{seq}",
                "input_audio_buffer.append",
                audio=base64.b64encode(audio_bytes(chunk)).decode(),
                sglang={"seq": seq},
            )
            seq += 1
        await self.send_event(websocket, "audio-end", self.audio_end_event)


class RealtimeTranscriptionRecognizer(RealtimeWebSocketRecognizer):
    """Native /v1/realtime streaming transcription; one WebSocket per utterance."""

    intent_query = {}
    audio_end_event = "sglang.input_audio.end"

    async def handle_event(
        self,
        websocket: object,
        exchange: RealtimeExchange,
        event: dict[str, object],
        audio: AsyncIterator[TimedChunk],
    ) -> list[TimedChunk]:
        event_type = event["type"]
        if event_type == "error":
            message = RealtimeError.model_validate(event).error.message
            raise RuntimeError(f"transcription failed: {message}")
        elif event_type == "session.created":
            await self.send_event(
                websocket,
                "session-update",
                "session.update",
                session={"output_modalities": ["text"]},
            )
        elif event_type == "session.updated" and not exchange.is_configured:
            exchange.is_configured = True
            exchange.sender = asyncio.ensure_future(self.send_audio(websocket, audio))
        elif event_type == "response.output_text.delta":
            exchange.text += RealtimeTextDelta.model_validate(event).delta
            return [transcript_chunk(exchange.text, False)]
        elif event_type == "response.output_text.done":
            exchange.final_text = RealtimeTextDone.model_validate(event).text
            return [transcript_chunk(exchange.final_text, True)]
        elif event_type == "sglang.input_audio.drained" and exchange.final_text is None:
            raise RuntimeError("drained before the final transcript")
        elif event_type == "sglang.input_audio.drained":
            await self.send_event(websocket, "session-close", "session.close")
        elif event_type == "session.closed" and exchange.final_text is None:
            raise RuntimeError("closed before the final transcript")
        elif event_type == "session.closed":
            exchange.is_finished = True
        else:
            pass
        return []


class StreamingTranscriptionRecognizer(RealtimeWebSocketRecognizer):

    intent_query = {"intent": "transcription"}
    audio_end_event = "transcription.done"

    async def handle_event(
        self,
        websocket: object,
        exchange: RealtimeExchange,
        event: dict[str, object],
        audio: AsyncIterator[TimedChunk],
    ) -> list[TimedChunk]:
        event_type = event["type"]
        if event_type == "error":
            message = RealtimeError.model_validate(event).error.message
            raise RuntimeError(f"transcription failed: {message}")
        elif event_type == "session.created":
            session: dict[str, str | int | None] = {"turn_detection": None}
            if self.config.language:
                session["language"] = self.config.language
            else:
                pass
            if self.config.decode_interval_ms is not None:
                session["decode_interval_ms"] = self.config.decode_interval_ms
            else:
                pass
            await self.send_event(
                websocket, "session-update", "session.update", session=session
            )
        elif event_type == "session.updated" and not exchange.is_configured:
            exchange.is_configured = True
            exchange.sender = asyncio.ensure_future(self.send_audio(websocket, audio))
        elif event_type == "transcription.segment":
            segment = TranscriptionSegmentEvent.model_validate(event)
            if not segment.is_final:
                return [transcript_chunk(segment.text, False)]
            else:
                pass
        elif event_type == "transcription.completed":
            completed = TranscriptionCompletedEvent.model_validate(event)
            exchange.final_text = completed.text
            exchange.is_finished = True
            return [transcript_chunk(completed.text.strip(), True, completed.language)]
        else:
            pass
        return []
