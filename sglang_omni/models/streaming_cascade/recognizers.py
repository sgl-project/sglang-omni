# SPDX-License-Identifier: Apache-2.0
"""Recognizer kinds: utterance transcripts from the session audio."""

import asyncio
import base64
import json
import re
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, replace
from typing import ClassVar, Literal
from urllib.parse import urlencode

import httpx
import numpy as np
from pydantic import BaseModel, Field
from scipy.optimize import linear_sum_assignment
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
    SpeakerSegment,
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
from sglang_omni.proto.streams import (
    PCM16_BYTES_PER_SAMPLE,
    SESSION_INPUT_SAMPLE_RATE_HZ,
)
from sglang_omni.scheduling.session import ChunkEmitter

PCM16_BYTES_PER_MS = PCM16_BYTES_PER_SAMPLE * SESSION_INPUT_SAMPLE_RATE_HZ // 1000
# One [start][Sxx]text[end] segment of MOSS-Transcribe-Diarize, times in seconds.
SPEAKER_SEGMENT = re.compile(
    r"\[(\d+(?:[.,]\d+)?)\]\s*\[(S\d+)\](.*?)\[(\d+(?:[.,]\d+)?)\]", re.DOTALL
)
MARKUP_TAG = re.compile(r"\[[^\[\]]*\]")
SPEAKER_TAG = re.compile(r"\[(S\d+)\]")
# note (fweilun): the silence between the recent utterances and the new one, as measured on movies800time; the model still joins one speaker's words across it at times, which a segment over STRADDLE_MS on both sides of it shows.
CONTEXT_SEPARATOR_MS = 600
STRADDLE_MS = 300


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


class SpeakerTranscriptionRecognizerConfig(ComponentConfig):
    """Speaker-attributed transcripts from the /audio/transcriptions endpoint of a MOSS-Transcribe-Diarize service; one final transcript per utterance.

    Each utterance is transcribed after the last context_s of the session's speech, so a speaker heard in it keeps one label; a speaker it does not hold gets a new label, and 0 transcribes each utterance alone.
    """

    kind: Literal["speaker_transcription"] = "speaker_transcription"
    base_url: str = Field(pattern=r"^https?://[^/]+")
    model: str
    context_s: float = Field(default=20, ge=0)
    api_key_env: str = "OPENAI_API_KEY"


def seconds_ms(seconds: str) -> float:
    return float(round(float(seconds.replace(",", ".")) * 1000))


def spoken_words(text: str) -> str:
    return " ".join(MARKUP_TAG.sub(" ", text).split())


def parse_speaker_segments(markup: str, duration_ms: float) -> list[SpeakerSegment]:
    """The segments of a MOSS-Transcribe-Diarize transcript; words without time markers are one segment over the whole audio, of the first speaker they name."""
    segments: list[SpeakerSegment] = []
    for start, speaker, text, end in SPEAKER_SEGMENT.findall(markup):
        words = spoken_words(text)
        if words:
            start_ms = seconds_ms(start)
            segments.append(
                SpeakerSegment(
                    speaker=speaker,
                    start_ms=start_ms,
                    end_ms=max(seconds_ms(end), start_ms),
                    text=words,
                )
            )
        else:
            pass
    words = spoken_words(markup)
    if segments or not words:
        return segments
    else:
        named = SPEAKER_TAG.search(markup)
        return [
            SpeakerSegment(
                speaker="" if named is None else named.group(1),
                start_ms=0,
                end_ms=duration_ms,
                text=words,
            )
        ]


def shifted(segment: SpeakerSegment, delta_ms: float) -> SpeakerSegment:
    return replace(
        segment,
        start_ms=segment.start_ms + delta_ms,
        end_ms=segment.end_ms + delta_ms,
    )


def match_speakers(
    known: list[SpeakerSegment], decoded: list[SpeakerSegment]
) -> dict[str, str]:
    """Each decoded label paired one to one with the known label it overlaps longest in all; a label that overlaps none stays unpaired."""
    decoded_labels = sorted({segment.speaker for segment in decoded})
    known_labels = sorted({segment.speaker for segment in known})
    overlap_ms = np.zeros((len(decoded_labels), len(known_labels)))
    for decoded_segment in decoded:
        for known_segment in known:
            overlap_ms[
                decoded_labels.index(decoded_segment.speaker),
                known_labels.index(known_segment.speaker),
            ] += max(
                0.0,
                min(decoded_segment.end_ms, known_segment.end_ms)
                - max(decoded_segment.start_ms, known_segment.start_ms),
            )
    rows, columns = linear_sum_assignment(overlap_ms, maximize=True)
    return {
        decoded_labels[row]: known_labels[column]
        for row, column in zip(rows, columns)
        if overlap_ms[row, column] > 0
    }


def speaker_lines(segments: list[SpeakerSegment]) -> str:
    """One "[S01] words" line per run of one speaker's segments."""
    lines: list[tuple[str, list[str]]] = []
    for segment in segments:
        if lines and lines[-1][0] == segment.speaker:
            lines[-1][1].append(segment.text)
        else:
            lines.append((segment.speaker, [segment.text]))
    return "\n".join(f"[{speaker}] {' '.join(words)}" for speaker, words in lines)


class SpeakerTranscriptionRecognizer(UtteranceRecognizer):
    """Transcribes each finished utterance with speaker labels that hold for the whole session."""

    def __init__(self, config: SpeakerTranscriptionRecognizerConfig) -> None:
        super().__init__()
        self.config = config
        self.client = httpx.AsyncClient(
            timeout=config.timeout_s, headers=bearer_headers(config.api_key_env)
        )
        self.speaker_count = 0
        # The last context_s of the session's speech, utterances apart by CONTEXT_SEPARATOR_MS of silence, and its segments with session labels on its own clock.
        self.context_pcm = b""
        self.context_segments: list[SpeakerSegment] = []

    def adopt(self, previous: StreamComponent) -> None:
        super().adopt(previous)
        if isinstance(previous, SpeakerTranscriptionRecognizer):
            self.speaker_count = previous.speaker_count
            self.context_pcm = previous.context_pcm
            self.context_segments = previous.context_segments
        else:
            pass

    def state_bytes(self) -> int:
        return len(self.context_pcm) + 4 * sum(
            len(segment.text) for segment in self.context_segments
        )

    async def transcribe(self, pcm: bytes) -> list[SpeakerSegment]:
        response = await request_transcription(
            self.client,
            self.config.base_url,
            self.config.model,
            pcm,
            SESSION_INPUT_SAMPLE_RATE_HZ,
            language=None,
            response_format="json",
        )
        response.raise_for_status()
        result = TranscriptionResult.model_validate(response.json())
        return parse_speaker_segments(result.text, len(pcm) / PCM16_BYTES_PER_MS)

    async def utterance(
        self, audio: AsyncIterator[TimedChunk]
    ) -> AsyncIterator[TimedChunk]:
        chunks = [chunk async for chunk in audio]
        if not chunks:
            yield transcript_chunk("", True)
            return
        else:
            pass
        segments = await self.attribute(
            b"".join(audio_bytes(chunk) for chunk in chunks)
        )
        yield transcript_chunk(
            speaker_lines(segments),
            True,
            segments=[shifted(segment, chunks[0].t_start_ms) for segment in segments],
        )

    async def attribute(self, pcm: bytes) -> list[SpeakerSegment]:
        """The utterance's segments with session labels on its own clock; the utterance then joins the context."""
        separator = (
            bytes(CONTEXT_SEPARATOR_MS * PCM16_BYTES_PER_MS)
            if self.context_pcm
            else b""
        )
        joined = self.context_pcm + separator + pcm
        offset_ms = (len(joined) - len(pcm)) / PCM16_BYTES_PER_MS
        if not self.context_pcm:
            utterance_segments, labels = await self.transcribe(pcm), {}
        else:
            joined_segments = await self.transcribe(joined)
            joined_to_session = match_speakers(self.context_segments, joined_segments)
            joined_utterance_segments = [
                shifted(segment, -offset_ms)
                for segment in joined_segments
                if (segment.start_ms + segment.end_ms) / 2 >= offset_ms
            ]
            straddles = any(
                segment.start_ms < offset_ms - STRADDLE_MS
                and segment.end_ms > offset_ms + STRADDLE_MS
                for segment in joined_segments
            )
            # note (fweilun): after the context the model may join the utterance's first words to the last context segment, or skip a short utterance; the utterance alone then gives the words, and the joined transcription their speakers.
            if joined_utterance_segments and not straddles:
                utterance_segments, labels = (
                    joined_utterance_segments,
                    joined_to_session,
                )
            else:
                utterance_segments = await self.transcribe(pcm)
                joined_on_utterance_clock = [
                    shifted(segment, -offset_ms) for segment in joined_segments
                ]
                labels = {
                    label: joined_to_session[joined_label]
                    for label, joined_label in match_speakers(
                        joined_on_utterance_clock, utterance_segments
                    ).items()
                    if joined_label in joined_to_session
                }
        for label in dict.fromkeys(segment.speaker for segment in utterance_segments):
            if label not in labels:
                self.speaker_count += 1
                labels[label] = f"S{self.speaker_count:02d}"
            else:
                pass
        segments = [
            replace(segment, speaker=labels[segment.speaker])
            for segment in utterance_segments
        ]
        cut_bytes = max(
            len(joined) - int(self.config.context_s * 1000) * PCM16_BYTES_PER_MS, 0
        )
        cut_ms = cut_bytes / PCM16_BYTES_PER_MS
        self.context_pcm = joined[cut_bytes:]
        self.context_segments = [
            shifted(segment, -cut_ms)
            for segment in [
                *self.context_segments,
                *(shifted(segment, offset_ms) for segment in segments),
            ]
            if segment.end_ms > cut_ms
        ]
        return segments

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
