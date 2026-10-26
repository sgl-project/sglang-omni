# SPDX-License-Identifier: Apache-2.0
"""The alignment kind: where in the reply text the reply audio is, as `reply.progress` controls."""

import asyncio
import base64
import logging
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Literal

import httpx
import numpy as np
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from sglang_omni.audio.spoken_prefix import (
    TRAILING_PUNCTUATION,
    NormalizedUnit,
    matched_prefix_end,
    normalized_units,
    spoken_prefix_end,
)
from sglang_omni.models.streaming_cascade.base import (
    PROBE_TIMEOUT_S,
    SPEECH_SAMPLE_RATE_HZ,
    ComponentConfig,
    StreamComponent,
    control_chunk,
    iterate,
    probe_models,
    stop_running,
    with_language,
)
from sglang_omni.models.streaming_cascade.numerals import spoken_text
from sglang_omni.models.streaming_cascade.recognizers import (
    TranscriptionResult,
    request_transcription,
)
from sglang_omni.proto.session import TimedChunk
from sglang_omni.proto.streams import (
    PCM_F32,
    REPLY_DELIVERED,
    REPLY_PROGRESS,
    TEXT_DELTA,
    TRUNCATE,
    chunk_language,
    control_type,
)
from sglang_omni.scheduling.session import ChunkEmitter

logger = logging.getLogger(__name__)

MAX_ALIGNMENT_AUDIO_BYTES = 32 * 1024 * 1024
WARM_UP_TEXT = "hello"
WARM_UP_SAMPLES = np.zeros(SPEECH_SAMPLE_RATE_HZ, dtype=np.float32)


def spoken_units(text: str, language: str) -> list[NormalizedUnit]:
    """Expanded readings with original offsets and complete source-unit boundaries."""
    complete_ends_chars = {unit.end_chars for unit in normalized_units(text)}
    spoken, origin = spoken_text(text, language)
    units: list[NormalizedUnit] = []
    for unit in normalized_units(spoken):
        original_end_chars = origin[unit.end_chars - 1] + 1
        next_start_chars = unit.end_chars
        while next_start_chars < len(spoken) and spoken[next_start_chars].isspace():
            next_start_chars += 1
        units.append(
            NormalizedUnit(
                unit.text,
                original_end_chars,
                has_joiner=unit.has_joiner,
                is_complete=original_end_chars in complete_ends_chars
                and (
                    next_start_chars == len(origin)
                    or origin[next_start_chars] + 1 != original_end_chars
                ),
            )
        )
    return units


class AlignmentBoundary(BaseModel):
    """The first `text_end_chars` characters of the request text end `audio_end_samples` into its audio."""

    model_config = ConfigDict(extra="forbid", strict=True)
    text_end_chars: int = Field(gt=0)
    audio_end_samples: int = Field(ge=0)


class AlignmentResult(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    boundaries: list[AlignmentBoundary]


class AlignmentConfig(ComponentConfig):
    """Reply progress for truncating an interrupted reply.

    `sentence` passes the synthesizer's sentence ends through; `forced` also
    asks the worker at `alignment_url` for the word ends of each sentence
    once its audio is complete; `asr` passes the sentence ends through and,
    on a cut, has the OpenAI-compatible transcription service at
    `heard_asr_url` recognize the audio of the sentence at the cut, so the
    part of its text the transcript says counts as heard. `forced` with
    `heard_asr_url` does the same for a sentence that has no word end at or
    after the cut, such as one an interruption cut off before its audio was
    complete.
    """

    kind: Literal["alignment"] = "alignment"
    method: Literal["sentence", "forced", "asr"] = "sentence"
    alignment_url: str = Field(default="", pattern=r"^(https?://[^/]+.*)?$")
    heard_asr_url: str = Field(default="", pattern=r"^(https?://[^/]+.*)?$")
    heard_asr_model: str = ""
    timeout_s: float = Field(default=2, gt=0)

    @model_validator(mode="after")
    def check_worker(self) -> "AlignmentConfig":
        if self.method == "forced" and not self.alignment_url:
            raise ValueError("the forced method needs alignment_url")
        elif self.method != "forced" and self.alignment_url:
            raise ValueError("alignment_url belongs to the forced method")
        elif bool(self.heard_asr_url) != bool(self.heard_asr_model):
            raise ValueError("heard_asr_url and heard_asr_model go together")
        elif self.method == "asr" and not self.heard_asr_url:
            raise ValueError("the asr method needs heard_asr_url and heard_asr_model")
        elif self.method == "sentence" and self.heard_asr_url:
            raise ValueError("heard_asr_url belongs to the asr and forced methods")
        else:
            return self

    def hears(self) -> bool:
        """Whether a transcription service recognizes the audio of interrupted sentences."""
        return bool(self.heard_asr_url)


async def probe_alignment_worker(config: AlignmentConfig) -> None:
    if config.method == "forced":
        worker_url = config.alignment_url.rstrip("/")
        await probe_models(f"{worker_url}/health", config.timeout_s, "")
        # Note (ruoyu): The worker's first request after startup ran past a sentence's timeout_s of 2 s on a cold container, and took 0.45 s on a restarted worker, so the first reply of a fresh deployment had no word ends; one alignment here pays that before the node reports ready.
        async with httpx.AsyncClient(timeout=PROBE_TIMEOUT_S) as client:
            try:
                response = await client.post(
                    f"{worker_url}/align",
                    json=alignment_request(
                        WARM_UP_TEXT, pcm16_bytes(WARM_UP_SAMPLES), None
                    ),
                )
            except httpx.TransportError as error:
                raise RuntimeError(
                    f"cannot reach {worker_url}/align: {error!r}"
                ) from error
        if response.status_code != 200:
            raise RuntimeError(f"{worker_url}/align answered {response.status_code}")
        else:
            pass
    else:
        pass
    if config.hears():
        await probe_models(
            f"{config.heard_asr_url.rstrip('/')}/models", config.timeout_s, ""
        )
    else:
        pass


@dataclass(frozen=True)
class SentenceAudio:
    """A spoken sentence of `turn` for the forced aligner: `text` begins at character `start` of the reply and its audio at `start_ms`; `language` is the reply's, None while unknown."""

    turn: int
    text: str
    start: int
    start_ms: float
    samples: np.ndarray
    language: str | None = None


@dataclass(frozen=True)
class HeardAudio:
    """The audio of one sentence of `turn` from its start to the cut at `audio_ms`, for the transcription service; `text` is the sentence's text from character `start` of the reply on."""

    turn: int
    text: str
    start: int
    audio_ms: float
    samples: np.ndarray
    language: str | None = None


@dataclass
class SpokenReply:
    """One reply's text, language and audio; `start` and `start_ms` open the sentence being spoken.

    With a transcription service the reply also keeps its ended sentences,
    the word ends the forced aligner found, the cut positions the client's
    truncate and the delivered audio report, and the cut last recognized.
    """

    turn: int
    text: str = ""
    language: str | None = None
    start: int = 0
    start_ms: float = 0.0
    samples: list[np.ndarray] = field(default_factory=list)
    sentences: list[SentenceAudio] = field(default_factory=list)
    word_ends: list[tuple[int, float]] = field(default_factory=list)
    client_ms: float | None = None
    delivered_ms: float | None = None
    heard_ms: float | None = None


class Alignment(StreamComponent):
    """Turns the reply text, the reply audio and the synthesizer's sentence ends into `reply.progress` controls.

    The synthesizer's `reply.progress` ends a sentence. A sentence cut off by
    an interruption never ends, so it reports nothing and counts as unheard,
    unless a transcription service hears it: the cut is the client's
    `truncate` position, never past the audio the graph delivered
    (`reply.delivered`), as in the dialog; the audio of the sentence at the
    cut, from its start to the cut, is recognized when no word end at or
    after the cut covers it, and the strict prefix of the sentence's text
    the transcript says ends at the cut.
    """

    def __init__(self, config: AlignmentConfig) -> None:
        self.config = config
        self.reply: SpokenReply | None = None
        self.client = httpx.AsyncClient(timeout=config.timeout_s)

    def adopt(self, previous: StreamComponent) -> None:
        if isinstance(previous, Alignment):
            self.reply = previous.reply
        else:
            pass

    def state_bytes(self) -> int:
        if self.reply is None:
            return 0
        else:
            return (
                len(self.reply.text) * 4
                + sum(samples.nbytes for samples in self.reply.samples)
                + sum(sentence.samples.nbytes for sentence in self.reply.sentences)
            )

    def reply_of(self, turn: int) -> SpokenReply:
        if self.reply is None or self.reply.turn != turn:
            self.reply = SpokenReply(turn)
        else:
            pass
        return self.reply

    async def process(
        self, inputs: AsyncIterator[TimedChunk], emit: ChunkEmitter
    ) -> None:
        sentences: asyncio.Queue[SentenceAudio | HeardAudio | None] = asyncio.Queue()
        worker = (
            asyncio.ensure_future(self.align_sentences(iterate(sentences), emit))
            if self.keeps_audio()
            else None
        )
        try:
            async for chunk in inputs:
                self.handle(chunk, emit, sentences)
            if worker is not None:
                sentences.put_nowait(None)
                await worker
            else:
                pass
        finally:
            await stop_running([worker])

    def keeps_audio(self) -> bool:
        return self.config.method == "forced" or self.config.hears()

    def handle(
        self,
        chunk: TimedChunk,
        emit: ChunkEmitter,
        sentences: asyncio.Queue[SentenceAudio | HeardAudio | None],
    ) -> None:
        event = control_type(chunk)
        if chunk.format == TEXT_DELTA:
            reply = self.reply_of(chunk.turn)
            reply.text += str(chunk.payload["text"])
            language = chunk_language(chunk)
            if language is not None:
                reply.language = language
            else:
                pass
        elif chunk.format == PCM_F32 and self.keeps_audio():
            self.reply_of(chunk.turn).samples.append(
                np.frombuffer(chunk.payload, dtype="<f4")
            )
        elif event == REPLY_PROGRESS:
            self.end_sentence(
                self.reply_of(chunk.turn),
                int(chunk.payload["text_end"]),
                float(chunk.payload["audio_ms"]),
                emit,
                sentences,
            )
        elif (
            event == REPLY_DELIVERED
            and self.config.hears()
            and self.reply is not None
            and self.reply.turn == chunk.turn
        ):
            self.reply.delivered_ms = float(chunk.payload["audio_ms"])
            self.cut(self.reply, sentences)
        elif (
            event == TRUNCATE
            and self.config.hears()
            and self.reply is not None
            and chunk.turn in (None, self.reply.turn)
        ):
            self.reply.client_ms = float(chunk.payload["t_ms"])
            self.cut(self.reply, sentences)
        else:
            pass

    def end_sentence(
        self,
        reply: SpokenReply,
        text_end: int,
        audio_ms: float,
        emit: ChunkEmitter,
        sentences: asyncio.Queue[SentenceAudio | HeardAudio | None],
    ) -> None:
        span = reply.text[reply.start : text_end]
        text = span.strip()
        progress(emit, reply.turn, text_end, audio_ms)
        if self.keeps_audio() and text and reply.samples:
            sentence = SentenceAudio(
                reply.turn,
                text,
                reply.start + len(span) - len(span.lstrip()),
                reply.start_ms,
                np.concatenate(reply.samples),
                reply.language,
            )
            if self.config.method == "forced":
                sentences.put_nowait(sentence)
            else:
                pass
            if self.config.hears():
                reply.sentences.append(sentence)
            else:
                pass
        else:
            pass
        reply.start, reply.start_ms, reply.samples = text_end, audio_ms, []

    def cut(
        self,
        reply: SpokenReply,
        sentences: asyncio.Queue[SentenceAudio | HeardAudio | None],
    ) -> None:
        """Queue the audio of the sentence at the cut for recognition when the cut moved before the one last recognized; it follows the sentences already queued for the forced aligner, so their word ends are known when it is recognized."""
        cut_ms = min(
            ms for ms in (reply.client_ms, reply.delivered_ms) if ms is not None
        )
        if reply.heard_ms is not None and cut_ms >= reply.heard_ms:
            pass
        else:
            reply.heard_ms = cut_ms
            heard = heard_audio(reply, cut_ms)
            if heard is None:
                pass
            else:
                sentences.put_nowait(heard)

    async def align_sentences(
        self, sentences: AsyncIterator[SentenceAudio | HeardAudio], emit: ChunkEmitter
    ) -> None:
        async for sentence in sentences:
            if isinstance(sentence, HeardAudio):
                text_end = await self.hear(sentence)
                if text_end > 0:
                    progress(
                        emit,
                        sentence.turn,
                        sentence.start + text_end,
                        sentence.audio_ms,
                    )
                else:
                    pass
            else:
                boundaries = await self.align(sentence)
                if self.reply is not None and self.reply.turn == sentence.turn:
                    self.reply.word_ends.extend(boundaries)
                else:
                    pass
                for text_end, audio_ms in boundaries:
                    progress(emit, sentence.turn, text_end, audio_ms)

    async def hear(self, heard: HeardAudio) -> int:
        """How many characters of the sentence's text its audio up to the cut says, by the transcript's strict prefix; 0 when a word end at or after the cut covers the sentence, or when the service fails or the first words differ."""
        sentence_end = heard.start + len(heard.text)
        if (
            self.reply is not None
            and self.reply.turn == heard.turn
            and any(
                heard.start < text_end <= sentence_end and audio_ms >= heard.audio_ms
                for text_end, audio_ms in self.reply.word_ends
            )
        ):
            return 0
        else:
            pass
        started = time.monotonic()
        try:
            response = await request_transcription(
                self.client,
                self.config.heard_asr_url,
                self.config.heard_asr_model,
                pcm16_bytes(heard.samples),
                SPEECH_SAMPLE_RATE_HZ,
                language=heard.language,
                response_format="json",
            )
        except httpx.TransportError as error:
            logger.warning("heard-speech transcription failed: %r", error)
            return 0
        if response.status_code != 200:
            logger.warning(
                "heard-speech transcription answered %s", response.status_code
            )
            return 0
        else:
            pass
        try:
            result = TranscriptionResult.model_validate_json(response.content)
        except ValidationError as error:
            logger.warning(
                "heard-speech transcription sent an invalid result: %s", error
            )
            return 0
        matched = spoken_prefix_end(heard.text, result.text)
        if heard.language in ("English", "Chinese"):
            expanded_end_chars = matched_prefix_end(
                spoken_units(heard.text, heard.language),
                spoken_units(result.text, heard.language),
            )
            matched = max(matched, expanded_end_chars)
        else:
            pass
        matched = (
            TRAILING_PUNCTUATION.match(heard.text, matched).end() if matched else 0
        )
        logger.info(
            "heard turn %d to %.0f ms: %d ms of sentence audio transcribed as %r in %.0f ms keeps %d of %d chars",
            heard.turn,
            heard.audio_ms,
            len(heard.samples) * 1000 // SPEECH_SAMPLE_RATE_HZ,
            result.text,
            (time.monotonic() - started) * 1000,
            matched,
            len(heard.text),
        )
        return matched

    async def align(self, sentence: SentenceAudio) -> list[tuple[int, float]]:
        """The word ends the worker finds in the sentence, as reply offsets; none when it fails, which leaves the sentence's own end."""
        pcm16 = pcm16_bytes(sentence.samples)
        if len(pcm16) > MAX_ALIGNMENT_AUDIO_BYTES:
            logger.warning("sentence audio exceeds the alignment worker's limit")
            return []
        else:
            pass
        try:
            response = await self.client.post(
                f"{self.config.alignment_url.rstrip('/')}/align",
                json=alignment_request(sentence.text, pcm16, sentence.language),
            )
        except httpx.TransportError as error:
            logger.warning("alignment worker failed: %r", error)
            return []
        if response.status_code != 200:
            logger.warning("alignment worker answered %s", response.status_code)
            return []
        else:
            pass
        try:
            result = AlignmentResult.model_validate_json(response.content)
        except ValidationError as error:
            logger.warning("alignment worker sent an invalid result: %s", error)
            return []
        ends = [boundary.text_end_chars for boundary in result.boundaries]
        samples = [boundary.audio_end_samples for boundary in result.boundaries]
        if (
            ends != sorted(ends)
            or samples != sorted(samples)
            or any(end > len(sentence.text) for end in ends)
            or any(sample > len(sentence.samples) for sample in samples)
        ):
            logger.warning("alignment worker sent boundaries outside the sentence")
            return []
        else:
            return [
                (
                    sentence.start + end,
                    sentence.start_ms + sample * 1000 / SPEECH_SAMPLE_RATE_HZ,
                )
                for end, sample in zip(ends, samples)
            ]

    async def close(self) -> None:
        await self.client.aclose()


def progress(emit: ChunkEmitter, turn: int, text_end: int, audio_ms: float) -> None:
    emit(control_chunk(REPLY_PROGRESS, turn=turn, text_end=text_end, audio_ms=audio_ms))


def pcm16_bytes(samples: np.ndarray) -> bytes:
    return (np.clip(samples, -1, 1) * 32767).astype("<i2").tobytes()


def alignment_request(
    text: str, pcm16: bytes, language: str | None
) -> dict[str, object]:
    """The worker's request for the word ends of text in its pcm16 audio."""
    return with_language(
        {
            "text": text,
            "audio": base64.b64encode(pcm16).decode(),
            "sample_rate_hz": SPEECH_SAMPLE_RATE_HZ,
            "audio_format": "pcm16",
        },
        language,
    )


def heard_audio(reply: SpokenReply, cut_ms: float) -> HeardAudio | None:
    """The audio of the reply's sentence at `cut_ms` from the sentence's start to the cut; None when the cut falls on a sentence end or outside the reply's audio."""
    ended = [
        sentence
        for sentence in reply.sentences
        if sentence.start_ms
        < cut_ms
        < sentence.start_ms + len(sentence.samples) * 1000 / SPEECH_SAMPLE_RATE_HZ
    ]
    if ended:
        sentence = ended[0]
        text, start, start_ms, samples = (
            sentence.text,
            sentence.start,
            sentence.start_ms,
            sentence.samples,
        )
    elif reply.samples and cut_ms > reply.start_ms:
        span = reply.text[reply.start :]
        text = span.strip()
        start = reply.start + len(span) - len(span.lstrip())
        start_ms, samples = reply.start_ms, np.concatenate(reply.samples)
    else:
        return None
    cut = samples[: round((cut_ms - start_ms) * SPEECH_SAMPLE_RATE_HZ / 1000)]
    if text and len(cut):
        return HeardAudio(reply.turn, text, start, cut_ms, cut, reply.language)
    else:
        return None
