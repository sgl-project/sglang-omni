# SPDX-License-Identifier: Apache-2.0
"""Synthesizer kinds: reply audio from the reply text."""

import asyncio
import re
from collections.abc import AsyncIterator
from typing import Literal

import httpx
import numpy as np
from pydantic import Field

from sglang_omni.models.streaming_cascade.base import (
    SPEECH_SAMPLE_RATE_HZ,
    ComponentConfig,
    Replier,
    Replies,
    StreamComponent,
    audio_chunk,
    bearer_headers,
    control_chunk,
    iterate,
    replying,
    sentences,
)
from sglang_omni.models.streaming_cascade.numerals import spoken_text
from sglang_omni.proto.session import TimedChunk
from sglang_omni.proto.streams import (
    REPLY_PROGRESS,
    TEXT_DELTA,
    TURN_DONE,
    chunk_language,
    control_type,
)
from sglang_omni.scheduling.session import ChunkEmitter

# Note (ruoyu): Qwen3-TTS reads an emoji as stray syllables or runs on for minutes and spells out a link target.
MARKDOWN_LINK = re.compile(r"\[([^\]]*)\]\([^)]*\)")
UNSPOKEN_SYMBOLS = re.compile(
    "[\U0001f000-\U0001faff\u2600-\u27bf\u2b50\u2b55\ufe0e\ufe0f\u200d\u20e3*`]"
)
# Note (Haoling Pu): Qwen3-TTS reads a heading, list or quote marker as a word, garbles a table row and can run on for 40 s or more after either, and reads a bare URL out symbol by symbol.
LINE_MARKER = re.compile(r"(?m)^[ \t]*(?:#{1,6}|[-+\u2022>])[ \t]+")
TABLE_EDGE_PIPE = re.compile(r"(?m)^[ \t]*\||\|[ \t]*$")
TABLE_PIPE = re.compile(r"[ \t]*\|[ \t]*")
BARE_URL = re.compile(
    r"(?<![A-Za-z0-9])(?:https?://|www\.)(?:www\.)?([A-Za-z0-9.-]+)(?::[0-9]+)?(?:[/?#][!-{}~]*?)??(?=[.,!?;:)\]]*(?:[^!-~]|[|]|$))"
)


def has_chinese_characters(sentence: str) -> bool:
    return any("\u4e00" <= char <= "\u9fff" for char in sentence)


class OpenAISpeechSynthesizerConfig(ComponentConfig):
    """Sentence-chunked `/v1/audio/speech` streaming PCM, one request per sentence.

    `language` is the service's language tag: empty leaves it to the
    service, `auto` follows the language the recognizer heard from the
    user, and any other name tags every sentence without Chinese characters.
    Without a known user language `auto` tags by script like `English`.
    A sentence tagged English, with no Chinese characters, reaches the
    service with its numbers read out as words.
    """

    kind: Literal["openai_speech"] = "openai_speech"
    base_url: str = Field(pattern=r"^https?://[^/]+")
    model: str
    voice: str = "default"
    reference_audio_path: str = ""
    reference_text: str = ""
    language: str = ""
    max_sentence_words: int = Field(default=24, gt=0)
    api_key_env: str = "OPENAI_API_KEY"


class TurnSpeaker(Replier):
    """Speaks each reply's text deltas as they stream and ends the turn after its audio.

    `speak_reply` turns one reply's deltas into emitted audio; an `interrupt`
    control of that turn cancels it and the rest of the reply's text is dropped.
    """

    async def speak_reply(
        self, deltas: AsyncIterator[TimedChunk], turn: int, emit: ChunkEmitter
    ) -> None:
        raise NotImplementedError

    def progress(
        self, emit: ChunkEmitter, turn: int, text_end: int, audio_ms: float
    ) -> None:
        """Tell the dialog that the reply's first text_end characters have audio ending at audio_ms, so a truncate can cut the history where the listener stopped."""
        emit(
            control_chunk(
                REPLY_PROGRESS, turn=turn, text_end=text_end, audio_ms=audio_ms
            )
        )

    async def process(
        self, inputs: AsyncIterator[TimedChunk], emit: ChunkEmitter
    ) -> None:
        deltas: asyncio.Queue[TimedChunk | None] = asyncio.Queue()

        async def speak_then_end(turn: int) -> None:
            await self.speak_reply(iterate(deltas), turn, emit)
            emit(control_chunk(TURN_DONE, turn=turn))

        async def handle(chunk: TimedChunk, replies: Replies) -> None:
            nonlocal deltas
            if chunk.format == TEXT_DELTA and chunk.turn in replies.interrupted:
                # Note (Junnan Li): The generator may emit a few deltas after the interrupt before it stops.
                pass
            elif chunk.format == TEXT_DELTA:
                if not replies.is_running(chunk.turn):
                    await replies.finish()
                    deltas = asyncio.Queue()
                    replies.start(chunk.turn, speak_then_end(chunk.turn))
                else:
                    pass
                deltas.put_nowait(chunk)
            elif control_type(chunk) == TURN_DONE:
                if replies.is_running(chunk.turn):
                    deltas.put_nowait(None)
                elif chunk.turn in replies.interrupted:
                    replies.interrupted.discard(chunk.turn)
                else:
                    emit(control_chunk(TURN_DONE, turn=chunk.turn))
            else:
                pass

        await replying(inputs, handle, self.replies)


class OpenAISpeechSynthesizer(TurnSpeaker):
    """Speaks each reply sentence by sentence; turn.done follows the reply's audio."""

    sample_rate_hz = SPEECH_SAMPLE_RATE_HZ

    def __init__(self, config: OpenAISpeechSynthesizerConfig) -> None:
        super().__init__()
        self.config = config
        self.client = httpx.AsyncClient(
            timeout=config.timeout_s, headers=bearer_headers(config.api_key_env)
        )
        self.session_language: str | None = None

    def adopt(self, previous: StreamComponent) -> None:
        if isinstance(previous, OpenAISpeechSynthesizer):
            self.session_language = previous.session_language
        else:
            pass

    def sentence_language(self, sentence: str) -> str:
        """The language tag of one sentence."""
        if self.config.language == "auto" and self.session_language is not None:
            return self.session_language
        elif has_chinese_characters(sentence):
            # Note (Junnan Li): A sentence with CJK characters is spoken as Chinese whatever the configured language, so a reply in the user's language keeps the right tag.
            return "Chinese"
        elif self.config.language == "auto":
            return "English"
        else:
            return self.config.language

    async def follow_language(
        self, deltas: AsyncIterator[TimedChunk]
    ) -> AsyncIterator[TimedChunk]:
        """The reply's deltas, recording the user language they carry."""
        async for chunk in deltas:
            language = chunk_language(chunk)
            if language is not None:
                self.session_language = language
            else:
                pass
            yield chunk

    async def speak(self, sentence: str) -> AsyncIterator[np.ndarray]:
        body: dict[str, object] = {
            "model": self.config.model,
            "voice": self.config.voice,
            "input": sentence,
            "response_format": "pcm",
            "stream": True,
        }
        if self.config.language:
            language = self.sentence_language(sentence)
            body["language"] = language
            if language == "English" and not has_chinese_characters(sentence):
                # Note (ruoyu): A Base voice cloned from a Chinese reference read the digits of English sentences as Chinese numerals in 12 of 50 takes and the same numbers written as words in 0 of 50; the aligner reads the digits with the same words.
                body["input"] = spoken_text(sentence, language)[0]
            else:
                pass
        else:
            pass
        if self.config.reference_audio_path:
            body["references"] = [
                {
                    "audio_path": self.config.reference_audio_path,
                    **(
                        {"text": self.config.reference_text}
                        if self.config.reference_text
                        else {}
                    ),
                }
            ]
        else:
            pass
        async with self.client.stream(
            "POST", f"{self.config.base_url.rstrip('/')}/audio/speech", json=body
        ) as response:
            response.raise_for_status()
            rate = response.headers.get("X-Sample-Rate")
            if rate is None:
                raise ValueError(
                    "speech service sent no X-Sample-Rate header, so its PCM rate is unknown"
                )
            elif int(rate) != self.sample_rate_hz:
                raise ValueError(
                    f"speech service streams {rate} Hz, expected {self.sample_rate_hz}"
                )
            else:
                pass
            carry = b""
            async for data in response.aiter_bytes():
                data = carry + data
                usable = len(data) - len(data) % 2
                carry = data[usable:]
                if usable:
                    yield np.frombuffer(data[:usable], dtype="<i2").astype(
                        "<f4"
                    ) / 32768
                else:
                    pass

    async def speak_reply(
        self, deltas: AsyncIterator[TimedChunk], turn: int, emit: ChunkEmitter
    ) -> None:
        text_end = 0
        audio_ms = 0.0
        async for span in sentences(
            self.follow_language(deltas), self.config.max_sentence_words
        ):
            text = UNSPOKEN_SYMBOLS.sub("", MARKDOWN_LINK.sub(r"\1", span))
            text = TABLE_PIPE.sub(
                ", ",
                TABLE_EDGE_PIPE.sub("", LINE_MARKER.sub("", BARE_URL.sub(r"\1", text))),
            )
            sentence = " ".join(text.split())
            # Note (ruoyu): Qwen3-TTS answers 500 to a sentence with no word in it, which would close the session.
            if re.search(r"\w", sentence):
                async for samples in self.speak(sentence):
                    audio_ms += len(samples) * 1000 / self.sample_rate_hz
                    emit(audio_chunk(samples, self.sample_rate_hz, turn))
            else:
                pass
            text_end += len(span)
            self.progress(emit, turn, text_end, audio_ms)

    async def close(self) -> None:
        await self.client.aclose()
