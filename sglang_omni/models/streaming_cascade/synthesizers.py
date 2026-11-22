# SPDX-License-Identifier: Apache-2.0
"""Synthesizer kinds: reply audio from the reply text."""

import asyncio
import logging
import re
from collections.abc import AsyncIterator
from typing import Literal

import httpx
import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from sglang_omni.models.streaming_cascade.base import (
    SPEECH_SAMPLE_RATE_HZ,
    ComponentConfig,
    Replier,
    Replies,
    StreamComponent,
    audio_chunk,
    bearer_headers,
    blank_screen_blocks,
    control_chunk,
    iterate,
    replying,
    sentences,
    text_chunk,
    with_language,
)
from sglang_omni.models.streaming_cascade.numerals import spoken_text
from sglang_omni.proto.session import TimedChunk
from sglang_omni.proto.streams import (
    PCM_F32,
    REPLY_PROGRESS,
    REPLY_SENTENCE,
    REPLY_TRUNCATED,
    SPEECH_ERROR,
    SPOKEN_TEXT_END,
    TEXT_DELTA,
    TOOL_CALL,
    TURN_DONE,
    chunk_language,
    chunk_style,
    control_type,
)
from sglang_omni.proto.tools import is_partial_call, tool_calls_of
from sglang_omni.scheduling.session import ChunkEmitter

logger = logging.getLogger(__name__)

# The error of a speech that ended without audio, which a text with no words to speak leaves and so does a service that answers with none.
NOTHING_TO_SPEAK = "no audio was produced: the text may have no words to speak"

# Note (ruoyu): Qwen3-TTS reads an emoji as stray syllables or runs on for minutes and spells out a link target.
MARKDOWN_LINK = re.compile(r"\[([^\]]*)\]\([^)]*\)")
UNSPOKEN_SYMBOLS = re.compile(
    "[\U0001f000-\U0001faff\u2600-\u27bf\u2b50\u2b55\ufe0e\ufe0f\u200d\u20e3*`]"
)
# note (Haoling Pu): Qwen3-TTS Base cloning with a reference transcript opens each request with about 0.4 s of room tone, so with drop_quiet_start a request's audio starts this pre-roll before its first sample above SPEECH_ONSET_AMPLITUDE, a fraction of full scale.
SPEECH_ONSET_AMPLITUDE = 0.01
SPEECH_PREROLL_S = 0.05
# note (Haoling Pu): Qwen3-TTS reads a heading, list or quote marker as a word, garbles a table row and can run on for 40 s or more after either, and reads a bare URL out symbol by symbol.
LINE_MARKER = re.compile(r"(?m)^[ \t]*(?:#{1,6}|[-+\u2022>])[ \t]+")
TABLE_EDGE_PIPE = re.compile(r"(?m)^[ \t]*\||\|[ \t]*$")
TABLE_PIPE = re.compile(r"[ \t]*\|[ \t]*")
BARE_URL = re.compile(
    r"(?<![A-Za-z0-9])(?:https?://|www\.)(?:www\.)?([A-Za-z0-9.-]+)(?::[0-9]+)?(?:[/?#][!-{}~]*?)??(?=[.,!?;:)\]]*(?:[^!-~]|[|]|$))"
)


def has_chinese_characters(sentence: str) -> bool:
    return any("\u4e00" <= char <= "\u9fff" for char in sentence)


CJK_LANGUAGES = frozenset({"Chinese", "Cantonese", "Japanese", "Korean"})
KANA = re.compile("[\u3040-\u30ff]")
HANGUL = re.compile("[\uac00-\ud7af]")
LATIN_WORD = re.compile(r"[A-Za-z]+")
CJK_CHARACTER = re.compile("[\u3040-\u30ff\u4e00-\u9fff\uac00-\ud7af]")
LATIN_VOWELS = re.compile("[aeiouy]+", re.IGNORECASE)


class SpeechReference(BaseModel):
    """A recording the speech service clones, with its transcript."""

    model_config = ConfigDict(extra="forbid")

    audio_path: str
    text: str


class OpenAISpeechSynthesizerConfig(ComponentConfig):
    """Sentence-chunked `/v1/audio/speech` streaming PCM, one request per sentence.

    `language` is the service's language tag: empty leaves it to the
    service, `auto` follows the language the recognizer heard from the
    user except in a sentence whose script contradicts it and carries most of
    it, which is tagged by its script, and any other name tags every sentence
    without Chinese characters. Without a known user language `auto` tags by
    script like `English`. A sentence tagged English reaches the service with
    its numbers read out as words. sentence_pause_ms of silence goes before
    the audio of each sentence after the reply's first audio, but not inside a
    sentence cut by max_sentence_words, and a sentence that plays nothing
    sends none.
    styles maps a style name to the recording a sentence of that style
    clones, and a sentence of a style it lacks clones reference_audio_path;
    the language model names a style with a tag such as [calm].
    skip_screen_blocks leaves out of every request what the reply writes
    between <screen> and </screen>, a block still open included, and a
    stray </screen>.
    """

    kind: Literal["openai_speech"] = "openai_speech"
    base_url: str = Field(pattern=r"^https?://[^/]+")
    model: str
    voice: str = "default"
    reference_audio_path: str = ""
    reference_text: str = ""
    styles: dict[str, SpeechReference] = Field(default_factory=dict)
    language: str = ""
    max_sentence_words: int = Field(default=24, gt=0)
    drop_quiet_start: bool = False
    sentence_pause_ms: float = Field(default=0, ge=0)
    skip_screen_blocks: bool = False
    api_key_env: str = "OPENAI_API_KEY"


class TurnSpeaker(Replier):
    """Speaks each reply's text deltas as they stream and ends the turn after its audio.

    `speak_reply` turns one reply's deltas into emitted audio; an `interrupt`
    control of that turn cancels it and the rest of the reply's text is dropped.
    A tool_call chunk holds one speech of its own turn as a whole: its text is
    emitted as a text delta for the nodes that follow the spoken words, and a
    service failure ends the speech with the error and the end of its last
    spoken sentence on its turn.done, so the model is told instead of the
    Session failing. A reply or speech whose generation stopped at its token
    budget ends with REPLY_TRUNCATED on its turn.done.
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
        # Turns whose generation stopped at its token budget, until the turn.done that repeats it.
        truncated_turns: set[int] = set()

        def end_turn(turn: int, **fields: object) -> None:
            if turn in truncated_turns:
                truncated_turns.discard(turn)
                fields[REPLY_TRUNCATED] = True
            else:
                pass
            emit(control_chunk(TURN_DONE, turn=turn, **fields))

        async def speak_then_end(turn: int) -> None:
            await self.speak_reply(iterate(deltas), turn, emit)
            end_turn(turn)

        # The turn of the call whose text is arriving in pieces, until its complete chunk; a piece for it once its speech ended, cut or failed, is dropped.
        streamed_turn: int | None = None

        async def published(
            speech: asyncio.Queue[TimedChunk | None],
        ) -> AsyncIterator[TimedChunk]:
            async for text in iterate(speech):
                emit(text)
                yield text

        async def speak_call_then_end(
            turn: int, speech: asyncio.Queue[TimedChunk | None]
        ) -> None:
            audio_count = 0
            spoken_text_end = 0

            def emit_counting(chunk: TimedChunk) -> None:
                nonlocal audio_count, spoken_text_end
                if chunk.format == PCM_F32:
                    audio_count += 1
                elif control_type(chunk) == REPLY_PROGRESS:
                    spoken_text_end = int(chunk.payload["text_end"])
                else:
                    pass
                emit(chunk)

            try:
                await self.speak_reply(published(speech), turn, emit_counting)
            except httpx.HTTPError as error:
                logger.error(f"speech of turn {turn} failed: {error!r}")
                # note (Jingwen): the dialog reads reply.progress through the aligner, which can deliver it after this turn.done and out of order, so the failure carries where the speech stopped.
                end_turn(
                    turn,
                    **{SPEECH_ERROR: repr(error), SPOKEN_TEXT_END: spoken_text_end},
                )
            else:
                if audio_count == 0:
                    # note (Jinyan Su): a text of only symbols, such as "..." or an emoji, leaves nothing to speak, and the model must not be told it was heard.
                    logger.warning(f"speech of turn {turn} had nothing to speak")
                    end_turn(turn, **{SPEECH_ERROR: NOTHING_TO_SPEAK})
                else:
                    end_turn(turn)

        async def handle(chunk: TimedChunk, replies: Replies) -> None:
            nonlocal deltas, streamed_turn
            if chunk.format == TOOL_CALL and chunk.turn in replies.interrupted:
                # note (Jingwen): an interrupt can overtake a piece of its turn on the way here, and a speech started from that piece would wait for text that never arrives and hold every later speech.
                pass
            elif chunk.format == TOOL_CALL:
                [call] = tool_calls_of(chunk)
                text = text_chunk(
                    call.arguments["text"], chunk.turn, chunk_language(chunk)
                )
                is_streamed = chunk.turn == streamed_turn
                if not is_streamed:
                    await replies.finish()
                    deltas = asyncio.Queue()
                    replies.start(chunk.turn, speak_call_then_end(chunk.turn, deltas))
                    deltas.put_nowait(text)
                elif is_partial_call(chunk) and replies.is_running(chunk.turn):
                    deltas.put_nowait(text)
                else:
                    pass
                if is_partial_call(chunk):
                    streamed_turn = chunk.turn
                else:
                    streamed_turn = None
                    if is_streamed and not replies.is_running(chunk.turn):
                        pass
                    elif chunk.payload.get(REPLY_TRUNCATED) is True:
                        truncated_turns.add(chunk.turn)
                        deltas.put_nowait(None)
                    else:
                        deltas.put_nowait(None)
            elif chunk.format == TEXT_DELTA and chunk.turn in replies.interrupted:
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
            elif control_type(chunk) == TURN_DONE and chunk.turn in replies.interrupted:
                replies.interrupted.discard(chunk.turn)
            elif control_type(chunk) == TURN_DONE:
                if chunk.payload.get(REPLY_TRUNCATED) is True:
                    truncated_turns.add(chunk.turn)
                else:
                    pass
                if replies.is_running(chunk.turn):
                    deltas.put_nowait(None)
                else:
                    end_turn(chunk.turn)
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
        if KANA.search(sentence):
            script = "Japanese"
        elif HANGUL.search(sentence):
            script = "Korean"
        elif has_chinese_characters(sentence):
            script = "Chinese"
        elif any(char.isascii() and char.isalpha() for char in sentence):
            script = "English"
        else:
            script = ""
        # note (Haoling Pu): a sentence whose script contradicts the session language and carries most of it, as in a reply the user asked for in another language, is tagged by its script; a CJK character is about one syllable, so it is weighed against the vowel groups of the Latin words.
        latin_syllable_count = sum(
            max(len(LATIN_VOWELS.findall(word)), 1)
            for word in LATIN_WORD.findall(sentence)
        )
        is_mostly_cjk = len(CJK_CHARACTER.findall(sentence)) > latin_syllable_count
        is_switched = self.session_language is not None and (
            (
                script in ("Japanese", "Korean")
                and script != self.session_language
                and is_mostly_cjk
            )
            or (
                script == "Chinese"
                and self.session_language not in CJK_LANGUAGES
                and is_mostly_cjk
            )
            or (script == "English" and self.session_language in CJK_LANGUAGES)
        )
        if self.config.language == "auto" and is_switched:
            return script
        elif self.config.language == "auto" and self.session_language is not None:
            return self.session_language
        elif has_chinese_characters(sentence):
            # Note (Junnan Li): A sentence with CJK characters is spoken as Chinese whatever the configured language, so a reply in the user's language keeps the right tag.
            return "Chinese"
        elif self.config.language == "auto":
            return "English"
        else:
            return self.config.language

    async def follow_deltas(
        self,
        deltas: AsyncIterator[TimedChunk],
        delta_styles: list[tuple[int, str | None]],
    ) -> AsyncIterator[TimedChunk]:
        """The reply's deltas, recording the user language they carry and, in delta_styles, the reply offset each delta starts at with its style."""
        text_start = 0
        async for chunk in deltas:
            language = chunk_language(chunk)
            if language is not None:
                self.session_language = language
            else:
                pass
            delta_styles.append((text_start, chunk_style(chunk)))
            text_start += len(chunk.payload["text"])
            yield chunk

    async def speak(
        self, sentence: str, language: str | None, style: str | None
    ) -> AsyncIterator[np.ndarray]:
        body: dict[str, object] = {
            "model": self.config.model,
            "voice": self.config.voice,
            "input": sentence,
            "response_format": "pcm",
            "stream": True,
        }
        if language is not None:
            body["language"] = language
            if language == "English":
                # Note (ruoyu): A Base voice cloned from a Chinese reference read the digits of English sentences as Chinese numerals in 12 of 50 takes and the same numbers written as words in 0 of 50; the aligner reads the digits with the same words.
                body["input"] = spoken_text(sentence, language)[0]
            else:
                pass
        else:
            pass
        if style in self.config.styles:
            body["references"] = [self.config.styles[style].model_dump()]
        elif self.config.reference_audio_path:
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
            preroll_samples = round(SPEECH_PREROLL_S * self.sample_rate_hz)
            carry = b""
            held_preroll = np.zeros(0, dtype="<f4")
            is_playing = not self.config.drop_quiet_start
            async for data in response.aiter_bytes():
                data = carry + data
                usable = len(data) - len(data) % 2
                carry = data[usable:]
                samples = (
                    np.frombuffer(data[:usable], dtype="<i2").astype("<f4") / 32768
                )
                if not usable:
                    pass
                elif is_playing:
                    yield samples
                else:
                    audio = np.concatenate([held_preroll, samples])
                    loud_indices = np.flatnonzero(
                        np.abs(audio) > SPEECH_ONSET_AMPLITUDE
                    )
                    if loud_indices.size:
                        is_playing = True
                        yield audio[max(0, int(loud_indices[0]) - preroll_samples) :]
                    else:
                        held_preroll = audio[-preroll_samples:]

    async def speak_reply(
        self, deltas: AsyncIterator[TimedChunk], turn: int, emit: ChunkEmitter
    ) -> None:
        reply_text = ""
        text_end = 0
        audio_ms = 0.0
        pause_samples = round(
            self.config.sentence_pause_ms * self.sample_rate_hz / 1000
        )
        follows_sentence_end = True
        delta_styles: list[tuple[int, str | None]] = []
        async for span, ends_sentence in sentences(
            self.follow_deltas(deltas, delta_styles), self.config.max_sentence_words
        ):
            reply_text += span
            if self.config.skip_screen_blocks:
                # note (Haoling Pu): a span can start inside a block an earlier span opened, so the reply so far is blanked before the span is cut from it.
                spoken = blank_screen_blocks(reply_text)[text_end:]
            else:
                spoken = span
            text = UNSPOKEN_SYMBOLS.sub("", MARKDOWN_LINK.sub(r"\1", spoken))
            text = TABLE_PIPE.sub(
                ", ",
                TABLE_EDGE_PIPE.sub("", LINE_MARKER.sub("", BARE_URL.sub(r"\1", text))),
            )
            sentence = " ".join(text.split())
            language = (
                self.sentence_language(sentence) if self.config.language else None
            )
            emit(
                control_chunk(REPLY_SENTENCE, turn=turn, **with_language({}, language))
            )
            # note (Haoling Pu): each request ends at its last word, so back-to-back requests leave about 0.12 s between sentences, against about 0.35 s when one request reads the whole reply; a cut inside a sentence gets no pause.
            is_pause_due = bool(audio_ms and pause_samples and follows_sentence_end)
            plays_audio = False
            # Note (ruoyu): Qwen3-TTS answers 500 to a sentence with no word in it, which would close the session.
            if re.search(r"\w", sentence):
                # note (Haoling Pu): a span starts with the spacing after the sentence before it, which the model wrote before the tag of this one.
                spoken_start = text_end + len(span) - len(span.lstrip())
                style = next(
                    delta_style
                    for text_start, delta_style in reversed(delta_styles)
                    if text_start <= spoken_start
                )
                async for samples in self.speak(sentence, language, style):
                    if is_pause_due:
                        audio_ms += pause_samples * 1000 / self.sample_rate_hz
                        emit(
                            audio_chunk(
                                np.zeros(pause_samples, dtype="<f4"),
                                self.sample_rate_hz,
                                turn,
                            )
                        )
                        is_pause_due = False
                    else:
                        pass
                    plays_audio = True
                    audio_ms += len(samples) * 1000 / self.sample_rate_hz
                    emit(audio_chunk(samples, self.sample_rate_hz, turn))
            else:
                pass
            if plays_audio:
                follows_sentence_end = ends_sentence
            else:
                # note (Haoling Pu): a span that plays nothing, with no word in it or dropped whole by drop_quiet_start, sends no pause, so a sentence end before it or at its end still puts one pause before the next audio.
                follows_sentence_end = follows_sentence_end or ends_sentence
            text_end += len(span)
            self.progress(emit, turn, text_end, audio_ms)

    async def close(self) -> None:
        await self.client.aclose()
