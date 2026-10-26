# SPDX-License-Identifier: Apache-2.0
"""The dialog kind: the conversation history and the turns."""

import logging
import math
from collections import deque
from collections.abc import AsyncIterator
from typing import Literal

from pydantic import Field

from sglang_omni.models.streaming_cascade.base import (
    ChatMessage,
    ComponentConfig,
    StreamComponent,
    control_chunk,
    messages_chunk,
)
from sglang_omni.proto.session import TimedChunk
from sglang_omni.proto.streams import (
    COMMIT,
    INTERRUPT,
    MAX_HISTORY_CHARS,
    REPLY_DELIVERED,
    REPLY_HEARD,
    REPLY_PROGRESS,
    SPEECH_STARTED,
    SPEECH_STOPPED,
    TEXT_DELTA,
    TRANSCRIPT_FORMAT,
    TRUNCATE,
    TURN_DONE,
    TURN_EMPTY,
    TURN_INTERRUPTED,
    TURN_QUEUED,
    TURN_STARTED,
    chunk_language,
    control_type,
)
from sglang_omni.scheduling.session import ChunkEmitter

logger = logging.getLogger(__name__)


def heard_prefix(
    progress: list[tuple[int, float]], heard_ms: float, text: str
) -> tuple[int, bool]:
    """The characters of `text` up to the last sentence boundary whose audio ends at or before heard_ms, and whether the listener heard past it into text the cut drops."""
    text_end, audio_ms = max(
        (
            (boundary, boundary_ms)
            for boundary, boundary_ms in progress
            if boundary_ms <= heard_ms
        ),
        default=(0, 0.0),
    )
    return text_end, heard_ms > audio_ms and bool(text[text_end:].strip())


# note (Haoling Pu): a cut reply shown as is reads to the model as how it answers, and after a few interruptions it replies with fragments (#3063).
CUT_REPLY_NOTE = "A reply of yours that ends with … was cut off when the user interrupted it; answer the latest message in full."


def prompt_messages(messages: list[ChatMessage]) -> list[ChatMessage]:
    """The history as the model reads it: a reply the listener cut ends with … under a system note, and only such a reply ends with …."""
    prompt: list[ChatMessage] = []
    has_cut = False
    for message in messages:
        if message.is_cut:
            content = message.content.removesuffix("…")
            prompt.append(ChatMessage(role="assistant", content=f"{content}…"))
            has_cut = True
        elif message.role == "assistant":
            # note (Haoling Pu): the model sometimes ends a finished reply with …, which would read as cut and was followed by an empty reply in a live run.
            content = message.content.rstrip().removesuffix("…")
            prompt.append(ChatMessage(role="assistant", content=content))
        else:
            prompt.append(message)
    if has_cut:
        prompt[0] = ChatMessage(
            role="system", content=f"{prompt[0].content} {CUT_REPLY_NOTE}"
        )
    else:
        pass
    return prompt


class DialogConfig(ComponentConfig):
    """Conversation history; one messages chunk per answered utterance."""

    kind: Literal["dialog"] = "dialog"
    system_prompt: str = "You are a helpful voice assistant. Answer concisely."
    max_history_chars: int = Field(default=MAX_HISTORY_CHARS, gt=0)
    max_text_chars: int = Field(default=16000, gt=0)


class Dialog(StreamComponent):
    """Turns final transcripts into prompts and records the replies.

    Each final transcript opens the next turn with turn.queued. An utterance
    that arrives while a reply is running waits for the turn.done of that
    reply's turn, so history always alternates user and assistant. Speech
    over a reply, or an `interrupt` control, ends the reply: the dialog
    emits `interrupt` for the reply's turn to the nodes that take it as a
    control and publishes `turn.interrupted`; `reply.progress` controls from
    the synthesizer tell it how much of the reply was spoken. The interrupted
    reply stays in the history only up to the last sentence whose audio the
    graph delivered (`reply.delivered`), and a client `truncate` of that
    reply's turn moves the cut back to what the listener heard; from then on
    each cut is published as reply.heard. A reply the history keeps nothing
    of when the next reply begins leaves no entry, and the next utterance
    joins the user message it followed. The language the recognizer last
    heard in an answered turn is the Session's language, and every messages
    chunk carries it. A replacement adopts the history, the open reply and
    the waiting utterances, so new
    settings apply at once and reach the next prompt.

    Speech that started after an utterance ended and has not stopped when
    the utterance's reply begins is speech over that reply, so the reply
    ends as soon as it begins.
    """

    def __init__(self, config: DialogConfig) -> None:
        self.config = config
        self.emit: ChunkEmitter | None = None
        self.messages = [ChatMessage(role="system", content=config.system_prompt)]
        # Waiting utterances as (turn, transcript, language the recognizer heard).
        self.waiting: deque[tuple[int, str, str | None]] = deque()
        self.reply: list[str] | None = None
        self.reply_turn = 0
        self.turns = 0
        # Reply audio produced so far as (characters of the reply text with audio, audio ms), one entry per spoken sentence, for truncating an interrupted reply to what was heard.
        self.progress: list[tuple[int, float]] = []
        # Note (Junnan Li): The ended reply a truncate may still cut, as (turn, history index), until the next reply begins.
        self.truncatable: tuple[int, int] | None = None
        # Note (Junnan Li): The ended reply's full text, the reply audio the graph delivered (unknown, or all of it once the reply is done) and the last truncate's heard position, so progress that arrives later re-cuts it.
        self.truncatable_text = ""
        # note (Haoling Pu): an interrupted reply was cut even when the listener heard all the text it had produced.
        self.is_truncatable_interrupted = False
        self.delivered_ms: float | None = math.inf
        self.heard_ms: float | None = None
        self.language: str | None = None
        # note (luojiaxuan): From a speech.started to the speech.stopped or commit that ends the utterance; the node that reports speech also stops it at the end of the input.
        self.is_user_speaking = False

    def adopt(self, previous: StreamComponent) -> None:
        if isinstance(previous, Dialog):
            self.emit = previous.emit
            self.messages = [self.messages[0], *previous.messages[1:]]
            self.waiting = previous.waiting
            self.reply = previous.reply
            self.reply_turn = previous.reply_turn
            self.turns = previous.turns
            self.progress = previous.progress
            self.truncatable = previous.truncatable
            self.truncatable_text = previous.truncatable_text
            self.is_truncatable_interrupted = previous.is_truncatable_interrupted
            self.delivered_ms = previous.delivered_ms
            self.heard_ms = previous.heard_ms
            self.language = previous.language
            self.is_user_speaking = previous.is_user_speaking
        else:
            pass

    def state_bytes(self) -> int:
        chars = sum(len(message.content) for message in self.messages)
        chars += sum(len(text) for _, text, _ in self.waiting)
        chars += sum(len(part) for part in self.reply or ())
        return chars * 4

    def is_reply(self, chunk: TimedChunk) -> bool:
        return self.reply is not None and chunk.turn == self.reply_turn

    async def process(
        self, inputs: AsyncIterator[TimedChunk], emit: ChunkEmitter
    ) -> None:
        self.emit = emit
        async for chunk in inputs:
            self.handle(chunk)
            self.respond_to_waiting()

    async def control(self, event: TimedChunk) -> None:
        if control_type(event) == INTERRUPT and self.reply is not None:
            self.interrupt_reply()
            self.respond_to_waiting()
        elif control_type(event) == REPLY_PROGRESS and event.turn == self.reply_turn:
            self.progress.append(
                (int(event.payload["text_end"]), float(event.payload["audio_ms"]))
            )
            if self.truncatable is not None:
                self.truncate()
            else:
                pass
        else:
            pass

    def handle(self, chunk: TimedChunk) -> None:
        assert self.emit is not None
        if chunk.format == TRANSCRIPT_FORMAT and chunk.payload["is_final"]:
            self.turns += 1
            self.waiting.append(
                (
                    self.turns,
                    str(chunk.payload["text"]).strip(),
                    chunk_language(chunk),
                )
            )
            self.emit(control_chunk(TURN_QUEUED, turn=self.turns))
        elif chunk.format == TEXT_DELTA and self.is_reply(chunk):
            self.reply.append(str(chunk.payload["text"]))
            if sum(len(part) for part in self.reply) > self.config.max_text_chars:
                raise ValueError("reply exceeds max_text_chars")
            else:
                pass
        elif control_type(chunk) == TURN_DONE and self.is_reply(chunk):
            reply = "".join(self.reply)
            self.messages.append(ChatMessage(role="assistant", content=reply))
            self.truncatable = (self.reply_turn, len(self.messages) - 1)
            self.truncatable_text = reply
            self.is_truncatable_interrupted = False
            self.delivered_ms = math.inf
            self.reply = None
        elif control_type(chunk) == SPEECH_STARTED:
            self.is_user_speaking = True
            if self.reply is not None:
                self.interrupt_reply()
            else:
                pass
        elif control_type(chunk) in (SPEECH_STOPPED, COMMIT):
            self.is_user_speaking = False
        elif (
            control_type(chunk) == REPLY_DELIVERED
            and self.truncatable is not None
            and chunk.turn == self.truncatable[0]
        ):
            self.delivered_ms = float(chunk.payload["audio_ms"])
            self.truncate()
        elif control_type(chunk) == TRUNCATE:
            if self.truncatable is not None and chunk.turn in (
                None,
                self.truncatable[0],
            ):
                self.heard_ms = float(chunk.payload["t_ms"])
                self.truncate()
            else:
                logger.info(
                    "truncate of turn %s ignored: the reply that can still be cut is %s",
                    chunk.turn,
                    None if self.truncatable is None else self.truncatable[0],
                )
        else:
            pass

    def interrupt_reply(self) -> None:
        """End the open reply; the history keeps none of it until the graph reports the audio it delivered or a truncate says what was heard."""
        assert self.emit is not None and self.reply is not None
        turn = self.reply_turn
        self.truncatable_text = "".join(self.reply)
        self.is_truncatable_interrupted = True
        self.messages.append(ChatMessage(role="assistant"))
        self.truncatable = (turn, len(self.messages) - 1)
        self.delivered_ms = None
        self.truncate()
        self.reply = None
        self.emit(control_chunk(INTERRUPT, turn=turn))
        self.emit(control_chunk(TURN_INTERRUPTED, turn=turn))

    def respond_to_waiting(self) -> None:
        assert self.emit is not None
        while self.reply is None and self.waiting:
            for output in self.respond(*self.waiting.popleft()):
                self.emit(output)
            if self.reply is not None and self.is_user_speaking:
                self.interrupt_reply()
            else:
                pass

    def truncate(self) -> None:
        """Cut the ended reply in the history after the last sentence whose audio ended by the heard position: the client's truncate, never past the audio the graph delivered, and nothing when neither is known. Once the client's truncate arrived, every cut is reported as reply.heard so the client's caption matches the history."""
        assert self.emit is not None and self.truncatable is not None
        turn, index = self.truncatable
        heard_ms = min(
            (ms for ms in (self.heard_ms, self.delivered_ms) if ms is not None),
            default=0.0,
        )
        if math.isinf(heard_ms):
            pass
        else:
            content = self.truncatable_text
            text_end, uncertain = heard_prefix(self.progress, heard_ms, content)
            logger.info(
                "truncate turn %d at %.0f ms (client %s, delivered %s) keeps %d of %d chars (uncertain=%s, %d progress points)",
                turn,
                heard_ms,
                self.heard_ms,
                self.delivered_ms,
                text_end,
                len(content),
                uncertain,
                len(self.progress),
            )
            kept = content[:text_end].rstrip()
            self.messages[index] = ChatMessage(
                role="assistant",
                content=kept,
                uncertain=uncertain,
                is_cut=self.is_truncatable_interrupted or kept != content.rstrip(),
            )
            if self.heard_ms is not None:
                self.emit(control_chunk(REPLY_HEARD, turn=turn, text_end=text_end))
            else:
                pass

    def respond(self, turn: int, text: str, language: str | None) -> list[TimedChunk]:
        history_chars = sum(len(message.content) for message in self.messages)
        if not text:
            return [control_chunk(TURN_EMPTY, turn=turn)]
        elif len(text) > self.config.max_text_chars:
            raise ValueError("transcript exceeds max_text_chars")
        elif (
            history_chars + len(text) + self.config.max_text_chars
            > self.config.max_history_chars
        ):
            raise ValueError(
                "conversation exceeds max_history_chars; open a new session"
            )
        else:
            if self.messages[-1].role == "assistant" and not self.messages[-1].content:
                # note (luojiaxuan): Nothing of the last reply was heard, so the user kept talking; an empty assistant entry would only show the model an empty answer to copy.
                self.messages.pop()
                self.messages[-1] = ChatMessage(
                    role="user", content=f"{self.messages[-1].content} {text}"
                )
            else:
                self.messages.append(ChatMessage(role="user", content=text))
            if language is not None:
                self.language = language
            else:
                pass
            self.reply = []
            self.reply_turn = turn
            self.progress = []
            self.truncatable = None
            self.heard_ms = None
            return [
                control_chunk(TURN_STARTED, turn=turn),
                messages_chunk(prompt_messages(self.messages), turn, self.language),
            ]
