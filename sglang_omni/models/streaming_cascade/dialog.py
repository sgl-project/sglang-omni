# SPDX-License-Identifier: Apache-2.0
"""The dialog kind: the conversation history and the turns."""

import logging
import math
from collections import deque
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from pydantic import Field

from sglang_omni.models.streaming_cascade.base import (
    ChatImage,
    ChatMessage,
    ComponentConfig,
    StreamComponent,
    control_chunk,
    messages_chunk,
)
from sglang_omni.preprocessing.image import validate_upload_image
from sglang_omni.proto.session import TimedChunk
from sglang_omni.proto.streams import (
    COMMIT,
    IMAGE_FRAME,
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

if TYPE_CHECKING:
    from sglang_omni.models.streaming_cascade.config import ComponentGraphConfig
else:
    pass

logger = logging.getLogger(__name__)
MAX_UTF8_BYTES_PER_CHARACTER = 4


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
    max_history_images: int = Field(default=8, gt=0)
    max_history_image_bytes: int = Field(default=4 * 1024 * 1024, gt=0)


@dataclass(kw_only=True)
class QueuedUtterance:
    """A recognized question waiting for the preceding reply to finish."""

    turn_number: int
    transcript_text: str
    language: str | None
    images: tuple[ChatImage, ...]


@dataclass(kw_only=True)
class ImageUpload:
    """An uploaded image awaiting a question at or after its media timestamp."""

    timestamp_ms: float
    image: ChatImage


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
    each cut is published as reply.heard. In a graph that sends audio, the
    next reply after an interrupted one begins once the delivered audio or a
    truncate bounds the cut, so its prompt holds what could have been heard of it, and these
    reports keep cutting the ended reply until a later reply ends, other
    than one interrupted before its first text, so one that arrives after
    the next reply began still cuts the history later prompts read. A reply
    the history keeps nothing of leaves no entry once an utterance follows
    it, and that utterance joins the user message the reply answered. The
    language the recognizer last heard in an answered turn is the Session's
    language, and every messages chunk carries it. A replacement adopts the
    history, the open reply and
    the waiting utterances, so new
    settings apply at once and reach the next prompt.

    Speech that started after an utterance ended and has not stopped when
    the utterance's reply begins is speech over that reply, so the reply
    ends as soon as it begins.

    A nonempty transcript takes every image uploaded at or before its start,
    in upload order, and the history keeps the newest images that fit
    max_history_images and max_history_image_bytes, so follow-ups still see
    earlier pictures.
    """

    def __init__(self, config: DialogConfig) -> None:
        self.config: DialogConfig = config
        self.emit: ChunkEmitter | None = None
        self.messages: list[ChatMessage] = [
            ChatMessage(role="system", content=config.system_prompt)
        ]
        self.queued_utterances: deque[QueuedUtterance] = deque()
        self.pending_image_uploads: deque[ImageUpload] = deque()
        self.reply: list[str] | None = None
        self.reply_turn: int = 0
        self.turns: int = 0
        # Reply audio produced so far as (characters of the reply text with audio, audio ms), one entry per spoken sentence, for truncating an interrupted reply to what was heard.
        self.progress: list[tuple[int, float]] = []
        # note (luojiaxuan): The ended reply a truncate, its delivered audio or its progress may still cut, as (turn, history index), until a later reply ends, other than one interrupted before its first text, or its entry leaves the history.
        self.truncatable: tuple[int, int] | None = None
        # Note (Junnan Li): The ended reply's full text, the reply audio the graph delivered (unknown, or all of it once the reply is done) and the last truncate's heard position, so progress that arrives later re-cuts it.
        self.truncatable_text: str = ""
        # note (Haoling Pu): an interrupted reply was cut even when the listener heard all the text it had produced.
        self.is_truncatable_interrupted: bool = False
        # note (luojiaxuan): The ended reply's progress, which keeps arriving while the next reply runs.
        self.truncatable_progress: list[tuple[int, float]] = []
        self.delivered_ms: float | None = math.inf
        self.heard_ms: float | None = None
        self.language: str | None = None
        # note (luojiaxuan): From a speech.started to the speech.stopped or commit that ends the utterance; the node that reports speech also stops it at the end of the input.
        self.is_user_speaking: bool = False
        # note (luojiaxuan): Whether the graph reports the audio it delivered of a reply this dialog interrupts; a graph without audio does not.
        self.awaits_delivery: bool = True

    def place(self, settings: "ComponentGraphConfig", name: str) -> None:
        self.awaits_delivery = bool(settings.delivery_reporters(name))

    def adopt(self, previous: StreamComponent) -> None:
        if isinstance(previous, Dialog):
            self.emit = previous.emit
            self.messages = [self.messages[0], *previous.messages[1:]]
            self.queued_utterances = previous.queued_utterances
            self.reply = previous.reply
            self.reply_turn = previous.reply_turn
            self.turns = previous.turns
            self.progress = previous.progress
            self.truncatable = previous.truncatable
            self.truncatable_text = previous.truncatable_text
            self.is_truncatable_interrupted = previous.is_truncatable_interrupted
            self.truncatable_progress = previous.truncatable_progress
            self.delivered_ms = previous.delivered_ms
            self.heard_ms = previous.heard_ms
            self.language = previous.language
            self.is_user_speaking = previous.is_user_speaking
            self.pending_image_uploads = previous.pending_image_uploads
        else:
            pass

    def state_bytes(self) -> int:
        text_character_count = sum(len(message.content) for message in self.messages)
        text_character_count += sum(
            len(utterance.transcript_text) for utterance in self.queued_utterances
        )
        text_character_count += sum(len(text_delta) for text_delta in self.reply or ())
        image_byte_count = sum(
            len(image.image_bytes)
            for message in self.messages
            for image in message.images
        )
        image_byte_count += sum(
            len(image.image_bytes)
            for utterance in self.queued_utterances
            for image in utterance.images
        )
        image_byte_count += sum(
            len(upload.image.image_bytes) for upload in self.pending_image_uploads
        )
        return text_character_count * MAX_UTF8_BYTES_PER_CHARACTER + image_byte_count

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
        elif control_type(event) == REPLY_PROGRESS:
            boundary = (
                int(event.payload["text_end"]),
                float(event.payload["audio_ms"]),
            )
            if self.truncatable is not None and event.turn == self.truncatable[0]:
                self.truncatable_progress.append(boundary)
                self.truncate()
            elif self.reply is not None and event.turn == self.reply_turn:
                self.progress.append(boundary)
            else:
                pass
        else:
            pass

    def handle(self, chunk: TimedChunk) -> None:
        assert self.emit is not None
        if chunk.format == IMAGE_FRAME:
            image = ChatImage(
                media_type=validate_upload_image(chunk.payload),
                image_bytes=chunk.payload,
            )
            self.pending_image_uploads.append(
                ImageUpload(timestamp_ms=chunk.t_start_ms, image=image)
            )
            while (
                len(self.pending_image_uploads) > self.config.max_history_images
                or sum(
                    len(upload.image.image_bytes)
                    for upload in self.pending_image_uploads
                )
                > self.config.max_history_image_bytes
            ):
                evicted_upload = self.pending_image_uploads.popleft()
                logger.warning(
                    f"dropping the image uploaded at {evicted_upload.timestamp_ms:.0f} ms: pending images exceed the image history budget"
                )
        elif chunk.format == TRANSCRIPT_FORMAT and chunk.payload["is_final"]:
            transcript_text = str(chunk.payload["text"]).strip()
            images: tuple[ChatImage, ...] = ()
            if transcript_text:
                while (
                    self.pending_image_uploads
                    and self.pending_image_uploads[0].timestamp_ms <= chunk.t_start_ms
                ):
                    images = (*images, self.pending_image_uploads.popleft().image)
            else:
                pass
            self.turns += 1
            self.queued_utterances.append(
                QueuedUtterance(
                    turn_number=self.turns,
                    transcript_text=transcript_text,
                    language=chunk_language(chunk),
                    images=images,
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
            self.truncatable_progress, self.progress = self.progress, []
            self.delivered_ms = math.inf
            self.heard_ms = None
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
            # note (luojiaxuan): A truncate without a turn names the latest reply, which may have begun after the one a truncate can still cut.
            turn = self.reply_turn if chunk.turn is None else chunk.turn
            if self.truncatable is not None and turn == self.truncatable[0]:
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
        """End the open reply; the history keeps none of it until the graph reports the audio it delivered or a truncate says what was heard. A reply without text, such as one interrupted as it opens, keeps nothing whatever was delivered, so the reply before it stays the one a later report cuts."""
        assert self.emit is not None and self.reply is not None
        turn = self.reply_turn
        text = "".join(self.reply)
        self.messages.append(ChatMessage(role="assistant"))
        if text:
            self.truncatable_text = text
            self.is_truncatable_interrupted = True
            self.truncatable = (turn, len(self.messages) - 1)
            self.truncatable_progress, self.progress = self.progress, []
            self.delivered_ms = None
            self.heard_ms = None
            self.truncate()
        else:
            pass
        self.reply = None
        self.emit(control_chunk(INTERRUPT, turn=turn))
        self.emit(control_chunk(TURN_INTERRUPTED, turn=turn))

    def respond_to_waiting(self) -> None:
        assert self.emit is not None
        # note (luojiaxuan): The next prompt shows the interrupted reply's cut, which has no bound until the graph reports the audio it delivered or a truncate arrives.
        while (
            self.reply is None
            and self.queued_utterances
            and (
                not self.awaits_delivery
                or self.delivered_ms is not None
                or self.heard_ms is not None
            )
        ):
            utterance = self.queued_utterances.popleft()
            for output in self.respond(
                utterance.turn_number,
                utterance.transcript_text,
                utterance.language,
                utterance.images,
            ):
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
            text_end, uncertain = heard_prefix(
                self.truncatable_progress, heard_ms, content
            )
            logger.info(
                "truncate turn %d at %.0f ms (client %s, delivered %s) keeps %d of %d chars (uncertain=%s, %d progress points)",
                turn,
                heard_ms,
                self.heard_ms,
                self.delivered_ms,
                text_end,
                len(content),
                uncertain,
                len(self.truncatable_progress),
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
            if not self.messages[index].content and index < len(self.messages) - 1:
                self.join_around(index)
            else:
                pass

    def join_around(self, index: int) -> None:
        """Drop the reply at index, which the history keeps nothing of, and join the user messages around it: nothing of the reply was heard, so the user kept talking, and an empty assistant entry would only show the model an empty answer to copy."""
        self.messages[index - 1] = ChatMessage(
            role="user",
            content=f"{self.messages[index - 1].content} {self.messages[index + 1].content}",
            images=(
                *self.messages[index - 1].images,
                *(
                    image.model_copy(
                        update={
                            "content_offset": image.content_offset
                            + len(self.messages[index - 1].content)
                            + 1
                        }
                    )
                    for image in self.messages[index + 1].images
                ),
            ),
        )
        del self.messages[index : index + 2]
        if self.truncatable is not None and self.truncatable[1] == index:
            self.truncatable = None
        else:
            pass

    def respond(
        self,
        turn: int,
        text: str,
        language: str | None,
        images: tuple[ChatImage, ...] = (),
    ) -> list[TimedChunk]:
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
            self.messages.append(ChatMessage(role="user", content=text, images=images))
            if self.messages[-2].role == "assistant" and not self.messages[-2].content:
                self.join_around(len(self.messages) - 2)
            else:
                pass
            # note (Yuhao Chen): Evict single images, oldest first, so the newest pictures stay visible to follow-up questions.
            retained_image_count = 0
            retained_image_byte_count = 0
            for index in range(len(self.messages) - 1, 0, -1):
                message = self.messages[index]
                retained_images: list[ChatImage] = []
                for image in reversed(message.images):
                    if (
                        retained_image_count < self.config.max_history_images
                        and retained_image_byte_count + len(image.image_bytes)
                        <= self.config.max_history_image_bytes
                    ):
                        retained_images.insert(0, image)
                        retained_image_count += 1
                        retained_image_byte_count += len(image.image_bytes)
                    else:
                        pass
                if len(retained_images) != len(message.images):
                    self.messages[index] = message.model_copy(
                        update={"images": tuple(retained_images)}
                    )
                else:
                    pass
            if language is not None:
                self.language = language
            else:
                pass
            self.reply = []
            self.reply_turn = turn
            self.progress = []
            return [
                control_chunk(TURN_STARTED, turn=turn),
                messages_chunk(prompt_messages(self.messages), turn, self.language),
            ]
