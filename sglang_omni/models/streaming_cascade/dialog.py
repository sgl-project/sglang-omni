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
    ChatMessage,
    ComponentConfig,
    StreamComponent,
    control_chunk,
    messages_chunk,
    tool_round_messages,
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
    TASK_RESULT,
    TEXT_DELTA,
    TOOL_EXCHANGE,
    TRANSCRIPT_FORMAT,
    TRUNCATE,
    TURN_DONE,
    TURN_EMPTY,
    TURN_INTERRUPTED,
    TURN_QUEUED,
    TURN_STARTED,
    chunk_language,
    control_type,
    edge_event,
)
from sglang_omni.proto.tools import (
    TaskResult,
    ToolExchange,
    task_result_of,
    tool_exchange_of,
)
from sglang_omni.scheduling.session import ChunkEmitter

if TYPE_CHECKING:
    from sglang_omni.models.streaming_cascade.config import ComponentGraphConfig
else:
    pass

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
# note (zengjiahua): the notice says it is not the user's words, so the model reports the result instead of answering it as a request, and says the image is on screen, so the model does not read it out.
DEFAULT_TASK_NOTICE_TEMPLATES = {
    "succeeded": "[Task update, not said by the user] The {name} task {task_id} you started has finished: {summary}. The result is already on the user's screen. Tell the user in one short spoken sentence, in the language of the conversation. Do not describe the result in detail and do not call any tool.",
    "failed": "[Task update, not said by the user] The {name} task {task_id} you started failed: {error}. Tell the user in one short spoken sentence, in the language of the conversation, and ask whether to try again. Do not call any tool.",
}
MAX_NOTICE_FIELD_CHARS = 300


def assistant_messages(message: ChatMessage) -> list[ChatMessage]:
    """An assistant entry as the model reads it: its tool rounds expanded at their text offsets, a reply the listener cut ending with …, and only such a reply ending with …. A tool round past the cut keeps its calls and responses, since the task it started is real, but none of the text after the cut."""
    if message.is_cut:
        content = message.content.removesuffix("…")
    else:
        # note (Haoling Pu): the model sometimes ends a finished reply with …, which would read as cut and was followed by an empty reply in a live run.
        content = message.content.rstrip().removesuffix("…")
    expanded: list[ChatMessage] = []
    start = 0
    for exchange in message.tool_exchanges:
        end = min(exchange.text_offset, len(content))
        expanded.extend(tool_round_messages(content[start:end], exchange))
        start = end
    tail = content[start:]
    if message.is_cut and (tail or not message.tool_exchanges):
        expanded.append(ChatMessage(role="assistant", content=f"{tail}…"))
    elif tail or not message.tool_exchanges:
        expanded.append(ChatMessage(role="assistant", content=tail))
    else:
        pass
    return expanded


def prompt_messages(messages: list[ChatMessage]) -> list[ChatMessage]:
    """The history as the model reads it, a cut reply under a system note."""
    prompt: list[ChatMessage] = []
    for message in messages:
        if message.role == "assistant":
            prompt.extend(assistant_messages(message))
        else:
            prompt.append(ChatMessage(role=message.role, content=message.content))
    if any(message.is_cut for message in messages):
        prompt[0] = ChatMessage(
            role="system", content=f"{prompt[0].content} {CUT_REPLY_NOTE}"
        )
    else:
        pass
    return prompt


class DialogConfig(ComponentConfig):
    """Conversation history; one messages chunk per answered utterance and per task result the dialog speaks about."""

    kind: Literal["dialog"] = "dialog"
    system_prompt: str = "You are a helpful voice assistant. Answer concisely."
    max_history_chars: int = Field(default=MAX_HISTORY_CHARS, gt=0)
    max_text_chars: int = Field(default=16000, gt=0)
    # Fields: {name}, {task_id}, {summary}, {error}.
    task_notice_templates: dict[Literal["succeeded", "failed"], str] = Field(
        default_factory=lambda: dict(DEFAULT_TASK_NOTICE_TEMPLATES)
    )


@dataclass(kw_only=True)
class UserUtterance:
    turn: int
    text: str
    language: str | None


@dataclass(kw_only=True)
class TaskFollowUp:
    turn: int
    result: TaskResult


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

    A reply's tool rounds stay on its history entry and are expanded only in
    the prompt, so cuts and joins see one entry per reply. A task result the
    dialog speaks about opens a turn of its own once no reply runs and the
    user is not speaking; every task result also updates the tool response
    that started the task.
    """

    def __init__(self, config: DialogConfig) -> None:
        self.config = config
        self.emit: ChunkEmitter | None = None
        self.messages = [ChatMessage(role="system", content=config.system_prompt)]
        self.waiting: deque[UserUtterance | TaskFollowUp] = deque()
        self.reply: list[str] | None = None
        self.reply_exchanges: list[ToolExchange] = []
        self.reply_turn = 0
        self.turns = 0
        # Reply audio produced so far as (characters of the reply text with audio, audio ms), one entry per spoken sentence, for truncating an interrupted reply to what was heard.
        self.progress: list[tuple[int, float]] = []
        # note (luojiaxuan): The ended reply a truncate, its delivered audio or its progress may still cut, as (turn, history index), until a later reply ends, other than one interrupted before its first text, or its entry leaves the history.
        self.truncatable: tuple[int, int] | None = None
        # Note (Junnan Li): The ended reply's full text, the reply audio the graph delivered (unknown, or all of it once the reply is done) and the last truncate's heard position, so progress that arrives later re-cuts it.
        self.truncatable_text = ""
        # note (Haoling Pu): an interrupted reply was cut even when the listener heard all the text it had produced.
        self.is_truncatable_interrupted = False
        # note (luojiaxuan): The ended reply's progress, which keeps arriving while the next reply runs.
        self.truncatable_progress: list[tuple[int, float]] = []
        self.delivered_ms: float | None = math.inf
        self.heard_ms: float | None = None
        self.language: str | None = None
        # note (luojiaxuan): From a speech.started to the speech.stopped or commit that ends the utterance; the node that reports speech also stops it at the end of the input.
        self.is_user_speaking = False
        # note (luojiaxuan): Whether the graph reports the audio it delivered of a reply this dialog interrupts; a graph without audio does not.
        self.awaits_delivery = True

    def place(self, settings: "ComponentGraphConfig", name: str) -> None:
        self.awaits_delivery = bool(settings.delivery_reporters(name))

    def adopt(self, previous: StreamComponent) -> None:
        if isinstance(previous, Dialog):
            self.emit = previous.emit
            self.messages = [self.messages[0], *previous.messages[1:]]
            self.waiting = previous.waiting
            self.reply = previous.reply
            self.reply_exchanges = previous.reply_exchanges
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
        else:
            pass

    def state_bytes(self) -> int:
        chars = sum(len(message.content) for message in self.messages)
        chars += sum(
            len(entry.text)
            for entry in self.waiting
            if isinstance(entry, UserUtterance)
        )
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
        elif edge_event(event) == TASK_RESULT:
            self.take_task_result(task_result_of(event))
            self.respond_to_waiting()
        else:
            pass

    def take_task_result(self, result: TaskResult) -> None:
        """Record a task's end on the tool response that started it, and queue a turn about it when the task is to be spoken of."""
        assert self.emit is not None
        for exchange in [
            *self.reply_exchanges,
            *(
                exchange
                for message in self.messages
                for exchange in message.tool_exchanges
            ),
        ]:
            for response in exchange.responses:
                if (
                    isinstance(response.response, dict)
                    and response.response.get("task_id") == result.task_id
                ):
                    response.response = {**response.response, "status": result.status}
                else:
                    pass
        if result.follow_up == "speak" and result.status in ("succeeded", "failed"):
            self.turns += 1
            self.waiting.append(TaskFollowUp(turn=self.turns, result=result))
            self.emit(control_chunk(TURN_QUEUED, turn=self.turns))
        else:
            pass

    def handle(self, chunk: TimedChunk) -> None:
        assert self.emit is not None
        if chunk.format == TRANSCRIPT_FORMAT and chunk.payload["is_final"]:
            self.turns += 1
            self.waiting.append(
                UserUtterance(
                    turn=self.turns,
                    text=str(chunk.payload["text"]).strip(),
                    language=chunk_language(chunk),
                )
            )
            self.emit(control_chunk(TURN_QUEUED, turn=self.turns))
        elif chunk.format == TOOL_EXCHANGE and self.is_reply(chunk):
            self.reply_exchanges.append(tool_exchange_of(chunk))
        elif chunk.format == TEXT_DELTA and self.is_reply(chunk):
            self.reply.append(str(chunk.payload["text"]))
            if sum(len(part) for part in self.reply) > self.config.max_text_chars:
                raise ValueError("reply exceeds max_text_chars")
            else:
                pass
        elif control_type(chunk) == TURN_DONE and self.is_reply(chunk):
            reply = "".join(self.reply)
            self.messages.append(
                ChatMessage(
                    role="assistant",
                    content=reply,
                    tool_exchanges=self.reply_exchanges,
                )
            )
            self.reply_exchanges = []
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
        self.messages.append(
            ChatMessage(role="assistant", tool_exchanges=self.reply_exchanges)
        )
        self.reply_exchanges = []
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
            and self.waiting
            and (
                not self.awaits_delivery
                or self.delivered_ms is not None
                or self.heard_ms is not None
            )
            and not (
                isinstance(self.waiting[0], TaskFollowUp) and self.is_user_speaking
            )
        ):
            entry = self.waiting.popleft()
            if isinstance(entry, UserUtterance):
                outputs = self.respond(entry.turn, entry.text, entry.language)
            else:
                outputs = self.respond_to_task(entry)
            for output in outputs:
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
                tool_exchanges=self.messages[index].tool_exchanges,
            )
            if self.heard_ms is not None:
                self.emit(control_chunk(REPLY_HEARD, turn=turn, text_end=text_end))
            else:
                pass
            if is_empty_reply(self.messages[index]) and index < len(self.messages) - 1:
                self.join_around(index)
            else:
                pass

    def join_around(self, index: int) -> None:
        """Drop the reply at index, which the history keeps nothing of, and join the user messages around it: nothing of the reply was heard, so the user kept talking, and an empty assistant entry would only show the model an empty answer to copy. A task notice is not the user's words, so a reply next to one is only dropped."""
        if self.messages[index - 1].is_notice or self.messages[index + 1].is_notice:
            self.messages.pop(index)
        else:
            self.messages[index - 1] = ChatMessage(
                role="user",
                content=f"{self.messages[index - 1].content} {self.messages[index + 1].content}",
            )
            del self.messages[index : index + 2]
        if self.truncatable is not None and self.truncatable[1] == index:
            self.truncatable = None
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
            self.messages.append(ChatMessage(role="user", content=text))
            if self.messages[-2].role == "assistant" and is_empty_reply(
                self.messages[-2]
            ):
                self.join_around(len(self.messages) - 2)
            else:
                pass
            if language is not None:
                self.language = language
            else:
                pass
            return self.open_reply(turn)

    def respond_to_task(self, follow_up: TaskFollowUp) -> list[TimedChunk]:
        result = follow_up.result
        notice = self.config.task_notice_templates[result.status].format(
            name=result.name,
            task_id=result.task_id,
            summary=result.summary[:MAX_NOTICE_FIELD_CHARS],
            error=(result.error or "")[:MAX_NOTICE_FIELD_CHARS],
        )
        if self.messages[-1].role == "assistant" and is_empty_reply(self.messages[-1]):
            self.messages.pop()
            if self.truncatable is not None and self.truncatable[1] == len(
                self.messages
            ):
                self.truncatable = None
            else:
                pass
        else:
            pass
        self.messages.append(ChatMessage(role="user", content=notice, is_notice=True))
        return self.open_reply(follow_up.turn)

    def open_reply(self, turn: int) -> list[TimedChunk]:
        self.reply = []
        self.reply_exchanges = []
        self.reply_turn = turn
        self.progress = []
        return [
            control_chunk(TURN_STARTED, turn=turn),
            messages_chunk(prompt_messages(self.messages), turn, self.language),
        ]


def is_empty_reply(message: ChatMessage) -> bool:
    """An assistant entry the history keeps nothing of: no text and no tool round."""
    return not message.content and not message.tool_exchanges
