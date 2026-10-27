# SPDX-License-Identifier: Apache-2.0
"""The dialog kind: the conversation history and the turns."""

import json
import logging
import math
from collections import deque
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Literal

from pydantic import Field

from sglang_omni.models.streaming_cascade.base import (
    AssistantToolCall,
    ChatMessage,
    ComponentConfig,
    FunctionCall,
    StreamComponent,
    control_chunk,
    messages_chunk,
)
from sglang_omni.models.streaming_cascade.tools import ToolSettings
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
    TOOL_CALL,
    TOOL_RESPONSE,
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
from sglang_omni.proto.tools import (
    ToolCall,
    ToolResponse,
    tool_calls_of,
    tool_responses_of,
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
# The result the prompt shows for a call without one: still running, or cancelled when the user interrupted the reply that made it. Without the note the model repeats a running call when asked how it is going.
RUNNING_RESULT = '{"status": "running"}'
CANCELLED_RESULT = '{"status": "cancelled"}'
PENDING_CALL_NOTE = 'A tool result of {"status": "running"} means the call is still running and its result will arrive later: do not repeat that call, and if the user asks about it, say it is still in progress. A result of {"status": "cancelled"} means the user interrupted the call: repeat it only if the user asks again.'


def prompt_messages(
    messages: list[ChatMessage], running: frozenset[str]
) -> list[ChatMessage]:
    """The history as the model reads it: a reply the listener cut ends with … under a system note, and only such a reply ends with …; a tool call without a result shows a running or cancelled status under a system note."""
    answered = {message.tool_call_id for message in messages if message.role == "tool"}
    prompt: list[ChatMessage] = []
    notes: list[str] = []
    for message in messages:
        if message.is_cut:
            content = message.content.removesuffix("…")
            prompt.append(message.model_copy(update={"content": f"{content}…"}))
            if CUT_REPLY_NOTE not in notes:
                notes.append(CUT_REPLY_NOTE)
            else:
                pass
        elif message.role == "assistant":
            # note (Haoling Pu): the model sometimes ends a finished reply with …, which would read as cut and was followed by an empty reply in a live run.
            content = message.content.rstrip().removesuffix("…")
            prompt.append(message.model_copy(update={"content": content}))
        else:
            prompt.append(message)
        for call in message.tool_calls or ():
            if call.id in answered:
                pass
            else:
                status = RUNNING_RESULT if call.id in running else CANCELLED_RESULT
                prompt.append(
                    ChatMessage(role="tool", content=status, tool_call_id=call.id)
                )
                if PENDING_CALL_NOTE not in notes:
                    notes.append(PENDING_CALL_NOTE)
                else:
                    pass
    if notes:
        prompt[0] = ChatMessage(
            role="system", content=" ".join([prompt[0].content, *notes])
        )
    else:
        pass
    return prompt


def tool_message(response: ToolResponse) -> ChatMessage:
    """The history entry answering a call: the response as JSON text, or as it is when the tool answered text."""
    answer = response.response
    return ChatMessage(
        role="tool",
        content=(
            answer
            if isinstance(answer, str)
            else json.dumps(answer, ensure_ascii=False)
        ),
        tool_call_id=response.id,
    )


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

    A reply whose generator calls tools records what it said so far with
    the calls as one assistant entry. The reply awaits the calls of awaited
    tools: once every one has its `tool_response` the tool entries join
    the history in call order and the dialog prompts the generator again
    for the same turn, whose text then ends the reply. A call of an
    unawaited tool runs on while the reply ends, and its result, like the
    result of a call an interrupt retained, reports in a turn of its own:
    once no utterance waits, no reply is open and the user is not
    speaking, the results arrived so far join the history in arrival
    order and the generator answers them. The prompt shows a running or
    cancelled status for a call without a result, so the model neither
    repeats the call nor waits for it. An interrupt of a reply keeps the
    responses its awaited calls already had and cancels or retains the
    calls still running, as their tools say; a response to a call the
    dialog no longer awaits is dropped. The turn's whole text, before and
    after the calls, is what the synthesizer's progress counts, so a cut
    of the reply applies to the text after its last calls, or to the
    entry that made them when nothing followed.
    """

    def __init__(self, config: DialogConfig) -> None:
        self.config = config
        self.emit: ChunkEmitter | None = None
        self.messages = [ChatMessage(role="system", content=config.system_prompt)]
        # Waiting utterances as (turn, transcript, language the recognizer heard).
        self.waiting: deque[tuple[int, str, str | None]] = deque()
        self.reply: list[str] | None = None
        self.reply_turn = 0
        # Characters of the open reply's text that its earlier assistant entries, the ones that called tools, already hold.
        self.reply_start = 0
        # The open reply's tool calls, by call id, and the responses to the awaited ones so far.
        self.reply_calls: dict[str, ToolCall] = {}
        self.responses: dict[str, ToolResponse] = {}
        # The calls of ended replies still running, whose results report in a turn of their own, and the results waiting for that turn.
        self.running_calls: dict[str, ToolCall] = {}
        self.arrived: list[ToolResponse] = []
        # Each tool's settings, from the tools nodes answering the calls this dialog reads; a tool they lack is awaited and cancelled.
        self.tools: dict[str, ToolSettings] = {}
        self.turns = 0
        # Reply audio produced so far as (characters of the reply text with audio, audio ms), one entry per spoken sentence, for truncating an interrupted reply to what was heard.
        self.progress: list[tuple[int, float]] = []
        # note (luojiaxuan): The ended reply a truncate, its delivered audio or its progress may still cut, as (turn, history index), until a later reply ends, other than one interrupted before its first text, or its entry leaves the history.
        self.truncatable: tuple[int, int] | None = None
        # Note (Junnan Li): The ended reply's full text, the reply audio the graph delivered (unknown, or all of it once the reply is done) and the last truncate's heard position, so progress that arrives later re-cuts it.
        self.truncatable_text = ""
        self.truncatable_start = 0
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
        self.tools = {}
        for source, formats in settings.node(name).edges().items():
            if TOOL_CALL in formats:
                self.tools.update(settings.tool_settings(source))
            else:
                pass

    def adopt(self, previous: StreamComponent) -> None:
        if isinstance(previous, Dialog):
            self.emit = previous.emit
            self.messages = [self.messages[0], *previous.messages[1:]]
            self.waiting = previous.waiting
            self.reply = previous.reply
            self.reply_turn = previous.reply_turn
            self.reply_start = previous.reply_start
            self.reply_calls = previous.reply_calls
            self.responses = previous.responses
            self.running_calls = previous.running_calls
            self.arrived = previous.arrived
            self.turns = previous.turns
            self.progress = previous.progress
            self.truncatable = previous.truncatable
            self.truncatable_text = previous.truncatable_text
            self.truncatable_start = previous.truncatable_start
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
        chars += sum(len(text) for _, text, _ in self.waiting)
        chars += sum(len(part) for part in self.reply or ())
        return chars * 4

    def is_reply(self, chunk: TimedChunk) -> bool:
        return self.reply is not None and chunk.turn == self.reply_turn

    def awaits(self, call: ToolCall) -> bool:
        """Whether the reply that made the call waits for its result; a tool no tools node declares is awaited."""
        return call.name not in self.tools or self.tools[call.name].awaited

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
        elif chunk.format == TOOL_CALL and self.is_reply(chunk):
            self.record_calls(tool_calls_of(chunk))
        elif chunk.format == TOOL_RESPONSE:
            self.record_responses(tool_responses_of(chunk))
        elif control_type(chunk) == TURN_DONE and self.is_reply(chunk):
            self.end_reply()
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

    def record_calls(self, calls: list[ToolCall]) -> None:
        """Keep what the open reply said so far with its tool calls as one assistant entry, and await the responses of the awaited ones."""
        assert self.reply is not None
        reply = "".join(self.reply)
        self.messages.append(
            ChatMessage(
                role="assistant",
                content=reply[self.reply_start :],
                tool_calls=[
                    AssistantToolCall(
                        id=call.id,
                        function=FunctionCall(
                            name=call.name,
                            arguments=json.dumps(call.arguments, ensure_ascii=False),
                        ),
                    )
                    for call in calls
                ],
            )
        )
        self.reply_start = len(reply)
        self.reply_calls.update({call.id: call for call in calls})
        self.responses = {}

    def record_responses(self, responses: list[ToolResponse]) -> None:
        """Keep the responses the open reply awaits and, once every awaited call has one, append the tool entries in call order and continue the reply; the response of a call no reply awaits reports in a turn of its own, and one of a call the dialog forgot is dropped."""
        assert self.emit is not None
        for response in responses:
            if response.id in self.reply_calls and self.awaits(
                self.reply_calls[response.id]
            ):
                self.responses[response.id] = response
            elif response.id in self.reply_calls or response.id in self.running_calls:
                calls = (
                    self.reply_calls
                    if response.id in self.reply_calls
                    else self.running_calls
                )
                calls.pop(response.id)
                self.arrived.append(response)
            else:
                logger.info(
                    f"tool response {response.id} of {response.name} dropped: no call awaits it"
                )
        awaited = [
            call_id for call_id, call in self.reply_calls.items() if self.awaits(call)
        ]
        if awaited and set(awaited) <= set(self.responses):
            for call_id in awaited:
                self.messages.append(tool_message(self.responses.pop(call_id)))
                self.reply_calls.pop(call_id)
            self.emit(messages_chunk(self.prompt(), self.reply_turn, self.language))
        else:
            pass

    def end_reply(self) -> None:
        """Close the open reply into the history: the text after its last calls becomes its entry, or the entry that made the calls stays its last when nothing followed them, and that entry is the one a later report cuts; the calls still running report in turns of their own."""
        assert self.reply is not None
        reply = "".join(self.reply)
        if not reply[self.reply_start :] and self.messages[-1].tool_calls:
            # The entry that made the calls holds the reply's text from where its earlier entries end.
            self.truncatable_start = self.reply_start - len(self.messages[-1].content)
        else:
            self.messages.append(
                ChatMessage(role="assistant", content=reply[self.reply_start :])
            )
            self.truncatable_start = self.reply_start
        self.truncatable = (self.reply_turn, len(self.messages) - 1)
        self.truncatable_text = reply
        self.is_truncatable_interrupted = False
        self.truncatable_progress, self.progress = self.progress, []
        self.delivered_ms = math.inf
        self.heard_ms = None
        self.reply = None
        self.running_calls.update(self.reply_calls)
        self.reply_calls = {}
        self.responses = {}

    def interrupt_reply(self) -> None:
        """End the open reply; the history keeps none of it until the graph reports the audio it delivered or a truncate says what was heard. A reply without text, such as one interrupted as it opens, keeps nothing whatever was delivered, so the reply before it stays the one a later report cuts. The responses its awaited calls already had join the history, and each call still running is cancelled or retained as its tool says."""
        assert self.emit is not None and self.reply is not None
        turn = self.reply_turn
        text = "".join(self.reply)
        index = len(self.messages) - 1
        is_after_calls = not text[self.reply_start :] and bool(
            self.messages[index].tool_calls
        )
        if is_after_calls:
            entry_start = self.reply_start - len(self.messages[index].content)
        else:
            entry_start = self.reply_start
        for call_id, call in self.reply_calls.items():
            if call_id in self.responses:
                self.messages.append(tool_message(self.responses[call_id]))
            elif (
                call.name in self.tools
                and self.tools[call.name].on_interrupt == "retain"
            ):
                self.running_calls[call_id] = call
            else:
                pass
        if is_after_calls:
            pass
        else:
            self.messages.append(ChatMessage(role="assistant"))
            index = len(self.messages) - 1
        if text[entry_start:]:
            self.truncatable_text = text
            self.truncatable_start = entry_start
            self.is_truncatable_interrupted = True
            self.truncatable = (turn, index)
            self.truncatable_progress, self.progress = self.progress, []
            self.delivered_ms = None
            self.heard_ms = None
            self.truncate()
        else:
            pass
        self.reply = None
        self.reply_calls = {}
        self.responses = {}
        self.emit(control_chunk(INTERRUPT, turn=turn))
        self.emit(control_chunk(TURN_INTERRUPTED, turn=turn))

    def respond_to_waiting(self) -> None:
        assert self.emit is not None
        # note (luojiaxuan): The next prompt shows the interrupted reply's cut, which has no bound until the graph reports the audio it delivered or a truncate arrives.
        while self.reply is None and (
            not self.awaits_delivery
            or self.delivered_ms is not None
            or self.heard_ms is not None
        ):
            if self.waiting:
                outputs = self.respond(*self.waiting.popleft())
            elif self.arrived and not self.is_user_speaking:
                # Results report once no utterance waits, and not over the user's speech, where a reply would open and end at once.
                self.turns += 1
                outputs = [
                    control_chunk(TURN_QUEUED, turn=self.turns),
                    *self.report(self.turns),
                ]
            else:
                break
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
            kept = content[self.truncatable_start : text_end].rstrip()
            self.messages[index] = self.messages[index].model_copy(
                update={
                    "content": kept,
                    "uncertain": uncertain,
                    "is_cut": self.is_truncatable_interrupted
                    or kept != content[self.truncatable_start :].rstrip(),
                }
            )
            if self.heard_ms is not None:
                self.emit(control_chunk(REPLY_HEARD, turn=turn, text_end=text_end))
            else:
                pass
            if (
                not kept
                and not self.messages[index].tool_calls
                and index < len(self.messages) - 1
            ):
                self.join_around(index)
            else:
                pass

    def join_around(self, index: int) -> None:
        """Drop the reply at index, which the history keeps nothing of, and join the user messages around it: nothing of the reply was heard, so the user kept talking, and an empty assistant entry would only show the model an empty answer to copy. After a reply that called tools, the calls and their results stay in the history and only the empty entry goes."""
        if self.messages[index - 1].role == "user":
            self.messages[index - 1] = ChatMessage(
                role="user",
                content=f"{self.messages[index - 1].content} {self.messages[index + 1].content}",
            )
            del self.messages[index : index + 2]
        else:
            del self.messages[index]
        if self.truncatable is not None and self.truncatable[1] == index:
            self.truncatable = None
        else:
            pass

    def respond(self, turn: int, text: str, language: str | None) -> list[TimedChunk]:
        if not text:
            return [control_chunk(TURN_EMPTY, turn=turn)]
        elif len(text) > self.config.max_text_chars:
            raise ValueError("transcript exceeds max_text_chars")
        else:
            self.messages.append(ChatMessage(role="user", content=text))
            if (
                self.messages[-2].role == "assistant"
                and not self.messages[-2].content
                and not self.messages[-2].tool_calls
            ):
                self.join_around(len(self.messages) - 2)
            else:
                pass
            if language is not None:
                self.language = language
            else:
                pass
            return self.open_reply(turn)

    def report(self, turn: int) -> list[TimedChunk]:
        """Open the turn that reports the tool results arrived so far, which join the history in arrival order."""
        assert self.arrived
        self.messages.extend(tool_message(response) for response in self.arrived)
        self.arrived = []
        return self.open_reply(turn)

    def open_reply(self, turn: int) -> list[TimedChunk]:
        """Open the turn's reply to the history as it stands, which must leave room for the reply's text."""
        history_chars = sum(len(message.content) for message in self.messages)
        if history_chars + self.config.max_text_chars > self.config.max_history_chars:
            raise ValueError(
                "conversation exceeds max_history_chars; open a new session"
            )
        else:
            pass
        self.reply = []
        self.reply_turn = turn
        self.reply_start = 0
        self.reply_calls = {}
        self.responses = {}
        self.progress = []
        return [
            control_chunk(TURN_STARTED, turn=turn),
            messages_chunk(self.prompt(), turn, self.language),
        ]

    def prompt(self) -> list[ChatMessage]:
        """The history as the model reads it, with the calls still running, and those whose results wait for their turn, shown as running."""
        return prompt_messages(
            self.messages,
            frozenset(self.reply_calls)
            | frozenset(self.running_calls)
            | frozenset(response.id for response in self.arrived),
        )
