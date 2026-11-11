# SPDX-License-Identifier: Apache-2.0
"""The dialog kind: the conversation history and the turns."""

import json
import logging
import math
import re
from collections import deque
from collections.abc import AsyncIterator
from dataclasses import dataclass
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
from sglang_omni.proto.session import TimedChunk
from sglang_omni.proto.streams import (
    COMMIT,
    INTERRUPT,
    MAX_HISTORY_CHARS,
    REPLY_DELIVERED,
    REPLY_HEARD,
    REPLY_PROGRESS,
    SPEECH_ERROR,
    SPEECH_STARTED,
    SPEECH_STOPPED,
    TEXT_DELTA,
    TOOL_CALL,
    TRANSCRIPT_FORMAT,
    TRUNCATE,
    TURN_DONE,
    TURN_EMPTY,
    TURN_INTERRUPTED,
    TURN_QUEUED,
    TURN_STARTED,
    chunk_language,
    chunk_style,
    control_type,
)
from sglang_omni.proto.tools import (
    ToolCall,
    is_partial_call,
    tool_call_chunk,
    tool_calls_of,
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


# note (Haoling Pu): a short utterance or a noise ("嗯。", "Okay.") is often recognized in the wrong language, so it neither changes the session language nor leads the reply into its language.
MIN_LANGUAGE_WORDS = 3
MIN_LANGUAGE_CJK_CHARS = 3
CJK_CHAR = re.compile("[\u3040-\u30ff\u4e00-\u9fff\uac00-\ud7af]")
SHORT_UTTERANCE_NOTE = "Reply in {language}, the language of this conversation; the user's last message is too short to tell their language."
# note (Haoling Pu): a cut reply shown as is reads to the model as how it answers, and after a few interruptions it replies with fragments (#3063).
CUT_REPLY_NOTE = "A reply of yours that ends with … was cut off when the user interrupted it; answer the latest message in full."
# A second call of the speech tool while one is spoken is refused rather than queued or overlapped.
SPEECH_BUSY = "BUSY"
# note (Jingwen): only when a speech was cut, since a longer note of rules on every prompt took Qwen3.8-27B from 7 of 8 to 4 of 8 samples following a client's new instructions.
CUT_SPEECH_NOTE = 'A tool result of {"status": "interrupted", "heard_prefix": ...} means the user cut you off and heard only heard_prefix.'
# note (Jingwen): without this Qwen3.8-27B said the whole text again after a speech that failed part-way in 13 of 20 samples, and with it went on from heard_prefix in 20 of 20.
FAILED_SPEECH_NOTE = 'A tool result of {"status": "failed", "heard_prefix": ...} means your speech broke off after the user heard heard_prefix; say only what comes after it.'
# note (Jingwen): with thinking off Qwen3.8-27B called the speech tool in 12 of 24 first prompts without this and in 36 of 36 with it, and no prompt wording closed the gap. Every prompt uses it, so the model is never prompted once it has spoken for the utterance: with a `silent` tool or a `silent` action to choose instead, it repeated its answer after a completed speech in 10 of 49 and 22 of 50 samples.
SPEECH_TOOL_CHOICE = "required"


def tagged(content: str, style_starts: list[tuple[int, str]]) -> str:
    """The reply with the tag [style] back before the first text of each style it keeps."""
    for text_start, style in reversed(style_starts):
        if text_start < len(content):
            content = f"{content[:text_start]}[{style}] {content[text_start:]}"
        else:
            pass
    return content


def prompt_messages(messages: list[ChatMessage]) -> list[ChatMessage]:
    """The history as the model reads it: a reply the listener cut ends with … under a system note, only such a reply ends with …, and a reply shows the style tags the model wrote in it; a speech the listener cut is explained by a system note too."""
    prompt: list[ChatMessage] = []
    notes: list[str] = []
    for message in messages:
        if message.role == "tool":
            prompt.append(message)
            is_failed = '"status": "failed"' in message.content
            note = FAILED_SPEECH_NOTE if is_failed else CUT_SPEECH_NOTE
            if message.is_cut and note not in notes:
                notes.append(note)
            else:
                pass
        elif message.is_cut:
            content = message.content.removesuffix("…")
            prompt.append(
                ChatMessage(
                    role="assistant",
                    content=f"{tagged(content, message.style_starts)}…",
                )
            )
            if CUT_REPLY_NOTE not in notes:
                notes.append(CUT_REPLY_NOTE)
            else:
                pass
        elif message.role == "assistant":
            # note (Haoling Pu): the model sometimes ends a finished reply with …, which would read as cut and was followed by an empty reply in a live run.
            content = message.content.rstrip().removesuffix("…")
            # note (Haoling Pu): decoding without a regex, under a prompt that asked for a tag only when a feeling fit, Qwen3.8-27B tagged 7 of 16 emotional turns of six-turn English conversations when its earlier replies were shown without their tags, against 16 of 16 with them.
            prompt.append(
                message.model_copy(
                    update={"content": tagged(content, message.style_starts)}
                )
            )
        else:
            prompt.append(message)
    if notes:
        prompt[0] = ChatMessage(
            role="system", content=" ".join([prompt[0].content, *notes])
        )
    else:
        pass
    return prompt


def message_chars(message: ChatMessage) -> int:
    """The characters an entry adds to the prompt: its text and the arguments of its calls, which hold the text of a speech."""
    return len(message.content) + sum(
        len(call.function.arguments) for call in message.tool_calls or ()
    )


def assistant_tool_call(call: ToolCall) -> AssistantToolCall:
    """A call as the history shows it; arguments the model did not write as a JSON object, which the chat template cannot render, show as none."""
    arguments = call.arguments if isinstance(call.arguments, dict) else {}
    return AssistantToolCall(
        id=call.id,
        function=FunctionCall(
            name=call.name, arguments=json.dumps(arguments, ensure_ascii=False)
        ),
    )


def tool_message(
    tool_call_id: str, result: dict[str, str], is_cut: bool = False
) -> ChatMessage:
    return ChatMessage(
        role="tool",
        content=json.dumps(result, ensure_ascii=False),
        tool_call_id=tool_call_id,
        is_cut=is_cut,
    )


@dataclass(kw_only=True)
class Speech:
    """The text being spoken and the history entry that answers it: the tool entry of a call, or the assistant entry of text the model wrote without calling the tool."""

    text: str
    entry_index: int
    # True while the model is still writing the call: `text` is what reached the synthesizer so far.
    is_streaming: bool = False


# The arguments of a call up to the opening quote of its `text` value, with any scalar arguments written before it.
TEXT_VALUE_START = re.compile(
    r'\s*\{(?:\s*"(?:[^"\\]|\\.)*"\s*:\s*(?:"(?:[^"\\]|\\.)*"|[^,{}\[\]"]+)\s*,)*\s*"text"\s*:\s*"'
)
HIGH_SURROGATE_ESCAPE = re.compile(r"\\u[dD][89abAB][0-9a-fA-F]{2}")


class SpokenTextReader:
    """Reads the `text` value out of a speech call's arguments while the model is still writing them."""

    def __init__(self) -> None:
        self.raw = ""
        self.text = ""
        self.read_end: int | None = None
        self.is_closed = False

    def feed(self, fragment: str) -> None:
        """Add the next piece of the arguments; `text` grows by what it completes."""
        self.raw += fragment
        if self.read_end is None:
            match = TEXT_VALUE_START.match(self.raw)
            if match is None:
                return
            else:
                self.read_end = match.end()
        else:
            pass
        end = self.read_end
        # Where the first half of a character written as two escapes begins, while its second half may still be on its way.
        held: int | None = None
        while not self.is_closed:
            if end >= len(self.raw):
                break
            elif self.raw[end] == '"':
                self.is_closed = True
            else:
                if self.raw[end] != "\\":
                    step = 1
                elif self.raw[end + 1 : end + 2] == "u":
                    step = 6
                else:
                    step = 2
                if end + step > len(self.raw):
                    break
                else:
                    is_high = step == 6 and HIGH_SURROGATE_ESCAPE.match(
                        self.raw, end, end + 6
                    )
                    held = end if is_high else None
                    end += step
        if held is not None and not self.is_closed:
            end = held
        else:
            pass
        try:
            self.text += json.loads(f'"{self.raw[self.read_end:end]}"')
        except json.JSONDecodeError:
            self.is_closed = True
        self.read_end = end


class DialogConfig(ComponentConfig):
    """Conversation history; one messages chunk per answered utterance.

    `speech_tool` names the tool the generator speaks through; without it the synthesizer reads the generator's text. `max_inferences_per_utterance` bounds how many times the model is prompted for one utterance, so calls that keep being refused or failing cannot go on forever.
    """

    kind: Literal["dialog"] = "dialog"
    system_prompt: str = "You are a helpful voice assistant. Answer concisely."
    max_history_chars: int = Field(default=MAX_HISTORY_CHARS, gt=0)
    max_text_chars: int = Field(default=16000, gt=0)
    speech_tool: str | None = None
    max_inferences_per_utterance: int = Field(default=4, gt=0)


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
    truncate bounds the cut, so its prompt holds what could have been heard of
    it, and these reports keep cutting the ended reply until a later reply
    ends, other than one interrupted before its first text, so one that
    arrives after the next reply began still cuts the history later prompts
    read. A reply the history keeps nothing of leaves no entry once an
    utterance follows it, and that utterance joins the user message the reply
    answered. The language the recognizer heard in the last utterance of at
    least 3 words or 3 Chinese, Japanese or Korean characters is the
    Session's language, and every messages chunk carries it; a shorter
    utterance leaves it unchanged, and when the recognizer heard another
    language in it, that turn's latest user message asks for a reply in the
    Session's language. A replacement adopts the history, the open reply and
    the waiting utterances, so new settings apply at once and reach the next
    prompt.

    A reply whose deltas carry a style shows the model, in later prompts,
    the tag [style] before the first text of each style; the history,
    its cuts and their offsets count only the reply text.

    Speech that started after an utterance ended and has not stopped when
    the utterance's reply begins is speech over that reply, so the reply
    ends as soon as it begins.

    With a `speech_tool`, the model speaks by calling it: the dialog hands the
    one call it accepts to the synthesizer, the Session speaks one at a time
    and every other call is answered at once with an error. Each speech is a
    turn, so the turn fence, the paced turn.done, the reply's audio timeline
    and the client's truncate all apply to it as to a reply. The call's answer
    enters the history as `{"status": "interrupted", "heard_prefix": ...}`
    until the speech ends, and every later report of what was heard rewrites
    it, so the next prompt holds the last known position. Every prompt
    requires a tool call, so the model speaks once for an utterance and is not
    prompted again when that speech completes: it reads the answer in the
    prompt for the next utterance. It is prompted again, with no turn
    announced, only when a call was refused or its speech failed, unless the
    user is speaking or an utterance waits; the turn is announced only if the
    model then speaks. The speech starts with the first words of the call's
    text while the model still writes the rest. Text the model writes without
    a call is spoken as it is, so the user does not sit through a silent
    reply. New instructions are repeated beside the next utterance in its
    prompt only, since a long past speech otherwise outweighs them.
    """

    def __init__(self, config: DialogConfig) -> None:
        self.config = config
        self.emit: ChunkEmitter | None = None
        self.messages = [ChatMessage(role="system", content=config.system_prompt)]
        # Waiting utterances as (turn, transcript, language the recognizer heard).
        self.waiting: deque[tuple[int, str, str | None]] = deque()
        self.reply: list[str] | None = None
        # note (Haoling Pu): the open reply's style changes, each as (characters of the reply before it, style).
        self.reply_style_starts: list[tuple[int, str]] = []
        self.reply_turn = 0
        self.turns = 0
        # The speech the synthesizer holds: the TTS status of the Session, on while it is not None.
        self.speech: Speech | None = None
        # The latest turn the client was told about; a prompt after a speech stays unannounced until the model speaks.
        self.announced_turn = 0
        self.should_follow_up = False
        self.inference_count = 0
        # Reads the text of the call the model is writing; None between calls.
        self.text_reader: SpokenTextReader | None = None
        # note (Jingwen): with a long past speech in the history and a call required, Qwen3.8-27B followed a client's new instructions in 3 of 12 samples, and in 10 of 12 with them repeated beside the next utterance.
        self.has_new_instructions = False
        # Reply audio produced so far as (characters of the reply text with audio, audio ms), one entry per spoken sentence, for truncating an interrupted reply to what was heard.
        self.progress: list[tuple[int, float]] = []
        # note (luojiaxuan): The ended reply a truncate, its delivered audio or its progress may still cut, as (turn, history index), until a later reply ends, other than one interrupted before its first text, or its entry leaves the history.
        self.truncatable: tuple[int, int] | None = None
        # Note (Junnan Li): The ended reply's full text, the reply audio the graph delivered (unknown, or all of it once the reply is done) and the last truncate's heard position, so progress that arrives later re-cuts it.
        self.truncatable_text = ""
        self.truncatable_style_starts: list[tuple[int, str]] = []
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
            self.reply_style_starts = previous.reply_style_starts
            self.reply_turn = previous.reply_turn
            self.turns = previous.turns
            self.speech = previous.speech
            self.announced_turn = previous.announced_turn
            self.should_follow_up = previous.should_follow_up
            self.inference_count = previous.inference_count
            self.text_reader = previous.text_reader
            self.has_new_instructions = (
                previous.has_new_instructions
                or self.config.system_prompt != previous.config.system_prompt
            )
            self.progress = previous.progress
            self.truncatable = previous.truncatable
            self.truncatable_text = previous.truncatable_text
            self.truncatable_style_starts = previous.truncatable_style_starts
            self.is_truncatable_interrupted = previous.is_truncatable_interrupted
            self.truncatable_progress = previous.truncatable_progress
            self.delivered_ms = previous.delivered_ms
            self.heard_ms = previous.heard_ms
            self.language = previous.language
            self.is_user_speaking = previous.is_user_speaking
        else:
            pass

    def state_bytes(self) -> int:
        chars = sum(message_chars(message) for message in self.messages)
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
            style = chunk_style(chunk)
            if style is not None and (
                not self.reply_style_starts or self.reply_style_starts[-1][1] != style
            ):
                self.reply_style_starts.append(
                    (sum(len(part) for part in self.reply), style)
                )
            else:
                pass
            self.reply.append(str(chunk.payload["text"]))
            if sum(len(part) for part in self.reply) > self.config.max_text_chars:
                raise ValueError("reply exceeds max_text_chars")
            else:
                pass
        elif chunk.format == TOOL_CALL and self.is_reply(chunk):
            if is_partial_call(chunk):
                self.stream_speech(tool_calls_of(chunk)[0])
            else:
                self.start_speech(tool_calls_of(chunk))
        elif control_type(chunk) == TURN_DONE and self.is_reply(chunk):
            if self.speech is not None:
                self.end_speech(chunk)
            elif self.config.speech_tool is not None:
                self.end_inference()
            else:
                reply = "".join(self.reply)
                self.messages.append(
                    ChatMessage(
                        role="assistant",
                        content=reply,
                        style_starts=self.reply_style_starts,
                    )
                )
                self.end_reply(
                    len(self.messages) - 1, reply, self.reply_style_starts
                )
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
            turn = self.announced_turn if chunk.turn is None else chunk.turn
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

    def end_reply(
        self, entry_index: int, text: str, style_starts: list[tuple[int, str]]
    ) -> None:
        """The reply whose text is `text` reached the client whole; until a later reply ends, a truncate can still cut its history entry."""
        self.truncatable = (self.reply_turn, entry_index)
        self.truncatable_text = text
        self.truncatable_style_starts = style_starts
        self.is_truncatable_interrupted = False
        self.truncatable_progress, self.progress = self.progress, []
        self.delivered_ms = math.inf
        self.heard_ms = None
        self.reply = None

    def speech_call_error(self, call: ToolCall, is_busy: bool) -> str | None:
        """Why the call cannot be spoken; None when it can."""
        text = call.arguments.get("text") if isinstance(call.arguments, dict) else None
        if call.name != self.config.speech_tool:
            return f"unknown tool {call.name}"
        elif is_busy:
            return SPEECH_BUSY
        elif not isinstance(text, str) or not text.strip():
            return "text must be a non-empty string"
        elif len(text) > self.config.max_text_chars:
            return f"text exceeds {self.config.max_text_chars} characters"
        else:
            return None

    def start_speech(self, calls: list[ToolCall]) -> None:
        """Record the model's calls and speak the first one that can be; any other is answered at once, so the next prompt holds one answer per call."""
        assert self.reply is not None
        self.text_reader = None
        if self.speech is not None and self.speech.is_streaming:
            self.finish_streamed_speech(calls)
            return
        else:
            pass
        self.messages.append(
            ChatMessage(
                role="assistant",
                content="".join(self.reply),
                tool_calls=[assistant_tool_call(call) for call in calls],
            )
        )
        self.reply = []
        spoken: ToolCall | None = None
        for call in calls:
            error = self.speech_call_error(call, spoken is not None)
            if error is None:
                spoken = call
                self.messages.append(
                    tool_message(call.id, {"status": "interrupted", "heard_prefix": ""})
                )
                self.speech = Speech(
                    text=call.arguments["text"], entry_index=len(self.messages) - 1
                )
            else:
                logger.warning(f"turn {self.reply_turn}: {call.name} refused: {error}")
                self.messages.append(
                    tool_message(call.id, {"status": "failed", "error": error})
                )
        if spoken is None:
            self.should_follow_up = True
            self.end_silent_inference()
        else:
            self.speak(spoken)

    def announce_turn(self) -> None:
        """Tell the client of a turn it has not heard of, now that the model speaks in it."""
        assert self.emit is not None
        if self.announced_turn != self.reply_turn:
            self.announced_turn = self.reply_turn
            self.emit(control_chunk(TURN_QUEUED, turn=self.reply_turn))
            self.emit(control_chunk(TURN_STARTED, turn=self.reply_turn))
        else:
            pass

    def speak(self, call: ToolCall) -> None:
        """Hand the call to the synthesizer, announcing a turn the client has not heard of."""
        assert self.emit is not None
        self.announce_turn()
        self.emit(tool_call_chunk([call], self.reply_turn, self.language))

    def stream_speech(self, call: ToolCall) -> None:
        """One more piece of the arguments of the call the model is writing. The speech starts with the first words of its text and each later piece follows them to the synthesizer, so the first sentence is spoken while the model writes the rest."""
        assert self.emit is not None and self.reply is not None
        if call.name != self.config.speech_tool or not isinstance(call.arguments, str):
            return
        elif self.speech is not None and not self.speech.is_streaming:
            return
        else:
            pass
        if self.text_reader is None:
            self.text_reader = SpokenTextReader()
        else:
            pass
        self.text_reader.feed(call.arguments)
        text = self.text_reader.text[: self.config.max_text_chars]
        if self.speech is None and not text.strip():
            return
        elif self.speech is None:
            self.messages.append(
                ChatMessage(role="assistant", content="".join(self.reply))
            )
            self.messages.append(
                tool_message(call.id, {"status": "interrupted", "heard_prefix": ""})
            )
            self.reply = []
            self.speech = Speech(
                text="", entry_index=len(self.messages) - 1, is_streaming=True
            )
            self.announce_turn()
        else:
            pass
        speech = self.speech
        delta = text[len(speech.text) :]
        if delta:
            speech.text = text
            self.record_streamed_calls(
                [ToolCall(id=call.id, name=call.name, arguments={"text": text})]
            )
            self.emit(
                tool_call_chunk(
                    [ToolCall(id=call.id, name=call.name, arguments={"text": delta})],
                    self.reply_turn,
                    self.language,
                    is_partial=True,
                )
            )
        else:
            pass

    def record_streamed_calls(self, calls: list[ToolCall]) -> None:
        """Show the calls of the speech being streamed in its assistant entry, which sits before the call's answer."""
        assert self.speech is not None
        index = self.speech.entry_index - 1
        self.messages[index] = self.messages[index].model_copy(
            update={"tool_calls": [assistant_tool_call(call) for call in calls]}
        )

    def finish_streamed_speech(self, calls: list[ToolCall]) -> None:
        """The model finished the call whose text is already being spoken: the synthesizer is told the text is complete, and any other call is answered at once. A call the token budget cut has no arguments to read, so what was spoken is its text."""
        assert self.emit is not None and self.speech is not None
        speech = self.speech
        first = calls[0]
        arguments = first.arguments if isinstance(first.arguments, dict) else {}
        whole = arguments.get("text")
        if isinstance(whole, str) and whole.startswith(speech.text):
            text = whole[: self.config.max_text_chars]
        else:
            text = speech.text
        rest = text[len(speech.text) :]
        spoken = ToolCall(
            id=first.id, name=first.name, arguments={**arguments, "text": text}
        )
        self.record_streamed_calls([spoken, *calls[1:]])
        for call in calls[1:]:
            error = self.speech_call_error(call, True)
            logger.warning(f"turn {self.reply_turn}: {call.name} refused: {error}")
            self.messages.append(
                tool_message(call.id, {"status": "failed", "error": error})
            )
        speech.text = text
        speech.is_streaming = False
        if rest:
            self.emit(
                tool_call_chunk(
                    [ToolCall(id=first.id, name=first.name, arguments={"text": rest})],
                    self.reply_turn,
                    self.language,
                    is_partial=True,
                )
            )
        else:
            pass
        self.emit(tool_call_chunk([spoken], self.reply_turn, self.language))

    def end_speech(self, chunk: TimedChunk) -> None:
        """The speech's audio reached the client, or its synthesis failed; a call is answered, and a text the model wrote without a call ends the utterance's replies."""
        assert self.speech is not None
        speech = self.speech
        entry = self.messages[speech.entry_index]
        error = chunk.payload.get(SPEECH_ERROR)
        self.speech = None
        if entry.role != "tool":
            self.end_reply(speech.entry_index, speech.text, [])
        elif error is None:
            self.messages[speech.entry_index] = tool_message(
                entry.tool_call_id, {"status": "completed"}
            )
            self.end_reply(speech.entry_index, speech.text, [])
        else:
            result = {"status": "failed", "error": str(error)}
            # note (Jinyan Su): the sentences whose audio was complete before the service failed reached the client, so the model is told where to go on from.
            heard = (
                speech.text[: self.progress[-1][0]].rstrip() if self.progress else ""
            )
            if heard:
                result["heard_prefix"] = heard
            else:
                pass
            self.messages[speech.entry_index] = tool_message(
                entry.tool_call_id, result, is_cut=bool(heard)
            )
            self.progress = []
            self.reply = None
            self.should_follow_up = True

    def end_inference(self) -> None:
        """The model ended a completion without calling anything, as a service that ignores `tool_choice` lets it: what it wrote is spoken, and writing nothing leaves the turn silent."""
        assert self.reply is not None
        text = "".join(self.reply).strip()
        if text:
            logger.warning(
                f"turn {self.reply_turn}: the model wrote text without calling {self.config.speech_tool}; speaking it"
            )
            self.messages.append(ChatMessage(role="assistant", content=text))
            self.speech = Speech(text=text, entry_index=len(self.messages) - 1)
            self.reply = []
            self.speak(
                ToolCall(
                    id=f"call_{self.reply_turn}_text",
                    name=str(self.config.speech_tool),
                    arguments={"text": text},
                )
            )
        else:
            self.end_silent_inference()

    def end_silent_inference(self) -> None:
        """The prompt led to no speech: a turn the client was told about ends empty, and the history shows the unanswered utterance as `interrupt_reply` does."""
        assert self.emit is not None
        self.reply = None
        if self.messages[-1].role == "user":
            self.messages.append(ChatMessage(role="assistant"))
        else:
            pass
        if self.announced_turn == self.reply_turn:
            self.emit(control_chunk(TURN_EMPTY, turn=self.reply_turn))
        else:
            pass

    def follow_up(self) -> None:
        """Prompt the model again after a call was refused or its speech failed, with no turn announced, since the utterance has no spoken answer yet."""
        assert self.emit is not None
        self.should_follow_up = False
        if self.inference_count >= self.config.max_inferences_per_utterance:
            logger.warning(
                f"the model was prompted {self.inference_count} times for one utterance; not prompting it again"
            )
        else:
            self.turns += 1
            self.inference_count += 1
            self.text_reader = None
            self.reply = []
            self.reply_turn = self.turns
            self.progress = []
            self.emit(
                messages_chunk(
                    prompt_messages(self.messages),
                    self.reply_turn,
                    self.language,
                    SPEECH_TOOL_CHOICE,
                )
            )

    def interrupt_reply(self) -> None:
        """End the open reply; the history keeps none of it until the graph reports the audio it delivered or a truncate says what was heard. A reply without text, such as one interrupted as it opens, keeps nothing whatever was delivered, so the reply before it stays the one a later report cuts. With a speech tool only spoken text counts: the model's other text and a prompt that has not led to speech leave nothing, and the client hears of no turn it was not told about."""
        assert self.emit is not None and self.reply is not None
        turn = self.reply_turn
        entry_index = len(self.messages)
        style_starts: list[tuple[int, str]] = []
        if self.speech is not None:
            text = self.speech.text
            entry_index = self.speech.entry_index
        elif self.config.speech_tool is None:
            text = "".join(self.reply)
            style_starts = self.reply_style_starts
            self.messages.append(ChatMessage(role="assistant"))
        elif self.messages[-1].role == "user":
            text = ""
            self.messages.append(ChatMessage(role="assistant"))
        else:
            text = ""
        if text:
            self.truncatable_text = text
            self.truncatable_style_starts = style_starts
            self.is_truncatable_interrupted = True
            self.truncatable = (turn, entry_index)
            self.truncatable_progress, self.progress = self.progress, []
            self.delivered_ms = None
            self.heard_ms = None
            self.truncate()
        else:
            pass
        self.reply = None
        self.speech = None
        self.text_reader = None
        self.should_follow_up = False
        self.emit(control_chunk(INTERRUPT, turn=turn))
        if self.announced_turn == turn:
            self.emit(control_chunk(TURN_INTERRUPTED, turn=turn))
        else:
            pass

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
        ):
            for output in self.respond(*self.waiting.popleft()):
                self.emit(output)
            if self.reply is not None and self.is_user_speaking:
                self.interrupt_reply()
            else:
                pass
        if (
            self.should_follow_up
            and self.reply is None
            and not self.waiting
            and not self.is_user_speaking
        ):
            self.follow_up()
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
            is_cut = self.is_truncatable_interrupted or kept != content.rstrip()
            entry = self.messages[index]
            if entry.role != "tool":
                self.messages[index] = ChatMessage(
                    role="assistant",
                    content=kept,
                    uncertain=uncertain,
                    is_cut=is_cut,
                    style_starts=self.truncatable_style_starts,
                )
            elif is_cut:
                self.messages[index] = tool_message(
                    entry.tool_call_id,
                    {"status": "interrupted", "heard_prefix": kept},
                    is_cut=True,
                )
            else:
                pass
            if self.heard_ms is not None:
                self.emit(control_chunk(REPLY_HEARD, turn=turn, text_end=text_end))
            else:
                pass
            if not self.messages[index].content and index < len(self.messages) - 1:
                self.join_around(index)
            else:
                pass

    def join_around(self, index: int) -> None:
        """Drop the reply at index, which the history keeps nothing of, and join the user messages around it: nothing of the reply was heard, so the user kept talking, and an empty assistant entry would only show the model an empty answer to copy. A reply after an answered call has no user message to join."""
        if self.messages[index - 1].role == "user":
            self.messages[index - 1] = ChatMessage(
                role="user",
                content=f"{self.messages[index - 1].content} {self.messages[index + 1].content}",
            )
            del self.messages[index : index + 2]
            removed = 2
        else:
            self.messages.pop(index)
            removed = 1
        if self.truncatable is not None and self.truncatable[1] == index:
            self.truncatable = None
        elif self.truncatable is not None and self.truncatable[1] > index:
            self.truncatable = (self.truncatable[0], self.truncatable[1] - removed)
        else:
            pass
        # note (yxs): a late truncate can empty the reply before a speech that is still being spoken, and the speech's entry moves with the entries removed before it.
        if self.speech is not None and self.speech.entry_index > index:
            self.speech.entry_index -= removed
        else:
            pass

    def respond(self, turn: int, text: str, language: str | None) -> list[TimedChunk]:
        history_chars = sum(message_chars(message) for message in self.messages)
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
            if self.messages[-2].role == "assistant" and not self.messages[-2].content:
                self.join_around(len(self.messages) - 2)
            else:
                pass
            is_clear = (
                len(CJK_CHAR.findall(text)) >= MIN_LANGUAGE_CJK_CHARS
                or len(text.split()) >= MIN_LANGUAGE_WORDS
            )
            is_short_switch = (
                not is_clear
                and language is not None
                and self.language is not None
                and language != self.language
            )
            if language is not None and is_clear:
                self.language = language
            else:
                pass
            prompt = prompt_messages(self.messages)
            if is_short_switch:
                note = SHORT_UTTERANCE_NOTE.format(language=self.language)
                prompt[-1] = ChatMessage(
                    role="user", content=f"{prompt[-1].content}\n\n[{note}]"
                )
            else:
                pass
            if self.has_new_instructions and self.config.speech_tool is not None:
                prompt[-1] = ChatMessage(
                    role="user",
                    content=f"{prompt[-1].content}\n\n[{self.config.system_prompt}]",
                )
            else:
                pass
            self.has_new_instructions = False
            self.reply = []
            self.reply_style_starts = []
            self.reply_turn = turn
            self.announced_turn = turn
            self.progress = []
            self.should_follow_up = False
            self.inference_count = 1
            self.text_reader = None
            return [
                control_chunk(TURN_STARTED, turn=turn),
                messages_chunk(
                    prompt,
                    turn,
                    self.language,
                    (None if self.config.speech_tool is None else SPEECH_TOOL_CHOICE),
                ),
            ]
