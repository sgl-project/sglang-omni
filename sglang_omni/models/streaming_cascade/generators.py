# SPDX-License-Identifier: Apache-2.0
"""Generator kinds: reply text, or the tool calls the model makes, from the conversation."""

import json
import re
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Literal

import httpx
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from sglang_omni.models.streaming_cascade.base import (
    ChatMessage,
    ComponentConfig,
    Replier,
    Replies,
    bearer_headers,
    control_chunk,
    message_dict,
    messages_of,
    probe_openai_service,
    replying,
    text_chunk,
)
from sglang_omni.models.streaming_cascade.synthesizers import (
    OpenAISpeechSynthesizerConfig,
)
from sglang_omni.proto.session import TimedChunk
from sglang_omni.proto.streams import (
    MESSAGES,
    REPLY_TRUNCATED,
    TEXT_DELTA,
    TURN_DONE,
    chunk_language,
)
from sglang_omni.proto.tools import ToolCall, tool_call_chunk
from sglang_omni.scheduling.session import ChunkEmitter

if TYPE_CHECKING:
    from sglang_omni.models.streaming_cascade.config import ComponentGraphConfig
else:
    pass

# The request fields this kind sets itself; `extra_body` adds others.
CHAT_REQUEST_FIELDS = frozenset(
    {
        "model",
        "messages",
        "stream",
        "max_tokens",
        "temperature",
        "tools",
        "tool_choice",
        "continue_final_message",
    }
)
WARM_UP_MESSAGES = [ChatMessage(role="user", content="Hi")]


class ToolFunction(BaseModel):
    """One function the model may call, as the chat API's function object."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    description: str = ""
    parameters: dict[str, object] | None = None


class OpenAIChatGeneratorConfig(ComponentConfig):
    """tools are the functions the model may call; a completion that calls them ends in a tool_call chunk instead of turn.done.

    required_call_opening is the text the model writes to open the call a prompt requires, such as the qwen3_coder tags that open a tts_start call up to its text; it goes as an assistant message the model continues, so decoding starts at the call's argument.
    """

    kind: Literal["openai_chat"] = "openai_chat"
    base_url: str = Field(pattern=r"^https?://[^/]+")
    model: str
    max_tokens: int = Field(default=512, gt=0)
    temperature: float | None = Field(default=None, ge=0, le=2)
    api_key_env: str = "OPENAI_API_KEY"
    extra_body: dict[str, object] = Field(default_factory=dict)
    constrain_style_tags: bool = False
    tools: list[ToolFunction] = Field(default_factory=list)
    required_call_opening: str = ""

    @field_validator("tools")
    @classmethod
    def check_tool_names(cls, tools: list[ToolFunction]) -> list[ToolFunction]:
        names = [tool.name for tool in tools]
        if len(set(names)) != len(names):
            raise ValueError(f"tools must have distinct names, got {names}")
        else:
            return tools

    @field_validator("extra_body")
    @classmethod
    def check_extra_body(cls, extra_body: dict[str, object]) -> dict[str, object]:
        if CHAT_REQUEST_FIELDS & set(extra_body):
            raise ValueError(
                f"extra_body cannot set {sorted(CHAT_REQUEST_FIELDS & set(extra_body))}"
            )
        else:
            return extra_body

    @model_validator(mode="after")
    def check_style_tag_regex(self) -> "OpenAIChatGeneratorConfig":
        if self.constrain_style_tags and "regex" in self.extra_body:
            raise ValueError(
                "extra_body cannot set regex when constrain_style_tags builds it"
            )
        else:
            return self


class FunctionDelta(BaseModel):
    name: str | None = None
    arguments: str | None = None


class ToolCallDelta(BaseModel):
    """One fragment of a streamed tool call; index joins the fragments of one call, whose id and name come first and whose arguments arrive as text pieces."""

    index: int
    id: str | None = None
    function: FunctionDelta | None = None


class ChatDelta(BaseModel):
    content: str | None = None
    tool_calls: list[ToolCallDelta] | None = None


class ChatChoice(BaseModel):
    delta: ChatDelta
    finish_reason: str | None = None


class ChatEvent(BaseModel):
    choices: list[ChatChoice]


def parse_arguments(text: str) -> object:
    """The arguments object the model wrote; text that is not JSON, such as a call the token budget cut, stays as it is, so the model is answered with an error instead of the node failing."""
    try:
        return json.loads(text) if text else {}
    except json.JSONDecodeError:
        return text


class OpenAIChatGenerator(Replier):
    """One streamed completion per messages chunk, closed by turn.done, or by a tool_call chunk when the model called tools; the deltas carry the messages chunk's language.

    A tag such as [calm] naming a style of a synthesizer that reads the
    deltas leaves the reply text, and the deltas after it carry that style.
    """

    def __init__(self, config: OpenAIChatGeneratorConfig) -> None:
        super().__init__()
        self.config = config
        self.client = httpx.AsyncClient(
            timeout=config.timeout_s, headers=bearer_headers(config.api_key_env)
        )
        self.styles: list[str] = []

    def place(self, settings: "ComponentGraphConfig", name: str) -> None:
        self.styles = sorted(
            {
                style
                for node in settings.nodes
                if isinstance(node.component, OpenAISpeechSynthesizerConfig)
                and TEXT_DELTA in node.edges().get(name, frozenset())
                for style in node.component.styles
            }
        )

    async def process(
        self, inputs: AsyncIterator[TimedChunk], emit: ChunkEmitter
    ) -> None:

        async def handle(chunk: TimedChunk, replies: Replies) -> None:
            if chunk.format == MESSAGES:
                await replies.finish()
                replies.start(
                    chunk.turn,
                    self.reply(
                        messages_of(chunk),
                        chunk.turn,
                        chunk_language(chunk),
                        emit,
                        chunk.payload.get("tool_choice"),
                    ),
                )
            else:
                pass

        await replying(inputs, handle, self.replies)

    async def reply(
        self,
        messages: list[ChatMessage],
        turn: int,
        language: str | None,
        emit: ChunkEmitter,
        tool_choice: str | None = None,
    ) -> None:
        """Stream the reply's text deltas, each carrying the user's language and the style the last tag named for the synthesizer; a completion that called tools ends in their tool_call chunk and leaves the turn to the node that reads it, any other in turn.done. Either end carries REPLY_TRUNCATED when the completion stopped at max_tokens."""
        # note (Haoling Pu): Qwen3.8-27B wrote [Sympathetic] in 6 of 740 replies, which a case-sensitive match left in the text and the speech service spoke.
        style_names = {name.lower(): name for name in self.styles}
        style_tag = re.compile(
            r"\[(" + "|".join(map(re.escape, self.styles)) + r")\]", re.IGNORECASE
        )
        style: str | None = None
        partial_tag = ""
        is_after_tag = False
        ids: dict[int, str] = {}
        names: dict[int, str] = {}
        arguments: dict[int, list[str]] = {}
        is_truncated = False
        async for choice in self.complete(messages, tool_choice):
            delta = choice.delta
            if choice.finish_reason == "length":
                is_truncated = True
            else:
                pass
            received = partial_tag + (delta.content or "")
            tag_start = received.rfind("[")
            tail = received[tag_start:].lower()
            if (
                tag_start >= 0
                and not tail.endswith("]")
                and any(f"[{name}]".startswith(tail) for name in style_names)
            ):
                received, partial_tag = received[:tag_start], received[tag_start:]
            else:
                partial_tag = ""
            pieces = style_tag.split(received) if self.styles else [received]
            for index, piece in enumerate(pieces):
                if index % 2 == 1:
                    style = style_names[piece.lower()]
                    is_after_tag = True
                elif is_after_tag and not piece.strip():
                    pass
                elif is_after_tag:
                    emit(text_chunk(piece.lstrip(), turn, language, style))
                    is_after_tag = False
                elif piece:
                    emit(text_chunk(piece, turn, language, style))
                else:
                    pass
            for fragment in delta.tool_calls or ():
                function = fragment.function or FunctionDelta()
                if fragment.id:
                    ids[fragment.index] = fragment.id
                else:
                    pass
                if function.name:
                    names[fragment.index] = function.name
                else:
                    pass
                if function.arguments:
                    arguments.setdefault(fragment.index, []).append(function.arguments)
                else:
                    pass
                # The first call's arguments go out as they arrive, so a dialog that speaks through it can start before the completion ends.
                if function.arguments and fragment.index == 0 and 0 in names:
                    emit(
                        tool_call_chunk(
                            [
                                ToolCall(
                                    id=ids.get(0, f"call_{turn}_0"),
                                    name=names[0],
                                    arguments=function.arguments,
                                )
                            ],
                            turn,
                            is_partial=True,
                        )
                    )
                else:
                    pass
        if partial_tag:
            emit(text_chunk(partial_tag, turn, language, style))
        else:
            pass
        calls = [
            ToolCall(
                id=ids.get(index, f"call_{turn}_{index}"),
                name=name,
                arguments=parse_arguments("".join(arguments.get(index, []))),
            )
            for index, name in sorted(names.items())
        ]
        if calls:
            emit(tool_call_chunk(calls, turn, is_truncated=is_truncated))
        elif is_truncated:
            emit(control_chunk(TURN_DONE, turn=turn, **{REPLY_TRUNCATED: True}))
        else:
            emit(control_chunk(TURN_DONE, turn=turn))

    def request_body(
        self, messages: list[ChatMessage], tool_choice: str | None = None
    ) -> dict[str, object]:
        message_dicts = [message_dict(message) for message in messages]
        body: dict[str, object] = {
            "model": self.config.model,
            "messages": message_dicts,
            "stream": True,
            "max_tokens": self.config.max_tokens,
        }
        if self.config.temperature is not None:
            body["temperature"] = self.config.temperature
        else:
            pass
        if self.config.constrain_style_tags and self.styles:
            # note (Haoling Pu): built from the styles where the node is placed, so a configure that changes them changes the regex too; a non-whitespace character must follow the tag, since a reply of the tag alone would stream no text and the turn would be silent.
            body["regex"] = (
                r"\[("
                + "|".join(map(re.escape, self.styles))
                + r")\] [^\[\]\s][^\[\]]*"
            )
        else:
            pass
        if self.config.tools:
            body["tools"] = [
                {
                    "type": "function",
                    "function": tool.model_dump(exclude_none=True),
                }
                for tool in self.config.tools
            ]
        else:
            pass
        if tool_choice == "required" and self.config.required_call_opening:
            body["messages"] = [
                *message_dicts,
                {"role": "assistant", "content": self.config.required_call_opening},
            ]
            body["continue_final_message"] = True
            # Note (wenyao): the opening already commits to the call, and the grammar of a required choice would make the model open a second one.
            body["tool_choice"] = "auto"
        elif tool_choice is not None:
            body["tool_choice"] = tool_choice
        else:
            pass
        return {**body, **self.config.extra_body}

    async def complete(
        self, messages: list[ChatMessage], tool_choice: str | None = None
    ) -> AsyncIterator[ChatChoice]:
        finished = False
        async with self.client.stream(
            "POST",
            f"{self.config.base_url.rstrip('/')}/chat/completions",
            json=self.request_body(messages, tool_choice),
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line.startswith("data:"):
                    continue
                else:
                    pass
                encoded = line[5:].strip()
                if encoded == "[DONE]":
                    if not finished:
                        raise RuntimeError("LLM ended without a stop finish reason")
                    else:
                        break
                else:
                    pass
                event = ChatEvent.model_validate_json(encoded)
                for choice in event.choices:
                    if choice.finish_reason is None:
                        pass
                    elif choice.finish_reason not in ("stop", "length", "tool_calls"):
                        raise RuntimeError(f"LLM stopped with {choice.finish_reason}")
                    else:
                        finished = True
                    yield choice
            if not finished:
                raise RuntimeError("LLM stream ended before completion")
            else:
                pass

    async def close(self) -> None:
        await self.client.aclose()


async def probe_chat_service(config: OpenAIChatGeneratorConfig) -> None:
    """Require the service to list its models and to stream one token sampled with the node's settings."""
    await probe_openai_service(config)
    # Note (ruoyu): The chat service warms itself up with a greedy prefill, so a fresh deployment compiled its sampling kernel on the first real request and answered it 72 s late; one sampled token here moves that into startup.
    warm_up_body = dict(config.extra_body)
    if "min_tokens" in warm_up_body:
        warm_up_body["min_tokens"] = 0
    else:
        pass
    generator = OpenAIChatGenerator(
        config.model_copy(update={"max_tokens": 1, "extra_body": warm_up_body})
    )
    completions_url = f"{config.base_url.rstrip('/')}/chat/completions"
    try:
        async for _ in generator.complete(WARM_UP_MESSAGES):
            pass
    except httpx.HTTPStatusError as error:
        raise RuntimeError(
            f"{completions_url} answered {error.response.status_code}"
        ) from error
    except httpx.TransportError as error:
        raise RuntimeError(f"cannot reach {completions_url}: {error!r}") from error
    finally:
        await generator.close()
