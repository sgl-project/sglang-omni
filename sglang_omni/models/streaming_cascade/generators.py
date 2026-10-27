# SPDX-License-Identifier: Apache-2.0
"""Generator kinds: reply text, or the tool calls the model makes, from the conversation."""

import json
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Literal

import httpx
from pydantic import BaseModel, ConfigDict, Field, field_validator

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
from sglang_omni.proto.session import TimedChunk
from sglang_omni.proto.streams import MESSAGES, TURN_DONE, chunk_language
from sglang_omni.proto.tools import ToolCall, tool_call_chunk
from sglang_omni.scheduling.session import ChunkEmitter

if TYPE_CHECKING:
    from sglang_omni.models.streaming_cascade.config import ComponentGraphConfig
else:
    pass

# The request fields this kind sets itself; `extra_body` adds others.
CHAT_REQUEST_FIELDS = frozenset(
    {"model", "messages", "stream", "max_tokens", "temperature", "tools"}
)
WARM_UP_MESSAGES = [ChatMessage(role="user", content="Hi")]


class ToolFunction(BaseModel):
    """One function the model may call, as the chat API's function object; a tools node of the Session runs it by name."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    description: str = ""
    parameters: dict[str, object] | None = None
    strict: bool | None = None


class OpenAIChatGeneratorConfig(ComponentConfig):
    """`tools` are the functions the model may call; a reply that calls them ends in a tool_call chunk instead of turn.done."""

    kind: Literal["openai_chat"] = "openai_chat"
    base_url: str = Field(pattern=r"^https?://[^/]+")
    model: str
    max_tokens: int = Field(default=512, gt=0)
    temperature: float | None = Field(default=None, ge=0, le=2)
    api_key_env: str = "OPENAI_API_KEY"
    extra_body: dict[str, object] = Field(default_factory=dict)
    tools: list[ToolFunction] = Field(default_factory=list)

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


class FunctionDelta(BaseModel):
    name: str | None = None
    arguments: str | None = None


class ToolCallDelta(BaseModel):
    """One fragment of a streamed tool call; `index` joins the fragments of one call, whose id and name come first and whose arguments arrive as text pieces."""

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
    """The arguments object the model wrote; text that is not JSON stays as it is, so the tool runner answers the model with an error instead of the node failing."""
    try:
        return json.loads(text) if text else {}
    except json.JSONDecodeError:
        return text


class OpenAIChatGenerator(Replier):
    """One streamed completion per messages chunk, closed by turn.done, or by a tool_call chunk when the model called a tool the reply awaits; the deltas carry the messages chunk's language."""

    def __init__(self, config: OpenAIChatGeneratorConfig) -> None:
        super().__init__()
        self.config = config
        self.client = httpx.AsyncClient(
            timeout=config.timeout_s, headers=bearer_headers(config.api_key_env)
        )
        # The tools whose calls end the reply instead of holding it open for their results.
        self.unawaited: frozenset[str] = frozenset()

    def place(self, settings: "ComponentGraphConfig", name: str) -> None:
        self.unawaited = frozenset(
            tool_name
            for tool_name, tool in settings.tool_settings(name).items()
            if not tool.awaited
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
                        messages_of(chunk), chunk.turn, chunk_language(chunk), emit
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
    ) -> None:
        """Stream the reply's text deltas, each carrying the user's language for the synthesizer; a completion that called tools ends in their tool_call chunk, and leaves the turn open for the dialog's follow-up prompt unless every call is of an unawaited tool."""
        ids: dict[int, str] = {}
        names: dict[int, str] = {}
        arguments: dict[int, list[str]] = {}
        async for delta in self.complete(messages):
            if delta.content:
                emit(text_chunk(delta.content, turn, language))
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
        calls = [
            ToolCall(
                id=ids.get(index, f"call_{turn}_{index}"),
                name=name,
                arguments=parse_arguments("".join(arguments.get(index, []))),
            )
            for index, name in sorted(names.items())
        ]
        if calls:
            emit(tool_call_chunk(calls, turn))
        else:
            pass
        if all(call.name in self.unawaited for call in calls):
            emit(control_chunk(TURN_DONE, turn=turn))
        else:
            pass

    def request_body(self, messages: list[ChatMessage]) -> dict[str, object]:
        body: dict[str, object] = {
            "model": self.config.model,
            "messages": [message_dict(message) for message in messages],
            "stream": True,
            "max_tokens": self.config.max_tokens,
        }
        if self.config.temperature is not None:
            body["temperature"] = self.config.temperature
        else:
            pass
        if self.config.tools:
            body["tools"] = [
                {"type": "function", "function": tool.model_dump(exclude_none=True)}
                for tool in self.config.tools
            ]
        else:
            pass
        return {**body, **self.config.extra_body}

    async def complete(self, messages: list[ChatMessage]) -> AsyncIterator[ChatDelta]:
        finished = False
        async with self.client.stream(
            "POST",
            f"{self.config.base_url.rstrip('/')}/chat/completions",
            json=self.request_body(messages),
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
                    yield choice.delta
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
