# SPDX-License-Identifier: Apache-2.0
"""Generator kinds: reply text from the conversation."""

import asyncio
import json
import logging
from collections.abc import AsyncIterator
from typing import Literal

import httpx
from pydantic import BaseModel, Field, field_validator

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
    tool_round_messages,
)
from sglang_omni.proto.session import TimedChunk
from sglang_omni.proto.streams import (
    INTERRUPT,
    MESSAGES,
    TOOL_RESPONSE,
    TURN_DONE,
    chunk_language,
    edge_event,
)
from sglang_omni.proto.tools import (
    ToolCall,
    ToolExchange,
    ToolResponse,
    tool_call_chunk,
    tool_exchange_chunk,
    tool_responses_of,
)
from sglang_omni.scheduling.session import ChunkEmitter

logger = logging.getLogger(__name__)

# The request fields this kind sets itself; `extra_body` adds others.
CHAT_REQUEST_FIELDS = frozenset(
    {"model", "messages", "stream", "max_tokens", "temperature", "tools"}
)
WARM_UP_MESSAGES = [ChatMessage(role="user", content="Hi")]
FINISH_REASONS = ("stop", "length", "tool_calls")
# Text a chat model writes for a tool call when the service does not parse its tool-call format.
TOOL_CALL_MARKUP = ("<tool_call>", "<function=", "<TOOLCALL>")
TOOL_PROBE_MAX_TOKENS = 128


class OpenAIChatGeneratorConfig(ComponentConfig):
    kind: Literal["openai_chat"] = "openai_chat"
    base_url: str = Field(pattern=r"^https?://[^/]+")
    model: str
    max_tokens: int = Field(default=512, gt=0)
    temperature: float | None = Field(default=None, ge=0, le=2)
    api_key_env: str = "OPENAI_API_KEY"
    extra_body: dict[str, object] = Field(default_factory=dict)
    # Chat-completions function definitions; a reply calls them through the node that reads its tool_call.
    tools: list[dict[str, object]] = Field(default_factory=list)
    # Tool rounds in one reply; the request after the last round offers no tools, so the reply ends in text.
    max_tool_rounds: int = Field(default=2, gt=0)
    tool_response_timeout_s: float = Field(default=10, gt=0)

    @field_validator("extra_body")
    @classmethod
    def check_extra_body(cls, extra_body: dict[str, object]) -> dict[str, object]:
        if CHAT_REQUEST_FIELDS & set(extra_body):
            raise ValueError(
                f"extra_body cannot set {sorted(CHAT_REQUEST_FIELDS & set(extra_body))}"
            )
        else:
            return extra_body


class ChatFunctionDelta(BaseModel):
    name: str | None = None
    arguments: str | None = None


class ChatToolCallDelta(BaseModel):
    index: int
    id: str | None = None
    function: ChatFunctionDelta = Field(default_factory=ChatFunctionDelta)


class ChatDelta(BaseModel):
    content: str | None = None
    tool_calls: list[ChatToolCallDelta] | None = None


class ChatChoice(BaseModel):
    delta: ChatDelta
    finish_reason: str | None = None


class ChatEvent(BaseModel):
    choices: list[ChatChoice]


class OpenAIChatGenerator(Replier):
    """One streamed completion per messages chunk, closed by turn.done; the deltas carry the messages chunk's language.

    A completion that ends in tool calls emits them as one tool_call chunk of
    the reply's turn, waits for the tool_response a control edge brings back,
    emits the round as tool_exchange and continues the reply with the
    responses in its prompt.
    """

    def __init__(self, config: OpenAIChatGeneratorConfig) -> None:
        super().__init__()
        self.config = config
        self.client = httpx.AsyncClient(
            timeout=config.timeout_s, headers=bearer_headers(config.api_key_env)
        )
        self.pending_responses: dict[int, asyncio.Future[list[ToolResponse]]] = {}

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

    async def control(self, event: TimedChunk) -> None:
        if edge_event(event) == TOOL_RESPONSE:
            pending = self.pending_responses.get(event.turn)
            if pending is not None and not pending.done():
                pending.set_result(tool_responses_of(event))
            else:
                logger.info(f"tool response of turn {event.turn} has no waiting reply")
        elif edge_event(event) == INTERRUPT:
            await super().control(event)
        else:
            pass

    async def reply(
        self,
        messages: list[ChatMessage],
        turn: int,
        language: str | None,
        emit: ChunkEmitter,
    ) -> None:
        """Stream the reply's text deltas, each carrying the user's language for the synthesizer, through at most max_tool_rounds tool rounds."""
        history = list(messages)
        spoken_chars = 0
        for round_index in range(self.config.max_tool_rounds + 1):
            round_text = ""
            argument_texts: dict[int, str] = {}
            call_ids: dict[int, str] = {}
            call_names: dict[int, str] = {}
            async for delta in self.complete(history):
                # note (zengjiahua): with thinking on, the reasoning parser leaves the blank lines after the thinking block in the content, which a caption shows and a speech request reads.
                content = (
                    delta.content
                    if spoken_chars + len(round_text)
                    else (delta.content or "").lstrip()
                )
                if content:
                    emit(text_chunk(content, turn, language))
                    round_text += content
                else:
                    pass
                for call_delta in delta.tool_calls or ():
                    if call_delta.id is not None:
                        call_ids[call_delta.index] = call_delta.id
                    else:
                        pass
                    if call_delta.function.name is not None:
                        call_names[call_delta.index] = (
                            call_names.get(call_delta.index, "")
                            + call_delta.function.name
                        )
                    else:
                        pass
                    argument_texts[call_delta.index] = argument_texts.get(
                        call_delta.index, ""
                    ) + (call_delta.function.arguments or "")
            spoken_chars += len(round_text)
            calls = [
                ToolCall(
                    id=call_ids.get(index, f"call_{turn}_{round_index}_{index}"),
                    name=call_names.get(index, ""),
                    arguments=decoded_arguments(argument_texts.get(index, "")),
                )
                for index in sorted(call_names.keys() | argument_texts.keys())
            ]
            if not calls:
                break
            elif round_index == self.config.max_tool_rounds:
                logger.warning(
                    f"turn {turn} asked for tools after {round_index} tool rounds; the calls are dropped"
                )
                break
            else:
                responses = await self.tool_responses(calls, turn, emit)
                exchange = ToolExchange(
                    text_offset=spoken_chars, calls=calls, responses=responses
                )
                emit(tool_exchange_chunk(exchange, turn))
                history.extend(tool_round_messages(round_text, exchange))
        emit(control_chunk(TURN_DONE, turn=turn))

    async def tool_responses(
        self, calls: list[ToolCall], turn: int, emit: ChunkEmitter
    ) -> list[ToolResponse]:
        """Emit the calls and wait for their responses; a tool that does not answer in time answers with an error."""
        pending: asyncio.Future[list[ToolResponse]] = (
            asyncio.get_running_loop().create_future()
        )
        self.pending_responses[turn] = pending
        emit(tool_call_chunk(calls, turn))
        try:
            return await asyncio.wait_for(pending, self.config.tool_response_timeout_s)
        except TimeoutError:
            logger.warning(f"tools of turn {turn} did not answer in time")
            return [
                ToolResponse(
                    id=call.id,
                    name=call.name,
                    response={"error": "the tool did not answer in time"},
                )
                for call in calls
            ]
        finally:
            self.pending_responses.pop(turn, None)

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
            body["tools"] = self.config.tools
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
                    elif choice.finish_reason not in FINISH_REASONS:
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


def decoded_arguments(argument_text: str) -> object:
    """A call's streamed arguments as JSON; text that is not JSON stays text, which a tool answers with an error."""
    try:
        return json.loads(argument_text) if argument_text else {}
    except json.JSONDecodeError:
        return argument_text


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
        config.model_copy(
            update={"max_tokens": 1, "extra_body": warm_up_body, "tools": []}
        )
    )
    completions_url = f"{config.base_url.rstrip('/')}/chat/completions"
    try:
        async for _ in generator.complete(WARM_UP_MESSAGES):
            pass
        if config.tools:
            await probe_tool_parser(
                OpenAIChatGenerator(
                    config.model_copy(
                        update={
                            "max_tokens": TOOL_PROBE_MAX_TOKENS,
                            "extra_body": warm_up_body,
                        }
                    )
                )
            )
        else:
            pass
    except httpx.HTTPStatusError as error:
        raise RuntimeError(
            f"{completions_url} answered {error.response.status_code}"
        ) from error
    except httpx.TransportError as error:
        raise RuntimeError(f"cannot reach {completions_url}: {error!r}") from error
    finally:
        await generator.close()


async def probe_tool_parser(generator: OpenAIChatGenerator) -> None:
    """Require a tool request to answer without tool-call markup in its text, which the speech node would read aloud: the service parses calls only with a tool-call parser that matches the model."""
    name = str(generator.config.tools[0]["function"]["name"])
    text = ""
    has_calls = False
    try:
        async for delta in generator.complete(
            [ChatMessage(role="user", content=f"Call the {name} tool now.")]
        ):
            text += delta.content or ""
            has_calls = has_calls or bool(delta.tool_calls)
    finally:
        await generator.close()
    if any(marker in text for marker in TOOL_CALL_MARKUP):
        raise RuntimeError(
            f"{generator.config.model} wrote a tool call as text; start the chat service with a --tool-call-parser that matches the model"
        )
    elif not has_calls:
        logger.warning(
            f"{generator.config.model} answered a tool request without calling {name}"
        )
    else:
        pass
