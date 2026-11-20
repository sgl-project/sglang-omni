# SPDX-License-Identifier: Apache-2.0
"""Generator kinds: reply text from the conversation."""

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
)
from sglang_omni.proto.session import TimedChunk
from sglang_omni.proto.streams import MESSAGES, TURN_DONE, WAIT_REPLY, chunk_language
from sglang_omni.scheduling.session import ChunkEmitter

# The request fields this kind sets itself; `extra_body` adds others.
CHAT_REQUEST_FIELDS = frozenset(
    {"model", "messages", "stream", "max_tokens", "temperature"}
)
WARM_UP_MESSAGES = [ChatMessage(role="user", content="Hi")]
WAIT_REPLY_NOTE = f"When the user asks you to wait, hold on, listen, not interrupt, or not reply until they finish speaking, output only {WAIT_REPLY} instead of acknowledging that you will wait, and keep outputting {WAIT_REPLY} on hesitation fillers or partial items until the user finishes their thought or asks a question."


class OpenAIChatGeneratorConfig(ComponentConfig):
    kind: Literal["openai_chat"] = "openai_chat"
    base_url: str = Field(pattern=r"^https?://[^/]+")
    model: str
    max_tokens: int = Field(default=512, gt=0)
    temperature: float | None = Field(default=None, ge=0, le=2)
    api_key_env: str = "OPENAI_API_KEY"
    extra_body: dict[str, object] = Field(default_factory=dict)

    @field_validator("extra_body")
    @classmethod
    def check_extra_body(cls, extra_body: dict[str, object]) -> dict[str, object]:
        if CHAT_REQUEST_FIELDS & set(extra_body):
            raise ValueError(
                f"extra_body cannot set {sorted(CHAT_REQUEST_FIELDS & set(extra_body))}"
            )
        else:
            return extra_body


class ChatDelta(BaseModel):
    content: str | None = None


class ChatChoice(BaseModel):
    delta: ChatDelta
    finish_reason: str | None = None


class ChatEvent(BaseModel):
    choices: list[ChatChoice]


class OpenAIChatGenerator(Replier):
    """One streamed completion per messages chunk, closed by turn.done; the deltas carry the messages chunk's language."""

    def __init__(self, config: OpenAIChatGeneratorConfig) -> None:
        super().__init__()
        self.config = config
        self.client = httpx.AsyncClient(
            timeout=config.timeout_s, headers=bearer_headers(config.api_key_env)
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
        """Stream the reply's text deltas, each carrying the user's language for the synthesizer."""
        buffered = ""
        is_streaming = False
        async for delta in self.complete(messages):
            if not delta.content:
                pass
            elif is_streaming:
                emit(text_chunk(delta.content, turn, language))
            else:
                buffered += delta.content
                stripped = buffered.lstrip()
                if WAIT_REPLY.startswith(stripped):
                    pass
                elif not stripped.startswith(WAIT_REPLY):
                    emit(text_chunk(buffered, turn, language))
                    is_streaming = True
                elif stripped.removeprefix(WAIT_REPLY).strip():
                    emit(
                        text_chunk(
                            stripped.removeprefix(WAIT_REPLY).lstrip(), turn, language
                        )
                    )
                    is_streaming = True
                else:
                    pass
        if is_streaming:
            pass
        elif buffered.strip() == WAIT_REPLY:
            emit(text_chunk(WAIT_REPLY, turn, language))
        elif buffered:
            emit(text_chunk(buffered, turn, language))
        else:
            pass
        emit(control_chunk(TURN_DONE, turn=turn))

    def request_body(self, messages: list[ChatMessage]) -> dict[str, object]:
        if messages and messages[0].role == "system":
            messages = [
                ChatMessage(
                    role="system",
                    content=f"{messages[0].content} {WAIT_REPLY_NOTE}",
                ),
                *messages[1:],
            ]
        else:
            pass
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
                    elif choice.finish_reason not in ("stop", "length"):
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
