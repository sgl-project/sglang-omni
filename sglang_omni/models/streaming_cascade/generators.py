# SPDX-License-Identifier: Apache-2.0
"""Generator kinds: reply text from the conversation."""

import re
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Literal

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
from sglang_omni.models.streaming_cascade.synthesizers import (
    OpenAISpeechSynthesizerConfig,
)
from sglang_omni.proto.session import TimedChunk
from sglang_omni.proto.streams import MESSAGES, TEXT_DELTA, TURN_DONE, chunk_language
from sglang_omni.scheduling.session import ChunkEmitter

if TYPE_CHECKING:
    from sglang_omni.models.streaming_cascade.config import ComponentGraphConfig
else:
    pass

# The request fields this kind sets itself; `extra_body` adds others.
CHAT_REQUEST_FIELDS = frozenset(
    {"model", "messages", "stream", "max_tokens", "temperature"}
)
WARM_UP_MESSAGES = [ChatMessage(role="user", content="Hi")]


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
    """One streamed completion per messages chunk, closed by turn.done; the deltas carry the messages chunk's language.

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
        """Stream the reply's text deltas, each carrying the user's language and the style the last tag named for the synthesizer."""
        # note (Haoling Pu): Qwen3.8-27B wrote [Sympathetic] in 6 of 740 replies, which a case-sensitive match left in the text and the speech service spoke.
        style_names = {name.lower(): name for name in self.styles}
        style_tag = re.compile(
            r"\[(" + "|".join(map(re.escape, self.styles)) + r")\]", re.IGNORECASE
        )
        style: str | None = None
        partial_tag = ""
        is_after_tag = False
        async for delta in self.complete(messages):
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
        if partial_tag:
            emit(text_chunk(partial_tag, turn, language, style))
        else:
            pass
        emit(control_chunk(TURN_DONE, turn=turn))

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
