# SPDX-License-Identifier: Apache-2.0
"""Read text chat streams for multimodal benchmark clients."""

from __future__ import annotations

import time

import aiohttp
from pydantic import BaseModel, Field, JsonValue

from benchmarks.benchmarker.data import RequestResult, finish_reason_from_server


class TextDelta(BaseModel):
    content: str | None = None


class TextChoice(BaseModel):
    delta: TextDelta = Field(default_factory=TextDelta)
    finish_reason: str | None = None


class TokenUsage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0


class TextStreamEvent(BaseModel):
    choices: list[TextChoice] = Field(default_factory=list)
    usage: TokenUsage | None = None
    error: JsonValue = None


async def consume_text_chat_stream(
    response: aiohttp.ClientResponse,
    result: RequestResult,
    start_time: float,
) -> None:
    """Collect text and usage, rejecting failed or truncated streams."""
    text_parts: list[str] = []
    seen_done = False
    async for raw_line in response.content:
        line = raw_line.decode("utf-8").strip()
        if not line.startswith("data:"):
            continue
        else:
            payload = line[len("data:") :].strip()
        if payload == "[DONE]":
            seen_done = True
            break
        else:
            event = TextStreamEvent.model_validate_json(payload)
        if event.error is not None:
            raise ValueError(f"Chat stream failed: {event.error}")
        else:
            pass
        for choice in event.choices:
            if choice.delta.content:
                if result.text_ttft_s is None:
                    result.text_ttft_s = time.perf_counter() - start_time
                else:
                    pass
                text_parts.append(choice.delta.content)
            else:
                pass
            if choice.finish_reason is not None:
                result.finish_reason = finish_reason_from_server(choice.finish_reason)
            else:
                pass
        if event.usage is not None:
            result.prompt_tokens = event.usage.prompt_tokens
            result.completion_tokens = event.usage.completion_tokens
        else:
            pass
    if not seen_done:
        raise ValueError("Chat stream ended without [DONE]")
    elif not text_parts:
        raise ValueError("Chat stream returned no text")
    else:
        result.text = "".join(text_parts)
        result.is_success = True
        result.engine_time_s = time.perf_counter() - start_time
        if result.completion_tokens > 0 and result.engine_time_s > 0:
            result.tok_per_s = result.completion_tokens / result.engine_time_s
        else:
            pass
