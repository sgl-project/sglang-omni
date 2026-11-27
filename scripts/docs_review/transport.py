# SPDX-License-Identifier: Apache-2.0
"""Model transport and response validation for documentation reviews."""

from __future__ import annotations

import logging
import os
import time
import urllib.error
import urllib.request
from concurrent.futures import Future
from http.client import HTTPException
from threading import Thread
from typing import Literal
from urllib.parse import urlsplit

from pydantic import BaseModel, Field, ValidationError

from scripts.docs_review.models import ModelSettings, ReasoningEffort
from scripts.docs_review.tools import (
    MAX_TOOL_CALLS,
    ReadOnlyToolExecutor,
    ToolDefinition,
)

LOGGER = logging.getLogger(__name__)
MODEL_TIMEOUT_SECONDS = 60
MAX_REVIEW_DURATION_SECONDS = 240
MAX_CHAT_RESPONSE_BYTES = 1_048_576
MAX_TOOL_CALLS_PER_MESSAGE = 2 * MAX_TOOL_CALLS


class ReviewerConfigurationError(ValueError):
    """The reviewer could not prepare a model request."""


class ReviewerError(Exception):
    """The model call or its response could not be used."""


class ChatMessage(BaseModel):
    role: Literal["system", "user"]
    content: str


class ToolCallFunction(BaseModel):
    name: str
    arguments: str


class ToolCall(BaseModel):
    id: str = Field(min_length=1)
    type: Literal["function"] = "function"
    function: ToolCallFunction


class AssistantToolMessage(BaseModel):
    role: Literal["assistant"]
    content: str | None = None
    tool_calls: list[ToolCall]


class ToolResultMessage(BaseModel):
    role: Literal["tool"]
    tool_call_id: str
    content: str


RequestMessage = ChatMessage | AssistantToolMessage | ToolResultMessage


class ChatRequest(BaseModel):
    """JSON body sent to an OpenAI-compatible chat endpoint."""

    model_name: str = Field(serialization_alias="model")
    messages: list[RequestMessage]
    tools: list[ToolDefinition] | None = None
    tool_choice: Literal["auto", "none"] | None = None
    max_tokens: int | None = Field(default=None, gt=0)
    max_completion_tokens: int | None = Field(default=None, gt=0)
    temperature: Literal[0] | None = None
    reasoning_effort: ReasoningEffort | None = None

    def to_json(self) -> str:
        return self.model_dump_json(by_alias=True, exclude_none=True)


class OpenAICompatibleReviewer:
    """Read-only model call. No shell, web search, or GitHub writes."""

    def __init__(
        self,
        settings: ModelSettings,
        prompt_text: str,
        *,
        toolbox: ReadOnlyToolExecutor | None = None,
    ) -> None:
        endpoint = urlsplit(settings.endpoint_url)
        if endpoint.scheme not in ("http", "https") or not endpoint.hostname:
            raise ReviewerConfigurationError("Model endpoint must be an HTTP(S) URL")
        else:
            self.authorization = authorization_header(settings)
        self.settings = settings
        self.prompt_text = prompt_text
        self.toolbox = toolbox

    def review(self, bundle_text: str) -> str:
        request = build_chat_request(bundle_text, self.settings, self.prompt_text)
        tool_rounds = self.settings.max_tool_rounds if self.toolbox is not None else 0
        deadline = (
            time.monotonic() + MAX_REVIEW_DURATION_SECONDS if tool_rounds else None
        )
        if (
            self.toolbox is not None
            and tool_rounds
            and not self.toolbox.budget_exhausted()
        ):
            request.tools = self.toolbox.tool_definitions()
            request.tool_choice = "auto"
        else:
            pass

        for round_index in range(tool_rounds + 1):
            try:
                choice = self.request_choice(request, deadline)
            except urllib.error.HTTPError as error:
                error.close()
                if round_index == 0 and request.tools and error.code in (400, 422):
                    LOGGER.warning(
                        f"Model endpoint rejected tools (HTTP {error.code}); retrying without tools"
                    )
                    request.tools = None
                    request.tool_choice = None
                    continue
                else:
                    raise ReviewerError("model request failed") from error
            tool_calls = choice.message.tool_calls
            if request.tools is None or not tool_calls:
                return choice.final_content()
            elif choice.finish_reason not in (None, "stop", "tool_calls"):
                raise ReviewerError(
                    f"model response was incomplete: {choice.finish_reason}"
                )
            else:
                assert self.toolbox is not None
                request.messages.append(
                    AssistantToolMessage(
                        role="assistant",
                        content=choice.message.content,
                        tool_calls=tool_calls,
                    )
                )
            for tool_call in tool_calls:
                remaining_request_seconds(deadline)
                content = self.toolbox.execute(
                    tool_call.function.name,
                    tool_call.function.arguments,
                    deadline_seconds=deadline,
                )
                request.messages.append(
                    ToolResultMessage(
                        role="tool", tool_call_id=tool_call.id, content=content
                    )
                )
            if round_index + 1 == tool_rounds or self.toolbox.budget_exhausted():
                request.tools = None
                request.tool_choice = None
                request.messages.append(
                    ChatMessage(
                        role="user",
                        content="Tool budget reached. Return the final JSON list of findings now and nothing else.",
                    )
                )
            else:
                pass
        raise ReviewerError("model review exceeded tool round limit")

    def request_choice(
        self, request: ChatRequest, deadline: float | None
    ) -> ChatResponseChoice:
        http_request = urllib.request.Request(
            self.settings.endpoint_url,
            data=request.to_json().encode(),
            headers={
                "Authorization": self.authorization,
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            if deadline is None:
                with urllib.request.urlopen(
                    http_request, timeout=MODEL_TIMEOUT_SECONDS
                ) as response:
                    payload_bytes = response.read()
            else:
                # DNS and HTTP headers may block beyond urllib's per-socket timeout.
                result: Future[bytes] = Future()
                Thread(
                    target=read_tool_response,
                    args=(http_request, deadline, result),
                    daemon=True,
                ).start()
                payload_bytes = result.result(
                    timeout=max(0, deadline - time.monotonic())
                )
        except urllib.error.HTTPError:
            raise
        except (HTTPException, OSError) as error:
            remaining_request_seconds(deadline)
            raise ReviewerError("model request failed") from error
        else:
            choice = parse_chat_choice(payload_bytes)
            remaining_request_seconds(deadline)
            return choice


def read_tool_response(
    http_request: urllib.request.Request, deadline: float, result: Future[bytes]
) -> None:
    try:
        with urllib.request.urlopen(
            http_request, timeout=remaining_request_seconds(deadline)
        ) as response:
            payload = bytearray()
            while len(payload) <= MAX_CHAT_RESPONSE_BYTES:
                remaining_request_seconds(deadline)
                chunk = response.read1(MAX_CHAT_RESPONSE_BYTES + 1 - len(payload))
                if chunk:
                    payload.extend(chunk)
                else:
                    break
            if len(payload) > MAX_CHAT_RESPONSE_BYTES:
                raise ReviewerError("model response exceeded size limit")
            else:
                pass
    except BaseException as error:
        result.set_exception(error)
    else:
        result.set_result(bytes(payload))


def remaining_request_seconds(deadline: float | None) -> float:
    if deadline is None:
        return MODEL_TIMEOUT_SECONDS
    else:
        remaining_seconds = deadline - time.monotonic()
    if remaining_seconds <= 0:
        raise ReviewerError("model review exceeded duration limit")
    else:
        return min(MODEL_TIMEOUT_SECONDS, remaining_seconds)


def build_chat_request(
    bundle_text: str, settings: ModelSettings, prompt_text: str
) -> ChatRequest:
    messages: list[RequestMessage] = [
        ChatMessage(role="system", content=prompt_text),
        ChatMessage(role="user", content=bundle_text),
    ]
    if settings.reasoning_effort is None:
        return ChatRequest(
            model_name=settings.model_name,
            messages=messages,
            max_tokens=settings.max_output_tokens,
            temperature=0,
        )
    else:
        return ChatRequest(
            model_name=settings.model_name,
            messages=messages,
            max_completion_tokens=settings.max_output_tokens,
            reasoning_effort=settings.reasoning_effort,
        )


def authorization_header(settings: ModelSettings) -> str:
    variable_name = settings.api_key_environment_variable
    secret = os.environ.get(variable_name, "").strip()
    if not secret:
        raise ReviewerConfigurationError(
            f"Missing API key environment variable {variable_name}"
        )
    else:
        return f"Bearer {secret}"


def parse_chat_content(payload_bytes: bytes) -> str:
    return parse_chat_choice(payload_bytes).final_content()


def parse_chat_choice(payload_bytes: bytes) -> ChatResponseChoice:
    try:
        response = ChatResponse.model_validate_json(payload_bytes)
    except ValidationError as error:
        raise ReviewerError("model response did not match the chat schema") from error
    else:
        return response.choices[0]


class ChatResponseMessage(BaseModel):
    content: str | None = Field(default=None, strict=True)
    tool_calls: list[ToolCall] | None = Field(
        default=None, max_length=MAX_TOOL_CALLS_PER_MESSAGE
    )


class ChatResponseChoice(BaseModel):
    message: ChatResponseMessage
    finish_reason: str | None = None

    def final_content(self) -> str:
        if self.finish_reason not in (None, "stop"):
            raise ReviewerError(f"model response was incomplete: {self.finish_reason}")
        elif self.message.tool_calls or self.message.content is None:
            raise ReviewerError("model response did not contain a final answer")
        else:
            return self.message.content


class ChatResponse(BaseModel):
    choices: list[ChatResponseChoice] = Field(min_length=1)
