# SPDX-License-Identifier: Apache-2.0
"""Model transport and response validation for documentation reviews."""

from __future__ import annotations

import os
import urllib.error
import urllib.request
from typing import Literal

from pydantic import BaseModel, Field, ValidationError

from scripts.docs_review.models import ModelSettings, ReasoningEffort

MODEL_TIMEOUT_SECONDS = 60


class ReviewerError(Exception):
    """The model call or its response could not be used."""


class ChatMessage(BaseModel):
    role: Literal["system", "user"]
    content: str


class ChatRequest(BaseModel):
    """JSON body sent to an OpenAI-compatible chat endpoint."""

    model_name: str = Field(serialization_alias="model")
    messages: list[ChatMessage]
    max_tokens: int | None = Field(default=None, gt=0)
    max_completion_tokens: int | None = Field(default=None, gt=0)
    temperature: Literal[0] | None = None
    reasoning_effort: ReasoningEffort | None = None

    def to_json(self) -> str:
        return self.model_dump_json(by_alias=True, exclude_none=True)


class OpenAICompatibleReviewer:
    """Read-only model call. No shell, web search, or GitHub writes."""

    def __init__(self, settings: ModelSettings, prompt_text: str) -> None:
        self.settings = settings
        self.prompt_text = prompt_text

    def review(self, bundle_text: str) -> str:
        request = build_chat_request(bundle_text, self.settings, self.prompt_text)
        http_request = urllib.request.Request(
            self.settings.endpoint_url,
            data=request.to_json().encode(),
            headers={
                "Authorization": authorization_header(self.settings),
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(
                http_request, timeout=MODEL_TIMEOUT_SECONDS
            ) as response:
                payload_bytes = response.read()
        except urllib.error.URLError as error:
            raise ReviewerError("model request failed") from error
        else:
            return parse_chat_content(payload_bytes)


def build_chat_request(
    bundle_text: str, settings: ModelSettings, prompt_text: str
) -> ChatRequest:
    messages = [
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
    try:
        secret = os.environ[variable_name]
    except KeyError as error:
        raise ReviewerError(
            f"Missing API key environment variable {variable_name}"
        ) from error
    else:
        return f"Bearer {secret}"


def parse_chat_content(payload_bytes: bytes) -> str:
    try:
        response = ChatResponse.model_validate_json(payload_bytes)
    except ValidationError as error:
        raise ReviewerError("model response did not match the chat schema") from error
    else:
        choice = response.choices[0]
    if choice.finish_reason not in (None, "stop"):
        raise ReviewerError(f"model response was incomplete: {choice.finish_reason}")
    else:
        return choice.message.content


class ChatResponseMessage(BaseModel):
    content: str = Field(strict=True)


class ChatResponseChoice(BaseModel):
    message: ChatResponseMessage
    finish_reason: str | None = None


class ChatResponse(BaseModel):
    choices: list[ChatResponseChoice] = Field(min_length=1)
