# SPDX-License-Identifier: Apache-2.0
"""Client request, streaming, and result types."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


@dataclass
class Message:
    """Chat-style message."""

    role: str
    content: Any

    def to_dict(self) -> dict[str, Any]:
        return {"role": self.role, "content": self.content}


@dataclass
class UsageInfo:
    """Token usage details."""

    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    engine_time_s: float | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "UsageInfo | None":
        if not data:
            return None
        return cls(
            prompt_tokens=data.get("prompt_tokens"),
            completion_tokens=data.get("completion_tokens"),
            total_tokens=data.get("total_tokens"),
            engine_time_s=data.get("engine_time_s"),
        )

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
        }
        if self.engine_time_s is not None:
            d["engine_time_s"] = self.engine_time_s
        return d


@dataclass
class SamplingParams:
    """Sampling configuration."""

    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1
    min_p: float = 0.0
    repetition_penalty: float = 1.0
    stop: list[str] = field(default_factory=list)
    stop_token_ids: list[int] = field(default_factory=list)
    seed: int | None = None
    max_new_tokens: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "min_p": self.min_p,
            "repetition_penalty": self.repetition_penalty,
            "stop": list(self.stop),
            "stop_token_ids": list(self.stop_token_ids),
            "seed": self.seed,
            "max_new_tokens": self.max_new_tokens,
        }


@dataclass
class GenerateRequest:
    """Client-level request (API-agnostic)."""

    model: str | None = None

    prompt: str | None = None
    prompt_token_ids: list[int] | None = None
    messages: list[Message] | None = None

    sampling: SamplingParams = field(default_factory=SamplingParams)
    stage_sampling: dict[str, SamplingParams] | None = None
    stage_params: dict[str, dict[str, Any]] | None = None
    stream: bool = True
    max_tokens: int | None = None

    # Multi-modal support
    output_modalities: list[str] | None = None

    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "prompt": self.prompt,
            "prompt_token_ids": self.prompt_token_ids,
            "messages": [m.to_dict() for m in self.messages] if self.messages else None,
            "sampling": self.sampling.to_dict(),
            "stage_sampling": (
                {key: params.to_dict() for key, params in self.stage_sampling.items()}
                if self.stage_sampling
                else None
            ),
            "stage_params": self.stage_params,
            "stream": self.stream,
            "max_tokens": self.max_tokens,
            "output_modalities": self.output_modalities,
            "metadata": dict(self.metadata),
        }


@dataclass
class GenerateChunk:
    """Streaming chunk from the client."""

    request_id: str
    index: int = 0
    token_ids: list[int] = field(default_factory=list)
    text: str = ""
    logprobs: list[float] | None = None
    finish_reason: str | None = None
    usage: UsageInfo | None = None
    stage_id: int | None = None
    stage_name: str | None = None
    modality: str = "text"
    # Multi-modal output data (e.g. audio waveform bytes, image bytes)
    audio_data: Any = None
    sample_rate: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "index": self.index,
            "token_ids": list(self.token_ids),
            "text": self.text,
            "logprobs": self.logprobs,
            "finish_reason": self.finish_reason,
            "usage": self.usage.to_dict() if self.usage else None,
            "stage_id": self.stage_id,
            "stage_name": self.stage_name,
            "modality": self.modality,
            "audio_data": self.audio_data,
            "sample_rate": self.sample_rate,
        }


class AbortLevel(Enum):
    """Abort severity."""

    SOFT = "soft"
    HARD = "hard"


@dataclass
class AbortResult:
    """Abort response from the client."""

    success: bool
    level_applied: AbortLevel
    partial_output: GenerateChunk | None = None


# ---------------------------------------------------------------------------
# High-level result types
# ---------------------------------------------------------------------------


@dataclass
class CompletionAudio:
    """Audio data from a non-streaming completion."""

    id: str
    data: str  # base64
    transcript: str | None = None


@dataclass
class CompletionResult:
    """Result of a non-streaming completion call."""

    request_id: str
    text: str
    audio: CompletionAudio | None = None
    finish_reason: str = "stop"
    usage: UsageInfo | None = None


@dataclass
class CompletionStreamChunk:
    """A single chunk from a streaming completion call."""

    request_id: str
    text: str = ""
    modality: str = "text"
    audio_b64: str | None = None  # already base64-encoded
    finish_reason: str | None = None
    usage: UsageInfo | None = None
    stage_name: str | None = None


@dataclass
class SpeechResult:
    """Result of a text-to-speech call."""

    audio_bytes: bytes
    mime_type: str
    format: str  # actual format used
    usage: UsageInfo | None = None


class ClientError(Exception):
    """Error raised by the Client layer."""
