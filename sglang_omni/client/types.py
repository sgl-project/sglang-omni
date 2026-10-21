# SPDX-License-Identifier: Apache-2.0
"""Client request, streaming, and result types."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Literal

from typing_extensions import TypedDict


class TokenUsageDict(TypedDict):
    prompt_tokens: int | None
    completion_tokens: int | None
    total_tokens: int | None


class UsageInfoDict(TokenUsageDict, total=False):
    engine_time_s: float


class ImageInputTokensDetailsDict(TypedDict):
    text_tokens: int | None
    image_tokens: int


class ImageUsageDict(TypedDict, total=False):
    input_tokens: int
    total_tokens: int
    input_tokens_details: ImageInputTokensDetailsDict


# note (Yucheng Hu): finish reason reported on speech surfaces when the model
# did not say how generation ended.
UNKNOWN_FINISH_REASON = "unknown"


@dataclass(frozen=True, kw_only=True)
class DiarizationSegment:
    """One speaker's active interval, in seconds from the recording start."""

    start: float
    end: float
    speaker: str


@dataclass(frozen=True, kw_only=True)
class DiarizationResult:
    """Standalone speaker activity, with overlapping intervals permitted."""

    duration: float
    segments: list[DiarizationSegment]


DiarizationOperation = Literal["open", "append", "finish", "close", "reset"]


@dataclass
class Message:
    """Chat-style message."""

    role: str
    content: object

    def to_dict(self) -> dict[str, object]:
        return {"role": self.role, "content": self.content}


@dataclass
class UsageInfo:
    """Token usage details."""

    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    engine_time_s: float | None = None

    @classmethod
    def from_dict(cls, data: Mapping[str, object] | None) -> "UsageInfo | None":
        if not data:
            return None
        else:
            pass
        return cls(
            prompt_tokens=data.get("prompt_tokens"),
            completion_tokens=data.get("completion_tokens"),
            total_tokens=data.get("total_tokens"),
            engine_time_s=data.get("engine_time_s"),
        )

    def to_dict(self) -> UsageInfoDict:
        d: UsageInfoDict = {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
        }
        if self.engine_time_s is not None:
            d["engine_time_s"] = self.engine_time_s
        else:
            pass
        return d


@dataclass
class SamplingParams:
    """Sampling configuration."""

    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    min_p: float | None = None
    repetition_penalty: float | None = None
    stop: list[str] = field(default_factory=list)
    stop_token_ids: list[int] = field(default_factory=list)
    seed: int | None = None
    max_new_tokens: int | None = None

    def to_dict(self) -> dict[str, object]:
        # note (ratish): a field the caller left unset stays out, so the stage
        # request builder applies its own model default.
        values: dict[str, object] = {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "min_p": self.min_p,
            "repetition_penalty": self.repetition_penalty,
            "seed": self.seed,
            "max_new_tokens": self.max_new_tokens,
        }
        return {
            **{name: value for name, value in values.items() if value is not None},
            "stop": list(self.stop),
            "stop_token_ids": list(self.stop_token_ids),
        }


@dataclass
class GenerateRequest:
    """Client-level request (API-agnostic)."""

    model: str | None = None

    prompt: str | dict[str, object] | None = None
    prompt_token_ids: list[int] | None = None
    messages: list[Message] | None = None

    sampling: SamplingParams = field(default_factory=SamplingParams)
    stage_sampling: dict[str, SamplingParams] | None = None
    stage_params: dict[str, dict[str, object]] | None = None
    extra_params: dict[str, object] = field(default_factory=dict)
    stream: bool = True
    max_tokens: int | None = None

    # Multi-modal support
    output_modalities: list[str] | None = None
    multimodal_train_inputs: dict[str, object] | None = None

    metadata: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
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
            "extra_params": dict(self.extra_params),
            "stream": self.stream,
            "max_tokens": self.max_tokens,
            "output_modalities": self.output_modalities,
            "multimodal_train_inputs": self.multimodal_train_inputs,
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
    output_token_logprobs: list[list[float | int]] | None = None
    omni_rollout: dict[str, object] | None = None
    finish_reason: str | None = None
    # note (Yucheng Hu): None preserves typed producers; empty marks a synthetic terminal reason.
    model_finish_reason: str | None = None
    usage: UsageInfo | None = None
    weight_version: str | None = None
    stage_id: int | None = None
    stage_name: str | None = None
    modality: str = "text"
    language: str | None = None
    audio_data: object = None
    media: list[dict[str, object]] | None = None
    sample_rate: int | None = None
    image_b64: str | None = None
    images_b64: list[str] | None = None
    image_usage: ImageUsageDict | None = None
    diarization: DiarizationResult | None = None
    image: str | None = None  # base64-encoded PNG from the terminal result

    @property
    def reported_finish_reason(self) -> str:
        if self.model_finish_reason is None:
            return self.finish_reason or ""
        else:
            return self.model_finish_reason

    def to_dict(self) -> dict[str, object]:
        return {
            "request_id": self.request_id,
            "index": self.index,
            "token_ids": list(self.token_ids),
            "text": self.text,
            "logprobs": self.logprobs,
            "output_token_logprobs": self.output_token_logprobs,
            "omni_rollout": self.omni_rollout,
            "finish_reason": self.finish_reason,
            "model_finish_reason": self.reported_finish_reason,
            "usage": self.usage.to_dict() if self.usage else None,
            "weight_version": self.weight_version,
            "stage_id": self.stage_id,
            "stage_name": self.stage_name,
            "modality": self.modality,
            "language": self.language,
            "audio_data": self.audio_data,
            "media": self.media,
            "sample_rate": self.sample_rate,
            "image_b64": self.image_b64,
            "images_b64": self.images_b64,
            "image_usage": self.image_usage,
            "image": self.image,
        }


@dataclass(frozen=True)
class UMMSegment:
    """One accepted output of an interleaved (UMM) request, in order.

    data is None only in the final result of a streamed request, which already
    delivered each segment with its data.
    """

    execution_id: str
    segment_index: int
    kind: str
    data: object = None

    @classmethod
    def from_dict(cls, value: object) -> UMMSegment:
        fields = {"type", "execution_id", "segment_index", "kind", "data"}
        if (
            not isinstance(value, dict)
            or value.get("type") != "segment"
            or not set(value) <= fields
            or not isinstance(value.get("execution_id"), str)
            or type(value.get("segment_index")) is not int
            or value["segment_index"] < 0
            or not isinstance(value.get("kind"), str)
        ):
            raise ValueError(f"Invalid UMM segment {value!r:.200}")
        else:
            return cls(
                value["execution_id"],
                value["segment_index"],
                value["kind"],
                value.get("data"),
            )

    def to_dict(self) -> dict[str, object]:
        result: dict[str, object] = {
            "type": "segment",
            "execution_id": self.execution_id,
            "segment_index": self.segment_index,
            "kind": self.kind,
        }
        if self.data is not None:
            result["data"] = self.data
        else:
            pass
        return result


@dataclass
class UMMGenerateChunk(GenerateChunk):
    """A chunk of an interleaved (UMM) request.

    A streamed chunk carries one segment. The final chunk lists every segment,
    and its text joins the text segments that carry data.
    """

    segment: UMMSegment | None = None
    segments: list[UMMSegment] | None = None

    def to_dict(self) -> dict[str, object]:
        result = super().to_dict()
        if self.segment is not None:
            result["segment"] = self.segment.to_dict()
        else:
            pass
        if self.segments is not None:
            result["segments"] = [segment.to_dict() for segment in self.segments]
        else:
            pass
        return result


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
    # Saved media of a generation stage. Delivered files belong to the
    # caller. The server removes them only when it drops the completion
    # before any caller receives it.
    media: list[dict[str, object]] | None = None
    finish_reason: str = "stop"
    usage: UsageInfo | None = None
    output_token_logprobs: list[list[float | int]] | None = None
    omni_rollout: dict[str, object] | None = None
    weight_version: str | None = None
    language: str | None = None
    diarization: DiarizationResult | None = None
    image: str | None = None  # base64-encoded PNG


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
class UMMCompletionResult(CompletionResult):
    """Result of an interleaved (UMM) request.

    text joins the text segments, and segments keeps every accepted output in
    order with its data.
    """

    segments: list[UMMSegment] = field(default_factory=list)


@dataclass
class UMMCompletionStreamChunk(CompletionStreamChunk):
    """A streamed chunk of an interleaved (UMM) request with its segment."""

    segment: UMMSegment | None = None


@dataclass
class SpeechResult:
    """Result of a text-to-speech call."""

    audio_bytes: bytes
    mime_type: str
    format: str
    sample_rate: int | None = None
    usage: UsageInfo | None = None
    finish_reason: str = UNKNOWN_FINISH_REASON


class ClientError(Exception):
    """Error raised by the Client layer."""
