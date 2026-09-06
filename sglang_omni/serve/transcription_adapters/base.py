# SPDX-License-Identifier: Apache-2.0
"""Model-specific transcription output adapters.

Ported from the upstream SGLang transcription_adapters pattern: subclass
TranscriptionAdapter, decorate with register_transcription_adapter("Key"), and
the /v1/audio/transcriptions handler resolves the right adapter by matching Key
as a substring against the served model's HF architectures.

Only the pieces the omni HTTP layer needs are kept: chunk orchestration, markup
post-processing, and verbose_json segment building. Sampling and prompt token
construction remain in each model's pipeline (stages.py / request_builders.py).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable

from sglang_omni.serve.protocol import (
    TranscriptionSegment,
    TranscriptionVerboseResponse,
)


class TranscriptionAdapter(ABC):
    """Abstract base for model-specific transcription output handling."""

    @property
    def supports_segment_timestamps(self) -> bool:
        """Whether segments carry model-derived timestamps."""
        return False

    @property
    def requires_ordered_chunk_decoding(self) -> bool:
        return False

    def chunk_prompt(
        self,
        *,
        caller_prompt: str | None,
        previous_text: str | None,
        is_first_decoded_chunk: bool,
    ) -> str | None:
        return caller_prompt

    def should_retry_chunk_without_context(self, text: str) -> bool:
        """Whether ordered chunk decoding should retry once without context."""
        return False

    def postprocess_text(self, text: str) -> str:
        """Strip model-specific markers from the decoded text.

        Default is an identity pass-through; override for models that emit
        special-token or diarization syntax.
        """
        return text

    def postprocess_plain_text(self, text: str) -> str:
        """Post-process ``response_format=text`` without changing legacy output.

        Existing adapters historically returned their raw model text for this
        format. Models that require clean plain text can opt in without
        changing that behavior for every other transcription pipeline.
        """
        return text

    def resolve_language(
        self,
        raw_text: str,
        requested_language: str | None,
    ) -> str | None:
        """Resolve the response language before model markup is removed.

        Most models do not emit a language marker, so the default preserves
        the caller's value. Adapters for language-detecting models can inspect
        ``raw_text`` and return the detected locale instead.
        """
        return requested_language

    @abstractmethod
    def build_verbose_response(
        self,
        text: str,
        language: str | None,
        audio_duration_s: float,
    ) -> TranscriptionVerboseResponse:
        """Build a verbose_json response with segments / timestamps."""

    def build_timestamped_response(
        self,
        text: str,
        language: str | None,
        audio_duration_s: float,
    ) -> TranscriptionVerboseResponse:
        """Build a response whose segments have model-derived timestamps."""
        raise ValueError("model did not produce segment timestamps")

    def build_verbose_response_from_chunks(
        self,
        *,
        text: str,
        chunks: list[tuple[float, float, str]],
        language: str | None,
        audio_duration_s: float,
    ) -> TranscriptionVerboseResponse:
        """verbose_json for chunked long audio: one segment per chunk."""
        segments: list[TranscriptionSegment] = []
        for start_s, end_s, chunk_text in chunks:
            stripped = self.postprocess_text(chunk_text).strip()
            if not stripped:
                continue
            segments.append(
                TranscriptionSegment(
                    id=len(segments),
                    start=round(start_s, 2),
                    end=round(end_s, 2),
                    text=stripped,
                )
            )
        return TranscriptionVerboseResponse(
            language=language,
            duration=round(max(float(audio_duration_s), 0.0), 2),
            text=text,
            segments=segments,
        )


class DefaultTranscriptionAdapter(TranscriptionAdapter):
    """Fallback: emit the whole transcript as a single segment."""

    def build_verbose_response(
        self,
        text: str,
        language: str | None,
        audio_duration_s: float,
    ) -> TranscriptionVerboseResponse:
        text = text.strip()
        segments = (
            [
                TranscriptionSegment(
                    id=0,
                    start=0.0,
                    end=round(max(float(audio_duration_s), 0.0), 2),
                    text=text,
                )
            ]
            if text
            else []
        )
        return TranscriptionVerboseResponse(
            language=language,
            duration=round(max(float(audio_duration_s), 0.0), 2),
            text=text,
            segments=segments,
        )


_ADAPTER_REGISTRY: dict[str, type[TranscriptionAdapter]] = {}
_DEFAULT_ADAPTER_KEY = "__default__"
_ADAPTER_REGISTRY[_DEFAULT_ADAPTER_KEY] = DefaultTranscriptionAdapter


def register_transcription_adapter(
    key: str,
) -> Callable[[type[TranscriptionAdapter]], type[TranscriptionAdapter]]:
    """Class decorator registering a TranscriptionAdapter under key.

    key is matched as a substring against the model's HF architectures at
    resolve time (e.g. MossTranscribeDiarize matches
    MossTranscribeDiarizeForConditionalGeneration).
    """

    def decorator(cls: type[TranscriptionAdapter]) -> type[TranscriptionAdapter]:
        _ADAPTER_REGISTRY[key] = cls
        return cls

    return decorator


def resolve_adapter(architectures: list[str] | None) -> TranscriptionAdapter:
    """Pick an adapter by matching architecture names against the registry."""
    for arch in architectures or []:
        if not arch:
            continue
        for key, adapter_cls in _ADAPTER_REGISTRY.items():
            if key != _DEFAULT_ADAPTER_KEY and key in arch:
                return adapter_cls()
    return _ADAPTER_REGISTRY[_DEFAULT_ADAPTER_KEY]()
