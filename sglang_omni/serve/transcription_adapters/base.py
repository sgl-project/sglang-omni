# SPDX-License-Identifier: Apache-2.0
"""Model-specific transcription output adapters.

Ported from the upstream SGLang transcription_adapters pattern: subclass
TranscriptionAdapter, decorate with register_transcription_adapter("Key"), and
the /v1/audio/transcriptions handler resolves the right adapter by matching Key
as a substring against the served model's HF architectures.

Only the pieces the omni HTTP layer needs are kept: markup post-processing and
verbose_json segment building. Sampling / prompt construction already live in
each model's pipeline (stages.py / request_builders.py), so they are
intentionally not part of this interface.
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

    def postprocess_text(self, text: str) -> str:
        """Strip model-specific markers from the decoded text.

        Default is an identity pass-through; override for models that emit
        special-token or diarization syntax.
        """
        return text

    @abstractmethod
    def build_verbose_response(
        self,
        text: str,
        language: str | None,
        audio_duration_s: float,
    ) -> TranscriptionVerboseResponse:
        """Build a verbose_json response with segments / timestamps."""


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
