# SPDX-License-Identifier: Apache-2.0
"""Context-audio input normalization and cache keys for CSM TTS."""

from __future__ import annotations

import base64
from collections.abc import Mapping
from pathlib import Path

from sglang_omni.models.csm_tts.utils import AudioSource, as_media_bytes
from sglang_omni.preprocessing.cache_key import (
    hash_media_item,
    reference_path_cache_key,
)

CONSUMED_CONTEXT_INPUT_KEYS = frozenset(
    {
        "context",
        "references",
        "reference_audio",
        "reference_text",
        "reference_codes",
        "reference_speaker",
    }
)


def as_audio_source(value: object) -> AudioSource:
    """Narrow one request-input value to a loadable audio source."""
    if isinstance(value, (str, Path, Mapping)):
        return value
    else:
        raise TypeError(
            f"context audio must be a path, a url, or a media mapping, got "
            f"{type(value).__name__}"
        )


def context_audio_cache_key(context_audio: AudioSource) -> str | None:
    """Stable cache key for one context-audio input."""
    if isinstance(context_audio, (str, Path)):
        return reference_path_cache_key(context_audio)
    else:
        pass
    path = context_audio.get("audio_path") or context_audio.get("path")
    if isinstance(path, (str, Path)):
        return reference_path_cache_key(path)
    else:
        pass
    raw_bytes = context_audio.get("bytes")
    if isinstance(raw_bytes, (bytes, bytearray, str)):
        return hash_media_item(as_media_bytes(raw_bytes))
    else:
        pass
    encoded = context_audio.get("base64") or context_audio.get("data")
    if isinstance(encoded, str):
        return hash_media_item(base64.b64decode(encoded))
    elif isinstance(encoded, (bytes, bytearray)):
        return hash_media_item(bytes(encoded))
    else:
        return None


def drop_consumed_context_inputs(inputs: Mapping[str, object]) -> dict[str, object]:
    """Copy of inputs without context / reference keys already consumed upstream."""
    return {
        key: value
        for key, value in inputs.items()
        if key not in CONSUMED_CONTEXT_INPUT_KEYS
    }


def normalize_context_inputs(
    inputs: Mapping[str, object], default_speaker: int
) -> list[dict[str, object]]:
    """Fold references / reference_audio into context segments."""
    context = inputs.get("context")
    if context:
        if not isinstance(context, list):
            raise TypeError(f"context must be a list, got {type(context).__name__}")
        else:
            pass
        return [dict(segment) for segment in context]
    else:
        pass

    segments: list[dict[str, object]] = []
    references = inputs.get("references")
    if references and isinstance(references, list):
        for reference in references:
            if not isinstance(reference, Mapping):
                continue
            else:
                pass
            segment: dict[str, object] = {
                "speaker": reference.get("speaker", default_speaker),
                "text": reference.get("text") or "",
            }
            if "bytes" in reference or "base64" in reference or "data" in reference:
                segment["audio"] = reference
            else:
                segment["audio"] = reference.get("audio_path") or reference.get("path")
            segments.append(segment)
        return segments
    else:
        pass

    reference_audio = inputs.get("reference_audio")
    if reference_audio is not None:
        segments.append(
            {
                "speaker": inputs.get("reference_speaker", default_speaker),
                "text": inputs.get("reference_text") or "",
                "audio": reference_audio,
            }
        )
    else:
        pass
    return segments


__all__ = [
    "CONSUMED_CONTEXT_INPUT_KEYS",
    "as_audio_source",
    "context_audio_cache_key",
    "drop_consumed_context_inputs",
    "normalize_context_inputs",
]
