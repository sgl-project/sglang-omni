# SPDX-License-Identifier: Apache-2.0
"""EasyMagpie preset voices.

Each context is a fixed block of prompt rows. The engine keeps every context
on the GPU in one table, so requests carry only the context id and its frame
count instead of the rows themselves.

A model directory provides contexts in one of two layouts under
``speaker_embeddings/``:

* ``contexts.json`` + ``contexts.safetensors``: a registry mapping each voice
  and language to a context id, its tensor, and its paired context text.
* ``<voice>.pt``: one context per voice, without paired text or languages.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import torch

from sglang_omni.models.easymagpie_tts.payload_types import DEFAULT_VOICE

SPEAKER_SUBDIR = "speaker_embeddings"
REGISTRY_FILE = "contexts.json"

# Speech API language names -> primary language subtags used by the registry.
LANGUAGE_CODES = {
    "arabic": "ar",
    "chinese": "zh",
    "english": "en",
    "french": "fr",
    "german": "de",
    "hindi": "hi",
    "italian": "it",
    "japanese": "ja",
    "korean": "ko",
    "portuguese": "pt",
    "russian": "ru",
    "spanish": "es",
    "vietnamese": "vi",
}


def language_code(language: str) -> str:
    """Primary language subtag for an API language name or tag; "" for none/auto."""
    value = language.strip().casefold()
    if value in ("", "auto"):
        return ""
    else:
        return LANGUAGE_CODES.get(value, value.replace("_", "-").split("-")[0])


@dataclass(frozen=True)
class VoiceCatalog:
    """Context rows plus the voice/language registry that selects among them."""

    rows: dict[str, torch.Tensor]
    default_voice: str
    context_texts: dict[str, str] = field(default_factory=dict)
    voice_contexts: dict[str, dict[str, str]] = field(default_factory=dict)
    langs_supported: dict[str, list[str]] = field(default_factory=dict)

    @property
    def voices(self) -> list[str]:
        return sorted(self.voice_contexts or self.rows)

    def resolve(self, voice: str, language: str) -> str:
        """Context id for a request's voice and language.

        A missing voice falls back to the language's first voice, then to the
        default voice; a missing language picks the voice's first context.
        """
        name = voice.strip()
        if name.casefold() == "default":
            name = ""
        else:
            pass
        code = language_code(language)
        if self.langs_supported and code and code not in self.langs_supported:
            raise ValueError(
                f"EasyMagpie has no voice for language {language!r}; "
                f"supported languages: {sorted(self.langs_supported)}"
            )
        else:
            pass
        if not name and code in self.langs_supported:
            name = self.langs_supported[code][0]
        elif not name:
            name = self.default_voice
        else:
            pass
        contexts = self.voice_contexts.get(name)
        if contexts is None and name in self.rows and not self.voice_contexts:
            return name
        elif contexts is None:
            raise ValueError(
                f"Unknown EasyMagpie voice {name!r}; available voices: {self.voices}"
            )
        elif not code:
            return next(iter(contexts.values()))
        elif code not in contexts:
            raise ValueError(
                f"EasyMagpie voice {name!r} does not support language {language!r}; "
                f"it supports {sorted(contexts)}"
            )
        else:
            return contexts[code]


def checked_rows(
    name: str, embedding: torch.Tensor, embedding_dim: int
) -> torch.Tensor:
    if embedding.ndim != 2 or embedding.shape[1] != embedding_dim:
        raise ValueError(
            f"EasyMagpie context {name} must be [frames, {embedding_dim}], "
            f"got {tuple(embedding.shape)}"
        )
    else:
        pass
    return embedding.detach().to(torch.float16)


def load_registry(directory: Path, embedding_dim: int) -> VoiceCatalog:
    from safetensors.torch import load_file

    registry = json.loads((directory / REGISTRY_FILE).read_text(encoding="utf-8"))
    if registry.get("schema_version") != 1:
        raise ValueError(f"{directory / REGISTRY_FILE} must use schema_version 1")
    else:
        pass
    tensors = load_file(str(directory / registry["tensor_bundle"]), device="cpu")
    contexts = registry["contexts"]
    rows, texts = {}, {}
    for context_id, entry in contexts.items():
        rows[context_id] = checked_rows(
            context_id, tensors[entry["tensor_key"]], embedding_dim
        )
        texts[context_id] = str(entry["context_text"])
    # Without voice_contexts, voices are the context ids themselves.
    voice_contexts = dict(registry.get("voice_contexts") or {})
    for voice, languages in voice_contexts.items():
        unknown = sorted(set(languages.values()) - set(rows))
        if not languages or unknown:
            raise ValueError(
                f"EasyMagpie voice {voice!r} must map languages to known contexts"
            )
        else:
            pass
    default_voice = str(registry["default_speaker_id"])
    if default_voice not in (voice_contexts or rows):
        raise ValueError(
            f"EasyMagpie default voice {default_voice!r} is not registered"
        )
    else:
        pass
    return VoiceCatalog(
        rows=rows,
        default_voice=default_voice,
        context_texts=texts,
        voice_contexts=voice_contexts,
        langs_supported=dict(registry.get("langs_supported") or {}),
    )


def load_voice_catalog(checkpoint: Path, embedding_dim: int) -> VoiceCatalog:
    directory = checkpoint / SPEAKER_SUBDIR
    if (directory / REGISTRY_FILE).is_file():
        return load_registry(directory, embedding_dim)
    else:
        pass
    rows = {}
    for path in sorted(directory.glob("*.pt")):
        loaded = torch.load(path, map_location="cpu", weights_only=True)
        if isinstance(loaded, dict):
            embedding = loaded["speaker_encoding"]
        else:
            embedding = loaded
        rows[path.stem] = checked_rows(str(path), embedding, embedding_dim)
    if not rows:
        raise ValueError(f"No EasyMagpie voices found under {directory}")
    else:
        pass
    default_voice = DEFAULT_VOICE if DEFAULT_VOICE in rows else next(iter(rows))
    return VoiceCatalog(rows=rows, default_voice=default_voice)


def load_speaker_embeddings(
    checkpoint: Path, embedding_dim: int
) -> dict[str, torch.Tensor]:
    """Every context as a [frames, embedding_dim] float16 tensor, by context id."""
    return load_voice_catalog(checkpoint, embedding_dim).rows


@dataclass
class SpeakerTable:
    """Every context's rows back to back, with each context's first row and length."""

    rows: torch.Tensor
    spans: dict[str, tuple[int, int]]

    @classmethod
    def from_voices(
        cls,
        voices: dict[str, torch.Tensor],
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> SpeakerTable:
        spans, start = {}, 0
        for name, embedding in voices.items():
            spans[name] = (start, int(embedding.shape[0]))
            start += int(embedding.shape[0])
        rows = torch.cat(list(voices.values()), dim=0).to(device=device, dtype=dtype)
        return cls(rows=rows, spans=spans)


__all__ = [
    "SPEAKER_SUBDIR",
    "SpeakerTable",
    "VoiceCatalog",
    "language_code",
    "load_speaker_embeddings",
    "load_voice_catalog",
]
