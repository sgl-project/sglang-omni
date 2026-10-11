# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json

import pytest
import torch
from safetensors.torch import save_file

from sglang_omni.models.easymagpie_tts.speakers import (
    SPEAKER_SUBDIR,
    language_code,
    load_voice_catalog,
)


def write_registry(checkpoint, **overrides) -> None:
    voices = checkpoint / SPEAKER_SUBDIR
    voices.mkdir()
    save_file(
        {
            "Aria": torch.ones(2, 8),
            "context_fr": torch.full((2, 8), 2.0),
            "Remy": torch.zeros(3, 8),
        },
        voices / "contexts.safetensors",
    )
    registry = {
        "schema_version": 1,
        "default_speaker_id": "Aria",
        "tensor_bundle": "contexts.safetensors",
        "contexts": {
            "Aria": {"tensor_key": "Aria", "context_text": "aria en", "prompt_len": 9},
            "context_fr": {
                "tensor_key": "context_fr",
                "context_text": "aria fr",
                "prompt_len": 9,
            },
            "Remy": {"tensor_key": "Remy", "context_text": "remy fr", "prompt_len": 10},
        },
        "langs_supported": {"en": ["Aria"], "fr": ["Remy", "Aria"]},
        "voice_contexts": {
            "Aria": {"en": "Aria", "fr": "context_fr"},
            "Remy": {"fr": "Remy"},
        },
        **overrides,
    }
    (voices / "contexts.json").write_text(json.dumps(registry))


def test_registry_resolves_voice_and_language_to_a_context(tmp_path) -> None:
    write_registry(tmp_path)
    catalog = load_voice_catalog(tmp_path, 8)

    assert catalog.voices == ["Aria", "Remy"]
    assert catalog.resolve("", "") == "Aria"
    assert catalog.resolve("default", "Auto") == "Aria"
    assert catalog.resolve("Aria", "French") == "context_fr"
    assert catalog.resolve("", "fr-FR") == "Remy"
    assert catalog.resolve("Remy", "") == "Remy"
    assert catalog.context_texts["context_fr"] == "aria fr"
    assert catalog.rows["Remy"].shape == (3, 8)
    assert catalog.rows["Remy"].dtype == torch.float16


def test_registry_rejects_unsupported_voice_language_pairs(tmp_path) -> None:
    write_registry(tmp_path)
    catalog = load_voice_catalog(tmp_path, 8)

    with pytest.raises(ValueError, match="does not support language 'English'"):
        catalog.resolve("Remy", "English")
    with pytest.raises(ValueError, match="supported languages: \\['en', 'fr'\\]"):
        catalog.resolve("Aria", "German")
    with pytest.raises(ValueError, match="available voices: \\['Aria', 'Remy'\\]"):
        catalog.resolve("context_fr", "")


def test_registry_validates_contexts_and_default_voice(tmp_path) -> None:
    write_registry(tmp_path, default_speaker_id="Nobody")
    with pytest.raises(ValueError, match="default voice 'Nobody'"):
        load_voice_catalog(tmp_path, 8)


def test_registry_checks_the_embedding_width(tmp_path) -> None:
    write_registry(tmp_path)
    with pytest.raises(ValueError, match="must be \\[frames, 16\\]"):
        load_voice_catalog(tmp_path, 16)


def test_per_voice_files_ignore_language(tmp_path) -> None:
    voices = tmp_path / SPEAKER_SUBDIR
    voices.mkdir()
    torch.save({"speaker_encoding": torch.ones(3, 8)}, voices / "eng.pt")
    torch.save(torch.zeros(2, 8), voices / "alt.pt")
    catalog = load_voice_catalog(tmp_path, 8)

    assert catalog.default_voice == "eng"
    assert catalog.resolve("", "French") == "eng"
    assert catalog.resolve("alt", "") == "alt"
    assert catalog.context_texts == {}


def test_language_codes_accept_api_names_and_tags() -> None:
    assert language_code("Vietnamese") == "vi"
    assert language_code("pt_BR") == "pt"
    assert language_code("auto") == ""
