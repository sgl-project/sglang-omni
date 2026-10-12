# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Preset voice contexts: ``speaker_embeddings/contexts.{safetensors,json}``.

Uses the context registry format of the Nemotron-Speech repository builder:

* A CSV row ``speaker_name, languages, context_text`` yields one context per
  ``(speaker, primary language)``. A speaker's first context keeps the speaker
  name as its context id; further languages get ``context_<sha256[:32]>``.
* Text-only contexts all share NeMo's omitted-audio tensor (empty context codes
  through BOS/EOS, frame stacking, and the speaker encoder); voices differ only
  by their paired ``context_text``.
* ``prompt_len`` is ``task_tokens + context frames + len(context_text tokens)``.
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import save_file

from easymagpie_converter.talker import write_json

LOG = logging.getLogger(__name__)

SPEAKER_SUBDIR = "speaker_embeddings"
BUNDLE_FILE = "contexts.safetensors"
REGISTRY_FILE = "contexts.json"
BUNDLE_FORMAT = "easymagpie-speaker-contexts-v1"
VOICE_LABEL = re.compile(r"[\x20-\x7e]{1,64}\Z")
REQUIRED_COLUMNS = {"speaker_name", "languages", "context_text"}


@dataclass(frozen=True)
class ContextSpec:
    context_id: str
    speaker: str
    language: str
    context_text: str


@dataclass
class ContextPlan:
    contexts: list[ContextSpec]
    default_speaker_id: str
    voice_contexts: dict[str, dict[str, str]]
    langs_supported: dict[str, list[str]]


def is_voice_label(value: Any) -> bool:
    return (
        isinstance(value, str)
        and bool(VOICE_LABEL.fullmatch(value))
        and value.strip() == value
        and not any(character in value for character in "/\\:")
        and ".." not in value
    )


def primary_language(value: str) -> str:
    subtags = value.split("-")
    valid = (
        value.isascii()
        and len(value) <= 35
        and 2 <= len(subtags[0]) <= 8
        and subtags[0].isalpha()
        and all(1 <= len(subtag) <= 8 and subtag.isalnum() for subtag in subtags[1:])
    )
    if not valid:
        raise ValueError(f"invalid context language {value!r}")
    return subtags[0].lower()


def plan_contexts(csv_path: Path, default_voice: str | None) -> ContextPlan:
    """Parse the context CSV into context ids and the voice/language registry."""
    selected: dict[tuple[str, str], str] = {}
    with csv_path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        missing = REQUIRED_COLUMNS - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{csv_path} is missing columns {sorted(missing)}")
        for row in reader:
            text = (row.get("context_text") or "").strip()
            if not text:
                continue
            speaker = row.get("speaker_name")
            if not is_voice_label(speaker):
                raise ValueError(f"invalid speaker_name {speaker!r}")
            languages = (row.get("languages") or "").strip()
            if not languages:
                raise ValueError(f"speaker {speaker!r} has no languages")
            for language in re.split(r"[,;\s]+", languages):
                pair = (speaker, primary_language(language))
                if pair in selected:
                    raise ValueError(f"duplicate context for {pair}")
                selected[pair] = text
    if not selected:
        raise ValueError(f"{csv_path} contains no usable contexts")

    speakers = {speaker for speaker, _ in selected}
    default_speaker_id = default_voice or next(iter(selected))[0]
    if default_speaker_id not in speakers:
        raise ValueError(f"default voice {default_speaker_id!r} is not in {csv_path}")
    contexts: list[ContextSpec] = []
    voice_contexts: dict[str, dict[str, str]] = {}
    langs_supported: dict[str, list[str]] = {}
    for (speaker, language), text in selected.items():
        languages = voice_contexts.setdefault(speaker, {})
        if languages:
            digest = hashlib.sha256(json.dumps([speaker, language]).encode()).hexdigest()
            context_id = f"context_{digest[:32]}"
            if context_id in speakers:
                raise ValueError(f"generated context id {context_id} collides with a voice")
        else:
            context_id = speaker
        languages[language] = context_id
        langs_supported.setdefault(language, []).append(speaker)
        contexts.append(ContextSpec(context_id, speaker, language, text))
    return ContextPlan(contexts, default_speaker_id, voice_contexts, langs_supported)


@torch.no_grad()
def text_only_context(model: Any, context_text: str) -> torch.Tensor:
    """NeMo's omitted-audio context rows ``[frames, embedding_dim]`` (float32)."""
    device = next(model.parameters()).device
    tokens = model.tokenizer.encode(context_text, tokenizer_name=model.text_conditioning_tokenizer_name)
    embedding, _, _, audio_lens = model.prepare_context_tensors(
        context_text_tokens=torch.tensor([tokens], dtype=torch.long, device=device),
        context_text_tokens_lens=torch.tensor([len(tokens)], dtype=torch.long, device=device),
        context_audio_codes=torch.zeros(
            1, model.data_num_audio_codebooks, 0, dtype=torch.long, device=device
        ),
        context_audio_codes_lens=torch.zeros(1, dtype=torch.long, device=device),
    )
    task_tokens = int(model.task_embedding is not None and model.training_modes[0].mode_idx is not None)
    frames = int(audio_lens[0].item())
    return embedding[0, task_tokens : task_tokens + frames].float().cpu().contiguous()


def validate_tensor(tensor: torch.Tensor, embedding_dim: int) -> None:
    if tensor.ndim != 2 or tensor.shape[0] <= 0 or tensor.shape[1] != embedding_dim:
        raise ValueError(
            f"context tensor has shape {tuple(tensor.shape)}; expected [frames, {embedding_dim}]"
        )
    if not bool(torch.isfinite(tensor).all()):
        raise ValueError("context tensor contains non-finite values")


def write_contexts(model: Any, output: Path, plan: ContextPlan, config: dict[str, Any]) -> dict[str, Any]:
    """Write the tensor bundle and registry; return the registry."""
    from transformers import AutoTokenizer

    embedding_dim = int(config["embedding_dim"])
    shared = text_only_context(model, plan.contexts[0].context_text)
    validate_tensor(shared, embedding_dim)
    task_tokens = 1 if int(config.get("num_task_embeddings", 0)) > 0 else 0
    tokenizer = AutoTokenizer.from_pretrained(output, local_files_only=True)
    text_vocab_size = int(config["text_vocab_size"])

    tensors: dict[str, torch.Tensor] = {}
    entries: dict[str, dict[str, Any]] = {}
    for context in plan.contexts:
        token_ids = tokenizer.encode(context.context_text, add_special_tokens=False)
        if not token_ids or any(not 0 <= token < text_vocab_size for token in token_ids):
            raise ValueError(f"context {context.context_id!r} produced out-of-vocabulary token ids")
        tensors[context.context_id] = shared.clone()
        entries[context.context_id] = {
            "tensor_key": context.context_id,
            "context_text": context.context_text,
            "prompt_len": task_tokens + int(shared.shape[0]) + len(token_ids),
        }

    directory = output / SPEAKER_SUBDIR
    directory.mkdir(parents=True, exist_ok=True)
    save_file(tensors, directory / BUNDLE_FILE, metadata={"format": BUNDLE_FORMAT})
    registry = {
        "schema_version": 1,
        "default_speaker_id": plan.default_speaker_id,
        "tensor_bundle": BUNDLE_FILE,
        "contexts": entries,
        "langs_supported": plan.langs_supported,
        "voice_contexts": plan.voice_contexts,
    }
    write_json(directory / REGISTRY_FILE, registry)
    LOG.info(
        "Saved %d contexts for %d voices (%d context frames each)",
        len(entries),
        len(plan.voice_contexts),
        int(shared.shape[0]),
    )
    return registry


__all__ = ["ContextPlan", "ContextSpec", "plan_contexts", "write_contexts"]
