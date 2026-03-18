# SPDX-License-Identifier: Apache-2.0
"""Model-agnostic text preprocessing utilities."""

from __future__ import annotations

import json
from typing import Any, Mapping

from transformers.utils.hub import cached_file


def load_chat_template(model_path: str) -> str | None:
    """Load chat_template.json from the local HF cache if available."""
    try:
        path = cached_file(model_path, "chat_template.json", local_files_only=True)
    except Exception:
        return None

    try:
        with open(path, encoding="utf-8") as f:
            payload = json.load(f)
    except Exception:
        return None

    if not isinstance(payload, Mapping):
        return None
    template = payload.get("chat_template")
    return template if isinstance(template, str) and template else None


def ensure_chat_template(tokenizer: Any, *, model_path: str) -> None:
    """Ensure tokenizer.chat_template is populated when possible."""
    if getattr(tokenizer, "chat_template", None):
        return
    template = load_chat_template(model_path)
    if template:
        tokenizer.chat_template = template


def normalize_messages(messages: Any) -> list[dict[str, str]]:
    """Normalize chat messages into a list of {role, content} dicts."""
    if not isinstance(messages, list):
        raise ValueError("Preprocessing expects a list of chat messages")

    normalized: list[dict[str, str]] = []
    for message in messages:
        if not isinstance(message, dict):
            raise ValueError("Each message must be a dict with role/content")
        content = message.get("content", "")
        if not isinstance(content, str):
            content = json.dumps(content, ensure_ascii=True)
        normalized.append({"role": message.get("role", "user"), "content": content})
    return normalized


def append_modality_placeholders(
    messages: list[dict[str, str]],
    *,
    placeholders: Mapping[str, str],
    counts: Mapping[str, int],
) -> list[dict[str, str]]:
    """Append modality placeholders to the last message.

    This keeps the policy simple and model-agnostic: the caller controls both
    placeholder strings and modality counts.
    """
    if not messages:
        return messages

    pieces: list[str] = []
    for modality, placeholder in placeholders.items():
        count = int(counts.get(modality, 0))
        if count > 0 and placeholder:
            pieces.append(placeholder * count)

    if not pieces:
        return messages

    updated = [dict(m) for m in messages]
    updated[-1]["content"] = f"{updated[-1]['content']}\n{''.join(pieces)}"
    return updated


def apply_chat_template(tokenizer: Any, messages: list[dict[str, str]]) -> str:
    """Apply the tokenizer's chat template with a generation prompt."""
    if not hasattr(tokenizer, "apply_chat_template"):
        raise ValueError("Tokenizer does not support apply_chat_template")
    if not getattr(tokenizer, "chat_template", None):
        raise ValueError("Tokenizer chat_template is not set")
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
