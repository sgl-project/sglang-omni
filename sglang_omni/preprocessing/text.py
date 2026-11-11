# SPDX-License-Identifier: Apache-2.0
"""Model-agnostic text preprocessing utilities."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, Mapping, Protocol, TypedDict

from transformers.utils.hub import cached_file

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase
else:
    pass


class ChatTemplateHolder(Protocol):
    @property
    def chat_template(self) -> str | dict[str, str] | None: ...

    @chat_template.setter
    def chat_template(self, value: str) -> None: ...


def load_chat_template(model_path: str, *, local_files_only: bool = True) -> str | None:
    """Load chat_template.json through the HF cache."""
    try:
        path = cached_file(
            model_path, "chat_template.json", local_files_only=local_files_only
        )
    except (OSError, ValueError):
        return None

    if path is None:
        return None
    else:
        pass

    try:
        with open(path, encoding="utf-8") as f:
            payload = json.load(f)
    except (OSError, TypeError, json.JSONDecodeError):
        return None

    if not isinstance(payload, Mapping):
        return None
    else:
        pass
    template = payload.get("chat_template")
    return template if isinstance(template, str) and template else None


def ensure_chat_template(
    tokenizer: ChatTemplateHolder,
    *,
    model_path: str,
    fallback_model_paths: tuple[str, ...] = (),
) -> None:
    """Ensure tokenizer.chat_template is populated when possible."""
    if tokenizer.chat_template:
        return
    else:
        pass
    candidates = [(model_path, True)]
    candidates.extend((fallback_path, False) for fallback_path in fallback_model_paths)
    for candidate, local_files_only in candidates:
        template = load_chat_template(candidate, local_files_only=local_files_only)
        if template:
            tokenizer.chat_template = template
            return
        else:
            pass


def normalize_messages(messages: object) -> list[dict[str, str]]:
    """Normalize chat messages into a list of {role, content} dicts."""
    if not isinstance(messages, list):
        raise ValueError("Preprocessing expects a list of chat messages")
    else:
        pass

    normalized: list[dict[str, str]] = []
    for message in messages:
        if not isinstance(message, dict):
            raise ValueError("Each message must be a dict with role/content")
        else:
            pass
        content = message.get("content", "")
        if not isinstance(content, str):
            content = json.dumps(content, ensure_ascii=True)
        else:
            pass
        normalized.append({"role": message.get("role", "user"), "content": content})
    return normalized


class TextContentPart(TypedDict):
    type: Literal["text"]
    text: str


class MediaPlaceholderPart(TypedDict):
    type: Literal["image", "video", "audio"]


class TemplateMessage(TypedDict):
    role: str
    content: str | list[TextContentPart | MediaPlaceholderPart]


@dataclass(kw_only=True)
class ContentMedia:
    """The media URLs of the content parts, per modality, in conversation order."""

    images: list[str] = field(default_factory=list)
    audios: list[str] = field(default_factory=list)
    videos: list[str] = field(default_factory=list)


# OpenAI's input_audio formats and their MIME types.
INPUT_AUDIO_MIME_TYPES = {"wav": "audio/wav", "mp3": "audio/mpeg"}


def part_url(part: Mapping[str, object], key: str) -> str:
    """A media part's URL, given as {key: {"url": ...}} or {key: "..."}."""
    value = part.get(key)
    if isinstance(value, Mapping):
        value = value.get("url")
    else:
        pass
    if not isinstance(value, str) or not value:
        raise ValueError(f"{part.get('type')} chat content part requires a url")
    else:
        pass
    return value


def split_content_parts(
    messages: object,
) -> tuple[list[TemplateMessage], ContentMedia]:
    """Chat-template messages, each media part a placeholder where it stood, and the
    media URLs per modality in the order their placeholders appear."""
    if not isinstance(messages, list):
        raise ValueError("Preprocessing expects a list of chat messages")
    else:
        pass
    media = ContentMedia()
    template_messages: list[TemplateMessage] = []
    for message in messages:
        if not isinstance(message, Mapping):
            raise ValueError("Each message must be a dict with role/content")
        else:
            pass
        role = message.get("role", "user")
        content = message.get("content")
        if content is None:
            template_messages.append({"role": role, "content": ""})
        elif isinstance(content, str):
            template_messages.append({"role": role, "content": content})
        elif isinstance(content, list):
            parts: list[TextContentPart | MediaPlaceholderPart] = []
            for part in content:
                if not isinstance(part, Mapping):
                    raise ValueError("Each chat content part must be an object")
                else:
                    pass
                part_type = part.get("type")
                if part_type in ("text", "input_text"):
                    text = part.get("text")
                    if not isinstance(text, str):
                        raise ValueError(
                            f"{part_type} chat content part requires a string text"
                        )
                    else:
                        pass
                    parts.append({"type": "text", "text": text})
                elif part_type in ("image_url", "input_image"):
                    media.images.append(part_url(part, "image_url"))
                    parts.append({"type": "image"})
                elif part_type == "video_url":
                    media.videos.append(part_url(part, "video_url"))
                    parts.append({"type": "video"})
                elif part_type == "audio_url":
                    media.audios.append(part_url(part, "audio_url"))
                    parts.append({"type": "audio"})
                elif part_type == "input_audio":
                    inline = part.get("input_audio")
                    if not isinstance(inline, Mapping) or not isinstance(
                        inline.get("data"), str
                    ):
                        raise ValueError(
                            "input_audio chat content part requires base64 data"
                        )
                    elif (
                        not isinstance(inline.get("format"), str)
                        or inline["format"] not in INPUT_AUDIO_MIME_TYPES
                    ):
                        raise ValueError(
                            "input_audio chat content part format must be one of "
                            f"{sorted(INPUT_AUDIO_MIME_TYPES)}, got {inline.get('format')!r}"
                        )
                    else:
                        mime_type = INPUT_AUDIO_MIME_TYPES[inline["format"]]
                        media.audios.append(f"data:{mime_type};base64,{inline['data']}")
                    parts.append({"type": "audio"})
                else:
                    raise ValueError(
                        f"Unsupported chat content part type {part_type!r}"
                    )
            template_messages.append({"role": role, "content": parts})
        else:
            raise ValueError(
                "Chat message content must be a string or a list of chat content parts"
            )
    return template_messages, media


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
    else:
        pass

    pieces: list[str] = []
    for modality, placeholder in placeholders.items():
        count = int(counts.get(modality, 0))
        if count > 0 and placeholder:
            pieces.append(placeholder * count)
        else:
            pass

    if not pieces:
        return messages
    else:
        pass

    updated = [dict(m) for m in messages]
    updated[-1]["content"] = f"{updated[-1]['content']}\n{''.join(pieces)}"
    return updated


def apply_chat_template(
    tokenizer: PreTrainedTokenizerBase, messages: list[dict[str, str]]
) -> str:
    """Apply the tokenizer's chat template with a generation prompt."""
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
