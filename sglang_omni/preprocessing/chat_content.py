# SPDX-License-Identifier: Apache-2.0
"""OpenAI chat content parts as chat-template parts, with each media part's URL in order."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Literal, TypedDict


class TextPart(TypedDict):
    type: Literal["text"]
    text: str


class MediaPart(TypedDict):
    type: Literal["image", "audio", "video"]


class TemplateMessage(TypedDict):
    role: str
    content: str | list[TextPart | MediaPart]


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


def template_part(part: object, media: ContentMedia) -> TextPart | MediaPart:
    """One content part as the chat template takes it; its media URL joins media."""
    if not isinstance(part, Mapping):
        raise ValueError("Each chat content part must be an object")
    else:
        pass
    part_type = part.get("type")
    if part_type in ("text", "input_text"):
        text = part.get("text")
        if not isinstance(text, str):
            raise ValueError(f"{part_type} chat content part requires a string text")
        else:
            pass
        return {"type": "text", "text": text}
    elif part_type in ("image_url", "input_image"):
        media.images.append(part_url(part, "image_url"))
        return {"type": "image"}
    elif part_type == "video_url":
        media.videos.append(part_url(part, "video_url"))
        return {"type": "video"}
    elif part_type == "audio_url":
        media.audios.append(part_url(part, "audio_url"))
        return {"type": "audio"}
    elif part_type == "input_audio":
        inline = part.get("input_audio")
        if not isinstance(inline, Mapping) or not isinstance(inline.get("data"), str):
            raise ValueError("input_audio chat content part requires base64 data")
        elif inline.get("format") not in INPUT_AUDIO_MIME_TYPES:
            raise ValueError(
                "input_audio chat content part format must be one of "
                f"{sorted(INPUT_AUDIO_MIME_TYPES)}, got {inline.get('format')!r}"
            )
        else:
            mime_type = INPUT_AUDIO_MIME_TYPES[inline["format"]]
            media.audios.append(f"data:{mime_type};base64,{inline['data']}")
        return {"type": "audio"}
    else:
        raise ValueError(f"Unsupported chat content part type {part_type!r}")


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
            parts = [template_part(part, media) for part in content]
            template_messages.append({"role": role, "content": parts})
        else:
            raise ValueError(
                "Chat message content must be a string or a list of chat content parts"
            )
    return template_messages, media
