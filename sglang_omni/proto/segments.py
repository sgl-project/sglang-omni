# SPDX-License-Identifier: Apache-2.0
"""Ordered text and inline image segments shared by SDK and HTTP output."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field
from typing_extensions import TypedDict


class InlineImage(TypedDict):
    kind: Literal["image"]
    mime_type: Literal["image/png"]
    url: str
    sha256: str
    size_bytes: int


class UMMSegment(TypedDict):
    type: Literal["segment"]
    session_id: str
    segment_index: int
    kind: Literal["text", "image"]
    data: str | InlineImage


class InterleavedGenerationParams(BaseModel):
    """Controls for text-only interleaved image generation."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False, strict=True)

    mode: Literal["interleaved"]
    max_frames: int = Field(default=10, ge=1)
    text_max_new_tokens: int = Field(default=8192, ge=1)
    image_max_new_tokens: int = Field(default=1500, ge=1)
    dllm_steps: int = Field(default=32, ge=1)
    cfg_scale: float = Field(default=0.0, ge=0.0)
    cfg_text_scale: float = Field(default=7.5, ge=0.0)
    cfg_image_scale: float = Field(default=1.5, ge=0.0)
    cfg_rescale: float = Field(default=0.7, ge=0.0, le=1.0)
    decoder_steps: int | None = Field(default=None, ge=1)
    seed: int | None = Field(default=None, ge=0)
    max_image_tokens: int = Field(default=4096, ge=1)
    format: Literal["png"] = "png"
    decode_mode: Literal["normal", "decoder-turbo"] = "normal"


class CompletionImage(BaseModel):
    """Decoded image referenced by ordered assistant content."""

    id: str = Field(min_length=1)
    data: str
    format: Literal["png"] = "png"
    width: int = Field(gt=0)
    height: int = Field(gt=0)


class TextSegment(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["text"]
    text: str


class ImageRefSegment(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["image_ref"]
    image_id: str = Field(min_length=1)


def normalize_interleaved_content(
    content: list[dict[str, object]], images: list[dict[str, object]]
) -> list[dict[str, object]]:
    """Require unique image references in the same order as the image table."""
    image_ids = [CompletionImage.model_validate(image).id for image in images]
    segments = [
        (TextSegment if part.get("type") == "text" else ImageRefSegment)
        .model_validate(part)
        .model_dump()
        for part in content
    ]
    references = [part["image_id"] for part in segments if part["type"] == "image_ref"]
    if len(set(image_ids)) != len(image_ids) or references != image_ids:
        raise ValueError("interleaved content must reference each image once in order")
    else:
        pass
    return segments


def validate_interleaved_inputs(
    messages: list[dict[str, object]],
    modalities: list[str] | None,
    *,
    has_media: bool,
) -> None:
    if modalities is not None and (
        len(modalities) != 2 or set(modalities) != {"text", "image"}
    ):
        raise ValueError("interleaved generation requires text and image modalities")
    else:
        pass
    if has_media:
        raise ValueError("interleaved generation requires text-only input")
    else:
        pass
    for message in messages:
        if not isinstance(message, dict):
            raise ValueError("interleaved generation requires chat messages")
        else:
            pass
        content = message.get("content", "")
        if isinstance(content, str):
            continue
        else:
            pass
        if not isinstance(content, list) or any(
            not isinstance(part, str)
            and not (
                isinstance(part, dict)
                and part.get("type") == "text"
                and isinstance(part.get("text"), str)
            )
            for part in content
        ):
            raise ValueError("interleaved generation requires text-only input")
        else:
            pass
