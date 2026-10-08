# SPDX-License-Identifier: Apache-2.0
"""Load ordered chat content while retaining audio turn ownership."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TypedDict

import numpy as np
import numpy.typing as npt
import torch
from PIL import Image

from sglang_omni.models.minicpm_o.video_frontend import load_timed_video
from sglang_omni.preprocessing.audio import ensure_audio_list_async
from sglang_omni.preprocessing.image import ensure_image_list_async

IMAGE_PLACEHOLDER = "<image>./</image>"
AUDIO_PLACEHOLDER = "<audio>./</audio>"


def first_batch_item(
    batch_values: torch.Tensor | list[torch.Tensor] | list[list[torch.Tensor]],
) -> torch.Tensor | list[torch.Tensor] | None:
    """Remove the single-request batch dimension from processor outputs."""
    if isinstance(batch_values, list):
        return batch_values[0] if batch_values else None
    else:
        return batch_values[0]


def video_to_images(video: object) -> list[Image.Image]:
    """Convert normalized video pixels to RGB frames."""
    if isinstance(video, list) and all(
        isinstance(frame, Image.Image) for frame in video
    ):
        return [frame.convert("RGB") for frame in video]
    elif isinstance(video, (torch.Tensor, np.ndarray)):
        frame_tensor = torch.as_tensor(video)
    else:
        raise ValueError(
            "MiniCPM-o video inputs must be frame images or a pixel tensor"
        )
    if frame_tensor.ndim != 4:
        raise ValueError(
            f"MiniCPM-o video inputs must have shape (T, C, H, W), got {tuple(frame_tensor.shape)}"
        )
    elif frame_tensor.shape[1] in (1, 3, 4):
        frame_tensor = frame_tensor.permute(0, 2, 3, 1)
    elif frame_tensor.shape[-1] not in (1, 3, 4):
        raise ValueError(
            f"MiniCPM-o video frames must have 1, 3, or 4 channels, got {tuple(frame_tensor.shape)}"
        )
    else:
        pass
    frame_tensor = frame_tensor.detach().cpu()
    if (
        frame_tensor.is_floating_point()
        and frame_tensor.numel()
        and float(frame_tensor.max()) <= 1.0
    ):
        frame_tensor = frame_tensor * 255.0
    else:
        pass
    frame_tensor = frame_tensor.clamp(0, 255).to(torch.uint8)
    return [Image.fromarray(frame.numpy()).convert("RGB") for frame in frame_tensor]


class RenderedMessage(TypedDict):
    role: str
    content: str


@dataclass(kw_only=True)
class RenderedChat:
    messages: list[RenderedMessage] = field(default_factory=list)
    images: list[Image.Image] = field(default_factory=list)
    audios: list[npt.NDArray[np.float32]] = field(default_factory=list)
    audio_turn_indices: list[int] = field(default_factory=list)
    has_video: bool = False
    has_interleaved_video_audio: bool = False


def normalize_message_contents(messages: object) -> str | list[RenderedMessage]:
    """Join text parts using the chat template's content separator."""
    if isinstance(messages, str):
        return messages
    elif not isinstance(messages, list):
        raise ValueError("Chat messages must be a string or list")
    else:
        normalized_messages: list[RenderedMessage] = []
        for message in messages:
            if not isinstance(message, dict):
                raise ValueError("Each chat message must contain role and content")
            else:
                role = message.get("role", "user")
                content = message.get("content", "")
            if not isinstance(role, str):
                raise ValueError("Chat role must be a string")
            elif isinstance(content, list):
                content_fragments: list[str] = []
                for part in content:
                    if isinstance(part, str):
                        content_fragments.append(part)
                    elif (
                        isinstance(part, dict)
                        and part.get("type") == "text"
                        and isinstance(part.get("text"), str)
                    ):
                        content_fragments.append(part["text"])
                    else:
                        raise ValueError("Unsupported MiniCPM-o text content part")
                content = "\n".join(content_fragments)
            elif not isinstance(content, str):
                raise ValueError("Chat content must be a string or ordered parts")
            else:
                pass
            normalized_messages.append({**message, "role": role, "content": content})
        return normalized_messages


def messages_with_media_placeholders(
    messages: object, *, image_count: int, audio_count: int
) -> list[RenderedMessage]:
    """Prepend top-level media placeholders to the last user message."""
    normalized_messages = normalize_message_contents(messages)
    assert isinstance(normalized_messages, list)
    rendered_messages: list[RenderedMessage] = []
    for message_index, message in enumerate(normalized_messages):
        if message_index == len(normalized_messages) - 1 and message["role"] == "user":
            content_fragments = (
                [IMAGE_PLACEHOLDER] * image_count
                + [AUDIO_PLACEHOLDER] * audio_count
                + [message["content"]]
            )
            rendered_messages.append(
                {**message, "content": "\n".join(content_fragments)}
            )
        else:
            rendered_messages.append(message)
    return rendered_messages


def has_inline_media(messages: object) -> bool:
    if not isinstance(messages, list):
        return False
    else:
        return any(
            isinstance(message, dict)
            and isinstance(message.get("content"), list)
            and any(
                isinstance(part, (Image.Image, np.ndarray))
                or isinstance(part, dict)
                and part.get("type") != "text"
                for part in message["content"]
            )
            for message in messages
        )


async def render_ordered_chat(
    messages: object,
    *,
    use_audio_in_video: bool = False,
    video_fps: float | None = None,
    video_max_frames: int | None = None,
    video_min_pixels: int | None = None,
    video_max_pixels: int | None = None,
    video_total_pixels: int | None = None,
) -> RenderedChat:
    if not isinstance(messages, list):
        raise ValueError("Chat messages must be a list")
    else:
        pass
    rendered_chat = RenderedChat()
    message_content_fragments: list[list[str]] = []
    for turn_index, message in enumerate(messages):
        if not isinstance(message, dict):
            raise ValueError("Each chat message must contain role and content")
        else:
            pass
        role = message.get("role", "user")
        content = message.get("content", "")
        if not isinstance(role, str):
            raise ValueError("Chat role must be a string")
        elif isinstance(content, str):
            rendered_chat.messages.append({"role": role, "content": content})
            message_content_fragments.append([content])
            continue
        elif not isinstance(content, list):
            raise ValueError("Chat content must be a string or ordered parts")
        else:
            pass
        content_fragments: list[str] = []
        for part in content:
            if isinstance(part, str):
                content_fragments.append(part)
            elif isinstance(part, Image.Image):
                rendered_chat.images.append(part.convert("RGB"))
                content_fragments.append(IMAGE_PLACEHOLDER)
            elif isinstance(part, np.ndarray):
                rendered_chat.audios.append(part.astype(np.float32, copy=False))
                rendered_chat.audio_turn_indices.append(turn_index)
                content_fragments.append(AUDIO_PLACEHOLDER)
            elif isinstance(part, dict):
                part_type = part.get("type")
                if part_type == "text":
                    text = part.get("text")
                    if not isinstance(text, str):
                        raise ValueError("Text content parts require a string text")
                    else:
                        content_fragments.append(text)
                elif part_type in ("image_url", "audio_url", "input_audio"):
                    media_reference = part.get(part_type)
                    if part_type == "input_audio":
                        if (
                            not isinstance(media_reference, dict)
                            or not isinstance(media_reference.get("data"), str)
                            or not isinstance(media_reference.get("format"), str)
                        ):
                            raise ValueError(
                                "input_audio requires base64 data and format"
                            )
                        else:
                            media_url = f"data:audio/{media_reference['format']};base64,{media_reference['data']}"
                    elif isinstance(media_reference, str):
                        media_url = media_reference
                    elif isinstance(media_reference, dict) and isinstance(
                        media_reference.get("url"), str
                    ):
                        media_url = media_reference["url"]
                    else:
                        raise ValueError(f"{part_type} requires a media URL")
                    if part_type == "image_url":
                        images = await ensure_image_list_async([media_url])
                        image = images[0]
                        assert isinstance(image, Image.Image)
                        rendered_chat.images.append(image)
                        content_fragments.append(IMAGE_PLACEHOLDER)
                    else:
                        audios = await ensure_audio_list_async(
                            [media_url], target_sr=16000
                        )
                        audio_waveform = audios[0]
                        assert isinstance(audio_waveform, np.ndarray)
                        rendered_chat.audios.append(audio_waveform)
                        rendered_chat.audio_turn_indices.append(turn_index)
                        content_fragments.append(AUDIO_PLACEHOLDER)
                elif part_type == "video_url":
                    media_reference = part.get("video_url")
                    if isinstance(media_reference, str):
                        media_url = media_reference
                        use_audio = use_audio_in_video
                    elif isinstance(media_reference, dict) and isinstance(
                        media_reference.get("url"), str
                    ):
                        media_url = media_reference["url"]
                        use_audio = bool(
                            media_reference.get("use_audio", use_audio_in_video)
                        )
                        if media_reference.get("stack_frames", 1) != 1:
                            raise ValueError(
                                "MiniCPM-o video stack_frames currently supports only 1"
                            )
                        else:
                            pass
                    else:
                        raise ValueError("video_url requires a media URL")
                    video = await load_timed_video(
                        media_url,
                        use_audio=use_audio,
                        fps=video_fps,
                        max_frames=video_max_frames,
                        min_pixels=video_min_pixels,
                        max_pixels=video_max_pixels,
                        total_pixels=video_total_pixels,
                    )
                    rendered_chat.has_video = True
                    rendered_chat.has_interleaved_video_audio = (
                        rendered_chat.has_interleaved_video_audio
                        or bool(video.audio_segments)
                    )
                    for frame_index, frame in enumerate(video.frames):
                        rendered_chat.images.append(frame)
                        content_fragments.append(IMAGE_PLACEHOLDER)
                        if video.audio_segments:
                            rendered_chat.audios.append(
                                video.audio_segments[frame_index]
                            )
                            rendered_chat.audio_turn_indices.append(turn_index)
                            content_fragments.append(AUDIO_PLACEHOLDER)
                        else:
                            pass
                else:
                    raise ValueError(f"Unsupported MiniCPM-o content type: {part_type}")
            else:
                raise ValueError("Unsupported MiniCPM-o content part")
        rendered_chat.messages.append(
            {"role": role, "content": "\n".join(content_fragments)}
        )
        message_content_fragments.append(content_fragments)
    if rendered_chat.has_interleaved_video_audio:
        for message, content_fragments in zip(
            rendered_chat.messages, message_content_fragments
        ):
            message["content"] = "".join(content_fragments)
    else:
        pass
    return rendered_chat
