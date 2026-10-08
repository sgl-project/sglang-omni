# SPDX-License-Identifier: Apache-2.0
"""Load ordered chat content while retaining audio turn ownership."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
from PIL import Image

from sglang_omni.models.minicpm_o.video_frontend import load_timed_video
from sglang_omni.preprocessing.audio import ensure_audio_list_async
from sglang_omni.preprocessing.image import ensure_image_list_async

IMAGE_PLACEHOLDER = "<image>./</image>"
AUDIO_PLACEHOLDER = "<audio>./</audio>"


@dataclass(kw_only=True)
class RenderedChat:
    messages: list[dict[str, str]] = field(default_factory=list)
    images: list[Image.Image] = field(default_factory=list)
    audios: list[npt.NDArray[np.float32]] = field(default_factory=list)
    audio_turn_indices: list[int] = field(default_factory=list)
    has_video: bool = False
    omni_mode: bool = False


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
    rendered = RenderedChat()
    message_pieces: list[list[str]] = []
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
            rendered.messages.append({"role": role, "content": content})
            message_pieces.append([content])
            continue
        elif not isinstance(content, list):
            raise ValueError("Chat content must be a string or ordered parts")
        else:
            pass
        pieces: list[str] = []
        for part in content:
            if isinstance(part, str):
                pieces.append(part)
            elif isinstance(part, Image.Image):
                rendered.images.append(part.convert("RGB"))
                pieces.append(IMAGE_PLACEHOLDER)
            elif isinstance(part, np.ndarray):
                rendered.audios.append(part.astype(np.float32, copy=False))
                rendered.audio_turn_indices.append(turn_index)
                pieces.append(AUDIO_PLACEHOLDER)
            elif isinstance(part, dict):
                part_type = part.get("type")
                if part_type == "text":
                    text = part.get("text")
                    if not isinstance(text, str):
                        raise ValueError("Text content parts require a string text")
                    else:
                        pieces.append(text)
                elif part_type in ("image_url", "audio_url", "input_audio"):
                    media = part.get(str(part_type))
                    if part_type == "input_audio":
                        if (
                            not isinstance(media, dict)
                            or not isinstance(media.get("data"), str)
                            or not isinstance(media.get("format"), str)
                        ):
                            raise ValueError(
                                "input_audio requires base64 data and format"
                            )
                        else:
                            source = (
                                f"data:audio/{media['format']};base64,{media['data']}"
                            )
                    elif isinstance(media, str):
                        source = media
                    elif isinstance(media, dict) and isinstance(media.get("url"), str):
                        source = media["url"]
                    else:
                        raise ValueError(f"{part_type} requires a media URL")
                    if part_type == "image_url":
                        images = await ensure_image_list_async([source])
                        if len(images) != 1 or not isinstance(images[0], Image.Image):
                            raise ValueError("Image content did not decode to an image")
                        else:
                            rendered.images.append(images[0])
                            pieces.append(IMAGE_PLACEHOLDER)
                    else:
                        audios = await ensure_audio_list_async(
                            [source], target_sr=16000
                        )
                        if len(audios) != 1 or not isinstance(audios[0], np.ndarray):
                            raise ValueError(
                                "Audio content did not decode to a waveform"
                            )
                        else:
                            rendered.audios.append(audios[0])
                            rendered.audio_turn_indices.append(turn_index)
                            pieces.append(AUDIO_PLACEHOLDER)
                elif part_type == "video_url":
                    media = part.get("video_url")
                    if isinstance(media, str):
                        source = media
                        use_audio = use_audio_in_video
                    elif isinstance(media, dict) and isinstance(media.get("url"), str):
                        source = media["url"]
                        use_audio = bool(media.get("use_audio", use_audio_in_video))
                        if media.get("stack_frames", 1) != 1:
                            raise ValueError(
                                "MiniCPM-o video stack_frames currently supports only 1"
                            )
                        else:
                            pass
                    else:
                        raise ValueError("video_url requires a media URL")
                    video = await load_timed_video(
                        source,
                        use_audio=use_audio,
                        fps=video_fps,
                        max_frames=video_max_frames,
                        min_pixels=video_min_pixels,
                        max_pixels=video_max_pixels,
                        total_pixels=video_total_pixels,
                    )
                    rendered.has_video = True
                    rendered.omni_mode = rendered.omni_mode or bool(
                        video.audio_segments
                    )
                    for frame_index, frame in enumerate(video.frames):
                        rendered.images.append(frame)
                        pieces.append(IMAGE_PLACEHOLDER)
                        if video.audio_segments:
                            rendered.audios.append(video.audio_segments[frame_index])
                            rendered.audio_turn_indices.append(turn_index)
                            pieces.append(AUDIO_PLACEHOLDER)
                        else:
                            pass
                else:
                    raise ValueError(f"Unsupported MiniCPM-o content type: {part_type}")
            else:
                raise ValueError("Unsupported MiniCPM-o content part")
        rendered.messages.append({"role": role, "content": "\n".join(pieces)})
        message_pieces.append(pieces)
    if rendered.omni_mode:
        for message, pieces in zip(rendered.messages, message_pieces):
            message["content"] = "".join(pieces)
    else:
        pass
    return rendered
