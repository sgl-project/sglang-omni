# SPDX-License-Identifier: Apache-2.0
"""Preprocessing and decode stages of the LFM3 pipeline (LFM3-Audio and LFM3-Omni)."""

from __future__ import annotations

import asyncio
import base64
import io
from collections.abc import Mapping
from pathlib import Path

import av
import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, ProcessorMixin
from transformers.video_processing_utils import BaseVideoProcessor
from transformers.video_utils import VideoMetadata

from sglang_omni.models.lfm3_audio.io import DEFAULT_MAX_NEW_TOKENS, Lfm3AudioState
from sglang_omni.preprocessing.audio import ensure_audio_list_async
from sglang_omni.preprocessing.base import MediaIO, is_url
from sglang_omni.preprocessing.image import ensure_image_list_async
from sglang_omni.preprocessing.resource_connector import get_global_resource_connector
from sglang_omni.proto import StagePayload
from sglang_omni.scheduling.pipeline_state import load_state, store_state
from sglang_omni.scheduling.simple_scheduler import SimpleScheduler
from sglang_omni.utils.audio_payload import audio_waveform_payload
from sglang_omni.utils.checkpoint import resolve_checkpoint
from sglang_omni.utils.device import resolve_concrete_device

INPUT_SAMPLE_RATE = 16000
OUTPUT_SAMPLE_RATE = 24000
AUDIO_START = "<|audio_start|>"
MIXED_START = "<|mixed_start|>"
# An empty reasoning block: text answers come without the model's thinking.
NO_THINKING = "<think></think>"
DEFAULT_TTS_LANGUAGE = "American English"
DEFAULT_TTS_VOICE = "female"
TTS_VOICES = frozenset({"female", "male"})
INTERLEAVED_SYSTEM_PROMPT = "Respond with interleaved text and audio."
AUDIO_PART_TYPES = frozenset({"audio", "audio_url", "input_audio"})
IMAGE_PART_TYPES = frozenset({"image", "image_url", "input_image"})
VIDEO_PART_TYPES = frozenset({"video", "video_url"})


def load_processor(model_path: str) -> ProcessorMixin:
    return AutoProcessor.from_pretrained(
        resolve_checkpoint(model_path), trust_remote_code=True
    )


def tts_system_prompt(tts_params: Mapping[str, object]) -> str:
    """The checkpoint's TTS instruction; the request's instructions replace it verbatim."""
    instructions = tts_params.get("instructions")
    if isinstance(instructions, str) and instructions.strip():
        return instructions.strip()
    else:
        pass
    voice = str(tts_params.get("voice") or DEFAULT_TTS_VOICE).lower()
    if voice in ("", "default"):
        voice = DEFAULT_TTS_VOICE
    elif voice not in TTS_VOICES:
        raise ValueError(f"LFM3 voices are {sorted(TTS_VOICES)}, got {voice!r}")
    else:
        pass
    language = str(tts_params.get("language") or DEFAULT_TTS_LANGUAGE)
    return f"Perform TTS in {language} using a {voice} voice."


def as_messages(inputs: object) -> list[dict[str, object]]:
    """Chat messages from a prompt string, a message list, or a dict holding messages and media lists."""
    if isinstance(inputs, str):
        return [{"role": "user", "content": inputs}]
    elif isinstance(inputs, dict) and "messages" in inputs:
        messages = inputs["messages"]
    else:
        messages = inputs
    if not isinstance(messages, list) or not all(
        isinstance(message, Mapping) for message in messages
    ):
        raise ValueError("LFM3 takes chat messages or a text prompt, not token ids")
    else:
        return [dict(message) for message in messages]


def part_source(part: Mapping[str, object], kind: str) -> object:
    """The media of a content part: input_audio data, else the kind_url field (an object with a url, or a string), else the kind field."""
    if kind == "audio" and part.get("type") == "input_audio":
        audio = part["input_audio"]
        if not isinstance(audio, Mapping) or "data" not in audio:
            raise ValueError("input_audio needs a data field")
        else:
            pass
        return f"data:audio/{audio.get('format', 'wav')};base64,{audio['data']}"
    else:
        value = part.get(f"{kind}_url", part.get(kind, part.get(f"input_{kind}")))
        source = value.get("url") if isinstance(value, Mapping) else value
        if source is None:
            raise ValueError(f"A {part.get('type')} content part needs its {kind}")
        else:
            return source


class VideoBytesIO(MediaIO[bytes]):
    """A video's encoded bytes, read through the resource connector's media policy."""

    def load_bytes(self, data: bytes) -> bytes:
        return data

    def load_base64(self, media_type: str, data: str) -> bytes:
        del media_type
        return base64.b64decode(data, validate=True)

    def load_file(self, filepath: Path) -> bytes:
        return filepath.read_bytes()


async def load_video_bytes(source: object) -> bytes:
    """A video URL (http, data or file) or local path, under the server's media policy."""
    connector = get_global_resource_connector()
    if isinstance(source, str) and is_url(source):
        return await connector.load_resource_async(source, VideoBytesIO())
    elif isinstance(source, (str, Path)):
        return connector.load_local_path(source, VideoBytesIO())
    else:
        raise ValueError(f"Unsupported LFM3 video input {type(source).__name__}")


def load_video_frames(
    video_bytes: bytes, video_processor: BaseVideoProcessor
) -> tuple[np.ndarray, VideoMetadata]:
    """The frames the checkpoint's video processor samples, decoded with PyAV, and their metadata.

    Only the sampled frames are kept; the metadata carries their indices and the clip's fps, from which the processor writes the frame timestamps.
    """
    with av.open(io.BytesIO(video_bytes)) as container:
        stream = container.streams.video[0]
        fps = float(stream.average_rate or stream.base_rate or 0) or None
        total_frames = stream.frames or sum(
            1 for packet in container.demux(stream) if packet.size
        )
    if not total_frames:
        raise ValueError("LFM3 found no frames in the video")
    else:
        pass
    frame_indices = [
        int(index)
        for index in video_processor.sample_frames(
            VideoMetadata(total_num_frames=total_frames, fps=fps)
        )
    ]
    # note (ruilin): the sampled indices are distinct and increasing, so one pass keeps only them, in order.
    wanted_indices = set(frame_indices)
    decoded_frames: list[np.ndarray] = []
    with av.open(io.BytesIO(video_bytes)) as container:
        for index, frame in enumerate(container.decode(video=0)):
            if index in wanted_indices:
                decoded_frames.append(frame.to_ndarray(format="rgb24"))
            else:
                pass
            if index >= frame_indices[-1]:
                break
            else:
                pass
    if len(decoded_frames) != len(frame_indices):
        raise ValueError(
            f"LFM3 decoded {len(decoded_frames)} of {len(frame_indices)} sampled video frames"
        )
    else:
        pass
    metadata = VideoMetadata(
        total_num_frames=total_frames, fps=fps, frames_indices=frame_indices
    )
    return np.stack(decoded_frames), metadata


def audio_sampling(params: Mapping[str, object], is_tts: bool) -> dict[str, object]:
    """Speech-frame sampling: a TTS request's sampling fields, or a chat request's talker_ fields."""
    prefix = "" if is_tts else "talker_"
    return {
        "audio_temperature": params.get(f"{prefix}temperature"),
        "audio_top_k": params.get(f"{prefix}top_k"),
        "audio_repetition_penalty": params.get(f"{prefix}repetition_penalty"),
    }


class Media:
    """A request's input clips (16 kHz), images and videos, in placeholder order."""

    def __init__(self) -> None:
        self.audio: list[np.ndarray] = []
        self.images: list[Image.Image] = []
        self.videos: list[np.ndarray] = []
        self.video_metadata: list[VideoMetadata] = []

    def processor_inputs(self) -> dict[str, object]:
        inputs: dict[str, object] = {}
        if self.audio:
            inputs["audio"] = self.audio
        else:
            pass
        if self.images:
            inputs["images"] = self.images
        else:
            pass
        if self.videos:
            # Already sampled by load_video_frames, with their indices in the metadata.
            inputs.update(
                videos=self.videos,
                video_metadata=self.video_metadata,
                do_sample_frames=False,
            )
        else:
            pass
        return inputs


class Lfm3AudioPreprocessor:
    """Builds the prompt ids, the input clips' log-mel features and LFM3-Omni's image and video patches."""

    def __init__(self, model_path: str) -> None:
        self.processor = load_processor(model_path)
        self.tokenizer = self.processor.tokenizer
        # note (ruilin): only LFM3-Omni's processor has image and video processors.
        self.has_vision = "image_processor" in self.processor.attributes

    async def __call__(self, payload: StagePayload) -> StagePayload:
        request = payload.request
        params = request.params or {}
        metadata = request.metadata or {}
        tts_params = metadata.get("tts_params") or {}
        is_tts = metadata.get("task") == "tts"
        media = Media()
        if is_tts:
            inputs = request.inputs
            text = inputs if isinstance(inputs, str) else (inputs or {}).get("text", "")
            messages = [
                {"role": "system", "content": tts_system_prompt(tts_params)},
                {"role": "user", "content": text},
            ]
            marker = AUDIO_START
            text_only = False
        else:
            messages = as_messages(request.inputs)
            for message in messages:
                content = message.get("content")
                if isinstance(content, list):
                    if not all(isinstance(part, Mapping) for part in content):
                        raise ValueError("LFM3 message content parts must be objects")
                    else:
                        pass
                    message["content"] = [
                        await self.read_part(dict(part), media) for part in content
                    ]
                else:
                    pass
            extra = request.inputs if isinstance(request.inputs, dict) else {}
            for kind, keys in (
                ("audio", ("audio", "audios")),
                ("image", ("image", "images")),
                ("video", ("video", "videos")),
            ):
                sources = next((extra[key] for key in keys if extra.get(key)), None)
                if sources is None:
                    continue
                else:
                    pass
                for source in sources if isinstance(sources, list) else [sources]:
                    part = await self.read_part({"type": kind, kind: source}, media)
                    last_content = messages[-1].get("content") or []
                    messages[-1]["content"] = (
                        [{"type": "text", "text": last_content}]
                        if isinstance(last_content, str)
                        else list(last_content)
                    ) + [part]
            speaks = "audio" in (metadata.get("output_modalities") or [])
            if speaks and not any(
                message.get("role") == "system" for message in messages
            ):
                messages.insert(
                    0, {"role": "system", "content": INTERLEAVED_SYSTEM_PROMPT}
                )
            else:
                pass
            marker = MIXED_START if speaks else NO_THINKING
            text_only = not speaks
        rendered = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        # note (ruilin): LFM3-Omni's template already opens the answer with an empty reasoning block.
        if marker == NO_THINKING and rendered.endswith(NO_THINKING):
            marker = ""
        else:
            pass
        encoded = self.processor(
            text=rendered + marker,
            sampling_rate=INPUT_SAMPLE_RATE,
            return_tensors="pt",
            **media.processor_inputs(),
        )
        if (
            params.get("stop")
            or params.get("stop_token_ids")
            or params.get("epsilon_cutoff")
        ):
            raise ValueError(
                "LFM3 does not support stop sequences, stop_token_ids or epsilon_cutoff"
            )
        else:
            pass
        text_sampling = (
            {}
            if is_tts
            else {
                name: params[name]
                for name in (
                    "temperature",
                    "top_p",
                    "top_k",
                    "min_p",
                    "repetition_penalty",
                )
                if params.get(name) is not None
            }
        )
        state = Lfm3AudioState(
            input_ids=encoded["input_ids"][0].tolist(),
            input_features=encoded.get("input_features"),
            feature_attention_mask=encoded.get("feature_attention_mask"),
            pixel_values=encoded.get("pixel_values"),
            image_grid_thw=encoded.get("image_grid_thw"),
            pixel_values_videos=encoded.get("pixel_values_videos"),
            video_grid_thw=encoded.get("video_grid_thw"),
            text_only=text_only,
            max_new_tokens=int(params.get("max_new_tokens") or DEFAULT_MAX_NEW_TOKENS),
            seed=params.get("seed"),
            **text_sampling,
            **audio_sampling(params, is_tts),
        )
        return store_state(payload, state)

    async def read_part(
        self, part: dict[str, object], media: Media
    ) -> dict[str, object]:
        """Load a content part's audio, image or video; the template gets a bare placeholder part."""
        kind = part.get("type")
        if kind in AUDIO_PART_TYPES:
            media.audio.extend(
                await ensure_audio_list_async(
                    part_source(part, "audio"), target_sr=INPUT_SAMPLE_RATE
                )
            )
            return {"type": "audio"}
        elif kind in IMAGE_PART_TYPES or kind in VIDEO_PART_TYPES:
            if not self.has_vision:
                raise ValueError(
                    "This LFM3 checkpoint takes text and audio; images and videos need LFM3-Omni"
                )
            else:
                pass
            if kind in IMAGE_PART_TYPES:
                media.images.extend(
                    await ensure_image_list_async(part_source(part, "image"))
                )
                return {"type": "image"}
            else:
                video_bytes = await load_video_bytes(part_source(part, "video"))
                frames, metadata = await asyncio.to_thread(
                    load_video_frames, video_bytes, self.processor.video_processor
                )
                media.videos.append(frames)
                media.video_metadata.append(metadata)
                return {"type": "video"}
        else:
            return part


def create_preprocessing_executor(model_path: str) -> SimpleScheduler:
    return SimpleScheduler(Lfm3AudioPreprocessor(model_path))


class Lfm3AudioDecoder:
    """Detokenizes the text and decodes the speech frames with LiquidCodec (24 kHz)."""

    def __init__(self, model_path: str, device: str) -> None:
        processor = load_processor(model_path)
        self.tokenizer = processor.tokenizer
        self.codec = processor.audio_tokenizer.to(device=device).eval()
        self.device = device

    @torch.inference_mode()
    def __call__(self, payload: StagePayload) -> StagePayload:
        state = load_state(payload, Lfm3AudioState)
        text = self.tokenizer.decode(
            state.generated_ids, skip_special_tokens=True
        ).strip()
        output_data = state.to_dict()
        output_data["text"] = text
        if state.audio_codes is not None:
            waveform = self.codec.decode(
                state.audio_codes.T[None].to(self.device, dtype=torch.long)
            ).audio_values[0]
            output_data.update(
                audio_waveform_payload(waveform, sample_rate=OUTPUT_SAMPLE_RATE)
            )
        else:
            pass
        payload.data = output_data
        return payload


def create_decode_executor(
    model_path: str, *, device: str | None = None, gpu_id: int | None = None
) -> SimpleScheduler:
    decoder = Lfm3AudioDecoder(model_path, str(resolve_concrete_device(device, gpu_id)))
    return SimpleScheduler(decoder)
