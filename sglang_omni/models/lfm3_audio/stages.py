# SPDX-License-Identifier: Apache-2.0
"""Stage executors of the LFM3 pipeline (LFM3-Audio, LFM3-Omni): preprocessing, generation, decode."""

from __future__ import annotations

import asyncio
import base64
import binascii
import logging
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch

from sglang_omni.models.lfm3_audio.io import Lfm3AudioState
from sglang_omni.preprocessing.audio import AudioMediaIO
from sglang_omni.preprocessing.base import is_url
from sglang_omni.preprocessing.image import ensure_image_list_async
from sglang_omni.preprocessing.resource_connector import get_global_resource_connector
from sglang_omni.proto import StagePayload
from sglang_omni.scheduling.pipeline_state import load_state, store_state
from sglang_omni.scheduling.simple_scheduler import SimpleScheduler
from sglang_omni.utils.audio_payload import audio_waveform_payload
from sglang_omni.utils.checkpoint import resolve_checkpoint
from sglang_omni.utils.device import resolve_concrete_device

if TYPE_CHECKING:
    from sglang_omni.scheduling.omni_scheduler import OmniScheduler
else:
    pass

logger = logging.getLogger(__name__)

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


def load_processor(model_path: str):
    from transformers import AutoProcessor

    return AutoProcessor.from_pretrained(
        resolve_checkpoint(model_path), trust_remote_code=True
    )


def tts_system_prompt(tts_params: Mapping[str, object]) -> str:
    """The checkpoint's TTS instruction; `instructions` replaces it verbatim."""
    instructions = tts_params.get("instructions")
    if isinstance(instructions, str) and instructions.strip():
        return instructions.strip()
    else:
        pass
    voice = str(tts_params.get("voice") or DEFAULT_TTS_VOICE).lower()
    if voice in ("", "default"):
        voice = DEFAULT_TTS_VOICE
    elif voice not in TTS_VOICES:
        raise ValueError(f"LFM3-Audio voices are {sorted(TTS_VOICES)}, got {voice!r}")
    else:
        pass
    language = str(tts_params.get("language") or DEFAULT_TTS_LANGUAGE)
    return f"Perform TTS in {language} using a {voice} voice."


def as_messages(inputs: object) -> list[dict]:
    if isinstance(inputs, str):
        return [{"role": "user", "content": inputs}]
    elif isinstance(inputs, list):
        return [dict(message) for message in inputs]
    elif isinstance(inputs, dict) and "messages" in inputs:
        return [dict(message) for message in inputs["messages"]]
    else:
        raise ValueError(
            f"LFM3-Audio cannot read chat inputs of type {type(inputs).__name__}"
        )


def audio_source(part: Mapping[str, object]) -> object:
    kind = part.get("type")
    if kind == "input_audio":
        audio = part["input_audio"]
        return f"data:audio/{audio['format']};base64,{audio['data']}"
    elif kind == "audio_url":
        url = part["audio_url"]
        return url["url"] if isinstance(url, Mapping) else url
    else:
        return part.get("audio")


def media_source(part: Mapping[str, object], kind: str) -> object:
    """The image or video of a content part: `{kind}_url` ({"url": ...} or a string), or `{kind}`."""
    value = part.get(f"{kind}_url", part.get(kind, part.get(f"input_{kind}")))
    if isinstance(value, Mapping):
        return value.get("url")
    else:
        return value


def video_file(source: object) -> object:
    """A video the processor can open: paths and http(s) URLs as given, a data URL as a temp file."""
    if isinstance(source, str) and source.startswith("data:"):
        header, _, encoded = source.partition(",")
        if ";base64" not in header:
            raise ValueError("LFM3 video data URLs must be base64")
        else:
            pass
        try:
            payload = base64.b64decode(encoded, validate=True)
        except binascii.Error as exc:
            raise ValueError("LFM3 video data URL is not valid base64") from exc
        suffix = (
            "." + header[len("data:video/") :].split(";")[0]
            if header.startswith("data:video/")
            else ".mp4"
        )
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as handle:
            handle.write(payload)
        return handle.name
    else:
        return source


def load_video_frames(
    source: object, video_processor: object
) -> tuple[np.ndarray, object]:
    """The frames the checkpoint's video processor samples, decoded with PyAV, and their metadata.

    Only the sampled frames are kept; the metadata carries their indices and the
    clip's fps, from which the processor writes the frame timestamps.
    """
    import av
    from transformers.video_utils import VideoMetadata

    with av.open(str(source)) as container:
        stream = container.streams.video[0]
        fps = float(stream.average_rate or stream.base_rate or 0) or None
        total = stream.frames or sum(1 for _ in container.demux(stream) if _.size)
    if not total:
        raise ValueError(f"LFM3 found no video frames in {source!r}")
    else:
        pass
    indices = video_processor.sample_frames(
        VideoMetadata(total_num_frames=total, fps=fps)
    )
    # The sampled indices are distinct and increasing: one decoding pass keeps them in order.
    wanted = {int(index) for index in indices}
    last = max(wanted)
    decoded: list[np.ndarray] = []
    with av.open(str(source)) as container:
        for index, frame in enumerate(container.decode(video=0)):
            if index in wanted:
                decoded.append(frame.to_ndarray(format="rgb24"))
            else:
                pass
            if index >= last:
                break
            else:
                pass
    if len(decoded) != len(wanted):
        raise ValueError(
            f"LFM3 decoded {len(decoded)} of {len(wanted)} sampled video frames"
        )
    else:
        pass
    metadata = VideoMetadata(
        total_num_frames=total,
        fps=fps,
        frames_indices=[int(index) for index in indices],
    )
    return np.stack(decoded), metadata


async def load_waveform(source: object, media_io: AudioMediaIO) -> np.ndarray:
    if isinstance(source, np.ndarray):
        return source.astype(np.float32)
    elif isinstance(source, str) and is_url(source):
        waveform, _ = await get_global_resource_connector().load_resource_async(
            source, media_io
        )
        return waveform
    elif isinstance(source, (str, Path)):
        waveform, _ = media_io.load_file(Path(source))
        return waveform
    else:
        raise ValueError(f"Unsupported LFM3-Audio audio input {type(source).__name__}")


def audio_sampling(params: Mapping[str, object], is_tts: bool) -> dict[str, object]:
    """Speech-frame sampling: a TTS request's sampling fields, or a chat request's `talker_*` fields."""
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
        self.images: list[object] = []
        self.videos: list[np.ndarray] = []
        self.video_metadata: list[object] = []
        # Decoded video data URLs, removed once the processor has read them.
        self.temp_files: list[str] = []

    def cleanup(self) -> None:
        for path in self.temp_files:
            try:
                os.unlink(path)
            except FileNotFoundError:
                pass
        self.temp_files.clear()

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
        self.media_io = AudioMediaIO(target_sr=INPUT_SAMPLE_RATE)
        self.has_vision = getattr(self.processor, "image_processor", None) is not None

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
                    parts = []
                    for part in content:
                        parts.append(await self.read_part(dict(part), media))
                    message["content"] = parts
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
                    last = messages[-1]["content"]
                    messages[-1]["content"] = (
                        [{"type": "text", "text": last}]
                        if isinstance(last, str)
                        else list(last)
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
        # LFM3-Omni's template already opens the answer with an empty reasoning block.
        if marker == NO_THINKING and rendered.endswith(NO_THINKING):
            marker = ""
        else:
            pass
        try:
            encoded = self.processor(
                text=rendered + marker,
                sampling_rate=INPUT_SAMPLE_RATE,
                return_tensors="pt",
                **media.processor_inputs(),
            )
        finally:
            media.cleanup()
        state = Lfm3AudioState(
            input_ids=encoded["input_ids"][0].tolist(),
            input_features=encoded.get("input_features"),
            feature_attention_mask=encoded.get("feature_attention_mask"),
            pixel_values=encoded.get("pixel_values"),
            image_grid_thw=encoded.get("image_grid_thw"),
            pixel_values_videos=encoded.get("pixel_values_videos"),
            video_grid_thw=encoded.get("video_grid_thw"),
            text_only=text_only,
            max_new_tokens=int(params.get("max_new_tokens") or 2048),
            temperature=0.0 if is_tts else float(params.get("temperature") or 0.0),
            top_p=1.0 if is_tts else float(params.get("top_p") or 1.0),
            top_k=-1 if is_tts else int(params.get("top_k") or -1),
            seed=params.get("seed"),
            **audio_sampling(params, is_tts),
        )
        return store_state(payload, state)

    async def read_part(self, part: dict, media: "Media") -> dict:
        """Load a content part's audio, image or video; the template gets a bare placeholder part."""
        kind = part.get("type")
        if kind in AUDIO_PART_TYPES:
            media.audio.append(await load_waveform(audio_source(part), self.media_io))
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
                    await ensure_image_list_async(media_source(part, "image"))
                )
                return {"type": "image"}
            else:
                source = media_source(part, "video")
                path = video_file(source)
                if path is not source:
                    media.temp_files.append(path)
                else:
                    pass
                frames, metadata = await asyncio.to_thread(
                    load_video_frames, path, self.processor.video_processor
                )
                media.videos.append(frames)
                media.video_metadata.append(metadata)
                return {"type": "video"}
        else:
            return part


def create_preprocessing_executor(model_path: str) -> SimpleScheduler:
    return SimpleScheduler(Lfm3AudioPreprocessor(model_path))


def create_generation_executor(
    model_path: str,
    *,
    device: str | None = None,
    gpu_id: int | None = None,
    server_args_overrides: Mapping[str, object] | None = None,
) -> OmniScheduler:
    from sglang_omni.models.lfm3_audio.engine_builder import Lfm3AudioEngineBuilder

    return Lfm3AudioEngineBuilder().build(
        model_path,
        device=device,
        gpu_id=gpu_id,
        server_args_overrides=server_args_overrides,
    )


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
        data: dict[str, object] = state.to_dict()
        data["text"] = text
        codes = state.audio_codes
        if codes is not None and codes.numel():
            waveform = self.codec.decode(
                codes.T[None].to(self.device, dtype=torch.long)
            ).audio_values[0]
            data.update(
                audio_waveform_payload(waveform, sample_rate=OUTPUT_SAMPLE_RATE)
            )
        else:
            pass
        payload.data = data
        return payload


def create_decode_executor(
    model_path: str, *, device: str | None = None, gpu_id: int | None = None
) -> SimpleScheduler:
    decoder = Lfm3AudioDecoder(model_path, str(resolve_concrete_device(device, gpu_id)))
    return SimpleScheduler(decoder)
