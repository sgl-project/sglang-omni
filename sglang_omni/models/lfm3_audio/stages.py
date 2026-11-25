# SPDX-License-Identifier: Apache-2.0
"""Preprocessing and decode stages of the LFM3 pipeline (LFM3-Audio and LFM3-Omni)."""

from __future__ import annotations

import asyncio
import math
from collections.abc import Mapping
from dataclasses import dataclass

import torch
from transformers import AutoProcessor, ProcessorMixin

from sglang_omni.models.lfm3_audio.media import (
    REQUEST_ERROR_PREFIX,
    Media,
    load_video_bytes,
    load_video_frames,
    part_source,
)
from sglang_omni.models.lfm3_audio.payload_types import (
    DEFAULT_MAX_NEW_TOKENS,
    Lfm3AudioState,
)
from sglang_omni.preprocessing.audio import ensure_audio_list_async
from sglang_omni.preprocessing.image import ensure_image_list_async
from sglang_omni.proto import OmniRequest, StagePayload
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
# Undecodable bytes, a bad data URL or a path outside the media policy: the request's fault.
MEDIA_READ_ERRORS = (OSError, ValueError, RuntimeError)
# Extra media lists a chat request may carry beside its messages, appended to the last message.
EXTRA_MEDIA_KEYS = (
    ("audio", ("audio", "audios")),
    ("image", ("image", "images")),
    ("video", ("video", "videos")),
)


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
        raise ValueError(
            f"{REQUEST_ERROR_PREFIX} voices are {sorted(TTS_VOICES)}, got {voice!r}"
        )
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
        raise ValueError(
            f"{REQUEST_ERROR_PREFIX} send chat messages or a text prompt, not token ids"
        )
    else:
        return [dict(message) for message in messages]


def is_finite_number(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


@dataclass(frozen=True, kw_only=True)
class SpeechSamplingParams:
    """The request's speech-frame sampling fields; None takes the checkpoint's default."""

    temperature: float | None
    top_k: int | None
    repetition_penalty: float | None


def speech_sampling_params(
    params: Mapping[str, object], is_tts: bool
) -> SpeechSamplingParams:
    """A TTS request's sampling fields, or a chat request's talker_ fields, checked before they reach the batch."""
    prefix = "" if is_tts else "talker_"
    temperature = params.get(f"{prefix}temperature")
    top_k = params.get(f"{prefix}top_k")
    repetition_penalty = params.get(f"{prefix}repetition_penalty")
    # note (ruilin): the frame sampler runs on the whole batch, so a value it rejects would fail every request beside this one.
    if temperature is not None and not (
        is_finite_number(temperature) and temperature >= 0
    ):
        raise ValueError(
            f"{REQUEST_ERROR_PREFIX} {prefix}temperature must be a number >= 0"
        )
    elif top_k is not None and not (
        isinstance(top_k, int) and not isinstance(top_k, bool) and top_k >= 0
    ):
        raise ValueError(
            f"{REQUEST_ERROR_PREFIX} {prefix}top_k must be an integer >= 0"
        )
    elif repetition_penalty is not None and not (
        is_finite_number(repetition_penalty) and repetition_penalty > 0
    ):
        raise ValueError(
            f"{REQUEST_ERROR_PREFIX} {prefix}repetition_penalty must be a number > 0"
        )
    else:
        return SpeechSamplingParams(
            temperature=temperature,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
        )


def text_sampling_params(params: Mapping[str, object]) -> dict[str, object]:
    """A chat request's text sampling fields, checked against the ranges the engine accepts."""
    checks = {
        "temperature": lambda value: is_finite_number(value) and value >= 0,
        "top_p": lambda value: is_finite_number(value) and 0 < value <= 1,
        "top_k": lambda value: isinstance(value, int)
        and not isinstance(value, bool)
        and value >= -1,
        "min_p": lambda value: is_finite_number(value) and 0 <= value <= 1,
        "repetition_penalty": lambda value: is_finite_number(value) and 0 < value <= 2,
    }
    sampling = {}
    for name, in_range in checks.items():
        value = params.get(name)
        if value is None:
            continue
        elif not in_range(value):
            raise ValueError(f"{REQUEST_ERROR_PREFIX} {name}={value!r} is out of range")
        else:
            sampling[name] = value
    return sampling


def request_seed(params: Mapping[str, object]) -> int | None:
    seed = params.get("seed")
    if seed is None or (
        isinstance(seed, int) and not isinstance(seed, bool) and 0 <= seed < 2**63
    ):
        return seed
    else:
        raise ValueError(
            f"{REQUEST_ERROR_PREFIX} seed must be an integer in [0, 2**63)"
        )


def tts_messages(
    request: OmniRequest, tts_params: Mapping[str, object]
) -> list[dict[str, object]]:
    inputs = request.inputs
    text = inputs if isinstance(inputs, str) else (inputs or {}).get("text", "")
    return [
        {"role": "system", "content": tts_system_prompt(tts_params)},
        {"role": "user", "content": text},
    ]


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
        is_tts = metadata.get("task") == "tts"
        if (
            params.get("stop")
            or params.get("stop_token_ids")
            or params.get("epsilon_cutoff")
        ):
            raise ValueError(
                f"{REQUEST_ERROR_PREFIX} stop sequences, stop_token_ids and epsilon_cutoff are not supported"
            )
        else:
            pass
        speech_sampling = speech_sampling_params(params, is_tts)
        media = Media()
        if is_tts:
            messages = tts_messages(request, metadata.get("tts_params") or {})
            # note (ruilin): TTS answers in speech only; its text stays greedy.
            marker, text_only, text_sampling = AUDIO_START, False, {}
        else:
            messages = await self.chat_messages(request, media)
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
            text_sampling = text_sampling_params(params)
        encoded = self.processor(
            text=self.render_prompt(messages, marker),
            sampling_rate=INPUT_SAMPLE_RATE,
            return_tensors="pt",
            **media.processor_inputs(),
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
            max_new_tokens=params.get("max_new_tokens") or DEFAULT_MAX_NEW_TOKENS,
            seed=request_seed(params),
            audio_temperature=speech_sampling.temperature,
            audio_top_k=speech_sampling.top_k,
            audio_repetition_penalty=speech_sampling.repetition_penalty,
            **text_sampling,
        )
        return store_state(payload, state)

    def render_prompt(self, messages: list[dict[str, object]], marker: str) -> str:
        """The chat template's prompt, opened with the answer's mode marker."""
        rendered = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        # note (ruilin): LFM3-Omni's template already opens the answer with an empty reasoning block.
        if marker == NO_THINKING and rendered.endswith(NO_THINKING):
            return rendered
        else:
            return rendered + marker

    async def chat_messages(
        self, request: OmniRequest, media: Media
    ) -> list[dict[str, object]]:
        """The request's messages with every media part loaded into media and replaced by its placeholder."""
        messages = as_messages(request.inputs)
        for message in messages:
            content = message.get("content")
            if isinstance(content, list):
                if not all(isinstance(part, Mapping) for part in content):
                    raise ValueError(
                        f"{REQUEST_ERROR_PREFIX} message content parts must be objects"
                    )
                else:
                    pass
                message["content"] = [
                    await self.read_part(dict(part), media) for part in content
                ]
            else:
                pass
        extra = request.inputs if isinstance(request.inputs, dict) else {}
        for kind, keys in EXTRA_MEDIA_KEYS:
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
        return messages

    async def read_part(
        self, part: dict[str, object], media: Media
    ) -> dict[str, object]:
        """Load a content part's audio, image or video; the template gets a bare placeholder part."""
        kind = part.get("type")
        if kind in AUDIO_PART_TYPES:
            media_kind = "audio"
        elif kind in IMAGE_PART_TYPES:
            media_kind = "image"
        elif kind in VIDEO_PART_TYPES:
            media_kind = "video"
        else:
            return part
        if media_kind != "audio" and not self.has_vision:
            raise ValueError(
                f"{REQUEST_ERROR_PREFIX} this checkpoint takes text and audio; images and videos need LFM3-Omni"
            )
        else:
            pass
        source = part_source(part, media_kind)
        try:
            await self.load_media(media_kind, source, media)
        except MEDIA_READ_ERRORS as error:
            if str(error).startswith(REQUEST_ERROR_PREFIX):
                raise
            else:
                raise ValueError(
                    f"{REQUEST_ERROR_PREFIX} could not read the {media_kind} input: {error}"
                ) from error
        return {"type": media_kind}

    async def load_media(self, media_kind: str, source: object, media: Media) -> None:
        if media_kind == "audio":
            media.audio.extend(
                await ensure_audio_list_async(source, target_sr=INPUT_SAMPLE_RATE)
            )
        elif media_kind == "image":
            media.images.extend(await ensure_image_list_async(source))
        else:
            video_bytes = await load_video_bytes(source)
            frames, metadata = await asyncio.to_thread(
                load_video_frames, video_bytes, self.processor.video_processor
            )
            media.videos.append(frames)
            media.video_metadata.append(metadata)


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
