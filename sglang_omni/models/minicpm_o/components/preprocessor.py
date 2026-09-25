# SPDX-License-Identifier: Apache-2.0
"""Render MiniCPM-o prompts and extract media features and placeholder bounds."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
import torch
from PIL import Image
from transformers import AutoProcessor, AutoTokenizer

from sglang_omni.models.minicpm_o.payload_types import MiniCPMOPipelineState
from sglang_omni.models.minicpm_o.routing import should_generate_audio_output
from sglang_omni.models.weight_loader import resolve_model_path
from sglang_omni.preprocessing.audio import (
    AudioMediaIO,
    compute_audio_cache_key,
    ensure_audio_list_async,
)
from sglang_omni.preprocessing.image import (
    compute_image_cache_key,
    ensure_image_list_async,
)
from sglang_omni.preprocessing.video import (
    compute_video_cache_key,
    ensure_video_list_async,
)
from sglang_omni.proto import StagePayload

if TYPE_CHECKING:
    from transformers import ProcessorMixin
else:
    pass

IMAGE_PLACEHOLDER = "<image>./</image>"
AUDIO_PLACEHOLDER = "<audio>./</audio>"

# note (MayDomine): task prompts match the checkpoint's audio-understanding template.
ASR_PROMPT_ZH = "请仔细听这段音频片段，并将其内容逐字记录。"
ASR_PROMPT_EN = (
    "Please listen to the audio snippet carefully and transcribe the content."
)


def first_batch_item(value: object) -> object:
    """Unwrap the batch dimension of a processor output (batch size is 1)."""
    if isinstance(value, list):
        return value[0] if value else None
    else:
        pass
    if isinstance(value, torch.Tensor):
        return value[0]
    else:
        pass
    return value


def video_to_images(video: object) -> list[Image.Image]:
    """Convert one decoded video (T, C, H, W) tensor to RGB frames."""
    if isinstance(video, list) and all(
        isinstance(frame, Image.Image) for frame in video
    ):
        return [frame.convert("RGB") for frame in video]
    else:
        pass

    frames = video if isinstance(video, torch.Tensor) else torch.as_tensor(video)
    if frames.ndim != 4:
        raise ValueError(
            "MiniCPM-o video inputs must have shape (T, C, H, W), "
            f"got {tuple(frames.shape)}"
        )
    else:
        pass
    if frames.shape[1] in (1, 3, 4):
        frames = frames.permute(0, 2, 3, 1)
    elif frames.shape[-1] not in (1, 3, 4):
        raise ValueError(
            "MiniCPM-o video frames must have 1, 3, or 4 channels, "
            f"got {tuple(frames.shape)}"
        )
    else:
        pass

    frames = frames.detach().cpu()
    if frames.is_floating_point() and frames.numel() and float(frames.max()) <= 1.0:
        frames = frames * 255.0
    else:
        pass
    frames = frames.clamp(0, 255).to(torch.uint8)
    return [Image.fromarray(frame.numpy()).convert("RGB") for frame in frames]


class MiniCPMOPreprocessor:
    def __init__(
        self,
        model_path: str,
        *,
        speech_enabled: bool = False,
    ) -> None:
        local_dir = str(resolve_model_path(model_path))
        self.tokenizer = AutoTokenizer.from_pretrained(
            local_dir, trust_remote_code=True
        )
        # note (MayDomine): text-only requests do not need Whisper feature extraction.
        self.model_dir = local_dir
        self._processor = None  # noqa: leading-underscore
        self.speech_enabled = speech_enabled

    def speech_to_text_inputs(
        self, payload: StagePayload, inputs: Mapping[str, object]
    ) -> tuple[list[dict[str, str]], list[npt.NDArray[np.float32]]]:
        """Turn a transcription upload into a chat turn plus audio list."""
        params = payload.request.params or {}
        language = str(params.get("language") or "").lower()
        prompt = ASR_PROMPT_ZH if language.startswith("zh") else ASR_PROMPT_EN
        audio, _ = AudioMediaIO(target_sr=16000).load_bytes(inputs["audio_bytes"])
        return [{"role": "user", "content": prompt}], [audio]

    def should_use_tts_template(self, payload: StagePayload) -> bool:
        return self.speech_enabled and should_generate_audio_output(payload)

    @property
    def processor(self) -> ProcessorMixin:
        if self._processor is None:  # noqa: leading-underscore
            self._processor = AutoProcessor.from_pretrained(  # noqa: leading-underscore
                self.model_dir, trust_remote_code=True
            )
        else:
            pass
        return self._processor  # noqa: leading-underscore

    async def __call__(self, payload: StagePayload) -> StagePayload:
        inputs = payload.request.inputs
        raw_images = None
        raw_audios = None
        raw_videos = None
        use_audio_in_video = False
        video_params: dict[str, object] = {}
        if isinstance(inputs, dict) and inputs.get("audio_bytes") is not None:
            messages, raw_audios = self.speech_to_text_inputs(payload, inputs)
        elif isinstance(inputs, dict):
            messages = inputs.get("messages", [])
            raw_images = inputs.get("images")
            raw_audios = inputs.get("audio") or inputs.get("audios")
            raw_videos = inputs.get("videos") or inputs.get("video")
            use_audio_in_video = bool(inputs.get("use_audio_in_video", False))
            video_params = {
                key: inputs.get(key)
                for key in (
                    "video_fps",
                    "video_max_frames",
                    "video_min_pixels",
                    "video_max_pixels",
                    "video_total_pixels",
                )
                if inputs.get(key) is not None
            }
        else:
            messages = inputs

        if raw_images or raw_audios or raw_videos:
            return await self.preprocess_multimodal(
                payload,
                messages,
                raw_images=raw_images,
                raw_audios=raw_audios,
                raw_videos=raw_videos,
                use_audio_in_video=use_audio_in_video,
                video_params=video_params,
            )
        else:
            pass

        if (
            isinstance(messages, list)
            and messages
            and all(isinstance(token, int) for token in messages)
        ):
            # note (MayDomine): rollout prompt ids must match the caller's exactly.
            prompt_text = ""
            input_ids = torch.tensor(messages, dtype=torch.long)
        else:
            prompt_text = self.render_chat_template(
                messages, use_tts_template=self.should_use_tts_template(payload)
            )
            encoded = self.tokenizer(prompt_text, return_tensors="pt")
            input_ids = encoded["input_ids"][0].to(dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)

        state = MiniCPMOPipelineState(
            prompt={
                "prompt_text": prompt_text,
                "input_ids": input_ids,
                "attention_mask": attention_mask,
            },
            stream_state={"token_ids": [], "text": ""},
        )
        payload.data = state.to_dict()
        payload.request.inputs = None
        return payload

    def render_chat_template(
        self, messages: object, *, use_tts_template: bool = False
    ) -> str:
        if isinstance(messages, str):
            return messages
        else:
            pass
        messages = self.normalize_message_contents(messages)
        return self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
            use_tts_template=use_tts_template,
            enable_thinking=False,
        )

    @staticmethod
    def normalize_message_contents(messages: object) -> object:
        """Convert OpenAI text-part content to the string form expected by MiniCPM."""
        if not isinstance(messages, list):
            return messages
        else:
            pass
        normalized = []
        for message in messages:
            if not isinstance(message, dict):
                normalized.append(message)
                continue
            else:
                pass
            content = message.get("content", "")
            if isinstance(content, list):
                content = "".join(
                    str(part.get("text", ""))
                    for part in content
                    if isinstance(part, dict) and part.get("type") == "text"
                )
            else:
                pass
            normalized.append({**message, "content": content})
        return normalized

    def messages_with_media_placeholders(
        self,
        messages: Sequence[Mapping[str, object]],
        *,
        num_images: int,
        num_audios: int,
    ) -> list[Mapping[str, object]]:
        """Prepend media placeholders to the last user message."""
        result: list[Mapping[str, object]] = []
        messages = self.normalize_message_contents(messages)
        for i, msg in enumerate(messages):
            if i == len(messages) - 1 and msg.get("role", "user") == "user":
                parts = (
                    [IMAGE_PLACEHOLDER] * num_images
                    + [AUDIO_PLACEHOLDER] * num_audios
                    + [str(msg.get("content", ""))]
                )
                result.append({**msg, "content": "\n".join(parts)})
            else:
                result.append(msg)
        return result

    async def preprocess_multimodal(
        self,
        payload: StagePayload,
        messages: object,
        *,
        raw_images: object,
        raw_audios: object,
        raw_videos: object,
        use_audio_in_video: bool,
        video_params: Mapping[str, object],
    ) -> StagePayload:
        video_kwargs = {
            key.removeprefix("video_"): value for key, value in video_params.items()
        }
        image_cache_key = compute_image_cache_key(raw_images)
        video_cache_key = (
            compute_video_cache_key(raw_videos, **video_kwargs) if raw_videos else None
        )

        images = await ensure_image_list_async(raw_images)
        if raw_videos:
            videos, _, video_audios = await ensure_video_list_async(
                raw_videos,
                **video_kwargs,
                extract_audio=use_audio_in_video,
                audio_target_sr=16000,
            )
        else:
            videos, video_audios = [], None
        video_images = [frame for video in videos for frame in video_to_images(video)]
        images.extend(video_images)
        audios = await ensure_audio_list_async(raw_audios, target_sr=16000)
        if video_audios:
            audios.extend(audio for audio in video_audios if audio is not None)
        else:
            pass
        audio_cache_key = compute_audio_cache_key(audios)

        cache_keys = [key for key in (image_cache_key, video_cache_key) if key]
        image_cache_key = "|".join(cache_keys) if cache_keys else None

        if isinstance(messages, list) and not (
            messages and all(isinstance(token, int) for token in messages)
        ):
            messages = self.messages_with_media_placeholders(
                messages, num_images=len(images), num_audios=len(audios)
            )
        else:
            pass
        prompt_text = self.render_chat_template(
            messages,
            use_tts_template=bool(audios) or self.should_use_tts_template(payload),
        )

        # Match the checkpoint's video recipe; the policy covers mixed images too.
        video_options = (
            {"max_slice_nums": 1, "use_image_id": False} if raw_videos else {}
        )
        processed = self.processor(
            prompt_text,
            images=[images] if images else None,
            audios=[audios] if audios else None,
            return_tensors="pt",
            **video_options,
        )

        input_ids = processed["input_ids"][0].to(dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)

        mm_inputs: dict[str, object] = {}
        encoder_inputs: dict[str, dict[str, object]] = {}
        if images:
            image_bound = first_batch_item(processed["image_bound"])
            # note (MayDomine): slice order must match the placeholder bound order.
            pixel_values = [
                slice_tensor
                for per_image in processed["pixel_values"][0]
                for slice_tensor in (
                    per_image if isinstance(per_image, list) else [per_image]
                )
            ]
            tgt_sizes = first_batch_item(processed["tgt_sizes"])
            mm_inputs["image"] = {"bounds": image_bound, "cache_key": image_cache_key}
            encoder_inputs["image_encoder"] = {
                "pixel_values": pixel_values,
                "tgt_sizes": tgt_sizes,
                "cache_key": image_cache_key,
            }
        else:
            pass
        if audios:
            audio_bounds = first_batch_item(processed["audio_bounds"])
            audio_feature_lens = first_batch_item(processed["audio_feature_lens"])
            mm_inputs["audio"] = {"bounds": audio_bounds, "cache_key": audio_cache_key}
            encoder_inputs["audio_encoder"] = {
                "audio_features": processed["audio_features"],
                "audio_feature_lens": audio_feature_lens,
                "cache_key": audio_cache_key,
            }
        else:
            pass

        state = MiniCPMOPipelineState(
            prompt={
                "prompt_text": prompt_text,
                "input_ids": input_ids,
                "attention_mask": attention_mask,
            },
            mm_inputs=mm_inputs,
            encoder_inputs=encoder_inputs,
            stream_state={"token_ids": [], "text": ""},
        )
        payload.data = state.to_dict()
        payload.request.inputs = None
        for key in (
            "audios",
            "audio",
            "images",
            "videos",
            "video",
            "video_fps",
            "video_max_frames",
            "video_min_pixels",
            "video_max_pixels",
            "video_total_pixels",
        ):
            payload.request.metadata.pop(key, None)
        return payload
