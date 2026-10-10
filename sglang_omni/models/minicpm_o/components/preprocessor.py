# SPDX-License-Identifier: Apache-2.0
"""Render MiniCPM-o prompts and extract media features and placeholder bounds."""

from __future__ import annotations

import re
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import numpy.typing as npt
import torch
from PIL import Image
from transformers import AutoProcessor, AutoTokenizer, ProcessorMixin

from sglang_omni.models.minicpm_o.payload_types import (
    AudioEncoderInputs,
    ImageEncoderInputs,
    MiniCPMOPipelineState,
    ModalityInputs,
    StreamState,
)
from sglang_omni.models.minicpm_o.prompt_frontend import AUDIO_PLACEHOLDER
from sglang_omni.models.minicpm_o.prompt_frontend import (
    IMAGE_PLACEHOLDER as IMAGE_PLACEHOLDER,
)
from sglang_omni.models.minicpm_o.prompt_frontend import (
    RenderedChat,
    first_batch_item,
    has_inline_media,
    messages_with_media_placeholders,
    normalize_message_contents,
    render_ordered_chat,
    video_to_images,
)
from sglang_omni.models.minicpm_o.routing import should_generate_audio_output
from sglang_omni.models.minicpm_o.video_frontend import (
    VideoProcessingOptions,
    load_timed_video,
)
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
from sglang_omni.preprocessing.video import compute_video_cache_key
from sglang_omni.proto import StagePayload

TTS_READ_PROMPT_EN = "Please read the following text out loud in English: "
TTS_READ_PROMPT_ZH = "请用中文朗读以下文本: "
# Forwarded to the talker as talker_<field>; the seed already reaches it.
TALKER_SPEECH_SAMPLING_FIELDS = (
    "max_new_tokens",
    "temperature",
    "top_p",
    "top_k",
    "repetition_penalty",
)

# note (MayDomine): task prompts match the checkpoint's audio-understanding template.
ASR_PROMPT_ZH = "请仔细听这段音频片段，并将其内容逐字记录。"
ASR_PROMPT_EN = (
    "Please listen to the audio snippet carefully and transcribe the content."
)


def is_chinese(text: str) -> bool:
    return bool(re.search(r"[\u4e00-\u9fff]", text))


def tts_read_prompt(text: str, language: str | None) -> str:
    if language in (None, "Auto"):
        language = "Chinese" if is_chinese(text) else "English"
    else:
        pass
    return TTS_READ_PROMPT_ZH if language == "Chinese" else TTS_READ_PROMPT_EN


class MiniCPMOPreprocessor:
    def __init__(
        self,
        model_path: str,
        *,
        speech_enabled: bool = False,
    ) -> None:
        local_model_directory = str(resolve_model_path(model_path))
        self.tokenizer = AutoTokenizer.from_pretrained(
            local_model_directory, trust_remote_code=True
        )
        # note (MayDomine): text-only requests do not need Whisper feature extraction.
        self.model_dir: str = local_model_directory
        self._processor: ProcessorMixin | None = None  # noqa: leading-underscore
        self.speech_enabled: bool = speech_enabled

    def speech_to_text_inputs(
        self, payload: StagePayload, inputs: Mapping[str, object]
    ) -> tuple[list[dict[str, str]], list[npt.NDArray[np.float32]]]:
        """Turn a transcription upload into a chat turn plus audio list."""
        request_parameters = payload.request.params or {}
        language = str(request_parameters.get("language") or "").lower()
        prompt = ASR_PROMPT_ZH if language.startswith("zh") else ASR_PROMPT_EN
        audio_bytes = inputs["audio_bytes"]
        if not isinstance(audio_bytes, bytes):
            raise ValueError("Transcription input requires audio bytes")
        else:
            audio_waveform, sample_rate = AudioMediaIO(target_sr=16000).load_bytes(
                audio_bytes
            )
        # note (Tianyao Wu): the recipe puts a blank line between prompt and audio.
        message = {"role": "user", "content": f"{prompt}\n\n{AUDIO_PLACEHOLDER}"}
        return [message], [audio_waveform]

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
        params = payload.request.params or {}
        metadata = payload.request.metadata or {}
        known_tts_text = None
        if metadata.get("task") == "tts":
            known_tts_text = inputs["text"] if isinstance(inputs, Mapping) else inputs
            tts_params = metadata.get("tts_params") or {}
            read_prompt = tts_read_prompt(known_tts_text, tts_params.get("language"))
            inputs = [{"role": "user", "content": f"{read_prompt}{known_tts_text}"}]
            explicit_fields = tts_params.get("explicit_generation_params") or []
            talker_params = {
                f"talker_{field}": value
                for field, value in params.items()
                if field in TALKER_SPEECH_SAMPLING_FIELDS
                and field in explicit_fields
                and value is not None
            }
            params = {**params, **talker_params}
            payload.request.params = params
        else:
            pass
        raw_images = None
        raw_audios = None
        raw_videos = None
        use_audio_in_video = False
        video_options = VideoProcessingOptions()
        media_placeholders_placed = False
        audio_turn_indices: list[int] | None = None
        has_inline_video = False
        rendered_chat: RenderedChat | None = None
        if isinstance(inputs, dict) and inputs.get("audio_bytes") is not None:
            messages, raw_audios = self.speech_to_text_inputs(payload, inputs)
            media_placeholders_placed = True
        elif isinstance(inputs, dict):
            messages = inputs.get("messages", [])
            raw_images = inputs.get("images")
            raw_audios = inputs.get("audio") or inputs.get("audios")
            raw_videos = inputs.get("videos") or inputs.get("video")
            use_audio_in_video = bool(inputs.get("use_audio_in_video", False))
            video_options = VideoProcessingOptions.model_validate(inputs)
        else:
            messages = inputs

        if known_tts_text is not None:
            if not isinstance(known_tts_text, str) or not known_tts_text.strip():
                raise ValueError("speech input must be nonempty text")
            elif not self.should_use_tts_template(payload):
                raise ValueError("speech requests require the speech pipeline")
            elif raw_images or raw_audios or raw_videos:
                raise ValueError("speech requests take text only")
            elif params.get("stream", False):
                raise ValueError("MiniCPM-o speech output does not stream")
            else:
                pass
        else:
            pass

        if raw_videos and not has_inline_media(messages):
            video_sources = raw_videos if isinstance(raw_videos, list) else [raw_videos]
            has_only_video_sources = all(
                isinstance(source, str) for source in video_sources
            )
            if use_audio_in_video and not has_only_video_sources:
                raise ValueError(
                    "Video audio interleaving requires file or URL video sources"
                )
            elif not has_only_video_sources:
                pass
            elif (
                not isinstance(messages, list)
                or not messages
                or messages[-1].get("role") != "user"
            ):
                raise ValueError("Top-level video inputs require a final user message")
            else:
                image_contents = await ensure_image_list_async(raw_images)
                audio_contents = await ensure_audio_list_async(
                    raw_audios, target_sr=16000
                )
                content = messages[-1].get("content", "")
                text_contents = (
                    content
                    if isinstance(content, list)
                    else ([] if content == "" else [content])
                )
                messages = [
                    *messages[:-1],
                    {
                        **messages[-1],
                        "content": [
                            *image_contents,
                            *audio_contents,
                            *[
                                {
                                    "type": "video_url",
                                    "video_url": {
                                        "url": source,
                                        "use_audio": use_audio_in_video,
                                    },
                                }
                                for source in video_sources
                            ],
                            *text_contents,
                        ],
                    },
                ]
                raw_images = raw_audios = raw_videos = None
        else:
            pass

        if has_inline_media(messages):
            if raw_images or raw_audios or raw_videos:
                raise ValueError(
                    "Inline media cannot be combined with top-level images, audios or videos"
                )
            else:
                rendered_chat = await render_ordered_chat(
                    messages,
                    use_audio_in_video=use_audio_in_video,
                    video_fps=video_options.video_fps,
                    video_max_frames=video_options.video_max_frames,
                    video_min_pixels=video_options.video_min_pixels,
                    video_max_pixels=video_options.video_max_pixels,
                    video_total_pixels=video_options.video_total_pixels,
                )
                messages = rendered_chat.messages
                raw_images = rendered_chat.images
                raw_audios = rendered_chat.audios
                audio_turn_indices = rendered_chat.audio_turn_indices
                media_placeholders_placed = True
                has_inline_video = rendered_chat.has_video
        else:
            pass

        if raw_images or raw_audios or raw_videos:
            return await self.preprocess_multimodal(
                payload,
                messages,
                raw_images=raw_images,
                raw_audios=raw_audios,
                raw_videos=raw_videos,
                video_options=video_options,
                media_placeholders_placed=media_placeholders_placed,
                audio_turn_indices=audio_turn_indices,
                has_inline_video=has_inline_video,
                rendered_chat=rendered_chat,
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

        known_tts_output_ids = None
        if known_tts_text is not None:
            tts_bos_token_id = self.tokenizer.convert_tokens_to_ids("<|tts_bos|>")
            tts_eos_token_id = self.tokenizer.convert_tokens_to_ids("<|tts_eos|>")
            if int(input_ids[-1]) != tts_bos_token_id:
                raise ValueError("speech prompt must end at the TTS boundary")
            else:
                pass
            known_tts_output_ids = self.tokenizer.encode(
                known_tts_text, add_special_tokens=False
            )
            if not known_tts_output_ids or any(
                token_id in (tts_bos_token_id, tts_eos_token_id)
                for token_id in known_tts_output_ids
            ):
                raise ValueError("speech input has no speakable tokens")
            else:
                pass
            suffix = torch.tensor(
                [*known_tts_output_ids, tts_eos_token_id], dtype=torch.long
            )
            input_ids = torch.cat((input_ids, suffix))
            attention_mask = torch.ones_like(input_ids)
        else:
            pass

        stream_state: StreamState = {"token_ids": [], "text": ""}
        state = MiniCPMOPipelineState(
            prompt={
                "prompt_text": prompt_text,
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "known_tts_output_ids": known_tts_output_ids,
            },
            stream_state=stream_state,
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
        messages = normalize_message_contents(messages)
        return self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
            use_tts_template=use_tts_template,
            enable_thinking=False,
        )

    async def preprocess_multimodal(
        self,
        payload: StagePayload,
        messages: object,
        *,
        raw_images: object,
        raw_audios: object,
        raw_videos: object,
        video_options: VideoProcessingOptions,
        media_placeholders_placed: bool,
        audio_turn_indices: list[int] | None = None,
        has_inline_video: bool = False,
        rendered_chat: RenderedChat | None = None,
    ) -> StagePayload:
        images = (
            list(rendered_chat.images)
            if rendered_chat is not None
            else await ensure_image_list_async(raw_images)
        )
        if raw_videos:
            video_sources = raw_videos if isinstance(raw_videos, list) else [raw_videos]
            videos: list[list[Image.Image]] = []
            for source in video_sources:
                if isinstance(source, (str, Path)):
                    video = await load_timed_video(
                        str(source),
                        use_audio=False,
                        fps=video_options.video_fps,
                        max_frames=video_options.video_max_frames,
                        min_pixels=video_options.video_min_pixels,
                        max_pixels=video_options.video_max_pixels,
                        total_pixels=video_options.video_total_pixels,
                    )
                    videos.append(video.frames)
                else:
                    videos.append(video_to_images(source))
        else:
            videos = []
        # note (Yuhao Chen): image and video cache identities use separate source media.
        image_cache_key = compute_image_cache_key(images)
        video_cache_key = compute_video_cache_key(
            videos,
            fps=video_options.video_fps,
            max_frames=video_options.video_max_frames,
            min_pixels=video_options.video_min_pixels,
            max_pixels=video_options.video_max_pixels,
            total_pixels=video_options.video_total_pixels,
        )
        video_images = [frame for video in videos for frame in video_to_images(video)]
        images.extend(video_images)
        audios = (
            list(rendered_chat.audios)
            if rendered_chat is not None
            else await ensure_audio_list_async(raw_audios, target_sr=16000)
        )
        audio_cache_key = compute_audio_cache_key(audios)
        if audio_cache_key is not None and audio_turn_indices is not None:
            audio_cache_key = f"{audio_cache_key}|parts={audio_turn_indices}"
        else:
            pass

        cache_keys = [
            cache_key for cache_key in (image_cache_key, video_cache_key) if cache_key
        ]
        image_cache_key = "|".join(cache_keys) if cache_keys else None
        # note (Yuhao Chen): slicing policy changes cached embeddings for identical pixels.
        processor_video_options = (
            {"max_slice_nums": 1, "use_image_id": False}
            if raw_videos or has_inline_video
            else {}
        )
        if image_cache_key is not None:
            image_processing_policy = (
                "video-v1" if processor_video_options else "default-v1"
            )
            image_cache_key = f"{image_cache_key}|policy={image_processing_policy}"
        else:
            pass

        if (
            not media_placeholders_placed
            and isinstance(messages, list)
            and not (messages and all(isinstance(token, int) for token in messages))
        ):
            messages = messages_with_media_placeholders(
                messages, image_count=len(images), audio_count=len(audios)
            )
        else:
            pass
        prompt_text = self.render_chat_template(
            messages,
            use_tts_template=bool(audios) or self.should_use_tts_template(payload),
        )

        processed = self.processor(
            prompt_text,
            images=[images] if images else None,
            audios=[audios] if audios else None,
            **(
                {"audio_parts": [audio_turn_indices]}
                if audio_turn_indices is not None
                else {}
            ),
            return_tensors="pt",
            **processor_video_options,
        )

        input_ids = processed["input_ids"][0].to(dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)

        modality_inputs: dict[str, ModalityInputs] = {}
        encoder_inputs: dict[str, ImageEncoderInputs | AudioEncoderInputs] = {}
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
            target_image_sizes = first_batch_item(processed["tgt_sizes"])
            modality_inputs["image"] = {
                "bounds": image_bound,
                "cache_key": image_cache_key,
            }
            encoder_inputs["image_encoder"] = {
                "pixel_values": pixel_values,
                "tgt_sizes": target_image_sizes,
                "cache_key": image_cache_key,
            }
        else:
            pass
        if audios:
            audio_bounds = first_batch_item(processed["audio_bounds"])
            audio_feature_lengths = first_batch_item(processed["audio_feature_lens"])
            modality_inputs["audio"] = {
                "bounds": audio_bounds,
                "cache_key": audio_cache_key,
            }
            encoder_inputs["audio_encoder"] = {
                "audio_features": processed["audio_features"],
                "audio_feature_lens": audio_feature_lengths,
                "cache_key": audio_cache_key,
            }
        else:
            pass

        stream_state: StreamState = {"token_ids": [], "text": ""}
        state = MiniCPMOPipelineState(
            prompt={
                "prompt_text": prompt_text,
                "input_ids": input_ids,
                "attention_mask": attention_mask,
            },
            mm_inputs=modality_inputs,
            encoder_inputs=encoder_inputs,
            stream_state=stream_state,
        )
        payload.data = state.to_dict()
        payload.request.inputs = None
        for metadata_field in (
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
            payload.request.metadata.pop(metadata_field, None)
        return payload
