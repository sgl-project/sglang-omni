# SPDX-License-Identifier: Apache-2.0
"""Model-specific preprocessor for Qwen3-Omni."""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any, Mapping

import torch
from transformers import AutoFeatureExtractor, AutoImageProcessor, AutoTokenizer
from transformers.models.qwen3_omni_moe.processing_qwen3_omni_moe import (
    Qwen3OmniMoeProcessor,
)

from sglang_omni.models.qwen3_omni.io import PipelineState
from sglang_omni.models.weight_loader import resolve_model_path
from sglang_omni.preprocessing import (
    build_audio_mm_inputs,
    build_image_mm_inputs,
    build_video_mm_inputs,
    compute_audio_cache_key,
    compute_image_cache_key,
    compute_video_cache_key,
    ensure_audio_list_async,
    ensure_chat_template,
    ensure_image_list_async,
    ensure_video_list_async,
    normalize_messages,
)
from sglang_omni.proto import StagePayload

logger = logging.getLogger(__name__)


def _resolve_local_model_dir(model_id: str) -> str:
    """Resolve a local model directory, downloading snapshot when needed."""
    return str(resolve_model_path(model_id, local_files_only=False))


def _load_preprocessor_config(model_dir: str) -> Mapping[str, Any]:
    cfg_path = Path(model_dir) / "preprocessor_config.json"
    with open(cfg_path, encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, Mapping):
        raise ValueError("preprocessor_config.json is not a mapping")
    return payload


class _StubVideoProcessor:
    """Minimal video processor to avoid remote downloads when video is unused."""

    def __init__(self, *, merge_size: int, temporal_patch_size: int):
        self.merge_size = merge_size
        self.temporal_patch_size = temporal_patch_size

    def __call__(
        self, *args: Any, **kwargs: Any
    ) -> Any:  # pragma: no cover - defensive
        raise NotImplementedError("Video inputs are not supported in this pipeline yet")


def _build_processor_local(model_dir: str, model_id: str) -> Qwen3OmniMoeProcessor:
    cfg = _load_preprocessor_config(model_dir)
    merge_size = int(cfg.get("merge_size", 2))
    temporal_patch_size = int(cfg.get("temporal_patch_size", 2))

    tokenizer = AutoTokenizer.from_pretrained(
        model_dir,
        trust_remote_code=True,
        local_files_only=True,
    )
    ensure_chat_template(tokenizer, model_id=model_dir)

    image_processor = AutoImageProcessor.from_pretrained(
        model_dir,
        trust_remote_code=True,
        local_files_only=True,
    )
    feature_extractor = AutoFeatureExtractor.from_pretrained(
        model_dir,
        trust_remote_code=True,
        local_files_only=True,
    )
    video_processor = _StubVideoProcessor(
        merge_size=merge_size,
        temporal_patch_size=temporal_patch_size,
    )
    processor = Qwen3OmniMoeProcessor(
        image_processor=image_processor,
        video_processor=video_processor,
        feature_extractor=feature_extractor,
        tokenizer=tokenizer,
    )
    ensure_chat_template(processor.tokenizer, model_id=model_dir)
    return processor


class Qwen3OmniPreprocessor:
    """CPU-side preprocessing and tokenization using the HF processor."""

    def __init__(self, model_id: str):
        self.model_id = model_id
        self.model_dir = _resolve_local_model_dir(model_id)
        try:
            self.processor = Qwen3OmniMoeProcessor.from_pretrained(
                self.model_dir,
                trust_remote_code=True,
                local_files_only=True,
            )
        except Exception:
            # The hub cache may not include video_preprocessor_config.json.
            # Build the processor locally with a stub video processor instead.
            self.processor = _build_processor_local(self.model_dir, model_id)
        self.tokenizer = self.processor.tokenizer
        ensure_chat_template(self.tokenizer, model_id=self.model_dir)

    def _build_multimodal_messages(
        self,
        messages: list[dict[str, Any]],
        *,
        num_images: int,
        num_audios: int,
        num_videos: int,
    ) -> list[dict[str, Any]]:
        """Convert simple messages to HF's structured multimodal format."""
        if num_images == 0 and num_audios == 0 and num_videos == 0:
            return messages

        result: list[dict[str, Any]] = []
        for i, msg in enumerate(messages):
            role = msg.get("role", "user")
            content = msg.get("content", "")

            # Only inject placeholders into the last user message
            if i == len(messages) - 1 and role == "user":
                content_parts: list[dict[str, Any]] = []
                # Placeholders come BEFORE text (Qwen3-Omni format)
                for _ in range(num_images):
                    content_parts.append({"type": "image"})
                for _ in range(num_videos):
                    content_parts.append({"type": "video"})
                for _ in range(num_audios):
                    content_parts.append({"type": "audio"})
                content_parts.append({"type": "text", "text": content})
                result.append({"role": role, "content": content_parts})
            else:
                result.append(msg)

        return result

    async def __call__(self, payload: StagePayload) -> StagePayload:
        inputs = payload.request.inputs
        if isinstance(inputs, dict):
            messages = inputs.get("messages", [])
            raw_images = inputs.get("images")
            raw_videos = inputs.get("videos") or inputs.get("video")
            raw_audios = inputs.get("audio") or inputs.get("audios")
            audio_target_sr = int(inputs.get("audio_target_sr", 16000))
            video_fps = inputs.get("video_fps")
            use_audio_in_video = inputs.get("use_audio_in_video")
            video_seconds_per_chunk = inputs.get("video_seconds_per_chunk")
            video_position_id_per_seconds = inputs.get("video_position_id_per_seconds")
            audio_from_video = False
            num_explicit_audios = 0

            # Compute cache keys BEFORE conversion (paths are cheap to hash)
            image_cache_key = compute_image_cache_key(raw_images)
            audio_cache_key = compute_audio_cache_key(raw_audios)
            video_cache_key = compute_video_cache_key(raw_videos)

            # Count explicit audio inputs (for placeholder insertion)
            if raw_audios:
                num_explicit_audios = (
                    len(raw_audios) if isinstance(raw_audios, list) else 1
                )

            # Use async versions for concurrent loading
            # If we need audio from video, extract it during video loading to avoid duplicate downloads
            extract_audio_from_video_flag = bool(use_audio_in_video and raw_videos)

            images, videos_result, audios_result = await asyncio.gather(
                ensure_image_list_async(raw_images),
                ensure_video_list_async(
                    raw_videos,
                    fps=video_fps,
                    extract_audio=extract_audio_from_video_flag,
                    audio_target_sr=audio_target_sr,
                ),
                ensure_audio_list_async(raw_audios, target_sr=audio_target_sr),
            )
            videos, sampled_video_fps, extracted_audio_from_video = videos_result

            # Merge extracted audio from videos with explicit audio (if any)
            if extracted_audio_from_video:
                # Filter out None values (videos without audio)
                extracted_audio_from_video = [
                    audio for audio in extracted_audio_from_video if audio is not None
                ]
                if extracted_audio_from_video:
                    audio_from_video = True
                    # Merge with explicit audio
                    if audios_result:
                        if isinstance(audios_result, list):
                            audios = audios_result + extracted_audio_from_video
                        else:
                            audios = [audios_result] + extracted_audio_from_video
                    else:
                        audios = extracted_audio_from_video
                else:
                    audios = audios_result
            else:
                audios = audios_result
        else:
            messages = inputs
            images = []
            videos = []
            audios = []
            image_cache_key = None
            audio_cache_key = None
            video_cache_key = None
            video_fps = None
            sampled_video_fps = None
            use_audio_in_video = None
            video_seconds_per_chunk = None
            video_position_id_per_seconds = None
            audio_from_video = False
            num_explicit_audios = 0

        messages_norm = normalize_messages(messages)
        # Insert placeholders:
        # - Explicit audio files get independent audio placeholders
        # - Video audio (when use_audio_in_video=True) is handled by video token, no separate placeholder
        num_audios_for_placeholder = num_explicit_audios
        messages_mm = self._build_multimodal_messages(
            messages_norm,
            num_images=len(images),
            num_audios=num_audios_for_placeholder,
            num_videos=len(videos),
        )
        prompt_text = self.processor.apply_chat_template(
            messages_mm,
            add_generation_prompt=True,
            tokenize=False,
        )

        videos_kwargs: dict[str, Any] = {}
        if sampled_video_fps is not None:
            videos_kwargs["fps"] = (
                sampled_video_fps[0]
                if len(sampled_video_fps) == 1
                else sampled_video_fps
            )
        elif video_fps is not None:
            videos_kwargs["fps"] = video_fps
        if use_audio_in_video is not None:
            videos_kwargs["use_audio_in_video"] = bool(use_audio_in_video)
        if video_seconds_per_chunk is not None:
            videos_kwargs["seconds_per_chunk"] = float(video_seconds_per_chunk)
        if video_position_id_per_seconds is not None:
            videos_kwargs["position_id_per_seconds"] = float(
                video_position_id_per_seconds
            )
        if videos:
            # torchcodec backend expects a non-None device string
            videos_kwargs.setdefault("device", "cpu")
        processor_kwargs: dict[str, Any] = {}
        if videos_kwargs:
            processor_kwargs["videos_kwargs"] = videos_kwargs

        hf_inputs = self.processor(
            text=prompt_text,
            images=images or None,
            videos=videos or None,
            audio=audios or None,
            add_special_tokens=False,
            return_tensors="pt",
            **processor_kwargs,
        )

        input_ids = hf_inputs["input_ids"][0]
        attention_mask = hf_inputs.get("attention_mask")
        if isinstance(attention_mask, torch.Tensor):
            attention_mask = attention_mask[0]
        else:
            attention_mask = torch.ones_like(input_ids)

        mm_inputs: dict[str, Any] = {
            "image": build_image_mm_inputs(hf_inputs),
            "audio": build_audio_mm_inputs(hf_inputs),
            "video": build_video_mm_inputs(hf_inputs),
        }
        if use_audio_in_video is not None:
            mm_inputs["video"]["use_audio_in_video"] = bool(use_audio_in_video)

        # Build encoder_inputs with cache_key for efficient caching
        image_encoder_inputs = {**mm_inputs["image"], **mm_inputs["video"]}
        if image_cache_key:
            image_encoder_inputs["cache_key"] = image_cache_key

        audio_encoder_inputs = {**mm_inputs["audio"]}
        if audio_cache_key:
            audio_encoder_inputs["cache_key"] = audio_cache_key

        encoder_inputs: dict[str, dict[str, Any]] = {}
        image_encoder_inputs = {
            k: v for k, v in image_encoder_inputs.items() if v is not None
        }
        if (
            image_encoder_inputs.get("pixel_values") is not None
            or image_encoder_inputs.get("pixel_values_videos") is not None
        ):
            encoder_inputs["image_encoder"] = image_encoder_inputs
        else:
            encoder_inputs["image_encoder"] = {"_skip": True, "_result": {}}
        if audio_encoder_inputs.get("input_features") is not None:
            encoder_inputs["audio_encoder"] = audio_encoder_inputs
        else:
            encoder_inputs["audio_encoder"] = {"_skip": True, "_result": {}}

        state = PipelineState(
            raw_inputs=inputs,
            mm_inputs=mm_inputs,
            prompt={
                "prompt_text": prompt_text,
                "input_ids": input_ids,
                "attention_mask": attention_mask,
            },
            encoder_inputs=encoder_inputs,
            stream_state={"token_ids": [], "text": ""},
        )
        payload.data = state.to_dict()
        return payload
