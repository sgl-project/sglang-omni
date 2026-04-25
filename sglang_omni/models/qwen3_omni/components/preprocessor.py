# SPDX-License-Identifier: Apache-2.0
"""Model-specific preprocessor for Qwen3-Omni."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any

import torch
from transformers.models.qwen3_omni_moe.processing_qwen3_omni_moe import (
    Qwen3OmniMoeProcessor,
)

from sglang_omni.models.qwen3_omni.io import PipelineState
from sglang_omni.models.qwen3_omni.pipeline.engine_io import (
    DEFAULT_THINKER_MAX_NEW_TOKENS,
    validate_prompt_seq_len,
)
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


def _resolve_local_model_dir(model_path: str) -> str:
    """Resolve a local model directory without eagerly hydrating full snapshots."""
    path = Path(model_path)
    if path.exists():
        return str(path)
    try:
        return str(resolve_model_path(model_path, local_files_only=True))
    except (FileNotFoundError, OSError) as exc:
        logger.warning(
            "Local-only model resolution failed for %s; falling back to hub id",
            model_path,
            exc_info=exc,
        )
        return model_path


def _combine_cache_keys(*keys: str | None) -> str | None:
    parts = [key for key in keys if key]
    if not parts:
        return None
    return "|".join(parts)


def _contextualize_cache_key(base_key: str | None, **context: Any) -> str | None:
    if base_key is None:
        return None
    parts = [base_key]
    for key in sorted(context):
        value = context[key]
        if value is not None:
            parts.append(f"{key}={value}")
    return "|".join(parts)


class Qwen3OmniPreprocessor:
    """CPU-side preprocessing and tokenization using the HF processor."""

    def __init__(self, model_path: str, max_seq_len: int | None = None):
        self.model_path = model_path
        self.max_seq_len = max_seq_len
        self.model_dir = _resolve_local_model_dir(model_path)
        try:
            self.processor = Qwen3OmniMoeProcessor.from_pretrained(
                self.model_dir,
                trust_remote_code=True,
                local_files_only=True,
            )
        except (OSError, ValueError, RuntimeError):
            if Path(model_path).exists():
                raise
            self.processor = Qwen3OmniMoeProcessor.from_pretrained(
                model_path,
                trust_remote_code=True,
                local_files_only=False,
            )
            self.model_dir = str(resolve_model_path(model_path, local_files_only=False))
        self.tokenizer = self.processor.tokenizer
        ensure_chat_template(self.tokenizer, model_path=self.model_dir)

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
            # TODO (Qiujiang): route video options through the unified runtime parameter path.
            video_fps = inputs.get("video_fps", 1.0)
            video_max_frames = inputs.get("video_max_frames")
            video_min_pixels = inputs.get("video_min_pixels")
            video_max_pixels = inputs.get("video_max_pixels")
            video_total_pixels = inputs.get("video_total_pixels")
            use_audio_in_video = inputs.get("use_audio_in_video")
            video_seconds_per_chunk = inputs.get("video_seconds_per_chunk")
            video_position_id_per_seconds = inputs.get("video_position_id_per_seconds")
            audio_from_video = False
            num_explicit_audios = 0

            # Compute cache keys BEFORE conversion (paths are cheap to hash)
            image_cache_key = compute_image_cache_key(raw_images)
            raw_audio_cache_key = compute_audio_cache_key(raw_audios)
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
                    max_frames=video_max_frames,
                    min_pixels=video_min_pixels,
                    max_pixels=video_max_pixels,
                    total_pixels=video_total_pixels,
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
            audio_target_sr = None
            image_cache_key = None
            raw_audio_cache_key = None
            video_cache_key = None
            audio_target_sr = 16000
            video_fps = None
            video_max_frames = None
            video_min_pixels = None
            video_max_pixels = None
            video_total_pixels = None
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
        if video_max_frames is not None:
            videos_kwargs["max_frames"] = int(video_max_frames)
        if video_min_pixels is not None:
            videos_kwargs["min_pixels"] = int(video_min_pixels)
        if video_max_pixels is not None:
            videos_kwargs["max_pixels"] = int(video_max_pixels)
        if video_total_pixels is not None:
            videos_kwargs["total_pixels"] = int(video_total_pixels)
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
        validate_prompt_seq_len(
            input_ids,
            max_seq_len=self.max_seq_len,
            max_new_tokens=payload.request.params.get(
                "max_new_tokens", DEFAULT_THINKER_MAX_NEW_TOKENS
            ),
            request_id=payload.request_id,
        )

        mm_inputs: dict[str, Any] = {
            "image": build_image_mm_inputs(hf_inputs),
            "audio": build_audio_mm_inputs(hf_inputs),
            "video": build_video_mm_inputs(hf_inputs),
        }
        if use_audio_in_video is not None:
            mm_inputs["video"]["use_audio_in_video"] = bool(use_audio_in_video)

        # Build encoder_inputs with cache_key for efficient caching.
        # Include preprocessing parameters that materially change encoder outputs.
        image_encoder_inputs = {**mm_inputs["image"], **mm_inputs["video"]}
        effective_video_fps: tuple[float, ...] | None = None
        if sampled_video_fps is not None:
            effective_video_fps = tuple(float(fps) for fps in sampled_video_fps)
        elif video_fps is not None:
            effective_video_fps = (float(video_fps),)

        contextual_video_cache_key = _contextualize_cache_key(
            video_cache_key,
            fps=effective_video_fps,
            max_frames=video_max_frames,
            min_pixels=video_min_pixels,
            max_pixels=video_max_pixels,
            total_pixels=video_total_pixels,
        )
        combined_cache_key = _combine_cache_keys(
            image_cache_key, contextual_video_cache_key
        )
        if combined_cache_key:
            image_encoder_inputs["cache_key"] = combined_cache_key

        audio_encoder_inputs = {**mm_inputs["audio"]}
        contextualized_audio_cache_key = _contextualize_cache_key(
            raw_audio_cache_key,
            target_sr=audio_target_sr,
        )
        if audio_from_video:
            contextualized_audio_cache_key = _combine_cache_keys(
                contextualized_audio_cache_key,
                _contextualize_cache_key(
                    video_cache_key,
                    extracted_audio=True,
                    target_sr=audio_target_sr,
                ),
            )
        if contextualized_audio_cache_key:
            audio_encoder_inputs["cache_key"] = contextualized_audio_cache_key

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
