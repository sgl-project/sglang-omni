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

from sglang_omni.cache import CacheKey, get_global_cache_plane
from sglang_omni.models.qwen3_omni.payload_types import Qwen3OmniPipelineState
from sglang_omni.models.qwen3_omni.request_builders import build_lightweight_mm_inputs
from sglang_omni.models.qwen3_omni.session_media import (
    SessionMediaEntry,
    SessionMediaRegistry,
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
from sglang_omni.profiler.event_recorder import emit as _emit_event
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


def _has_media_refs(*values: Any) -> bool:
    return any(value is not None and value != [] and value != () for value in values)


def _metadata_or_input_value(
    inputs: Any,
    metadata: dict[str, Any],
    key: str,
) -> Any:
    if key in metadata:
        return metadata[key]
    if isinstance(inputs, dict):
        return inputs.get(key)
    return None


def _truthy_media_flag(value: Any) -> bool:
    if isinstance(value, str):
        return value.lower() not in ("", "0", "false", "no", "none")
    if isinstance(value, (list, tuple, set, dict)):
        return bool(value)
    return bool(value)


def _session_id_from(inputs: Any, metadata: dict[str, Any]) -> str | None:
    value = metadata.get("session_id")
    if value is None and isinstance(inputs, dict):
        value = inputs.get("session_id")
    if value is None:
        return None
    value = str(value)
    return value or None


def _encoder_cache_ready(*, kind: str, cache_key: str, stage_name: str) -> bool:
    entry = get_global_cache_plane().peek(
        CacheKey(
            namespace="qwen3_omni",
            kind=kind,
            digest=cache_key,
            stage_name=stage_name,
        )
    )
    return entry is not None


def _metadata_cache_value(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: _metadata_cache_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_metadata_cache_value(item) for item in value)
    return value


def _iter_metadata_rows(value: Any):
    if value is None:
        return iter([])
    if isinstance(value, torch.Tensor):
        return iter(value)
    return iter(value)


DEFAULT_THINKER_MAX_NEW_TOKENS = 2048
QWEN3_OMNI_CHAT_TEMPLATE_FALLBACK_MODEL = "Qwen/Qwen3-Omni-30B-A3B-Instruct"


def validate_prompt_seq_len(
    input_ids: torch.Tensor,
    *,
    max_seq_len: int | None,
    max_new_tokens: int = DEFAULT_THINKER_MAX_NEW_TOKENS,
    request_id: str | None = None,
) -> None:
    if max_seq_len is None:
        return
    prompt_len = int(input_ids.numel())
    if prompt_len >= max_seq_len:
        logger.info(
            f"rejecting request {request_id}: prompt {prompt_len} tokens "
            f">= max_seq_len {max_seq_len}"
        )
        raise ValueError(
            f"The input ({prompt_len} tokens) is longer than the model's "
            f"context length ({max_seq_len} tokens)."
        )
    total_tokens = prompt_len + int(max_new_tokens)
    if total_tokens >= max_seq_len:
        logger.info(
            f"rejecting request {request_id}: prompt {prompt_len} + "
            f"max_new_tokens {int(max_new_tokens)} = {total_tokens} tokens "
            f">= max_seq_len {max_seq_len}"
        )
        raise ValueError(
            f"Requested token count exceeds the model's maximum context length "
            f"of {max_seq_len} tokens. You requested a total of {total_tokens} "
            f"tokens: {prompt_len} tokens from the input messages and "
            f"{int(max_new_tokens)} tokens for the completion. Please reduce "
            f"the number of tokens in the input messages or the completion to "
            f"fit within the limit."
        )


def _is_pretokenized_prompt(inputs: Any) -> bool:
    """True when a rollout request carries pre-tokenized prompt ids.

    Miles RL rollout sends the exact prompt token ids it trains on, so those
    ids must bypass the chat template + HF processor to keep rollout and
    training tokens identical. A list of message dicts goes the normal path.
    """
    return (
        isinstance(inputs, list)
        and bool(inputs)
        and all(isinstance(token, int) for token in inputs)
    )


class Qwen3OmniPreprocessor:
    """CPU-side preprocessing and tokenization using the HF processor."""

    def __init__(
        self,
        model_path: str,
        max_seq_len: int | None = None,
        *,
        video_fps: float | None = None,
        video_max_frames: int | None = None,
        video_min_pixels: int | None = None,
        video_max_pixels: int | None = None,
        video_total_pixels: int | None = None,
    ):
        self.model_path = model_path
        self.max_seq_len = max_seq_len
        self.default_video_fps = float(video_fps) if video_fps is not None else None
        self.default_video_max_frames = (
            int(video_max_frames) if video_max_frames is not None else None
        )
        self.default_video_min_pixels = (
            int(video_min_pixels) if video_min_pixels is not None else None
        )
        self.default_video_max_pixels = (
            int(video_max_pixels) if video_max_pixels is not None else None
        )
        self.default_video_total_pixels = (
            int(video_total_pixels) if video_total_pixels is not None else None
        )
        self.session_media = SessionMediaRegistry()
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
        ensure_chat_template(
            self.tokenizer,
            model_path=self.model_dir,
            fallback_model_paths=(QWEN3_OMNI_CHAT_TEMPLATE_FALLBACK_MODEL,),
        )
        if not getattr(self.processor, "chat_template", None) and getattr(
            self.tokenizer, "chat_template", None
        ):
            self.processor.chat_template = self.tokenizer.chat_template

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
        _emit_event(
            request_id=payload.request_id,
            stage=None,
            event_name="preprocess_start",
        )
        try:
            result = await self._call_impl(payload)
        finally:
            _emit_event(
                request_id=payload.request_id,
                stage=None,
                event_name="preprocess_end",
            )
        return result

    def _finalize_state(
        self,
        payload: StagePayload,
        *,
        input_ids: "torch.Tensor",
        attention_mask: "torch.Tensor",
        prompt_text: str,
        full_mm_inputs: dict[str, Any],
        encoder_inputs: dict[str, dict[str, Any]],
    ) -> StagePayload:
        """Assemble the thinker-ready pipeline state (single source of shape)."""
        state = Qwen3OmniPipelineState(
            mm_inputs=build_lightweight_mm_inputs(full_mm_inputs),
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

    def _preprocess_pretokenized(
        self, payload: StagePayload, token_ids: list[int]
    ) -> StagePayload:
        """Build thinker state directly from pre-tokenized prompt ids.

        Skips the chat template + HF processor so the thinker runs the exact
        tokens the RL trainer computes gradients on (text-only; multimodal ids
        still go through the normal messages path).
        """
        input_ids = torch.tensor(token_ids, dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)
        validate_prompt_seq_len(
            input_ids,
            max_seq_len=self.max_seq_len,
            max_new_tokens=payload.request.params.get(
                "max_new_tokens", DEFAULT_THINKER_MAX_NEW_TOKENS
            ),
            request_id=payload.request_id,
        )
        return self._finalize_state(
            payload,
            input_ids=input_ids,
            attention_mask=attention_mask,
            prompt_text="",
            full_mm_inputs={},
            encoder_inputs={
                "image_encoder": {"_skip": True, "_result": {}},
                "audio_encoder": {"_skip": True, "_result": {}},
            },
        )

    def _preprocess_cached_session_media(
        self,
        payload: StagePayload,
        *,
        messages: Any,
        entry: SessionMediaEntry,
    ) -> StagePayload | None:
        encoder_inputs: dict[str, dict[str, Any]] = {}
        if entry.image_cache_key:
            if entry.num_images and entry.image_grid_thw is None:
                return None
            if entry.num_videos and entry.video_grid_thw is None:
                return None
            if not _encoder_cache_ready(
                kind="image_encoder_output",
                cache_key=entry.image_cache_key,
                stage_name="image_encoder",
            ):
                return None
            encoder_inputs["image_encoder"] = {
                "cache_key": entry.image_cache_key,
                "_active": True,
                "_cache_only": True,
            }
        else:
            encoder_inputs["image_encoder"] = {"_skip": True, "_result": {}}

        if entry.audio_cache_key:
            if entry.audio_placeholder_lengths is None:
                return None
            if not _encoder_cache_ready(
                kind="audio_encoder_output",
                cache_key=entry.audio_cache_key,
                stage_name="audio_encoder",
            ):
                return None
            encoder_inputs["audio_encoder"] = {
                "cache_key": entry.audio_cache_key,
                "_active": True,
                "_cache_only": True,
            }
        else:
            encoder_inputs["audio_encoder"] = {"_skip": True, "_result": {}}

        if not any(
            stage_inputs.get("_active") for stage_inputs in encoder_inputs.values()
        ):
            return None

        messages_norm = normalize_messages(messages)
        messages_mm = self._build_multimodal_messages(
            messages_norm,
            num_images=entry.num_images,
            num_audios=entry.num_audios,
            num_videos=entry.num_videos,
        )
        prompt_text = self.processor.apply_chat_template(
            messages_mm,
            add_generation_prompt=True,
            tokenize=False,
        )
        tokenize_text = prompt_text
        replace_tokens = getattr(
            self.processor,
            "replace_multimodal_special_tokens",
            None,
        )
        if callable(replace_tokens):
            tokenize_text = replace_tokens(
                [prompt_text],
                _iter_metadata_rows(entry.audio_placeholder_lengths),
                _iter_metadata_rows(entry.image_grid_thw),
                _iter_metadata_rows(entry.video_grid_thw),
                video_second_per_grid=_iter_metadata_rows(entry.video_second_per_grid),
                use_audio_in_video=bool(entry.use_audio_in_video),
                position_id_per_seconds=entry.video_position_id_per_seconds,
                seconds_per_chunk=entry.video_seconds_per_chunk,
            )[0]
        tokenized = self.tokenizer(
            tokenize_text,
            add_special_tokens=False,
            return_tensors="pt",
        )
        input_ids = tokenized["input_ids"][0]
        attention_mask = tokenized.get("attention_mask")
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
        return self._finalize_state(
            payload,
            input_ids=input_ids,
            attention_mask=attention_mask,
            prompt_text=prompt_text,
            full_mm_inputs={},
            encoder_inputs=encoder_inputs,
        )

    async def _call_impl(self, payload: StagePayload) -> StagePayload:
        inputs = payload.request.inputs
        if _is_pretokenized_prompt(inputs):
            return self._preprocess_pretokenized(payload, inputs)
        metadata = payload.request.metadata
        if not isinstance(metadata, dict):
            metadata = {}
        session_id = _session_id_from(inputs, metadata)
        if session_id is not None and _truthy_media_flag(
            _metadata_or_input_value(inputs, metadata, "clear_session_media")
        ):
            self.session_media.clear(session_id)
        metadata_images = _metadata_or_input_value(inputs, metadata, "images")
        metadata_videos = _metadata_or_input_value(
            inputs, metadata, "videos"
        ) or _metadata_or_input_value(inputs, metadata, "video")
        metadata_audios = _metadata_or_input_value(
            inputs, metadata, "audio"
        ) or _metadata_or_input_value(inputs, metadata, "audios")
        reuse_session_media = session_id is not None and _truthy_media_flag(
            _metadata_or_input_value(inputs, metadata, "reuse_session_media")
        )
        if (
            isinstance(inputs, dict)
            or _has_media_refs(
                metadata_images,
                metadata_videos,
                metadata_audios,
            )
            or reuse_session_media
        ):
            messages = (
                inputs.get("messages", []) if isinstance(inputs, dict) else inputs
            )
            raw_images = metadata_images
            raw_videos = metadata_videos
            raw_audios = metadata_audios
            reuse_entry: SessionMediaEntry | None = None
            if reuse_session_media and not _has_media_refs(
                raw_images, raw_videos, raw_audios
            ):
                reuse_entry = self.session_media.get(session_id)
                if reuse_entry is not None:
                    cached_payload = self._preprocess_cached_session_media(
                        payload,
                        messages=messages,
                        entry=reuse_entry,
                    )
                    if cached_payload is not None:
                        return cached_payload
                    if not _has_media_refs(
                        reuse_entry.raw_images,
                        reuse_entry.raw_videos,
                        reuse_entry.raw_audios,
                    ):
                        raise RuntimeError(
                            "session media cache is not ready and the session "
                            "does not retain raw media references; resend media"
                        )
                    raw_images = reuse_entry.raw_images
                    raw_videos = reuse_entry.raw_videos
                    raw_audios = reuse_entry.raw_audios
            audio_target_sr_value = _metadata_or_input_value(
                inputs, metadata, "audio_target_sr"
            )
            audio_target_sr = int(
                audio_target_sr_value if audio_target_sr_value is not None else 16000
            )
            if reuse_entry is not None:
                audio_target_sr_value = _metadata_or_input_value(
                    inputs, metadata, "audio_target_sr"
                )
                audio_target_sr = int(
                    audio_target_sr_value
                    if audio_target_sr_value is not None
                    else reuse_entry.audio_target_sr
                )
            video_fps = _metadata_or_input_value(inputs, metadata, "video_fps")
            if video_fps is None:
                video_fps = self.default_video_fps
            if reuse_entry is not None and video_fps is None:
                video_fps = reuse_entry.video_fps
            video_max_frames = _metadata_or_input_value(
                inputs, metadata, "video_max_frames"
            )
            if video_max_frames is None:
                video_max_frames = self.default_video_max_frames
            if reuse_entry is not None and video_max_frames is None:
                video_max_frames = reuse_entry.video_max_frames
            video_min_pixels = _metadata_or_input_value(
                inputs, metadata, "video_min_pixels"
            )
            if video_min_pixels is None:
                video_min_pixels = self.default_video_min_pixels
            if reuse_entry is not None and video_min_pixels is None:
                video_min_pixels = reuse_entry.video_min_pixels
            video_max_pixels = _metadata_or_input_value(
                inputs, metadata, "video_max_pixels"
            )
            if video_max_pixels is None:
                video_max_pixels = self.default_video_max_pixels
            if reuse_entry is not None and video_max_pixels is None:
                video_max_pixels = reuse_entry.video_max_pixels
            video_total_pixels = _metadata_or_input_value(
                inputs, metadata, "video_total_pixels"
            )
            if video_total_pixels is None:
                video_total_pixels = self.default_video_total_pixels
            if reuse_entry is not None and video_total_pixels is None:
                video_total_pixels = reuse_entry.video_total_pixels
            use_audio_in_video = _metadata_or_input_value(
                inputs, metadata, "use_audio_in_video"
            )
            if reuse_entry is not None and use_audio_in_video is None:
                use_audio_in_video = reuse_entry.use_audio_in_video
            video_seconds_per_chunk = _metadata_or_input_value(
                inputs, metadata, "video_seconds_per_chunk"
            )
            if reuse_entry is not None and video_seconds_per_chunk is None:
                video_seconds_per_chunk = reuse_entry.video_seconds_per_chunk
            video_position_id_per_seconds = _metadata_or_input_value(
                inputs, metadata, "video_position_id_per_seconds"
            )
            if reuse_entry is not None and video_position_id_per_seconds is None:
                video_position_id_per_seconds = (
                    reuse_entry.video_position_id_per_seconds
                )
            audio_from_video = False
            num_explicit_audios = 0
            resolved_video_fps = float(video_fps) if video_fps is not None else None
            resolved_video_max_frames = (
                int(video_max_frames) if video_max_frames is not None else None
            )
            resolved_video_min_pixels = (
                int(video_min_pixels) if video_min_pixels is not None else None
            )
            resolved_video_max_pixels = (
                int(video_max_pixels) if video_max_pixels is not None else None
            )
            resolved_video_total_pixels = (
                int(video_total_pixels) if video_total_pixels is not None else None
            )
            resolved_video_seconds_per_chunk = (
                float(video_seconds_per_chunk)
                if video_seconds_per_chunk is not None
                else None
            )
            resolved_video_position_id_per_seconds = (
                float(video_position_id_per_seconds)
                if video_position_id_per_seconds is not None
                else None
            )

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
                    fps=resolved_video_fps,
                    max_frames=resolved_video_max_frames,
                    min_pixels=resolved_video_min_pixels,
                    max_pixels=resolved_video_max_pixels,
                    total_pixels=resolved_video_total_pixels,
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
            raw_images = None
            raw_videos = None
            raw_audios = None
            images = []
            videos = []
            audios = []
            image_cache_key = None
            raw_audio_cache_key = None
            video_cache_key = None
            audio_target_sr = 16000
            video_fps = self.default_video_fps
            video_max_frames = self.default_video_max_frames
            video_min_pixels = self.default_video_min_pixels
            video_max_pixels = self.default_video_max_pixels
            video_total_pixels = self.default_video_total_pixels
            sampled_video_fps = None
            use_audio_in_video = None
            video_seconds_per_chunk = None
            video_position_id_per_seconds = None
            audio_from_video = False
            num_explicit_audios = 0
            resolved_video_fps = None
            resolved_video_max_frames = None
            resolved_video_min_pixels = None
            resolved_video_max_pixels = None
            resolved_video_total_pixels = None
            resolved_video_seconds_per_chunk = None
            resolved_video_position_id_per_seconds = None

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
        elif resolved_video_fps is not None:
            videos_kwargs["fps"] = resolved_video_fps
        if resolved_video_max_frames is not None:
            videos_kwargs["max_frames"] = resolved_video_max_frames
        if resolved_video_min_pixels is not None:
            videos_kwargs["min_pixels"] = resolved_video_min_pixels
        if resolved_video_max_pixels is not None:
            videos_kwargs["max_pixels"] = resolved_video_max_pixels
        if resolved_video_total_pixels is not None:
            videos_kwargs["total_pixels"] = resolved_video_total_pixels
        if use_audio_in_video is not None:
            videos_kwargs["use_audio_in_video"] = bool(use_audio_in_video)
        if resolved_video_seconds_per_chunk is not None:
            videos_kwargs["seconds_per_chunk"] = resolved_video_seconds_per_chunk
        if resolved_video_position_id_per_seconds is not None:
            videos_kwargs["position_id_per_seconds"] = float(
                resolved_video_position_id_per_seconds
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

        full_mm_inputs: dict[str, Any] = {
            "image": build_image_mm_inputs(hf_inputs),
            "audio": build_audio_mm_inputs(hf_inputs),
            "video": build_video_mm_inputs(hf_inputs),
        }
        if use_audio_in_video is not None:
            full_mm_inputs["video"]["use_audio_in_video"] = bool(use_audio_in_video)

        # Build encoder_inputs with cache_key for efficient caching.
        # Include preprocessing parameters that materially change encoder outputs.
        image_encoder_inputs = {
            **full_mm_inputs["image"],
            **full_mm_inputs["video"],
        }
        effective_video_fps: tuple[float, ...] | None = None
        if sampled_video_fps is not None:
            effective_video_fps = tuple(float(fps) for fps in sampled_video_fps)
        elif resolved_video_fps is not None:
            effective_video_fps = (resolved_video_fps,)

        contextual_video_cache_key = _contextualize_cache_key(
            video_cache_key,
            fps=effective_video_fps,
            max_frames=resolved_video_max_frames,
            min_pixels=resolved_video_min_pixels,
            max_pixels=resolved_video_max_pixels,
            total_pixels=resolved_video_total_pixels,
            seconds_per_chunk=resolved_video_seconds_per_chunk,
        )
        combined_cache_key = _combine_cache_keys(
            image_cache_key, contextual_video_cache_key
        )
        if combined_cache_key:
            image_encoder_inputs["cache_key"] = combined_cache_key

        audio_encoder_inputs = {**full_mm_inputs["audio"]}
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

        if session_id is not None and _has_media_refs(
            raw_images, raw_videos, raw_audios
        ):
            self.session_media.put(
                session_id,
                SessionMediaEntry(
                    raw_images=raw_images,
                    raw_videos=raw_videos,
                    raw_audios=raw_audios,
                    image_cache_key=combined_cache_key,
                    audio_cache_key=contextualized_audio_cache_key,
                    num_images=len(images),
                    num_videos=len(videos),
                    num_audios=num_audios_for_placeholder,
                    image_grid_thw=_metadata_cache_value(
                        full_mm_inputs["image"].get("image_grid_thw")
                    ),
                    video_grid_thw=_metadata_cache_value(
                        full_mm_inputs["video"].get("video_grid_thw")
                    ),
                    video_second_per_grid=_metadata_cache_value(
                        full_mm_inputs["video"].get("video_second_per_grid")
                    ),
                    audio_target_sr=audio_target_sr,
                    video_fps=resolved_video_fps,
                    video_max_frames=resolved_video_max_frames,
                    video_min_pixels=resolved_video_min_pixels,
                    video_max_pixels=resolved_video_max_pixels,
                    video_total_pixels=resolved_video_total_pixels,
                    use_audio_in_video=(
                        bool(use_audio_in_video)
                        if use_audio_in_video is not None
                        else None
                    ),
                    video_seconds_per_chunk=resolved_video_seconds_per_chunk,
                    video_position_id_per_seconds=(
                        resolved_video_position_id_per_seconds
                    ),
                ),
            )

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

        return self._finalize_state(
            payload,
            input_ids=input_ids,
            attention_mask=attention_mask,
            prompt_text=prompt_text,
            full_mm_inputs=full_mm_inputs,
            encoder_inputs=encoder_inputs,
        )
