# SPDX-License-Identifier: Apache-2.0
"""Text and vision preprocessing for Cosmos3-Nano."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any

import torch
from transformers import AutoProcessor, AutoTokenizer

from sglang_omni.models.cosmos3.payload_types import Cosmos3PipelineState
from sglang_omni.preprocessing import (
    compute_image_cache_key,
    compute_video_cache_key,
    ensure_image_list_async,
    ensure_video_list_async,
)
from sglang_omni.preprocessing.text import ensure_chat_template
from sglang_omni.proto import StagePayload

logger = logging.getLogger(__name__)

DEFAULT_MAX_NEW_TOKENS = 2048
VISION_STAGE = "vision_encoder"


def load_cosmos3_tokenizer(model_path: str) -> Any:
    """Load tokenizer assets without hydrating the checkpoint weights."""

    local_only = Path(model_path).exists()
    if not local_only:
        try:
            return AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True,
                local_files_only=True,
            )
        except (OSError, ValueError):
            pass
    return AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        local_files_only=local_only,
    )


def load_cosmos3_processor(model_path: str) -> Any:
    """Load the Qwen3-VL processor shipped with Cosmos3-Nano."""

    local_only = Path(model_path).exists()
    if not local_only:
        try:
            return AutoProcessor.from_pretrained(
                model_path,
                trust_remote_code=True,
                local_files_only=True,
            )
        except (OSError, ValueError):
            pass
    return AutoProcessor.from_pretrained(
        model_path,
        trust_remote_code=True,
        local_files_only=local_only,
    )


def _normalize_text_messages(inputs: Any) -> list[dict[str, str]]:
    if isinstance(inputs, dict):
        if "messages" not in inputs:
            raise ValueError("Cosmos3 text preprocessing expects a messages field")
        inputs = inputs["messages"]

    if not isinstance(inputs, list) or not inputs:
        raise ValueError(
            "Cosmos3 text preprocessing expects a non-empty list of chat messages"
        )

    messages: list[dict[str, str]] = []
    for index, message in enumerate(inputs):
        if not isinstance(message, dict):
            raise TypeError(f"Message {index} must be a dict with role/content")
        role = message.get("role")
        content = message.get("content")
        if not isinstance(role, str) or not role:
            raise ValueError(f"Message {index} must have a non-empty string role")
        if not isinstance(content, str):
            raise TypeError(
                f"Message {index} content must be a string; multimodal content "
                "must be passed with top-level images/videos"
            )
        messages.append({"role": role, "content": content})
    return messages


def _flatten_single_batch(value: Any, *, name: str) -> torch.Tensor:
    tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
    if tensor.ndim == 2:
        if tensor.shape[0] != 1:
            raise ValueError(f"Tokenizer returned batched {name}; expected one prompt")
        tensor = tensor[0]
    if tensor.ndim != 1:
        raise ValueError(
            f"Tokenizer returned invalid {name} shape {tuple(tensor.shape)}"
        )
    return tensor.to(dtype=torch.long)


def validate_prompt_seq_len(
    input_ids: torch.Tensor,
    *,
    max_seq_len: int | None,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    request_id: str | None = None,
) -> None:
    """Apply the same prompt budget contract as the existing AR pipelines."""

    if max_seq_len is None:
        return
    prompt_len = int(input_ids.numel())
    if prompt_len >= max_seq_len:
        logger.info(
            "rejecting request %s: prompt %d tokens >= max_seq_len %d",
            request_id,
            prompt_len,
            max_seq_len,
        )
        raise ValueError(
            f"The input ({prompt_len} tokens) is longer than the model's "
            f"context length ({max_seq_len} tokens)."
        )

    total_tokens = prompt_len + int(max_new_tokens)
    if total_tokens >= max_seq_len:
        logger.info(
            "rejecting request %s: prompt %d + max_new_tokens %d >= max_seq_len %d",
            request_id,
            prompt_len,
            int(max_new_tokens),
            max_seq_len,
        )
        raise ValueError(
            "Requested token count exceeds the model's maximum context length "
            f"of {max_seq_len} tokens. You requested {prompt_len} prompt tokens "
            f"and {int(max_new_tokens)} completion tokens."
        )


class Cosmos3TextPreprocessor:
    """Build the Cosmos3 prompt and standalone vision-encoder inputs on CPU."""

    def __init__(
        self,
        model_path: str,
        max_seq_len: int | None = None,
        *,
        tokenizer: Any | None = None,
        processor: Any | None = None,
        enable_vision: bool = True,
    ) -> None:
        self.model_path = model_path
        self.max_seq_len = max_seq_len
        self.enable_vision = enable_vision
        self.processor = processor
        if self.processor is None and tokenizer is None and enable_vision:
            self.processor = load_cosmos3_processor(model_path)
        self.tokenizer = tokenizer or (
            self.processor.tokenizer
            if self.processor is not None
            else load_cosmos3_tokenizer(model_path)
        )
        ensure_chat_template(self.tokenizer, model_path=model_path)
        if not getattr(self.tokenizer, "chat_template", None):
            raise ValueError(
                f"Tokenizer for {model_path!r} does not define a chat template"
            )

    def _tokenize_messages(self, inputs: Any) -> tuple[torch.Tensor, torch.Tensor, str]:
        messages = _normalize_text_messages(inputs)
        prompt_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        encoded = self.tokenizer(
            prompt_text,
            add_special_tokens=False,
            return_tensors="pt",
        )
        input_ids = _flatten_single_batch(encoded["input_ids"], name="input_ids")
        raw_attention_mask = encoded.get("attention_mask")
        attention_mask = (
            torch.ones_like(input_ids)
            if raw_attention_mask is None
            else _flatten_single_batch(raw_attention_mask, name="attention_mask")
        )
        if attention_mask.shape != input_ids.shape:
            raise ValueError("Tokenizer returned mismatched input_ids/attention_mask")
        return input_ids, attention_mask, prompt_text

    @staticmethod
    def _multimodal_messages(
        messages: list[dict[str, str]],
        *,
        num_images: int,
        num_videos: int,
    ) -> list[dict[str, Any]]:
        if num_images == 0 and num_videos == 0:
            return messages

        result: list[dict[str, Any]] = []
        target = max(
            (
                index
                for index, message in enumerate(messages)
                if message["role"] == "user"
            ),
            default=len(messages) - 1,
        )
        for index, message in enumerate(messages):
            if index != target:
                result.append(message)
                continue
            content: list[dict[str, str]] = []
            content.extend({"type": "image"} for _ in range(num_images))
            content.extend({"type": "video"} for _ in range(num_videos))
            content.append({"type": "text", "text": message["content"]})
            result.append({"role": message["role"], "content": content})
        return result

    async def _tokenize_multimodal(
        self,
        inputs: dict[str, Any],
    ) -> tuple[dict[str, torch.Tensor], str, str | None]:
        if self.processor is None:
            raise ValueError("Cosmos3 media inputs require the checkpoint processor")

        raw_images = inputs.get("images")
        raw_videos = inputs.get("videos") or inputs.get("video")
        video_fps = inputs.get("video_fps")
        video_max_frames = inputs.get("video_max_frames")
        video_min_pixels = inputs.get("video_min_pixels")
        video_max_pixels = inputs.get("video_max_pixels")
        video_total_pixels = inputs.get("video_total_pixels")
        image_cache_key = compute_image_cache_key(raw_images)
        video_cache_key = compute_video_cache_key(raw_videos)

        images, video_result = await asyncio.gather(
            ensure_image_list_async(raw_images),
            ensure_video_list_async(
                raw_videos,
                fps=float(video_fps) if video_fps is not None else None,
                max_frames=(
                    int(video_max_frames) if video_max_frames is not None else None
                ),
                min_pixels=(
                    int(video_min_pixels) if video_min_pixels is not None else None
                ),
                max_pixels=(
                    int(video_max_pixels) if video_max_pixels is not None else None
                ),
                total_pixels=(
                    int(video_total_pixels) if video_total_pixels is not None else None
                ),
            ),
        )
        videos, sampled_fps, _ = video_result
        messages = _normalize_text_messages(inputs)
        messages_mm = self._multimodal_messages(
            messages,
            num_images=len(images),
            num_videos=len(videos),
        )
        prompt_text = self.processor.apply_chat_template(
            messages_mm,
            tokenize=False,
            add_generation_prompt=True,
        )
        processor_kwargs: dict[str, Any] = {}
        if videos:
            videos_kwargs: dict[str, Any] = {"device": "cpu"}
            if sampled_fps:
                videos_kwargs["fps"] = (
                    sampled_fps[0] if len(sampled_fps) == 1 else sampled_fps
                )
            elif video_fps is not None:
                videos_kwargs["fps"] = float(video_fps)
            if video_max_frames is not None:
                videos_kwargs["max_frames"] = int(video_max_frames)
            if video_min_pixels is not None:
                videos_kwargs["min_pixels"] = int(video_min_pixels)
            if video_max_pixels is not None:
                videos_kwargs["max_pixels"] = int(video_max_pixels)
            if video_total_pixels is not None:
                videos_kwargs["total_pixels"] = int(video_total_pixels)
            processor_kwargs["videos_kwargs"] = videos_kwargs

        encoded = self.processor(
            text=prompt_text,
            images=images or None,
            videos=videos or None,
            add_special_tokens=False,
            return_tensors="pt",
            **processor_kwargs,
        )
        if video_cache_key is not None:
            effective_fps = tuple(sampled_fps or ()) or (
                (float(video_fps),) if video_fps is not None else None
            )
            video_cache_key = "|".join(
                str(part)
                for part in (
                    video_cache_key,
                    f"fps={effective_fps}",
                    f"max_frames={video_max_frames}",
                    f"min_pixels={video_min_pixels}",
                    f"max_pixels={video_max_pixels}",
                    f"total_pixels={video_total_pixels}",
                )
            )
        cache_parts = [part for part in (image_cache_key, video_cache_key) if part]
        return dict(encoded), prompt_text, "|".join(cache_parts) or None

    @staticmethod
    def _use_pretokenized_inputs(inputs: Any) -> bool:
        return (
            isinstance(inputs, list)
            and bool(inputs)
            and all(isinstance(token_id, int) for token_id in inputs)
        )

    async def __call__(self, payload: StagePayload) -> StagePayload:
        inputs = payload.request.inputs
        if self._use_pretokenized_inputs(inputs):
            input_ids = torch.tensor(inputs, dtype=torch.long)
            attention_mask = torch.ones_like(input_ids)
            prompt_text = ""
            mm_token_type_ids = torch.zeros_like(input_ids)
            mm_inputs: dict[str, Any] = {}
            vision_inputs: dict[str, Any] = {"_skip": True, "_result": {}}
        else:
            if not isinstance(inputs, dict):
                inputs = {"messages": inputs}
            for key in (
                "images",
                "videos",
                "video_fps",
                "video_max_frames",
                "video_min_pixels",
                "video_max_pixels",
                "video_total_pixels",
            ):
                if (
                    inputs.get(key) is None
                    and payload.request.metadata.get(key) is not None
                ):
                    inputs[key] = payload.request.metadata[key]

            if inputs.get("audios") or inputs.get("audio"):
                raise ValueError("Cosmos3 audio preprocessing is not implemented yet")
            has_media = bool(
                inputs.get("images") or inputs.get("videos") or inputs.get("video")
            )
            if has_media:
                if not self.enable_vision:
                    raise ValueError(
                        "Cosmos3 text-only pipeline does not accept image/video input"
                    )
                encoded, prompt_text, cache_key = await self._tokenize_multimodal(
                    inputs
                )
                input_ids = _flatten_single_batch(
                    encoded["input_ids"], name="input_ids"
                )
                raw_attention_mask = encoded.get("attention_mask")
                attention_mask = (
                    torch.ones_like(input_ids)
                    if raw_attention_mask is None
                    else _flatten_single_batch(
                        raw_attention_mask, name="attention_mask"
                    )
                )
                raw_token_types = encoded.get("mm_token_type_ids")
                if raw_token_types is None:
                    raise ValueError(
                        "Cosmos3 processor did not return mm_token_type_ids"
                    )
                mm_token_type_ids = _flatten_single_batch(
                    raw_token_types,
                    name="mm_token_type_ids",
                )
                mm_inputs = {
                    key: encoded[key]
                    for key in ("image_grid_thw", "video_grid_thw")
                    if isinstance(encoded.get(key), torch.Tensor)
                }
                vision_inputs = {
                    key: encoded[key]
                    for key in (
                        "pixel_values",
                        "image_grid_thw",
                        "pixel_values_videos",
                        "video_grid_thw",
                    )
                    if isinstance(encoded.get(key), torch.Tensor)
                }
                if cache_key is not None:
                    vision_inputs["cache_key"] = cache_key
            else:
                input_ids, attention_mask, prompt_text = self._tokenize_messages(inputs)
                mm_token_type_ids = torch.zeros_like(input_ids)
                mm_inputs = {}
                vision_inputs = {"_skip": True, "_result": {}}

        max_new_tokens = payload.request.params.get(
            "max_new_tokens", DEFAULT_MAX_NEW_TOKENS
        )
        if max_new_tokens is None:
            max_new_tokens = DEFAULT_MAX_NEW_TOKENS
        validate_prompt_seq_len(
            input_ids,
            max_seq_len=self.max_seq_len,
            max_new_tokens=int(max_new_tokens),
            request_id=payload.request_id,
        )

        state = Cosmos3PipelineState(
            prompt={
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "prompt_text": prompt_text,
                "mm_token_type_ids": mm_token_type_ids,
            },
            mm_inputs=mm_inputs,
            encoder_inputs={VISION_STAGE: vision_inputs},
            stream_state={"token_ids": [], "text": ""},
        )
        payload.data = state.to_dict()
        payload.request.inputs = None
        for key in ("images", "videos"):
            payload.request.metadata.pop(key, None)
        return payload


__all__ = [
    "DEFAULT_MAX_NEW_TOKENS",
    "Cosmos3TextPreprocessor",
    "load_cosmos3_processor",
    "load_cosmos3_tokenizer",
    "validate_prompt_seq_len",
]
