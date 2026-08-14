# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 The CogView team, Tsinghua University & ZhipuAI and The HuggingFace Team. All rights reserved.
# Portions adapted and modified from inclusionAI/LLaDA2.0-Uni (Apache-2.0):
# decoder/utils.py, encoder/image_tokenizer.py, and scripts/image_edit.py at
# commit 3457030a9c737f77f38ad5ff657e7659243d3444.
"""Preprocessor for LLaDA2-Uni: tokenize text and prepare image inputs."""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any

import torch
from PIL import Image

from sglang_omni.models.llada2_uni.components.common import (
    load_llada2_image_token_offset,
    load_llada2_tokenizer,
    resolve_local_model_dir,
)
from sglang_omni.models.llada2_uni.config import (
    DEFAULT_THINKER_MAX_NEW_TOKENS,
    IMAGE_STAGE,
)
from sglang_omni.models.llada2_uni.payload_types import LLaDA2UniPipelineState
from sglang_omni.models.weight_loader import resolve_model_path
from sglang_omni.preprocessing.image import (
    compute_image_cache_key,
    ensure_image_list_async,
)
from sglang_omni.proto import StagePayload

# LLaDA2-Uni chat template tokens
ROLE_HUMAN = "<role>HUMAN</role>"
ROLE_ASSISTANT = "<role>ASSISTANT</role>"
ROLE_SYSTEM = "<role>SYSTEM</role>"
DEFAULT_SYSTEM_PROMPT = "detailed thinking off"
SYSTEM_PROMPT_T2I = "You are a text-to-image generation assistant."
EDIT_SYSTEM_PROMPT = "You are an image editing assistant."
UNCOND_TEXT = "<uncondition>"

# Image special token strings
SOI_TOKEN = "<|image|>"  # id=156901
EOI_TOKEN = "<|/image|>"  # id=156902
BOI_TOKEN = "<boi>"  # id=156904

# Internal-only placeholder replaced after image encoding. It is deliberately
# outside the vocabulary and is not the checkpoint-specific VQ token offset.
DUMMY_IMAGE_TOKEN_ID = -200

DEFAULT_IMAGE_HEIGHT = 1024
DEFAULT_IMAGE_WIDTH = 1024
DEFAULT_RESOLUTION_MULTIPLIER = 2

# Pixel budgets for image resize (single-image / multi-image)
SINGLE_IMAGE_MIN_PIXELS = 128 * 128
SINGLE_IMAGE_MAX_PIXELS = 800 * 800
MULTI_IMAGE_MIN_PIXELS = 128 * 128
MULTI_IMAGE_MAX_PIXELS = 448 * 448

logger = logging.getLogger(__name__)


def _positive_int(value: Any, *, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f"image_generation.{name} must be a positive integer")
    return value


def _resolve_native_output_size(
    image_generation: dict[str, Any],
) -> tuple[int, int]:
    size = image_generation.get("size")
    if size is None:
        height = _positive_int(
            image_generation.get("height", DEFAULT_IMAGE_HEIGHT), name="height"
        )
        width = _positive_int(
            image_generation.get("width", DEFAULT_IMAGE_WIDTH), name="width"
        )
        return height, width

    if not isinstance(size, str):
        raise ValueError("image_generation.size must use WIDTHxHEIGHT")
    dimensions = size.strip().lower().split("x")
    if len(dimensions) != 2:
        raise ValueError("image_generation.size must use WIDTHxHEIGHT")
    try:
        width = _positive_int(int(dimensions[0]), name="width")
        height = _positive_int(int(dimensions[1]), name="height")
    except (TypeError, ValueError) as exc:
        raise ValueError("image_generation.size must use WIDTHxHEIGHT") from exc
    return height, width


def resolve_native_image_grid(
    image_generation: dict[str, Any],
) -> tuple[int, int, int, int, int]:
    """Resolve requested output pixels to the decoder's semantic token grid."""
    mode = image_generation.get("mode", "normal")
    if mode != "normal":
        raise ValueError("image_generation.mode must be 'normal'")

    output_format = image_generation.get("format", "png")
    if not isinstance(output_format, str) or output_format.strip().lower() != "png":
        raise ValueError("image_generation.format must be 'png'")

    height, width = _resolve_native_output_size(image_generation)
    resolution_multiplier = _positive_int(
        image_generation.get("resolution_multiplier", DEFAULT_RESOLUTION_MULTIPLIER),
        name="resolution_multiplier",
    )
    pixel_stride = 16 * resolution_multiplier
    if height % pixel_stride or width % pixel_stride:
        raise ValueError(
            "image_generation width and height must be divisible by "
            f"{pixel_stride} for resolution_multiplier={resolution_multiplier}"
        )
    return (
        height // pixel_stride,
        width // pixel_stride,
        height,
        width,
        resolution_multiplier,
    )


def _resolve_edit_cfg_scales(
    image_generation: dict[str, Any],
) -> tuple[float, float]:
    if "cfg_text_scale" in image_generation:
        text_scale = float(image_generation["cfg_text_scale"])
    elif "cfg_scale" in image_generation:
        legacy_scale = float(image_generation["cfg_scale"])
        text_scale = 0.0 if legacy_scale == 1.0 else legacy_scale
    else:
        text_scale = 4.0
    image_scale = float(image_generation.get("cfg_image_scale", 0.0))
    if text_scale < 0.0 or image_scale < 0.0:
        raise ValueError("image editing CFG scales must be non-negative")
    return text_scale, image_scale


def _validate_native_edit_params(image_generation: dict[str, Any]) -> int:
    if image_generation.get("mode", "normal") != "normal":
        raise ValueError("image_generation.mode must be 'normal'")
    output_format = image_generation.get("format", "png")
    if not isinstance(output_format, str) or output_format.strip().lower() != "png":
        raise ValueError("image_generation.format must be 'png'")
    if any(key in image_generation for key in ("width", "height", "size")):
        raise ValueError("image editing follows source dimensions")
    return _positive_int(
        image_generation.get("resolution_multiplier", DEFAULT_RESOLUTION_MULTIPLIER),
        name="resolution_multiplier",
    )


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
    if total_tokens > max_seq_len:
        logger.info(
            "rejecting request %s: prompt %d + max_new_tokens %d = %d tokens "
            ">= max_seq_len %d",
            request_id,
            prompt_len,
            int(max_new_tokens),
            total_tokens,
            max_seq_len,
        )
        raise ValueError(
            f"Requested token count exceeds the model's maximum context length "
            f"of {max_seq_len} tokens. You requested a total of {total_tokens} "
            f"tokens: {prompt_len} tokens from the input messages and "
            f"{int(max_new_tokens)} tokens for the completion. Please reduce "
            f"the number of tokens in the input messages or the completion to "
            f"fit within the limit."
        )


def _compute_target_dims(
    height: int,
    width: int,
    min_pixels: int,
    max_pixels: int,
    factor: int,
) -> tuple[int, int]:
    """Scale dimensions to fit within [min_pixels, max_pixels], aligned to factor."""
    new_h = max(round(height / factor) * factor, factor)
    new_w = max(round(width / factor) * factor, factor)

    if new_h * new_w > max_pixels:
        scale = math.sqrt(max_pixels / (height * width))
        new_h = max(math.floor(height * scale / factor) * factor, factor)
        new_w = max(math.floor(width * scale / factor) * factor, factor)
    elif new_h * new_w < min_pixels:
        scale = math.sqrt(min_pixels / (height * width))
        new_h = math.ceil(height * scale / factor) * factor
        new_w = math.ceil(width * scale / factor) * factor

    return new_h, new_w


def _resize_and_center_crop(
    img: Image.Image,
    target_h: int,
    target_w: int,
    factor: int,
) -> Image.Image:
    """Resize a PIL Image to cover the target area, then center-crop to a factor-aligned size."""
    width, height = img.size
    scale = max(target_h / height, target_w / width)
    resize_h = int(round(height * scale))
    resize_w = int(round(width * scale))
    img = img.resize((resize_w, resize_h), resample=Image.BICUBIC)

    crop_h = max((resize_h // factor) * factor, target_h)
    crop_w = max((resize_w // factor) * factor, target_w)
    top = (resize_h - crop_h) // 2
    left = (resize_w - crop_w) // 2
    return img.crop((left, top, left + crop_w, top + crop_h))


def _resize_images(
    images: list[Image.Image],
    factor: int,
) -> list[Image.Image]:
    """Resize PIL Images to fit within pixel budgets, preserving aspect ratio."""
    if len(images) == 1:
        min_pixels, max_pixels = SINGLE_IMAGE_MIN_PIXELS, SINGLE_IMAGE_MAX_PIXELS
    else:
        min_pixels, max_pixels = MULTI_IMAGE_MIN_PIXELS, MULTI_IMAGE_MAX_PIXELS

    result = []
    for img in images:
        width, height = img.size
        target_h, target_w = _compute_target_dims(
            height, width, min_pixels, max_pixels, factor
        )
        result.append(_resize_and_center_crop(img, target_h, target_w, factor))
    return result


def _center_crop(image: Image.Image, crop_size: tuple[int, int]) -> Image.Image:
    crop_width, crop_height = crop_size
    width, height = image.size
    left = max(0, (width - crop_width) // 2)
    top = max(0, (height - crop_height) // 2)
    return image.crop((left, top, left + crop_width, top + crop_height)).resize(
        (crop_width, crop_height), Image.Resampling.LANCZOS
    )


def _generate_crop_sizes(
    num_patches: int,
    patch_size: int,
    max_ratio: float = 4.0,
) -> list[tuple[int, int]]:
    if max_ratio < 1.0:
        raise ValueError("max_ratio must be at least one")
    sizes: list[tuple[int, int]] = []
    width_patches, height_patches = num_patches, 1
    while width_patches > 0:
        if (
            max(width_patches, height_patches) / min(width_patches, height_patches)
            <= max_ratio
        ):
            sizes.append((width_patches * patch_size, height_patches * patch_size))
        if (height_patches + 1) * width_patches <= num_patches:
            height_patches += 1
        else:
            width_patches -= 1
    return sizes


def preprocess_image_edit(images: list[Image.Image], factor: int) -> list[Image.Image]:
    """Apply the reference generation crop budget to one edit source image."""
    crop_sizes = _generate_crop_sizes((512 // factor) ** 2, factor)
    cropped: list[Image.Image] = []
    for image in images:
        width, height = image.size
        crop_size = max(
            crop_sizes,
            key=lambda size: (
                min(size[0] / width, size[1] / height)
                / max(size[0] / width, size[1] / height)
            ),
        )
        cropped.append(_center_crop(image, crop_size))
    return cropped


def edit_image_pixel_values(
    images: list[Image.Image],
    *,
    patch_size: int,
    temporal_patch_size: int,
    merge_size: int,
    image_mean: list[float],
    image_std: list[float],
    rescale_factor: float,
) -> dict[str, torch.Tensor]:
    """Patchify edit inputs with the reference float32 operation order."""
    import torchvision.transforms.v2.functional as tv_functional

    patches: list[torch.Tensor] = []
    grids: list[list[int]] = []
    for image in images:
        if image.mode != "RGB":
            image = image.convert("RGB")
        tensor = tv_functional.to_dtype(
            tv_functional.to_image(image), dtype=torch.float32, scale=False
        )
        height, width = tensor.shape[-2:]
        tensor = tensor * rescale_factor
        mean = torch.tensor(image_mean, dtype=tensor.dtype).view(-1, 1, 1)
        std = torch.tensor(image_std, dtype=tensor.dtype).view(-1, 1, 1)
        tensor = (tensor - mean) / std
        if tensor.ndim == 3:
            tensor = tensor.unsqueeze(0)
        if tensor.shape[0] % temporal_patch_size:
            repetitions = temporal_patch_size - tensor.shape[0] % temporal_patch_size
            tensor = torch.cat(
                [tensor, tensor[-1:].repeat(repetitions, 1, 1, 1)], dim=0
            )
        grid_t = tensor.shape[0] // temporal_patch_size
        grid_h = height // patch_size
        grid_w = width // patch_size
        channels = tensor.shape[1]
        tensor = tensor.unsqueeze(0).view(
            1,
            grid_t,
            temporal_patch_size,
            channels,
            grid_h // merge_size,
            merge_size,
            patch_size,
            grid_w // merge_size,
            merge_size,
            patch_size,
        )
        tensor = tensor.permute(0, 1, 4, 7, 5, 8, 3, 2, 6, 9)
        patches.append(
            tensor.reshape(
                grid_t * grid_h * grid_w,
                channels * temporal_patch_size * patch_size * patch_size,
            )
        )
        grids.append([grid_t, grid_h, grid_w])
    return {
        "pixel_values": torch.cat(patches, dim=0),
        "image_grid_thw": torch.tensor(grids, dtype=torch.long),
    }


class LLaDA2Preprocessor:
    """Preprocessor for LLaDA2-Uni model (text + image)."""

    def __init__(self, model_path: str, max_seq_len: int | None = None):
        self._max_seq_len = max_seq_len
        self._model_dir = resolve_local_model_dir(model_path)
        self._tokenizer = load_llada2_tokenizer(model_path)
        self._image_token_offset = load_llada2_image_token_offset(model_path)

        # Load HF Qwen2VLImageProcessor (do_resize=False, crop handles sizing)
        from transformers import Qwen2VLImageProcessor

        tokenizer_path = str(Path(self._model_dir) / "image_tokenizer")

        try:
            self._image_processor = Qwen2VLImageProcessor.from_pretrained(
                tokenizer_path,
                local_files_only=True,
                do_resize=False,  # Disable resize, use manual crop instead
                merge_size=1,
            )
        except (OSError, ValueError, RuntimeError):
            if Path(model_path).exists():
                raise
            self._image_processor = Qwen2VLImageProcessor.from_pretrained(
                model_path,
                trust_remote_code=True,
                local_files_only=False,
                subfolder="image_tokenizer",
                do_resize=False,
                merge_size=1,
            )
            self._model_dir = str(
                resolve_model_path(model_path, local_files_only=False)
            )

        def _rescale_and_normalize(
            images,
            do_rescale,
            rescale_factor,
            do_normalize,
            image_mean,
            image_std,
        ):
            if do_rescale:
                images = images * rescale_factor
            if do_normalize:
                mean = torch.tensor(
                    image_mean, dtype=images.dtype, device=images.device
                ).view(-1, 1, 1)
                std = torch.tensor(
                    image_std, dtype=images.dtype, device=images.device
                ).view(-1, 1, 1)
                images = (images - mean) / std
            return images

        self._image_processor.rescale_and_normalize = _rescale_and_normalize
        self._merge_size = self._image_processor.merge_size
        self._factor = self._image_processor.patch_size * self._merge_size

        # Cache special token IDs
        self._eoi_id = self._tokenizer.convert_tokens_to_ids(EOI_TOKEN)
        self._boi_id = self._tokenizer.convert_tokens_to_ids(BOI_TOKEN)
        self._soi_id = self._tokenizer.convert_tokens_to_ids(SOI_TOKEN)

    async def __call__(self, payload: StagePayload) -> StagePayload:
        request = payload.request
        raw_inputs = request.inputs
        if isinstance(raw_inputs, list):
            messages = raw_inputs
            raw_images, image_counts_per_msg = self._extract_raw_images(messages)
        else:
            messages = raw_inputs.get("messages", [])
            raw_images = raw_inputs.get("images")
            if raw_images is None:
                raw_images, image_counts_per_msg = self._extract_raw_images(messages)
            else:
                image_counts_per_msg = None

        self._validate_messages(messages)
        request_metadata = (
            request.metadata if isinstance(request.metadata, dict) else {}
        )
        image_generation = request_metadata.get("image_generation")
        if image_generation is not None and not isinstance(image_generation, dict):
            raise ValueError("image_generation must be an object")

        output_modalities = request_metadata.get("output_modalities")
        if isinstance(output_modalities, str):
            output_modalities = (output_modalities,)
        wants_image = isinstance(output_modalities, (list, tuple, set)) and "image" in {
            str(modality).lower() for modality in output_modalities
        }
        task_kind = "chat"
        if wants_image:
            image_generation = image_generation or {}
            request_metadata["image_generation"] = image_generation
            task_kind = "edit" if raw_images else "t2i"
            if task_kind == "edit":
                _validate_native_edit_params(image_generation)
            else:
                resolve_native_image_grid(image_generation)

        image_cache_key = compute_image_cache_key(raw_images)

        images = await ensure_image_list_async(raw_images) if raw_images else []

        if task_kind == "edit":
            if len(images) != 1:
                raise ValueError("image editing requires exactly one source image")
            return self._build_edit_payload(
                payload,
                messages,
                images,
                request_metadata,
                image_generation,
            )

        encoder_inputs: dict[str, dict[str, Any]] = {}
        image_token_counts: list[int] = []
        image_parts_by_msg: dict[int, list[str]] = {}

        if images:
            cropped = _resize_images(images, self._factor)
            img_result = self._image_processor(images=cropped, return_tensors="pt")
            pixel_values = img_result["pixel_values"]
            image_grid_thw = img_result["image_grid_thw"]
            image_enc_inputs: dict[str, Any] = {
                "pixel_values": pixel_values,
                "image_grid_thw": image_grid_thw,
            }
            if image_cache_key:
                image_enc_inputs["cache_key"] = image_cache_key
            encoder_inputs[IMAGE_STAGE] = image_enc_inputs

            if image_counts_per_msg is None:
                last_user_idx = max(len(messages) - 1, 0)
                for i, m in enumerate(messages):
                    if m.get("role", "user") == "user":
                        last_user_idx = i
                image_counts_per_msg = [(last_user_idx, len(images))]

            img_idx = 0
            for msg_idx, count in image_counts_per_msg:
                parts: list[str] = []
                for _ in range(count):
                    t, h, w = image_grid_thw[img_idx].tolist()
                    h_token = f"<|reserved_token_{h}|>"
                    w_token = f"<|reserved_token_{w}|>"
                    num_image_tokens = t * h * w
                    img_header = f"{SOI_TOKEN}{h_token}{w_token}{BOI_TOKEN}"
                    image_token_counts.append(num_image_tokens)
                    parts.extend([img_header, EOI_TOKEN])
                    img_idx += 1
                image_parts_by_msg[msg_idx] = parts
        else:
            encoder_inputs[IMAGE_STAGE] = {"_skip": True, "_result": {}}

        text_prompt = self._build_prompt(
            messages,
            image_parts_by_msg=image_parts_by_msg,
            task_kind=task_kind,
        )
        input_ids = self._tokenizer.encode(text_prompt, add_special_tokens=False)

        if image_token_counts:
            input_ids = self._insert_image_placeholders(input_ids, image_token_counts)

        generation_state: dict[str, Any] = {}
        if task_kind == "t2i":
            assert isinstance(image_generation, dict)
            grid_h, grid_w, height, width, resolution_multiplier = (
                resolve_native_image_grid(image_generation)
            )
            input_ids.extend(self._build_image_header_ids(grid_h, grid_w))
            cfg_scale = float(image_generation.get("cfg_scale", 4.0))
            if cfg_scale <= 0.0:
                raise ValueError("image_generation.cfg_scale must be positive")
            generation_state = {
                "image_grid": {"height": grid_h, "width": grid_w},
                "output_size": {"height": height, "width": width},
                "resolution_multiplier": resolution_multiplier,
            }
            if cfg_scale > 1.0:
                generation_state["cfg"] = {
                    "unconditional_input_ids": self._build_t2i_unconditional_ids(
                        grid_h, grid_w
                    ),
                    "scale": cfg_scale,
                    "rescale": float(image_generation.get("cfg_rescale", 0.7)),
                }

        input_ids_tensor = torch.tensor([input_ids], dtype=torch.long)

        max_new_tokens = request.params.get(
            "max_new_tokens", DEFAULT_THINKER_MAX_NEW_TOKENS
        )
        if task_kind == "t2i":
            max_new_tokens = grid_h * grid_w

        validate_prompt_seq_len(
            input_ids_tensor,
            max_seq_len=self._max_seq_len,
            max_new_tokens=max_new_tokens,
            request_id=payload.request_id,
        )

        prompt = {"input_ids": input_ids_tensor}

        state = LLaDA2UniPipelineState(
            prompt=prompt,
            encoder_inputs=encoder_inputs,
            generation_state=generation_state,
            request_metadata=request_metadata,
            task_kind=task_kind,
            image_token_offset=self._image_token_offset,
        )
        return StagePayload(
            request_id=payload.request_id,
            request=payload.request,
            data=state.to_dict(),
        )

    @staticmethod
    def _extract_raw_images(
        messages: list[dict[str, Any]],
    ) -> tuple[list[Any], list[tuple[int, int]]]:
        """Return (images, image_counts_per_msg) with per-message image counts."""
        raw_images: list[Any] = []
        image_counts_per_msg: list[tuple[int, int]] = []
        for msg_idx, msg in enumerate(messages):
            msg_count = 0
            content = msg.get("content", "")
            if isinstance(content, list):
                for item in content:
                    if not isinstance(item, dict):
                        continue
                    if item.get("type") == "image_url":
                        url = item.get("image_url", {})
                        if isinstance(url, dict):
                            url = url.get("url", "")
                        if url:
                            raw_images.append(url)
                            msg_count += 1
                    elif item.get("type") == "image":
                        img = item.get("image", "")
                        if img:
                            raw_images.append(img)
                            msg_count += 1
            if msg_count > 0:
                image_counts_per_msg.append((msg_idx, msg_count))
        return raw_images, image_counts_per_msg

    @staticmethod
    def _validate_messages(messages: list[dict[str, Any]]) -> None:
        if not isinstance(messages, list):
            raise ValueError("Preprocessing expects a list of chat messages")
        for message in messages:
            if not isinstance(message, dict):
                raise ValueError("Each message must be a dict with role/content")

    def _build_prompt(
        self,
        messages: list[dict[str, Any]],
        image_parts_by_msg: dict[int, list[str]] | None = None,
        task_kind: str = "chat",
    ) -> str:
        """Build LLaDA2-Uni chat format prompt.

        Image blocks are inserted at the start of their originating message's
        content via *image_parts_by_msg* (message index -> header/footer tokens).
        """
        parts: list[str] = []
        system_prompt = (
            SYSTEM_PROMPT_T2I if task_kind == "t2i" else DEFAULT_SYSTEM_PROMPT
        )
        parts.append(f"{ROLE_SYSTEM} {system_prompt} ")

        for msg_idx, msg in enumerate(messages):
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if role == "system":
                continue

            role_tag = ROLE_HUMAN if role == "user" else ROLE_ASSISTANT

            img_prefix = ""
            if image_parts_by_msg and msg_idx in image_parts_by_msg:
                img_prefix = "".join(image_parts_by_msg[msg_idx])

            if isinstance(content, str):
                parts.append(f"{role_tag}{img_prefix}{content}")
            elif isinstance(content, list):
                text_parts: list[str] = []
                for item in content:
                    if isinstance(item, dict):
                        item_type = item.get("type", "text")
                        if item_type == "text":
                            text_parts.append(item.get("text", ""))
                    elif isinstance(item, str):
                        text_parts.append(item)
                parts.append(f"{role_tag}{img_prefix}{''.join(text_parts)}")
            else:
                parts.append(f"{role_tag}{img_prefix}{content}")

        parts.append(ROLE_ASSISTANT)
        return "".join(parts)

    def _build_image_header_ids(self, grid_h: int, grid_w: int) -> list[int]:
        header = [self._soi_id]
        header.extend(
            self._tokenizer.encode(
                f"<|reserved_token_{grid_h}|>", add_special_tokens=False
            )
        )
        header.extend(
            self._tokenizer.encode(
                f"<|reserved_token_{grid_w}|>", add_special_tokens=False
            )
        )
        header.append(self._boi_id)
        return header

    def _build_t2i_unconditional_ids(self, grid_h: int, grid_w: int) -> list[int]:
        prompt = (
            f"{ROLE_SYSTEM} {SYSTEM_PROMPT_T2I} "
            f"{ROLE_HUMAN}{UNCOND_TEXT}{ROLE_ASSISTANT}"
        )
        input_ids = self._tokenizer.encode(prompt, add_special_tokens=False)
        input_ids.extend(self._build_image_header_ids(grid_h, grid_w))
        return input_ids

    @staticmethod
    def _extract_user_instruction(messages: list[dict[str, Any]]) -> str:
        for message in reversed(messages):
            if message.get("role", "user") != "user":
                continue
            content = message.get("content", "")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                parts = []
                for item in content:
                    if isinstance(item, str):
                        parts.append(item)
                    elif isinstance(item, dict) and item.get("type", "text") == "text":
                        parts.append(str(item.get("text", "")))
                return "".join(parts)
        return ""

    def _build_edit_payload(
        self,
        payload: StagePayload,
        messages: list[dict[str, Any]],
        images: list[Image.Image],
        request_metadata: dict[str, Any],
        image_generation: dict[str, Any],
    ) -> StagePayload:
        instruction = self._extract_user_instruction(messages)
        if not instruction.strip():
            raise ValueError("image editing requires a non-empty instruction")
        resolution_multiplier = _validate_native_edit_params(image_generation)

        cropped = preprocess_image_edit(images, self._factor)
        processor = self._image_processor
        image_inputs = edit_image_pixel_values(
            cropped,
            patch_size=processor.patch_size,
            temporal_patch_size=processor.temporal_patch_size,
            merge_size=processor.merge_size,
            image_mean=processor.image_mean,
            image_std=processor.image_std,
            rescale_factor=processor.rescale_factor,
        )
        image_grid_thw = image_inputs["image_grid_thw"]
        grid_t, grid_h, grid_w = (int(value) for value in image_grid_thw[0].tolist())
        num_source_tokens = grid_t * grid_h * grid_w

        source_block = (
            f"{SOI_TOKEN}<|reserved_token_{grid_h}|>"
            f"<|reserved_token_{grid_w}|>{BOI_TOKEN}{EOI_TOKEN}"
        )
        prompt = (
            f"{ROLE_SYSTEM} {EDIT_SYSTEM_PROMPT} "
            f"{ROLE_HUMAN}{source_block}{instruction}{ROLE_ASSISTANT}"
        )
        conditional = self._tokenizer.encode(prompt, add_special_tokens=False)
        conditional = self._insert_image_placeholders(conditional, [num_source_tokens])
        conditional.extend(self._build_image_header_ids(grid_h, grid_w))

        text_scale, image_scale = _resolve_edit_cfg_scales(image_generation)
        cfg: dict[str, Any] = {}
        if text_scale > 0.0 or image_scale > 0.0:
            unconditional_prompt = (
                f"{ROLE_SYSTEM} {EDIT_SYSTEM_PROMPT} "
                f"{ROLE_HUMAN}{source_block}{UNCOND_TEXT}{ROLE_ASSISTANT}"
            )
            unconditional = self._tokenizer.encode(
                unconditional_prompt, add_special_tokens=False
            )
            unconditional = self._insert_image_placeholders(
                unconditional, [num_source_tokens]
            )
            unconditional.extend(self._build_image_header_ids(grid_h, grid_w))
            cfg = {
                "unconditional_input_ids": unconditional,
                "text_scale": text_scale,
                "rescale": float(image_generation.get("cfg_rescale", 0.7)),
            }
        if image_scale > 0.0:
            no_image_prompt = (
                f"{ROLE_SYSTEM} {EDIT_SYSTEM_PROMPT} "
                f"{ROLE_HUMAN}{SOI_TOKEN}{instruction}{ROLE_ASSISTANT}"
            )
            no_image = self._tokenizer.encode(no_image_prompt, add_special_tokens=False)
            no_image.extend(self._build_image_header_ids(grid_h, grid_w))
            cfg["no_image_input_ids"] = no_image
            cfg["image_scale"] = image_scale

        input_ids_tensor = torch.tensor([conditional], dtype=torch.long)
        validate_prompt_seq_len(
            input_ids_tensor,
            max_seq_len=self._max_seq_len,
            max_new_tokens=grid_h * grid_w,
            request_id=payload.request_id,
        )
        state = LLaDA2UniPipelineState(
            prompt={"input_ids": input_ids_tensor},
            encoder_inputs={IMAGE_STAGE: image_inputs},
            generation_state={
                "image_grid": {"height": grid_h, "width": grid_w},
                "output_size": {
                    "height": grid_h * 16 * resolution_multiplier,
                    "width": grid_w * 16 * resolution_multiplier,
                },
                **({"cfg": cfg} if cfg else {}),
            },
            request_metadata=request_metadata,
            task_kind="edit",
            image_token_offset=self._image_token_offset,
        )
        return StagePayload(
            request_id=payload.request_id,
            request=payload.request,
            data=state.to_dict(),
        )

    def _insert_image_placeholders(
        self,
        input_ids: list[int],
        image_token_counts: list[int],
    ) -> list[int]:
        new_ids: list[int] = []
        cursor = 0
        search_start = 0

        for image_idx, num_tokens in enumerate(image_token_counts):
            boi_idx = next(
                (
                    i
                    for i in range(search_start, len(input_ids))
                    if input_ids[i] == self._boi_id
                ),
                None,
            )
            if boi_idx is None:
                raise ValueError(
                    f"Expected image block {image_idx} but no matching <boi> token was found"
                )

            eoi_idx = next(
                (
                    i
                    for i in range(boi_idx + 1, len(input_ids))
                    if input_ids[i] == self._eoi_id
                ),
                None,
            )
            if eoi_idx is None:
                raise ValueError(
                    f"No <eoi> token found after <boi> for image block {image_idx}"
                )

            new_ids.extend(input_ids[cursor : boi_idx + 1])
            new_ids.extend([DUMMY_IMAGE_TOKEN_ID] * num_tokens)
            cursor = eoi_idx
            search_start = eoi_idx + 1

        new_ids.extend(input_ids[cursor:])
        return new_ids
