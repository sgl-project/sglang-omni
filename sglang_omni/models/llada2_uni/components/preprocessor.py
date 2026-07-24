# SPDX-License-Identifier: Apache-2.0
"""Preprocessor for LLaDA2-Uni: tokenize text and prepare image inputs."""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any

import torch
from PIL import Image

from sglang_omni.models.llada2_uni.components.common import (
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
T2I_SYSTEM_PROMPT = "You are a text-to-image generation assistant."
T2I_UNCONDITION_PROMPT = "<uncondition>"

# Image special token strings
SOI_TOKEN = "<|image|>"  # id=156901
EOI_TOKEN = "<|/image|>"  # id=156902
BOI_TOKEN = "<boi>"  # id=156904

IMAGE_TOKEN_OFFSET = 157184  # VQ codebook indices are offset by this value
DUMMY_IMAGE_TOKEN_ID = IMAGE_TOKEN_OFFSET  # <IMAGE0>, used as placeholder

DEFAULT_IMAGE_OUTPUT_WIDTH = 1024
DEFAULT_IMAGE_OUTPUT_HEIGHT = 1024
DEFAULT_IMAGE_GENERATION_STEPS = 16
DEFAULT_IMAGE_BLOCK_LENGTH = 32
DEFAULT_IMAGE_CFG_SCALE = 4.0
DEFAULT_IMAGE_GEN_LENGTH = 1088
DEFAULT_IMAGE_DECODER_STEPS = 8
DEFAULT_IMAGE_RESOLUTION_MULTIPLIER = 2
DEFAULT_IMAGE_DECODE_MODE = "decoder-turbo"
DEFAULT_IMAGE_FORMAT = "png"
NORMAL_IMAGE_DECODER_STEPS = 50

# Pixel budgets for image resize (single-image / multi-image)
SINGLE_IMAGE_MIN_PIXELS = 128 * 128
SINGLE_IMAGE_MAX_PIXELS = 800 * 800
MULTI_IMAGE_MIN_PIXELS = 128 * 128
MULTI_IMAGE_MAX_PIXELS = 448 * 448

logger = logging.getLogger(__name__)


def _normalize_modalities(value: Any) -> set[str]:
    if value is None:
        return {"text"}
    if isinstance(value, str):
        return {value.lower()}
    if isinstance(value, (list, tuple, set)):
        return {str(item).lower() for item in value}
    return {"text"}


def _is_image_output_request(request: Any) -> bool:
    metadata = getattr(request, "metadata", {}) or {}
    return "image" in _normalize_modalities(metadata.get("output_modalities"))


def _get_config_int(
    config: dict[str, Any],
    *names: str,
    default: int,
    minimum: int = 1,
) -> int:
    value = default
    for name in names:
        if config.get(name) is not None:
            value = config[name]
            break
    value = int(value)
    if value < minimum:
        raise ValueError(
            f"Image generation parameter {names[0]!r} must be >= {minimum}"
        )
    return value


def _get_config_float(
    config: dict[str, Any],
    name: str,
    *,
    default: float,
) -> float:
    value = config.get(name, default)
    return float(value)


def _get_image_format(config: dict[str, Any]) -> str:
    image_format = str(
        config.get("format", config.get("response_format", DEFAULT_IMAGE_FORMAT))
    ).lower()
    if image_format == "jpg":
        image_format = "jpeg"
    if image_format not in {"png", "jpeg"}:
        raise ValueError("LLaDA2-Uni image output format must be 'png' or 'jpeg'.")
    return image_format


def _get_decode_mode(config: dict[str, Any]) -> str:
    decode_mode = str(config.get("decode_mode", DEFAULT_IMAGE_DECODE_MODE)).lower()
    if decode_mode == "turbo":
        decode_mode = "decoder-turbo"
    if decode_mode == "decoder":
        decode_mode = "normal"
    if decode_mode not in {"normal", "decoder-turbo"}:
        raise ValueError(
            "LLaDA2-Uni image decode_mode must be 'normal' or 'decoder-turbo'."
        )
    return decode_mode


def _extract_last_user_text(messages: list[dict[str, Any]]) -> str:
    text = ""
    for msg in messages:
        if msg.get("role", "user") != "user":
            continue
        content = msg.get("content", "")
        if isinstance(content, str):
            text = content
        elif isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, dict):
                    if item.get("type", "text") == "text":
                        parts.append(str(item.get("text", "")))
                elif isinstance(item, str):
                    parts.append(item)
            text = "".join(parts)
        else:
            text = str(content)
    return text


def build_image_generation_config(request: Any) -> dict[str, Any]:
    """Build LLaDA2 text-to-image request metadata from API image config."""
    metadata = getattr(request, "metadata", {}) or {}
    params = getattr(request, "params", {}) or {}
    raw_config = metadata.get("image_config") or {}
    if not isinstance(raw_config, dict):
        raise ValueError("image output configuration must be a JSON object")

    resolution_multiplier = _get_config_int(
        raw_config,
        "resolution_multiplier",
        default=DEFAULT_IMAGE_RESOLUTION_MULTIPLIER,
    )
    width = _get_config_int(
        raw_config,
        "width",
        "image_w",
        "w",
        default=DEFAULT_IMAGE_OUTPUT_WIDTH,
    )
    height = _get_config_int(
        raw_config,
        "height",
        "image_h",
        "h",
        default=DEFAULT_IMAGE_OUTPUT_HEIGHT,
    )

    token_factor = 16 * resolution_multiplier
    if height % token_factor != 0 or width % token_factor != 0:
        raise ValueError(
            "LLaDA2-Uni image output width and height must be divisible by "
            f"{token_factor} for resolution_multiplier={resolution_multiplier}."
        )

    token_grid_h = height // token_factor
    token_grid_w = width // token_factor
    num_image_tokens = token_grid_h * token_grid_w

    requested_gen_length = raw_config.get("gen_length")
    if requested_gen_length is None:
        requested_gen_length = params.get("max_new_tokens")
    gen_length = (
        max(num_image_tokens, DEFAULT_IMAGE_GEN_LENGTH)
        if requested_gen_length is None
        else int(requested_gen_length)
    )
    if gen_length < num_image_tokens:
        raise ValueError(
            "LLaDA2-Uni image output gen_length must be at least "
            f"{num_image_tokens} for {width}x{height} output."
        )

    decode_mode = _get_decode_mode(raw_config)
    default_decoder_steps = (
        DEFAULT_IMAGE_DECODER_STEPS
        if decode_mode == "decoder-turbo"
        else NORMAL_IMAGE_DECODER_STEPS
    )

    image_generation: dict[str, Any] = {
        "type": "image",
        "width": width,
        "height": height,
        "token_grid_h": token_grid_h,
        "token_grid_w": token_grid_w,
        "num_image_tokens": num_image_tokens,
        "gen_length": gen_length,
        "steps": _get_config_int(
            raw_config,
            "steps",
            default=DEFAULT_IMAGE_GENERATION_STEPS,
        ),
        "block_length": _get_config_int(
            raw_config,
            "block_length",
            default=DEFAULT_IMAGE_BLOCK_LENGTH,
        ),
        "cfg_scale": _get_config_float(
            raw_config,
            "cfg_scale",
            default=DEFAULT_IMAGE_CFG_SCALE,
        ),
        "decoder_steps": _get_config_int(
            raw_config,
            "decoder_steps",
            default=default_decoder_steps,
        ),
        "resolution_multiplier": resolution_multiplier,
        "decode_mode": decode_mode,
        "format": _get_image_format(raw_config),
        "image_token_offset": IMAGE_TOKEN_OFFSET,
    }
    if raw_config.get("seed") is not None:
        image_generation["seed"] = int(raw_config["seed"])
    elif params.get("seed") is not None:
        image_generation["seed"] = int(params["seed"])
    return image_generation


def _encode_text(tokenizer: Any, text: str) -> list[int]:
    if hasattr(tokenizer, "encode"):
        return list(tokenizer.encode(text, add_special_tokens=False))
    return list(tokenizer(text).input_ids)


def build_image_generation_prompt(
    *,
    tokenizer: Any,
    messages: list[dict[str, Any]],
    image_generation: dict[str, Any],
) -> dict[str, Any]:
    """Build the LLaDA2 text-to-image prompt and unconditional CFG prompt."""
    prompt = _extract_last_user_text(messages)
    token_grid_h = int(image_generation["token_grid_h"])
    token_grid_w = int(image_generation["token_grid_w"])
    image_header = (
        f"{SOI_TOKEN}<|reserved_token_{token_grid_h}|>"
        f"<|reserved_token_{token_grid_w}|>{BOI_TOKEN}"
    )

    prefix = f"{ROLE_SYSTEM} {T2I_SYSTEM_PROMPT} {ROLE_HUMAN}"
    image_header_ids = (
        _encode_text(tokenizer, SOI_TOKEN)
        + _encode_text(tokenizer, f"<|reserved_token_{token_grid_h}|>")
        + _encode_text(tokenizer, f"<|reserved_token_{token_grid_w}|>")
        + _encode_text(tokenizer, BOI_TOKEN)
    )
    conditional_ids = (
        _encode_text(tokenizer, prefix)
        + _encode_text(tokenizer, prompt)
        + _encode_text(tokenizer, ROLE_ASSISTANT)
        + image_header_ids
    )
    unconditional_ids = (
        _encode_text(tokenizer, prefix)
        + _encode_text(tokenizer, T2I_UNCONDITION_PROMPT)
        + _encode_text(tokenizer, ROLE_ASSISTANT)
        + image_header_ids
    )

    return {
        "input_ids": conditional_ids,
        "uncond_ids": unconditional_ids,
        "prompt": prompt,
        "image_header": image_header,
    }


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


class LLaDA2Preprocessor:
    """Preprocessor for LLaDA2-Uni model (text + image)."""

    def __init__(self, model_path: str, max_seq_len: int | None = None):
        self._max_seq_len = max_seq_len
        self._model_dir = resolve_local_model_dir(model_path)
        self._tokenizer = load_llada2_tokenizer(model_path)

        # Load HF Qwen2VLImageProcessor (do_resize=False, crop handles sizing)
        from transformers import Qwen2VLImageProcessor

        tokenizer_path = str(Path(self._model_dir) / "image_tokenizer")

        try:
            self._image_processor = Qwen2VLImageProcessor.from_pretrained(
                tokenizer_path,
                local_files_only=True,
                do_resize=False,  # Disable resize, use manual crop instead
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
            )
            self._model_dir = str(
                resolve_model_path(model_path, local_files_only=False)
            )
        self._merge_size = self._image_processor.merge_size
        self._factor = self._image_processor.patch_size * self._merge_size

        # Cache special token IDs
        self._eoi_id = self._tokenizer.convert_tokens_to_ids(EOI_TOKEN)
        self._boi_id = self._tokenizer.convert_tokens_to_ids(BOI_TOKEN)

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

        if _is_image_output_request(request):
            if raw_images:
                raise ValueError(
                    "LLaDA2-Uni image output request path currently supports "
                    "text-to-image requests only; image editing is not wired yet."
                )
            image_generation = build_image_generation_config(request)
            image_prompt = build_image_generation_prompt(
                tokenizer=self._tokenizer,
                messages=messages,
                image_generation=image_generation,
            )
            image_generation.update(
                {
                    "uncond_ids": image_prompt["uncond_ids"],
                    "text_prompt": image_prompt["prompt"],
                    "image_header": image_prompt["image_header"],
                }
            )
            input_ids_tensor = torch.tensor(
                [image_prompt["input_ids"]],
                dtype=torch.long,
            )
            validate_prompt_seq_len(
                input_ids_tensor,
                max_seq_len=self._max_seq_len,
                max_new_tokens=int(image_generation["gen_length"]),
                request_id=payload.request_id,
            )
            state = LLaDA2UniPipelineState(
                prompt={"input_ids": input_ids_tensor},
                encoder_inputs={IMAGE_STAGE: {"_skip": True, "_result": {}}},
                image_generation=image_generation,
            )
            return StagePayload(
                request_id=payload.request_id,
                request=payload.request,
                data=state.to_dict(),
            )

        image_cache_key = compute_image_cache_key(raw_images)

        images = await ensure_image_list_async(raw_images) if raw_images else []

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

            merge_sq = self._merge_size**2
            img_idx = 0
            for msg_idx, count in image_counts_per_msg:
                parts: list[str] = []
                for _ in range(count):
                    t, h, w = image_grid_thw[img_idx].tolist()
                    h_token = f"<|reserved_token_{h}|>"
                    w_token = f"<|reserved_token_{w}|>"
                    num_image_tokens = t * h * w // merge_sq
                    img_header = f"{SOI_TOKEN}{h_token}{w_token}{BOI_TOKEN}"
                    image_token_counts.append(num_image_tokens)
                    parts.extend([img_header, EOI_TOKEN])
                    img_idx += 1
                image_parts_by_msg[msg_idx] = parts
        else:
            encoder_inputs[IMAGE_STAGE] = {"_skip": True, "_result": {}}

        text_prompt = self._build_prompt(
            messages, image_parts_by_msg=image_parts_by_msg
        )
        input_ids = self._tokenizer.encode(text_prompt, add_special_tokens=False)

        if image_token_counts:
            input_ids = self._insert_image_placeholders(input_ids, image_token_counts)

        input_ids_tensor = torch.tensor([input_ids], dtype=torch.long)

        validate_prompt_seq_len(
            input_ids_tensor,
            max_seq_len=self._max_seq_len,
            max_new_tokens=request.params.get(
                "max_new_tokens", DEFAULT_THINKER_MAX_NEW_TOKENS
            ),
            request_id=payload.request_id,
        )

        prompt = {"input_ids": input_ids_tensor}

        state = LLaDA2UniPipelineState(
            prompt=prompt,
            encoder_inputs=encoder_inputs,
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
    ) -> str:
        """Build LLaDA2-Uni chat format prompt.

        Image blocks are inserted at the start of their originating message's
        content via *image_parts_by_msg* (message index -> header/footer tokens).
        """
        parts: list[str] = []

        parts.append(f"{ROLE_SYSTEM} {DEFAULT_SYSTEM_PROMPT} ")

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
