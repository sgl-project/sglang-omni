# SPDX-License-Identifier: Apache-2.0
"""Preprocessor for LLaDA2-Uni: tokenize text and prepare image inputs."""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any

import torch
from PIL import Image, ImageOps

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
DEFAULT_SYSTEM_PROMPT = "You are a multimodal understanding assistant."
SYSTEM_PROMPT_T2I = "You are a text-to-image generation assistant."
SYSTEM_PROMPT_T2I_THINKING = (
    "You are a text-to-image generation assistant with a thinking process."
)
EDIT_SYSTEM_PROMPT = "You are an image editing assistant."
UNCOND_TEXT = "<uncondition>"
DEFAULT_T2I_IMAGE_H = 1024
DEFAULT_T2I_IMAGE_W = 1024

# Image special token strings
SOI_TOKEN = "<|image|>"  # id=156901
EOI_TOKEN = "<|/image|>"  # id=156902
BOI_TOKEN = "<boi>"  # id=156904

IMAGE_TOKEN_OFFSET = 157184  # VQ codebook indices are offset by this value
DUMMY_IMAGE_TOKEN_ID = IMAGE_TOKEN_OFFSET  # <IMAGE0>, used as placeholder

# Pixel budgets for image resize (single-image / multi-image)
SINGLE_IMAGE_MIN_PIXELS = 128 * 128
SINGLE_IMAGE_MAX_PIXELS = 800 * 800
MULTI_IMAGE_MIN_PIXELS = 128 * 128
MULTI_IMAGE_MAX_PIXELS = 448 * 448

logger = logging.getLogger(__name__)


def align_cfg_unconditional_input_ids(
    tokenizer: Any,
    conditional_input_ids: list[int],
    unconditional_input_ids: list[int],
) -> tuple[list[int], int]:
    """Left-pad a CFG companion for the low_confidence_cfg scheduler contract."""
    left_pad_len = len(conditional_input_ids) - len(unconditional_input_ids)
    if left_pad_len < 0:
        raise ValueError(
            "CFG unconditional input cannot be longer than conditional input"
        )
    if not left_pad_len:
        return list(unconditional_input_ids), 0
    mask_id = getattr(tokenizer, "mask_token_id", None)
    if mask_id is None:
        raise ValueError("LLaDA2 tokenizer has no mask_token_id for CFG padding")
    return [int(mask_id)] * left_pad_len + list(unconditional_input_ids), left_pad_len


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


def compute_target_dims(
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


def resize_and_center_crop(
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


def resize_images(
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
        target_h, target_w = compute_target_dims(
            height, width, min_pixels, max_pixels, factor
        )
        result.append(resize_and_center_crop(img, target_h, target_w, factor))
    return result


def preprocess_image_edit(images: list[Image.Image], factor: int) -> list[Image.Image]:
    """Resize and center-crop edit inputs to a supported grid without padding."""
    num_patches = (512 // factor) ** 2
    candidates = []
    wp, hp = num_patches, 1
    while wp > 0:
        if max(wp, hp) / min(wp, hp) <= 4.0:
            candidates.append((wp * factor, hp * factor))
        if (hp + 1) * wp <= num_patches:
            hp += 1
        else:
            wp -= 1
    if not candidates:
        raise ValueError("Image patch factor exceeds the edit pixel budget")
    result = []
    for img in images:
        w, h = img.size
        cw, ch = max(
            candidates,
            key=lambda size: (
                min(size[0] / w, size[1] / h) / max(size[0] / w, size[1] / h),
                size,
            ),
        )
        result.append(
            img.copy()
            if img.size == (cw, ch)
            else ImageOps.fit(
                img, (cw, ch), method=Image.Resampling.LANCZOS, centering=(0.5, 0.5)
            )
        )
    return result


def edit_image_pixel_values(
    images: list[Image.Image],
    patch_size: int,
    temporal_patch_size: int,
    merge_size: int,
    image_mean: list[float],
    image_std: list[float],
    rescale_factor: float,
) -> dict[str, torch.Tensor]:
    """Patchify edit inputs with the reference's float32 normalization order."""
    import numpy as np

    patches, grids = [], []
    mean = torch.tensor(image_mean, dtype=torch.float32).view(-1, 1, 1)
    std = torch.tensor(image_std, dtype=torch.float32).view(-1, 1, 1)
    for img in images:
        values = torch.from_numpy(np.array(img.convert("RGB"))).permute(2, 0, 1).float()
        values = (values * rescale_factor - mean) / std
        _, h, w = values.shape
        gh, gw = h // patch_size, w // patch_size
        values = values.unsqueeze(0).repeat(temporal_patch_size, 1, 1, 1)
        # Match Qwen2VL's spatial merge ordering without merging VQ tokens.
        values = values.reshape(
            1,
            temporal_patch_size,
            3,
            gh // merge_size,
            merge_size,
            patch_size,
            gw // merge_size,
            merge_size,
            patch_size,
        ).permute(0, 3, 6, 4, 7, 2, 1, 5, 8)
        patches.append(values.reshape(gh * gw, 3 * temporal_patch_size * patch_size**2))
        grids.append([1, gh, gw])
    return {
        "pixel_values": torch.cat(patches),
        "image_grid_thw": torch.tensor(grids, dtype=torch.long),
    }


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
        self._soi_id = self._tokenizer.convert_tokens_to_ids(SOI_TOKEN)

    async def __call__(self, payload: StagePayload) -> StagePayload:
        request = payload.request
        raw_inputs = request.inputs
        if isinstance(raw_inputs, list):
            messages = raw_inputs
            self.validate_messages(messages)
            raw_images, image_counts_per_msg = self.extract_raw_images(messages)
        else:
            messages = raw_inputs.get("messages", [])
            self.validate_messages(messages)
            raw_images = raw_inputs.get("images")
            if raw_images is None:
                raw_images, image_counts_per_msg = self.extract_raw_images(messages)
            else:
                image_counts_per_msg = None

        metadata = request.metadata if isinstance(request.metadata, dict) else {}
        image_generation = metadata.get("image_generation")
        if image_generation is None and "image" in metadata.get(
            "output_modalities", []
        ):
            image_generation = {}
            metadata = {**metadata, "image_generation": image_generation}
        task_kind = "chat"
        if isinstance(image_generation, dict):
            task_kind = "edit" if raw_images else "t2i"
        if task_kind == "edit":
            if image_generation.get("mode") == "thinking":
                raise ValueError("Thinking mode only supports text-to-image generation")
            self.require_edit_instruction(messages)
        image_cache_key = compute_image_cache_key(raw_images)

        images = await ensure_image_list_async(raw_images) if raw_images else []
        if task_kind == "edit":
            return self.build_edit_payload(payload, messages, images, metadata)

        encoder_inputs: dict[str, dict[str, Any]] = {}
        image_token_counts: list[int] = []
        image_parts_by_msg: dict[int, list[str]] = {}

        if images:
            cropped = resize_images(images, self._factor)
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
                    # The VQ encoder has no PatchMerger: one token per patch.
                    num_image_tokens = t * h * w
                    img_header = f"{SOI_TOKEN}{h_token}{w_token}{BOI_TOKEN}"
                    image_token_counts.append(num_image_tokens)
                    parts.extend([img_header, EOI_TOKEN])
                    img_idx += 1
                image_parts_by_msg[msg_idx] = parts
        else:
            encoder_inputs[IMAGE_STAGE] = {"_skip": True, "_result": {}}

        thinking_mode = (
            task_kind == "t2i" and image_generation.get("mode") == "thinking"
        )
        text_prompt = self.build_prompt(
            messages,
            image_parts_by_msg=image_parts_by_msg,
            task_kind="t2i_thinking" if thinking_mode else task_kind,
        )
        input_ids = self._tokenizer.encode(text_prompt, add_special_tokens=False)

        if image_token_counts:
            input_ids = self.insert_image_placeholders(input_ids, image_token_counts)

        stream_state: dict[str, Any] = {}
        max_new_tokens = request.params.get(
            "max_new_tokens", DEFAULT_THINKER_MAX_NEW_TOKENS
        )
        if task_kind == "t2i":
            ig = image_generation
            image_h = int(ig.get("image_h", DEFAULT_T2I_IMAGE_H))
            image_w = int(ig.get("image_w", DEFAULT_T2I_IMAGE_W))
            if image_h <= 0 or image_w <= 0 or image_h % 32 or image_w % 32:
                raise ValueError(
                    "Image generation dimensions must be positive multiples of 32"
                )
            grid_h, grid_w = image_h // 32, image_w // 32
            stream_state["image_info"] = [{"grid_h": grid_h, "grid_w": grid_w}]
            stream_state["max_seq_len"] = self._max_seq_len
            if ig.get("dllm_steps") is not None:
                stream_state["dllm_steps"] = int(ig["dllm_steps"])
            cfg_scale = float(ig.get("cfg_scale", 1.0))
            stream_state["cfg_scale"] = cfg_scale
            stream_state["cfg_rescale"] = float(ig.get("cfg_rescale", 0.7))
            mode = ig.get("mode", "normal")
            if mode == "thinking":
                max_new_tokens = DEFAULT_THINKER_MAX_NEW_TOKENS + grid_h * grid_w
            elif mode == "normal":
                input_ids.extend(self.build_t2i_header_ids(grid_h, grid_w))
                max_new_tokens = grid_h * grid_w
                if cfg_scale > 1.0:
                    uncond = self._tokenizer.encode(
                        self.build_prompt(
                            [{"role": "user", "content": UNCOND_TEXT}], task_kind="t2i"
                        ),
                        add_special_tokens=False,
                    ) + self.build_t2i_header_ids(grid_h, grid_w)
                    self.set_cfg_branch(stream_state, input_ids, uncond)
            else:
                raise ValueError(f"Unsupported image generation mode: {mode!r}")

        input_ids_tensor = torch.tensor([input_ids], dtype=torch.long)

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
            request_metadata=metadata,
            task_kind=task_kind,
            stream_state=stream_state,
            thinking_phase="text" if thinking_mode else None,
        )
        return StagePayload(
            request_id=payload.request_id,
            request=payload.request,
            data=state.to_dict(),
        )

    @staticmethod
    def extract_raw_images(
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
    def validate_messages(messages: list[dict[str, Any]]) -> None:
        if not isinstance(messages, list):
            raise ValueError("Preprocessing expects a list of chat messages")
        for message in messages:
            if not isinstance(message, dict):
                raise ValueError("Each message must be a dict with role/content")

    def build_prompt(
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

        system_prompt = {
            "t2i": SYSTEM_PROMPT_T2I,
            "t2i_thinking": SYSTEM_PROMPT_T2I_THINKING,
            "edit": EDIT_SYSTEM_PROMPT,
        }.get(task_kind, DEFAULT_SYSTEM_PROMPT)
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

    def build_t2i_header_ids(self, grid_h: int, grid_w: int) -> list[int]:
        return (
            [self._soi_id]
            + self._tokenizer.encode(
                f"<|reserved_token_{grid_h}|><|reserved_token_{grid_w}|>",
                add_special_tokens=False,
            )
            + [self._boi_id]
        )

    @staticmethod
    def extract_user_prompt_text(messages: list[dict[str, Any]]) -> str:
        for msg in reversed(messages):
            if msg.get("role", "user") != "user":
                continue
            content = msg.get("content", "")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                return "".join(
                    item if isinstance(item, str) else item.get("text", "")
                    for item in content
                    if isinstance(item, str)
                    or isinstance(item, dict)
                    and item.get("type", "text") == "text"
                )
        return ""

    @classmethod
    def require_edit_instruction(cls, messages: list[dict[str, Any]]) -> str:
        instruction = cls.extract_user_prompt_text(messages)
        if not instruction.strip():
            raise ValueError("Image editing requires a non-empty instruction")
        return instruction

    def set_cfg_branch(
        self,
        stream_state: dict[str, Any],
        conditional: list[int],
        unconditional: list[int],
        *,
        branch: str = "uncond",
    ) -> None:
        ids, pad_len = align_cfg_unconditional_input_ids(
            self._tokenizer, conditional, unconditional
        )
        stream_state[f"{branch}_input_ids"] = ids
        stream_state[f"{branch}_left_pad_len"] = pad_len

    def build_edit_payload(
        self,
        payload: StagePayload,
        messages: list[dict[str, Any]],
        images: list[Image.Image],
        request_metadata: dict[str, Any],
    ) -> StagePayload:
        instruction = self.require_edit_instruction(messages)
        if len(images) != 1:
            raise ValueError("Image editing requires exactly one source image")
        processor = self._image_processor
        encoded = edit_image_pixel_values(
            preprocess_image_edit(images, self._factor),
            patch_size=processor.patch_size,
            temporal_patch_size=processor.temporal_patch_size,
            merge_size=processor.merge_size,
            image_mean=processor.image_mean,
            image_std=processor.image_std,
            rescale_factor=processor.rescale_factor,
        )
        t, h, w = encoded["image_grid_thw"][0].tolist()
        num_tokens = t * h * w
        encoder_inputs = {IMAGE_STAGE: encoded}
        grid_h, grid_w = t * h // self._merge_size, w // self._merge_size
        source = (
            f"{SOI_TOKEN}<|reserved_token_{h}|><|reserved_token_{w}|>"
            f"{BOI_TOKEN}{EOI_TOKEN}"
        )
        header = self.build_t2i_header_ids(grid_h, grid_w)

        def encode_prompt(text: str, *, with_image: bool = True) -> list[int]:
            prompt = self.build_prompt(
                [{"role": "user", "content": text}],
                image_parts_by_msg={0: [source if with_image else SOI_TOKEN]},
                task_kind="edit",
            )
            ids = self._tokenizer.encode(prompt, add_special_tokens=False)
            if with_image:
                ids = self.insert_image_placeholders(ids, [num_tokens])
            return ids + header

        input_ids = encode_prompt(instruction)
        input_tensor = torch.tensor([input_ids], dtype=torch.long)
        validate_prompt_seq_len(
            input_tensor,
            max_seq_len=self._max_seq_len,
            max_new_tokens=grid_h * grid_w,
            request_id=payload.request_id,
        )
        ig = request_metadata["image_generation"]
        stream_state: dict[str, Any] = {
            "image_info": [{"grid_h": grid_h, "grid_w": grid_w}],
        }
        if ig.get("dllm_steps") is not None:
            stream_state["dllm_steps"] = int(ig["dllm_steps"])
        if "cfg_text_scale" in ig:
            text_scale = float(ig["cfg_text_scale"])
        elif "cfg_scale" in ig:
            legacy_scale = float(ig["cfg_scale"])
            text_scale = 0.0 if legacy_scale == 1.0 else legacy_scale
        else:
            text_scale = 4.0
        image_scale = float(ig.get("cfg_image_scale", 0.0))
        if text_scale > 0.0 or image_scale > 0.0:
            self.set_cfg_branch(stream_state, input_ids, encode_prompt(UNCOND_TEXT))
            stream_state["cfg_scale"] = text_scale
            stream_state["cfg_rescale"] = float(ig.get("cfg_rescale", 0.7))
            if image_scale > 0.0:
                self.set_cfg_branch(
                    stream_state,
                    input_ids,
                    encode_prompt(instruction, with_image=False),
                    branch="uncond_img",
                )
                stream_state["cfg_image_scale"] = image_scale
        state = LLaDA2UniPipelineState(
            prompt={"input_ids": input_tensor},
            encoder_inputs=encoder_inputs,
            request_metadata=request_metadata,
            task_kind="edit",
            stream_state=stream_state,
        )
        return StagePayload(
            request_id=payload.request_id, request=payload.request, data=state.to_dict()
        )

    def insert_image_placeholders(
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
