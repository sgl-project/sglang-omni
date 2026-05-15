# SPDX-License-Identifier: Apache-2.0
"""Preprocessor for LLaDA2-Uni: tokenize text and prepare image inputs."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import PIL.Image
import torch

from sglang_omni.models.llada2_uni.components.common import (
    load_llada2_config,
    load_llada2_tokenizer,
    resolve_local_model_dir,
)
from sglang_omni.models.llada2_uni.payload_types import PipelineState, PromptInputs
from sglang_omni.models.weight_loader import resolve_model_path
from sglang_omni.preprocessing.image import ensure_image_list_async
from sglang_omni.proto import StagePayload

logger = logging.getLogger(__name__)

# LLaDA2-Uni chat template tokens
ROLE_HUMAN = "<role>HUMAN</role>"
ROLE_ASSISTANT = "<role>ASSISTANT</role>"
ROLE_SYSTEM = "<role>SYSTEM</role>"
DEFAULT_SYSTEM_PROMPT = "detailed thinking off"

# Image special token strings
SOI_TOKEN = "<|image|>"  # id=156901
EOI_TOKEN = "<|/image|>"  # id=156902
BOI_TOKEN = "<boi>"  # id=156904

IMAGE_TOKEN_OFFSET = 157184  # VQ codebook indices are offset by this value
DUMMY_IMAGE_TOKEN_ID = IMAGE_TOKEN_OFFSET  # <IMAGE0>, used as placeholder

IMAGE_STAGE = "image_encoder"


def _center_crop(pil_image, crop_size):
    cw, ch = crop_size
    w, h = pil_image.size
    left = max(0, (w - cw) // 2)
    top = max(0, (h - ch) // 2)
    return pil_image.crop((left, top, left + cw, top + ch)).resize(
        (cw, ch), PIL.Image.LANCZOS
    )


def _var_center_crop(pil_image, crop_size_list):
    w, h = pil_image.size
    rem_percent = [
        min(cw / w, ch / h) / max(cw / w, ch / h) for cw, ch in crop_size_list
    ]
    crop_size = max(zip(rem_percent, crop_size_list))[1]
    return _center_crop(pil_image, crop_size)


def _generate_crop_size_list(num_patches, patch_size, max_ratio=4.0):
    assert max_ratio >= 1.0
    crop_size_list = []
    wp, hp = num_patches, 1
    while wp > 0:
        if max(wp, hp) / min(wp, hp) <= max_ratio:
            crop_size_list.append((wp * patch_size, hp * patch_size))
        if (hp + 1) * wp <= num_patches:
            hp += 1
        else:
            wp -= 1
    return crop_size_list


def _crop_images(images: list[PIL.Image.Image], factor: int) -> list[PIL.Image.Image]:
    """Crop PIL images to valid patch-aligned sizes for LLaDA2 ViT."""
    crop_size_list = _generate_crop_size_list((512 // factor) ** 2, factor)
    return [_var_center_crop(img, crop_size_list) for img in images]


class LLaDA2Preprocessor:
    """Preprocessor for LLaDA2-Uni model (text + image)."""

    def __init__(self, model_path: str):
        self._model_path = model_path
        self._model_dir = resolve_local_model_dir(model_path)
        self._config = load_llada2_config(model_path)
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
        raw_inputs = request.inputs if hasattr(request, "inputs") else {}
        if isinstance(raw_inputs, list):
            messages = raw_inputs
            raw_images = self._extract_raw_images(messages)
        else:
            messages = raw_inputs.get("messages", [])
            raw_images = raw_inputs.get("images")
            if raw_images is None:
                raw_images = self._extract_raw_images(messages)

        # Load images asynchronously via framework utility
        images = await ensure_image_list_async(raw_images) if raw_images else []

        # Prepare encoder inputs and build prompt
        encoder_inputs: dict[str, dict[str, Any]] = {}
        prompt_text_parts: list[str] = []
        image_info_list: list[dict] = []  # track per-image metadata for token splicing

        if images:
            cropped = _crop_images(images, self._factor)
            img_result = self._image_processor(images=cropped, return_tensors="pt")
            pixel_values = img_result["pixel_values"]
            image_grid_thw = img_result["image_grid_thw"]
            encoder_inputs[IMAGE_STAGE] = {
                "pixel_values": pixel_values,
                "image_grid_thw": image_grid_thw,
            }

            # Build image placeholder tokens for each image
            for i in range(image_grid_thw.shape[0]):
                t, h, w = image_grid_thw[i].tolist()
                # h, w are in patch grid units; convert to reserved token indices
                h_token = f"<|reserved_token_{h}|>"
                w_token = f"<|reserved_token_{w}|>"
                # Number of VQ tokens = t * h * w (before spatial merge)
                merge_sq = self._merge_size**2
                num_image_tokens = t * h * w // merge_sq

                # Image header: <|image|> <h> <w> <boi>
                img_header = f"{SOI_TOKEN}{h_token}{w_token}{BOI_TOKEN}"

                image_info_list.append(
                    {
                        "num_tokens": num_image_tokens,
                    }
                )
                prompt_text_parts.append(img_header)

            prompt_text_parts.append(EOI_TOKEN)
        else:
            encoder_inputs[IMAGE_STAGE] = {"_skip": True, "_result": {}}

        # Build text prompt from messages
        text_prompt = self._build_prompt(messages, prompt_text_parts)
        input_ids = self._tokenizer.encode(text_prompt, add_special_tokens=False)

        # If we have images, insert placeholder VQ token IDs into input_ids
        if image_info_list:
            input_ids = self._insert_image_placeholders(input_ids, image_info_list)

        input_ids_tensor = torch.tensor([input_ids], dtype=torch.long)

        prompt: PromptInputs = {
            "input_ids": input_ids_tensor,
            "prompt_text": text_prompt,
        }

        state = PipelineState(
            raw_inputs=raw_inputs,
            prompt=prompt,
            encoder_inputs=encoder_inputs,
        )
        # Store image info for the merge step
        if image_info_list:
            state.stream_state["image_info"] = image_info_list

        return StagePayload(
            request_id=payload.request_id,
            request=payload.request,
            data=state.to_dict(),
        )

    @staticmethod
    def _extract_raw_images(messages: list[dict[str, Any]]) -> list[Any]:
        raw_images: list[Any] = []
        for msg in messages:
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
                    elif item.get("type") == "image":
                        img = item.get("image", "")
                        if img:
                            raw_images.append(img)
        return raw_images

    def _build_prompt(
        self,
        messages: list[dict[str, Any]],
        image_prefix_parts: list[str] | None = None,
    ) -> str:
        """Build LLaDA2-Uni chat format prompt."""
        # Format must match HF `modeling_llada2uni_moe.py:_build_chat` byte-for-byte
        # (`<role>SYSTEM</role> {sys} <role>HUMAN</role>{user}<role>ASSISTANT</role>`)
        parts: list[str] = []

        parts.append(f"{ROLE_SYSTEM} {DEFAULT_SYSTEM_PROMPT} ")

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if role == "system":
                continue

            role_tag = ROLE_HUMAN if role == "user" else ROLE_ASSISTANT

            if isinstance(content, str):
                parts.append(f"{role_tag}{content}")
            elif isinstance(content, list):
                text_parts: list[str] = []
                for item in content:
                    if isinstance(item, dict):
                        item_type = item.get("type", "text")
                        if item_type == "text":
                            text_parts.append(item.get("text", ""))
                        # image items are handled separately via image_prefix_parts
                    elif isinstance(item, str):
                        text_parts.append(item)
                parts.append(f"{role_tag}{''.join(text_parts)}")
            else:
                parts.append(f"{role_tag}{content}")

        # Insert image tokens before the last ASSISTANT tag
        # Format: <|image|><h><w><boi> [VQ placeholders] <|/image|> question_text
        if image_prefix_parts:
            # Images go right before the user text in the last user message
            # We insert them at the start of the last user message content
            # The actual format from understand_image: img_header + image_tokens + eoi + question
            # We build: <role>HUMAN</role> img_prefixes question
            # Find last HUMAN message and prepend image tokens
            last_human_idx = None
            for i, part in enumerate(parts):
                if ROLE_HUMAN in part:
                    last_human_idx = i

            img_prefix = "".join(image_prefix_parts)
            if last_human_idx is not None:
                # Insert image header + eoi at start of user content
                original = parts[last_human_idx]
                # After <role>HUMAN</role>, before the text
                marker = ROLE_HUMAN
                idx = original.find(marker) + len(marker)
                parts[last_human_idx] = original[:idx] + img_prefix + original[idx:]
            else:
                # No user message found; prepend before ASSISTANT
                parts.insert(-1 if parts else 0, img_prefix)

        parts.append(ROLE_ASSISTANT)
        return "".join(parts)

    def _insert_image_placeholders(
        self,
        input_ids: list[int],
        image_info_list: list[dict],
    ) -> list[int]:
        boi_idx = next(
            (i for i, tid in enumerate(input_ids) if tid == self._boi_id), None
        )
        if boi_idx is None:
            logger.warning(
                "No <boi> token found in input_ids; cannot insert image placeholders"
            )
            return input_ids

        eoi_idx = next(
            (i for i, tid in enumerate(input_ids) if tid == self._eoi_id), None
        )
        if eoi_idx is None:
            raise ValueError("No <eoi> token found after <boi>; malformed prompt")

        all_image_tokens = sum(
            ([DUMMY_IMAGE_TOKEN_ID] * info["num_tokens"] for info in image_info_list),
            [],
        )
        return input_ids[: boi_idx + 1] + all_image_tokens + input_ids[eoi_idx:]
