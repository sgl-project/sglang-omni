# SPDX-License-Identifier: Apache-2.0
"""Frontend preprocessing for Qwen3-Omni."""

from __future__ import annotations

import json
from typing import Any

import torch
from transformers.models.qwen3_omni_moe.processing_qwen3_omni_moe import (
    Qwen3OmniMoeProcessor,
)

from sglang_omni.proto import StagePayload

IMAGE_PLACEHOLDER = "<|vision_start|><|image_pad|><|vision_end|>"
AUDIO_PLACEHOLDER = "<|audio_start|><|audio_pad|><|audio_end|>"


def _normalize_messages(messages: Any) -> list[dict[str, str]]:
    if not isinstance(messages, list):
        raise ValueError("Qwen3-Omni frontend expects a list of chat messages")
    normalized: list[dict[str, str]] = []
    for message in messages:
        if not isinstance(message, dict):
            raise ValueError("Each message must be a dict with role/content")
        content = message.get("content", "")
        if not isinstance(content, str):
            content = json.dumps(content, ensure_ascii=True)
        normalized.append({"role": message.get("role", "user"), "content": content})
    return normalized


def _append_placeholders(
    messages: list[dict[str, str]],
    *,
    num_images: int,
    num_audios: int,
) -> list[dict[str, str]]:
    if not messages:
        return messages
    placeholders = (IMAGE_PLACEHOLDER * num_images) + (AUDIO_PLACEHOLDER * num_audios)
    if not placeholders:
        return messages
    updated = [dict(m) for m in messages]
    updated[-1]["content"] = f"{updated[-1]['content']}\n{placeholders}"
    return updated


class Qwen3OmniFrontend:
    """CPU-side preprocessing and tokenization using the HF processor."""

    def __init__(self, model_id: str):
        self.model_id = model_id
        self.processor = Qwen3OmniMoeProcessor.from_pretrained(
            model_id,
            trust_remote_code=True,
        )
        self.tokenizer = self.processor.tokenizer

    def _apply_chat_template(self, messages: list[dict[str, str]]) -> str:
        if not hasattr(self.tokenizer, "apply_chat_template"):
            raise ValueError("Tokenizer does not support apply_chat_template")
        if not getattr(self.tokenizer, "chat_template", None):
            raise ValueError("Tokenizer chat_template is not set for this model")
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    def __call__(self, payload: StagePayload) -> StagePayload:
        inputs = payload.request.inputs
        if isinstance(inputs, dict):
            messages = inputs.get("messages", [])
            images = inputs.get("images") or []
            audios = inputs.get("audio") or inputs.get("audios") or []
        else:
            messages = inputs
            images = []
            audios = []

        messages_norm = _normalize_messages(messages)
        messages_with_mm = _append_placeholders(
            messages_norm,
            num_images=len(images),
            num_audios=len(audios),
        )
        prompt_text = self._apply_chat_template(messages_with_mm)

        hf_inputs = self.processor(
            text=prompt_text,
            images=images or None,
            audio=audios or None,
            return_tensors="pt",
        )

        input_ids = hf_inputs["input_ids"][0]
        attention_mask = hf_inputs.get("attention_mask")
        if isinstance(attention_mask, torch.Tensor):
            attention_mask = attention_mask[0]
        else:
            attention_mask = torch.ones_like(input_ids)

        mm_inputs: dict[str, Any] = {
            "image": {
                "pixel_values": hf_inputs.get("pixel_values"),
                "image_grid_thw": hf_inputs.get("image_grid_thw"),
            },
            "audio": {
                "input_features": hf_inputs.get("input_features"),
                "feature_attention_mask": hf_inputs.get("feature_attention_mask"),
                "audio_feature_lengths": hf_inputs.get("audio_feature_lengths"),
            },
        }

        payload.data = {
            "raw_inputs": inputs,
            "mm_inputs": mm_inputs,
            "prompt": {
                "prompt_text": prompt_text,
                "input_ids": input_ids,
                "attention_mask": attention_mask,
            },
            "engine_inputs": {
                "image_encoder": mm_inputs["image"],
                "audio_encoder": mm_inputs["audio"],
            },
        }
        return payload
