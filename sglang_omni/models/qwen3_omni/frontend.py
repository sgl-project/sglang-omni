# SPDX-License-Identifier: Apache-2.0
"""Model-specific frontend preprocessing for Qwen3-Omni."""

from __future__ import annotations

from typing import Any

import torch
from transformers.models.qwen3_omni_moe.processing_qwen3_omni_moe import (
    Qwen3OmniMoeProcessor,
)

from sglang_omni.frontends import (
    append_modality_placeholders,
    apply_chat_template,
    build_audio_mm_inputs,
    build_image_mm_inputs,
    ensure_audio_list,
    ensure_chat_template,
    ensure_image_list,
    normalize_messages,
)
from sglang_omni.proto import StagePayload

IMAGE_PLACEHOLDER = "<|vision_start|><|image_pad|><|vision_end|>"
AUDIO_PLACEHOLDER = "<|audio_start|><|audio_pad|><|audio_end|>"


class Qwen3OmniFrontend:
    """CPU-side preprocessing and tokenization using the HF processor."""

    def __init__(self, model_id: str):
        self.model_id = model_id
        self.processor = Qwen3OmniMoeProcessor.from_pretrained(
            model_id,
            trust_remote_code=True,
        )
        self.tokenizer = self.processor.tokenizer
        ensure_chat_template(self.tokenizer, model_id=model_id)

    def _apply_chat_template(self, messages: list[dict[str, str]]) -> str:
        prompt_text = apply_chat_template(self.tokenizer, messages)
        if not prompt_text:
            raise ValueError("Failed to build prompt_text from chat template")
        return prompt_text

    def __call__(self, payload: StagePayload) -> StagePayload:
        inputs = payload.request.inputs
        if isinstance(inputs, dict):
            messages = inputs.get("messages", [])
            images = ensure_image_list(inputs.get("images"))
            audios = ensure_audio_list(inputs.get("audio") or inputs.get("audios"))
        else:
            messages = inputs
            images = []
            audios = []

        messages_norm = normalize_messages(messages)
        messages_with_mm = append_modality_placeholders(
            messages_norm,
            placeholders={"image": IMAGE_PLACEHOLDER, "audio": AUDIO_PLACEHOLDER},
            counts={"image": len(images), "audio": len(audios)},
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
            "image": build_image_mm_inputs(hf_inputs),
            "audio": build_audio_mm_inputs(hf_inputs),
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
