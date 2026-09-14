# SPDX-License-Identifier: MIT
# Copyright (C) 2026 Tencent. All rights reserved.
# Derived from Tencent-Hunyuan/AuK; see LICENSE for the MIT permission notice.
"""AuK conditioning encoder: frozen Qwen2.5-Omni-3B Thinker."""

from __future__ import annotations

import logging
from typing import Any

import torch

from sglang_omni.models.auk.constants import NO_PROMPT_AUDIO_MARKER

logger = logging.getLogger(__name__)


def build_messages(instruction: str, has_reference_audio: bool) -> list[dict[str, Any]]:
    """Build the single-turn ChatML message list AuK is trained on."""
    text = instruction
    if not has_reference_audio and not text.endswith(NO_PROMPT_AUDIO_MARKER):
        text = text + NO_PROMPT_AUDIO_MARKER

    content: list[dict[str, Any]] = [{"type": "text", "text": text}]
    if has_reference_audio:
        content.append({"type": "audio", "audio": None})
    return [{"role": "user", "content": content}]


class AuKConditionEncoder:
    """Frozen Qwen2.5-Omni Thinker used as AuK's instruction/reference encoder."""

    def __init__(
        self,
        model_path: str,
        *,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.bfloat16,
    ):
        from transformers import (
            Qwen2_5OmniProcessor,
            Qwen2_5OmniThinkerForConditionalGeneration,
        )

        class AuKThinker(Qwen2_5OmniThinkerForConditionalGeneration):
            _keys_to_ignore_on_load_unexpected = [
                *(
                    Qwen2_5OmniThinkerForConditionalGeneration._keys_to_ignore_on_load_unexpected
                    or []
                ),
                r"^(talker|token2wav)\.",
            ]

        self.model_path = model_path
        self.device = torch.device(device)
        self.dtype = dtype

        logger.info(
            "AuK: loading Qwen2.5-Omni Thinker from %s; "
            "checkpoint talker/token2wav branches are unused",
            model_path,
        )
        self.processor = Qwen2_5OmniProcessor.from_pretrained(model_path)
        model = AuKThinker.from_pretrained(model_path, torch_dtype=dtype)
        model.visual = None
        model.lm_head = torch.nn.Identity()
        model.requires_grad_(False)
        model.eval()
        self.model = model.to(device=self.device, dtype=torch.float32)

    @property
    def num_hidden_layers(self) -> int:
        return int(self.model.config.text_config.num_hidden_layers)

    @torch.no_grad()
    def encode(self, messages, audio):
        return self.encode_batch([messages], [audio])[0]

    @torch.no_grad()
    def encode_batch(self, messages, audios):
        """Encode padded requests together, returning only each request's valid tokens."""
        formatted = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        kwargs = dict(text=formatted, padding=True, return_tensors="pt")
        references = [audio for audio in audios if audio is not None]
        if references:
            kwargs["audio"] = references
        inputs = self.processor(**kwargs)
        inputs = {k: v.to(self.device) for k, v in inputs.items() if torch.is_tensor(v)}
        outputs = self.model(**inputs, output_hidden_states=True, use_cache=False)
        hidden = torch.stack(outputs.hidden_states, dim=1)
        masks = inputs["attention_mask"].bool()
        return [(item[:, mask], mask[mask]) for item, mask in zip(hidden, masks)]
