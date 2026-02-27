# SPDX-License-Identifier: Apache-2.0
"""Preprocessor component for Qwen3-ASR."""

from __future__ import annotations

from typing import Any

from sglang_omni.models.qwen3_asr.modeling.processing_qwen3_asr import Qwen3ASRProcessor
from sglang_omni.models.qwen3_asr.pipeline.state_io import load_state, store_state
from sglang_omni.proto import StagePayload


class Qwen3ASRPreprocessor:
    """Preprocessor that wraps Qwen3ASRProcessor."""

    def __init__(self, model_id: str) -> None:
        self.processor = Qwen3ASRProcessor.from_pretrained(
            model_id, trust_remote_code=True
        )

    def __call__(self, payload: StagePayload) -> StagePayload:
        state = load_state(payload)
        raw_inputs = state.raw_inputs

        text = raw_inputs.get("text", "")
        audio = raw_inputs.get("audio")

        # Build messages and apply chat template like original repo
        messages = [
            {"role": "system", "content": text or ""},
            {"role": "user", "content": [{"type": "audio", "audio": ""}]},
        ]
        prompt = self.processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        # Append <asr_text> to signal ASR task
        prompt += "<asr_text>"

        # Qwen3ASRProcessor expects 'audio' and 'text'
        processed = self.processor(text=prompt, audio=audio, return_tensors="pt")

        state.prompt = {
            "input_ids": processed["input_ids"][0],
            "attention_mask": processed["attention_mask"][0],
            "prompt_text": prompt,
        }

        if "input_features" in processed:
            state.mm_inputs["audio"] = {
                "input_features": processed["input_features"],
                "feature_attention_mask": processed["feature_attention_mask"],
            }

        return store_state(payload, state)

    @property
    def tokenizer(self) -> Any:
        return self.processor.tokenizer
