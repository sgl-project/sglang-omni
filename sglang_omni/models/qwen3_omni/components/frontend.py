# SPDX-License-Identifier: Apache-2.0
"""Model-specific frontend preprocessing for Qwen3-Omni."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import torch
from transformers import AutoFeatureExtractor, AutoImageProcessor, AutoTokenizer
from transformers.models.qwen3_omni_moe.processing_qwen3_omni_moe import (
    Qwen3OmniMoeProcessor,
)
from transformers.utils.hub import cached_file

from sglang_omni.frontends import (
    build_audio_mm_inputs,
    build_image_mm_inputs,
    ensure_audio_list,
    ensure_chat_template,
    ensure_image_list,
    normalize_messages,
)
from sglang_omni.proto import StagePayload


def _resolve_local_model_dir(model_id: str) -> str:
    """Resolve the local snapshot directory for a cached model."""
    config_path = cached_file(model_id, "config.json", local_files_only=True)
    return str(Path(config_path).parent)


def _load_preprocessor_config(model_dir: str) -> Mapping[str, Any]:
    cfg_path = Path(model_dir) / "preprocessor_config.json"
    with open(cfg_path, encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, Mapping):
        raise ValueError("preprocessor_config.json is not a mapping")
    return payload


class _StubVideoProcessor:
    """Minimal video processor to avoid remote downloads when video is unused."""

    def __init__(self, *, merge_size: int, temporal_patch_size: int):
        self.merge_size = merge_size
        self.temporal_patch_size = temporal_patch_size

    def __call__(
        self, *args: Any, **kwargs: Any
    ) -> Any:  # pragma: no cover - defensive
        raise NotImplementedError("Video inputs are not supported in this pipeline yet")


def _build_processor_local(model_dir: str, model_id: str) -> Qwen3OmniMoeProcessor:
    cfg = _load_preprocessor_config(model_dir)
    merge_size = int(cfg.get("merge_size", 2))
    temporal_patch_size = int(cfg.get("temporal_patch_size", 2))

    tokenizer = AutoTokenizer.from_pretrained(
        model_dir,
        trust_remote_code=True,
        local_files_only=True,
    )
    ensure_chat_template(tokenizer, model_id=model_dir)

    image_processor = AutoImageProcessor.from_pretrained(
        model_dir,
        trust_remote_code=True,
        local_files_only=True,
    )
    feature_extractor = AutoFeatureExtractor.from_pretrained(
        model_dir,
        trust_remote_code=True,
        local_files_only=True,
    )
    video_processor = _StubVideoProcessor(
        merge_size=merge_size,
        temporal_patch_size=temporal_patch_size,
    )
    processor = Qwen3OmniMoeProcessor(
        image_processor=image_processor,
        video_processor=video_processor,
        feature_extractor=feature_extractor,
        tokenizer=tokenizer,
    )
    ensure_chat_template(processor.tokenizer, model_id=model_dir)
    return processor


class Qwen3OmniFrontend:
    """CPU-side preprocessing and tokenization using the HF processor."""

    def __init__(self, model_id: str):
        self.model_id = model_id
        self.model_dir = _resolve_local_model_dir(model_id)
        try:
            self.processor = Qwen3OmniMoeProcessor.from_pretrained(
                self.model_dir,
                trust_remote_code=True,
                local_files_only=True,
            )
        except Exception:
            # The hub cache may not include video_preprocessor_config.json.
            # Build the processor locally with a stub video processor instead.
            self.processor = _build_processor_local(self.model_dir, model_id)
        self.tokenizer = self.processor.tokenizer
        ensure_chat_template(self.tokenizer, model_id=self.model_dir)

    def _build_multimodal_messages(
        self,
        messages: list[dict[str, Any]],
        *,
        num_images: int,
        num_audios: int,
    ) -> list[dict[str, Any]]:
        """Convert simple messages to HF's structured multimodal format."""
        if num_images == 0 and num_audios == 0:
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
                for _ in range(num_audios):
                    content_parts.append({"type": "audio"})
                content_parts.append({"type": "text", "text": content})
                result.append({"role": role, "content": content_parts})
            else:
                result.append(msg)

        return result

    def __call__(self, payload: StagePayload) -> StagePayload:
        inputs = payload.request.inputs
        if isinstance(inputs, dict):
            messages = inputs.get("messages", [])
            images = ensure_image_list(inputs.get("images"))
            audio_target_sr = int(inputs.get("audio_target_sr", 16000))
            audios = ensure_audio_list(
                inputs.get("audio") or inputs.get("audios"),
                target_sr=audio_target_sr,
            )
        else:
            messages = inputs
            images = []
            audios = []

        messages_norm = normalize_messages(messages)
        messages_mm = self._build_multimodal_messages(
            messages_norm,
            num_images=len(images),
            num_audios=len(audios),
        )
        prompt_text = self.processor.apply_chat_template(
            messages_mm,
            add_generation_prompt=True,
            tokenize=False,
        )

        hf_inputs = self.processor(
            text=prompt_text,
            images=images or None,
            audio=audios or None,
            add_special_tokens=False,
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
            "encoder_inputs": {
                "image_encoder": mm_inputs["image"],
                "audio_encoder": mm_inputs["audio"],
            },
            "stream_state": {"token_ids": [], "text": ""},
        }
        return payload
