# SPDX-License-Identifier: Apache-2.0
"""Model-specific preprocessor for MiniCPM-V (2.6 and 4.5) and MiniCPM-o.

This is the core preprocessing module that handles:
- LLaVA-style slice-based image tokenization
- Audio feature extraction for MiniCPM-o (Whisper-based)

Supported versions:
- MiniCPM-V 2.6: SigLIP-400M + MiniCPM-3.0 LLM
- MiniCPM-V 4.5: SigLIP2-400M + Qwen3-8B LLM + 3D-Resampler
- MiniCPM-o 2.6: Full audio support with Whisper encoder

Key differences from Qwen3-Omni:
- Images are sliced into multiple patches (up to 9 slices + 1 thumbnail)
- Each slice produces fixed-length tokens via SigLIP/SigLIP2
- `tgt_sizes` tracks each slice's patch grid size for 2D RoPE
- No `image_grid_thw` - MiniCPM-V uses different position encoding

Note: The HF AutoProcessor handles version-specific tokenization automatically.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any

import torch

from sglang_omni.models.minicpm_v.io import PipelineState
from sglang_omni.models.weight_loader import resolve_model_path
from sglang_omni.preprocessing import (
    compute_audio_cache_key,
    compute_image_cache_key,
    ensure_audio_list_async,
    ensure_chat_template,
    ensure_image_list_async,
    normalize_messages,
)
from sglang_omni.preprocessing.audio import build_audio_mm_inputs
from sglang_omni.proto import StagePayload

logger = logging.getLogger(__name__)


def _resolve_local_model_dir(model_path: str) -> str:
    """Resolve a local model directory without eagerly hydrating full snapshots."""
    path = Path(model_path)
    if path.exists():
        return str(path)
    try:
        return str(resolve_model_path(model_path, local_files_only=True))
    except (FileNotFoundError, OSError) as exc:
        logger.warning(
            "Local-only model resolution failed for %s; falling back to hub id",
            model_path,
            exc_info=exc,
        )
        return model_path


def build_minicpm_image_mm_inputs(hf_inputs: dict[str, Any]) -> dict[str, Any]:
    """Extract MiniCPM-V specific image tensors from HF processor outputs.

    Unlike Qwen3-Omni which uses `image_grid_thw`, MiniCPM-V uses:
    - `pixel_values`: List[Tensor[num_slices, C, H, W]] or Tensor[total_slices, C, H, W]
    - `tgt_sizes`: Tensor[total_slices, 2] - patch grid (height, width) for each slice
    - `image_sizes`: Original image dimensions (optional)
    """
    pixel_values = hf_inputs.get("pixel_values")
    tgt_sizes = hf_inputs.get("tgt_sizes")
    image_sizes = hf_inputs.get("image_sizes")

    # Handle different pixel_values formats from processor
    if isinstance(pixel_values, list):
        # List of tensors per image - concatenate for batch processing
        if pixel_values and isinstance(pixel_values[0], torch.Tensor):
            # Track slice counts per image for later reconstruction
            slice_lengths = [pv.shape[0] for pv in pixel_values]
            pixel_values = torch.cat(pixel_values, dim=0)
        else:
            slice_lengths = []
    elif isinstance(pixel_values, torch.Tensor):
        # Single tensor with all slices
        slice_lengths = [pixel_values.shape[0]]
    else:
        slice_lengths = []

    result: dict[str, Any] = {
        "pixel_values": pixel_values,
        "tgt_sizes": tgt_sizes,
        "slice_lengths": slice_lengths,
    }
    if image_sizes is not None:
        result["image_sizes"] = image_sizes
    return result


class MiniCPMVPreprocessor:
    """CPU-side preprocessing and tokenization using the HF processor.

    This preprocessor handles:
    - MiniCPM-V's LLaVA-style image processing
    - MiniCPM-o's Whisper-based audio processing (if audio inputs provided)

    Processing flow:
    1. Uses AutoProcessor.from_pretrained to load the custom MiniCPM processor
    2. Processor automatically handles image slicing based on slice_mode config
    3. Extracts tgt_sizes for 2D position encoding in the Resampler
    4. For audio: extracts mel-spectrogram features for the Whisper encoder

    The processor outputs contain placeholder tokens (<image>, <audio>) that mark
    where embeddings should be inserted after encoding.
    """

    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model_dir = _resolve_local_model_dir(model_path)

        # MiniCPM-V/o uses AutoProcessor which loads the custom processing class
        from transformers import AutoProcessor

        try:
            self.processor = AutoProcessor.from_pretrained(
                self.model_dir,
                trust_remote_code=True,
                local_files_only=True,
            )
        except (OSError, ValueError, RuntimeError):
            if Path(model_path).exists():
                raise
            self.processor = AutoProcessor.from_pretrained(
                model_path,
                trust_remote_code=True,
                local_files_only=False,
            )
            self.model_dir = str(resolve_model_path(model_path, local_files_only=False))

        self.tokenizer = self.processor.tokenizer
        ensure_chat_template(self.tokenizer, model_path=self.model_dir)

        # Check if this model supports audio (MiniCPM-o vs MiniCPM-V)
        from sglang_omni.models.minicpm_v.components.common import has_audio_support

        self._supports_audio = has_audio_support(model_path)

    def _build_multimodal_messages(
        self,
        messages: list[dict[str, Any]],
        *,
        num_images: int,
        num_audios: int = 0,
    ) -> list[dict[str, Any]]:
        """Convert simple messages to HF's structured multimodal format.

        MiniCPM-V/o expects images to be marked with {"type": "image"} and
        audio with {"type": "audio"} in the content list.
        """
        if num_images == 0 and num_audios == 0:
            return messages

        result: list[dict[str, Any]] = []
        for i, msg in enumerate(messages):
            role = msg.get("role", "user")
            content = msg.get("content", "")

            # Only inject placeholders into the last user message
            if i == len(messages) - 1 and role == "user":
                content_parts: list[dict[str, Any]] = []
                # MiniCPM format: images come before text, then audio
                for _ in range(num_images):
                    content_parts.append({"type": "image"})
                for _ in range(num_audios):
                    content_parts.append({"type": "audio"})
                content_parts.append({"type": "text", "text": content})
                result.append({"role": role, "content": content_parts})
            else:
                result.append(msg)

        return result

    async def __call__(self, payload: StagePayload) -> StagePayload:
        inputs = payload.request.inputs
        if isinstance(inputs, dict):
            messages = inputs.get("messages", [])
            raw_images = inputs.get("images")
            raw_audios = inputs.get("audio") or inputs.get("audios")
            audio_target_sr = int(inputs.get("audio_target_sr", 16000))

            # Compute cache keys BEFORE conversion (paths are cheap to hash)
            image_cache_key = compute_image_cache_key(raw_images)
            audio_cache_key = compute_audio_cache_key(raw_audios)

            # Use async versions for concurrent loading
            images, audios = await asyncio.gather(
                ensure_image_list_async(raw_images),
                ensure_audio_list_async(raw_audios, target_sr=audio_target_sr)
                if raw_audios and self._supports_audio
                else asyncio.coroutine(lambda: [])(),
            )
        else:
            messages = inputs
            images = []
            audios = []
            image_cache_key = None
            audio_cache_key = None
            audio_target_sr = 16000

        messages_norm = normalize_messages(messages)
        messages_mm = self._build_multimodal_messages(
            messages_norm,
            num_images=len(images),
            num_audios=len(audios) if audios else 0,
        )

        # Apply chat template to get the prompt text
        prompt_text = self.processor.apply_chat_template(
            messages_mm,
            add_generation_prompt=True,
            tokenize=False,
        )

        # Process through HF processor to get tokenized inputs and image/audio tensors
        processor_kwargs: dict[str, Any] = {
            "text": prompt_text,
            "images": images or None,
            "return_tensors": "pt",
        }
        # Add audio if supported and present
        if audios and self._supports_audio:
            processor_kwargs["audio"] = audios

        hf_inputs = self.processor(**processor_kwargs)

        input_ids = hf_inputs["input_ids"][0]
        attention_mask = hf_inputs.get("attention_mask")
        if isinstance(attention_mask, torch.Tensor):
            attention_mask = attention_mask[0]
        else:
            attention_mask = torch.ones_like(input_ids)

        # Build MiniCPM-V/o specific multimodal inputs
        mm_inputs: dict[str, Any] = {
            "image": build_minicpm_image_mm_inputs(hf_inputs),
        }
        # Add audio inputs if present
        if audios and self._supports_audio:
            mm_inputs["audio"] = build_audio_mm_inputs(hf_inputs)
        else:
            mm_inputs["audio"] = {}

        # Build encoder_inputs with cache_key for efficient caching
        image_encoder_inputs = {**mm_inputs["image"]}
        if image_cache_key:
            image_encoder_inputs["cache_key"] = image_cache_key

        encoder_inputs: dict[str, dict[str, Any]] = {}

        # Image encoder inputs
        image_encoder_inputs = {
            k: v for k, v in image_encoder_inputs.items() if v is not None
        }
        if image_encoder_inputs.get("pixel_values") is not None:
            encoder_inputs["image_encoder"] = image_encoder_inputs
        else:
            encoder_inputs["image_encoder"] = {"_skip": True, "_result": {}}

        # Audio encoder inputs (MiniCPM-o only)
        if audios and self._supports_audio:
            audio_encoder_inputs = {**mm_inputs["audio"]}
            if audio_cache_key:
                audio_encoder_inputs["cache_key"] = f"{audio_cache_key}@sr={audio_target_sr}"
            audio_encoder_inputs = {
                k: v for k, v in audio_encoder_inputs.items() if v is not None
            }
            if audio_encoder_inputs.get("input_features") is not None:
                encoder_inputs["audio_encoder"] = audio_encoder_inputs
            else:
                encoder_inputs["audio_encoder"] = {"_skip": True, "_result": {}}
        else:
            encoder_inputs["audio_encoder"] = {"_skip": True, "_result": {}}

        state = PipelineState(
            raw_inputs=inputs,
            mm_inputs=mm_inputs,
            prompt={
                "prompt_text": prompt_text,
                "input_ids": input_ids,
                "attention_mask": attention_mask,
            },
            encoder_inputs=encoder_inputs,
            stream_state={"token_ids": [], "text": ""},
        )
        payload.data = state.to_dict()
        return payload
