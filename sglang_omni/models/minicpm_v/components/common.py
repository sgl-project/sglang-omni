# SPDX-License-Identifier: Apache-2.0
"""Shared helpers for MiniCPM-V components."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sglang_omni.utils import load_hf_config


# Weight prefix constants for MiniCPM-V 2.6
# These match the HuggingFace checkpoint key structure
VISUAL_PREFIX = ("vpm.", "model.vpm.")  # SigLIP ViT
RESAMPLER_PREFIX = ("resampler.", "model.resampler.")  # Perceiver Resampler
LLM_PREFIX = ("llm.", "model.llm.")  # MiniCPM 3.0 LLM backbone
AUDIO_PREFIX = ("apm.", "model.apm.")  # MiniCPM-o 2.6 only (Whisper)


def load_minicpm_config(model_path: str, local_files_only: bool = True) -> Any:
    """Load the full MiniCPM-V config from HuggingFace.

    MiniCPM-V uses a flat config structure unlike Qwen3-Omni's nested thinker_config.
    """
    return load_hf_config(
        model_path, trust_remote_code=True, local_files_only=local_files_only
    )


def load_llm_config(model_path: str, local_files_only: bool = True) -> Any:
    """Load the LLM config from MiniCPM-V.

    MiniCPM-V stores LLM config directly on the main config object.
    The LLM is a MiniCPM 3.0 Llama-style model.
    """
    cfg = load_minicpm_config(model_path, local_files_only=local_files_only)
    # MiniCPM-V 2.6 has llm_config nested, but fallback to main config for older versions
    return getattr(cfg, "llm_config", cfg)


def load_vision_config(model_path: str, local_files_only: bool = True) -> Any:
    """Load the vision encoder config from MiniCPM-V."""
    cfg = load_minicpm_config(model_path, local_files_only=local_files_only)
    return getattr(cfg, "vision_config", None)


def load_audio_config(model_path: str, local_files_only: bool = True) -> Any:
    """Load the audio encoder config from MiniCPM-o.

    MiniCPM-o 2.6 uses a Whisper-based audio encoder.
    Returns None if the model doesn't support audio (MiniCPM-V vs MiniCPM-o).
    """
    cfg = load_minicpm_config(model_path, local_files_only=local_files_only)
    # Try different attribute names used by MiniCPM-o
    audio_cfg = getattr(cfg, "audio_config", None)
    if audio_cfg is None:
        audio_cfg = getattr(cfg, "apm_config", None)
    if audio_cfg is None:
        audio_cfg = getattr(cfg, "whisper_config", None)
    return audio_cfg


def has_audio_support(model_path: str) -> bool:
    """Check if the model supports audio input (MiniCPM-o vs MiniCPM-V)."""
    return load_audio_config(model_path) is not None


@dataclass(frozen=True)
class MiniCPMVSpec:
    """Lightweight spec extracted from the HF config.

    Key differences from Qwen3-Omni:
    - Uses `im_start_id` and `im_end_id` for image boundary tokens
    - Uses `slice_mode` for dynamic image slicing configuration
    - No spatial_merge_size (Resampler handles token reduction)
    """

    model_path: str
    im_start_id: int
    im_end_id: int
    image_token_id: int  # The actual image placeholder token ID
    slice_mode: str | None
    query_num: int  # Number of query tokens per slice in Resampler

    @classmethod
    def from_model_path(cls, model_path: str) -> "MiniCPMVSpec":
        cfg = load_minicpm_config(model_path)
        # MiniCPM-V uses im_start_id/im_end_id for image boundaries
        im_start_id = getattr(cfg, "im_start_id", None)
        im_end_id = getattr(cfg, "im_end_id", None)
        slice_mode = getattr(cfg, "slice_mode", None)

        # MiniCPM-V may use im_token_id or similar for the actual placeholder
        # Fall back to im_end_id if not found (common pattern)
        image_token_id = getattr(cfg, "im_token_id", None)
        if image_token_id is None:
            image_token_id = getattr(cfg, "image_token_id", None)
        if image_token_id is None:
            # Use im_end_id as the placeholder token (matches HF implementation)
            image_token_id = im_end_id

        # Query num from vision config or resampler config
        vision_cfg = getattr(cfg, "vision_config", None)
        query_num = getattr(vision_cfg, "query_num", 64) if vision_cfg else 64

        return cls(
            model_path=model_path,
            im_start_id=int(im_start_id) if im_start_id is not None else 151857,
            im_end_id=int(im_end_id) if im_end_id is not None else 151858,
            image_token_id=int(image_token_id) if image_token_id is not None else 151858,
            slice_mode=str(slice_mode) if slice_mode else None,
            query_num=int(query_num),
        )


def get_image_token_id(model_path: str) -> int:
    """Get the image placeholder token ID from config.

    This is the token ID that marks where image embeddings should be inserted.
    """
    spec = MiniCPMVSpec.from_model_path(model_path)
    return spec.image_token_id
