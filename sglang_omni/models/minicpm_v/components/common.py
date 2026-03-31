# SPDX-License-Identifier: Apache-2.0
"""Shared helpers for MiniCPM-V components.

Supports both MiniCPM-V 2.6 and MiniCPM-V 4.5:
- 2.6: SigLIP-400M + MiniCPM-3.0 LLM + Perceiver Resampler
- 4.5: SigLIP2-400M + Qwen3-8B LLM + 3D-Resampler
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Literal

from sglang_omni.utils import load_hf_config

logger = logging.getLogger(__name__)

# Weight prefix constants (same for 2.6 and 4.5)
# These match the HuggingFace checkpoint key structure
VISUAL_PREFIX = ("vpm.", "model.vpm.")  # SigLIP/SigLIP2 ViT
RESAMPLER_PREFIX = ("resampler.", "model.resampler.")  # Perceiver/3D Resampler
LLM_PREFIX = ("llm.", "model.llm.")  # MiniCPM-3.0 or Qwen3 LLM backbone
AUDIO_PREFIX = ("apm.", "model.apm.")  # MiniCPM-o only (Whisper)

# Version constants
MINICPM_VERSION_26 = 2.6
MINICPM_VERSION_45 = 4.5


def load_minicpm_config(model_path: str, local_files_only: bool = True) -> Any:
    """Load the full MiniCPM-V config from HuggingFace.

    MiniCPM-V uses a flat config structure unlike Qwen3-Omni's nested thinker_config.
    """
    return load_hf_config(
        model_path, trust_remote_code=True, local_files_only=local_files_only
    )


def get_minicpm_version(model_path: str) -> float:
    """Detect MiniCPM-V version from config.

    Returns:
        2.6 for MiniCPM-V 2.6 / MiniCPM-o 2.6
        4.5 for MiniCPM-V 4.5 / MiniCPM-o 4.5
    """
    cfg = load_minicpm_config(model_path)
    version = getattr(cfg, "version", None)
    if version is not None:
        return float(version)

    # Fallback heuristics:
    # 4.5 uses Qwen3 with vocab_size=151748, batch_3d_resampler=True
    # 2.6 uses MiniCPM-3.0 with smaller vocab
    vocab_size = getattr(cfg, "vocab_size", 0)
    has_3d_resampler = getattr(cfg, "batch_3d_resampler", False)

    if vocab_size >= 150000 or has_3d_resampler:
        logger.info(f"Detected MiniCPM-V 4.5 based on config heuristics")
        return MINICPM_VERSION_45

    return MINICPM_VERSION_26


def is_minicpm_v45(model_path: str) -> bool:
    """Check if model is MiniCPM-V 4.5 or later."""
    return get_minicpm_version(model_path) >= MINICPM_VERSION_45


def get_llm_type(model_path: str) -> Literal["minicpm", "qwen3"]:
    """Determine the LLM backbone type.

    Returns:
        'minicpm' for MiniCPM-V 2.6 (MiniCPM-3.0 backbone)
        'qwen3' for MiniCPM-V 4.5 (Qwen3-8B backbone)
    """
    if is_minicpm_v45(model_path):
        return "qwen3"
    return "minicpm"


def load_llm_config(model_path: str, local_files_only: bool = True) -> Any:
    """Load the LLM config from MiniCPM-V.

    For 2.6: MiniCPM-3.0 Llama-style LLM config
    For 4.5: Qwen3-8B config (different attention patterns)
    """
    cfg = load_minicpm_config(model_path, local_files_only=local_files_only)
    # Both versions store LLM config on main config object
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

    Supports both MiniCPM-V 2.6 and 4.5:
    - 2.6: SigLIP-400M + MiniCPM-3.0 LLM
    - 4.5: SigLIP2-400M + Qwen3-8B LLM + 3D-Resampler

    Key differences from Qwen3-Omni:
    - Uses `im_start_id` and `im_end_id` for image boundary tokens
    - Uses `slice_mode` for dynamic image slicing configuration
    - No spatial_merge_size (Resampler handles token reduction)
    """

    model_path: str
    version: float  # 2.6 or 4.5
    llm_type: Literal["minicpm", "qwen3"]  # LLM backbone type
    im_start_id: int
    im_end_id: int
    image_token_id: int  # The actual image placeholder token ID
    slice_mode: str | None
    query_num: int  # Number of query tokens per slice in Resampler
    # Vision encoder config
    vision_hidden_size: int
    vision_num_layers: int
    patch_size: int
    # LLM config
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    vocab_size: int
    # 4.5 specific
    batch_3d_resampler: bool
    max_slice_nums: int

    @classmethod
    def from_model_path(cls, model_path: str) -> "MiniCPMVSpec":
        cfg = load_minicpm_config(model_path)
        version = get_minicpm_version(model_path)
        llm_type = get_llm_type(model_path)

        # MiniCPM-V uses im_start_id/im_end_id for image boundaries
        im_start_id = getattr(cfg, "im_start_id", None)
        im_end_id = getattr(cfg, "im_end_id", None)
        slice_mode = getattr(cfg, "slice_mode", None)

        # MiniCPM-V may use im_token_id or similar for the actual placeholder
        image_token_id = getattr(cfg, "im_token_id", None)
        if image_token_id is None:
            image_token_id = getattr(cfg, "image_token_id", None)
        if image_token_id is None:
            image_token_id = im_end_id

        # Vision config
        vision_cfg = getattr(cfg, "vision_config", None) or {}
        if hasattr(vision_cfg, "to_dict"):
            vision_cfg = vision_cfg.to_dict()
        elif not isinstance(vision_cfg, dict):
            vision_cfg = {k: getattr(vision_cfg, k, None) for k in ["hidden_size", "num_hidden_layers", "patch_size", "query_num"]}

        vision_hidden_size = vision_cfg.get("hidden_size", 1152 if version >= 4.5 else 1024)
        vision_num_layers = vision_cfg.get("num_hidden_layers", 27 if version >= 4.5 else 24)
        patch_size = vision_cfg.get("patch_size", 14)
        query_num = getattr(cfg, "query_num", vision_cfg.get("query_num", 64))

        # Slice config
        slice_cfg = getattr(cfg, "slice_config", None) or {}
        if hasattr(slice_cfg, "to_dict"):
            slice_cfg = slice_cfg.to_dict()
        elif not isinstance(slice_cfg, dict):
            slice_cfg = {}
        max_slice_nums = slice_cfg.get("max_slice_nums", 9)

        # LLM config (differs between 2.6 MiniCPM-3.0 and 4.5 Qwen3)
        hidden_size = getattr(cfg, "hidden_size", 4096)
        num_hidden_layers = getattr(cfg, "num_hidden_layers", 36 if version >= 4.5 else 32)
        num_attention_heads = getattr(cfg, "num_attention_heads", 32)
        num_kv_heads = getattr(cfg, "num_key_value_heads", 8 if version >= 4.5 else num_attention_heads)
        vocab_size = getattr(cfg, "vocab_size", 151748 if version >= 4.5 else 128256)

        # 4.5 specific features
        batch_3d_resampler = getattr(cfg, "batch_3d_resampler", False)

        return cls(
            model_path=model_path,
            version=version,
            llm_type=llm_type,
            im_start_id=int(im_start_id) if im_start_id is not None else 151857,
            im_end_id=int(im_end_id) if im_end_id is not None else 151858,
            image_token_id=int(image_token_id) if image_token_id is not None else 151858,
            slice_mode=str(slice_mode) if slice_mode else None,
            query_num=int(query_num),
            vision_hidden_size=vision_hidden_size,
            vision_num_layers=vision_num_layers,
            patch_size=patch_size,
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_kv_heads,
            vocab_size=vocab_size,
            batch_3d_resampler=batch_3d_resampler,
            max_slice_nums=max_slice_nums,
        )


def get_image_token_id(model_path: str) -> int:
    """Get the image placeholder token ID from config.

    This is the token ID that marks where image embeddings should be inserted.
    """
    spec = MiniCPMVSpec.from_model_path(model_path)
    return spec.image_token_id
