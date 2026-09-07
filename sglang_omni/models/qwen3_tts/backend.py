# SPDX-License-Identifier: Apache-2.0
"""Execution backend selection for Qwen3-TTS."""

from __future__ import annotations

from enum import Enum


class Qwen3TTSBackend(str, Enum):
    """Model implementation selected for the Qwen3-TTS pipeline."""

    TORCH = "torch"
    MLX = "mlx"


def get_qwen3_tts_backend() -> Qwen3TTSBackend:
    """Return the model backend; device selection remains a Torch concern."""
    from sglang.srt.utils.tensor_bridge import use_mlx

    return Qwen3TTSBackend.MLX if use_mlx() else Qwen3TTSBackend.TORCH


__all__ = ["Qwen3TTSBackend", "get_qwen3_tts_backend"]
