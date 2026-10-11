# SPDX-License-Identifier: MIT
# Copyright (c) 2024 Prince Canuma and contributors.
# Derived from Blaizzy/mlx-audio CosyVoice3 PR #861 (commit 5272f213f8cc).
# Based on FunAudioLLM/CosyVoice (Apache-2.0, Copyright 2024-2025 Alibaba Inc).
# Modified for the non-streaming sglang-omni vocoder contract.

"""Native MLX Flow/DiT and HiFT backend for Fun-CosyVoice3."""

from .loader import FunCosyVoice3MlxVocoder

__all__ = ["FunCosyVoice3MlxVocoder"]
