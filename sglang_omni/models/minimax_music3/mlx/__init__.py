# SPDX-License-Identifier: Apache-2.0
"""Native MLX backend for MiniMax Music 3."""

from .loader import load_mlx_acoustic_model, load_mlx_ar_model

__all__ = ["load_mlx_acoustic_model", "load_mlx_ar_model"]
