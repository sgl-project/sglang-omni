# SPDX-License-Identifier: Apache-2.0
"""Abstract diffusion backend interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch
from PIL import Image


@dataclass
class ImageGenParams:
    """Parameters for a single image generation request."""

    width: int = 1024
    height: int = 1024
    num_inference_steps: int = 28
    # Tuned for the Z-Image-Turbo DiT (distilled, low-CFG); high CFG washes out.
    guidance_scale: float = 2.0
    seed: int | None = None
    negative_prompt: str = ""
    semantic_source: str | None = None
    enable_text_rendering: bool = False


class DiffusionBackend(ABC):
    """Small interface for model-specific diffusion inference."""

    @abstractmethod
    def load_models(self, model_path: str, device: torch.device) -> None:
        """Load DiT, VAE, text encoder(s), and scheduler."""
        ...

    @abstractmethod
    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        params: ImageGenParams,
        *,
        condition_embeds: list[torch.Tensor] | None = None,
        negative_condition_embeds: list[torch.Tensor] | None = None,
    ) -> Image.Image:
        """Denoise and decode an image.

        Callers either pass precomputed semantic condition embeddings or select
        an explicitly loaded backend-owned semantic encoder. Text-only backend
        fallback is intentionally outside this interface.
        """
        ...

    def unload(self) -> None:
        """Release GPU memory."""
