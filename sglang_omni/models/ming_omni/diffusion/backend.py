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
    guidance_scale: float = 7.0
    seed: int | None = None
    negative_prompt: str = ""


class DiffusionBackend(ABC):
    """Unified interface for SD3 and Z-Image diffusion inference."""

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
        """Full text-to-image pipeline: encode text, denoise, decode.

        If *condition_embeds* is provided, uses pre-computed semantic
        embeddings directly (bypasses text encoding). Otherwise falls back
        to the backend's built-in text encoder.
        """
        ...

    def unload(self) -> None:
        """Release GPU memory."""
