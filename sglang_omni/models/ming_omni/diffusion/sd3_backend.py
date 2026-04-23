# SPDX-License-Identifier: Apache-2.0
"""SD3 (Stable Diffusion 3) diffusion backend.

Ports the inference logic from Ming's pipeline_stable_diffusion_3.py.
Uses MMDiT (dual-stream joint attention) with CLIP + T5 text encoders.
"""

from __future__ import annotations

import logging

import torch
from PIL import Image

from sglang_omni.models.ming_omni.diffusion.backend import (
    DiffusionBackend,
    ImageGenParams,
)

logger = logging.getLogger(__name__)


class SD3Backend(DiffusionBackend):
    """SD3 diffusion backend using the diffusers StableDiffusion3Pipeline."""

    def __init__(self) -> None:
        self._pipe = None
        self._device: torch.device | None = None

    def load_models(self, model_path: str, device: torch.device) -> None:
        from diffusers import StableDiffusion3Pipeline

        logger.info("[SD3] Loading pipeline from %s", model_path)
        self._device = device
        self._pipe = StableDiffusion3Pipeline.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
        )
        self._pipe.to(device)
        logger.info("[SD3] Pipeline loaded on %s", device)

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        params: ImageGenParams,
        *,
        condition_embeds: list[torch.Tensor] | None = None,
        negative_condition_embeds: list[torch.Tensor] | None = None,
    ) -> Image.Image:
        if self._pipe is None:
            raise RuntimeError("SD3 pipeline not loaded")

        generator = None
        if params.seed is not None:
            generator = torch.Generator(device=self._device).manual_seed(params.seed)

        result = self._pipe(
            prompt=prompt,
            negative_prompt=params.negative_prompt or None,
            height=params.height,
            width=params.width,
            num_inference_steps=params.num_inference_steps,
            guidance_scale=params.guidance_scale,
            generator=generator,
            output_type="pil",
        )
        return result.images[0]

    def unload(self) -> None:
        if self._pipe is not None:
            del self._pipe
            self._pipe = None
            torch.cuda.empty_cache()
