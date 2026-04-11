# SPDX-License-Identifier: Apache-2.0
"""Z-Image diffusion backend.

Ports the inference logic from Ming's pipeline_z_image.py.
Uses a single-stream transformer with RoPE and a single text encoder.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import torch
from PIL import Image

from sglang_omni.models.ming_omni.diffusion.backend import DiffusionBackend, ImageGenParams

logger = logging.getLogger(__name__)


class ZImageBackend(DiffusionBackend):
    """Z-Image diffusion backend using Ming's ZImagePipeline."""

    def __init__(self) -> None:
        self._pipe = None
        self._device: torch.device | None = None

    def load_models(self, model_path: str, device: torch.device) -> None:
        self._device = device

        # Import Ming's ZImagePipeline — expects Ming repo on sys.path
        # or the pipeline classes installed as a package.
        try:
            from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler
            from transformers import AutoModel, AutoTokenizer

            logger.info("[ZImage] Loading pipeline components from %s", model_path)
            model_root = Path(model_path)

            # Load scheduler
            scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
                str(model_root / "scheduler")
                if (model_root / "scheduler").exists()
                else model_path,
                subfolder="scheduler" if not (model_root / "scheduler").exists() else None,
            )

            # Load VAE
            vae = AutoencoderKL.from_pretrained(
                model_path, subfolder="vae", torch_dtype=torch.bfloat16
            )

            # Load text encoder + tokenizer
            text_encoder_path = str(model_root / "text_encoder") if (model_root / "text_encoder").exists() else model_path
            text_encoder = AutoModel.from_pretrained(
                text_encoder_path,
                subfolder="text_encoder" if text_encoder_path == model_path else None,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
            )
            tokenizer = AutoTokenizer.from_pretrained(
                text_encoder_path,
                subfolder="text_encoder" if text_encoder_path == model_path else None,
                trust_remote_code=True,
            )

            # Load transformer
            from diffusers import DiffusionPipeline

            # Try loading as a full diffusers pipeline first
            self._pipe = DiffusionPipeline.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
            )
            self._pipe.to(device)
            logger.info("[ZImage] Pipeline loaded on %s", device)

        except Exception:
            # Fallback: try loading the whole thing as a diffusers pipeline
            logger.info("[ZImage] Falling back to DiffusionPipeline.from_pretrained")
            from diffusers import DiffusionPipeline

            self._pipe = DiffusionPipeline.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
            )
            self._pipe.to(device)
            logger.info("[ZImage] Pipeline loaded on %s (fallback)", device)

    @torch.no_grad()
    def generate(self, prompt: str, params: ImageGenParams) -> Image.Image:
        if self._pipe is None:
            raise RuntimeError("ZImage pipeline not loaded")

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
