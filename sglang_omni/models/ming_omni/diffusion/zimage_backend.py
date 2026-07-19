# SPDX-License-Identifier: Apache-2.0
"""Z-Image diffusion backend for explicit semantic conditioning.

Callers can pass thinker-fused semantic embeddings or explicitly configure the
backend to run the standalone Ming semantic encoder. ByT5 text rendering is also
available only when the stage is explicitly configured to load it.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import torch
from PIL import Image

from sglang_omni.models.ming_omni.diffusion.backend import (
    DiffusionBackend,
    ImageGenParams,
)

logger = logging.getLogger(__name__)

_QUOTE_PATTERNS = [
    r"\"(.*?)\"",
    r"\u201c(.*?)\u201d",
    r"\u2018(.*?)\u2019",
]


def _extract_render_text(prompt: str) -> str:
    """Extract the final quoted text span for optional ByT5 rendering."""
    texts: list[str] = []
    for pattern in _QUOTE_PATTERNS:
        texts.extend(re.findall(pattern, prompt))
    return texts[-1] if texts else ""


class ZImageBackend(DiffusionBackend):
    """Z-Image diffusion backend with precomputed semantic conditioning."""

    def __init__(self) -> None:
        self._pipe = None
        self._semantic_encoder = None
        self._text_encoder = None
        self._tokenizer = None
        self._device: torch.device | None = None

    def load_models(
        self,
        model_path: str,
        device: torch.device,
        *,
        ming_model_path: str | None = None,
        load_semantic_encoder: bool = False,
        load_byt5_text_encoder: bool = False,
    ) -> None:
        self._device = device

        # Fail fast on misconfiguration before loading the DiT (~22GB).
        if load_semantic_encoder and ming_model_path is None:
            raise ValueError("load_semantic_encoder=True requires ming_model_path")
        if load_byt5_text_encoder:
            if ming_model_path is None:
                raise ValueError("load_byt5_text_encoder=True requires ming_model_path")
            byt5_dir = Path(ming_model_path) / "byt5"
            if not byt5_dir.exists():
                raise RuntimeError(
                    f"ByT5 text rendering requested but {byt5_dir} does not exist"
                )

        from diffusers import (
            AutoencoderKL,
            FlowMatchEulerDiscreteScheduler,
            ZImagePipeline,
            ZImageTransformer2DModel,
        )

        logger.info("[ZImage] Loading pipeline components from %s", model_path)

        scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            model_path, subfolder="scheduler"
        )
        scheduler.config["use_dynamic_shifting"] = True

        vae = AutoencoderKL.from_pretrained(
            model_path, subfolder="vae", torch_dtype=torch.bfloat16
        )

        transformer = ZImageTransformer2DModel.from_pretrained(
            model_path, subfolder="transformer", torch_dtype=torch.bfloat16
        )
        logger.info(
            "[ZImage] Transformer loaded (cap_feat_dim=%d)",
            transformer.config.cap_feat_dim,
        )

        self._pipe = ZImagePipeline(
            scheduler=scheduler,
            vae=vae,
            transformer=transformer,
            text_encoder=None,
            tokenizer=None,
        )
        self._pipe = self._pipe.to(device)
        logger.info("[ZImage] Pipeline assembled on %s", device)

        if load_semantic_encoder:
            from sglang_omni.models.ming_omni.diffusion.semantic_encoder import (
                MingSemanticEncoder,
            )

            self._semantic_encoder = MingSemanticEncoder()
            self._semantic_encoder.load(ming_model_path, device)
            logger.info("[ZImage] Standalone semantic encoder ready")

        if load_byt5_text_encoder:
            from sglang_omni.models.ming_omni.diffusion.byt5_encoder import (
                load_byt5_text_encoder,
            )

            self._text_encoder, self._tokenizer = load_byt5_text_encoder(
                ming_model_path, device, dtype=torch.bfloat16
            )
            logger.info("[ZImage] ByT5 text encoder ready")

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
            raise RuntimeError("ZImage pipeline not loaded")

        generator = None
        if params.seed is not None:
            generator = torch.Generator(device=self._device).manual_seed(params.seed)

        if condition_embeds is not None:
            prompt_embeds = condition_embeds
            neg_embeds = (
                negative_condition_embeds
                if negative_condition_embeds is not None
                else [e * 0.0 for e in condition_embeds]
            )
        elif self._semantic_encoder is not None:
            prompt_embeds, neg_embeds = self._semantic_encoder.encode(prompt)
        else:
            raise RuntimeError(
                "standalone semantic encoder unavailable: "
                "condition_embeds were not provided"
            )

        if params.enable_text_rendering:
            render_text = _extract_render_text(prompt)
            if render_text:
                if self._text_encoder is None or self._tokenizer is None:
                    raise RuntimeError(
                        "ByT5 text rendering unavailable: encoder was not loaded"
                    )
                byt5_pos, byt5_neg = self._text_encoder.encode(
                    render_text,
                    tokenizer=self._tokenizer,
                    device=self._device,
                    max_length=256,
                )
                prompt_embeds = [
                    torch.cat([sem, byt.to(sem.device)], dim=0)
                    for sem, byt in zip(prompt_embeds, byt5_pos)
                ]
                neg_embeds = [
                    torch.cat([nsem, nbyt.to(nsem.device)], dim=0)
                    for nsem, nbyt in zip(neg_embeds, byt5_neg)
                ]

        result = self._pipe(
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=neg_embeds,
            height=params.height,
            width=params.width,
            num_inference_steps=params.num_inference_steps,
            guidance_scale=params.guidance_scale,
            generator=generator,
            max_sequence_length=512,
        )

        return result.images[0]

    def unload(self) -> None:
        if self._pipe is not None:
            del self._pipe
            self._pipe = None
        if self._semantic_encoder is not None:
            self._semantic_encoder.unload()
            self._semantic_encoder = None
        if self._text_encoder is not None:
            del self._text_encoder
            self._text_encoder = None
        self._tokenizer = None
        torch.cuda.empty_cache()
