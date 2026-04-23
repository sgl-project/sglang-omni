# SPDX-License-Identifier: Apache-2.0
"""Z-Image diffusion backend with semantic + ByT5 text encoding.

Loads the ZImage pipeline components (transformer, VAE, scheduler) and
optionally the MingSemanticEncoder (LLM + connector) and/or ByT5 text
encoder.  Semantic conditioning (LLM-derived) produces meaningful images;
ByT5 provides supplementary text rendering control.
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

# Patterns for extracting quoted text from prompts (matching
# processing_bailingmm2.py:get_text_from_prompt).
_QUOTE_PATTERNS = [
    r"\"(.*?)\"",  # straight double quotes
    r"\u201c(.*?)\u201d",  # curly double quotes ""
    r"\u2018(.*?)\u2019",  # curly single quotes ''
]


def _extract_render_text(prompt: str) -> str:
    """Extract text-to-render from a prompt by finding quoted substrings.

    Mirrors Ming's ``processing_bailingmm2.py:get_text_from_prompt``.
    Returns the last quoted substring found, or empty string if none.
    """
    texts: list[str] = []
    for pattern in _QUOTE_PATTERNS:
        texts.extend(re.findall(pattern, prompt))
    return texts[-1] if texts else ""


class ZImageBackend(DiffusionBackend):
    """Z-Image diffusion backend with semantic text conditioning."""

    def __init__(self) -> None:
        self._pipe = None
        self._text_encoder = None  # ByT5 (supplementary)
        self._tokenizer = None  # ByT5 tokenizer
        self._semantic_encoder = None  # Ming LLM + connector (primary)
        self._device: torch.device | None = None

    def load_models(
        self,
        model_path: str,
        device: torch.device,
        *,
        skip_semantic_encoder: bool = False,
    ) -> None:
        self._device = device

        from diffusers import (
            AutoencoderKL,
            FlowMatchEulerDiscreteScheduler,
            ZImagePipeline,
            ZImageTransformer2DModel,
        )

        logger.info("[ZImage] Loading pipeline components from %s", model_path)

        # 1. Scheduler
        scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            model_path, subfolder="scheduler"
        )
        scheduler.config["use_dynamic_shifting"] = True

        # 2. VAE
        vae = AutoencoderKL.from_pretrained(
            model_path, subfolder="vae", torch_dtype=torch.bfloat16
        )

        # 3. Transformer (ZImageTransformer2DModel)
        transformer = ZImageTransformer2DModel.from_pretrained(
            model_path, subfolder="transformer", torch_dtype=torch.bfloat16
        )
        logger.info(
            "[ZImage] Transformer loaded (cap_feat_dim=%d)",
            transformer.config.cap_feat_dim,
        )

        # 4. Assemble pipeline (text encoding handled separately)
        self._pipe = ZImagePipeline(
            scheduler=scheduler,
            vae=vae,
            transformer=transformer,
            text_encoder=None,
            tokenizer=None,
        )
        self._pipe = self._pipe.to(device)
        logger.info("[ZImage] Pipeline assembled on %s", device)

        # 5. Load semantic encoder (LLM + connector) — primary
        if skip_semantic_encoder:
            logger.info(
                "[ZImage] Skipping semantic encoder loading "
                "(skip_semantic_encoder=True)"
            )
            self._semantic_encoder = None
        else:
            try:
                from sglang_omni.models.ming_omni.diffusion.semantic_encoder import (
                    MingSemanticEncoder,
                )

                self._semantic_encoder = MingSemanticEncoder()
                self._semantic_encoder.load(model_path, device)
                logger.info("[ZImage] Semantic encoder (LLM + connector) ready")
            except Exception as e:
                logger.warning(
                    "[ZImage] Failed to load semantic encoder: %s. "
                    "Falling back to ByT5-only mode.",
                    e,
                )
                self._semantic_encoder = None

        # 6. Load ByT5 text encoder + mapper (supplementary)
        byt5_dir = Path(model_path) / "byt5"
        if byt5_dir.exists():
            from sglang_omni.models.ming_omni.diffusion.byt5_encoder import (
                load_byt5_text_encoder,
            )

            self._text_encoder, self._tokenizer = load_byt5_text_encoder(
                model_path, device, dtype=torch.bfloat16
            )
            logger.info("[ZImage] ByT5 text encoder ready")
        else:
            logger.warning(
                "[ZImage] No byt5/ directory found at %s — "
                "ByT5 text encoding will not be available.",
                model_path,
            )

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

        # --- Build condition embeddings ---
        prompt_embeds: list[torch.Tensor]
        neg_embeds: list[torch.Tensor]

        if condition_embeds is not None:
            # Pre-computed embeddings provided (e.g., from thinker hidden states)
            prompt_embeds = condition_embeds
            neg_embeds = (
                negative_condition_embeds
                if negative_condition_embeds is not None
                else [e * 0.0 for e in condition_embeds]
            )

            # Optionally concatenate ByT5 embeddings for text rendering
            if self._text_encoder is not None and self._tokenizer is not None:
                render_text = _extract_render_text(prompt)
                if render_text:
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

        elif self._semantic_encoder is not None:
            # Semantic encoding via LLM + connector
            prompt_embeds, neg_embeds = self._semantic_encoder.encode(prompt)

            # Optionally concatenate ByT5 embeddings for text rendering.
            # ByT5 encodes only the RENDER TEXT (text between quotes in the
            # prompt), not the full scene description.  This matches Ming's
            # production flow: processing_bailingmm2.py:get_text_from_prompt
            # extracts quoted text, and encode() wraps it as 'Text "...". '.
            if self._text_encoder is not None and self._tokenizer is not None:
                render_text = _extract_render_text(prompt)
                if render_text:
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

        elif self._text_encoder is not None and self._tokenizer is not None:
            # Fallback: ByT5-only encoding (text rendering mode)
            logger.warning(
                "[ZImage] Using ByT5-only encoding (no semantic encoder). "
                "Images may show text rendering instead of semantic content."
            )
            render_text = _extract_render_text(prompt) or prompt
            prompt_embeds, neg_embeds = self._text_encoder.encode(
                render_text,
                tokenizer=self._tokenizer,
                device=self._device,
                max_length=256,
            )

        else:
            # No text encoder at all — random embeddings
            logger.warning(
                "[ZImage] No text encoder — generating with random embeddings"
            )
            cap_feat_dim = self._pipe.transformer.config.cap_feat_dim
            prompt_embeds = [
                torch.randn(77, cap_feat_dim, device=self._device, dtype=torch.bfloat16)
            ]
            neg_embeds = [
                torch.zeros(77, cap_feat_dim, device=self._device, dtype=torch.bfloat16)
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
        if self._text_encoder is not None:
            del self._text_encoder
            self._text_encoder = None
        self._tokenizer = None
        if self._semantic_encoder is not None:
            self._semantic_encoder.unload()
            self._semantic_encoder = None
        torch.cuda.empty_cache()
