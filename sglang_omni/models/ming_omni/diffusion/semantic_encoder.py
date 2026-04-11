# SPDX-License-Identifier: Apache-2.0
"""Semantic text encoder using Ming's LLM + connector.

Loads the BailingMM2 model (LLM + connector + query tokens) to produce
2560-dim semantic embeddings for ZImage diffusion conditioning.

Architecture:
    text -> LLM tokenizer -> input_ids
    -> append multiscale query tokens (4x4, 8x8, 16x16)
    -> LLM forward pass (hidden_size=4096)
    -> extract query token hidden states (highest scale = 256 tokens)
    -> proj_in: Linear(4096, connector_hidden)
    -> connector: non-causal transformer
    -> proj_out: Linear(connector_hidden, 2560)
    -> L2 normalize
    -> [B, 256, 2560] semantic embeddings
"""

from __future__ import annotations

import logging
import sys

import torch

logger = logging.getLogger(__name__)


class MingSemanticEncoder:
    """Wraps the Ming LLM + connector to produce semantic condition embeddings.

    Usage::

        encoder = MingSemanticEncoder()
        encoder.load(model_path, device)
        condition_embeds, negative_embeds = encoder.encode("A cat on a windowsill")
        # condition_embeds: list of [num_tokens, 2560] tensors
        # negative_embeds: list of [num_tokens, 2560] tensors (zeros)
    """

    def __init__(self) -> None:
        self._model = None
        self._tokenizer = None
        self._device: torch.device | None = None

    def load(
        self,
        model_path: str,
        device: torch.device,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        """Load the BailingMM2 model with LLM + connector components.

        Args:
            model_path: Path to the Ming model directory.
            device: Target device.
            dtype: Model dtype (default bf16).
        """
        self._device = device
        logger.info("[SemanticEncoder] Loading BailingMM2 from %s", model_path)

        # Try loading via AutoModelForCausalLM with trust_remote_code first.
        # If the custom kwargs aren't handled, fall back to direct import.
        try:
            from transformers import AutoModelForCausalLM

            self._model = AutoModelForCausalLM.from_pretrained(
                model_path,
                trust_remote_code=True,
                torch_dtype=dtype,
                load_image_gen=True,
                load_image_gen_others=True,
                load_image_gen_diffusion=False,
                load_vlm=True,
                device_map=str(device),
            )
        except TypeError:
            # AutoModel may not pass custom kwargs; import the class directly.
            logger.info(
                "[SemanticEncoder] Falling back to direct model class import"
            )
            if model_path not in sys.path:
                sys.path.insert(0, model_path)
            from modeling_bailingmm2 import (  # type: ignore[import-not-found]
                BailingMM2NativeForConditionalGeneration,
            )

            self._model = (
                BailingMM2NativeForConditionalGeneration.from_pretrained(
                    model_path,
                    torch_dtype=dtype,
                    load_image_gen=True,
                    load_image_gen_others=True,
                    load_image_gen_diffusion=False,
                    load_vlm=True,
                    device_map=str(device),
                )
            )

        self._model.eval()
        logger.info("[SemanticEncoder] Model loaded on %s", device)

        from transformers import AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True
        )
        logger.info(
            "[SemanticEncoder] Tokenizer loaded (vocab_size=%d)",
            len(self._tokenizer),
        )

    @torch.no_grad()
    def encode(
        self,
        text: str | list[str],
        max_length: int = 512,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Encode text into semantic condition embeddings.

        Args:
            text: Text prompt(s) to encode.
            max_length: Max token length for text.

        Returns:
            (condition_embeds, negative_condition_embeds):
                Each is a list of tensors with shape ``[num_tokens, 2560]``.
        """
        if self._model is None or self._tokenizer is None:
            raise RuntimeError(
                "Semantic encoder not loaded. Call load() first."
            )

        if isinstance(text, str):
            text = [text]

        inputs = self._tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        input_ids = inputs.input_ids.to(self._device)
        attention_mask = inputs.attention_mask.to(self._device)

        condition_embeds = self._model.get_condition_embeds_for_image_gen(
            input_ids=input_ids,
            attention_mask=attention_mask,
            image_embeds=None,
            position_ids=None,
            use_cache=False,
            image_grid_thw=None,
            llm_hidden_states=None,
        )
        # condition_embeds: [B, 256, 2560]

        negative_condition_embeds = condition_embeds * 0.0

        pos_list = list(condition_embeds.unbind(dim=0))
        neg_list = list(negative_condition_embeds.unbind(dim=0))

        return pos_list, neg_list

    def unload(self) -> None:
        """Release GPU memory."""
        if self._model is not None:
            del self._model
            self._model = None
        self._tokenizer = None
        torch.cuda.empty_cache()
