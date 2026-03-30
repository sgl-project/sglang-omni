# SPDX-License-Identifier: Apache-2.0
"""Split thinker component for Ming-Omni.

Wraps the BailingMoeV2TextModel so it can accept pre-computed
audio encoder embeddings and merge them into the input sequence.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

from sglang_omni.models.ming_omni.thinker import (
    BailingMoeV2Config,
    BailingMoeV2TextModel,
)
from sglang_omni.models.weight_loader import (
    load_weights_by_prefix,
    resolve_dtype,
    resolve_model_path,
)
from sglang_omni.vendor.sglang.core import ForwardBatch
from sglang_omni.vendor.sglang.layers import ReplicatedLinear

logger = logging.getLogger(__name__)

# Prefixes for loading weights from Ming checkpoint
TEXT_MODEL_PREFIXES = ("model.model.", "model.")
LM_HEAD_PREFIXES = ("model.lm_head.", "lm_head.")


class MingSplitThinker(nn.Module):
    """Ming-Omni thinker that accepts pre-computed encoder embeddings.

    This wraps BailingMoeV2TextModel and adds:
    - Split weight loading (text model + LM head separately)
    - Audio embedding merging into input token embeddings
    - LM head for next-token prediction
    """

    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        dtype: str | None = None,
    ):
        super().__init__()
        self._model_path = model_path
        self._device = device
        self._dtype = resolve_dtype(dtype) if dtype else torch.bfloat16

        # Load config
        self._config = self._load_config()

        # Build text model
        self.text_model = BailingMoeV2TextModel(self._config)

        # Build LM head
        self.lm_head = ReplicatedLinear(
            self._config.hidden_size,
            self._config.vocab_size,
            bias=False,
        )

        # Load weights
        self._load_weights()

    def _load_config(self) -> BailingMoeV2Config:
        """Load BailingMoeV2Config from the Ming-Omni checkpoint."""
        resolved = resolve_model_path(self._model_path)
        config_path = Path(resolved) / "config.json"
        with open(config_path) as f:
            raw = json.load(f)

        llm_raw = raw.get("llm_config", raw)
        return BailingMoeV2Config(
            **{
                k: v
                for k, v in llm_raw.items()
                if k in BailingMoeV2Config.__init__.__code__.co_varnames
            }
        )

    def _load_weights(self) -> None:
        """Load text model and LM head weights."""
        logger.info("Loading Ming thinker text model from %s", self._model_path)

        # Load text model weights with prefix stripping
        text_weights = load_weights_by_prefix(
            self._model_path, prefix=TEXT_MODEL_PREFIXES
        )
        self.text_model.load_weights(text_weights)

        # Load LM head
        lm_head_weights = load_weights_by_prefix(
            self._model_path, prefix=LM_HEAD_PREFIXES
        )
        params_dict = dict(self.lm_head.named_parameters())
        for name, weight in lm_head_weights:
            if name in params_dict:
                param = params_dict[name]
                param.data.copy_(weight.to(param.dtype))

        # Handle weight tying
        if self._config.tie_word_embeddings:
            self._maybe_tie_weights()

    def _maybe_tie_weights(self) -> None:
        """Tie LM head weights to embedding weights if configured."""
        embed_weight = self.text_model.embed_tokens.weight
        lm_params = dict(self.lm_head.named_parameters())
        if "weight" in lm_params:
            lm_params["weight"].data = embed_weight.data

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: Optional[torch.Tensor] = None,
        audio_embeds: Optional[torch.Tensor] = None,
        audio_embed_lengths: Optional[torch.Tensor] = None,
        audio_placeholder_loc_lens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass with optional audio embedding injection.

        Args:
            input_ids: Token IDs [batch, seq_len]
            positions: Position IDs for RoPE
            forward_batch: SGLang forward batch context
            input_embeds: Pre-computed embeddings (overrides input_ids)
            audio_embeds: Audio encoder output [batch, audio_len, hidden]
            audio_embed_lengths: Per-segment audio lengths [batch, num_segments]
            audio_placeholder_loc_lens: Placeholder positions [batch, num_segments, 2]
        """
        # Get text embeddings
        if input_embeds is None:
            input_embeds = self.text_model.embed_tokens(input_ids)

        # Merge audio embeddings at placeholder positions
        if audio_embeds is not None and audio_placeholder_loc_lens is not None:
            input_embeds = self._merge_audio_embeddings(
                input_embeds,
                audio_embeds,
                audio_embed_lengths,
                audio_placeholder_loc_lens,
            )

        # Forward through text model
        hidden_states = self.text_model(
            input_ids=input_ids,
            positions=positions,
            forward_batch=forward_batch,
            input_embeds=input_embeds,
        )

        # LM head
        logits, _ = self.lm_head(hidden_states)
        return logits

    def _merge_audio_embeddings(
        self,
        input_embeds: torch.Tensor,
        audio_embeds: torch.Tensor,
        audio_embed_lengths: Optional[torch.Tensor],
        placeholder_loc_lens: torch.Tensor,
    ) -> torch.Tensor:
        """Patch audio embeddings into text embeddings at placeholder positions.

        Follows Ming's patch_continuous_features() logic.
        """
        batch_size = input_embeds.shape[0]
        for i in range(batch_size):
            audio_feat_start = 0
            for j in range(placeholder_loc_lens.shape[1]):
                placeholder_start = int(placeholder_loc_lens[i, j, 0].item())
                placeholder_len = int(placeholder_loc_lens[i, j, 1].item())
                if placeholder_len <= 0:
                    break

                if audio_embed_lengths is not None:
                    feat_len = int(audio_embed_lengths[i, j].item())
                else:
                    feat_len = placeholder_len

                target_len = min(feat_len, placeholder_len)
                input_embeds[i, placeholder_start : placeholder_start + target_len] = (
                    audio_embeds[i, audio_feat_start : audio_feat_start + target_len]
                )
                audio_feat_start += feat_len

        return input_embeds
