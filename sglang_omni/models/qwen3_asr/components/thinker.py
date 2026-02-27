# SPDX-License-Identifier: Apache-2.0
"""Thinker component for Qwen3-ASR."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from sglang_omni.models.qwen3_asr.components.common import load_thinker_config
from sglang_omni.models.qwen3_asr.modeling import modeling_qwen3_asr as hf_modeling
from sglang_omni.models.utils.hf import instantiate_module
from sglang_omni.models.weight_loader import load_module, resolve_dtype

TEXT_MODEL_PREFIX = ("thinker.model.", "model.")
LM_HEAD_PREFIX = ("thinker.lm_head.", "lm_head.")
TEXT_MODEL_CLASS = hf_modeling.Qwen3ASRThinkerTextModel


def _concat_features(value: Any) -> torch.Tensor | None:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, (list, tuple)):
        tensors = [v for v in value if isinstance(v, torch.Tensor)]
        if not tensors:
            return None
        return torch.cat(tensors, dim=0)
    return None


def _should_tie_embeddings(config: Any) -> bool:
    text_config = getattr(config, "text_config", None)
    if text_config is not None:
        return bool(getattr(text_config, "tie_word_embeddings", False))
    return bool(getattr(config, "tie_word_embeddings", False))


def _maybe_tie_weights(
    *,
    config: Any,
    text_model: nn.Module,
    lm_head: nn.Module,
) -> None:
    if not _should_tie_embeddings(config):
        return
    embed_tokens = getattr(text_model, "embed_tokens", None)
    if isinstance(embed_tokens, nn.Module) and hasattr(embed_tokens, "weight"):
        lm_head.weight = embed_tokens.weight


def _build_text_model(
    model_id: str,
    *,
    thinker_cfg: Any,
    torch_dtype: torch.dtype | None,
) -> nn.Module:
    text_cfg = thinker_cfg.text_config
    text_model = instantiate_module(TEXT_MODEL_CLASS, text_cfg)
    return load_module(
        text_model,
        model_id,
        prefix=TEXT_MODEL_PREFIX,
        dtype=torch_dtype,
        device=None,
        strict=True,
    )


def _build_lm_head(
    model_id: str,
    *,
    thinker_cfg: Any,
    torch_dtype: torch.dtype | None,
) -> nn.Module:
    if "forced_aligner" in thinker_cfg.model_type:
        lm_head = nn.Linear(
            thinker_cfg.text_config.hidden_size,
            thinker_cfg.classify_num,
            bias=False,
        )
    else:
        lm_head = nn.Linear(
            thinker_cfg.text_config.hidden_size,
            thinker_cfg.text_config.vocab_size,
            bias=False,
        )
    if not _should_tie_embeddings(thinker_cfg):
        lm_head = load_module(
            lm_head,
            model_id,
            prefix=LM_HEAD_PREFIX,
            dtype=torch_dtype,
            device=None,
            strict=True,
        )
    return lm_head


class Qwen3ASRSplitThinker(hf_modeling.Qwen3ASRThinkerForConditionalGeneration):
    """Thinker wrapper that accepts precomputed encoder embeddings."""

    def __init__(
        self,
        model_id: str,
        *,
        device: str = "cuda",
        dtype: str | torch.dtype | None = None,
    ) -> None:
        self.config = load_thinker_config(model_id)
        super().__init__(self.config)

        self._device = torch.device(device)
        torch_dtype = resolve_dtype(dtype)

        text_model = _build_text_model(
            model_id,
            thinker_cfg=self.config,
            torch_dtype=torch_dtype,
        )
        lm_head = _build_lm_head(
            model_id,
            thinker_cfg=self.config,
            torch_dtype=torch_dtype,
        )
        _maybe_tie_weights(config=self.config, text_model=text_model, lm_head=lm_head)

        self.model = text_model
        self.lm_head = lm_head
        # In split mode, the audio tower is handled by another component.
        self.audio_tower = None

        self.to(device=self._device, dtype=torch_dtype)

    def get_audio_features(
        self,
        input_features: torch.FloatTensor,
        *args: Any,
        **kwargs: Any,
    ) -> torch.FloatTensor:
        # In split mode, input_features are already the computed audio features
        return input_features

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        past_key_values: Any | None = None,
        inputs_embeds: torch.Tensor | None = None,
        use_cache: bool | None = None,
        cache_position: torch.Tensor | None = None,
        audio_embeds: torch.Tensor | None = None,
        input_features: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> hf_modeling.Qwen3ASRThinkerCausalLMOutputWithPast:
        # Map audio_embeds or input_features to the expected parameter name
        final_input_features = (
            audio_embeds if audio_embeds is not None else input_features
        )
        if final_input_features is not None:
            kwargs["input_features"] = _concat_features(final_input_features)

        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        position_ids=None,
        use_cache=True,
        audio_embeds=None,
        input_features=None,
        **kwargs,
    ):
        # Prefer audio_embeds but fall back to input_features
        final_input_features = (
            audio_embeds if audio_embeds is not None else input_features
        )
        if final_input_features is not None:
            kwargs["input_features"] = final_input_features

        model_inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            position_ids=position_ids,
            use_cache=use_cache,
            **kwargs,
        )

        # In split mode, we pass the precomputed features back as input_features
        if final_input_features is not None:
            model_inputs["input_features"] = final_input_features

        if cache_position is not None and cache_position[0] != 0:
            model_inputs["input_features"] = None

        return model_inputs
