# SPDX-License-Identifier: Apache-2.0
"""Thinker component for Qwen3-Omni."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from accelerate import init_empty_weights
from transformers.models.qwen3_omni_moe import modeling_qwen3_omni_moe as hf_modeling

from sglang_omni.models.qwen3_omni.components.common import load_thinker_config
from sglang_omni.models.utils.hf import instantiate_module
from sglang_omni.models.weight_loader import load_module, resolve_dtype

TEXT_MODEL_PREFIX = ("thinker.model.", "model.")
LM_HEAD_PREFIX = ("thinker.lm_head.", "lm_head.")
TEXT_MODEL_CLASS = hf_modeling.Qwen3OmniMoeThinkerTextModel


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
    # Prefer text_config.tie_word_embeddings when it exists, as nested configs
    # may have different settings than the top-level config.
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


def _build_thinker_shell(
    thinker_cfg: Any,
) -> hf_modeling.Qwen3OmniMoeThinkerForConditionalGeneration:
    with init_empty_weights():
        thinker = hf_modeling.Qwen3OmniMoeThinkerForConditionalGeneration(thinker_cfg)
    return thinker


class Qwen3OmniSplitThinker(nn.Module):
    """Thinker wrapper that accepts precomputed encoder embeddings."""

    def __init__(
        self,
        model_id: str,
        *,
        device: str = "cuda",
        dtype: str | torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self._device = torch.device(device)
        torch_dtype = resolve_dtype(dtype)
        thinker_cfg = load_thinker_config(model_id)

        text_model = _build_text_model(
            model_id,
            thinker_cfg=thinker_cfg,
            torch_dtype=torch_dtype,
        )
        lm_head = _build_lm_head(
            model_id,
            thinker_cfg=thinker_cfg,
            torch_dtype=torch_dtype,
        )
        _maybe_tie_weights(config=thinker_cfg, text_model=text_model, lm_head=lm_head)

        self.thinker = _build_thinker_shell(thinker_cfg)
        self.thinker.model = text_model
        self.thinker.lm_head = lm_head
        # Move only the text model and LM head to the thinker device.
        self.thinker.model = self.thinker.model.to(self._device)
        self.thinker.lm_head = self.thinker.lm_head.to(self._device)

    def to(self, *args, **kwargs):  # type: ignore[override]
        """Move only the active text components; avoid meta tensor errors."""
        device = kwargs.get("device")
        dtype = kwargs.get("dtype")
        if device is None and args:
            device = args[0]

        if device is not None:
            self._device = torch.device(device)

        if device is not None and dtype is not None:
            self.thinker.model = self.thinker.model.to(device=device, dtype=dtype)
            self.thinker.lm_head = self.thinker.lm_head.to(device=device, dtype=dtype)
        elif device is not None:
            self.thinker.model = self.thinker.model.to(device=device)
            self.thinker.lm_head = self.thinker.lm_head.to(device=device)
        elif dtype is not None:
            self.thinker.model = self.thinker.model.to(dtype=dtype)
            self.thinker.lm_head = self.thinker.lm_head.to(dtype=dtype)
        return self

    def _merge_embeddings(
        self,
        *,
        input_ids: torch.Tensor,
        inputs_embeds: torch.Tensor,
        image_embeds: torch.Tensor | None,
        audio_embeds: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        image_mask, _, audio_mask = self.thinker.get_placeholder_mask(
            input_ids,
            inputs_embeds=inputs_embeds,
            image_features=image_embeds,
        )
        image_mask_out = image_mask if image_embeds is not None else None

        if image_embeds is not None:
            image_embeds = image_embeds.to(
                device=inputs_embeds.device,
                dtype=inputs_embeds.dtype,
            )
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

        if audio_embeds is not None:
            audio_token_count = int(
                (input_ids == self.thinker.config.audio_token_id).sum().item()
            )
            if audio_token_count != int(audio_embeds.shape[0]):
                raise ValueError(
                    "Audio placeholder count mismatch: "
                    f"tokens={audio_token_count} embeds={audio_embeds.shape[0]}"
                )
            audio_embeds = audio_embeds.to(
                device=inputs_embeds.device,
                dtype=inputs_embeds.dtype,
            )
            inputs_embeds = inputs_embeds.masked_scatter(audio_mask, audio_embeds)

        return inputs_embeds, image_mask_out

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        image_embeds: torch.Tensor | list[torch.Tensor] | None = None,
        audio_embeds: torch.Tensor | list[torch.Tensor] | None = None,
        **kwargs: Any,
    ):
        image_embeds_t = _concat_features(image_embeds)
        audio_embeds_t = _concat_features(audio_embeds)
        deepstack_visual_embeds = kwargs.pop("deepstack_visual_embeds", None)
        visual_pos_masks = kwargs.pop("visual_pos_masks", None)

        if inputs_embeds is not None:
            inputs_embeds = inputs_embeds.to(self._device)

        # Track whether we manually merged embeddings
        manual_merge_done = False

        if inputs_embeds is None and (
            image_embeds_t is not None or audio_embeds_t is not None
        ):
            inputs_embeds = self.thinker.get_input_embeddings()(
                input_ids.to(self._device)
            )

        if inputs_embeds is not None and (
            image_embeds_t is not None or audio_embeds_t is not None
        ):
            image_embeds_t = (
                image_embeds_t.to(self._device) if image_embeds_t is not None else None
            )
            audio_embeds_t = (
                audio_embeds_t.to(self._device) if audio_embeds_t is not None else None
            )
            inputs_embeds, image_mask = self._merge_embeddings(
                input_ids=input_ids.to(self._device),
                inputs_embeds=inputs_embeds,
                image_embeds=image_embeds_t,
                audio_embeds=audio_embeds_t,
            )
            manual_merge_done = True
        else:
            image_mask = None

        # Only pass deepstack_visual_embeds if we haven't already merged embeddings
        # This avoids the HF model trying to merge visual features twice
        if deepstack_visual_embeds is not None and not manual_merge_done:
            kwargs["deepstack_visual_embeds"] = deepstack_visual_embeds
            if visual_pos_masks is not None:
                kwargs["visual_pos_masks"] = visual_pos_masks
            elif image_mask is not None:
                kwargs["visual_pos_masks"] = image_mask

        return self.thinker(
            input_ids=input_ids.to(self._device),
            attention_mask=(
                attention_mask.to(self._device)
                if isinstance(attention_mask, torch.Tensor)
                else None
            ),
            inputs_embeds=inputs_embeds,
            **kwargs,
        )
