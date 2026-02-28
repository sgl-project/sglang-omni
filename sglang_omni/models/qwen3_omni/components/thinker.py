# SPDX-License-Identifier: Apache-2.0
"""Thinker component for Qwen3-Omni."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from accelerate import init_empty_weights
from transformers.models.qwen3_omni_moe import modeling_qwen3_omni_moe as hf_modeling

from sglang_omni.models.qwen3_omni.components.common import load_thinker_config
from sglang_omni.models.weight_loader import load_module, resolve_dtype
from sglang_omni.utils import instantiate_module

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
        video_embeds: torch.Tensor | None,
        audio_embeds: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        # Follow HF implementation order: audio -> image -> video
        # Get masks as needed, matching HF's pattern of calling get_placeholder_mask
        # for each modality with updated inputs_embeds

        image_mask_out = None
        video_mask_out = None

        # 1. Process audio first (matches HF order)
        if audio_embeds is not None:
            _, _, audio_mask = self.thinker.get_placeholder_mask(
                input_ids,
                inputs_embeds=inputs_embeds,
            )
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

        # 2. Process image (matches HF order)
        if image_embeds is not None:
            image_mask, _, _ = self.thinker.get_placeholder_mask(
                input_ids,
                inputs_embeds=inputs_embeds,
                image_features=image_embeds,
            )
            image_mask_out = image_mask
            image_embeds = image_embeds.to(
                device=inputs_embeds.device,
                dtype=inputs_embeds.dtype,
            )
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

        # 3. Process video last (matches HF order)
        if video_embeds is not None:
            _, video_mask, _ = self.thinker.get_placeholder_mask(
                input_ids,
                inputs_embeds=inputs_embeds,
                video_features=video_embeds,
            )
            video_mask_out = video_mask
            video_token_count = int(
                (input_ids == self.thinker.config.video_token_id).sum().item()
            )
            if video_token_count != int(video_embeds.shape[0]):
                raise ValueError(
                    "Video placeholder count mismatch: "
                    f"tokens={video_token_count} embeds={video_embeds.shape[0]}"
                )
            video_embeds = video_embeds.to(
                device=inputs_embeds.device,
                dtype=inputs_embeds.dtype,
            )
            inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

        return inputs_embeds, image_mask_out, video_mask_out

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        image_embeds: torch.Tensor | list[torch.Tensor] | None = None,
        video_embeds: torch.Tensor | list[torch.Tensor] | None = None,
        audio_embeds: torch.Tensor | list[torch.Tensor] | None = None,
        **kwargs: Any,
    ):
        image_embeds_t = _concat_features(image_embeds)
        video_embeds_t = _concat_features(video_embeds)
        audio_embeds_t = _concat_features(audio_embeds)
        deepstack_visual_embeds = kwargs.pop("deepstack_visual_embeds", None)
        image_deepstack_visual_embeds = kwargs.pop(
            "image_deepstack_visual_embeds", None
        )
        video_deepstack_visual_embeds = kwargs.pop(
            "video_deepstack_visual_embeds", None
        )
        visual_pos_masks = kwargs.pop("visual_pos_masks", None)

        if inputs_embeds is not None:
            inputs_embeds = inputs_embeds.to(self._device)

        # Track whether we manually merged embeddings
        manual_merge_done = False

        if inputs_embeds is None and (
            image_embeds_t is not None
            or video_embeds_t is not None
            or audio_embeds_t is not None
        ):
            inputs_embeds = self.thinker.get_input_embeddings()(
                input_ids.to(self._device)
            )

        if inputs_embeds is not None and (
            image_embeds_t is not None
            or video_embeds_t is not None
            or audio_embeds_t is not None
        ):
            image_embeds_t = (
                image_embeds_t.to(self._device) if image_embeds_t is not None else None
            )
            video_embeds_t = (
                video_embeds_t.to(self._device) if video_embeds_t is not None else None
            )
            audio_embeds_t = (
                audio_embeds_t.to(self._device) if audio_embeds_t is not None else None
            )
            inputs_embeds, image_mask, video_mask = self._merge_embeddings(
                input_ids=input_ids.to(self._device),
                inputs_embeds=inputs_embeds,
                image_embeds=image_embeds_t,
                video_embeds=video_embeds_t,
                audio_embeds=audio_embeds_t,
            )
            manual_merge_done = True
        else:
            image_mask = None
            video_mask = None

        if deepstack_visual_embeds is None and (
            image_deepstack_visual_embeds or video_deepstack_visual_embeds
        ):
            if image_deepstack_visual_embeds and video_deepstack_visual_embeds:
                if image_mask is None or video_mask is None:
                    raise ValueError(
                        "Missing visual masks for merged deepstack embeddings."
                    )
                # Follow HF implementation: visual_pos_masks = image_mask | video_mask
                # Both masks have shape (batch_size, seq_len, hidden_size)
                visual_pos_masks_local = image_mask | video_mask

                # Extract boolean masks for visual positions only
                # image_mask[visual_pos_masks_local] gives a 1D boolean array of length visual_pos_masks_local.sum()
                # indicating which positions in the visual set are image positions
                # This matches HF's implementation: image_mask_joint = image_mask[visual_pos_masks]
                image_mask_joint = image_mask[visual_pos_masks_local]
                video_mask_joint = video_mask[visual_pos_masks_local]

                merged: list[torch.Tensor] = []
                for img_embed, vid_embed in zip(
                    image_deepstack_visual_embeds, video_deepstack_visual_embeds
                ):
                    # embed_joint length = visual_pos_masks_local.sum() (all visual positions)
                    num_visual_pos = int(visual_pos_masks_local.sum().item())
                    embed_joint = img_embed.new_zeros(
                        num_visual_pos, img_embed.shape[-1]
                    )
                    # img_embed length should match image_mask_joint.sum() (image positions in visual set)
                    # This should equal all image positions if all images are visual
                    if image_mask_joint.any():
                        num_image_in_visual = int(image_mask_joint.sum().item())
                        if num_image_in_visual == img_embed.shape[0]:
                            # All image positions are in visual set, direct assignment
                            embed_joint[image_mask_joint, :] = img_embed
                        else:
                            raise ValueError(
                                f"Image embed length mismatch: "
                                f"img_embed.shape[0]={img_embed.shape[0]}, "
                                f"image_mask_joint.sum()={num_image_in_visual}"
                            )

                    if video_mask_joint.any():
                        num_video_in_visual = int(video_mask_joint.sum().item())
                        if num_video_in_visual == vid_embed.shape[0]:
                            # All video positions are in visual set, direct assignment
                            embed_joint[video_mask_joint, :] = vid_embed
                        else:
                            raise ValueError(
                                f"Video embed length mismatch: "
                                f"vid_embed.shape[0]={vid_embed.shape[0]}, "
                                f"video_mask_joint.sum()={num_video_in_visual}"
                            )
                    merged.append(embed_joint)
                deepstack_visual_embeds = merged
                if visual_pos_masks is None:
                    visual_pos_masks = visual_pos_masks_local
            elif image_deepstack_visual_embeds:
                deepstack_visual_embeds = image_deepstack_visual_embeds
            elif video_deepstack_visual_embeds:
                deepstack_visual_embeds = video_deepstack_visual_embeds

        # Only pass deepstack_visual_embeds if we haven't already merged embeddings
        # This avoids the HF model trying to merge visual features twice
        if deepstack_visual_embeds is not None and not manual_merge_done:
            kwargs["deepstack_visual_embeds"] = deepstack_visual_embeds
            if visual_pos_masks is not None:
                kwargs["visual_pos_masks"] = visual_pos_masks
            elif image_mask is not None or video_mask is not None:
                if image_mask is None:
                    kwargs["visual_pos_masks"] = video_mask
                elif video_mask is None:
                    kwargs["visual_pos_masks"] = image_mask
                else:
                    kwargs["visual_pos_masks"] = image_mask | video_mask

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
