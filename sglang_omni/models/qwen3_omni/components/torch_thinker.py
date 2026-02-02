# SPDX-License-Identifier: Apache-2.0
"""Torch-native thinker wrapper for Qwen3-Omni."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from sglang_omni.models.qwen3_omni.components.common import concat_features
from sglang_omni.models.qwen3_omni.components.mrope import get_multimodal_rope_index
from sglang_omni.models.qwen3_omni.components.torch_common import (
    load_config_dict,
    strip_audio_config,
)
from sglang_omni.models.qwen3_omni.modeling import (
    ThinkerOutput as TorchThinkerOutput,
    Qwen3OmniThinker as TorchThinker,
    _build_position_ids,
    _maybe_apply_causal_mask,
)
from sglang_omni.models.weight_loader import load_weights_by_prefixes, resolve_dtype


class Qwen3OmniTorchThinker(nn.Module):
    """Torch-native thinker that accepts multimodal embeddings."""

    def __init__(
        self,
        model_path: str,
        *,
        device: str = "cuda",
        dtype: str | torch.dtype | None = None,
    ) -> None:
        super().__init__()
        torch_dtype = resolve_dtype(dtype)
        config = load_config_dict(model_path)
        config = strip_audio_config(config)

        self._device = torch.device(device)
        self.config = config
        self.thinker = TorchThinker(config)

        thinker_cfg = config.get("thinker_config", config)
        self.audio_token_id = thinker_cfg.get("audio_token_id")
        self.image_token_id = thinker_cfg.get("image_token_id")
        self.video_token_id = thinker_cfg.get("video_token_id")
        self.vision_start_token_id = thinker_cfg.get("vision_start_token_id")
        self.audio_start_token_id = thinker_cfg.get("audio_start_token_id")
        self.position_id_per_seconds = thinker_cfg.get("position_id_per_seconds", 1)
        vision_cfg = thinker_cfg.get("vision_config", {})
        self.spatial_merge_size = int(vision_cfg.get("spatial_merge_size", 1))
        self._rope_deltas: torch.Tensor | None = None

        # Load only thinker + lm_head weights (skip talker/code2wav/audio/vision)
        state_dict = load_weights_by_prefixes(
            model_path,
            prefixes="thinker.model.",
            device=str(self._device),
            dtype=torch_dtype,
        )
        lm_head_dict = load_weights_by_prefixes(
            model_path,
            prefixes="thinker.lm_head.",
            device=str(self._device),
            dtype=torch_dtype,
        )
        state_dict.update({f"lm_head.{k}": v for k, v in lm_head_dict.items()})
        self.thinker.load_state_dict(state_dict, strict=True, assign=True)
        if torch_dtype is not None:
            self.thinker = self.thinker.to(dtype=torch_dtype)
        self.thinker = self.thinker.to(self._device)
        self.thinker.eval()

    def get_input_embeddings(self) -> nn.Embedding:
        return self.thinker.get_input_embeddings()

    def _merge_embeddings(
        self,
        *,
        input_ids: torch.Tensor,
        inputs_embeds: torch.Tensor,
        image_embeds: torch.Tensor | None,
        audio_embeds: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        image_mask = None
        if image_embeds is not None and self.image_token_id is not None:
            token_count = int((input_ids == int(self.image_token_id)).sum().item())
            if token_count != int(image_embeds.shape[0]):
                raise ValueError(
                    "Image placeholder count mismatch: "
                    f"tokens={token_count} embeds={image_embeds.shape[0]}"
                )
            image_mask = (input_ids == int(self.image_token_id)).unsqueeze(-1).expand_as(
                inputs_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

        if audio_embeds is not None and self.audio_token_id is not None:
            token_count = int((input_ids == int(self.audio_token_id)).sum().item())
            if token_count != int(audio_embeds.shape[0]):
                raise ValueError(
                    "Audio placeholder count mismatch: "
                    f"tokens={token_count} embeds={audio_embeds.shape[0]}"
                )
            audio_mask = (input_ids == int(self.audio_token_id)).unsqueeze(-1).expand_as(
                inputs_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(audio_mask, audio_embeds)

        return inputs_embeds, image_mask

    def _apply_deepstack(
        self,
        hidden_states: torch.Tensor,
        visual_pos_masks: torch.Tensor | None,
        visual_embeds: torch.Tensor,
    ) -> torch.Tensor:
        if visual_pos_masks is None:
            return hidden_states
        if visual_pos_masks.dim() == 3:
            visual_pos_masks = visual_pos_masks[..., 0]
        visual_pos_masks = visual_pos_masks.to(hidden_states.device)
        visual_embeds = visual_embeds.to(hidden_states.device, hidden_states.dtype)
        local = hidden_states[visual_pos_masks, :].clone() + visual_embeds
        hidden_states[visual_pos_masks, :] = local
        return hidden_states

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        past_key_values: list | None = None,
        inputs_embeds: torch.Tensor | None = None,
        use_cache: bool = False,
        output_hidden_states: bool = False,
        cache_position: torch.Tensor | None = None,
        image_embeds: torch.Tensor | list[torch.Tensor] | None = None,
        audio_embeds: torch.Tensor | list[torch.Tensor] | None = None,
        deepstack_visual_embeds: list[torch.Tensor] | None = None,
        visual_pos_masks: torch.Tensor | None = None,
        image_grid_thw: torch.Tensor | None = None,
        video_grid_thw: torch.Tensor | None = None,
        feature_attention_mask: torch.Tensor | None = None,
        audio_feature_lengths: torch.Tensor | None = None,
        use_audio_in_video: bool = False,
        video_second_per_grid: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> TorchThinkerOutput:
        del kwargs
        if input_ids is None and inputs_embeds is None:
            raise ValueError("input_ids or inputs_embeds must be provided")

        if input_ids is not None and input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        if attention_mask is not None and attention_mask.dim() == 1:
            attention_mask = attention_mask.unsqueeze(0)

        image_embeds_t = concat_features(image_embeds)
        audio_embeds_t = concat_features(audio_embeds)

        if inputs_embeds is None:
            if input_ids is None:
                raise ValueError("input_ids required to build inputs_embeds")
            inputs_embeds = self.thinker.get_input_embeddings()(input_ids.to(self._device))

        inputs_embeds = inputs_embeds.to(self._device)
        if input_ids is not None:
            input_ids = input_ids.to(self._device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self._device)

        if image_embeds_t is not None or audio_embeds_t is not None:
            if input_ids is None:
                raise ValueError("input_ids required for multimodal merge")
            image_embeds_t = (
                image_embeds_t.to(self._device) if image_embeds_t is not None else None
            )
            audio_embeds_t = (
                audio_embeds_t.to(self._device) if audio_embeds_t is not None else None
            )
            inputs_embeds, image_mask = self._merge_embeddings(
                input_ids=input_ids,
                inputs_embeds=inputs_embeds,
                image_embeds=image_embeds_t,
                audio_embeds=audio_embeds_t,
            )
            if visual_pos_masks is None:
                visual_pos_masks = image_mask

        if feature_attention_mask is not None:
            audio_feature_lengths = torch.sum(feature_attention_mask.to(self._device), dim=1)
        if audio_feature_lengths is not None:
            audio_feature_lengths = audio_feature_lengths.to(self._device)
        if image_grid_thw is not None:
            image_grid_thw = image_grid_thw.to(self._device)
        if video_grid_thw is not None:
            video_grid_thw = video_grid_thw.to(self._device)
        if video_second_per_grid is not None:
            video_second_per_grid = video_second_per_grid.to(self._device)

        if attention_mask is not None and position_ids is None:
            if (
                cache_position is None
                or (cache_position is not None and cache_position[0] == 0)
                or self._rope_deltas is None
            ):
                if input_ids is None:
                    raise ValueError("input_ids required for rope index computation")
                delta0 = (1 - attention_mask).sum(dim=-1).unsqueeze(1)
                position_ids, rope_deltas = get_multimodal_rope_index(
                    input_ids=input_ids,
                    image_grid_thw=image_grid_thw,
                    video_grid_thw=video_grid_thw,
                    attention_mask=attention_mask,
                    use_audio_in_video=use_audio_in_video,
                    audio_seqlens=audio_feature_lengths,
                    second_per_grids=video_second_per_grid,
                    spatial_merge_size=self.spatial_merge_size,
                    image_token_id=self.image_token_id,
                    video_token_id=self.video_token_id,
                    audio_token_id=self.audio_token_id,
                    vision_start_token_id=self.vision_start_token_id,
                    audio_start_token_id=self.audio_start_token_id,
                    position_id_per_seconds=self.position_id_per_seconds,
                )
                rope_deltas = rope_deltas - delta0
                self._rope_deltas = rope_deltas
            else:
                if input_ids is None:
                    raise ValueError("input_ids required for cache position handling")
                batch_size, seq_length = input_ids.shape
                delta = cache_position[0] + self._rope_deltas if cache_position is not None else 0
                position_ids = torch.arange(seq_length, device=input_ids.device)
                position_ids = position_ids.view(1, -1).expand(batch_size, -1)
                position_ids = position_ids.add(delta)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

        seq_length = inputs_embeds.shape[1]
        if position_ids is None:
            position_ids = _build_position_ids(seq_length, inputs_embeds.device)
        attention_mask = _maybe_apply_causal_mask(
            attention_mask, seq_length, inputs_embeds.device, allow_none=False
        )
        position_embeddings = self.thinker.rotary_emb(inputs_embeds, position_ids)

        hidden_states = inputs_embeds
        all_hidden_states = () if output_hidden_states else None
        new_past_key_values = [] if use_cache else None

        for layer_idx, layer in enumerate(self.thinker.layers):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            past_kv = past_key_values[layer_idx] if past_key_values is not None else None
            hidden_states, new_kv = layer(
                hidden_states,
                attention_mask=attention_mask,
                position_embeddings=position_embeddings,
                past_key_value=past_kv,
                use_cache=use_cache,
            )
            if use_cache:
                new_past_key_values.append(new_kv)

            if deepstack_visual_embeds is not None and layer_idx < len(
                deepstack_visual_embeds
            ):
                hidden_states = self._apply_deepstack(
                    hidden_states,
                    visual_pos_masks,
                    deepstack_visual_embeds[layer_idx],
                )

        hidden_states = self.thinker.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        logits = self.thinker.lm_head(hidden_states)

        return TorchThinkerOutput(
            sequences=logits.argmax(dim=-1),
            hidden_states=all_hidden_states,
            logits=logits,
            past_key_values=new_past_key_values if use_cache else None,
            last_hidden_state=hidden_states,
        )
