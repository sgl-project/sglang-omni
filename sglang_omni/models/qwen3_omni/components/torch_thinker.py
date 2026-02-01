# SPDX-License-Identifier: Apache-2.0
"""Torch-native thinker wrapper for Qwen3-Omni."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

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
from sglang_omni.models.weight_loader import resolve_dtype
from sglang_omni.models.weight_utils import weights_iterator


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


def _get_feat_extract_output_lengths(input_lengths: torch.Tensor) -> torch.Tensor:
    input_lengths_leave = input_lengths % 100
    feat_lengths = (input_lengths_leave - 1) // 2 + 1
    output_lengths = ((feat_lengths - 1) // 2 + 1 - 1) // 2 + 1 + (input_lengths // 100) * 13
    return output_lengths


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
        if torch_dtype is not None:
            self.thinker = self.thinker.to(dtype=torch_dtype)
        self.thinker = self.thinker.to(self._device)
        self.thinker.eval()

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

        # Load weights from checkpoint
        self.thinker.load_weights(weights_iterator(model_path))

    def get_input_embeddings(self) -> nn.Embedding:
        return self.thinker.get_input_embeddings()

    def _get_llm_pos_ids_for_vision(
        self,
        start_idx: int,
        vision_idx: int,
        spatial_merge_size: int,
        t_index: torch.Tensor,
        grid_hs: torch.Tensor,
        grid_ws: torch.Tensor,
    ) -> torch.Tensor:
        llm_grid_h = grid_hs[vision_idx] // spatial_merge_size
        llm_grid_w = grid_ws[vision_idx] // spatial_merge_size
        device = grid_hs.device
        h_index = (
            torch.arange(llm_grid_h, device=device)
            .view(1, -1, 1)
            .expand(len(t_index), -1, llm_grid_w)
            .flatten()
            .float()
        )
        w_index = (
            torch.arange(llm_grid_w, device=device)
            .view(1, 1, -1)
            .expand(len(t_index), llm_grid_h, -1)
            .flatten()
            .float()
        )
        t_index = t_index.to(device=device, dtype=torch.float)
        t_index = t_index.view(-1, 1).expand(-1, llm_grid_h * llm_grid_w).flatten().float()
        llm_pos_ids = torch.stack([t_index, h_index, w_index])
        return llm_pos_ids + start_idx

    def _get_rope_index(
        self,
        *,
        input_ids: torch.Tensor,
        image_grid_thw: torch.Tensor | None,
        video_grid_thw: torch.Tensor | None,
        attention_mask: torch.Tensor,
        use_audio_in_video: bool,
        audio_seqlens: torch.Tensor | None,
        second_per_grids: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        spatial_merge_size = self.spatial_merge_size
        image_token_id = self.image_token_id
        video_token_id = self.video_token_id
        audio_token_id = self.audio_token_id
        vision_start_token_id = self.vision_start_token_id
        audio_start_token_id = self.audio_start_token_id
        position_id_per_seconds = self.position_id_per_seconds

        if attention_mask is not None:
            attention_mask = attention_mask == 1

        position_ids = torch.zeros(
            3,
            input_ids.shape[0],
            input_ids.shape[1],
            dtype=torch.float,
            device=input_ids.device,
        )

        mrope_position_deltas: list[torch.Tensor] = []
        image_idx = 0
        video_idx = 0
        audio_idx = 0

        total_input_ids = input_ids
        for batch_idx, batch_ids in enumerate(total_input_ids):
            if attention_mask is not None:
                batch_ids = batch_ids[attention_mask[batch_idx]]

            vision_start_indices = (
                torch.argwhere(batch_ids == int(vision_start_token_id)).squeeze(1)
                if vision_start_token_id is not None
                else torch.tensor([], device=batch_ids.device, dtype=torch.long)
            )
            vision_tokens = batch_ids[vision_start_indices + 1] if vision_start_indices.numel() else batch_ids[:0]
            audio_nums = (
                torch.sum(batch_ids == int(audio_start_token_id))
                if audio_start_token_id is not None
                else torch.tensor(0, device=batch_ids.device)
            )
            image_nums = (
                (vision_tokens == int(image_token_id)).sum()
                if image_token_id is not None
                else torch.tensor(0, device=batch_ids.device)
            )
            if use_audio_in_video:
                video_nums = (
                    (vision_tokens == int(audio_start_token_id)).sum()
                    if audio_start_token_id is not None
                    else torch.tensor(0, device=batch_ids.device)
                )
            else:
                video_nums = (
                    (vision_tokens == int(video_token_id)).sum()
                    if video_token_id is not None
                    else torch.tensor(0, device=batch_ids.device)
                )

            input_tokens = batch_ids.tolist()
            llm_pos_ids_list: list[torch.Tensor] = []
            st = 0
            remain_images = int(image_nums.item())
            remain_videos = int(video_nums.item())
            remain_audios = int(audio_nums.item())
            multimodal_nums = (
                remain_images + remain_audios
                if use_audio_in_video
                else remain_images + remain_videos + remain_audios
            )

            for _ in range(multimodal_nums):
                st_idx = llm_pos_ids_list[-1].max() + 1 if llm_pos_ids_list else 0

                if (
                    (image_token_id in input_tokens or video_token_id in input_tokens)
                    and (remain_videos > 0 or remain_images > 0)
                    and vision_start_token_id is not None
                ):
                    ed_vision_start = input_tokens.index(int(vision_start_token_id), st)
                else:
                    ed_vision_start = len(input_tokens) + 1

                if audio_token_id in input_tokens and remain_audios > 0 and audio_start_token_id is not None:
                    ed_audio_start = input_tokens.index(int(audio_start_token_id), st)
                else:
                    ed_audio_start = len(input_tokens) + 1

                min_ed = min(ed_vision_start, ed_audio_start)
                text_len = min_ed - st
                if text_len:
                    llm_pos_ids_list.append(
                        torch.arange(text_len, device=input_ids.device).view(1, -1).expand(3, -1)
                        + st_idx
                    )
                    st_idx += text_len

                if min_ed == ed_vision_start and ed_vision_start + 1 == ed_audio_start:
                    bos_len, eos_len = 2, 2
                else:
                    bos_len, eos_len = 1, 1

                llm_pos_ids_list.append(
                    torch.arange(bos_len, device=input_ids.device).view(1, -1).expand(3, -1) + st_idx
                )
                st_idx += bos_len

                # Audio only
                if min_ed == ed_audio_start:
                    if audio_seqlens is None:
                        raise ValueError("audio_feature_lengths is required for audio rope index")
                    audio_len = _get_feat_extract_output_lengths(audio_seqlens[audio_idx])
                    llm_pos_ids = (
                        torch.arange(audio_len, device=input_ids.device).view(1, -1).expand(3, -1) + st_idx
                    )
                    llm_pos_ids_list.append(llm_pos_ids)

                    st += int(text_len + bos_len + audio_len + eos_len)
                    audio_idx += 1
                    remain_audios -= 1

                # Image only
                elif min_ed == ed_vision_start and image_token_id is not None and batch_ids[ed_vision_start + 1] == int(
                    image_token_id
                ):
                    if image_grid_thw is None:
                        raise ValueError("image_grid_thw is required for image rope index")
                    grid_t = image_grid_thw[image_idx][0]
                    grid_hs = image_grid_thw[:, 1]
                    grid_ws = image_grid_thw[:, 2]
                    t_index = (
                        torch.arange(grid_t, device=input_ids.device).float()
                        * float(position_id_per_seconds)
                    )
                    llm_pos_ids = self._get_llm_pos_ids_for_vision(
                        st_idx, image_idx, spatial_merge_size, t_index, grid_hs, grid_ws
                    )
                    image_len = image_grid_thw[image_idx].prod() // (spatial_merge_size**2)
                    llm_pos_ids_list.append(llm_pos_ids)

                    st += int(text_len + bos_len + image_len + eos_len)
                    image_idx += 1
                    remain_images -= 1

                # Video only
                elif min_ed == ed_vision_start and video_token_id is not None and batch_ids[ed_vision_start + 1] == int(
                    video_token_id
                ):
                    if video_grid_thw is None:
                        raise ValueError("video_grid_thw is required for video rope index")
                    grid_t = video_grid_thw[video_idx][0]
                    grid_hs = video_grid_thw[:, 1]
                    grid_ws = video_grid_thw[:, 2]
                    if second_per_grids is None:
                        raise ValueError("video_second_per_grid is required for video rope index")
                    t_index = (
                        torch.arange(grid_t, device=input_ids.device).float()
                        * float(second_per_grids[video_idx].cpu().float())
                        * float(position_id_per_seconds)
                    )
                    llm_pos_ids = self._get_llm_pos_ids_for_vision(
                        st_idx, video_idx, spatial_merge_size, t_index, grid_hs, grid_ws
                    )
                    video_len = video_grid_thw[video_idx].prod() // (spatial_merge_size**2)
                    llm_pos_ids_list.append(llm_pos_ids)

                    st += int(text_len + bos_len + video_len + eos_len)
                    video_idx += 1
                    remain_videos -= 1

                # Audio in video
                elif min_ed == ed_vision_start and ed_vision_start + 1 == ed_audio_start:
                    if audio_seqlens is None or video_grid_thw is None or second_per_grids is None:
                        raise ValueError("audio + video inputs required for audio-in-video rope index")
                    audio_len = _get_feat_extract_output_lengths(audio_seqlens[audio_idx])
                    audio_llm_pos_ids = (
                        torch.arange(audio_len, device=input_ids.device).view(1, -1).expand(3, -1) + st_idx
                    )
                    grid_t = video_grid_thw[video_idx][0]
                    grid_hs = video_grid_thw[:, 1]
                    grid_ws = video_grid_thw[:, 2]
                    t_index = (
                        torch.arange(grid_t, device=input_ids.device).float()
                        * float(second_per_grids[video_idx].cpu().float())
                        * float(position_id_per_seconds)
                    )
                    video_llm_pos_ids = self._get_llm_pos_ids_for_vision(
                        st_idx, video_idx, spatial_merge_size, t_index, grid_hs, grid_ws
                    )
                    video_data_index = 0
                    audio_data_index = 0
                    while (
                        video_data_index < video_llm_pos_ids.shape[-1]
                        and audio_data_index < audio_llm_pos_ids.shape[-1]
                    ):
                        if video_llm_pos_ids[0][video_data_index] <= audio_llm_pos_ids[0][audio_data_index]:
                            llm_pos_ids_list.append(
                                video_llm_pos_ids[:, video_data_index : video_data_index + 1]
                            )
                            video_data_index += 1
                        else:
                            llm_pos_ids_list.append(
                                audio_llm_pos_ids[:, audio_data_index : audio_data_index + 1]
                            )
                            audio_data_index += 1
                    if video_data_index < video_llm_pos_ids.shape[-1]:
                        llm_pos_ids_list.append(
                            video_llm_pos_ids[:, video_data_index : video_llm_pos_ids.shape[-1]]
                        )
                    if audio_data_index < audio_llm_pos_ids.shape[-1]:
                        llm_pos_ids_list.append(
                            audio_llm_pos_ids[:, audio_data_index : audio_llm_pos_ids.shape[-1]]
                        )

                    video_len = video_grid_thw[video_idx].prod() // (spatial_merge_size**2)
                    st += int(text_len + bos_len + audio_len + video_len + eos_len)
                    audio_idx += 1
                    video_idx += 1
                    remain_videos -= 1
                    remain_audios -= 1

                st_idx = llm_pos_ids_list[-1].max() + 1 if llm_pos_ids_list else 0
                llm_pos_ids_list.append(
                    torch.arange(eos_len, device=input_ids.device).view(1, -1).expand(3, -1) + st_idx
                )

            if st < len(input_tokens):
                st_idx = llm_pos_ids_list[-1].max() + 1 if llm_pos_ids_list else 0
                text_len = len(input_tokens) - st
                llm_pos_ids_list.append(
                    torch.arange(text_len, device=input_ids.device).view(1, -1).expand(3, -1) + st_idx
                )

            llm_positions = torch.cat([item.float() for item in llm_pos_ids_list], dim=1).reshape(3, -1)
            if attention_mask is not None:
                position_ids[..., batch_idx, attention_mask[batch_idx] == 1] = llm_positions.to(
                    position_ids.device
                )
            else:
                position_ids[..., batch_idx, :] = llm_positions.to(position_ids.device)
            mrope_position_deltas.append(llm_positions.max() + 1 - len(batch_ids))

        mrope_position_deltas_tensor = (
            torch.tensor(mrope_position_deltas, device=input_ids.device).unsqueeze(1)
            if mrope_position_deltas
            else torch.zeros((input_ids.shape[0], 1), device=input_ids.device)
        )
        return position_ids, mrope_position_deltas_tensor

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

        image_embeds_t = _concat_features(image_embeds)
        audio_embeds_t = _concat_features(audio_embeds)

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
                position_ids, rope_deltas = self._get_rope_index(
                    input_ids=input_ids,
                    image_grid_thw=image_grid_thw,
                    video_grid_thw=video_grid_thw,
                    attention_mask=attention_mask,
                    use_audio_in_video=use_audio_in_video,
                    audio_seqlens=audio_feature_lengths,
                    second_per_grids=video_second_per_grid,
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
