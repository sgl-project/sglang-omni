# SPDX-License-Identifier: Apache-2.0
"""Multimodal MRoPE position index computation for Qwen3-Omni."""

from __future__ import annotations

import torch

from sglang_omni.models.qwen3_omni.components.torch_common import (
    get_feat_extract_output_lengths,
)


def get_llm_pos_ids_for_vision(
    start_idx: int,
    vision_idx: int,
    spatial_merge_size: int,
    t_index: torch.Tensor,
    grid_hs: torch.Tensor,
    grid_ws: torch.Tensor,
) -> torch.Tensor:
    """Compute 3D (t, h, w) position IDs for a vision segment."""
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


def get_multimodal_rope_index(
    *,
    input_ids: torch.Tensor,
    image_grid_thw: torch.Tensor | None,
    video_grid_thw: torch.Tensor | None,
    attention_mask: torch.Tensor,
    use_audio_in_video: bool,
    audio_seqlens: torch.Tensor | None,
    second_per_grids: torch.Tensor | None,
    spatial_merge_size: int,
    image_token_id: int | None,
    video_token_id: int | None,
    audio_token_id: int | None,
    vision_start_token_id: int | None,
    audio_start_token_id: int | None,
    position_id_per_seconds: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute multimodal MRoPE position indices for mixed text/image/audio/video inputs."""
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
                audio_len = get_feat_extract_output_lengths(audio_seqlens[audio_idx])
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
                llm_pos_ids = get_llm_pos_ids_for_vision(
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
                llm_pos_ids = get_llm_pos_ids_for_vision(
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
                audio_len = get_feat_extract_output_lengths(audio_seqlens[audio_idx])
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
                video_llm_pos_ids = get_llm_pos_ids_for_vision(
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
