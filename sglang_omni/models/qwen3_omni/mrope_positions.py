# SPDX-License-Identifier: Apache-2.0
"""Vectorized Qwen3-Omni M-RoPE position index (bit-identical to sglang HF port)."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch


def linear_mrope_positions(
    seq_len: int,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.long,
) -> tuple[torch.Tensor, torch.Tensor]:
    """arange broadcast to [3, seq] with delta 0."""
    # Note (guozhihao): clone so MultimodalInputs owns a contiguous buffer
    # (expand returns a view).
    positions = (
        torch.arange(seq_len, device=device, dtype=dtype)
        .unsqueeze(0)
        .expand(3, -1)
        .clone()
    )
    delta = torch.zeros((1, 1), device=device, dtype=dtype)
    return positions, delta


def talker_can_use_linear_mrope(
    input_ids: torch.Tensor,
    model_inputs: dict[str, Any],
    thinker_config: Any,
) -> bool:
    """True when linear arange+delta0 matches full mm MRoPE."""
    # Note (guozhihao): talker uses MRotaryEmbedding; decode is
    # seq_len + delta - 1. Mm placeholders + grids make positions/delta
    # diverge from arange+0 (#1149 Part B), so only short-circuit when no
    # multimodal segment would be emitted (no grids, or grids but no
    # vision_start/audio_start in input_ids).
    has_image = model_inputs.get("image_grid_thw") is not None
    has_video = model_inputs.get("video_grid_thw") is not None
    if not has_image and not has_video:
        return True

    ids = input_ids.view(-1)
    vision_start = int(thinker_config.vision_start_token_id)
    audio_start = int(thinker_config.audio_start_token_id)
    if not (ids == vision_start).any() and not (ids == audio_start).any():
        return True
    return False


def _feat_extract_output_lengths(input_lengths: int) -> int:
    """Audio encoder output length (matches HF / sglang port)."""
    input_lengths_leave = input_lengths % 100
    feat_lengths = (input_lengths_leave - 1) // 2 + 1
    return int(((feat_lengths - 1) // 2 + 1 - 1) // 2 + 1 + (input_lengths // 100) * 13)


def _linear_pos_ids(length: int, st_idx: float) -> np.ndarray:
    """[3, length] identical arange on each M-RoPE axis."""
    if length <= 0:
        return np.zeros((3, 0), dtype=np.float32)
    row = np.arange(length, dtype=np.float32) + np.float32(st_idx)
    return np.stack([row, row, row], axis=0)


def _vision_t_index(
    grid_t: int,
    second_per_grid: float,
    position_id_per_seconds: float,
) -> np.ndarray:
    """Temporal ids matching sglang (arange(t) * sec) * pps float32 order."""
    # Note (guozhihao): left-assoc; precomputing sec * pps changes FP32 rounding.
    t = np.arange(grid_t, dtype=np.float32)
    return (t * np.float32(second_per_grid)) * np.float32(position_id_per_seconds)


def _vision_pos_ids(
    st_idx: float,
    grid_h: int,
    grid_w: int,
    t_index: np.ndarray,
) -> np.ndarray:
    """Vision [t, h, w] block via meshgrid."""
    tt, hh, ww = np.meshgrid(
        t_index.astype(np.float32, copy=False),
        np.arange(grid_h, dtype=np.float32),
        np.arange(grid_w, dtype=np.float32),
        indexing="ij",
    )
    return np.stack(
        [tt.ravel(), hh.ravel(), ww.ravel()],
        axis=0,
    ) + np.float32(st_idx)


def _merge_audio_in_video(video_pos: np.ndarray, audio_pos: np.ndarray) -> np.ndarray:
    """Merge video/audio columns by temporal id; ties prefer video."""
    if video_pos.shape[1] == 0:
        return audio_pos
    if audio_pos.shape[1] == 0:
        return video_pos
    primary = np.concatenate([video_pos[0], audio_pos[0]])
    secondary = np.concatenate(
        [
            np.zeros(video_pos.shape[1], dtype=np.float32),
            np.ones(audio_pos.shape[1], dtype=np.float32),
        ]
    )
    order = np.lexsort((secondary, primary))
    return np.concatenate([video_pos, audio_pos], axis=1)[:, order]


def get_rope_index_qwen3_omni_vectorized(
    spatial_merge_size: int,
    image_token_id: int,
    video_token_id: int,
    vision_start_token_id: int,
    tokens_per_second: int | None = None,
    input_ids: torch.LongTensor | None = None,
    image_grid_thw: torch.LongTensor | None = None,
    video_grid_thw: torch.LongTensor | None = None,
    second_per_grid_ts: torch.Tensor | None = None,
    **kwargs: Any,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Drop-in for get_rope_index_qwen3_omni with vectorized blocks."""
    del tokens_per_second

    audio_token_id = kwargs["audio_token_id"]
    audio_start_token_id = kwargs["audio_start_token_id"]
    position_id_per_seconds = kwargs["position_id_per_seconds"]
    use_audio_in_video = bool(kwargs.get("use_audio_in_video", False))
    audio_seqlens = kwargs.get("audio_seqlens", None)
    second_per_grids = second_per_grid_ts

    if input_ids is None or (image_grid_thw is None and video_grid_thw is None):
        assert input_ids is not None
        seq_len = input_ids.shape[1]
        position_ids = (
            torch.arange(seq_len, device=input_ids.device)
            .unsqueeze(0)
            .expand(3, -1, -1)
            .to(input_ids.device)
        )
        max_position_ids = position_ids.max(0, keepdim=False)[0].max(-1, keepdim=True)[
            0
        ]
        mrope_position_deltas = max_position_ids + 1 - seq_len
        return position_ids, mrope_position_deltas

    device = input_ids.device
    batch_size, seq_len = input_ids.shape[0], input_ids.shape[1]
    position_ids = torch.zeros(3, batch_size, seq_len, dtype=torch.float, device=device)

    image_grid_np = (
        image_grid_thw.detach().cpu().numpy()
        if isinstance(image_grid_thw, torch.Tensor)
        else None
    )
    video_grid_np = (
        video_grid_thw.detach().cpu().numpy()
        if isinstance(video_grid_thw, torch.Tensor)
        else None
    )
    second_per_grids_np = None
    if second_per_grids is not None:
        if isinstance(second_per_grids, torch.Tensor):
            second_per_grids_np = second_per_grids.detach().cpu().float().numpy()
        else:
            second_per_grids_np = np.asarray(second_per_grids, dtype=np.float32)

    audio_seqlens_list: list[int] | None = None
    if audio_seqlens is not None:
        if isinstance(audio_seqlens, torch.Tensor):
            audio_seqlens_list = [int(x) for x in audio_seqlens.detach().cpu().tolist()]
        else:
            audio_seqlens_list = [int(x) for x in audio_seqlens]

    merge = int(spatial_merge_size)
    mrope_position_deltas: list[float] = []
    image_idx = video_idx = audio_idx = 0

    for batch_i in range(batch_size):
        current_input_ids = input_ids[batch_i]
        input_tokens = current_input_ids.tolist()
        n_tokens = len(input_tokens)

        vision_start_indices = torch.argwhere(
            current_input_ids == vision_start_token_id
        ).squeeze(1)
        image_nums = video_nums = 0
        if vision_start_indices.numel() > 0:
            vision_tokens = current_input_ids[vision_start_indices + 1]
            image_nums = int((vision_tokens == image_token_id).sum().item())
            if use_audio_in_video:
                video_nums = int((vision_tokens == audio_start_token_id).sum().item())
            else:
                video_nums = int((vision_tokens == video_token_id).sum().item())
        audio_nums = int((current_input_ids == audio_start_token_id).sum().item())

        remain_images, remain_videos, remain_audios = (
            image_nums,
            video_nums,
            audio_nums,
        )
        multimodal_nums = (
            image_nums + audio_nums
            if use_audio_in_video
            else image_nums + video_nums + audio_nums
        )

        blocks: list[np.ndarray] = []
        st = 0
        has_image_or_video_token = (
            image_token_id in input_tokens or video_token_id in input_tokens
        )
        has_audio_token = audio_token_id in input_tokens

        for _ in range(multimodal_nums):
            st_idx = float(blocks[-1].max() + 1) if blocks else 0.0

            if has_image_or_video_token and (remain_videos > 0 or remain_images > 0):
                ed_vision_start = input_tokens.index(vision_start_token_id, st)
            else:
                ed_vision_start = n_tokens + 1

            if has_audio_token and remain_audios > 0:
                ed_audio_start = input_tokens.index(audio_start_token_id, st)
            else:
                ed_audio_start = n_tokens + 1

            min_ed = min(ed_vision_start, ed_audio_start)
            text_len = min_ed - st
            if text_len != 0:
                blocks.append(_linear_pos_ids(text_len, st_idx))
                st_idx += text_len

            if min_ed == ed_vision_start and ed_vision_start + 1 == ed_audio_start:
                bos_len, eos_len = 2, 2
            else:
                bos_len, eos_len = 1, 1
            blocks.append(_linear_pos_ids(bos_len, st_idx))
            st_idx += bos_len

            if min_ed == ed_audio_start:
                assert audio_seqlens_list is not None
                audio_len = _feat_extract_output_lengths(audio_seqlens_list[audio_idx])
                blocks.append(_linear_pos_ids(audio_len, st_idx))
                st += text_len + bos_len + audio_len + eos_len
                audio_idx += 1
                remain_audios -= 1

            elif (
                min_ed == ed_vision_start
                and current_input_ids[ed_vision_start + 1] == image_token_id
            ):
                assert image_grid_np is not None
                grid_t = int(image_grid_np[image_idx, 0])
                grid_h = int(image_grid_np[image_idx, 1]) // merge
                grid_w = int(image_grid_np[image_idx, 2]) // merge
                t_index = _vision_t_index(grid_t, 1.0, float(position_id_per_seconds))
                blocks.append(_vision_pos_ids(st_idx, grid_h, grid_w, t_index))
                image_len = int(image_grid_np[image_idx].prod()) // (merge * merge)
                st += text_len + bos_len + image_len + eos_len
                image_idx += 1
                remain_images -= 1

            elif (
                min_ed == ed_vision_start
                and current_input_ids[ed_vision_start + 1] == video_token_id
            ):
                assert video_grid_np is not None and second_per_grids_np is not None
                grid_t = int(video_grid_np[video_idx, 0])
                grid_h = int(video_grid_np[video_idx, 1]) // merge
                grid_w = int(video_grid_np[video_idx, 2]) // merge
                t_index = _vision_t_index(
                    grid_t,
                    float(second_per_grids_np[video_idx]),
                    float(position_id_per_seconds),
                )
                blocks.append(_vision_pos_ids(st_idx, grid_h, grid_w, t_index))
                video_len = int(video_grid_np[video_idx].prod()) // (merge * merge)
                st += text_len + bos_len + video_len + eos_len
                video_idx += 1
                remain_videos -= 1

            elif min_ed == ed_vision_start and ed_vision_start + 1 == ed_audio_start:
                assert audio_seqlens_list is not None
                assert video_grid_np is not None and second_per_grids_np is not None
                audio_len = _feat_extract_output_lengths(audio_seqlens_list[audio_idx])
                audio_pos = _linear_pos_ids(audio_len, st_idx)
                grid_t = int(video_grid_np[video_idx, 0])
                grid_h = int(video_grid_np[video_idx, 1]) // merge
                grid_w = int(video_grid_np[video_idx, 2]) // merge
                t_index = _vision_t_index(
                    grid_t,
                    float(second_per_grids_np[video_idx]),
                    float(position_id_per_seconds),
                )
                video_pos = _vision_pos_ids(st_idx, grid_h, grid_w, t_index)
                merged = _merge_audio_in_video(video_pos, audio_pos)
                blocks.append(merged)
                video_len = int(video_grid_np[video_idx].prod()) // (merge * merge)
                st += text_len + bos_len + audio_len + video_len + eos_len
                audio_idx += 1
                video_idx += 1
                remain_videos -= 1
                remain_audios -= 1

            if min_ed == ed_vision_start and ed_vision_start + 1 == ed_audio_start:
                # Note (guozhihao): AIV uses last-column max; oracle appends
                # per column.
                next_st = float(blocks[-1][:, -1].max() + 1)
            else:
                next_st = float(blocks[-1].max() + 1) if blocks else 0.0
            blocks.append(_linear_pos_ids(eos_len, next_st))
        if st < n_tokens:
            st_idx = float(blocks[-1].max() + 1) if blocks else 0.0
            blocks.append(_linear_pos_ids(n_tokens - st, st_idx))

        llm_positions = np.concatenate(blocks, axis=1).astype(np.float32, copy=False)
        position_ids[:, batch_i, :] = torch.from_numpy(llm_positions)
        mrope_position_deltas.append(float(llm_positions.max() + 1 - n_tokens))

    deltas = torch.tensor(
        mrope_position_deltas, device=device, dtype=torch.float
    ).unsqueeze(1)
    return position_ids, deltas
