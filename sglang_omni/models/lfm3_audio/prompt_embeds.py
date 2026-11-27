# SPDX-License-Identifier: Apache-2.0
"""Prefill input embeddings of an LFM3 request: the prompt with its media embeddings, and on a retraction's replay the inputs of the tokens generated before it."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from sglang_omni.models.lfm3_audio.request_builders import Lfm3AudioRequestData
from sglang_omni.models.lfm3_audio.sglang_model import Lfm3AudioSGLangModel
from sglang_omni.scheduling.types import SchedulerRequest


def prefill_input_embeds(
    model: Lfm3AudioSGLangModel, request: SchedulerRequest
) -> torch.Tensor:
    """The prompt's embeddings (encode_prompts), followed on a replay by the inputs of the tokens generated before the retraction."""
    request_data = request.data
    sglang_request = request_data.req
    if sglang_request.prefix_len != 0:
        raise RuntimeError(
            "LFM3 prefills a request whole; serve it with the radix cache off"
        )
    else:
        pass
    prompt_embeds = request_data.prefill_input_embeds
    output_ids = sglang_request.output_ids
    if sglang_request.extend_len != prompt_embeds.shape[0] + len(output_ids):
        raise RuntimeError(
            f"LFM3 prefill of {request.request_id} covers {sglang_request.extend_len} positions, "
            f"not its {prompt_embeds.shape[0]} prompt and {len(output_ids)} output tokens; "
            "serve it with chunked prefill off"
        )
    elif not output_ids:
        return prompt_embeds
    else:
        pass
    output_id_tensor = torch.tensor(output_ids, device=prompt_embeds.device)
    output_embeds = model.get_input_embeddings()(output_id_tensor)
    is_frame = output_id_tensor == model.tokens.audio_out_token_id
    if int(is_frame.sum()) != len(request_data.feedback_frame_codes):
        raise RuntimeError(
            f"LFM3 replay of {request.request_id} has {int(is_frame.sum())} frame positions "
            f"for {len(request_data.feedback_frame_codes)} fed frames"
        )
    elif request_data.feedback_frame_codes:
        output_embeds[is_frame] = model.frame_embeddings(
            torch.stack(request_data.feedback_frame_codes).to(output_embeds.device)
        ).to(output_embeds.dtype)
    else:
        pass
    # note (ruilin): the queued embedding of the last frame is in output_embeds now.
    request_data.pending_feedback_queue.clear()
    return torch.cat([prompt_embeds, output_embeds], dim=0)


def encode_prompts(
    model: Lfm3AudioSGLangModel, batch: list[Lfm3AudioRequestData]
) -> None:
    """Set each request's prompt embeddings: one encoder call per media kind for the whole batch, then each request's share at its placeholders."""
    if not batch:
        return
    else:
        pass
    embed_tokens = model.get_input_embeddings()
    device = embed_tokens.weight.device
    media_by_request = encode_batch_media(model, batch, device)
    for request_data, media in zip(batch, media_by_request, strict=True):
        token_ids = request_data.input_ids.to(device)
        prompt_embeds = embed_tokens(token_ids)
        for placeholder_token_id, media_embeds in media:
            positions = (token_ids == placeholder_token_id).nonzero(as_tuple=True)[0]
            prompt_embeds[positions] = media_embeds.to(prompt_embeds.dtype)
        request_data.prefill_input_embeds = prompt_embeds.detach()


def encode_batch_media(
    model: Lfm3AudioSGLangModel,
    batch: list[Lfm3AudioRequestData],
    device: torch.device,
) -> list[list[tuple[int, torch.Tensor]]]:
    """(placeholder token id, embeddings) of each request's audio clips, images and videos; consumes the inputs."""
    tokens = model.tokens
    media: list[list[tuple[int, torch.Tensor]]] = [[] for _ in batch]
    audio_rows = [
        row
        for row, request_data in enumerate(batch)
        if request_data.input_features is not None
    ]
    if audio_rows:
        frames = max(batch[row].input_features.shape[-1] for row in audio_rows)
        features = torch.cat(
            [
                F.pad(
                    batch[row].input_features,
                    (0, frames - batch[row].input_features.shape[-1]),
                )
                for row in audio_rows
            ]
        )
        feature_mask = torch.cat(
            [
                F.pad(
                    batch[row].feature_attention_mask,
                    (0, frames - batch[row].feature_attention_mask.shape[-1]),
                )
                for row in audio_rows
            ]
        )
        share_out(
            media,
            audio_rows,
            batch,
            tokens.audio_token_id,
            model.encode_audio(features.to(device), feature_mask.to(device)),
        )
    else:
        pass
    vision_inputs = (
        (
            tokens.image_token_id,
            [(data.pixel_values, data.image_grid_thw) for data in batch],
        ),
        (
            tokens.video_token_id,
            [(data.pixel_values_videos, data.video_grid_thw) for data in batch],
        ),
    )
    for placeholder_token_id, inputs in vision_inputs:
        rows = [row for row, (patches, _) in enumerate(inputs) if patches is not None]
        if rows:
            pixel_values = torch.cat([inputs[row][0] for row in rows])
            grid_thw = torch.cat([inputs[row][1] for row in rows])
            share_out(
                media,
                rows,
                batch,
                placeholder_token_id,
                model.encode_vision(pixel_values.to(device), grid_thw.to(device)),
            )
        else:
            pass
    for request_data in batch:
        request_data.input_features = None
        request_data.feature_attention_mask = None
        request_data.pixel_values = None
        request_data.image_grid_thw = None
        request_data.pixel_values_videos = None
        request_data.video_grid_thw = None
    return media


def share_out(
    media: list[list[tuple[int, torch.Tensor]]],
    rows: list[int],
    batch: list[Lfm3AudioRequestData],
    placeholder_token_id: int,
    embeds: torch.Tensor,
) -> None:
    """Split one encoder call's rows among the requests in order, by each prompt's placeholder count."""
    counts = [int((batch[row].input_ids == placeholder_token_id).sum()) for row in rows]
    if sum(counts) != embeds.shape[0]:
        raise ValueError(
            f"{sum(counts)} placeholders of token {placeholder_token_id} for "
            f"{embeds.shape[0]} media embeddings"
        )
    else:
        pass
    for row, share in zip(rows, embeds.split(counts), strict=True):
        media[row].append((placeholder_token_id, share))
