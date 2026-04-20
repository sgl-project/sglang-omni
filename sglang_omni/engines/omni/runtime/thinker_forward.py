# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from typing import Any

import torch


def thinker_forward_omni(
    outer_model: Any,
    attn_backend: Any,
    forward_batch: Any,
    input_embeds: torch.Tensor,
    deepstack_visual_embeds: list | None,
    visual_pos_masks: torch.Tensor | None,
) -> Any:
    """Shared thinker forward used by rank 0 and TP followers.

    Both ranks MUST call this even if follower discards the result —
    logits_processor does a vocab-parallel all-gather; skipping desyncs NCCL.
    """
    attn_backend.init_forward_metadata(forward_batch)

    positions = forward_batch.positions
    if forward_batch.mrope_positions is not None:
        positions = forward_batch.mrope_positions

    ds_input = None
    if deepstack_visual_embeds is not None and visual_pos_masks is not None:
        device = input_embeds.device
        dtype = input_embeds.dtype
        layer_tensors = [
            t.to(device=device, dtype=dtype) for t in deepstack_visual_embeds
        ]
        ds_concat = torch.cat(layer_tensors, dim=-1)
        full_ds = torch.zeros(
            input_embeds.shape[0],
            ds_concat.shape[-1],
            device=device,
            dtype=dtype,
        )
        full_ds[visual_pos_masks] = ds_concat
        ds_input = full_ds

    hidden_states = outer_model.model(
        input_ids=None,
        positions=positions,
        forward_batch=forward_batch,
        input_embeds=input_embeds,
        input_deepstack_embeds=ds_input,
    )

    return outer_model.logits_processor(
        forward_batch.input_ids,
        hidden_states,
        outer_model.lm_head,
        forward_batch,
    )
