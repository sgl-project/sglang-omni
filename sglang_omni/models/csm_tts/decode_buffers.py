# SPDX-License-Identifier: Apache-2.0
"""CUDA-graph decode buffers for CSM: fill before the step, pack after it."""

from __future__ import annotations

import torch
from sglang.srt.model_executor.forward_batch_info import ForwardBatch

from sglang_omni.models.csm_tts.generation_params import (
    decode_sampling_params,
    depth_sampling_params,
)
from sglang_omni.models.csm_tts.model import CsmTTSModel
from sglang_omni.models.csm_tts.sampler import K_MAX
from sglang_omni.scheduling.types import SchedulerRequest


def populate_decode_buffers(
    model: CsmTTSModel,
    forward_batch: ForwardBatch,
    requests: list[SchedulerRequest],
    *,
    is_lookahead: bool,
    async_enabled: bool,
) -> None:
    """Fill decode buffers. Extra slots reuse the reset padding slot."""
    batch_size = int(forward_batch.batch_size)
    num_requests = len(requests)
    if batch_size < num_requests:
        raise ValueError(
            f"forward_batch.batch_size ({batch_size}) < len(requests) "
            f"({num_requests})"
        )
    else:
        pass

    model.sampler_pool.reset_sampler_slot(model.padding_sampler_slot)

    sampler_slots: list[int] = [
        model.acquire_sampler_slot(request.request_id) for request in requests
    ]
    sampler_slots.extend([model.padding_sampler_slot] * (batch_size - num_requests))
    model.cg_row_indices[:batch_size] = torch.tensor(
        sampler_slots,
        dtype=torch.long,
        device=model.cg_row_indices.device,
    )

    if async_enabled and is_lookahead and num_requests > 0:
        # async lookahead can still launch a finished slot; park on padding.
        active_sampler_slots = model.cg_row_indices[:num_requests]
        generation_done = model.sampler_pool.generation_done[active_sampler_slots]
        model.cg_row_indices[:num_requests] = torch.where(
            generation_done,
            torch.full_like(active_sampler_slots, model.padding_sampler_slot),
            active_sampler_slots,
        )
    else:
        pass

    temperatures, top_ps, top_ks = decode_sampling_params(
        forward_batch.sampling_info, num_requests
    )
    temperatures.extend([1.0] * (batch_size - num_requests))
    top_ps.extend([1.0] * (batch_size - num_requests))
    model.cg_temperature[:batch_size] = torch.tensor(
        temperatures,
        dtype=torch.float32,
        device=model.cg_temperature.device,
    )
    model.cg_top_p[:batch_size] = torch.tensor(
        top_ps,
        dtype=torch.float32,
        device=model.cg_top_p.device,
    )
    top_k_values = [K_MAX if top_k is None else top_k for top_k in top_ks]
    top_k_values.extend([K_MAX] * (batch_size - num_requests))
    model.cg_top_k_buf[:batch_size] = torch.tensor(
        top_k_values,
        dtype=torch.long,
        device=model.cg_top_k_buf.device,
    )

    depth_temperatures, depth_top_ks = depth_sampling_params(requests)
    depth_temperatures.extend([1.0] * (batch_size - num_requests))
    depth_top_ks.extend([K_MAX] * (batch_size - num_requests))
    model.cg_depth_temperature[:batch_size] = torch.tensor(
        depth_temperatures,
        dtype=torch.float32,
        device=model.cg_depth_temperature.device,
    )
    model.cg_depth_top_k_buf[:batch_size] = torch.tensor(
        depth_top_ks,
        dtype=torch.long,
        device=model.cg_depth_top_k_buf.device,
    )

    batch_sampler_slots = model.cg_row_indices[:batch_size]
    pool = model.sampler_pool
    model.cg_active_generation_done[:batch_size] = pool.generation_done[
        batch_sampler_slots
    ]
    model.cg_active_last_codes[:batch_size] = pool.last_codes[batch_sampler_slots]


def pack_decode_staging(model: CsmTTSModel, num_requests: int) -> torch.Tensor:
    """Scatter active buffers into the pool and pack the D2H staging rows."""
    active_sampler_slots = model.cg_row_indices[:num_requests]
    pool = model.sampler_pool
    pool.generation_done[active_sampler_slots] = model.cg_active_generation_done[
        :num_requests
    ]
    pool.last_codes[active_sampler_slots] = model.cg_active_last_codes[:num_requests]
    pool.frames_emitted[active_sampler_slots] += (~model.cg_was_done[:num_requests]).to(
        torch.int32
    )

    num_codebooks = model.cg_codes_BN.shape[1]
    staging = model.cg_collect_staging
    staging[:num_requests, :num_codebooks] = model.cg_codes_BN[:num_requests]
    staging[:num_requests, num_codebooks] = model.cg_was_done[:num_requests]
    staging[:num_requests, num_codebooks + 1] = model.cg_active_generation_done[
        :num_requests
    ]
    return staging


__all__ = ["pack_decode_staging", "populate_decode_buffers"]
