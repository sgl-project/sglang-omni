# SPDX-License-Identifier: Apache-2.0
"""Per-request generation parameters for CSM codebook-0 and the depth decoder."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo

from sglang_omni.models.csm_tts.sampler import K_MAX
from sglang_omni.scheduling.types import SchedulerRequest


@dataclass
class CsmGenerationParams:
    """Per-request sampling for codebook-0 and the depth decoder."""

    temperature: float = 0.9
    top_k: int = 50
    top_p: float | None = None
    depth_temperature: float = 0.9
    depth_top_k: int = 50
    seed: int | None = None


@dataclass(frozen=True, kw_only=True)
class CsmSamplingTensors:
    """One prefill batch of sampling knobs, one row per request."""

    temperature: torch.Tensor
    top_p: torch.Tensor | None
    top_k: torch.Tensor
    depth_temperature: torch.Tensor
    depth_top_k: torch.Tensor


def flat_sampling_values(
    values: torch.Tensor | None,
) -> list[float] | list[int] | None:
    """One sampling tensor flattened to a host list, or None when unset."""
    if values is None:
        return None
    else:
        return values.detach().cpu().flatten().tolist()


def normalize_top_k(top_k: int | float | str) -> int:
    """Clamp top_k to (0, K_MAX]; anything else is a no-op filter."""
    try:
        clamped = int(top_k)
    except (TypeError, ValueError):
        return K_MAX
    return clamped if 0 < clamped <= K_MAX else K_MAX


def generation_params_for_batch(
    sampling_info: SamplingBatchInfo | None,
    depth_params: list[tuple[float, int]] | None,
    batch_size: int,
) -> list[CsmGenerationParams]:
    """Per-row prefill sampling, defaults where the caller set nothing."""
    temperatures = top_ps = top_ks = None
    if sampling_info is not None:
        temperatures = flat_sampling_values(sampling_info.temperatures)
        top_ps = flat_sampling_values(sampling_info.top_ps)
        top_ks = flat_sampling_values(sampling_info.top_ks)
    else:
        pass

    params: list[CsmGenerationParams] = []
    for batch_index in range(batch_size):
        generation = CsmGenerationParams()
        if temperatures is not None:
            generation.temperature = float(temperatures[batch_index])
        else:
            pass
        if top_ps is not None:
            generation.top_p = float(top_ps[batch_index])
        else:
            pass
        if top_ks is not None:
            generation.top_k = normalize_top_k(top_ks[batch_index])
        else:
            pass
        if depth_params is not None and batch_index < len(depth_params):
            depth_temperature, depth_top_k = depth_params[batch_index]
            generation.depth_temperature = float(depth_temperature)
            generation.depth_top_k = normalize_top_k(depth_top_k)
        else:
            pass
        params.append(generation)
    return params


def sampling_tensors_for_batch(
    generation_params: list[CsmGenerationParams], device: torch.device
) -> CsmSamplingTensors:
    """Move one prefill batch of sampling knobs onto the sampler's device."""
    has_top_p = any(params.top_p is not None for params in generation_params)
    if has_top_p:
        top_p = torch.tensor(
            [
                1.0 if params.top_p is None else params.top_p
                for params in generation_params
            ],
            dtype=torch.float32,
            device=device,
        )
    else:
        top_p = None
    return CsmSamplingTensors(
        temperature=torch.tensor(
            [params.temperature for params in generation_params],
            dtype=torch.float32,
            device=device,
        ),
        top_p=top_p,
        top_k=torch.tensor(
            [normalize_top_k(params.top_k) for params in generation_params],
            dtype=torch.long,
            device=device,
        ),
        depth_temperature=torch.tensor(
            [params.depth_temperature for params in generation_params],
            dtype=torch.float32,
            device=device,
        ),
        depth_top_k=torch.tensor(
            [normalize_top_k(params.depth_top_k) for params in generation_params],
            dtype=torch.long,
            device=device,
        ),
    )


def decode_sampling_params(
    sampling_info: SamplingBatchInfo | None, num_requests: int
) -> tuple[list[float], list[float], list[int | None]]:
    """Codebook-0 temperature / top_p / top_k for one decode step."""
    if sampling_info is None or num_requests == 0:
        return ([1.0] * num_requests, [1.0] * num_requests, [None] * num_requests)
    else:
        pass

    raw_temperatures = flat_sampling_values(sampling_info.temperatures)
    raw_top_ps = flat_sampling_values(sampling_info.top_ps)
    raw_top_ks = flat_sampling_values(sampling_info.top_ks)

    if raw_temperatures is None:
        temperatures = [1.0] * num_requests
    else:
        temperatures = [float(value) for value in raw_temperatures[:num_requests]]
    if raw_top_ps is None:
        top_ps = [1.0] * num_requests
    else:
        top_ps = [float(value) for value in raw_top_ps[:num_requests]]
    if raw_top_ks is None:
        top_ks: list[int | None] = [None] * num_requests
    else:
        top_ks = [
            int(value) if 0 < int(value) <= K_MAX else None
            for value in raw_top_ks[:num_requests]
        ]
    return temperatures, top_ps, top_ks


def depth_sampling_params(
    requests: list[SchedulerRequest],
) -> tuple[list[float], list[int]]:
    """Depth-decoder temperature / top_k per request."""
    depth_temperatures: list[float] = []
    depth_top_ks: list[int] = []
    for scheduler_request in requests:
        request_data = scheduler_request.data
        depth_temperatures.append(float(request_data.depth_temperature))
        depth_top_ks.append(normalize_top_k(request_data.depth_top_k))
    return depth_temperatures, depth_top_ks


__all__ = [
    "CsmGenerationParams",
    "CsmSamplingTensors",
    "decode_sampling_params",
    "depth_sampling_params",
    "flat_sampling_values",
    "generation_params_for_batch",
    "normalize_top_k",
    "sampling_tensors_for_batch",
]
