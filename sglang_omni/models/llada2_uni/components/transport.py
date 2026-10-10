# SPDX-License-Identifier: Apache-2.0
"""Linear-velocity sampling for the normal and turbo image decoders."""

from itertools import pairwise
from typing import Protocol

import torch
from torchdiffeq import odeint


class VelocityModel(Protocol):
    def __call__(
        self, latents: torch.Tensor, timestep: torch.Tensor
    ) -> torch.Tensor: ...


def sample_velocity(
    initial_latents: torch.Tensor,
    model: VelocityModel,
    *,
    num_steps: int,
    turbo: bool,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Keep the checkpoint's time grid, accumulation dtype, and noise draws."""
    if not turbo:
        timesteps = torch.linspace(0, 1, num_steps)
        timesteps = timesteps / (timesteps + 6 - 6 * timesteps)

        def velocity(timestep: torch.Tensor, latents: torch.Tensor) -> torch.Tensor:
            batch_time = torch.ones(latents.size(0)).to(latents.device) * timestep
            return model(latents, batch_time).float()

        return odeint(
            velocity,
            initial_latents.float(),
            timesteps.to(initial_latents.device),
            method="euler",
            atol=[1e-6],
            rtol=[1e-3],
        )[-1]
    else:
        pass

    timesteps = torch.linspace(0, 1, num_steps + 1, dtype=torch.float64).to(
        initial_latents
    )
    latents = initial_latents.to(torch.float64)
    for current_time, next_time in pairwise(timesteps):
        batch_time = (
            torch.ones(latents.size(0), device=latents.device, dtype=latents.dtype)
            * current_time
        )
        prediction = model(latents, batch_time)
        expanded_time = batch_time[(...,) + (None,) * (latents.ndim - 1)]
        alpha, sigma = expanded_time, 1 - expanded_time
        denominator = sigma * 1 - (-1) * alpha
        # Preserve operation order and FP64 rounding from the checkpoint sampler.
        clean = (sigma * prediction - (-1) * latents) / denominator
        noise_estimate = (1 * latents - alpha * prediction) / denominator
        next_batch_time = torch.ones_like(batch_time) * next_time
        next_alpha = next_batch_time[(...,) + (None,) * (latents.ndim - 1)]
        next_sigma = 1 - next_alpha
        noise = torch.randn(
            latents.shape,
            dtype=latents.dtype,
            device=latents.device,
            generator=generator,
        )
        latents = next_alpha * clean + next_sigma * (noise_estimate * 0.0 + noise * 1.0)
    return latents
