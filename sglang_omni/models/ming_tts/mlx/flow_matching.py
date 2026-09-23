# SPDX-License-Identifier: MIT
# Copyright (c) 2025 inclusionAI
# Adapted from the Omni Ming-TTS flow_matching.py inference solver.

from __future__ import annotations

import math

import mlx.core as mx
import mlx.nn as nn

from .acoustic import DiT
from .config import AcousticConfig


def build_cfm_timesteps(steps: int = 10, *, use_epss: bool = True) -> mx.array:
    if steps < 1:
        raise ValueError("CFM requires at least one integration step")
    predefined = {
        5: [0, 2, 4, 8, 16, 32],
        6: [0, 2, 4, 6, 8, 16, 32],
        7: [0, 2, 4, 6, 8, 16, 24, 32],
        10: [0, 2, 4, 6, 8, 12, 16, 20, 24, 28, 32],
        12: [0, 2, 4, 6, 8, 10, 12, 14, 16, 20, 24, 28, 32],
        16: [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 20, 24, 28, 32],
    }
    if use_epss and steps in predefined:
        return mx.array(predefined[steps], dtype=mx.float32) / 32
    return mx.linspace(0, 1, steps + 1)


def _expand_batch_param(value: float | mx.array, *, batch_size: int) -> mx.array:
    tensor = mx.array(value, dtype=mx.float32)
    if tensor.size == 1:
        return mx.broadcast_to(tensor.reshape(1, 1, 1), (batch_size, 1, 1))
    return tensor.reshape(batch_size, 1, 1)


class CFM(nn.Module):
    def __init__(self, model: DiT) -> None:
        super().__init__()
        self.model = model

    def sample(
        self,
        noise: mx.array,
        c: mx.array,
        latent_history: mx.array,
        timesteps: mx.array,
        sde_random: mx.array,
        *,
        cfg_scale: float | mx.array = 1.0,
        sigma: float | mx.array = 0.25,
        temperature: float | mx.array = 1.5,
        sway_sampling_coef: float | None = -1.0,
    ) -> mx.array:
        batch_size = noise.shape[0]
        cfg = _expand_batch_param(cfg_scale, batch_size=batch_size)
        sigma = _expand_batch_param(sigma, batch_size=batch_size)
        temperature = _expand_batch_param(temperature, batch_size=batch_size)
        # Solver state stays FP32 even with BF16 acoustic weights.
        x = noise.transpose(0, 2, 1).astype(mx.float32)
        t = timesteps
        if sway_sampling_coef is not None:
            t = t + sway_sampling_coef * (mx.cos(math.pi / 2 * t) - 1 + t)
        steps = t.shape[0] - 1
        for step in range(steps):
            dt = t[step + 1] - t[step]
            pred, null = mx.split(
                self.model.forward_with_cfg(x, t[step], c, latent_history),
                2, axis=0,
            )
            velocity = pred + (pred - null) * cfg
            x = x + dt * velocity
            if step + 1 < steps:
                shift = sigma * mx.sqrt(temperature) * mx.sqrt(mx.abs(dt))
                x = x + shift * sde_random[step]
        return x


class FlowLoss(nn.Module):
    def __init__(self, config: AcousticConfig, latent_dim: int, llm_dim: int) -> None:
        super().__init__()
        self.cfm = CFM(DiT(config, latent_dim, llm_dim))

    def sample(
        self,
        z: mx.array,
        latent_history: mx.array,
        noise: mx.array,
        timesteps: mx.array,
        sde_random: mx.array,
        *,
        cfg: float | mx.array = 1.0,
        sigma: float | mx.array = 0.25,
        temperature: float | mx.array = 0.0,
    ) -> mx.array:
        return self.cfm.sample(
            noise, z, latent_history, timesteps, sde_random,
            cfg_scale=cfg, sigma=sigma, temperature=temperature,
        )
