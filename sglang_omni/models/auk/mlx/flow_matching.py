# SPDX-License-Identifier: MIT
# Copyright (C) 2026 Tencent. All rights reserved.
"""Native MLX AuK and AuK-Flash flow-matching inference."""

from __future__ import annotations

import math
import secrets
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal

import mlx.core as mx
import mlx.nn as nn

from sglang_omni.models.auk.mlx.dit import AuKDit


def request_key(
    seed: int | None, *, stream: Literal["reference", "generation"] = "generation"
) -> mx.array:
    """Independent request-local streams; seeds do not imply cross-backend parity."""
    if seed is None:
        seed = secrets.randbits(64)
    elif not -(2**63) <= seed < 2**64:
        raise ValueError("AuK seed must be between -2**63 and 2**64 - 1")
    else:
        pass
    return mx.random.split(mx.random.key(seed % 2**64), 2)[
        {"reference": 0, "generation": 1}[stream]
    ]


def fuse_hidden_states(
    hidden_states: mx.array, layer_weights: mx.array, layer_scale: mx.array
) -> mx.array:
    normalized = mx.fast.layer_norm(hidden_states[:, 1:], None, None, eps=1e-5)
    weights = mx.softmax(layer_weights, axis=0)
    return mx.sum(normalized * weights[None, :, None, None], axis=1) * layer_scale


def build_time_grid(
    steps: int,
    sway_sampling_coef: float | None = None,
    t_grid: Sequence[float] | None = None,
) -> mx.array:
    if t_grid is not None:
        grid = mx.array(t_grid, dtype=mx.float32)
        if grid.ndim != 1 or grid.size < 2:
            raise ValueError("t_grid must hold at least two time points")
        else:
            return grid
    elif steps < 1:
        raise ValueError("AuK nfe must be positive")
    else:
        grid = mx.linspace(0, 1, steps + 1, dtype=mx.float32)
        if sway_sampling_coef is not None:
            grid = grid + sway_sampling_coef * (mx.cos(math.pi / 2 * grid) - 1 + grid)
        else:
            pass
        return grid


@dataclass(kw_only=True)
class AuKSampleItem:
    conditioning: mx.array
    text_mask: mx.array
    target_frames: int
    ref_latent: mx.array | None = None
    seed: int | None = None
    ref_length: int = 0
    noise: mx.array | None = None


def pack_sequences(tensors: Sequence[mx.array]) -> mx.array:
    max_length = max(tensor.shape[0] for tensor in tensors)
    return mx.stack(
        [
            mx.pad(
                tensor,
                [(0, max_length - tensor.shape[0])] + [(0, 0)] * (tensor.ndim - 1),
            )
            for tensor in tensors
        ]
    )


class AuKFlowMatching(nn.Module):
    def __init__(self, transformer: AuKDit, num_llm_layers: int) -> None:
        super().__init__()
        self.transformer = transformer
        self.layer_weights = mx.zeros((num_llm_layers,))
        self.layer_scale = mx.ones((1,))

    @staticmethod
    def sanitize(weights: dict[str, mx.array]) -> dict[str, mx.array]:
        return {
            name: (
                value.transpose(0, 2, 1)
                if name.endswith(
                    ("conv_pos_embed.conv1d.0.weight", "conv_pos_embed.conv1d.2.weight")
                )
                else value
            )
            for name, value in weights.items()
        }

    def sample(
        self,
        item: AuKSampleItem,
        *,
        steps: int,
        cfg_strength: float,
        sway_sampling_coef: float | None = None,
        t_grid: Sequence[float] | None = None,
    ) -> mx.array:
        return self.sample_batch(
            [item],
            steps=steps,
            cfg_strength=cfg_strength,
            sway_sampling_coef=sway_sampling_coef,
            t_grid=t_grid,
        )[0]

    def sample_batch(
        self,
        items: Sequence[AuKSampleItem],
        *,
        steps: int,
        cfg_strength: float,
        sway_sampling_coef: float | None = None,
        t_grid: Sequence[float] | None = None,
    ) -> list[mx.array]:
        if not items:
            raise ValueError("AuK sampling requires at least one request")
        else:
            pass
        dim, weight_dtype = self.transformer.latent_dim, self.transformer.dtype
        references, noises = [], []
        for item in items:
            if item.target_frames < 1:
                raise ValueError("AuK target_frames must be positive")
            else:
                pass
            reference = (
                mx.zeros((0, dim)) if item.ref_latent is None else item.ref_latent
            )
            if not 0 <= item.ref_length <= reference.shape[0]:
                raise ValueError("AuK ref_length must fit the reference latents")
            else:
                pass
            references.append(reference)
            if item.noise is None:
                noise = mx.random.normal(
                    (item.target_frames, dim),
                    dtype=mx.float32,
                    key=request_key(item.seed),
                )
            elif item.noise.shape != (item.target_frames, dim):
                raise ValueError(
                    "AuK noise shape must match target_frames and latent_dim"
                )
            else:
                noise = item.noise.astype(mx.float32)
            noises.append(noise)
        ref = pack_sequences(references).astype(weight_dtype)
        ref_mask = (
            mx.arange(ref.shape[1])[None, :]
            < mx.array([item.ref_length for item in items])[:, None]
        )
        text = pack_sequences([item.conditioning for item in items]).astype(
            weight_dtype
        )
        text_mask = pack_sequences([item.text_mask for item in items])
        projected_text = self.transformer.project_text(text)
        y0 = pack_sequences(noises)
        mask = audio_positions = joint_positions = None
        if len(items) > 1:
            target_positions = mx.arange(y0.shape[1])[None, :]
            mask = (
                target_positions
                < mx.array([item.target_frames for item in items])[:, None]
            )
            ref_sizes = mx.array([reference.shape[0] for reference in references])[
                :, None
            ]
            text_sizes = mx.array([item.conditioning.shape[0] for item in items])[
                :, None
            ]
            audio_positions = mx.concatenate(
                [
                    mx.broadcast_to(
                        mx.arange(ref.shape[1]), (len(items), ref.shape[1])
                    ),
                    target_positions + ref_sizes,
                ],
                axis=1,
            )
            joint_positions = mx.concatenate(
                [
                    mx.broadcast_to(
                        mx.arange(text.shape[1]), (len(items), text.shape[1])
                    ),
                    audio_positions + text_sizes,
                ],
                axis=1,
            )
        else:
            pass

        def velocity(time: mx.array, x: mx.array) -> mx.array:
            prediction = self.transformer(
                x=x.astype(weight_dtype),
                text=text,
                time=time,
                mask=mask,
                c_mask=text_mask,
                ref=ref,
                ref_mask=ref_mask,
                audio_positions=audio_positions,
                joint_positions=joint_positions,
                projected_text=projected_text,
                cfg_infer=cfg_strength >= 1e-5,
            )
            if cfg_strength < 1e-5:
                return prediction
            else:
                conditional, unconditional = mx.split(prediction, 2, axis=0)
                return conditional + (conditional - unconditional) * cfg_strength

        time = build_time_grid(steps, sway_sampling_coef, t_grid)
        result = integrate(velocity, y0, time)
        return [latent[: item.target_frames] for item, latent in zip(items, result)]


def integrate(
    velocity: Callable[[mx.array, mx.array], mx.array], y0: mx.array, time: mx.array
) -> mx.array:
    y = y0
    for step in range(time.size - 1):
        y = y + (time[step + 1] - time[step]) * velocity(time[step], y)
        mx.eval(y)
    return y
