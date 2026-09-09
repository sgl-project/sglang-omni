# SPDX-License-Identifier: MIT
# Copyright (C) 2026 Tencent. All rights reserved.
# Derived from Tencent-Hunyuan/AuK; see LICENSE for the MIT permission notice.

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils.rnn import pad_sequence

from sglang_omni.models.auk.dit import AuKDit


def request_generator(
    seed: int | None, device: torch.device | str
) -> torch.Generator | None:
    if seed is None:
        return None
    return torch.Generator(device=device).manual_seed(int(seed))


def fuse_hidden_states(hidden_states, layer_weights, layer_scale):
    d_llm = hidden_states.shape[-1]
    stacked = F.layer_norm(hidden_states[:, 1:], [d_llm])
    weights = F.softmax(layer_weights, dim=0)
    return (stacked * weights[None, :, None, None]).sum(dim=1) * layer_scale


def build_time_grid(
    steps: int,
    sway_sampling_coef: float | None = None,
    t_grid: Sequence[float] | None = None,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    if t_grid is not None:
        grid = torch.tensor(list(t_grid), device=device, dtype=torch.float32)
        if grid.ndim != 1 or grid.numel() < 2:
            raise ValueError("t_grid must hold at least two time points")
        return grid
    if steps < 1:
        raise ValueError("AuK nfe must be positive")
    t = torch.linspace(0, 1, steps + 1, device=device, dtype=torch.float32)
    if sway_sampling_coef is not None:
        t = t + sway_sampling_coef * (torch.cos(torch.pi / 2 * t) - 1 + t)
    return t


@dataclass
class AuKSampleItem:
    conditioning: torch.Tensor
    text_mask: torch.Tensor
    target_frames: int
    ref_latent: torch.Tensor | None = None
    seed: int | None = None
    ref_length: int = 0


class AuKFlowMatching(nn.Module):
    """Velocity-field integration for variable-length request batches."""

    def __init__(self, transformer: AuKDit, num_llm_layers: int):
        super().__init__()
        self.transformer = transformer
        self.layer_weights = nn.Parameter(torch.zeros(num_llm_layers))
        self.layer_scale = nn.Parameter(torch.ones(1))

    def fuse(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return fuse_hidden_states(hidden_states, self.layer_weights, self.layer_scale)

    @torch.no_grad()
    def sample(
        self,
        item: AuKSampleItem,
        *,
        steps: int,
        cfg_strength: float,
        sway_sampling_coef: float | None = None,
        t_grid: Sequence[float] | None = None,
    ) -> torch.Tensor:
        return self.sample_batch(
            [item],
            steps=steps,
            cfg_strength=cfg_strength,
            sway_sampling_coef=sway_sampling_coef,
            t_grid=t_grid,
        )[0]

    @torch.no_grad()
    def sample_batch(
        self,
        items: Sequence[AuKSampleItem],
        *,
        steps: int,
        cfg_strength: float,
        sway_sampling_coef: float | None = None,
        t_grid: Sequence[float] | None = None,
    ) -> list[torch.Tensor]:
        device = next(self.parameters()).device
        dim = self.transformer.latent_dim

        def pack(tensors):
            return (
                tensors[0].unsqueeze(0)
                if len(tensors) == 1
                else pad_sequence(tensors, batch_first=True)
            )

        references = [
            (
                item.ref_latent
                if item.ref_latent is not None
                else torch.zeros(0, dim, device=device)
            )
            for item in items
        ]
        ref = pack(references)
        ref_mask = (
            torch.arange(ref.shape[1], device=device)[None, :]
            < torch.tensor([item.ref_length for item in items], device=device)[:, None]
        )
        text = pack([item.conditioning for item in items])
        text_mask = pack([item.text_mask for item in items])
        noise = []
        for item in items:
            generator = request_generator(item.seed, device)
            noise.append(
                torch.randn(
                    item.target_frames,
                    dim,
                    device=device,
                    dtype=torch.float32,
                    generator=generator,
                )
            )
        y0 = pack(noise)
        mask = audio_positions = joint_positions = None
        if len(items) > 1:
            target_positions = torch.arange(y0.shape[1], device=device)[None, :]
            mask = (
                target_positions
                < torch.tensor([item.target_frames for item in items], device=device)[
                    :, None
                ]
            )
            ref_sizes = torch.tensor(
                [ref.shape[0] for ref in references], device=device
            )[:, None]
            text_sizes = torch.tensor(
                [item.conditioning.shape[0] for item in items], device=device
            )[:, None]
            audio_positions = torch.cat(
                [
                    torch.arange(ref.shape[1], device=device)[None, :].expand(
                        len(items), -1
                    ),
                    target_positions + ref_sizes,
                ],
                dim=1,
            )
            joint_positions = torch.cat(
                [
                    torch.arange(text.shape[1], device=device)[None, :].expand(
                        len(items), -1
                    ),
                    audio_positions + text_sizes,
                ],
                dim=1,
            )

        def fn(t, x):
            kwargs = dict(
                x=x,
                text=text,
                time=t,
                mask=mask,
                c_mask=text_mask,
                ref=ref,
                ref_mask=ref_mask,
                cache=True,
                audio_positions=audio_positions,
                joint_positions=joint_positions,
            )
            if cfg_strength < 1e-5:
                return self.transformer(
                    **kwargs, drop_audio_cond=False, drop_text=False
                )
            pred = self.transformer(**kwargs, cfg_infer=True)
            v_cond, v_uncond = torch.chunk(pred, 2, dim=0)
            return v_cond + (v_cond - v_uncond) * cfg_strength

        t = build_time_grid(steps, sway_sampling_coef, t_grid, device=device)
        try:
            result = integrate(fn, y0, t)
            return [latent[: item.target_frames] for item, latent in zip(items, result)]
        finally:
            self.transformer.clear_cache()


def integrate(fn, y0: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Fixed-grid Euler integration, matching the released inference recipe."""
    y = y0
    for step in range(t.numel() - 1):
        y = y + (t[step + 1] - t[step]) * fn(t[step], y)
    return y
