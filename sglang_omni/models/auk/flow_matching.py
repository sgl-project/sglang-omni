# SPDX-License-Identifier: MIT
# Copyright (C) 2026 Tencent. All rights reserved.
# Derived from Tencent-Hunyuan/AuK; see LICENSE for the MIT permission notice.

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils.rnn import pad_sequence

from sglang_omni.models.auk.dit import AuKDit

if TYPE_CHECKING:
    from sglang_omni.models.auk.step_cuda_graph import AuKStepCudaGraphRunner


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


def _pad_rows(tensor: torch.Tensor, rows: int) -> torch.Tensor:
    """Pad axis 1 up to ``rows`` with zeros, i.e. False for a boolean mask."""
    extra = rows - tensor.shape[1]
    if extra <= 0:
        return tensor
    return F.pad(tensor, [0, 0] * (tensor.ndim - 2) + [0, extra])


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
        # The checkpoint stores these next to the DiT, so a strict load needs them
        # here; the conditioning stage reads its own copy of the same tensors.
        self.layer_weights = nn.Parameter(torch.zeros(num_llm_layers))
        self.layer_scale = nn.Parameter(torch.ones(1))

    @torch.no_grad()
    def sample(
        self,
        item: AuKSampleItem,
        *,
        steps: int,
        cfg_strength: float,
        sway_sampling_coef: float | None = None,
        t_grid: Sequence[float] | None = None,
        step_graph: AuKStepCudaGraphRunner | None = None,
    ) -> torch.Tensor:
        return self.sample_batch(
            [item],
            steps=steps,
            cfg_strength=cfg_strength,
            sway_sampling_coef=sway_sampling_coef,
            t_grid=t_grid,
            step_graph=step_graph,
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
        step_graph: AuKStepCudaGraphRunner | None = None,
    ) -> list[torch.Tensor]:
        """Integrate the velocity field for a batch of requests.

        With a ``step_graph`` the batch pads to that runner's shape buckets, so
        one captured step can be replayed for every NFE step.
        """
        device = next(self.parameters()).device
        dim = self.transformer.latent_dim
        # Inputs follow the backbone dtype; y stays fp32 through type promotion.
        weight_dtype = self.transformer.dtype

        def pack(tensors, rows=None):
            packed = (
                tensors[0].unsqueeze(0)
                if len(tensors) == 1
                else pad_sequence(tensors, batch_first=True)
            )
            return packed if rows is None else _pad_rows(packed, rows)

        references = [
            (
                item.ref_latent
                if item.ref_latent is not None
                else torch.zeros(0, dim, device=device)
            )
            for item in items
        ]
        # A runner declines a batch too wide to gain from a graph, and then this
        # batch neither pads nor binds: buckets carries both decisions.
        buckets = None
        if step_graph is not None:
            buckets = step_graph.pad_lengths(
                frames=max(item.target_frames for item in items),
                ref=max(reference.shape[0] for reference in references),
                text=max(item.conditioning.shape[0] for item in items),
                batch=len(items),
            )
        frame_rows, ref_rows, text_rows = (
            buckets if buckets is not None else (None, None, None)
        )

        ref = pack(references, ref_rows).to(weight_dtype)
        ref_mask = (
            torch.arange(ref.shape[1], device=device)[None, :]
            < torch.tensor([item.ref_length for item in items], device=device)[:, None]
        )
        text = pack([item.conditioning for item in items], text_rows).to(weight_dtype)
        text_mask = pack([item.text_mask for item in items], text_rows)
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
        y0 = pack(noise, frame_rows)
        mask = audio_positions = joint_positions = None
        # Positions come from the real lengths, so a padded batch places each
        # request's target frames where the unpadded single request would.
        if len(items) > 1 or buckets is not None:
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

        inputs = dict(
            text=text,
            mask=mask,
            c_mask=text_mask,
            ref=ref,
            ref_mask=ref_mask,
            # The projected text is constant per trajectory either way; a graph
            # holds it in its own buffers and must not write the python cache.
            cache=buckets is None,
            audio_positions=audio_positions,
            joint_positions=joint_positions,
        )

        def step(inputs, t, x):
            kwargs = dict(inputs, x=x.to(weight_dtype), time=t)
            if cfg_strength < 1e-5:
                return self.transformer(
                    **kwargs, drop_audio_cond=False, drop_text=False
                )
            pred = self.transformer(**kwargs, cfg_infer=True)
            v_cond, v_uncond = torch.chunk(pred, 2, dim=0)
            return v_cond + (v_cond - v_uncond) * cfg_strength

        t = build_time_grid(steps, sway_sampling_coef, t_grid, device=device)
        fn = None
        if buckets is not None:
            fn = step_graph.bind(step, inputs, x=y0, time=t[0], baked=(cfg_strength,))
        try:
            result = integrate(fn or partial(step, inputs), y0, t)
            return [latent[: item.target_frames] for item, latent in zip(items, result)]
        finally:
            self.transformer.clear_cache()


def integrate(fn, y0: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Fixed-grid Euler integration, matching the released inference recipe."""
    y = y0
    for step in range(t.numel() - 1):
        y = y + (t[step + 1] - t[step]) * fn(t[step], y)
    return y
