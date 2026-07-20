# SPDX-License-Identifier: MIT
# Copyright (c) 2025 inclusionAI
# Adapted from Ming-omni-tts/fm/CFM.py and Ming-omni-tts/fm/flowloss.py.

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from sglang_omni.models.ming_omni.talker.talker_module.cfm import get_epss_timesteps
from sglang_omni.models.ming_omni.talker.talker_module.dit import DiT


def build_cfm_timesteps(
    *,
    steps: int,
    device: torch.device,
    dtype: torch.dtype,
    use_epss: bool = True,
) -> torch.Tensor:
    if use_epss:
        return get_epss_timesteps(int(steps), device=device, dtype=dtype)
    return torch.linspace(0, 1, int(steps) + 1, device=device, dtype=dtype)


def build_cfm_sde_random(
    *,
    steps: int,
    device: torch.device,
    dtype: torch.dtype,
    batch_size: int,
    patch_size: int,
    latent_dim: int,
) -> torch.Tensor:
    return torch.randn(
        (int(steps) - 1, int(batch_size), int(patch_size), int(latent_dim)),
        device=device,
        dtype=dtype,
    )


def _expand_batch_param(
    value: float | torch.Tensor,
    *,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        tensor = value.to(device=device, dtype=dtype)
    else:
        tensor = torch.tensor(value, device=device, dtype=dtype)

    if tensor.ndim == 0 or int(tensor.numel()) == 1:
        return tensor.reshape(1, 1, 1).expand(int(batch_size), 1, 1)
    return tensor.reshape(int(batch_size), 1, 1)


class Solver:
    def __init__(
        self,
        func,
        y0: torch.Tensor,
        sigma: float | torch.Tensor = 0.25,
        temperature: float | torch.Tensor = 1.5,
    ) -> None:
        self.func = func
        self.y0 = y0
        self.sigma = sigma
        self.temperature = temperature

    def integrate(
        self,
        t: torch.Tensor,
        *,
        sde_random: torch.Tensor,
    ) -> torch.Tensor:
        step_count = int(t.shape[0]) - 1

        y0 = self.y0
        sampled = y0
        for step, (t0, t1) in enumerate(zip(t[:-1], t[1:])):
            dt = t1 - t0
            f0 = self.func(t0, y0)
            y1 = y0 + dt * f0
            sampled = y1

            if step + 1 < step_count:
                noise = sde_random[step]
                shift = self.sigma * (self.temperature**0.5) * (abs(dt) ** 0.5) * noise
                y0 = y1 + shift

        return sampled


class CFM(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(
        self,
        cond: torch.Tensor,
        target: torch.Tensor,
        latent_history: torch.Tensor,
        mask: torch.Tensor,
        patch_size: int,
    ) -> torch.Tensor:
        x1 = target
        batch, dtype = x1.shape[0], x1.dtype
        x0 = torch.randn_like(x1)
        time = torch.rand((batch,), dtype=dtype, device=self.device)
        t = time.unsqueeze(-1).unsqueeze(-1)
        x = (1 - t) * x0 + t * x1
        flow = x1 - x0

        pred = self.model(
            x=x,
            t=time,
            c=cond,
            latent_history=latent_history,
        )
        pred = pred[:, -patch_size:, :]

        loss = F.mse_loss(pred, flow, reduction="none")
        loss = loss[mask == 1]
        return loss.mean()

    @torch.no_grad()
    def sample(
        self,
        noise: torch.Tensor,
        c: torch.Tensor,
        latent_history: torch.Tensor,
        timesteps: torch.Tensor,
        sde_random: torch.Tensor,
        cfg_scale: float = 1.0,
        sway_sampling_coef: float | None = -1.0,
        sigma: float | torch.Tensor = 0.25,
        temperature: float | torch.Tensor = 1.5,
    ) -> torch.Tensor:
        fn, y0, t, sigma_tensor, temperature_tensor = self._prepare_sampling(
            noise=noise,
            c=c,
            latent_history=latent_history,
            timesteps=timesteps,
            cfg_scale=cfg_scale,
            sway_sampling_coef=sway_sampling_coef,
            sigma=sigma,
            temperature=temperature,
        )
        solver = Solver(fn, y0, sigma=sigma_tensor, temperature=temperature_tensor)
        return solver.integrate(t, sde_random=sde_random)

    def _prepare_sampling(
        self,
        *,
        noise: torch.Tensor,
        c: torch.Tensor,
        latent_history: torch.Tensor,
        timesteps: torch.Tensor,
        cfg_scale: float | torch.Tensor,
        sway_sampling_coef: float | None,
        sigma: float | torch.Tensor,
        temperature: float | torch.Tensor,
    ):
        batch_size = int(noise.shape[0])
        cfg_tensor = _expand_batch_param(
            cfg_scale,
            batch_size=batch_size,
            device=noise.device,
            dtype=noise.dtype,
        )
        sigma_tensor = _expand_batch_param(
            sigma,
            batch_size=batch_size,
            device=noise.device,
            dtype=noise.dtype,
        )
        temperature_tensor = _expand_batch_param(
            temperature,
            batch_size=batch_size,
            device=noise.device,
            dtype=noise.dtype,
        )

        def fn(t, x):
            pred_cfg = self.model.forward_with_cfg(
                x=x,
                t=t,
                c=c,
                latent_history=latent_history,
            )
            pred, null_pred = torch.chunk(pred_cfg, 2, dim=0)
            return pred + (pred - null_pred) * cfg_tensor

        y0 = noise.transpose(1, 2)
        t = timesteps
        if sway_sampling_coef is not None:
            t = t + sway_sampling_coef * (torch.cos(torch.pi / 2 * t) - 1 + t)

        return fn, y0, t, sigma_tensor, temperature_tensor


class FlowLoss(nn.Module):
    """Ming-Omni-TTS flow-matching latent head."""

    def __init__(
        self,
        z_channels: int,
        llm_cond_dim: int,
        patch_size: int | None = None,
        history_patch_size: int | None = None,
        **dit_kwargs,
    ) -> None:
        super().__init__()
        del patch_size, history_patch_size
        self.z_channels = z_channels
        self.cfm = CFM(
            model=DiT(
                in_channels=z_channels,
                llm_cond_dim=llm_cond_dim,
                **dit_kwargs,
            )
        )

    def forward(
        self,
        cond: torch.Tensor,
        target: torch.Tensor,
        latent_history: torch.Tensor,
        mask: torch.Tensor,
        patch_size: int,
    ) -> torch.Tensor:
        return self.cfm(
            cond=cond,
            target=target,
            latent_history=latent_history,
            mask=mask,
            patch_size=patch_size,
        )

    def sample(
        self,
        z: torch.Tensor,
        latent_history: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
        sde_random: torch.Tensor,
        cfg: float | torch.Tensor = 1.0,
        sigma: float | torch.Tensor = 0.25,
        temperature: float | torch.Tensor = 0,
    ) -> torch.Tensor:
        return self.cfm.sample(
            noise=noise,
            c=z,
            latent_history=latent_history,
            cfg_scale=cfg,
            sigma=sigma,
            temperature=temperature,
            timesteps=timesteps,
            sde_random=sde_random,
        )


__all__ = ["CFM", "FlowLoss", "Solver", "build_cfm_timesteps", "build_cfm_sde_random"]
