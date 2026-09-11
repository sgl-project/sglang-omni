# SPDX-License-Identifier: Apache-2.0
# Ported from OpenBMB/VoxCPM (Apache-2.0), src/voxcpm/modules/locdit/unified_cfm.py.
"""Flow-matching sampler that turns one AR step's hidden state into a latent patch."""

from __future__ import annotations

import torch
from pydantic import BaseModel

from sglang_omni.models.voxcpm2.components.local_dit import VoxCPMLocDiT


class CfmConfig(BaseModel):
    sigma_min: float = 1e-6
    solver: str = "euler"
    t_scheduler: str = "log-norm"
    inference_cfg_rate: float = 1.0


class UnifiedCFM(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        cfm_params: CfmConfig,
        estimator: VoxCPMLocDiT,
        mean_mode: bool = False,
    ):
        super().__init__()
        if cfm_params.solver != "euler":
            raise ValueError(
                f"VoxCPM2 only implements the euler solver, got {cfm_params.solver!r}"
            )
        self.solver = cfm_params.solver
        self.sigma_min = cfm_params.sigma_min
        self.t_scheduler = cfm_params.t_scheduler
        self.inference_cfg_rate = cfm_params.inference_cfg_rate
        self.in_channels = in_channels
        self.mean_mode = mean_mode
        self.estimator = estimator

    @torch.inference_mode()
    def forward(
        self,
        mu: torch.Tensor,
        n_timesteps: int,
        patch_size: int,
        cond: torch.Tensor,
        temperature: float = 1.0,
        cfg_value: float = 1.0,
        sway_sampling_coef: float = 1.0,
        use_cfg_zero_star: bool = True,
    ) -> torch.Tensor:
        batch = mu.shape[0]
        x = (
            torch.randn(
                (batch, self.in_channels, patch_size), device=mu.device, dtype=mu.dtype
            )
            * temperature
        )

        t_span = torch.linspace(1, 0, n_timesteps + 1, device=mu.device, dtype=mu.dtype)
        t_span = t_span + sway_sampling_coef * (
            torch.cos(torch.pi / 2 * t_span) - 1 + t_span
        )

        return self.solve_euler(
            x=x,
            t_span=t_span,
            mu=mu,
            cond=cond,
            cfg_value=cfg_value,
            use_cfg_zero_star=use_cfg_zero_star,
        )

    def optimized_scale(
        self, positive_flat: torch.Tensor, negative_flat: torch.Tensor
    ) -> torch.Tensor:
        dot_product = torch.sum(positive_flat * negative_flat, dim=1, keepdim=True)
        squared_norm = torch.sum(negative_flat**2, dim=1, keepdim=True) + 1e-8
        return dot_product / squared_norm

    def solve_euler(
        self,
        x: torch.Tensor,
        t_span: torch.Tensor,
        mu: torch.Tensor,
        cond: torch.Tensor,
        cfg_value: float = 1.0,
        use_cfg_zero_star: bool = True,
    ) -> torch.Tensor:
        t, dt = t_span[0], t_span[0] - t_span[1]
        batch = x.size(0)
        zero_init_steps = max(1, int(len(t_span) * 0.04))

        for step in range(1, len(t_span)):
            if use_cfg_zero_star and step <= zero_init_steps:
                dphi_dt = torch.zeros_like(x)
            else:
                x_in = torch.cat([x, x], dim=0)
                cond_in = torch.cat([cond, cond], dim=0)
                mu_in = torch.zeros(
                    [2 * batch, mu.size(1)], device=x.device, dtype=x.dtype
                )
                mu_in[:batch] = mu
                t_in = t.repeat(2 * batch)
                dt_in = (
                    dt.repeat(2 * batch) if self.mean_mode else torch.zeros_like(t_in)
                )

                dphi_dt = self.estimator(x_in, mu_in, t_in, cond_in, dt_in)
                dphi_dt, cfg_dphi_dt = torch.split(dphi_dt, [batch, batch], dim=0)

                if use_cfg_zero_star:
                    st_star = self.optimized_scale(
                        dphi_dt.view(batch, -1), cfg_dphi_dt.view(batch, -1)
                    )
                    st_star = st_star.view(batch, *([1] * (dphi_dt.ndim - 1)))
                else:
                    st_star = 1.0

                dphi_dt = cfg_dphi_dt * st_star + cfg_value * (
                    dphi_dt - cfg_dphi_dt * st_star
                )

            x = x - dt * dphi_dt
            t = t - dt
            if step < len(t_span) - 1:
                dt = t - t_span[step + 1]

        return x


__all__ = ["CfmConfig", "UnifiedCFM"]
