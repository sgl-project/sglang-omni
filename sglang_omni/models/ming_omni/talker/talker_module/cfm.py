from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def get_epss_timesteps(n, device, dtype):
    dt = 1 / 32
    predefined_timesteps = {
        5: [0, 2, 4, 8, 16, 32],
        6: [0, 2, 4, 6, 8, 16, 32],
        7: [0, 2, 4, 6, 8, 16, 24, 32],
        10: [0, 2, 4, 6, 8, 12, 16, 20, 24, 28, 32],
        12: [0, 2, 4, 6, 8, 10, 12, 14, 16, 20, 24, 28, 32],
        16: [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 20, 24, 28, 32],
    }
    t = predefined_timesteps.get(n, [])
    if not t:
        return torch.linspace(0, 1, n + 1, device=device, dtype=dtype)
    return dt * torch.tensor(t, device=device, dtype=dtype)


class CFM(nn.Module):
    def __init__(self, model, steps=10, sway_sampling_coef=-1):
        super().__init__()
        # transformer
        self.model = model

        self.steps = steps
        self.sway_sampling_coef = sway_sampling_coef

    @torch.no_grad()
    def sample(self, llm_cond, lat_cond, y0, t, sde_args, sde_rnd):
        # cfg_strength=2, sigma=0, temperature=0
        def fn(fn_t, x):
            pred_cfg = self.model.forward_with_cfg(x, fn_t, llm_cond, lat_cond, None)
            pred, null_pred = torch.chunk(pred_cfg, 2, dim=0)
            return pred + (pred - null_pred) * sde_args[0]

        if self.sway_sampling_coef is not None:
            t = t + self.sway_sampling_coef * (torch.cos(torch.pi / 2 * t) - 1 + t)

        trajectory = [y0]
        for step in range(self.steps):
            dt = t[step + 1] - t[step]
            y0 = y0 + fn(t[step], y0) * dt
            y0 = (
                y0
                + sde_args[1] * (sde_args[2] ** 0.5) * (dt.abs() ** 0.5) * sde_rnd[step]
            )
            trajectory.append(y0)

        sampled = trajectory[-1]
        out = sampled

        return out

    def forward(
        self, llm_cond, lat_cond, lat_tag, loss_weight, spk_emb=None, bat_size=None
    ):
        # mel is x1
        x1 = lat_tag.detach()

        # x0 is gaussian noise
        x0 = torch.randn_like(x1)

        # time step
        time = torch.rand(
            (llm_cond.shape[0],), dtype=llm_cond.dtype, device=llm_cond.device
        )

        # sample xt (φ_t(x) in the paper)
        t = time.unsqueeze(-1).unsqueeze(-1)
        xt = (1 - t) * x0 + t * x1
        flow = x1 - x0

        # forward
        pred = self.model(xt, time, llm_cond, lat_cond, spk_emb)[:, -lat_tag.shape[1] :]

        # flow matching loss
        loss = F.mse_loss(pred, flow, reduction="none")
        loss = loss * loss_weight

        return loss.sum()
