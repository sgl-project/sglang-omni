from __future__ import annotations

import torch
from einops import einsum, rearrange
from torch import nn
from torch.nn import functional

RMS_NORM_EPS = 1e-6
MIN_LOG_STD = -4.0


class RMSNorm(nn.Module):
    # Gemma's variant: the stored weight is an offset from 1.0, not the scale.
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(dim))

    def forward(self, hidden_TD):
        hidden = hidden_TD.float()
        hidden = hidden * torch.rsqrt(
            hidden.pow(2).mean(-1, keepdim=True) + RMS_NORM_EPS
        )
        return (hidden * (1.0 + self.weight.float())).type_as(hidden_TD)


class GatedMLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.act_fn = nn.GELU(approximate="tanh")

    def forward(self, hidden_TD):
        return self.down_proj(
            self.act_fn(self.gate_proj(hidden_TD)) * self.up_proj(hidden_TD)
        )


class MLPLayer(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int) -> None:
        super().__init__()
        self.pre_norm = RMSNorm(hidden_size)
        self.mlp = GatedMLP(hidden_size, intermediate_size)
        self.post_norm = RMSNorm(hidden_size)

    def forward(self, hidden_TD):
        return hidden_TD + self.post_norm(self.mlp(self.pre_norm(hidden_TD)))


class MoGHead(nn.Module):
    """Mixture of Gaussians over the codec's continuous latent.

    The 1024 means are factorised rank-64 so only the sampled component's
    mean is reconstructed, not all of them.
    """

    def __init__(self, config: dict) -> None:
        super().__init__()
        hidden_size = int(config["hidden_size"])
        intermediate_size = int(config["intermediate_size"])
        num_layers = int(config["num_layers"])
        self.num_predictions = int(config["num_predictions"])
        self.out_size = int(config["out_size"])
        self.low_rank = int(config["low_rank"])

        self.mlp_stack = nn.Sequential(
            *(MLPLayer(hidden_size, intermediate_size) for _ in range(num_layers)),
            RMSNorm(hidden_size),
        )
        self.proj_logits = nn.Linear(hidden_size, self.num_predictions, bias=False)
        self.proj_mus = nn.Linear(
            hidden_size, self.num_predictions * self.low_rank, bias=False
        )
        self.proj_logs = nn.Linear(hidden_size, 1, bias=False)
        self.proj_else = nn.Linear(hidden_size, self.out_size, bias=False)
        self.low_mat = nn.Parameter(
            torch.empty(self.num_predictions, self.out_size, self.low_rank)
        )

    def infer(
        self, hidden_TD, *, guidance_scale: float = 0.0, top_p: float | None = None
    ):
        """Sample one latent per position. Under guidance the conditioned and
        unconditioned halves arrive stacked along the batch axis."""
        hidden_TD = self.mlp_stack(hidden_TD)
        if guidance_scale > 0:
            conditioned, unconditioned = hidden_TD.chunk(2, dim=0)
            hidden_TD = conditioned + guidance_scale * (conditioned - unconditioned)

        logits_TN = self.proj_logits(hidden_TD)
        if top_p is not None:
            logits_TN = self._nucleus(logits_TN, top_p)
        # Gumbel-max over the log-softmax draws one component per position.
        with torch.autocast(device_type=hidden_TD.device.type, enabled=False):
            gumbel = -torch.log(-torch.log(torch.rand_like(logits_TN.float())))
            component_T = (
                functional.log_softmax(logits_TN.float(), dim=-1) + gumbel
            ).argmax(-1)

        coefficient_TR = self._component_matmul(
            hidden_TD, self.proj_mus.weight, component_T, self.low_rank
        )
        mean_TO = einsum(
            coefficient_TR,
            self.low_mat[component_T],
            "t r, t o r -> t o",
        )
        log_std_T1 = self.proj_logs(hidden_TD).clamp_min(MIN_LOG_STD)
        return mean_TO * torch.exp(log_std_T1) + self.proj_else(hidden_TD), log_std_T1

    def _component_matmul(self, hidden_TD, weight_ND, component_T, out_size):
        blocks = rearrange(weight_ND, "(n r) d -> n r d", r=out_size)
        return einsum(hidden_TD, blocks[component_T], "t d, t r d -> t r")

    @staticmethod
    def _nucleus(logits_TN, top_p: float):
        ordered, order = logits_TN.sort(dim=-1, descending=True)
        cumulative = ordered.softmax(dim=-1).cumsum(dim=-1)
        drop = cumulative - ordered.softmax(dim=-1) > top_p
        ordered = ordered.masked_fill(drop, float("-inf"))
        return ordered.scatter(-1, order, ordered)
