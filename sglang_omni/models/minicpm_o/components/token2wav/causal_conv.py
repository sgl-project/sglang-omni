# SPDX-License-Identifier: Apache-2.0
"""Shared causal convolution and explicit streaming history."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True, kw_only=True)
class ConvState:
    """Empty history enables streaming; no state disables caching."""

    history: torch.Tensor | None = None


class CausalConv1d(nn.Conv1d):
    def forward(
        self, x: torch.Tensor, state: ConvState | None = None
    ) -> tuple[torch.Tensor, ConvState | None]:
        history_length = (self.kernel_size[0] - 1) * self.dilation[0]
        if state is not None and state.history is not None:
            x = torch.cat((state.history, x), dim=2)
        else:
            x = F.pad(x, (history_length, 0))
        next_state = (
            ConvState(history=x[:, :, x.shape[2] - history_length :].clone())
            if state is not None
            else None
        )
        return super().forward(x), next_state
