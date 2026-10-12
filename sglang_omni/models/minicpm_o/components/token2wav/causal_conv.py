# SPDX-License-Identifier: Apache-2.0
"""Shared causal convolution and explicit streaming history."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True, kw_only=True)
class ConvState:
    """Empty history enables streaming; no state disables caching.

    With valid_frame_counts the next history ends at each row's last real frame.
    """

    history: torch.Tensor | None = None
    valid_frame_counts: torch.Tensor | None = None


class CausalConv1d(nn.Conv1d):
    def forward(
        self, x: torch.Tensor, state: ConvState | None = None
    ) -> tuple[torch.Tensor, ConvState | None]:
        history_length = (self.kernel_size[0] - 1) * self.dilation[0]
        if state is not None and state.history is not None:
            x = torch.cat((state.history, x), dim=2)
        else:
            x = F.pad(x, (history_length, 0))
        if state is None:
            next_state = None
        elif state.valid_frame_counts is None:
            next_state = ConvState(
                history=x[:, :, x.shape[2] - history_length :].clone()
            )
        else:
            positions = state.valid_frame_counts[:, None] + torch.arange(
                history_length, device=x.device
            )
            next_state = ConvState(
                history=torch.gather(
                    x, 2, positions[:, None, :].expand(-1, x.shape[1], -1)
                )
            )
        return super().forward(x), next_state
