from __future__ import annotations

import torch
from torch import nn


class AddFusion(nn.Module):
    def __init__(self, config: dict) -> None:
        super().__init__()
        self.user_weight = float(config["duplex_user_channel_weight"])
        self.text_weight = float(config["duplex_text_channel_weight"])
        self.function_weight = float(config["duplex_function_channel_weight"])

    def forward(
        self, acoustic, text, function: torch.Tensor | None = None
    ) -> torch.Tensor:
        output = (self.user_weight * acoustic) + (self.text_weight * text)
        if function is not None:
            output = output + (self.function_weight * function)
        return output
