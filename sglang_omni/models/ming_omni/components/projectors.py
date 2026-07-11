# SPDX-License-Identifier: Apache-2.0
"""Vision projector for Ming-Omni: maps vision encoder output to LLM hidden space.

Architecture (mlp_depth=2, default for Ming-flash-omni-2.0):
  ColumnParallelLinear(3584, 4096) → GELU → RowParallelLinear(4096, 4096)

Weight checkpoint mapping (direct, no remapping needed):
  linear_proj.0.weight → proj.0.weight
  linear_proj.0.bias   → proj.0.bias
  linear_proj.2.weight → proj.2.weight
  linear_proj.2.bias   → proj.2.bias
"""

from __future__ import annotations

import logging
from typing import Iterable, Tuple

import torch
import torch.nn as nn
from sglang.srt.layers.linear import ColumnParallelLinear, RowParallelLinear

from sglang_omni.models.weight_loader import default_weight_loader

logger = logging.getLogger(__name__)


class VisionProjector(nn.Module):
    """MLP projector from vision encoder output to LLM hidden space.

    Args:
        vision_dim: Vision encoder output dimension (out_hidden_size, e.g. 3584).
        llm_dim: LLM hidden dimension (e.g. 4096).
        mlp_depth: Number of linear layers. 1 = single linear, 2 = linear+GELU+linear.
    """

    def __init__(self, vision_dim: int, llm_dim: int, mlp_depth: int = 1) -> None:
        super().__init__()
        gather_first = mlp_depth == 1
        layers: list[nn.Module] = [
            ColumnParallelLinear(
                vision_dim,
                llm_dim,
                bias=True,
                gather_output=gather_first,
                quant_config=None,
            )
        ]
        for _ in range(1, mlp_depth):
            layers.append(nn.GELU())
            layers.append(
                RowParallelLinear(llm_dim, llm_dim, bias=True, quant_config=None)
            )
        self.proj = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.proj[0](x)
        for layer in self.proj[1:]:
            if isinstance(layer, RowParallelLinear):
                out, _ = layer(out)
            else:
                out = layer(out)
        return out

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]) -> set[str]:
        """Load weights with prefix normalization.

        Checkpoint keys like ``0.weight`` get prepended with ``proj.``
        to match the internal Sequential structure.
        """
        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()

        for name, loaded_weight in weights:
            if not name.startswith("proj."):
                name = f"proj.{name}"

            if name not in params_dict:
                logger.debug("Skipping unknown projector weight: %s", name)
                continue

            param = params_dict[name]
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, loaded_weight)
            loaded_params.add(name)

        return loaded_params
