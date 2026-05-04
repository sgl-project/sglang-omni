# SPDX-License-Identifier: Apache-2.0
"""Vision projector for Ming-Omni: maps vision encoder output to LLM hidden space.

Architecture (mlp_depth=2, default for Ming-flash-omni-2.0):
  Linear(3584, 4096) → GELU → Linear(4096, 4096)

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

from sglang_omni_v1.models.weight_loader import default_weight_loader

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
        layers: list[nn.Module] = [nn.Linear(vision_dim, llm_dim)]
        for _ in range(1, mlp_depth):
            layers.append(nn.GELU())
            layers.append(nn.Linear(llm_dim, llm_dim))
        self.proj = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)

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
