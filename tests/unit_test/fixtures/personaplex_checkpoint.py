# SPDX-License-Identifier: Apache-2.0
"""LM weights small enough to tell a PersonaPlex checkpoint from a Moshi one."""

from __future__ import annotations

from pathlib import Path

import torch
from safetensors.torch import save_file

PERSONAPLEX_DEPFORMER_STEPS = 16
MOSHI_DEPFORMER_STEPS = 8


def write_lm_weights(model_dir: Path, depformer_steps: int) -> None:
    """model.safetensors with one depformer output head per step and nothing else."""
    save_file(
        {f"linears.{step}.weight": torch.zeros(1) for step in range(depformer_steps)},
        str(model_dir / "model.safetensors"),
    )
