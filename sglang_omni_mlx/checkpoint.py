# SPDX-License-Identifier: Apache-2.0
"""Safetensors checkpoints into MLX modules, quantized as the checkpoint was."""

from __future__ import annotations

import glob
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn


def read_weights(model_directory: Path) -> dict[str, mx.array]:
    weights: dict[str, mx.array] = {}
    for path in sorted(glob.glob(str(model_directory / "*.safetensors"))):
        weights.update(mx.load(path))
    return weights


def load_weights(
    model: nn.Module,
    weights: dict[str, mx.array],
    quantization: dict[str, object] | None,
) -> None:
    """Every parameter of model from weights, which must name each one exactly once.

    With a quantization config, the layers whose weights carry scales are
    quantized first, so a quantized checkpoint loads into a float module tree.
    """
    if quantization is not None:
        nn.quantize(
            model,
            group_size=quantization["group_size"],
            bits=quantization["bits"],
            mode=quantization.get("mode", "affine"),
            class_predicate=lambda path, module: f"{path}.scales" in weights,
        )
    else:
        pass
    model.load_weights(list(weights.items()), strict=True)
    mx.eval(model.parameters())
