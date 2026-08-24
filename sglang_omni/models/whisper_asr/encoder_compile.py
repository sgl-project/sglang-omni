# SPDX-License-Identifier: Apache-2.0
"""torch.compile fusion for the fixed-shape Whisper encoder layers."""

from __future__ import annotations

import logging
from types import MethodType
from typing import Sequence

import torch
from torch import nn

logger = logging.getLogger(__name__)


def _bind_compiled_layers(layers: Sequence[nn.Module], *, mode: str | None) -> None:
    """Compile the shared layer forward once and bind it to every layer."""
    compiled = torch.compile(type(layers[0]).forward, mode=mode)
    for layer in layers:
        layer.forward = MethodType(compiled, layer)


def _unbind_layers(layers: Sequence[nn.Module]) -> None:
    for layer in layers:
        layer.__dict__.pop("forward", None)


def compile_encoder_layers(
    encoder: nn.Module,
    *,
    num_mel_bins: int,
    input_feature_len: int,
    mode: str | None = None,
) -> bool:
    """Fuse the Whisper encoder layers with ``torch.compile``."""
    layers = list(encoder.layers)
    parameter = next(encoder.parameters())
    if parameter.device.type != "cuda":
        logger.info("Whisper encoder compile skipped: encoder is not on CUDA")
        return False

    try:
        _bind_compiled_layers(layers, mode=mode)
        # note (Junnan Li): the first call runs Dynamo tracing and Inductor
        # codegen; pay it here instead of on the first serving request.
        with torch.no_grad(), torch.cuda.device(parameter.device):
            warmup = torch.zeros(
                1,
                num_mel_bins,
                input_feature_len,
                device=parameter.device,
                dtype=parameter.dtype,
            )
            encoder(warmup)
            torch.cuda.synchronize(parameter.device)
    except Exception:
        _unbind_layers(layers)
        logger.warning(
            "Whisper encoder compile failed; using eager encoder layers",
            exc_info=True,
        )
        return False
    logger.info("Compiled Whisper encoder layers count=%d mode=%s", len(layers), mode)
    return True


__all__ = ["compile_encoder_layers"]
