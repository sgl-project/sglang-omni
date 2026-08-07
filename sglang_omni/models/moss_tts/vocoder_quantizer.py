# SPDX-License-Identifier: Apache-2.0
"""Inference-only quantizer decode helpers for MOSS-TTS Delay."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def _pointwise_conv1d_parameters(
    module: nn.Module,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    if not isinstance(module, nn.Conv1d):
        raise TypeError(f"expected Conv1d, got {module.__class__.__name__}")
    if (
        module.kernel_size != (1,)
        or module.stride != (1,)
        or module.padding != (0,)
        or module.dilation != (1,)
        or module.groups != 1
    ):
        raise ValueError("cached quantizer decode requires pointwise Conv1d")
    weight = module.weight.detach().squeeze(-1).to(dtype=torch.float32)
    bias = None if module.bias is None else module.bias.detach().to(dtype=torch.float32)
    return weight, bias


class MossAudioTokenizerQuantizerDecoder:
    """Cache FP32 codebooks and projections used by residual codec decode."""

    def __init__(self, source: nn.Module) -> None:
        quantizers = list(getattr(source, "quantizers", ()))
        if not quantizers:
            raise ValueError("MOSS quantizer has no residual codebooks")

        codebooks: list[torch.Tensor] = []
        weights: list[torch.Tensor] = []
        biases: list[torch.Tensor | None] = []
        codebook_size = 0
        output_dim = 0
        for quantizer in quantizers:
            codebook = getattr(getattr(quantizer, "codebook", None), "weight", None)
            if not isinstance(codebook, torch.Tensor) or codebook.ndim != 2:
                raise TypeError("MOSS quantizer codebook must be a 2D tensor")
            weight, bias = _pointwise_conv1d_parameters(quantizer.out_proj)
            if int(codebook.shape[1]) != int(weight.shape[1]):
                raise ValueError("MOSS codebook and projection dimensions do not match")
            if not codebooks:
                codebook_size = int(codebook.shape[0])
                output_dim = int(weight.shape[0])
            elif (
                tuple(codebook.shape) != tuple(codebooks[0].shape)
                or int(weight.shape[0]) != output_dim
            ):
                raise ValueError(
                    "MOSS residual codebooks must share input and output sizes"
                )
            codebooks.append(codebook.detach().to(dtype=torch.float32))
            weights.append(weight)
            biases.append(bias)

        output_proj = getattr(source, "output_proj", None)
        if isinstance(output_proj, nn.Identity):
            output_weight = None
            output_bias = None
        else:
            output_weight, output_bias = _pointwise_conv1d_parameters(output_proj)
            if int(output_weight.shape[1]) != output_dim:
                raise ValueError(
                    "MOSS quantizer output projection has an unexpected input size"
                )

        self._num_quantizers = len(quantizers)
        self._output_dim = output_dim
        self._flat_codebooks = torch.stack(codebooks).flatten(0, 1)
        self._weights = tuple(weights)
        self._biases = tuple(biases)
        self._offsets = (
            torch.arange(
                self._num_quantizers,
                device=self._flat_codebooks.device,
                dtype=torch.long,
            )
            * codebook_size
        ).view(-1, 1, 1)
        self._output_weight = output_weight
        self._output_bias = output_bias

    def decode_codes(self, codes: torch.Tensor) -> torch.Tensor:
        if codes.ndim != 3:
            raise ValueError(
                f"MOSS quantizer codes must be [N, B, T], got {tuple(codes.shape)}"
            )
        num_quantizers = int(codes.shape[0])
        if num_quantizers <= 0 or num_quantizers > self._num_quantizers:
            raise ValueError(
                "MOSS quantizer codebook count must be within "
                f"[1, {self._num_quantizers}], got {num_quantizers}"
            )
        if codes.device != self._flat_codebooks.device:
            raise ValueError(
                "MOSS quantizer codes and cached weights must share one device"
            )

        _, batch_size, frames = codes.shape
        decoded = torch.zeros(
            batch_size,
            self._output_dim,
            frames,
            device=codes.device,
            dtype=torch.float32,
        )
        embedded = F.embedding(
            codes + self._offsets[:num_quantizers],
            self._flat_codebooks,
        )
        for index in range(num_quantizers):
            decoded.add_(
                F.conv1d(
                    embedded[index].transpose(1, 2),
                    self._weights[index].unsqueeze(-1),
                    self._biases[index],
                )
            )
        if self._output_weight is not None:
            decoded = F.conv1d(
                decoded,
                self._output_weight.unsqueeze(-1),
                self._output_bias,
            )
        return decoded


__all__ = ["MossAudioTokenizerQuantizerDecoder"]
