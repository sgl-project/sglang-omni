# SPDX-License-Identifier: Apache-2.0
"""Audio8 semantic and residual vector quantizers."""

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.nn.utils import weight_norm as legacy_weight_norm

from sglang_omni.models.audio8_tts.codec.attention import (
    CodecTransformerConfig,
    CodecWindowTransformer,
)
from sglang_omni.models.audio8_tts.codec.convolution import (
    CODEC_CHANNEL_COUNT,
    CausalConv1d,
    CausalConvTranspose1d,
)

SEMANTIC_CODEBOOK_SIZE = 4096
RESIDUAL_CODEBOOK_SIZE = 1024
RESIDUAL_CODEBOOK_COUNT = 9
CODEBOOK_COUNT = 1 + RESIDUAL_CODEBOOK_COUNT
CODEBOOK_FEATURES = 8


class VectorQuantizer(nn.Module):
    def __init__(self, codebook_size: int) -> None:
        super().__init__()
        self.in_proj: nn.Conv1d = legacy_weight_norm(
            nn.Conv1d(CODEC_CHANNEL_COUNT, CODEBOOK_FEATURES, kernel_size=1)
        )
        self.out_proj: nn.Conv1d = legacy_weight_norm(
            nn.Conv1d(CODEBOOK_FEATURES, CODEC_CHANNEL_COUNT, kernel_size=1)
        )
        self.codebook: nn.Embedding = nn.Embedding(codebook_size, CODEBOOK_FEATURES)

    def decode(self, indices: Tensor) -> Tensor:
        embedded = F.embedding(indices, self.codebook.weight)
        return self.out_proj(embedded.transpose(1, 2))

    def forward(self, features: Tensor) -> tuple[Tensor, Tensor]:
        projected = self.in_proj(features)
        batch_size, codebook_features, frame_count = projected.shape
        flattened = projected.transpose(1, 2).reshape(-1, codebook_features)
        flattened = F.normalize(flattened)
        codebook = F.normalize(self.codebook.weight)
        distances = (
            flattened.pow(2).sum(1, keepdim=True)
            - 2 * flattened @ codebook.t()
            + codebook.pow(2).sum(1, keepdim=True).t()
        )
        indices = (-distances).argmax(1).view(batch_size, frame_count)
        embedded = F.embedding(indices, self.codebook.weight).transpose(1, 2)
        quantized = projected + (embedded - projected).detach()
        return self.out_proj(quantized), indices


class ResidualQuantizer(nn.Module):
    def __init__(self, codebook_count: int, codebook_size: int) -> None:
        super().__init__()
        self.quantizers: nn.ModuleList = nn.ModuleList(
            [VectorQuantizer(codebook_size) for _ in range(codebook_count)]
        )

    def forward(self, features: Tensor) -> tuple[Tensor, Tensor]:
        quantized_sum = torch.zeros_like(features)
        residual = features
        indices: list[Tensor] = []
        for quantizer in self.quantizers:
            quantized, codebook_indices = quantizer(residual)
            quantized_sum = quantized_sum + quantized
            residual = residual - quantized
            indices.append(codebook_indices)
        return quantized_sum, torch.stack(indices, dim=1)

    def decode(self, codes: Tensor) -> Tensor:
        decoded = self.quantizers[0].decode(codes[:, 0])
        for index, quantizer in enumerate(self.quantizers[1:], start=1):
            decoded = decoded + quantizer.decode(codes[:, index])
        return decoded


class CodecConvNeXtBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.dwconv: CausalConv1d = CausalConv1d(
            CODEC_CHANNEL_COUNT,
            CODEC_CHANNEL_COUNT,
            kernel_size=7,
            groups=CODEC_CHANNEL_COUNT,
        )
        self.norm: nn.LayerNorm = nn.LayerNorm(CODEC_CHANNEL_COUNT, eps=1e-6)
        self.pwconv1 = nn.Linear(CODEC_CHANNEL_COUNT, 4 * CODEC_CHANNEL_COUNT)
        self.act: nn.GELU = nn.GELU()
        self.pwconv2 = nn.Linear(4 * CODEC_CHANNEL_COUNT, CODEC_CHANNEL_COUNT)
        self.gamma: nn.Parameter = nn.Parameter(1e-6 * torch.ones(CODEC_CHANNEL_COUNT))

    def forward(self, features: Tensor) -> Tensor:
        hidden_states = self.dwconv(features).permute(0, 2, 1)
        hidden_states = self.pwconv2(self.act(self.pwconv1(self.norm(hidden_states))))
        return features + (self.gamma * hidden_states).permute(0, 2, 1)


class DownsampleQuantizer(nn.Module):
    def __init__(self, post_config: CodecTransformerConfig) -> None:
        super().__init__()
        self.semantic_quantizer: ResidualQuantizer = ResidualQuantizer(
            1, SEMANTIC_CODEBOOK_SIZE
        )
        self.quantizer: ResidualQuantizer = ResidualQuantizer(
            RESIDUAL_CODEBOOK_COUNT, RESIDUAL_CODEBOOK_SIZE
        )
        self.downsample: nn.Sequential = nn.Sequential(
            *(
                nn.Sequential(
                    CausalConv1d(
                        CODEC_CHANNEL_COUNT,
                        CODEC_CHANNEL_COUNT,
                        kernel_size=2,
                        stride=2,
                    ),
                    CodecConvNeXtBlock(),
                )
                for _ in range(2)
            )
        )
        self.upsample: nn.Sequential = nn.Sequential(
            *(
                nn.Sequential(
                    CausalConvTranspose1d(
                        CODEC_CHANNEL_COUNT,
                        CODEC_CHANNEL_COUNT,
                        kernel_size=2,
                        stride=2,
                    ),
                    CodecConvNeXtBlock(),
                )
                for _ in range(2)
            )
        )
        self.pre_module: CodecWindowTransformer = CodecWindowTransformer(
            CodecTransformerConfig(
                layer_count=8,
                attention_head_count=16,
                hidden_size=CODEC_CHANNEL_COUNT,
                intermediate_size=3072,
            ),
            window_size=128,
        )
        self.post_module: CodecWindowTransformer = CodecWindowTransformer(
            post_config, window_size=128
        )

    def forward(self, features: Tensor) -> Tensor:
        features = self.pre_module(self.downsample(features))
        semantic_features, semantic_codes = self.semantic_quantizer(features)
        _, residual_codes = self.quantizer(features - semantic_features)
        return torch.cat((semantic_codes, residual_codes), dim=1)

    def decode(self, codes: Tensor) -> Tensor:
        semantic_codes = codes[:, :1].clamp(0, SEMANTIC_CODEBOOK_SIZE - 1)
        residual_codes = codes[:, 1:].clamp(0, RESIDUAL_CODEBOOK_SIZE - 1)
        semantic_features = self.semantic_quantizer.decode(semantic_codes)
        residual_features = self.quantizer.decode(residual_codes)
        return self.upsample(self.post_module(semantic_features + residual_features))
