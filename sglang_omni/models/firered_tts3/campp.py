# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CAM++ speaker encoder for FireRedTTS3-Base, adapted from 3D-Speaker."""

from collections import OrderedDict

import torch
import torchaudio
import torchaudio.compliance.kaldi as kaldi
from torch import nn

from sglang_omni.models.firered_tts3.campp_layers import (
    FCM,
    BasicResBlock,
    CAMDenseTDNNBlock,
    CAMDenseTDNNLayer,
    CAMLayer,
    DenseLayer,
    StatsPool,
    TDNNLayer,
    TransitLayer,
    get_nonlinear,
    statistics_pooling,
)


def extract_kaldi_mel(audio: torch.Tensor, audio_sr: int) -> torch.Tensor:
    audio_sample_rate = audio_sr
    audio = audio[:1]
    if audio_sample_rate != 16000:
        audio = torchaudio.functional.resample(audio, audio_sample_rate, 16000)
    else:
        pass
    mel_features = kaldi.fbank(
        audio,
        num_mel_bins=80,
        dither=0,
        sample_frequency=16000,
    )
    mel_features = mel_features - mel_features.mean(dim=0, keepdim=True)
    return mel_features


class CAMPPlus(nn.Module):
    def __init__(
        self,
        feat_dim: int = 80,
        embedding_size: int = 512,
        growth_rate: int = 32,
        bn_size: int = 4,
        init_channels: int = 128,
        config_str: str = "batchnorm-relu",
        memory_efficient: bool = True,
    ) -> None:
        super().__init__()

        self.head: FCM = FCM(feat_dim=feat_dim)
        channels = self.head.out_channels

        self.xvector: nn.Sequential = nn.Sequential(
            OrderedDict(
                [
                    (
                        "tdnn",
                        TDNNLayer(
                            channels,
                            init_channels,
                            5,
                            stride=2,
                            dilation=1,
                            padding=-1,
                            config_str=config_str,
                        ),
                    ),
                ]
            )
        )
        channels = init_channels
        for i, (num_layers, kernel_size, dilation) in enumerate(
            zip((12, 24, 16), (3, 3, 3), (1, 2, 2))
        ):
            block = CAMDenseTDNNBlock(
                num_layers=num_layers,
                in_channels=channels,
                out_channels=growth_rate,
                bn_channels=bn_size * growth_rate,
                kernel_size=kernel_size,
                dilation=dilation,
                config_str=config_str,
                memory_efficient=memory_efficient,
            )
            self.xvector.add_module(f"block{i + 1}", block)
            channels = channels + num_layers * growth_rate
            self.xvector.add_module(
                f"transit{i + 1}",
                TransitLayer(
                    channels,
                    channels // 2,
                    bias=False,
                    config_str=config_str,
                ),
            )
            channels //= 2

        self.xvector.add_module("out_nonlinear", get_nonlinear(config_str, channels))

        self.xvector.add_module("stats", StatsPool())
        self.xvector.add_module(
            "dense", DenseLayer(channels * 2, embedding_size, config_str="batchnorm_")
        )

        for module in self.modules():
            if isinstance(module, (nn.Conv1d, nn.Linear)):
                nn.init.kaiming_normal_(module.weight.data)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
                else:
                    pass
            else:
                pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1)
        x = self.head(x)
        x = self.xvector(x)
        return x


class CamppEmbedding(torch.nn.Module):
    """Extract a CAM++ speaker embedding from audio."""

    def __init__(self, feat_dim: int = 80, embedding_size: int = 512) -> None:
        super().__init__()
        self.model: CAMPPlus = CAMPPlus(
            feat_dim=feat_dim, embedding_size=embedding_size
        )
        self.model.eval()

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def forward(self, audio: torch.Tensor, audio_sr: int) -> torch.Tensor:
        mel_features = extract_kaldi_mel(audio, audio_sr)
        mel_features = mel_features.unsqueeze(0)
        with torch.no_grad():
            embedding = self.model.forward(mel_features.to(self.device))
        return embedding


__all__ = [
    "BasicResBlock",
    "CAMDenseTDNNBlock",
    "CAMDenseTDNNLayer",
    "CAMLayer",
    "CAMPPlus",
    "CamppEmbedding",
    "DenseLayer",
    "FCM",
    "StatsPool",
    "TDNNLayer",
    "TransitLayer",
    "extract_kaldi_mel",
    "get_nonlinear",
    "statistics_pooling",
]
