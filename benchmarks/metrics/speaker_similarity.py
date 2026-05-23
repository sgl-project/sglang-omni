# SPDX-License-Identifier: Apache-2.0
"""WavLM-large speaker-verification similarity for SeedTTS SIM."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.signal import resample_poly

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000

# Keys present in the published popsoda2002/seedtts-wavlm-sim
# wavlm_large_finetune.pth that are training-time only (AAM-softmax /
# ArcFace head used during fine-tuning) and not part of the inference
# scorer. Allowlisted so loading does not have to be silently lenient.
_HARMLESS_UNEXPECTED_KEYS = frozenset(
    {
        "loss_calculator.projection.weight",
    }
)


class Res2Conv1dReluBn(nn.Module):
    def __init__(
        self,
        channels: int,
        kernel_size: int = 1,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        bias: bool = True,
        scale: int = 4,
    ):
        super().__init__()
        self.scale = scale
        self.width = channels // scale
        self.nums = scale if scale == 1 else scale - 1
        self.convs = nn.ModuleList(
            [
                nn.Conv1d(
                    self.width,
                    self.width,
                    kernel_size,
                    stride,
                    padding,
                    dilation,
                    bias=bias,
                )
                for _ in range(self.nums)
            ]
        )
        self.bns = nn.ModuleList([nn.BatchNorm1d(self.width) for _ in range(self.nums)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = []
        spx = torch.split(x, self.width, 1)
        sp = None
        for i in range(self.nums):
            sp = spx[i] if i == 0 else sp + spx[i]
            sp = self.bns[i](F.relu(self.convs[i](sp)))
            out.append(sp)
        if self.scale != 1:
            out.append(spx[self.nums])
        return torch.cat(out, dim=1)


class Conv1dReluBn(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 1,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        bias: bool = True,
    ):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            bias=bias,
        )
        self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.bn(F.relu(self.conv(x)))


class SEConnect(nn.Module):
    def __init__(self, channels: int, se_bottleneck_dim: int = 128):
        super().__init__()
        self.linear1 = nn.Linear(channels, se_bottleneck_dim)
        self.linear2 = nn.Linear(se_bottleneck_dim, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x.mean(dim=2)
        out = F.relu(self.linear1(out))
        out = torch.sigmoid(self.linear2(out))
        return x * out.unsqueeze(2)


class SERes2Block(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int,
        dilation: int,
        scale: int,
        se_bottleneck_dim: int,
    ):
        super().__init__()
        self.Conv1dReluBn1 = Conv1dReluBn(in_channels, out_channels, kernel_size=1)
        self.Res2Conv1dReluBn = Res2Conv1dReluBn(
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            scale=scale,
        )
        self.Conv1dReluBn2 = Conv1dReluBn(out_channels, out_channels, kernel_size=1)
        self.SE_Connect = SEConnect(out_channels, se_bottleneck_dim)
        self.shortcut = (
            nn.Conv1d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x) if self.shortcut else x
        x = self.Conv1dReluBn1(x)
        x = self.Res2Conv1dReluBn(x)
        x = self.Conv1dReluBn2(x)
        x = self.SE_Connect(x)
        return x + residual


class AttentiveStatsPool(nn.Module):
    def __init__(self, in_dim: int, attention_channels: int = 128):
        super().__init__()
        self.linear1 = nn.Conv1d(in_dim, attention_channels, kernel_size=1)
        self.linear2 = nn.Conv1d(attention_channels, in_dim, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        alpha = torch.tanh(self.linear1(x))
        alpha = torch.softmax(self.linear2(alpha), dim=2)
        mean = torch.sum(alpha * x, dim=2)
        residuals = torch.sum(alpha * (x**2), dim=2) - mean**2
        std = torch.sqrt(residuals.clamp(min=1e-9))
        return torch.cat([mean, std], dim=1)


class ECAPATDNNWavLM(nn.Module):
    def __init__(
        self,
        *,
        feature_extract: nn.Module,
        feat_dim: int = 1024,
        channels: int = 512,
        emb_dim: int = 256,
    ):
        super().__init__()
        self.feature_extract = feature_extract
        for idx in (23, 11):
            self.feature_extract.model.encoder.layers[idx].self_attn.fp32_attention = (
                False
            )

        for param in self.feature_extract.parameters():
            param.requires_grad = False

        self.feature_selection = "hidden_states"
        self.feat_num = self.get_feat_num()
        self.feature_weight = nn.Parameter(torch.zeros(self.feat_num))
        self.instance_norm = nn.InstanceNorm1d(feat_dim)
        self.layer1 = Conv1dReluBn(feat_dim, channels, kernel_size=5, padding=2)
        self.layer2 = SERes2Block(
            channels, channels, 3, 1, 2, 2, scale=8, se_bottleneck_dim=128
        )
        self.layer3 = SERes2Block(
            channels, channels, 3, 1, 3, 3, scale=8, se_bottleneck_dim=128
        )
        self.layer4 = SERes2Block(
            channels, channels, 3, 1, 4, 4, scale=8, se_bottleneck_dim=128
        )
        self.conv = nn.Conv1d(channels * 3, 1536, kernel_size=1)
        self.pooling = AttentiveStatsPool(1536)
        self.bn = nn.BatchNorm1d(3072)
        self.linear = nn.Linear(3072, emb_dim)

    def get_feat_num(self) -> int:
        self.feature_extract.eval()
        wav = [
            torch.randn(SAMPLE_RATE).to(next(self.feature_extract.parameters()).device)
        ]
        with torch.no_grad():
            features = self.feature_extract(wav)[self.feature_selection]
        return len(features)

    def get_feat(self, x: list[torch.Tensor]) -> list[torch.Tensor]:
        wav_lengths = [wav.size(0) for wav in x]
        with torch.no_grad():
            x = self.feature_extract(x)[self.feature_selection]
        x = torch.stack(x, dim=0)
        weights = F.softmax(self.feature_weight, dim=-1)
        weights = weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        x = (weights * x).sum(dim=0)
        feat_lengths = self.get_feat_lengths(wav_lengths, x.device)
        return [
            self.instance_norm(torch.transpose(x[i : i + 1, :length], 1, 2) + 1e-6)
            for i, length in enumerate(feat_lengths)
        ]

    def get_feat_lengths(
        self,
        wav_lengths: list[int],
        device: torch.device,
    ) -> list[int]:
        lengths = torch.tensor(wav_lengths, device=device)
        for layer in self.feature_extract.model.feature_extractor.conv_layers:
            conv = layer[0]
            lengths = (
                torch.div(
                    lengths - conv.kernel_size[0],
                    conv.stride[0],
                    rounding_mode="floor",
                )
                + 1
            )
        return lengths.tolist()

    def encode_feat(self, x: torch.Tensor) -> torch.Tensor:
        out1 = self.layer1(x)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)
        out = torch.cat([out2, out3, out4], dim=1)
        out = F.relu(self.conv(out))
        out = self.bn(self.pooling(out))
        return self.linear(out)

    def forward(self, x: list[torch.Tensor]) -> torch.Tensor:
        return torch.cat([self.encode_feat(feat) for feat in self.get_feat(x)], dim=0)


class WavLMSpeakerSimilarity:
    def __init__(
        self,
        *,
        finetune_checkpoint: str | Path,
        wavlm_base: str | Path,
        device: str,
    ):
        finetune_checkpoint = str(finetune_checkpoint)
        wavlm_base = str(wavlm_base)
        self.model = ECAPATDNNWavLM(feature_extract=load_wavlm(wavlm_base))
        state_dict = torch.load(finetune_checkpoint, map_location="cpu")
        result = self.model.load_state_dict(state_dict["model"], strict=False)

        if result.missing_keys:
            raise RuntimeError(
                "WavLM SV checkpoint is missing required weights — refusing to "
                "score with a partially initialized model.\n"
                f"  finetune_checkpoint: {finetune_checkpoint}\n"
                f"  wavlm_base: {wavlm_base}\n"
                f"  missing_keys ({len(result.missing_keys)}): "
                f"{sorted(result.missing_keys)}"
            )

        unexpected = set(result.unexpected_keys) - _HARMLESS_UNEXPECTED_KEYS
        if unexpected:
            raise RuntimeError(
                "WavLM SV checkpoint has unrecognized weights not on the "
                "harmless-training-key allowlist — refusing to score with an "
                "unverified checkpoint structure.\n"
                f"  finetune_checkpoint: {finetune_checkpoint}\n"
                f"  wavlm_base: {wavlm_base}\n"
                f"  unexpected_keys (not allowlisted): {sorted(unexpected)}\n"
                f"  allowlist: {sorted(_HARMLESS_UNEXPECTED_KEYS)}"
            )

        if result.unexpected_keys:
            logger.info(
                "WavLM SV checkpoint has %d allowlisted unexpected keys "
                "(training-only artifacts): %s",
                len(result.unexpected_keys),
                sorted(result.unexpected_keys),
            )

        self.model.to(device)
        self.model.eval()
        self.device = device

    def embed(self, audio_paths: list[str]) -> torch.Tensor:
        audio = [load_audio(audio_path).to(self.device) for audio_path in audio_paths]
        with torch.no_grad():
            return self.model(audio)

    def score_batch(
        self,
        ref_audio_paths: list[str],
        wav_paths: list[str],
    ) -> list[float]:
        batch_size = len(ref_audio_paths)
        embeddings = self.embed(ref_audio_paths + wav_paths)
        scores = F.cosine_similarity(
            embeddings[:batch_size],
            embeddings[batch_size:],
            dim=-1,
        )
        return (scores.cpu() * 100.0).tolist()


def load_audio(audio_path: str) -> torch.Tensor:
    audio, sample_rate = sf.read(audio_path, dtype="float32")
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    if sample_rate != SAMPLE_RATE:
        gcd = np.gcd(sample_rate, SAMPLE_RATE)
        audio = resample_poly(audio, SAMPLE_RATE // gcd, sample_rate // gcd)
    return torch.from_numpy(np.asarray(audio, dtype=np.float32))


def load_wavlm(wavlm_base: str | Path):
    """Instantiate the WavLM upstream from the pip-installed s3prl package."""
    from s3prl.upstream.wavlm.hubconf import wavlm_local

    return wavlm_local(ckpt=str(wavlm_base))
