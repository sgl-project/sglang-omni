# SPDX-License-Identifier: Apache-2.0
"""Fuse windowed ASR and diarization encoder states."""

from __future__ import annotations

import math

import torch
from torch import nn

from sglang_omni.models.nemotron_diarization.speaker_cache import SpeakerCache
from sglang_omni.models.nemotron_transcribe.mel_features import normalize_features
from sglang_omni.models.nemotron_transcribe.sortformer_diarizer import (
    SortformerDiarizer,
)
from sglang_omni.models.nemotron_transcribe.speech_transformer import (
    SpeechTransformerEncoder,
)


def speaker_kernel(speaker_count: int, width: int) -> torch.Tensor:
    """Sinusoidal speaker table shaped (speakers, width)."""
    positions = torch.arange(speaker_count, dtype=torch.float32).unsqueeze(1)
    divider = torch.exp(
        torch.arange(0, width, 2, dtype=torch.float32) * -(math.log(10000.0) / width)
    )
    table = torch.zeros(speaker_count, width, dtype=torch.float32)
    table[:, 0::2] = torch.sin(positions * divider)
    table[:, 1::2] = torch.cos(positions * divider)
    return table


def match_module(values: torch.Tensor, module: nn.Module) -> torch.Tensor:
    parameter = next(module.parameters())
    return values.to(device=parameter.device, dtype=parameter.dtype)


def align_time(activity: torch.Tensor, frames: int) -> torch.Tensor:
    """Pad by repeating the last frame, or trim, so time matches the ASR chunk."""
    current = activity.shape[1]
    if current == frames:
        return activity
    elif current == 0:
        raise ValueError("speaker activity is empty")
    elif current < frames:
        return torch.cat(
            [activity, activity[:, -1:, :].repeat(1, frames - current, 1)], dim=1
        )
    else:
        return activity[:, :frames]


class ParallelExpertEncoder(nn.Module):
    """Fused ASR and diarization encoder. Output is (batch, width, time)."""

    def __init__(
        self,
        asr_encoder: SpeechTransformerEncoder,
        diarizer: SortformerDiarizer,
        asr_normalize: str | None,
        diar_normalize: str | None,
        speaker_activity_threshold: float,
        speaker_kernel_scale: float,
        online_frames: int,
        left_context_frames: int,
        right_context_frames: int,
        cache_limit: int,
        recent_limit: int,
        cache_update_frames: int,
    ) -> None:
        super().__init__()
        if asr_encoder.subsampling_factor != diarizer.encoder.subsampling_factor:
            raise ValueError("ASR and diarization subsampling factors differ")
        else:
            pass
        self.asr_encoder = asr_encoder
        self.asr_norm = nn.LayerNorm(asr_encoder.d_model)
        self.diar_norm = nn.LayerNorm(diarizer.sortformer_modules.speakers)
        self.diarization_model = diarizer
        self.asr_normalize = asr_normalize
        self.diar_normalize = diar_normalize
        self.speaker_activity_threshold = speaker_activity_threshold
        self.speaker_kernel_scale = speaker_kernel_scale
        self.online_frames = online_frames
        self.left_context_frames = left_context_frames
        self.right_context_frames = right_context_frames
        self.cache_limit = cache_limit
        self.recent_limit = recent_limit
        self.cache_update_frames = cache_update_frames
        self.register_buffer(
            "diar_kernel",
            speaker_kernel(diarizer.sortformer_modules.speakers, asr_encoder.d_model),
            persistent=False,
        )

    @property
    def subsampling_factor(self) -> int:
        return self.asr_encoder.subsampling_factor

    def forward(
        self, mel: torch.Tensor, mel_lengths: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """mel is (batch, bins, time). Returns fused states and encoder frame counts."""
        mel_lengths = mel_lengths.to(device=mel.device, dtype=torch.long)
        asr_mel = match_module(
            normalize_features(mel, mel_lengths, self.asr_normalize), self.asr_encoder
        )
        diar_mel = match_module(
            normalize_features(mel, mel_lengths, self.diar_normalize),
            self.diarization_model.encoder,
        )
        speakers = self.diarization_model.sortformer_modules.speakers
        batch = mel.shape[0]
        state = SpeakerCache(
            diar_mel,
            cache_size=self.cache_limit,
            fifo_size=self.recent_limit,
            update_period=self.cache_update_frames,
            batch=batch,
            width=self.diarization_model.encoder.d_model,
            speakers=speakers,
        )
        total = torch.zeros(
            (batch, 0, speakers),
            device=diar_mel.device,
            dtype=diar_mel.dtype,
        )
        limit = min(mel.shape[-1], int(mel_lengths.max().item()))
        chunk_frames = self.online_frames * self.subsampling_factor
        left_frames = self.left_context_frames * self.subsampling_factor
        right_frames = self.right_context_frames * self.subsampling_factor
        chunk_count = max(1, math.ceil(limit / chunk_frames)) if chunk_frames else 1
        asr_parts: list[torch.Tensor] = []
        diar_parts: list[torch.Tensor] = []
        for index in range(chunk_count):
            start = index * chunk_frames
            end = min(start + chunk_frames, limit)
            context_start = max(start - left_frames, 0)
            context_end = min(end + right_frames, limit)
            left_offset = start - context_start
            right_offset = context_end - end
            asr_lengths = (mel_lengths - context_start).clamp(
                min=0, max=context_end - context_start
            )
            encoded, _ = self.asr_encoder(
                asr_mel[:, :, context_start:context_end], asr_lengths
            )
            left_drop = left_offset // self.subsampling_factor
            core = self.output_frames(end) - self.output_frames(start)
            core = max(0, min(core, encoded.shape[-1] - left_drop))
            asr_parts.append(encoded[:, :, left_drop : left_drop + core])
            previous = total.shape[1]
            total = self.diarization_model.streaming_step(
                state,
                total,
                diar_mel[:, :, context_start:context_end],
                asr_lengths.to(diar_mel.device),
                left_offset,
                right_offset,
            )
            diar_parts.append(align_time(total[:, previous:], core))
        fused = self.fuse(torch.cat(asr_parts, dim=2), torch.cat(diar_parts, dim=1))
        return fused, self.encoded_lengths(mel_lengths, mel.shape[-1])

    def output_frames(self, mel_frames: int) -> int:
        return (mel_frames + self.subsampling_factor - 1) // self.subsampling_factor

    def encoded_lengths(
        self, mel_lengths: torch.Tensor, mel_frames: int
    ) -> torch.Tensor:
        valid = mel_lengths.clamp(min=0, max=mel_frames)
        return torch.div(
            valid + self.subsampling_factor - 1,
            self.subsampling_factor,
            rounding_mode="floor",
        )

    def fuse(self, asr_encoded: torch.Tensor, activity: torch.Tensor) -> torch.Tensor:
        """asr_encoded is (batch, width, time). activity is (batch, time, speakers)."""
        states = asr_encoded.transpose(1, 2)
        activity = align_time(activity, states.shape[1]).to(
            device=states.device, dtype=states.dtype
        )
        speaker_features = (activity > self.speaker_activity_threshold).to(states.dtype)
        infusion = self.diar_norm(speaker_features) @ self.diar_kernel.to(
            dtype=states.dtype
        )
        fused = self.asr_norm(states) + self.speaker_kernel_scale * infusion
        return fused.transpose(1, 2).contiguous()
