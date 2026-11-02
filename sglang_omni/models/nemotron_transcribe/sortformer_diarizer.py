# SPDX-License-Identifier: Apache-2.0
"""Windowed Sortformer inference on the ASR frame grid."""

import math

import torch
import torch.nn.functional as F
from torch import nn

from sglang_omni.models.nemotron_diarization.model import SortformerModules
from sglang_omni.models.nemotron_diarization.speaker_cache import SpeakerCache
from sglang_omni.models.nemotron_transcribe.speech_transformer import (
    SpeechTransformerEncoder,
)


def downsample_preds(predictions: torch.Tensor, factor: int) -> torch.Tensor:
    """Average non-overlapping speaker-probability windows. predictions is (batch, time, speakers)."""
    if factor == 1 or predictions.shape[1] == 0:
        return predictions
    else:
        pass
    pooled = F.avg_pool1d(
        predictions.transpose(1, 2),
        kernel_size=factor,
        stride=factor,
        ceil_mode=True,
        count_include_pad=False,
    )
    return pooled.transpose(1, 2)


class SortformerDiarizer(nn.Module):
    """Mel chunk to speaker probabilities on the ASR frame grid."""

    def __init__(
        self,
        encoder: SpeechTransformerEncoder,
        modules: SortformerModules,
        output_subsampling_factor: int,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.sortformer_modules = modules
        self.downsample_factor = output_subsampling_factor

    def streaming_step(
        self,
        state: SpeakerCache,
        total_predictions: torch.Tensor,
        mel_chunk: torch.Tensor,
        mel_lengths: torch.Tensor,
        left_offset: int,
        right_offset: int,
    ) -> torch.Tensor:
        """mel_chunk is (batch, bins, time). Returns predictions concatenated on time."""
        modules = self.sortformer_modules
        pre_encoded, pre_lengths = self.encoder.pre_encode(mel_chunk, mel_lengths)
        left_frames = round(left_offset / self.encoder.subsampling_factor)
        right_frames = math.ceil(right_offset / self.encoder.subsampling_factor)
        core = pre_encoded.shape[1] - left_frames - right_frames
        if core < 0:
            raise ValueError("streaming chunk is shorter than its context")
        else:
            pass
        prefix = state.cache.shape[1] + state.fifo.shape[1]
        combined = torch.cat([state.cache, state.fifo, pre_encoded], dim=1)
        encoded, encoded_lengths = self.encoder(
            combined, prefix + pre_lengths, bypass_pre_encode=True
        )
        high_predictions = modules(
            encoded.transpose(1, 2), encoded_lengths.unsqueeze(1)
        )
        state_predictions = self.mask_predictions(
            downsample_preds(high_predictions, modules.upsample).detach(),
            encoded_lengths,
        )
        state.update(
            pre_encoded[:, left_frames : left_frames + core],
            torch.cat(
                [
                    state_predictions[:, :prefix],
                    state_predictions[:, prefix + left_frames :],
                ],
                dim=1,
            ),
            modules.learnable_sil_emb,
        )
        start = (prefix + left_frames) * modules.upsample
        chunk_predictions = high_predictions[:, start : start + core * modules.upsample]
        chunk_predictions = downsample_preds(chunk_predictions, self.downsample_factor)
        return torch.cat([total_predictions, chunk_predictions], dim=1)

    @staticmethod
    def mask_predictions(
        predictions: torch.Tensor, lengths: torch.Tensor
    ) -> torch.Tensor:
        batch, frames, speakers = predictions.shape
        positions = torch.arange(frames, device=predictions.device).view(1, -1, 1)
        keep = positions.expand(batch, -1, speakers) < lengths.view(-1, 1, 1)
        return torch.where(
            keep,
            predictions,
            torch.zeros((), device=predictions.device, dtype=predictions.dtype),
        )
