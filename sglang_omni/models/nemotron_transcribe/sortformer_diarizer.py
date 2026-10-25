# SPDX-License-Identifier: Apache-2.0
"""Windowed Sortformer inference on the ASR frame grid."""

import math

import torch
from torch import nn

from sglang_omni.models.nemotron_transcribe.sortformer import (
    SortformerModules,
    StreamingState,
)
from sglang_omni.models.nemotron_transcribe.speech_transformer import (
    SpeechTransformerEncoder,
)


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

    def initial_state(self, batch: int) -> StreamingState:
        parameter = next(self.encoder.parameters())
        return self.sortformer_modules.initial_state(
            batch, parameter.device, parameter.dtype
        )

    def streaming_step(
        self,
        state: StreamingState,
        total_predictions: torch.Tensor,
        mel_chunk: torch.Tensor,
        mel_lengths: torch.Tensor,
        left_offset: int,
        right_offset: int,
    ) -> tuple[StreamingState, torch.Tensor]:
        """mel_chunk is (batch, bins, time). Returns state and predictions concatenated on time."""
        modules = self.sortformer_modules
        pre_encoded, pre_lengths = self.encoder.pre_encode(mel_chunk, mel_lengths)
        left_frames = round(left_offset / self.encoder.subsampling_factor)
        right_frames = math.ceil(right_offset / self.encoder.subsampling_factor)
        combined = torch.cat(
            [state.speaker_cache, state.recent_embeddings, pre_encoded], dim=1
        )
        combined_lengths = (
            state.speaker_cache.shape[1]
            + state.recent_embeddings.shape[1]
            + pre_lengths
        )
        encoded, encoded_lengths = self.encoder(
            combined, combined_lengths, bypass_pre_encode=True
        )
        hidden = encoded.transpose(1, 2)
        hidden = modules.encoder_proj(hidden)
        hidden = modules.upsample_hidden(hidden)
        time_mask = torch.arange(encoded.shape[-1], device=hidden.device).unsqueeze(
            0
        ) < encoded_lengths.unsqueeze(1)
        time_mask = time_mask.repeat_interleave(modules.upsample_factor, dim=1)
        logits = modules.forward_speaker_logits(hidden)
        high_predictions = torch.sigmoid(logits) * time_mask.unsqueeze(-1).to(
            logits.dtype
        )
        state_predictions = modules.downsample_preds(
            high_predictions, modules.upsample_factor
        ).detach()
        state_predictions = self.mask_predictions(state_predictions, encoded_lengths)
        saved_cache = state.speaker_cache.shape[1]
        saved_recent = state.recent_embeddings.shape[1]
        state = modules.update_cache(
            state, pre_encoded, state_predictions, left_frames, right_frames
        )
        core = pre_encoded.shape[1] - left_frames - right_frames
        start = (saved_cache + saved_recent + left_frames) * modules.upsample_factor
        chunk_predictions = high_predictions[
            :, start : start + core * modules.upsample_factor
        ]
        if self.downsample_factor > 1:
            chunk_predictions = modules.downsample_preds(
                chunk_predictions, self.downsample_factor
            )
        else:
            pass
        return state, torch.cat([total_predictions, chunk_predictions], dim=1)

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
