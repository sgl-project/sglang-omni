# SPDX-License-Identifier: Apache-2.0
"""Sortformer speaker heads and bounded streaming speaker caches."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass(kw_only=True)
class StreamingState:
    """Per-forward cache. Empty tensors share the activation dtype."""

    speaker_cache: torch.Tensor
    recent_embeddings: torch.Tensor
    recent_predictions: torch.Tensor | None = None
    speaker_cache_predictions: torch.Tensor | None = None
    cache_compressed: bool = False
    silence_embedding: torch.Tensor | None = None
    silence_frames: torch.Tensor | None = None


class SortformerModules(nn.Module):
    """Speaker head, upsampling, and streaming cache. Names match the checkpoint."""

    def __init__(
        self,
        speaker_count: int,
        fc_d_model: int,
        tf_d_model: int,
        dropout_rate: float,
        upsample_factor: int,
        spkcache_len: int,
        fifo_len: int,
        chunk_len: int,
        spkcache_update_period: int,
        spkcache_sil_frames_per_spk: int,
        pred_score_threshold: float,
        max_index: int,
        scores_boost_latest: float,
        sil_threshold: float,
        strong_boost_rate: float,
        weak_boost_rate: float,
        min_pos_scores_rate: float,
        use_learnable_sil_emb: bool,
    ) -> None:
        super().__init__()
        self.speaker_count = speaker_count
        self.fc_d_model = fc_d_model
        self.upsample_factor = upsample_factor
        self.spkcache_len = spkcache_len
        self.fifo_len = fifo_len
        self.chunk_len = chunk_len
        self.spkcache_update_period = spkcache_update_period
        self.spkcache_sil_frames_per_spk = spkcache_sil_frames_per_spk
        self.pred_score_threshold = pred_score_threshold
        self.max_index = max_index
        self.scores_boost_latest = scores_boost_latest
        self.sil_threshold = sil_threshold
        self.strong_boost_rate = strong_boost_rate
        self.weak_boost_rate = weak_boost_rate
        self.min_pos_scores_rate = min_pos_scores_rate
        self.use_learnable_sil_emb = use_learnable_sil_emb
        self.dropout = nn.Dropout(dropout_rate)
        self.hidden_to_spks = nn.Linear(2 * tf_d_model, speaker_count)
        self.hidden_to_spks.requires_grad_(False)
        self.first_hidden_to_hidden = nn.Linear(tf_d_model, tf_d_model)
        self.single_hidden_to_spks = nn.Linear(tf_d_model, speaker_count)
        if fc_d_model != tf_d_model:
            self.encoder_proj = nn.Linear(fc_d_model, tf_d_model)
        else:
            self.encoder_proj = nn.Identity()
        if upsample_factor > 1:
            self.subpixel_upsample = nn.Conv1d(
                tf_d_model, tf_d_model * upsample_factor, kernel_size=3, padding=1
            )
        else:
            self.subpixel_upsample = nn.Identity()
        if use_learnable_sil_emb:
            self.learnable_sil_emb = nn.Parameter(torch.zeros(fc_d_model))
        else:
            self.learnable_sil_emb = None

    def forward_speaker_logits(self, hidden: torch.Tensor) -> torch.Tensor:
        hidden = self.dropout(F.relu(hidden))
        hidden = self.first_hidden_to_hidden(hidden)
        hidden = self.dropout(F.relu(hidden))
        return self.single_hidden_to_spks(hidden)

    def upsample_hidden(self, hidden: torch.Tensor) -> torch.Tensor:
        if self.upsample_factor == 1:
            return hidden
        else:
            pass
        batch, frames, width = hidden.shape
        projected = self.subpixel_upsample(hidden.transpose(1, 2)).transpose(1, 2)
        return projected.reshape(batch, frames, self.upsample_factor, width).reshape(
            batch, frames * self.upsample_factor, width
        )

    @staticmethod
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

    def initial_state(
        self, batch: int, device: torch.device, dtype: torch.dtype
    ) -> StreamingState:
        return StreamingState(
            speaker_cache=torch.zeros(
                (batch, 0, self.fc_d_model), device=device, dtype=dtype
            ),
            recent_embeddings=torch.zeros(
                (batch, 0, self.fc_d_model), device=device, dtype=dtype
            ),
            silence_embedding=torch.zeros(
                (batch, self.fc_d_model), device=device, dtype=dtype
            ),
            silence_frames=torch.zeros((batch,), dtype=torch.long, device=device),
        )

    def update_cache(
        self,
        state: StreamingState,
        chunk: torch.Tensor,
        predictions: torch.Tensor,
        left_frames: int,
        right_frames: int,
    ) -> StreamingState:
        """Fold one pre-encoded chunk into the cache."""
        cache_frames = state.speaker_cache.shape[1]
        queue_frames = state.recent_embeddings.shape[1]
        core_frames = chunk.shape[1] - left_frames - right_frames
        if core_frames < 0:
            raise ValueError("streaming chunk is shorter than its context")
        else:
            pass
        state.recent_predictions = predictions[
            :, cache_frames : cache_frames + queue_frames
        ]
        core = chunk[:, left_frames : left_frames + core_frames]
        core_predictions = predictions[
            :,
            cache_frames
            + queue_frames
            + left_frames : cache_frames
            + queue_frames
            + left_frames
            + core_frames,
        ]
        state.recent_embeddings = torch.cat([state.recent_embeddings, core], dim=1)
        state.recent_predictions = torch.cat(
            [state.recent_predictions, core_predictions], dim=1
        )
        queued = queue_frames + core_frames
        if queued > self.fifo_len:
            pop = self.spkcache_update_period
            pop = max(pop, core_frames - self.fifo_len + queue_frames)
            pop = min(pop, queued)
            popped_embeddings = state.recent_embeddings[:, :pop]
            popped_predictions = state.recent_predictions[:, :pop]
            if not self.use_learnable_sil_emb:
                state.silence_embedding, state.silence_frames = self.silence_profile(
                    state.silence_embedding,
                    state.silence_frames,
                    popped_embeddings,
                    popped_predictions,
                )
            else:
                pass
            state.recent_embeddings = state.recent_embeddings[:, pop:]
            state.recent_predictions = state.recent_predictions[:, pop:]
            state.speaker_cache = torch.cat(
                [state.speaker_cache, popped_embeddings], dim=1
            )
            if state.cache_compressed and state.speaker_cache_predictions is not None:
                state.speaker_cache_predictions = torch.cat(
                    [state.speaker_cache_predictions, popped_predictions], dim=1
                )
            else:
                state.speaker_cache_predictions = torch.cat(
                    [predictions[:, :cache_frames], popped_predictions], dim=1
                )
            if state.speaker_cache.shape[1] > self.spkcache_len:
                compressed, compressed_predictions = self.compress_cache(
                    state.speaker_cache,
                    state.speaker_cache_predictions,
                    state.silence_embedding,
                )
                state.speaker_cache = compressed
                state.speaker_cache_predictions = compressed_predictions
                state.cache_compressed = True
            else:
                pass
        else:
            pass
        return state

    def silence_profile(
        self,
        mean_silence: torch.Tensor | None,
        silence_frames: torch.Tensor | None,
        embeddings: torch.Tensor,
        predictions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if mean_silence is None or silence_frames is None:
            raise RuntimeError("streaming state is missing its silence profile")
        else:
            pass
        silent = predictions.sum(dim=2) < self.sil_threshold
        count = silent.sum(dim=1)
        if not bool(count.any().item()):
            return mean_silence, silence_frames
        else:
            pass
        total_frames = silence_frames + count
        total = mean_silence * silence_frames.unsqueeze(1) + (
            embeddings * silent.unsqueeze(-1)
        ).sum(dim=1)
        return total / total_frames.clamp(min=1).unsqueeze(1), total_frames

    def compress_cache(
        self,
        embeddings: torch.Tensor,
        predictions: torch.Tensor,
        mean_silence: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Keep spkcache_len frames, ordered by speaker, plus learned silence slots."""
        batch, _, speakers = predictions.shape
        silence_parameter = self.learnable_sil_emb
        if self.use_learnable_sil_emb and silence_parameter is not None:
            silence = silence_parameter.to(
                dtype=embeddings.dtype, device=embeddings.device
            )
            silence = silence.unsqueeze(0).expand(batch, -1)
        elif mean_silence is not None:
            silence = mean_silence
        else:
            raise RuntimeError("speaker-cache compression has no silence embedding")
        per_speaker = self.spkcache_len // speakers - self.spkcache_sil_frames_per_spk
        scores = self.frame_scores(predictions)
        scores = self.drop_low_scores(
            predictions, scores, math.floor(per_speaker * self.min_pos_scores_rate)
        )
        if self.scores_boost_latest > 0:
            scores = scores.clone()
            scores[:, self.spkcache_len :, :] += self.scores_boost_latest
        else:
            pass
        scores = self.boost_top(
            scores, math.floor(per_speaker * self.strong_boost_rate), 2.0
        )
        scores = self.boost_top(
            scores, math.floor(per_speaker * self.weak_boost_rate), 1.0
        )
        if self.spkcache_sil_frames_per_spk > 0:
            pad = torch.full(
                (batch, self.spkcache_sil_frames_per_spk, speakers),
                float("inf"),
                device=scores.device,
                dtype=scores.dtype,
            )
            scores = torch.cat([scores, pad], dim=1)
        else:
            pass
        indices, disabled = self.top_indices(scores)
        return self.gather_cache(embeddings, predictions, indices, disabled, silence)

    def frame_scores(self, predictions: torch.Tensor) -> torch.Tensor:
        threshold = self.pred_score_threshold
        log_probs = torch.log(predictions.clamp(min=threshold))
        log_complement = torch.log((1.0 - predictions).clamp(min=threshold))
        complement_sum = (
            log_complement.sum(dim=2).unsqueeze(-1).expand(-1, -1, self.speaker_count)
        )
        return log_probs - log_complement + complement_sum - math.log(0.5)

    def drop_low_scores(
        self, predictions: torch.Tensor, scores: torch.Tensor, minimum_positive: int
    ) -> torch.Tensor:
        speech = predictions > 0.5
        scores = torch.where(
            speech,
            scores,
            torch.tensor(float("-inf"), device=scores.device, dtype=scores.dtype),
        )
        positive = scores > 0
        drop = (
            (~positive)
            & speech
            & (positive.sum(dim=1).unsqueeze(1) >= minimum_positive)
        )
        filler = torch.tensor(float("-inf"), device=scores.device, dtype=scores.dtype)
        return torch.where(drop, filler, scores)

    def boost_top(self, scores: torch.Tensor, count: int, scale: float) -> torch.Tensor:
        if count <= 0:
            return scores
        else:
            pass
        batch, _, speakers = scores.shape
        _, indices = torch.topk(scores, count, dim=1, largest=True, sorted=False)
        batch_index = torch.arange(batch, device=scores.device).view(batch, 1, 1)
        speaker_index = torch.arange(speakers, device=scores.device).view(
            1, 1, speakers
        )
        boosted = scores.clone()
        boosted[batch_index, indices, speaker_index] -= scale * math.log(0.5)
        return boosted

    def top_indices(self, scores: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch, frames, _ = scores.shape
        speech_frames = frames - self.spkcache_sil_frames_per_spk
        flat = scores.permute(0, 2, 1).reshape(batch, -1)
        values, indices = torch.topk(flat, self.spkcache_len, dim=1, sorted=False)
        placeholder = torch.tensor(self.max_index, device=scores.device)
        indices = torch.where(values != float("-inf"), indices, placeholder)
        indices, _ = torch.sort(indices, dim=1)
        disabled = indices == self.max_index
        indices = torch.remainder(indices, frames)
        disabled = disabled | (indices >= speech_frames)
        indices = indices.clone()
        indices[disabled] = 0
        return indices, disabled

    def gather_cache(
        self,
        embeddings: torch.Tensor,
        predictions: torch.Tensor,
        indices: torch.Tensor,
        disabled: torch.Tensor,
        silence: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        width = embeddings.shape[2]
        speakers = predictions.shape[2]
        gathered = torch.gather(
            embeddings, 1, indices.unsqueeze(-1).expand(-1, -1, width)
        )
        silence = silence.unsqueeze(1).expand(-1, self.spkcache_len, -1)
        gathered = torch.where(disabled.unsqueeze(-1), silence, gathered)
        gathered_predictions = torch.gather(
            predictions, 1, indices.unsqueeze(-1).expand(-1, -1, speakers)
        )
        zeros = torch.zeros((), device=predictions.device, dtype=predictions.dtype)
        gathered_predictions = torch.where(
            disabled.unsqueeze(-1), zeros, gathered_predictions
        )
        return gathered, gathered_predictions
