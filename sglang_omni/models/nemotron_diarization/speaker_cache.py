# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Request-local arrival-order cache for the published eight-speaker checkpoint.

Adapted from NVIDIA-NeMo/Speech sortformer_modules.py at revision
2c1a2f91d64566b5d391b83df42f9ab4cd810adb. Omni changes: synchronous, single
recording, inference only; learned silence embeddings and fixed scoring settings.
"""

import math

import torch


class SpeakerCache:
    def __init__(self, reference, *, cache_size, fifo_size, update_period):
        self.cache = reference.new_empty(1, 0, 512)
        self.fifo = reference.new_empty(1, 0, 512)
        self.cache_preds = reference.new_empty(1, 0, 8)
        self.compressed = False
        self.cache_size = cache_size
        self.fifo_size = fifo_size
        self.update_period = update_period

    def update(self, chunk, predictions, silence_embedding):
        cache_length, fifo_length = self.cache.shape[1], self.fifo.shape[1]
        count = chunk.shape[1]
        fifo = torch.cat([self.fifo, chunk], dim=1)
        fifo_preds = predictions[:, cache_length : cache_length + fifo_length + count]
        if fifo.shape[1] > self.fifo_size:
            pop = min(
                max(self.update_period, fifo.shape[1] - self.fifo_size), fifo.shape[1]
            )
            self.cache = torch.cat([self.cache, fifo[:, :pop]], dim=1)
            cache_preds = (
                self.cache_preds if self.compressed else predictions[:, :cache_length]
            )
            self.cache_preds = torch.cat([cache_preds, fifo_preds[:, :pop]], dim=1)
            fifo = fifo[:, pop:]
            if self.cache.shape[1] > self.cache_size:
                self.compress(silence_embedding)
                self.compressed = True
        self.fifo = fifo

    def compress(self, silence_embedding):
        preds = self.cache_preds
        log_probs = torch.log(torch.clamp(preds, min=0.25))
        log_other = torch.log(torch.clamp(1.0 - preds, min=0.25))
        scores = (
            log_probs - log_other + log_other.sum(dim=2, keepdim=True) - math.log(0.5)
        )
        scores = torch.where(preds > 0.5, scores, float("-inf"))
        per_speaker = self.cache_size // 8 - 1
        positive = scores > 0
        disable = (
            (~positive)
            * (preds > 0.5)
            * (positive.sum(dim=1, keepdim=True) >= math.floor(per_speaker * 0.5))
        )
        scores = torch.where(disable, float("-inf"), scores)
        scores[:, self.cache_size :] += 0.05
        for count, scale in [
            (math.floor(per_speaker * 0.75), 2),
            (math.floor(per_speaker * 1.5), 1),
        ]:
            indices = torch.topk(scores, count, dim=1, sorted=False).indices
            speakers = torch.arange(8, device=scores.device)[None, None, :]
            scores[
                torch.zeros(1, 1, 1, dtype=torch.long, device=scores.device),
                indices,
                speakers,
            ] -= scale * math.log(0.5)
        # One silence frame per speaker, followed by speaker-major ordering.
        scores = torch.cat([scores, scores.new_full((1, 1, 8), float("inf"))], dim=1)
        frame_count = scores.shape[1]
        values, indices = torch.topk(
            scores.permute(0, 2, 1).reshape(1, -1), self.cache_size, sorted=False
        )
        indices = (
            torch.where(values != float("-inf"), indices, 99999).sort(dim=1).values
        )
        disabled = indices == 99999
        indices = indices.remainder(frame_count)
        disabled |= indices >= frame_count - 1
        indices = indices.masked_fill(disabled, 0)
        embeddings = torch.gather(
            self.cache, 1, indices.unsqueeze(-1).expand(-1, -1, 512)
        )
        preds = torch.gather(preds, 1, indices.unsqueeze(-1).expand(-1, -1, 8))
        self.cache = torch.where(
            disabled.unsqueeze(-1), silence_embedding[None, None, :], embeddings
        )
        self.cache_preds = torch.where(disabled.unsqueeze(-1), 0.0, preds)
