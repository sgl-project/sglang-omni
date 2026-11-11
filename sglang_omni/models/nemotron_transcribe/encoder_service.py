# SPDX-License-Identifier: Apache-2.0
"""Encode Nemotron perception embeddings on a worker before decoder admission."""

from __future__ import annotations

import hashlib
import json
import queue
import time
from typing import TYPE_CHECKING

import torch
from sglang.srt.managers.schedule_batch import MultimodalDataItem

from sglang_omni.models.qwen3_asr.encoder_service import Qwen3ASRPreLMEncoderService
from sglang_omni.scheduling.pre_lm_encoder import QueueEntry, QueueSignal

if TYPE_CHECKING:
    from sglang_omni.models.nemotron_transcribe.sglang_model import (
        NemotronTranscribeForConditionalGeneration,
    )
else:
    pass

# note (yhayaran): Perception pads a batch to its longest waveform, so long clips encode alone.
SOLO_WAVEFORM_SECONDS = 60
_SAMPLE_RATE = 16000
_LENGTH_RATIO_LIMIT = 4


def build_cache_namespace(
    model: NemotronTranscribeForConditionalGeneration, *, model_path: str
) -> str:
    """Digest identifying this process's perception pipeline for cache keying."""
    config = model.config
    reference = next(model.perception.parameters())
    payload = {
        "model_path": model_path,
        "perception": config.perception,
        "pe_encoder_config": config.pe_encoder_config,
        "audio_token_estimator": config.audio_token_estimator,
        "dtype": str(reference.dtype),
        "device_type": reference.device.type,
        "hidden_size": int(model.perception.proj.out_features),
    }
    blob = json.dumps(payload, sort_keys=True, default=str).encode()
    return hashlib.blake2b(blob, digest_size=8).hexdigest()


def waveform_samples(item: MultimodalDataItem) -> int:
    feature = item.feature
    if isinstance(feature, torch.Tensor):
        return int(feature.numel())
    else:
        return 0


def clips_share_a_batch(first_samples: int, candidate_samples: int) -> bool:
    solo_samples = SOLO_WAVEFORM_SECONDS * _SAMPLE_RATE
    if first_samples > solo_samples or candidate_samples > solo_samples:
        return False
    else:
        pass
    if first_samples <= 0 or candidate_samples <= 0:
        return True
    else:
        pass
    return max(first_samples, candidate_samples) <= (
        min(first_samples, candidate_samples) * _LENGTH_RATIO_LIMIT
    )


class NemotronTranscribePreLMEncoderService(Qwen3ASRPreLMEncoderService):
    """Encode before admission, batching only clips of similar length."""

    ENCODE_TIMEOUT_S = 3600.0
    MODEL_NAME = "Nemotron Transcribe"
    WORKER_NAME = "nemotron-transcribe-audio-encode"
    held: QueueEntry[MultimodalDataItem, torch.Tensor] | None = None

    @staticmethod
    def encoder_module(
        model: NemotronTranscribeForConditionalGeneration,
    ) -> torch.nn.Module:
        return model.perception

    @staticmethod
    def output_hidden_size(model: NemotronTranscribeForConditionalGeneration) -> int:
        return int(model.perception.proj.out_features)

    def next_batch(
        self,
    ) -> tuple[list[QueueEntry[MultimodalDataItem, torch.Tensor]], bool]:
        first = self.held if self.held is not None else self.queue.get()
        self.held = None
        if first is QueueSignal.SHUTDOWN:
            return [], True
        else:
            pass
        batch = [first]
        solo_samples = SOLO_WAVEFORM_SECONDS * _SAMPLE_RATE
        if waveform_samples(first.item) > solo_samples:
            return batch, False
        else:
            pass
        deadline = time.monotonic() + self.max_batch_wait_s
        shutdown = False
        first_samples = waveform_samples(first.item)
        while len(batch) < self.max_batch_size:
            try:
                remaining = deadline - time.monotonic()
                queued = (
                    self.queue.get(timeout=remaining)
                    if remaining > 0
                    else self.queue.get_nowait()
                )
            except queue.Empty:
                break
            if queued is QueueSignal.SHUTDOWN:
                shutdown = True
                break
            else:
                pass
            if not clips_share_a_batch(first_samples, waveform_samples(queued.item)):
                self.held = queued
                break
            else:
                pass
            batch.append(queued)
        return batch, shutdown


__all__ = [
    "NemotronTranscribePreLMEncoderService",
    "build_cache_namespace",
    "clips_share_a_batch",
]
