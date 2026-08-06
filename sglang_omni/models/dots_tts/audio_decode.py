# SPDX-License-Identifier: Apache-2.0
"""Terminal audio decode stage for dots.tts."""

from __future__ import annotations

import time

import torch

from sglang_omni.models.dots_tts.components.vocoder.vocoder_inference import (
    VocoderInference,
)
from sglang_omni.models.dots_tts.payload_types import (
    DotsTtsState,
    load_dots_tts_state,
    store_dots_tts_state,
)
from sglang_omni.proto import StagePayload
from sglang_omni.scheduling.pipeline_state import build_usage
from sglang_omni.scheduling.vocoder_base import BatchVocoderBase
from sglang_omni.utils.audio_payload import audio_waveform_payload


class DotsTtsBatchVocoder(BatchVocoderBase):
    """AudioVAE latent decode on the shared batch-vocoder base.

    ``decode_latents`` runs one contiguous latent sequence per request and the
    sequences differ in length, so ``decode_batch`` stays a per-item loop; the
    base only supplies the scheduler shell.
    """

    def __init__(
        self,
        vocoder_inference: VocoderInference,
        *,
        sample_rate: int,
        device: torch.device,
        keep_latents: bool = False,
    ) -> None:
        self._vocoder_inference = vocoder_inference
        self._sample_rate = int(sample_rate)
        self._device = device
        self._keep_latents = bool(keep_latents)

    def prepare_item(
        self, payload: StagePayload
    ) -> tuple[DotsTtsState, torch.Tensor | None]:
        state = load_dots_tts_state(payload)
        latents = state.generated_latents
        if latents is not None:
            latents = latents.to(device=self._device, dtype=torch.float32)
        return state, latents

    async def decode_batch(
        self, items: list[tuple[DotsTtsState, torch.Tensor | None]]
    ) -> list[tuple[torch.Tensor, int]]:
        return [self._decode_item(state, latents) for state, latents in items]

    def store_result(
        self,
        payload: StagePayload,
        state: DotsTtsState,
        wav: torch.Tensor,
        sample_rate: int,
    ) -> StagePayload:
        state.sample_rate = int(sample_rate)
        state.duration_s = float(wav.numel() / int(sample_rate))
        if not self._keep_latents:
            state.generated_latents = None

        payload = store_dots_tts_state(payload, state)
        payload.data.update(
            audio_waveform_payload(
                wav,
                sample_rate=int(sample_rate),
                modality="audio",
                source_hint="dots.tts",
            )
        )
        usage = build_usage(state)
        if usage is not None:
            payload.data["usage"] = usage
        return payload

    def _decode_item(
        self, state: DotsTtsState, latents: torch.Tensor | None
    ) -> tuple[torch.Tensor, int]:
        if latents is None or latents.numel() == 0:
            raise RuntimeError("dots.tts audio decode requires generated latents")
        if latents.ndim != 3:
            raise RuntimeError(
                "dots.tts latents must have shape [1, frames, latent_dim], "
                f"got {tuple(latents.shape)}"
            )

        started = time.perf_counter()
        waveform = self._vocoder_inference.decode_latents(latents)
        state.audio_decode_time_s = time.perf_counter() - started
        return waveform.detach().reshape(-1), self._sample_rate


__all__ = ["DotsTtsBatchVocoder"]
