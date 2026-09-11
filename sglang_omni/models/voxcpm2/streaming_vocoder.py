# SPDX-License-Identifier: Apache-2.0
"""VoxCPM2 streaming vocoder: decodes latent patches as the engine emits them."""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from typing import Any

import torch

from sglang_omni.models.voxcpm2 import constants as C
from sglang_omni.models.voxcpm2.components.audio_vae import AudioVAE
from sglang_omni.models.voxcpm2.payload_types import VoxCPM2State
from sglang_omni.pipeline.stage.stream_queue import StreamItem
from sglang_omni.proto import StagePayload
from sglang_omni.scheduling.messages import OutgoingMessage
from sglang_omni.scheduling.pipeline_state import load_state, store_state
from sglang_omni.scheduling.streaming_simple_scheduler import StreamingSimpleScheduler

logger = logging.getLogger(__name__)


@dataclass
class _StreamState:
    patches: list[torch.Tensor] = field(default_factory=list)
    emitted_patches: int = 0
    next_decode_patches: int = 0


class VoxCPM2StreamingVocoder(StreamingSimpleScheduler):
    """Re-decodes a trailing window of patches per chunk instead of keeping state.

    The decoder is causal, so a chunk decoded on its own starts from silence and
    seams audibly. Prepending already-emitted patches and dropping their samples
    from the output gives the decoder that context back; the repeated patches are
    the price of not threading per-request convolution state through it.
    """

    def __init__(
        self,
        audio_vae: AudioVAE,
        *,
        device: str,
        patch_size: int,
        stream_stride: int = 8,
        stream_followup_stride: int = 4,
        overlap_patches: int = C.DEFAULT_STREAMING_PREFIX_LEN - 1,
        max_batch_size: int = 1,
    ) -> None:
        super().__init__(self._decode_payload, max_batch_size=max_batch_size)
        if min(stream_stride, stream_followup_stride) <= 0:
            raise ValueError("VoxCPM2 stream strides must be positive")
        if overlap_patches < 0:
            raise ValueError("VoxCPM2 overlap_patches must be >= 0")
        self._vae = audio_vae
        self._device = device
        self._stride = int(stream_stride)
        self._followup_stride = int(stream_followup_stride)
        self._overlap = int(overlap_patches)
        self._samples_per_patch = int(audio_vae.decode_chunk_size) * int(patch_size)
        self._states: dict[str, _StreamState] = {}
        self._lock = threading.Lock()

    def on_stream_chunk(
        self, request_id: str, item: StreamItem
    ) -> list[OutgoingMessage]:
        patch = _as_patch(item)
        if patch is None:
            return []
        with self._lock:
            state = self._states.setdefault(request_id, _StreamState())
            state.patches.append(patch)
            total = len(state.patches)
            threshold = state.next_decode_patches or self._stride
            if total < threshold:
                state.next_decode_patches = threshold
                return []
            audio = self._decode_window(state)
            state.next_decode_patches = total + self._followup_stride
        return self._emit(request_id, audio)

    def on_stream_done(self, request_id: str) -> list[OutgoingMessage]:
        with self._lock:
            state = self._states.get(request_id)
            if state is None or len(state.patches) <= state.emitted_patches:
                return []
            audio = self._decode_window(state)
        return self._emit(request_id, audio)

    def clear_stream_state(self, request_id: str) -> None:
        with self._lock:
            self._states.pop(request_id, None)

    def _decode_window(self, state: _StreamState) -> torch.Tensor | None:
        total = len(state.patches)
        window_start = max(0, state.emitted_patches - self._overlap)
        latents = _stack_patches(state.patches[window_start:]).to(
            device=self._device, dtype=torch.float32
        )
        audio = self._vae.decode(latents).squeeze(0).squeeze(0)
        drop = (state.emitted_patches - window_start) * self._samples_per_patch
        state.emitted_patches = total
        if audio.shape[-1] <= drop:
            return None
        return audio[drop:].detach().cpu()

    def _emit(
        self, request_id: str, audio: torch.Tensor | None
    ) -> list[OutgoingMessage]:
        if audio is None or audio.numel() == 0:
            return []
        return [
            OutgoingMessage(
                request_id=request_id,
                type="stream",
                data={"audio": audio, "sample_rate": self._vae.out_sample_rate},
            )
        ]

    def _decode_payload(self, payload: StagePayload) -> StagePayload:
        """Non-streaming path: decode everything the engine accumulated."""
        state = load_state(payload, VoxCPM2State)
        if state.generated_latents is None:
            raise ValueError("VoxCPM2 vocoder received a payload without latents")
        latents = torch.as_tensor(state.generated_latents)
        if latents.ndim == 2:
            latents = latents.unsqueeze(0)
        audio = self._vae.decode(
            latents.to(device=self._device, dtype=torch.float32),
            state.out_sample_rate,
        )
        state.generated_latents = None
        state.sample_rate = state.out_sample_rate
        payload = store_state(payload, state)
        payload.data["audio"] = audio.squeeze(1).detach().cpu()
        return payload


def _stack_patches(patches: list[torch.Tensor]) -> torch.Tensor:
    """``[n, patch_size, feat_dim]`` patches to the ``[1, feat_dim, frames]`` VAE input."""
    stacked = torch.stack(patches, dim=0)
    return stacked.permute(2, 0, 1).reshape(1, stacked.shape[2], -1)


def _as_patch(item: Any) -> torch.Tensor | None:
    data = getattr(item, "data", item)
    if isinstance(data, dict):
        data = data.get("patch")
    if data is None:
        return None
    return torch.as_tensor(data)


__all__ = ["VoxCPM2StreamingVocoder"]
