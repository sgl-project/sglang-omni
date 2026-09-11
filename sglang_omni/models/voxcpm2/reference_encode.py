# SPDX-License-Identifier: Apache-2.0
"""VoxCPM2 reference audio encoding."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from sglang_omni.models.voxcpm2.components.audio_vae import AudioVAE
from sglang_omni.models.voxcpm2.payload_types import VoxCPM2State
from sglang_omni.preprocessing.cache_key import reference_path_cache_key
from sglang_omni.proto import StagePayload
from sglang_omni.scheduling.pipeline_state import load_state, store_state
from sglang_omni.scheduling.reference_encoder import (
    ReferenceEncodeService,
    TensorReferenceEncodeHook,
)
from sglang_omni.utils.audio import load_audio


class _VoxCPM2ReferenceEncodeHook(TensorReferenceEncodeHook[tuple[str, str]]):
    """Cache reference latents by audio content and padding side."""

    model_revision = ""
    encoder_id = "voxcpm2_audio_vae"
    artifact_kind = "ref_latents"
    storage_dtype = torch.float32
    output_dtype = torch.float32

    def __init__(self, encoder: "VoxCPM2ReferenceEncoder", *, model_identity: str):
        self._encoder = encoder
        self.model_id = str(model_identity)
        self.encoder_config_hash = (
            f"sr{encoder.sample_rate}:patch{encoder.patch_size}:"
            f"latent{encoder.latent_dim}"
        )

    def input_key(self, item: tuple[str, str]) -> str | None:
        return reference_path_cache_key(item[0])

    def options_key(self, item: tuple[str, str]) -> str:
        return item[1]

    def encode_one(self, item: tuple[str, str]) -> torch.Tensor:
        return self._encoder.encode_audio(item[0], padding_side=item[1])


class VoxCPM2ReferenceEncoder:
    """Turns one reference or continuation audio into VoxCPM2 latent patches."""

    def __init__(
        self,
        audio_vae: AudioVAE,
        *,
        patch_size: int,
        cache_model_identity: str | None = None,
        cache_max_items: int | None = 256,
        cache_max_bytes: int | None = 64 * 1024 * 1024,
    ) -> None:
        self._audio_vae = audio_vae
        self.sample_rate = int(audio_vae.sample_rate)
        self.latent_dim = int(audio_vae.latent_dim)
        self.patch_size = int(patch_size)
        if self.patch_size <= 0:
            raise ValueError(f"VoxCPM2 patch_size must be > 0, got {patch_size}")
        self.device = next(audio_vae.parameters()).device
        self._service: ReferenceEncodeService | None = None
        if cache_model_identity is not None:
            self._service = ReferenceEncodeService(
                _VoxCPM2ReferenceEncodeHook(self, model_identity=cache_model_identity),
                max_items=cache_max_items,
                max_bytes=cache_max_bytes,
                log_prefix="VoxCPM2 ref cache",
            )

    def encode_audio(self, source: str, *, padding_side: str) -> torch.Tensor:
        """Encode one audio source into ``[frames, patch_size, latent_dim]``."""
        audio = load_audio(source, target_sample_rate=self.sample_rate, mono=True)
        waveform = torch.from_numpy(audio).unsqueeze(0)

        # note (Xinhao Tan): the padding side is not free. Upstream pads
        # continuation audio on the left and reference audio on the right, and
        # the two sit at opposite ends of the sequence, so flipping a side
        # shifts that audio against the text it is aligned to.
        patch_len = self.patch_size * self._audio_vae.hop_length
        remainder = waveform.shape[1] % patch_len
        if remainder:
            padding = patch_len - remainder
            waveform = F.pad(
                waveform, (padding, 0) if padding_side == "left" else (0, padding)
            )

        latents = self._audio_vae.encode(
            waveform.to(device=self.device, dtype=torch.float32), self.sample_rate
        ).cpu()
        return latents.view(self.latent_dim, -1, self.patch_size).permute(1, 2, 0)

    def _encode(self, source: str, padding_side: str) -> torch.Tensor:
        if self._service is not None:
            return self._service.get_or_encode(
                (source, padding_side), desc=repr(source)
            )
        return self.encode_audio(source, padding_side=padding_side)

    def encode_payload(self, payload: StagePayload) -> StagePayload:
        state = load_state(payload, VoxCPM2State)
        if state.reference_audio:
            state.ref_latents = self._encode(state.reference_audio, "right")
        if state.prompt_audio:
            state.prompt_latents = self._encode(state.prompt_audio, "left")
        return store_state(payload, state)


__all__ = ["VoxCPM2ReferenceEncoder"]
