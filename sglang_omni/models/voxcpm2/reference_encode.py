# SPDX-License-Identifier: Apache-2.0
"""VoxCPM2 reference audio encoding."""

from __future__ import annotations

import io
import os
from typing import Literal
from urllib.parse import unquote, urlparse

import httpx
import librosa
import torch
import torch.nn.functional as F

from sglang_omni.models.voxcpm2 import constants as C
from sglang_omni.models.voxcpm2.components.audio_vae import AudioVAE
from sglang_omni.models.voxcpm2.payload_types import VoxCPM2State
from sglang_omni.preprocessing.cache_key import reference_path_cache_key
from sglang_omni.proto import StagePayload
from sglang_omni.scheduling.pipeline_state import load_state, store_state
from sglang_omni.scheduling.reference_encoder import (
    ReferenceEncodeService,
    TensorReferenceEncodeHook,
)
from sglang_omni.utils.audio import decode_audio_data_uri

PaddingSide = Literal["left", "right"]


def load_reference_audio(source: str, sample_rate: int):
    # note (Xinhao Tan): upstream uses librosa to resample reference audio.
    # The shared torchaudio resampler changes encoded reference features and
    # caused repeated speech in the reference-English regression samples.
    if isinstance(source, (bytes, bytearray, memoryview)):
        source = io.BytesIO(bytes(source))
    else:
        raw = decode_audio_data_uri(source)
        if raw is not None:
            source = io.BytesIO(raw)
        elif source.startswith(("http://", "https://")):
            try:
                timeout = max(1, int(os.getenv("REQUEST_TIMEOUT", "5")))
            except ValueError:
                timeout = 5
            response = httpx.get(source, timeout=timeout, follow_redirects=True)
            response.raise_for_status()
            source = io.BytesIO(response.content)
        elif source.startswith("file://"):
            source = unquote(urlparse(source).path)
    audio, _ = librosa.load(source, sr=sample_rate, mono=True)
    return audio


class VoxCPM2ReferenceEncodeHook(TensorReferenceEncodeHook[tuple[str, PaddingSide]]):
    """Cache reference latents by audio content and padding side."""

    model_revision = ""
    encoder_id = "voxcpm2_audio_vae"
    artifact_kind = "ref_latents"
    storage_dtype = torch.float32
    output_dtype = torch.float32

    def __init__(self, encoder: "VoxCPM2ReferenceEncoder", *, model_identity: str):
        self.encoder = encoder
        self.model_id = str(model_identity)
        self.encoder_config_hash = (
            f"librosa:sr{encoder.sample_rate}:patch{encoder.patch_size}:"
            f"latent{encoder.latent_dim}"
        )

    def input_key(self, item: tuple[str, PaddingSide]) -> str | None:
        return reference_path_cache_key(item[0])

    def options_key(self, item: tuple[str, PaddingSide]) -> str:
        return item[1]

    def encode_one(self, item: tuple[str, PaddingSide]) -> torch.Tensor:
        return self.encoder.encode_audio(item[0], padding_side=item[1])


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
        self.audio_vae = audio_vae
        self.sample_rate = int(audio_vae.sample_rate)
        self.latent_dim = int(audio_vae.latent_dim)
        self.patch_size = int(patch_size)
        if self.patch_size <= 0:
            raise ValueError(f"VoxCPM2 patch_size must be > 0, got {patch_size}")
        self.device = next(audio_vae.parameters()).device
        if cache_model_identity is not None:
            self.service = ReferenceEncodeService(
                VoxCPM2ReferenceEncodeHook(self, model_identity=cache_model_identity),
                max_items=cache_max_items,
                max_bytes=cache_max_bytes,
                log_prefix="VoxCPM2 ref cache",
            )
        else:
            self.service = None

    def warmup(self) -> None:
        """Warm up the audio encoder before accepting requests.

        note (Xinhao Tan): TorchScript optimizes the encoder's Snake activation
        after its initial calls, which can change floating-point results.
        Small differences in encoded reference audio can affect the entire
        generated utterance. Run dummy audio through the encoder at startup
        so the first request also uses the warmed-up computation.
        """
        patch_len = self.patch_size * self.audio_vae.hop_length
        waveform = torch.zeros((1, patch_len), device=self.device, dtype=torch.float32)
        for _ in range(C.WARMUP_ITERATIONS):
            self.audio_vae.encode(waveform, self.sample_rate)

    def encode_audio(self, source: str, *, padding_side: PaddingSide) -> torch.Tensor:
        """Encode one audio source into ``[frames, patch_size, latent_dim]``."""
        audio = load_reference_audio(source, self.sample_rate)
        waveform = torch.from_numpy(audio).unsqueeze(0)

        # note (Xinhao Tan): the padding side is not free. Upstream pads
        # continuation audio on the left and reference audio on the right, and
        # the two sit at opposite ends of the sequence, so flipping a side
        # shifts that audio against the text it is aligned to.
        patch_len = self.patch_size * self.audio_vae.hop_length
        remainder = waveform.shape[1] % patch_len
        if remainder:
            padding = patch_len - remainder
            waveform = F.pad(
                waveform, (padding, 0) if padding_side == "left" else (0, padding)
            )

        latents = self.audio_vae.encode(
            waveform.to(device=self.device, dtype=torch.float32), self.sample_rate
        ).cpu()
        return latents.view(self.latent_dim, -1, self.patch_size).permute(1, 2, 0)

    def encode_cached(self, source: str, padding_side: PaddingSide) -> torch.Tensor:
        if self.service is not None:
            return self.service.get_or_encode((source, padding_side), desc=repr(source))
        return self.encode_audio(source, padding_side=padding_side)

    def encode_payload(self, payload: StagePayload) -> StagePayload:
        state = load_state(payload, VoxCPM2State)
        if state.reference_audio:
            state.ref_latents = self.encode_cached(state.reference_audio, "right")
        if state.prompt_audio:
            state.prompt_latents = self.encode_cached(state.prompt_audio, "left")
        return store_state(payload, state)


__all__ = ["VoxCPM2ReferenceEncoder"]
