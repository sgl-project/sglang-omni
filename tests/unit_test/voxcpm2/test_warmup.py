# SPDX-License-Identifier: Apache-2.0
"""VoxCPM2 warmup: the TorchScript activation is exercised before serving starts."""

from __future__ import annotations

import torch

from sglang_omni.models.voxcpm2 import constants as C
from sglang_omni.models.voxcpm2.reference_encode import VoxCPM2ReferenceEncoder
from sglang_omni.models.voxcpm2.streaming_vocoder import VoxCPM2StreamingVocoder

_PATCH_SIZE = 4
_LATENT_DIM = 8


class _RecordingAudioVAE:
    decode_chunk_size = 3
    out_sample_rate = 48000
    sample_rate = 16000
    hop_length = 2
    latent_dim = _LATENT_DIM

    def __init__(self):
        self.decoded: list[tuple[int, ...]] = []
        self.encoded: list[tuple[int, ...]] = []

    def decode(self, latents, sample_rate=None):
        del sample_rate
        self.decoded.append(tuple(latents.shape))
        return latents.new_zeros(1, 1, int(latents.shape[-1]) * self.decode_chunk_size)

    def encode(self, audio, sample_rate=None):
        del sample_rate
        self.encoded.append(tuple(audio.shape))
        return audio.new_zeros(1, _LATENT_DIM, 1)

    def parameters(self):
        yield torch.zeros(1)


def test_the_vocoder_decodes_before_the_stage_reports_ready():
    """Un-warmed, the first caller gets different audio from the same request later."""
    vae = _RecordingAudioVAE()
    vocoder = VoxCPM2StreamingVocoder(
        vae, device="cpu", patch_size=_PATCH_SIZE, stream_stride=4
    )
    assert vae.decoded == []

    vocoder.warmup_now()
    assert len(vae.decoded) == C.WARMUP_ITERATIONS
    assert all(shape[1] == _LATENT_DIM for shape in vae.decoded)


def test_the_reference_encoder_warms_the_activation_it_shares():
    vae = _RecordingAudioVAE()
    encoder = VoxCPM2ReferenceEncoder(
        vae, patch_size=_PATCH_SIZE, cache_model_identity=None
    )
    assert vae.encoded == []

    encoder.warmup()
    assert len(vae.encoded) == C.WARMUP_ITERATIONS


def test_warmup_does_not_leave_per_request_state_behind():
    """A warmup that registered a request would leak into the first real one."""
    vocoder = VoxCPM2StreamingVocoder(
        _RecordingAudioVAE(), device="cpu", patch_size=_PATCH_SIZE, stream_stride=4
    )
    vocoder.warmup_now()
    assert vocoder.stream_states == {}
