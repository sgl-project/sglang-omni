# SPDX-License-Identifier: Apache-2.0
"""Ming ASR pre-LM embeddings match encoding each mel on its own."""

from __future__ import annotations

import pytest
import torch

from sglang_omni.models.ming3_asr.encoder_service import Ming3ASRPreLMEncoderService
from sglang_omni.modules.ming_audio.audio_encoder import (
    AudioEncoderConfig,
    Ming3AudioEncoder,
)


def tiny_encoder() -> Ming3AudioEncoder:
    config = AudioEncoderConfig(
        n_mels=4,
        n_audio_ctx=32,
        n_audio_state=8,
        n_audio_head=2,
        n_audio_layer=2,
        downsampler_kernel_size_list=[3],
        downsampler_stride_list=[2],
        whisper_chunk_size=32,
    )
    return Ming3AudioEncoder(
        config,
        hidden_size=16,
        should_upcast_layer_norm=True,
        should_batch_chunks=False,
    ).eval()


def test_pre_lm_encode_matches_separate_mels() -> None:
    torch.manual_seed(0)
    encoder = tiny_encoder()
    long_mel = torch.randn(40, encoder.config.n_mels)
    short_mel = torch.randn(9, encoder.config.n_mels)
    service = Ming3ASRPreLMEncoderService(
        encoder,
        max_batch_size=8,
        max_batch_wait_ms=2,
    )
    try:
        combined = service.encode([long_mel, short_mel])
        long_embedding = service.submit([long_mel])
        short_embedding = service.submit([short_mel])
        with torch.inference_mode():
            expected = encoder([long_mel, short_mel])
            expected_long = encoder([long_mel])
            expected_short = encoder([short_mel])
        long_result = long_embedding.result(timeout=service.ENCODE_TIMEOUT_S)
        short_result = short_embedding.result(timeout=service.ENCODE_TIMEOUT_S)
    finally:
        service.close()
    torch.testing.assert_close(combined, expected, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(long_result, expected_long, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(short_result, expected_short, rtol=1e-5, atol=1e-5)
    with pytest.raises(RuntimeError, match="closed"):
        service.encode([short_mel])
