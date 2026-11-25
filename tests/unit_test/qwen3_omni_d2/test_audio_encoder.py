# SPDX-License-Identifier: Apache-2.0
"""The causal D2 mel frontend emits frames that later audio and chunking cannot change."""

import torch
from transformers import WhisperFeatureExtractor

from sglang_omni.models.qwen3_omni_d2.components.audio_encoder import CausalLogMel

SAMPLE_RATE_HZ = 16_000
INTERVAL_SAMPLES = 1_280


def frontend() -> CausalLogMel:
    extractor = WhisperFeatureExtractor(feature_size=128, sampling_rate=SAMPLE_RATE_HZ)
    return CausalLogMel(
        torch.from_numpy(extractor.mel_filters).float(),
        extractor.n_fft,
        extractor.hop_length,
    )


def test_frames_do_not_depend_on_chunking_or_later_audio() -> None:
    silence = torch.zeros(2 * INTERVAL_SAMPLES)
    speech = torch.randn(
        4 * INTERVAL_SAMPLES, generator=torch.Generator().manual_seed(0)
    )
    whole = frontend().push(torch.cat((silence, speech)))
    chunked = frontend()
    early = chunked.push(silence)
    pieces = [early]
    for start, end in ((0, 1000), (1000, 3333), (3333, speech.numel())):
        pieces.append(chunked.push(speech[start:end]))
    assert whole.shape == (128, 6 * INTERVAL_SAMPLES // 160)
    torch.testing.assert_close(torch.cat(pieces, dim=1), whole, rtol=0, atol=1e-5)
    frames = early.shape[1]
    torch.testing.assert_close(whole[:, :frames], early, rtol=0, atol=0)
    # An utterance-wide floor would lift the silent frames to within 8 log units of the speech peak.
    assert whole[:, :frames].max() < whole[:, frames:].max() - 8.0 / 4.0
