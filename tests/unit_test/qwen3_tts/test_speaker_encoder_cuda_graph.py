# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from sglang_omni.models.qwen3_tts.compat import (
    apply_qwen_tts_transformers_compatibility_patches,
)
from sglang_omni.models.qwen3_tts.speaker_encoder_cuda_graph import (
    SPEAKER_MEL_HOP,
    Qwen3TTSSpeakerEncoderCudaGraphRunner,
    encode_bucketed,
    reflect_index,
)

SAMPLE_RATE = 24000
NUM_MELS = 8
ENC_DIM = 8


def small_speaker_encoder(dtype: torch.dtype) -> torch.nn.Module:
    """The checkpoint's encoder class at toy widths: kernel 5 first, then dilations 2, 3, 4."""
    apply_qwen_tts_transformers_compatibility_patches()
    modeling = pytest.importorskip("qwen_tts.core.models.modeling_qwen3_tts")
    config = SimpleNamespace(
        mel_dim=NUM_MELS,
        enc_channels=[16, 16, 16, 16, 48],
        enc_kernel_sizes=[5, 3, 3, 3, 1],
        enc_dilations=[1, 2, 3, 4, 1],
        enc_res2net_scale=8,
        enc_se_channels=4,
        enc_attention_channels=4,
        enc_dim=ENC_DIM,
    )
    torch.manual_seed(7)
    return modeling.Qwen3TTSSpeakerEncoder(config).to(dtype).eval()


def test_reflect_index_gathers_the_reflect_pad_of_the_valid_frames() -> None:
    torch.manual_seed(1)
    x = torch.randn(1, 3, 12)
    for length, pad in ((12, 2), (9, 3), (5, 4)):
        index = reflect_index(torch.tensor([length]), 12, pad)
        assert index.shape == (12 + 2 * pad,)
        gathered = x.index_select(2, index)[:, :, : length + 2 * pad]
        expected = F.pad(x[:, :, :length], (pad, pad), mode="reflect")
        assert torch.equal(gathered, expected)


def test_bucketed_forward_matches_the_encoder_on_the_valid_frames() -> None:
    encoder = small_speaker_encoder(torch.float64)
    torch.manual_seed(2)
    with torch.inference_mode():
        for frames, width in ((32, 32), (20, 32), (5, 64)):
            mels = torch.randn(1, NUM_MELS, frames, dtype=torch.float64)
            eager = encoder(mels.transpose(1, 2))[0]
            padded = torch.randn(1, NUM_MELS, width, dtype=torch.float64)
            padded[:, :, :frames] = mels
            bucketed = encode_bucketed(
                encoder, padded, torch.tensor([frames]), frozenset({2, 3, 4})
            )
            assert bucketed.shape == eager.shape == (ENC_DIM,)
            assert torch.allclose(bucketed, eager, atol=1e-9, rtol=1e-9)


def test_mel_matches_the_checkpoint_front_end_bitwise() -> None:
    encoder = small_speaker_encoder(torch.float32)
    modeling = pytest.importorskip("qwen_tts.core.models.modeling_qwen3_tts")
    runner = Qwen3TTSSpeakerEncoderCudaGraphRunner(encoder, sample_rate=SAMPLE_RATE)
    torch.manual_seed(4)
    waveform = torch.rand(1, 40 * SPEAKER_MEL_HOP) * 2 - 1
    expected = modeling.mel_spectrogram(
        waveform,
        n_fft=1024,
        num_mels=NUM_MELS,
        sampling_rate=SAMPLE_RATE,
        hop_size=256,
        win_size=1024,
        fmin=0,
        fmax=12000,
    )
    assert torch.equal(runner.mel(waveform), expected)


@pytest.mark.accelerator
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_runner_replays_captured_buckets_and_encodes_the_rest() -> None:
    device = torch.device("cuda", torch.cuda.current_device())
    encoder = small_speaker_encoder(torch.float32).to(device)
    with torch.device(device):
        runner = Qwen3TTSSpeakerEncoderCudaGraphRunner(encoder, sample_rate=SAMPLE_RATE)
    assert runner.pads == {2, 3, 4}
    runner.capture((16, 32))
    assert sorted(runner.graphs) == [16, 32]

    rng = np.random.default_rng(3)
    clips = [
        rng.uniform(-1.0, 1.0, frames * SPEAKER_MEL_HOP).astype(np.float32)
        for frames in (32, 20, 5, 40)
    ]
    with torch.inference_mode():
        embeddings = [runner.embed(clip) for clip in clips]
        torch.cuda.synchronize(device)
        assert runner.replays == 3
        assert runner.misses == 1
        for clip, embedding in zip(clips, embeddings):
            mels = runner.mel(torch.from_numpy(clip).unsqueeze(0)).to(device)
            eager = encoder(mels.transpose(1, 2))[0]
            assert embedding.shape == (ENC_DIM,)
            assert torch.allclose(embedding, eager, atol=1e-4, rtol=1e-4)
