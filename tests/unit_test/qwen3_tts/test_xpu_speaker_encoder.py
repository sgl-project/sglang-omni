# SPDX-License-Identifier: Apache-2.0
"""Intel XPU cases for the Qwen3-TTS speaker encoder graphs.

Base clones a voice from the embedding this encoder returns, and its graphs capture
on an Intel XPU. Every case here captures the checkpoint's encoder class at toy
widths on the device, so none needs a checkpoint, and skips off an XPU.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from sglang_omni.models.qwen3_tts.compat import (
    apply_qwen_tts_transformers_compatibility_patches,
)
from sglang_omni.models.qwen3_tts.speaker_encoder_cuda_graph import (
    SPEAKER_MEL_HOP,
    Qwen3TTSSpeakerEncoderCudaGraphRunner,
)
from tests.unit_test.fixtures.accelerator import require_device_streams

SAMPLE_RATE = 24000
NUM_MELS = 8
ENCODER_DIM = 8
# The reference encoder's ladder, in its own frames, and its hop: 16 and 32 mel frames.
CODEC_FRAME_BUCKETS = (2, 4)
CODEC_HOP = 8 * SPEAKER_MEL_HOP


@pytest.fixture
def xpu_device() -> torch.device:
    device = require_device_streams()
    if device.type != "xpu":
        pytest.skip(f"needs an Intel XPU, found {device.type}")
    else:
        pass
    return device


@pytest.fixture
def captured_runner(
    xpu_device: torch.device,
) -> Qwen3TTSSpeakerEncoderCudaGraphRunner:
    """A runner whose buckets are captured on the XPU, rebuilt per case so the replay
    and miss counts a case reads are its own."""
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
        enc_dim=ENCODER_DIM,
    )
    torch.manual_seed(7)
    encoder = modeling.Qwen3TTSSpeakerEncoder(config).to(torch.float32).eval()
    encoder = encoder.to(xpu_device)
    with torch.device(xpu_device):
        runner = Qwen3TTSSpeakerEncoderCudaGraphRunner(encoder, sample_rate=SAMPLE_RATE)
    runner.capture(CODEC_FRAME_BUCKETS, CODEC_HOP)
    return runner


def clip_of(frames: int, seed: int) -> np.ndarray:
    generator = np.random.default_rng(seed)
    return generator.uniform(-1.0, 1.0, frames * SPEAKER_MEL_HOP).astype(np.float32)


def eager_embedding(
    runner: Qwen3TTSSpeakerEncoderCudaGraphRunner,
    clip: np.ndarray,
    device: torch.device,
) -> torch.Tensor:
    mels = runner.mel(torch.from_numpy(clip).unsqueeze(0)).to(device)
    return runner.encoder(mels.transpose(1, 2))[0]


@pytest.mark.accelerator
def test_every_bucket_captures_on_xpu_and_replays_what_the_encoder_computes(
    captured_runner: Qwen3TTSSpeakerEncoderCudaGraphRunner, xpu_device: torch.device
) -> None:
    """Both buckets capture through the XPU graph backend, and a clip shorter than its
    bucket replays the embedding the encoder computes from its own frames alone."""
    assert captured_runner.stats() == {
        "enabled": True,
        "disable_reason": None,
        "bucket_frames": [16, 32],
        "captured": [16, 32],
        "replays": 0,
        "misses": 0,
    }

    frames = (16, 5, 32, 20, 24)
    clips = [clip_of(count, seed) for seed, count in enumerate(frames)]
    with torch.inference_mode():
        embeddings = [captured_runner.embed(clip) for clip in clips]
        torch.xpu.synchronize(xpu_device)
        assert captured_runner.stats()["replays"] == len(frames)
        assert captured_runner.stats()["misses"] == 0
        for clip, embedding in zip(clips, embeddings):
            assert embedding.shape == (ENCODER_DIM,)
            assert embedding.device == xpu_device
            assert embedding.dtype == torch.float32
            assert bool(torch.isfinite(embedding).all())
            torch.testing.assert_close(
                embedding,
                eager_embedding(captured_runner, clip, xpu_device),
                atol=1e-4,
                rtol=1e-4,
            )


@pytest.mark.accelerator
def test_a_replayed_embedding_survives_the_next_replay_of_its_bucket_on_xpu(
    captured_runner: Qwen3TTSSpeakerEncoderCudaGraphRunner, xpu_device: torch.device
) -> None:
    """Two clips of one bucket replay into the same output buffer, so the first caller's
    embedding has to be a copy; compared bitwise, since a view onto that buffer would
    hold the second clip's values and still sit within any replay tolerance."""
    first_clip = clip_of(20, 11)
    second_clip = clip_of(24, 12)
    with torch.inference_mode():
        first = captured_runner.embed(first_clip)
        torch.xpu.synchronize(xpu_device)
        retained = first.clone()
        second = captured_runner.embed(second_clip)
        torch.xpu.synchronize(xpu_device)

    assert not torch.equal(retained, second)
    assert torch.equal(first, retained)


@pytest.mark.accelerator
def test_a_clip_past_the_largest_bucket_encodes_eagerly_on_xpu(
    captured_runner: Qwen3TTSSpeakerEncoderCudaGraphRunner, xpu_device: torch.device
) -> None:
    """Past the ladder XPU keeps the host mel and the plain encoder: the device mel the
    CUDA path switches to is a device FFT, which this platform has not taken. Bitwise,
    because the device mel would land within the replay tolerance of the host one."""
    clip = clip_of(40, 13)
    with torch.inference_mode():
        embedding = captured_runner.embed(clip)
        torch.xpu.synchronize(xpu_device)

        assert captured_runner.stats()["misses"] == 1
        assert captured_runner.stats()["replays"] == 0
        assert embedding.device == xpu_device
        assert bool(torch.isfinite(embedding).all())
        assert torch.equal(
            embedding, eager_embedding(captured_runner, clip, xpu_device)
        )
