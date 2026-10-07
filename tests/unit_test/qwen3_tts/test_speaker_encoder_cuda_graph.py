# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace
from typing import Literal

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from sglang_omni.models.qwen3_tts import speaker_encoder_cuda_graph
from sglang_omni.models.qwen3_tts.compat import (
    apply_qwen_tts_transformers_compatibility_patches,
)
from sglang_omni.models.qwen3_tts.speaker_encoder_cuda_graph import (
    SPEAKER_MEL_HOP,
    Qwen3TTSSpeakerEncoderCudaGraphRunner,
    encode_bucketed,
    masked_mean,
    reflect_index,
)
from tests.unit_test.fixtures.accelerator import require_device_streams

SAMPLE_RATE = 24000
NUM_MELS = 8
ENC_DIM = 8
PADS = frozenset({2, 3, 4})


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


def clip_of(frames: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.uniform(-1.0, 1.0, frames * SPEAKER_MEL_HOP).astype(np.float32)


def test_reflect_index_gathers_the_reflect_pad_of_the_valid_frames() -> None:
    torch.manual_seed(1)
    x = torch.randn(1, 3, 12)
    for length, pad in ((12, 2), (9, 3), (5, 4)):
        index = reflect_index(torch.tensor([length]), 12, pad)
        assert index.shape == (12 + 2 * pad,)
        gathered = x.index_select(2, index)[:, :, : length + 2 * pad]
        expected = F.pad(x[:, :, :length], (pad, pad), mode="reflect")
        assert torch.equal(gathered, expected)


def test_reflect_index_stays_inside_a_buffer_much_wider_than_the_clip() -> None:
    index = reflect_index(torch.tensor([5]), 64, 4)
    assert index.shape == (72,)
    assert int(index.min()) == 0
    assert int(index.max()) == 4


def test_bucketed_forward_matches_the_encoder_on_the_valid_frames() -> None:
    encoder = small_speaker_encoder(torch.float64)
    torch.manual_seed(2)
    with torch.inference_mode():
        for frames, width in ((32, 32), (20, 32), (5, 64)):
            mels = torch.randn(1, NUM_MELS, frames, dtype=torch.float64)
            eager = encoder(mels.transpose(1, 2))[0]
            padded = torch.randn(1, NUM_MELS, width, dtype=torch.float64)
            padded[:, :, :frames] = mels
            bucketed = encode_bucketed(encoder, padded, torch.tensor([frames]), PADS)
            assert bucketed.shape == eager.shape == (ENC_DIM,)
            assert torch.allclose(bucketed, eager, atol=1e-9, rtol=1e-9)


def test_bucketed_forward_is_bitwise_independent_of_the_buffer_tail() -> None:
    encoder = small_speaker_encoder(torch.float32)
    torch.manual_seed(5)
    mels = torch.randn(1, NUM_MELS, 20)
    with torch.inference_mode():
        outputs = []
        for tail_scale in (0.0, 1.0, 100.0):
            padded = torch.randn(1, NUM_MELS, 32) * tail_scale
            padded[:, :, :20] = mels
            outputs.append(encode_bucketed(encoder, padded, torch.tensor([20]), PADS))
    assert torch.equal(outputs[0], outputs[1])
    assert torch.equal(outputs[0], outputs[2])


def test_masked_mean_rounds_once_like_mean() -> None:
    x = torch.ones(1, 1, 288, dtype=torch.bfloat16)
    x[:, :, 256] = 0
    x[:, :, 257:] = 100
    mask = (torch.arange(288) < 257).to(x.dtype)[None, None]
    mean = masked_mean(x, mask, torch.tensor([257]))
    assert mean.dtype == torch.bfloat16
    assert torch.equal(mean, x[:, :, :257].mean(2, keepdim=True))
    assert float(mean) == 0.99609375
    x64 = x.double()
    mean64 = masked_mean(x64, mask.double(), torch.tensor([257]))
    assert mean64.dtype == torch.float64
    assert torch.equal(mean64, x64[:, :, :257].mean(2, keepdim=True))


def test_mel_matches_the_checkpoint_front_end_bitwise() -> None:
    encoder = small_speaker_encoder(torch.float32)
    modeling = pytest.importorskip("qwen_tts.core.models.modeling_qwen3_tts")
    default_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        runner = Qwen3TTSSpeakerEncoderCudaGraphRunner(encoder, sample_rate=SAMPLE_RATE)
    finally:
        torch.set_default_dtype(default_dtype)
    torch.manual_seed(4)
    for samples in (40 * SPEAKER_MEL_HOP, 40 * SPEAKER_MEL_HOP + 100):
        waveform = torch.rand(1, samples) * 2 - 1
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
        mel = runner.mel(waveform)
        assert mel.shape == (1, NUM_MELS, samples // SPEAKER_MEL_HOP)
        assert torch.equal(mel, expected)


def test_runner_without_graphs_runs_the_encoder_eagerly() -> None:
    encoder = small_speaker_encoder(torch.float32)
    runner = Qwen3TTSSpeakerEncoderCudaGraphRunner(encoder, sample_rate=SAMPLE_RATE)
    runner.capture((2, 4), 8 * SPEAKER_MEL_HOP)
    assert runner.graphs == {}
    clip = clip_of(20, 6)
    with torch.inference_mode():
        embedding = runner.embed(clip)
        eager = encoder(runner.mel(torch.from_numpy(clip).unsqueeze(0)).transpose(1, 2))
    assert runner.misses == 1
    assert runner.replays == 0
    assert torch.equal(embedding, eager[0])


def test_embed_rejects_a_clip_shorter_than_the_largest_reflect_pad() -> None:
    encoder = small_speaker_encoder(torch.float32)
    runner = Qwen3TTSSpeakerEncoderCudaGraphRunner(encoder, sample_rate=SAMPLE_RATE)
    with torch.inference_mode(), pytest.raises(AssertionError):
        runner.embed(clip_of(4, 8))


@pytest.mark.accelerator
def test_runner_replays_captured_buckets_and_encodes_the_rest() -> None:
    device = require_device_streams()
    device_module = torch.get_device_module(device)
    encoder = small_speaker_encoder(torch.float32).to(device)
    with torch.device(device):
        runner = Qwen3TTSSpeakerEncoderCudaGraphRunner(encoder, sample_rate=SAMPLE_RATE)
    assert runner.pads == PADS
    runner.capture((2, 4), 8 * SPEAKER_MEL_HOP)
    assert sorted(runner.graphs) == [16, 32]

    frames = (32, 20, 5, 24, 20, 40)
    seeds = (0, 1, 2, 3, 1, 5)
    clips = [clip_of(count, seed) for count, seed in zip(frames, seeds)]
    with torch.inference_mode():
        embeddings = [runner.embed(clip) for clip in clips]
        device_module.synchronize(device)
        assert runner.replays == 5
        assert runner.misses == 1
        for clip, embedding in zip(clips, embeddings):
            mels = runner.mel(torch.from_numpy(clip).unsqueeze(0)).to(device)
            eager = encoder(mels.transpose(1, 2))[0]
            assert embedding.shape == (ENC_DIM,)
            assert torch.allclose(embedding, eager, atol=1e-4, rtol=1e-4)
        assert torch.equal(embeddings[1], embeddings[4])
        assert torch.equal(embeddings[5], runner.embed(clips[5]))


def raise_out_of_memory() -> None:
    raise torch.OutOfMemoryError("device out of memory")


@pytest.mark.accelerator
@pytest.mark.parametrize(
    "failing_call, failure_mode",
    [(5, "out_of_memory"), (6, "synchronize")],
    ids=["warmup_of_the_second_bucket", "inside_the_capture_of_the_second_bucket"],
)
def test_a_failed_capture_leaves_no_graphs_and_the_runner_eager(
    monkeypatch,
    caplog,
    failing_call: int,
    failure_mode: Literal["out_of_memory", "synchronize"],
) -> None:
    device = require_device_streams()
    if failure_mode == "out_of_memory":
        fail = raise_out_of_memory
    else:
        # A synchronize inside a capture ends it, which is the other way a bucket
        # fails on real hardware.
        fail = torch.get_device_module(device).synchronize
    encoder = small_speaker_encoder(torch.float32).to(device)
    with torch.device(device):
        runner = Qwen3TTSSpeakerEncoderCudaGraphRunner(encoder, sample_rate=SAMPLE_RATE)
    calls: list[int] = []

    def failing_encode_bucketed(*args):
        calls.append(len(calls) + 1)
        if len(calls) == failing_call:
            fail()
        else:
            pass
        return encode_bucketed(*args)

    monkeypatch.setattr(
        speaker_encoder_cuda_graph, "encode_bucketed", failing_encode_bucketed
    )
    runner.capture((2, 4), 8 * SPEAKER_MEL_HOP)
    assert calls == list(range(1, failing_call + 1))
    assert runner.graphs == {}
    assert "speaker encoder graph capture disabled the runner" in caplog.text

    clip = clip_of(32, 9)
    with torch.inference_mode():
        embedding = runner.embed(clip)
        mels = runner.mel(torch.from_numpy(clip).unsqueeze(0)).to(device)
        eager = encoder(mels.transpose(1, 2))[0]
    assert runner.misses == 1
    assert runner.replays == 0
    assert torch.equal(embedding, eager)
