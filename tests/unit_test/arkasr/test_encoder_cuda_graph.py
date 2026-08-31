# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from transformers import WhisperConfig

from sglang_omni.models.arkasr.audio_lengths import arkasr_num_audio_tokens
from sglang_omni.models.arkasr.audio_tower import ArkAudioMLPAdapter
from sglang_omni.models.arkasr.configuration_arkasr import ArkasrConfig
from sglang_omni.models.arkasr.encoder_cuda_graph import (
    _PRECAPTURE_MEL_FRAMES,
    ArkasrEncoderCudaGraphRunner,
    _batch_buckets,
    _fit_bucket,
    _t_buckets_upto,
)
from sglang_omni.models.arkasr.sglang_model import ArkasrForConditionalGeneration


def test_batch_buckets_are_powers_of_two_plus_limit() -> None:
    assert _batch_buckets(8) == (1, 2, 4, 8)
    assert _batch_buckets(6) == (1, 2, 4, 6)
    assert _batch_buckets(16) == (1, 2, 4, 8, 16)
    assert _batch_buckets(1) == (1,)


def test_fit_bucket_rounds_up_within_captured_set() -> None:
    batch8 = _batch_buckets(8)
    assert _fit_bucket(1, batch8) == 1
    assert _fit_bucket(3, batch8) == 4
    assert _fit_bucket(5, batch8) == 8
    assert _fit_bucket(9, batch8) is None
    assert _fit_bucket(5, _batch_buckets(6)) == 6
    assert _fit_bucket(9, _batch_buckets(16)) == 16
    t_working = _t_buckets_upto(_PRECAPTURE_MEL_FRAMES, merge_factor=4)
    assert _fit_bucket(1, t_working) == 64
    assert _fit_bucket(64, t_working) == 64
    assert _fit_bucket(65, t_working) == 128
    assert _fit_bucket(1024, t_working) == 1024
    assert _fit_bucket(1025, t_working) is None


def test_t_buckets_upto_covers_working_set() -> None:
    buckets = _t_buckets_upto(_PRECAPTURE_MEL_FRAMES, merge_factor=4)
    assert buckets[0] == 64
    assert buckets[-1] == _PRECAPTURE_MEL_FRAMES
    assert all(b % 64 == 0 for b in buckets)
    assert buckets == tuple(range(64, _PRECAPTURE_MEL_FRAMES + 1, 64))
    # T_down = T/2 must be divisible by merge_factor so the adapter reshape
    # captured in the graph matches the eager path.
    for t in _t_buckets_upto(3000, merge_factor=4):
        assert t % 8 == 0


def _tiny_config() -> ArkasrConfig:
    whisper = WhisperConfig(
        d_model=32,
        encoder_layers=2,
        encoder_attention_heads=4,
        encoder_ffn_dim=64,
        num_mel_bins=8,
        max_source_positions=64,
    )
    return ArkasrConfig(
        whisper_config=whisper,
        merge_factor=4,
        hidden_size=48,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=64,
        vocab_size=256,
        audio_token_id=151663,
    )


def _tiny_ark_audio_mm_model() -> ArkasrForConditionalGeneration:
    model = ArkasrForConditionalGeneration.__new__(ArkasrForConditionalGeneration)
    nn.Module.__init__(model)
    model.audio_encoder = ArkAudioMLPAdapter(_tiny_config()).eval()
    model.encoder_max_batch_size = model.DEFAULT_ENCODER_MAX_BATCH_SIZE
    model.encoder_cuda_graph_runner = None
    return model


def _item(num_frames: int) -> SimpleNamespace:
    return SimpleNamespace(
        feature=torch.randn(1, 8, num_frames),
        feature_attention_mask=torch.ones(1, num_frames, dtype=torch.long),
    )


def test_get_audio_feature_routes_through_graph_runner() -> None:
    observed: dict[str, object] = {}

    class _Runner:
        def run(self, mel, lengths):
            observed["mel_shape"] = tuple(mel.shape)
            observed["lengths"] = list(lengths)
            batch = mel.shape[0]
            t_out = arkasr_num_audio_tokens(mel.shape[-1], 4)
            return torch.ones(batch, t_out, 48)

    model = _tiny_ark_audio_mm_model()
    model.encoder_cuda_graph_runner = _Runner()
    calls: list[tuple] = []

    def record_call(_module, args, kwargs):
        del args
        calls.append(kwargs.get("attention_mask"))

    handle = model.audio_encoder.register_forward_pre_hook(
        record_call, with_kwargs=True
    )
    try:
        out = model.get_audio_feature([_item(17), _item(9)])
    finally:
        handle.remove()

    assert observed["mel_shape"] == (2, 8, 17)
    assert observed["lengths"] == [17, 9]
    expected_rows = arkasr_num_audio_tokens(17, 4) + arkasr_num_audio_tokens(9, 4)
    assert out.shape == (expected_rows, 48)
    assert calls == []


def test_get_audio_feature_falls_back_to_eager_when_runner_declines() -> None:
    class _DecliningRunner:
        def run(self, mel, lengths):
            del mel, lengths
            return None

    model = _tiny_ark_audio_mm_model()
    model.encoder_cuda_graph_runner = _DecliningRunner()
    calls: list[object] = []

    def record_call(_module, args, kwargs):
        del args
        calls.append(None if kwargs.get("attention_mask") is None else "masked")

    handle = model.audio_encoder.register_forward_pre_hook(
        record_call, with_kwargs=True
    )
    try:
        out = model.get_audio_feature([_item(17), _item(9)])
    finally:
        handle.remove()

    assert calls == ["masked"]
    expected_rows = arkasr_num_audio_tokens(17, 4) + arkasr_num_audio_tokens(9, 4)
    assert out.shape == (expected_rows, 48)


def test_get_audio_feature_without_runner_keeps_unmasked_single_item() -> None:
    model = _tiny_ark_audio_mm_model()
    calls: list[object] = []

    def record_call(_module, args, kwargs):
        del args
        calls.append(kwargs.get("attention_mask"))

    handle = model.audio_encoder.register_forward_pre_hook(
        record_call, with_kwargs=True
    )
    try:
        out = model.get_audio_feature([_item(12)])
    finally:
        handle.remove()

    assert calls == [None]
    assert out.shape == (arkasr_num_audio_tokens(12, 4), 48)


def test_run_returns_none_on_cpu_encoder() -> None:
    encoder = ArkAudioMLPAdapter(_tiny_config()).eval()
    runner = ArkasrEncoderCudaGraphRunner(encoder, min_free_gb=0.0)
    mel = torch.randn(1, 8, 40)
    assert runner.run(mel, [40]) is None
    assert runner.captured_buckets == ()


@pytest.mark.accelerator
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_run_without_precapture_does_not_capture() -> None:
    encoder = ArkAudioMLPAdapter(_tiny_config()).eval().cuda()
    runner = ArkasrEncoderCudaGraphRunner(encoder, max_batch_size=4, min_free_gb=0.0)
    mel = torch.randn(1, 8, 40, device="cuda", dtype=encoder.dtype)
    assert runner.run(mel, [40]) is None
    assert runner.captured_buckets == ()


@pytest.mark.accelerator
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_graph_replay_matches_eager_masked_forward() -> None:
    encoder = ArkAudioMLPAdapter(_tiny_config()).eval().cuda()
    runner = ArkasrEncoderCudaGraphRunner(encoder, max_batch_size=4, min_free_gb=0.0)
    runner.capture_working_set(8, max_mel_frames=64)
    torch.manual_seed(0)
    real_t = 40
    lengths = [40, 17]
    mel = torch.randn(2, 8, real_t, device="cuda", dtype=encoder.dtype)
    mel[1, :, lengths[1] :] = 0

    graph_out = runner.run(mel, lengths)
    assert graph_out is not None, "encoder CUDA graph replay declined"
    assert (2, 64) in runner.captured_buckets

    t_bucket = 64
    padded = torch.zeros(2, 8, t_bucket, device="cuda", dtype=encoder.dtype)
    padded[:, :, :real_t] = mel
    ilens = torch.tensor(lengths, device="cuda", dtype=torch.long)
    mask = torch.arange(t_bucket, device="cuda").unsqueeze(0) < ilens.unsqueeze(1)
    with torch.no_grad():
        eager = encoder(padded, attention_mask=mask)
    torch.testing.assert_close(graph_out, eager, rtol=1e-3, atol=1e-3)


@pytest.mark.accelerator
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_capture_working_set_fills_batch_and_t_buckets() -> None:
    encoder = ArkAudioMLPAdapter(_tiny_config()).eval().cuda()
    runner = ArkasrEncoderCudaGraphRunner(
        encoder, max_batch_size=2, max_mel_frames=128, min_free_gb=0.0
    )
    runner.capture_working_set(8, max_mel_frames=64)
    assert runner.captured_buckets == ((1, 64), (2, 64))
    mel = torch.randn(1, 8, 40, device="cuda", dtype=encoder.dtype)
    out = runner.run(mel, [40])
    assert out is not None
    assert out.shape[0] == 1
    overflow = torch.randn(1, 8, 80, device="cuda", dtype=encoder.dtype)
    assert runner.run(overflow, [80]) is None
    assert runner.captured_buckets == ((1, 64), (2, 64))


@pytest.mark.accelerator
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_replay_failure_sticks_to_eager() -> None:
    encoder = ArkAudioMLPAdapter(_tiny_config()).eval().cuda()
    runner = ArkasrEncoderCudaGraphRunner(
        encoder, max_batch_size=1, max_mel_frames=64, min_free_gb=0.0
    )
    runner.capture_working_set(8, max_mel_frames=64)
    assert runner.captured_buckets == ((1, 64),)
    entry = runner._graphs[(1, 64)]

    def _boom() -> None:
        raise RuntimeError("replay boom")

    entry.graph.replay = _boom  # type: ignore[method-assign]
    mel = torch.randn(1, 8, 40, device="cuda", dtype=encoder.dtype)
    assert runner.run(mel, [40]) is None
    assert runner.captured_buckets == ()
    assert runner.run(mel, [40]) is None
    assert runner.captured_buckets == ()
