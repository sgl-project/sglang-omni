"""Streaming audio preserves causal history, reset, and caller-owned inputs."""

import pytest
import torch

from sglang_omni.models.nemotron_voicechat.conformer import (
    SAMPLES_PER_FRAME,
    AudioPerception,
    StreamingPerception,
)


@pytest.fixture(params=[0, 2, 7])
def perception_model(request: pytest.FixtureRequest) -> AudioPerception:
    torch.manual_seed(2188)
    model = AudioPerception(
        {
            "preprocessor": {
                "sample_rate": 16000,
                "n_fft": 512,
                "window_stride": 0.01,
                "window_size": 0.025,
                "features": 8,
            },
            "encoder": {
                "feat_in": 8,
                "d_model": 8,
                "subsampling_conv_channels": 4,
                "subsampling_factor": 8,
                "conv_kernel_size": 3,
                "use_bias": True,
                "ff_expansion_factor": 2,
                "n_heads": 2,
                "att_context_size": [int(request.param), 0],
                "n_layers": 2,
                "xscaling": True,
            },
            "output_dim": 8,
        }
    ).eval()
    model.preprocessor.featurizer.fb.uniform_(0.01, 0.1)
    model.preprocessor.featurizer.window.copy_(
        torch.hann_window(model.preprocessor.win_length)
    )
    return model


@torch.inference_mode()
def test_streaming_frames_and_flush_match_whole_audio(
    perception_model: AudioPerception,
) -> None:
    waveform = torch.randn(1, SAMPLES_PER_FRAME * 20) / 100
    stream = StreamingPerception(perception_model)
    frames = [stream.push(samples) for samples in waveform[0].split(SAMPLES_PER_FRAME)]
    frames.append(stream.flush())
    actual = torch.cat(frames).unsqueeze(0)
    assert actual.shape == (1, 21, 8)
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, perception_model(waveform), rtol=0, atol=0)


@torch.inference_mode()
def test_reset_discards_previous_audio(perception_model: AudioPerception) -> None:
    stream = StreamingPerception(perception_model)
    for samples in torch.randn(10, SAMPLES_PER_FRAME):
        stream.push(samples)
    stream.flush()
    stream.reset()
    fresh_stream = StreamingPerception(perception_model)
    for samples in torch.randn(10, SAMPLES_PER_FRAME) / 100:
        torch.testing.assert_close(
            stream.push(samples), fresh_stream.push(samples), rtol=0, atol=0
        )
    torch.testing.assert_close(stream.flush(), fresh_stream.flush(), rtol=0, atol=0)


@torch.inference_mode()
def test_stream_owns_history_when_caller_reuses_input(
    perception_model: AudioPerception,
) -> None:
    reused_input_stream = StreamingPerception(perception_model)
    reference_stream = StreamingPerception(perception_model)
    input_buffer = torch.empty(SAMPLES_PER_FRAME)
    for samples in torch.randn(10, SAMPLES_PER_FRAME) / 100:
        input_buffer.copy_(samples)
        torch.testing.assert_close(
            reused_input_stream.push(input_buffer),
            reference_stream.push(samples),
            rtol=0,
            atol=0,
        )
