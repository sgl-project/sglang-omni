from unittest.mock import Mock

import pytest
import torch

from sglang_omni.models.nemotron_voicechat import stages
from sglang_omni.models.nemotron_voicechat.conformer import (
    SAMPLES_PER_FRAME,
    AudioPerception,
    StreamingPerception,
)
from sglang_omni.models.nemotron_voicechat.payload_types import NemotronVoiceChatState
from sglang_omni.models.nemotron_voicechat.perception_graph import GraphPerception
from sglang_omni.proto.request import OmniRequest, StagePayload


@pytest.fixture(
    params=[
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="CUDA graph requires a GPU"
            ),
        ),
    ]
)
def execution_device(request: pytest.FixtureRequest) -> torch.device:
    return torch.device(request.param)


@pytest.fixture(params=[0, 2, 7])
def perception_model(
    request: pytest.FixtureRequest, execution_device: torch.device
) -> AudioPerception:
    torch.manual_seed(2188)
    model = (
        AudioPerception(
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
        )
        .to(execution_device)
        .eval()
    )
    model.preprocessor.featurizer.fb.uniform_(0.01, 0.1)
    model.preprocessor.featurizer.window.copy_(
        torch.hann_window(model.preprocessor.win_length, device=execution_device)
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


@pytest.mark.parametrize("stream_class", [StreamingPerception, GraphPerception])
@torch.inference_mode()
def test_reset_discards_previous_audio(
    perception_model: AudioPerception, stream_class: type[StreamingPerception]
) -> None:
    stream = stream_class(perception_model)
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


@pytest.mark.parametrize("stream_class", [StreamingPerception, GraphPerception])
@torch.inference_mode()
def test_stream_owns_history_when_caller_reuses_input(
    perception_model: AudioPerception, stream_class: type[StreamingPerception]
) -> None:
    reused_input_stream = stream_class(perception_model)
    reference_stream = StreamingPerception(perception_model)
    input_buffer = torch.empty(
        SAMPLES_PER_FRAME, device=perception_model.proj.weight.device
    )
    for samples in torch.randn(10, SAMPLES_PER_FRAME) / 100:
        input_buffer.copy_(samples)
        torch.testing.assert_close(
            reused_input_stream.push(input_buffer),
            reference_stream.push(samples),
            rtol=0,
            atol=0,
        )


@torch.inference_mode()
def test_graph_whole_audio_matches_eager_across_requests(
    perception_model: AudioPerception,
) -> None:
    stream = GraphPerception(perception_model)
    for frame_count in (1, 20, 2, 12):
        waveform = torch.randn(1, SAMPLES_PER_FRAME * frame_count) / 100
        expected = perception_model(waveform)
        actual = perception_model(waveform, stream=stream)
        assert actual.shape == (1, frame_count + 1, 8)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@torch.inference_mode()
def test_graph_outputs_survive_later_replays(perception_model: AudioPerception) -> None:
    stream = GraphPerception(perception_model)
    reference_stream = StreamingPerception(perception_model)
    outputs: list[torch.Tensor] = []
    expected_outputs: list[torch.Tensor] = []
    for samples in torch.randn(20, SAMPLES_PER_FRAME) / 100:
        outputs.append(stream.push(samples))
        expected_outputs.append(reference_stream.push(samples).clone())
    for actual, expected in zip(outputs, expected_outputs, strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("enable_cuda_graph", [True, False])
@torch.inference_mode()
def test_perception_factory_isolates_consecutive_requests(
    monkeypatch: pytest.MonkeyPatch,
    perception_model: AudioPerception,
    execution_device: torch.device,
    enable_cuda_graph: bool,
) -> None:
    monkeypatch.setattr(stages, "perception_config", Mock(return_value={}))
    monkeypatch.setattr(stages, "AudioPerception", Mock(return_value=perception_model))
    monkeypatch.setattr(stages, "load_module", Mock())
    scheduler = stages.create_perception_executor(
        "test-checkpoint",
        device=execution_device.type,
        enable_cuda_graph=enable_cuda_graph,
    )
    for request_index in range(2):
        waveform = torch.randn(SAMPLES_PER_FRAME * 20) / 100
        expected = perception_model(waveform.unsqueeze(0))[0].cpu()
        payload = StagePayload(
            str(request_index),
            OmniRequest(None),
            NemotronVoiceChatState(waveform=waveform, num_frames=20).to_dict(),
        )
        result = scheduler.fn(payload)
        actual = NemotronVoiceChatState.from_dict(result.data).acoustic_frames
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
