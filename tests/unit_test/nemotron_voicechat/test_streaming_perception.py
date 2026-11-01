import threading
from unittest.mock import Mock

import pytest
import torch

from sglang_omni.models.nemotron_voicechat import stages
from sglang_omni.models.nemotron_voicechat.conformer import (
    SAMPLES_PER_FRAME,
    AudioPerception,
    GraphPerception,
    StreamingPerception,
)
from sglang_omni.models.nemotron_voicechat.payload_types import NemotronVoiceChatState
from sglang_omni.proto.request import OmniRequest, StagePayload

REQUIRES_CUDA = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA graph requires a GPU"
)
STREAM_CASES = [
    pytest.param("cpu", StreamingPerception, id="cpu-StreamingPerception"),
    pytest.param(
        "cuda", StreamingPerception, marks=REQUIRES_CUDA, id="cuda-StreamingPerception"
    ),
    pytest.param(
        "cuda", GraphPerception, marks=REQUIRES_CUDA, id="cuda-GraphPerception"
    ),
]


@pytest.fixture(params=["cpu", pytest.param("cuda", marks=REQUIRES_CUDA)])
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


@pytest.mark.parametrize(
    ("execution_device", "stream_class"), STREAM_CASES, indirect=["execution_device"]
)
@torch.inference_mode()
def test_stream_owns_history_when_caller_reuses_input(
    perception_model: AudioPerception, stream_class: type[StreamingPerception]
) -> None:
    reused_input_stream = stream_class(perception_model, batch_size=1)
    reference_stream = StreamingPerception(perception_model, batch_size=1)
    input_buffer = torch.empty(
        1, SAMPLES_PER_FRAME, device=perception_model.proj.weight.device
    )
    for samples in torch.randn(10, 1, SAMPLES_PER_FRAME) / 100:
        input_buffer.copy_(samples)
        torch.testing.assert_close(
            reused_input_stream.push(input_buffer),
            reference_stream.push(samples),
            rtol=0,
            atol=0,
        )


@pytest.mark.parametrize(
    ("execution_device", "stream_class"), STREAM_CASES, indirect=["execution_device"]
)
@torch.inference_mode()
def test_stream_matches_fresh_encoding_across_requests(
    perception_model: AudioPerception, stream_class: type[StreamingPerception]
) -> None:
    stream = stream_class(perception_model, batch_size=1)
    for frame_count in (1, 20, 2, 12):
        waveform = torch.randn(SAMPLES_PER_FRAME * frame_count) / 100
        (expected,) = perception_model([waveform])
        (actual,) = perception_model([waveform], stream=stream)
        assert actual.shape == (frame_count + 1, 8)
        assert torch.isfinite(actual).all()
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize(
    ("execution_device", "stream_class"), STREAM_CASES, indirect=["execution_device"]
)
@pytest.mark.parametrize("batch_size", [4, 5])
@torch.inference_mode()
def test_batched_encoding_matches_each_clip_encoded_alone(
    monkeypatch: pytest.MonkeyPatch,
    perception_model: AudioPerception,
    stream_class: type[StreamingPerception],
    batch_size: int,
) -> None:
    # note (Xinhao Tan): under TF32 the conv result depends on the batch size.
    monkeypatch.setattr(torch.backends.cudnn, "allow_tf32", False)
    waveforms = [
        torch.randn(SAMPLES_PER_FRAME * frame_count) / 100
        for frame_count in (1, 20, 2, 12)
    ]
    batched = perception_model(
        waveforms, stream=stream_class(perception_model, batch_size=batch_size)
    )
    for waveform, rows_TD in zip(waveforms, batched, strict=True):
        (alone_TD,) = perception_model([waveform])
        assert rows_TD.shape == alone_TD.shape
        torch.testing.assert_close(
            rows_TD, alone_TD, rtol=0, atol=1e-5 * alone_TD.abs().max().item()
        )


@pytest.mark.parametrize(
    ("execution_device", "stream_class"), STREAM_CASES, indirect=["execution_device"]
)
@torch.inference_mode()
def test_a_clip_encodes_the_same_whatever_shares_its_batch(
    perception_model: AudioPerception, stream_class: type[StreamingPerception]
) -> None:
    stream = stream_class(perception_model, batch_size=3)
    kept = torch.randn(SAMPLES_PER_FRAME * 12) / 100
    first_rows, *_ = perception_model(
        [kept, torch.randn(SAMPLES_PER_FRAME * 20) / 100], stream=stream
    )
    second_rows, *_ = perception_model(
        [
            kept,
            torch.randn(SAMPLES_PER_FRAME * 3) / 100,
            torch.zeros(SAMPLES_PER_FRAME),
        ],
        stream=stream,
    )
    torch.testing.assert_close(first_rows, second_rows, rtol=0, atol=0)


@pytest.mark.parametrize(
    "execution_device", [pytest.param("cuda", marks=REQUIRES_CUDA)], indirect=True
)
@torch.inference_mode()
def test_graph_capture_lets_other_threads_use_cuda(
    monkeypatch: pytest.MonkeyPatch, perception_model: AudioPerception
) -> None:
    stream = GraphPerception(perception_model, batch_size=2)
    encode = stream.encode
    other_thread_errors: list[RuntimeError] = []

    def use_cuda_in_other_thread() -> None:
        try:
            torch.ones(1, device=stream.device).item()
        except RuntimeError as error:
            other_thread_errors.append(error)

    def encode_while_other_thread_uses_cuda(mel_BTM: torch.Tensor) -> torch.Tensor:
        if torch.cuda.is_current_stream_capturing():
            worker = threading.Thread(target=use_cuda_in_other_thread)
            worker.start()
            worker.join()
        else:
            pass
        return encode(mel_BTM)

    monkeypatch.setattr(stream, "encode", encode_while_other_thread_uses_cuda)
    stream.push(torch.zeros(2, SAMPLES_PER_FRAME, device=stream.device))
    assert other_thread_errors == []


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
        expected = perception_model([waveform])[0].cpu()
        payload = StagePayload(
            str(request_index),
            OmniRequest(None),
            NemotronVoiceChatState(waveform=waveform, num_frames=20).to_dict(),
        )
        result = scheduler.fn(payload)
        actual = NemotronVoiceChatState.from_dict(result.data).acoustic_frames
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("enable_cuda_graph", [True, False])
@torch.inference_mode()
def test_perception_factory_encodes_a_batch_like_each_request_alone(
    monkeypatch: pytest.MonkeyPatch,
    perception_model: AudioPerception,
    execution_device: torch.device,
    enable_cuda_graph: bool,
) -> None:
    # note (Xinhao Tan): under TF32 the conv result depends on the batch size.
    monkeypatch.setattr(torch.backends.cudnn, "allow_tf32", False)
    monkeypatch.setattr(stages, "perception_config", Mock(return_value={}))
    monkeypatch.setattr(stages, "AudioPerception", Mock(return_value=perception_model))
    monkeypatch.setattr(stages, "load_module", Mock())
    scheduler = stages.create_perception_executor(
        "test-checkpoint",
        device=execution_device.type,
        enable_cuda_graph=enable_cuda_graph,
    )
    frame_counts = (1, 20, 2, 12, 5)
    waveforms = [torch.randn(SAMPLES_PER_FRAME * count) / 100 for count in frame_counts]
    payloads = [
        StagePayload(
            str(request_index),
            OmniRequest(None),
            NemotronVoiceChatState(waveform=waveform, num_frames=count).to_dict(),
        )
        for request_index, (waveform, count) in enumerate(zip(waveforms, frame_counts))
    ]
    results = scheduler.batch_fn(payloads)
    for waveform, result in zip(waveforms, results, strict=True):
        expected = perception_model([waveform])[0].cpu()
        actual = NemotronVoiceChatState.from_dict(result.data).acoustic_frames
        torch.testing.assert_close(
            actual, expected, rtol=0, atol=1e-5 * expected.abs().max().item()
        )
