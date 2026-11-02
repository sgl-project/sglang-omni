from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from sglang_omni.models.ming_tts import audio_decode, audio_upsample
from sglang_omni.models.ming_tts.audio_decode import (
    AudioVAEFixedStreamingStateBank,
    AudioVAEFixedStreamingTransition,
)

CUDA_AVAILABLE = torch.cuda.is_available()
CUDA_BF16_AVAILABLE = CUDA_AVAILABLE and torch.cuda.get_device_capability()[0] >= 8
CUDA_TRITON_AVAILABLE = CUDA_BF16_AVAILABLE and audio_upsample.HAS_TRITON


def make_upsample_transition(
    cuda_device: torch.device, input_dtype: torch.dtype = torch.bfloat16
) -> AudioVAEFixedStreamingTransition:
    capacity = 1
    max_step_latents = 16
    hidden_size = 896
    transition = AudioVAEFixedStreamingTransition.__new__(
        AudioVAEFixedStreamingTransition
    )
    # note(chenye): Set backing fields directly so the fixture avoids loading an AudioVAE.
    transition._capacity = capacity  # noqa: leading-underscore
    transition._max_step_latents = max_step_latents  # noqa: leading-underscore
    transition._device = cuda_device  # noqa: leading-underscore
    transition._input_dtype = input_dtype  # noqa: leading-underscore
    transition.state = AudioVAEFixedStreamingStateBank(
        upsample_pending=torch.zeros(
            (capacity, max_step_latents, hidden_size),
            device=cuda_device,
            dtype=input_dtype,
        ),
        upsample_pending_lengths=torch.zeros(
            capacity, device=cuda_device, dtype=torch.long
        ),
        upsample_left_anchor=torch.zeros(
            (capacity, 1, hidden_size), device=cuda_device, dtype=input_dtype
        ),
        qwen_keys=torch.empty((0,), device=cuda_device),
        qwen_values=torch.empty((0,), device=cuda_device),
        qwen_positions=torch.empty((0,), device=cuda_device, dtype=torch.long),
        istft_audio_overlap=torch.empty((0,), device=cuda_device),
    )
    transition.upsampler = torch.nn.Upsample(
        scale_factor=4,
        mode="linear",
        align_corners=False,
        recompute_scale_factor=None,
    )
    transition.scale_factor = 4
    transition.max_frames = 2 * max_step_latents * transition.scale_factor
    transition.hidden_size = hidden_size
    return transition


@pytest.mark.accelerator
@pytest.mark.parametrize(
    ("input_shape", "is_noncontiguous"),
    [
        ((1, 18, 896), False),
        ((2, 22, 896), True),
        ((8, 34, 896), False),
        ((1, 1, 896), False),
        ((1, 9, 17), False),
    ],
)
@pytest.mark.skipif(
    not CUDA_TRITON_AVAILABLE, reason="requires supported CUDA and Triton"
)
def test_upsample_linear_bth_exactly_matches_pytorch(
    input_shape: tuple[int, int, int], is_noncontiguous: bool
) -> None:
    cuda_device = torch.device("cuda", torch.cuda.current_device())
    batch_size, input_time_steps, hidden_size = input_shape
    storage_time_steps = input_time_steps * (2 if is_noncontiguous else 1)
    timeline_storage = torch.randn(
        (batch_size, storage_time_steps, hidden_size),
        device=cuda_device,
        dtype=torch.bfloat16,
    )
    timeline = timeline_storage[:, ::2, :] if is_noncontiguous else timeline_storage
    assert timeline.is_contiguous() is (not is_noncontiguous)
    expected_bth = F.interpolate(
        timeline.transpose(1, 2),
        scale_factor=4,
        mode="linear",
        align_corners=False,
    ).transpose(1, 2)

    upsampled_bth = audio_upsample.upsample_linear_bth(timeline, scale_factor=4)

    assert upsampled_bth.shape == expected_bth.shape
    assert upsampled_bth.is_contiguous()
    assert upsampled_bth.stride() == (
        input_time_steps * 4 * hidden_size,
        hidden_size,
        1,
    )
    assert torch.equal(upsampled_bth, expected_bth)


@pytest.mark.accelerator
@pytest.mark.skipif(
    not CUDA_TRITON_AVAILABLE, reason="requires supported CUDA and Triton"
)
def test_upsample_linear_bth_cuda_graph_replay_tracks_input_updates() -> None:
    cuda_device = torch.device("cuda", torch.cuda.current_device())
    timeline = torch.randn((8, 34, 896), device=cuda_device, dtype=torch.bfloat16)
    audio_upsample.upsample_linear_bth(timeline, scale_factor=4)
    cuda_graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(cuda_graph):
        graph_output_bth = audio_upsample.upsample_linear_bth(timeline, scale_factor=4)

    cuda_graph.replay()
    torch.cuda.synchronize(cuda_device)
    expected_bth = F.interpolate(
        timeline.transpose(1, 2),
        scale_factor=4,
        mode="linear",
        align_corners=False,
    ).transpose(1, 2)
    assert graph_output_bth.is_contiguous()
    assert torch.equal(graph_output_bth, expected_bth)

    timeline.add_(0.25)
    cuda_graph.replay()
    torch.cuda.synchronize(cuda_device)
    expected_bth = F.interpolate(
        timeline.transpose(1, 2),
        scale_factor=4,
        mode="linear",
        align_corners=False,
    ).transpose(1, 2)
    assert torch.equal(graph_output_bth, expected_bth)


@pytest.mark.accelerator
@pytest.mark.skipif(
    not CUDA_TRITON_AVAILABLE, reason="requires supported CUDA and Triton"
)
def test_streaming_upsampling_dispatches_and_preserves_chunk_lengths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cuda_device = torch.device("cuda", torch.cuda.current_device())
    transition = make_upsample_transition(cuda_device)
    triton_scale_factors: list[int] = []
    original_upsample = audio_decode.upsample_linear_bth

    def record_triton_call(
        timeline: torch.Tensor, *, scale_factor: int
    ) -> torch.Tensor:
        triton_scale_factors.append(scale_factor)
        upsampled = original_upsample(timeline, scale_factor=scale_factor)
        native = F.interpolate(
            timeline.transpose(1, 2),
            scale_factor=scale_factor,
            mode="linear",
            align_corners=False,
        ).transpose(1, 2)
        assert torch.equal(upsampled, native)
        return upsampled

    monkeypatch.setattr(audio_decode, "upsample_linear_bth", record_triton_call)

    for current_length, terminal, expected_frame_length in (
        (8, False, 0),
        (16, False, 32),
        (8, True, 96),
    ):
        current = torch.zeros((1, 16, 896), device=cuda_device, dtype=torch.bfloat16)
        current[:, :current_length] = torch.randn(
            (1, current_length, 896), device=cuda_device, dtype=torch.bfloat16
        )
        current_lengths = torch.tensor([current_length], device=cuda_device)
        execution_mask = torch.ones((1,), device=cuda_device, dtype=torch.bool)
        terminal_mask = torch.tensor([terminal], device=cuda_device)
        frames, frame_lengths, next_state = transition.upsample(
            current, current_lengths, execution_mask, terminal_mask
        )

        assert frame_lengths.tolist() == [expected_frame_length]
        assert frames.shape == (1, 128, 896)
        assert torch.count_nonzero(frames[:, expected_frame_length:]) == 0
        (
            transition.state.upsample_pending,
            transition.state.upsample_pending_lengths,
            transition.state.upsample_left_anchor,
        ) = next_state

    assert triton_scale_factors == [4, 4, 4]


@pytest.mark.parametrize(
    ("device_type", "input_dtype", "has_triton"),
    [
        pytest.param("cpu", torch.float32, audio_upsample.HAS_TRITON, id="cpu"),
        pytest.param(
            "cuda",
            torch.bfloat16,
            False,
            marks=(
                pytest.mark.accelerator,
                pytest.mark.skipif(
                    not CUDA_BF16_AVAILABLE, reason="requires CUDA BF16"
                ),
            ),
            id="triton-unavailable",
        ),
        pytest.param(
            "cuda",
            torch.float32,
            True,
            marks=(
                pytest.mark.accelerator,
                pytest.mark.skipif(not CUDA_AVAILABLE, reason="requires CUDA"),
            ),
            id="unsupported-dtype",
        ),
    ],
)
def test_streaming_upsampling_uses_native_fallback(
    monkeypatch: pytest.MonkeyPatch,
    device_type: str,
    input_dtype: torch.dtype,
    has_triton: bool,
) -> None:
    device = (
        torch.device("cuda", torch.cuda.current_device())
        if device_type == "cuda"
        else torch.device("cpu")
    )
    transition = make_upsample_transition(device, input_dtype)
    current = torch.randn((1, 16, 896), device=device, dtype=input_dtype)
    current_lengths = torch.tensor([16], device=device)
    execution_mask = torch.ones((1,), device=device, dtype=torch.bool)
    terminal_mask = torch.ones((1,), device=device, dtype=torch.bool)
    native_outputs: list[torch.Tensor] = []
    native_upsample = transition.upsampler.forward
    triton_calls: list[torch.Tensor] = []

    def record_native_call(timeline: torch.Tensor) -> torch.Tensor:
        output = native_upsample(timeline)
        native_outputs.append(output.transpose(1, 2))
        return output

    def record_triton_call(
        timeline: torch.Tensor, *, scale_factor: int
    ) -> torch.Tensor:
        triton_calls.append(timeline)
        return audio_upsample.upsample_linear_bth(timeline, scale_factor=scale_factor)

    monkeypatch.setattr(audio_upsample, "HAS_TRITON", has_triton)
    monkeypatch.setattr(transition.upsampler, "forward", record_native_call)
    monkeypatch.setattr(audio_decode, "upsample_linear_bth", record_triton_call)

    frames, frame_lengths, _ = transition.upsample(
        current, current_lengths, execution_mask, terminal_mask
    )

    assert len(native_outputs) == 1
    assert not triton_calls
    assert frame_lengths.tolist() == [64]
    assert frames.shape == (1, 128, 896)
    assert torch.equal(frames[:, :64], native_outputs[0][:, 4:68].to(torch.float32))
    assert torch.count_nonzero(frames[:, 64:]) == 0
