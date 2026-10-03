# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest
import torch

from sglang_omni.models.fun_cosyvoice3 import stages
from sglang_omni.models.fun_cosyvoice3.config import (
    FUN_COSYVOICE3_DEFAULT_PREFIX_CUDA_GRAPH_CAPTURE_SHAPES,
)
from sglang_omni.models.fun_cosyvoice3.prefix_cuda_graph import (
    PrefixCudaGraphCache,
    PrefixCudaGraphEnvelope,
    prefix_cuda_graph_envelopes_from_capture_shapes,
    resolve_prefix_cuda_graph_max_slack,
    route_prefix_cuda_graph_envelope,
)


def test_prefix_cuda_graph_capture_requires_input_factory() -> None:
    parameter = inspect.signature(PrefixCudaGraphCache.capture).parameters[
        "capture_input_factory"
    ]
    assert parameter.default is inspect.Parameter.empty


def test_prepare_prefix_cuda_graph_capture_inputs_uses_warmup_conditioning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parameter = torch.nn.Parameter(torch.zeros(1))
    fake_flow = SimpleNamespace(
        token_mel_ratio=2,
        pre_lookahead_len=1,
        output_size=2,
        spk_embed_affine_layer=torch.nn.Linear(3, 2),
        parameters=lambda: iter((parameter,)),
    )
    fake_flow.flow = fake_flow

    warmup_token_counts: list[int] = []

    def make_warmup_flow_input(token_count: int) -> stages.FlowBatchInput:
        warmup_token_counts.append(token_count)
        return stages.FlowBatchInput(
            token=torch.full((1, token_count), 3, dtype=torch.int32),
            prompt_token=torch.full((1, 1), 2, dtype=torch.int32),
            prompt_feat=torch.full((1, 2, 2), 4.0),
            embedding=torch.full((1, 3), 5.0),
        )

    scheduler = SimpleNamespace(
        token_hop_len=1,
        make_warmup_flow_input=make_warmup_flow_input,
    )
    packed_batches: list[stages.PackedFlowBatch] = []
    original_pack_flow_inputs = stages.pack_flow_inputs

    def recording_pack_flow_inputs(
        flow_model: stages.FunCosyVoice3Flow,
        inputs: list[stages.FlowBatchInput],
    ) -> stages.PackedFlowBatch:
        packed = original_pack_flow_inputs(flow_model, inputs)
        packed_batches.append(packed)
        return packed

    monkeypatch.setattr(stages, "pack_flow_inputs", recording_pack_flow_inputs)

    token_condition = torch.arange(1, 25, dtype=torch.float32).reshape(2, 2, 6)
    noisy_mel = token_condition + 100
    prompt_mel = token_condition + 200
    speaker_embedding = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    time_span = torch.tensor([0.0, 0.5, 1.0])
    finalize_values: list[bool] = []

    def fake_prepare_flow_conditioning(
        flow_model: stages.FunCosyVoice3Flow,
        packed: stages.PackedFlowBatch,
        *,
        finalize: bool,
    ) -> stages.FlowConditioning:
        del flow_model, packed
        finalize_values.append(finalize)
        return stages.FlowConditioning(
            token_condition=token_condition,
            mel_lengths=(4, 6),
            speaker_embedding=speaker_embedding,
            prompt_mel=prompt_mel,
            noisy_mel=noisy_mel,
            time_span=time_span,
        )

    monkeypatch.setattr(
        stages, "prepare_flow_conditioning", fake_prepare_flow_conditioning
    )

    capture_inputs = stages.prepare_prefix_cuda_graph_capture_inputs(
        fake_flow,
        scheduler,
        (4, 6),
        device=torch.device("cpu"),
    )

    assert warmup_token_counts == [2, 3]
    assert len(packed_batches) == 1
    assert packed_batches[0].target_token_lengths == (2, 3)
    assert packed_batches[0].prompt_mel_lengths == (2, 2)
    assert finalize_values == [False]
    assert capture_inputs[0].shape == (1, 10, 2)
    assert capture_inputs[2].shape == (1, 10, 2)
    assert capture_inputs[4].shape == (1, 10, 2)
    assert torch.equal(capture_inputs[1], time_span)
    assert torch.equal(capture_inputs[3], speaker_embedding)
    assert torch.equal(
        capture_inputs[2],
        torch.cat(
            (
                token_condition[0, :, :4].transpose(0, 1),
                token_condition[1, :, :6].transpose(0, 1),
            ),
            dim=0,
        ).unsqueeze(0),
    )
    assert all(torch.count_nonzero(value) > 0 for value in capture_inputs)


def test_default_prefix_cuda_graph_capture_shapes_are_frozen_and_valid() -> None:
    expected_capture_shapes = (
        (1, 300, 300, 3072, (300,)),
        (1, 450, 450, 512, (450,)),
        (3, 400, 200, 1536, (200, 100, 100)),
        (4, 900, 400, 2048, (400, 200, 150, 150)),
        (2, 500, 350, 2048, (350, 150)),
        (3, 650, 350, 3072, (350, 150, 150)),
        (3, 1100, 400, 512, (400, 350, 350)),
        (1, 100, 100, 512, (100,)),
        (2, 200, 100, 512, (100, 100)),
        (3, 500, 300, 1024, (300, 100, 100)),
        (4, 1200, 400, 1024, (400, 300, 250, 250)),
        (3, 850, 350, 1024, (350, 250, 250)),
        (5, 1150, 350, 512, (350, 200, 200, 200, 200)),
        (8, 1700, 300, 1536, (300, 300, 250, 200, 200, 150, 150, 150)),
        (2, 600, 300, 512, (300, 300)),
        (4, 700, 300, 1024, (300, 150, 150, 100)),
        (6, 1450, 400, 2048, (400, 250, 200, 200, 200, 200)),
    )
    assert FUN_COSYVOICE3_DEFAULT_PREFIX_CUDA_GRAPH_CAPTURE_SHAPES == (
        expected_capture_shapes
    )
    stages.verify_prefix_cuda_graph_capture_shapes(
        FUN_COSYVOICE3_DEFAULT_PREFIX_CUDA_GRAPH_CAPTURE_SHAPES,
        chunk_frames=50,
    )
    envelopes = prefix_cuda_graph_envelopes_from_capture_shapes(
        FUN_COSYVOICE3_DEFAULT_PREFIX_CUDA_GRAPH_CAPTURE_SHAPES
    )
    assert len(envelopes) == 17
    names = [envelope.name for envelope in envelopes]
    assert len(set(names)) == len(names)
    assert names == [
        "B1-N300-M300-E3072",
        "B1-N450-M450-E512",
        "B3-N400-M200-E1536",
        "B4-N900-M400-E2048",
        "B2-N500-M350-E2048",
        "B3-N650-M350-E3072",
        "B3-N1100-M400-E512",
        "B1-N100-M100-E512",
        "B2-N200-M100-E512",
        "B3-N500-M300-E1024",
        "B4-N1200-M400-E1024",
        "B3-N850-M350-E1024",
        "B5-N1150-M350-E512",
        "B8-N1700-M300-E1536",
        "B2-N600-M300-E512",
        "B4-N700-M300-E1024",
        "B6-N1450-M400-E2048",
    ]
    assert (
        tuple(
            (
                envelope.batch_size,
                envelope.new_frame_count,
                envelope.max_new_frame_count,
                envelope.max_total_frame_count,
                envelope.capture_row_new_frames,
            )
            for envelope in envelopes
        )
        == expected_capture_shapes
    )


@pytest.mark.parametrize(
    ("capture_shapes", "message"),
    [
        (((1, 100, 100, 512, (75,)),), "multiples"),
        (((1, 100, 100, 512, (50,)),), "sum to N"),
        (((2, 200, 100, 512, (100,)),), "equal B"),
    ],
)
def test_verify_prefix_cuda_graph_capture_shapes_rejects_invalid_overrides(
    capture_shapes: tuple[tuple[int, int, int, int, tuple[int, ...]], ...],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        stages.verify_prefix_cuda_graph_capture_shapes(
            capture_shapes,
            chunk_frames=50,
        )


def test_prefix_cuda_graph_router_uses_supplied_chunk_frames() -> None:
    envelope = PrefixCudaGraphEnvelope("B1-N40-M40-E80", 1, 40, 40, 80, (40,))
    assert (
        route_prefix_cuda_graph_envelope(
            new_frame_counts=[40],
            total_frame_counts=[40],
            envelopes=(envelope,),
            chunk_frames=40,
            max_slack_frames=40,
        )
        == envelope
    )
    assert (
        route_prefix_cuda_graph_envelope(
            new_frame_counts=[50],
            total_frame_counts=[50],
            envelopes=(envelope,),
            chunk_frames=40,
            max_slack_frames=40,
        )
        is None
    )


def test_prefix_cuda_graph_shape_validation_uses_supplied_chunk_frames() -> None:
    valid_shapes = ((2, 120, 80, 160, (80, 40)),)
    assert (
        stages.verify_prefix_cuda_graph_capture_shapes(
            valid_shapes,
            chunk_frames=40,
        )
        == valid_shapes
    )
    with pytest.raises(ValueError, match="chunk size 40"):
        stages.verify_prefix_cuda_graph_capture_shapes(
            ((2, 120, 100, 160, (100, 20)),),
            chunk_frames=40,
        )


def test_prefix_cuda_graph_router_uses_configured_max_slack() -> None:
    envelope = PrefixCudaGraphEnvelope("B1-N200-M200-E200", 1, 200, 200, 200, (200,))
    assert (
        route_prefix_cuda_graph_envelope(
            new_frame_counts=[100],
            total_frame_counts=[100],
            envelopes=(envelope,),
            chunk_frames=50,
            max_slack_frames=50,
        )
        is None
    )
    assert (
        route_prefix_cuda_graph_envelope(
            new_frame_counts=[100],
            total_frame_counts=[100],
            envelopes=(envelope,),
            chunk_frames=50,
            max_slack_frames=100,
        )
        == envelope
    )


def test_prefix_cuda_graph_max_slack_defaults_to_two_chunks() -> None:
    assert resolve_prefix_cuda_graph_max_slack(50, None) == 100
    assert resolve_prefix_cuda_graph_max_slack(50, 50) == 50
    with pytest.raises(ValueError, match="greater than zero"):
        resolve_prefix_cuda_graph_max_slack(50, 0)
    with pytest.raises(ValueError, match="chunk size 50"):
        resolve_prefix_cuda_graph_max_slack(50, 75)


def test_prefix_cuda_graph_routes_physical_envelopes() -> None:
    envelopes = (
        PrefixCudaGraphEnvelope("B1-N100-M100-E512", 1, 100, 100, 512, (100,)),
        PrefixCudaGraphEnvelope("B2-N200-M100-E512", 2, 200, 100, 512, (100, 100)),
    )
    assert (
        route_prefix_cuda_graph_envelope(
            new_frame_counts=[50],
            total_frame_counts=[50],
            envelopes=envelopes,
            chunk_frames=50,
            max_slack_frames=100,
        )
        == envelopes[0]
    )
    assert (
        route_prefix_cuda_graph_envelope(
            new_frame_counts=[50, 50],
            total_frame_counts=[50, 50],
            envelopes=envelopes,
            chunk_frames=50,
            max_slack_frames=100,
        )
        == envelopes[1]
    )
    assert (
        route_prefix_cuda_graph_envelope(
            new_frame_counts=[50, 50],
            total_frame_counts=[50, 50],
            envelopes=(envelopes[0],),
            chunk_frames=50,
            max_slack_frames=100,
        )
        is None
    )
    assert (
        route_prefix_cuda_graph_envelope(
            new_frame_counts=[100, 50],
            total_frame_counts=[100, 50],
            envelopes=envelopes,
            chunk_frames=50,
            max_slack_frames=100,
        )
        == envelopes[1]
    )
    assert (
        route_prefix_cuda_graph_envelope(
            new_frame_counts=[100, 100],
            total_frame_counts=[100, 100],
            envelopes=envelopes,
            chunk_frames=50,
            max_slack_frames=100,
        )
        == envelopes[1]
    )
    assert (
        route_prefix_cuda_graph_envelope(
            new_frame_counts=[100],
            total_frame_counts=[100],
            envelopes=envelopes,
            chunk_frames=50,
            max_slack_frames=100,
        )
        == envelopes[0]
    )
    assert (
        route_prefix_cuda_graph_envelope(
            new_frame_counts=[75],
            total_frame_counts=[75],
            envelopes=envelopes,
            chunk_frames=50,
            max_slack_frames=100,
        )
        is None
    )
    assert (
        route_prefix_cuda_graph_envelope(
            new_frame_counts=[50],
            total_frame_counts=[50],
            envelopes=(
                PrefixCudaGraphEnvelope("B1-N200-M200-E512", 1, 200, 200, 512, (200,)),
            ),
            chunk_frames=50,
            max_slack_frames=100,
        )
        is None
    )
    assert (
        route_prefix_cuda_graph_envelope(
            new_frame_counts=[50],
            total_frame_counts=[50],
            envelopes=(
                PrefixCudaGraphEnvelope("B1-N125-M100-E512", 1, 125, 100, 512, (125,)),
            ),
            chunk_frames=50,
            max_slack_frames=100,
        )
        is None
    )
    assert (
        route_prefix_cuda_graph_envelope(
            new_frame_counts=[150],
            total_frame_counts=[150],
            envelopes=(
                PrefixCudaGraphEnvelope("B1-N200-M100-E512", 1, 200, 100, 512, (200,)),
            ),
            chunk_frames=50,
            max_slack_frames=100,
        )
        is None
    )
    assert (
        route_prefix_cuda_graph_envelope(
            new_frame_counts=[50],
            total_frame_counts=[600],
            envelopes=envelopes,
            chunk_frames=50,
            max_slack_frames=100,
        )
        is None
    )
    assert (
        route_prefix_cuda_graph_envelope(
            new_frame_counts=[100, 100],
            total_frame_counts=[100, 100],
            envelopes=(
                PrefixCudaGraphEnvelope(
                    "B2-N200-M100-E512", 2, 200, 100, 512, (100, 100)
                ),
                PrefixCudaGraphEnvelope(
                    "B2-N300-M150-E1024", 2, 300, 150, 1024, (150, 150)
                ),
            ),
            chunk_frames=50,
            max_slack_frames=100,
        ).name
        == "B2-N200-M100-E512"
    )


@pytest.mark.parametrize(
    ("kwargs", "exception", "message"),
    [
        (
            {"enable_dit_torch_compile": False, "flow_prefix_cache_gb": 24.0},
            ValueError,
            "enable_dit_torch_compile",
        ),
        (
            {"enable_dit_torch_compile": True, "flow_prefix_cache_gb": 0.0},
            ValueError,
            "flow_prefix_cache_gb",
        ),
        (
            {"enable_dit_torch_compile": True, "flow_prefix_cache_gb": 24.0},
            RuntimeError,
            "available CUDA",
        ),
    ],
)
def test_prefix_cuda_graph_explicit_enable_requires_cuda_prerequisites(
    monkeypatch: pytest.MonkeyPatch,
    kwargs: dict[str, bool | float],
    exception: type[Exception],
    message: str,
) -> None:
    monkeypatch.setattr(
        stages, "resolve_concrete_device", lambda device, gpu_id: torch.device("cpu")
    )

    with pytest.raises(exception, match=message):
        stages.create_vocoder_executor(
            "model",
            device="cpu",
            enable_flow_prefix_cuda_graph=True,
            **kwargs,
        )
