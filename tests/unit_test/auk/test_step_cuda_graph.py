# SPDX-License-Identifier: Apache-2.0
"""Shape declaration, lookup, and the eager fallbacks the step graph runner takes."""

from dataclasses import replace
from unittest.mock import Mock, patch

import pytest
import torch

from sglang_omni.models.auk.dit import AuKDit
from sglang_omni.models.auk.flow_matching import AuKFlowMatching, AuKSampleItem
from sglang_omni.models.auk.step_cuda_graph import (
    DEFAULT_CAPTURE_SHAPES,
    AuKGraphShape,
    AuKStepCudaGraphRunner,
    build_step_graph_runner,
    verify_capture_shapes,
)


def runner(*shapes: tuple[int, int, int, int]) -> AuKStepCudaGraphRunner:
    """A runner on the CPU with the given shapes declared and all captured."""
    built = AuKStepCudaGraphRunner(
        backend=Mock(),
        device=torch.device("cpu"),
        capture_shapes=shapes or DEFAULT_CAPTURE_SHAPES,
    )
    built.ready.update(built.declared)
    return built


def test_declared_shapes_are_deduplicated_and_ordered_by_padded_rows():
    verified = verify_capture_shapes(
        [(1, 448, 0, 192), (1, 192, 0, 192), (2, 192, 0, 192), (1, 192, 0, 192)]
    )
    assert verified == (
        AuKGraphShape(1, 192, 0, 192),
        AuKGraphShape(1, 448, 0, 192),
        AuKGraphShape(2, 192, 0, 192),
    )


@pytest.mark.parametrize(
    "shape", [(0, 192, 0, 192), (1, 0, 0, 192), (1, 192, -1, 192), (1, 192, 0, 0)]
)
def test_a_shape_with_no_rows_to_pad_to_is_refused(shape):
    with pytest.raises(ValueError, match="capture shapes need"):
        verify_capture_shapes([shape])


def test_an_empty_shape_list_is_refused():
    with pytest.raises(ValueError, match="must not be empty"):
        verify_capture_shapes([])


def test_fit_takes_the_cheapest_shape_that_covers_every_axis():
    fitted = runner((1, 192, 0, 192), (1, 448, 0, 192), (1, 448, 320, 384))
    assert fitted.fit(frames=100, ref=0, text=10, batch=1) == AuKGraphShape(
        1, 192, 0, 192
    )
    assert fitted.fit(frames=300, ref=0, text=10, batch=1) == AuKGraphShape(
        1, 448, 0, 192
    )
    # Cheapest by total rows, so a wider text count skips the narrower rungs.
    assert fitted.fit(frames=100, ref=0, text=300, batch=1) == AuKGraphShape(
        1, 448, 320, 384
    )


@pytest.mark.parametrize(
    "frames,ref,text,batch",
    [
        (500, 0, 192, 1),  # more frames than any rung
        (192, 400, 192, 1),  # a longer reference
        (192, 0, 500, 1),  # more text tokens
        (192, 0, 192, 3),  # a batch size never declared
        # Covered on every other axis, but only by a shape of another batch.
        (192, 300, 300, 1),
    ],
)
def test_fit_refuses_a_batch_wider_than_the_declared_shapes(frames, ref, text, batch):
    fitted = runner((1, 192, 0, 192), (2, 192, 320, 384))
    assert fitted.fit(frames=frames, ref=ref, text=text, batch=batch) is None
    assert fitted.pad_lengths(frames=frames, ref=ref, text=text, batch=batch) is None


def test_a_shape_that_never_captured_is_not_fitted():
    fitted = runner((1, 192, 0, 192))
    fitted.ready.clear()
    assert fitted.fit(frames=100, ref=0, text=10, batch=1) is None


def test_bind_outside_capture_declared_leaves_the_step_eager():
    """A request must never pay a capture: it synchronizes the whole device."""
    fitted = runner((1, 192, 0, 192))
    step = Mock()
    replay = fitted.bind(
        step,
        dict(text=torch.zeros(1, 192, 8)),
        x=torch.zeros(1, 192, 4),
        time=torch.zeros(()),
    )

    assert replay is None
    assert fitted.graphs == {}
    step.assert_not_called()


def test_the_graph_key_separates_shapes_dtypes_and_baked_values():
    keyed = runner()
    inputs = dict(text=torch.zeros(1, 192, 8), cache=False)
    x = torch.zeros(1, 192, 4)
    base = keyed.graph_key(inputs, x, (2.0,))

    assert keyed.graph_key(inputs, x, (2.0,)) == base
    assert keyed.graph_key(inputs, x, (0.0,)) != base
    assert keyed.graph_key(dict(inputs, cache=True), x, (2.0,)) != base
    assert keyed.graph_key(inputs, torch.zeros(1, 320, 4), (2.0,)) != base
    assert keyed.graph_key(inputs, x.to(torch.bfloat16), (2.0,)) != base


def test_a_warmup_is_required_before_a_capture_records():
    with pytest.raises(ValueError, match="warmup iteration"):
        AuKStepCudaGraphRunner(
            backend=Mock(), device=torch.device("cpu"), warmup_iters=0
        )


def test_a_shape_that_fails_to_capture_is_left_out_rather_than_raising(caplog):
    declared = runner((1, 192, 0, 192))
    declared.ready.clear()

    def run_trajectory(shape):
        raise RuntimeError("out of memory")

    declared.capture_declared(run_trajectory)

    assert declared.ready == set()
    assert "will run eager" in caplog.text


@pytest.mark.accelerator
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("weight_dtype", [torch.float32, torch.bfloat16])
def test_graph_replay_matches_eager_fallback_for_strided_reference(
    weight_dtype: torch.dtype, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A strided reference must produce the same sampled latents on replay and fallback."""
    torch.manual_seed(42)
    flow = (
        AuKFlowMatching(
            AuKDit(
                dim=32,
                heads=2,
                dim_head=16,
                latent_dim=8,
                text_hidden_dim=16,
                num_layers=1,
                num_single_layers=1,
            ),
            num_llm_layers=2,
        )
        .cuda()
        .to(weight_dtype)
        .eval()
        .requires_grad_(False)
    )
    for parameter in flow.parameters():
        torch.nn.init.uniform_(parameter, -0.2, 0.2)
    sample_item = AuKSampleItem(
        conditioning=torch.randn(8, 16, device="cuda"),
        text_mask=torch.ones(8, dtype=torch.bool, device="cuda"),
        target_frames=24,
        ref_latent=torch.randn(8, 16, device="cuda").transpose(0, 1),
        seed=1234,
        ref_length=16,
    )
    graph_runner = build_step_graph_runner(torch.device("cuda:0"), [(1, 32, 16, 8)])
    assert graph_runner is not None
    replay_count = 0
    original_replay = torch.cuda.CUDAGraph.replay

    def replay(graph: torch.cuda.CUDAGraph) -> None:
        nonlocal replay_count
        original_replay(graph)
        replay_count += 1

    with (
        torch.inference_mode(),
        torch.autocast(
            "cuda", dtype=torch.bfloat16, enabled=weight_dtype == torch.float32
        ),
    ):
        graph_runner.capture_declared(
            lambda shape: flow.sample_batch(
                [
                    replace(
                        sample_item,
                        target_frames=shape.frames,
                        ref_latent=sample_item.ref_latent.contiguous(),
                    )
                ],
                steps=2,
                cfg_strength=2.0,
                step_graph=graph_runner,
            )
        )
        monkeypatch.setattr(torch.cuda.CUDAGraph, "replay", replay)
        captured = flow.sample_batch(
            [sample_item], steps=2, cfg_strength=2.0, step_graph=graph_runner
        )
        assert replay_count > 0
        with patch.object(graph_runner, "bind", return_value=None):
            expected = flow.sample_batch(
                [sample_item], steps=2, cfg_strength=2.0, step_graph=graph_runner
            )
    torch.testing.assert_close(captured[0], expected[0], rtol=0, atol=0)
