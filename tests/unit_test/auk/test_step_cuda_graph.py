# SPDX-License-Identifier: Apache-2.0
"""Shape declaration, lookup, and the eager fallbacks the step graph runner takes."""

from contextlib import AbstractContextManager, nullcontext
from unittest.mock import Mock

import pytest
import torch

from sglang_omni.models.auk.dit import AuKDit
from sglang_omni.models.auk.flow_matching import AuKFlowMatching, AuKSampleItem
from sglang_omni.models.auk.stages import warmup_flow
from sglang_omni.models.auk.step_cuda_graph import (
    DEFAULT_CAPTURE_SHAPES,
    AuKGraphShape,
    AuKStepCudaGraphRunner,
    verify_capture_shapes,
)
from sglang_omni.platforms import current_platform


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


class CachingDeviceModule:
    def __init__(self) -> None:
        self.cached_bytes = 3 * 1024**3

    def empty_cache(self) -> None:
        self.cached_bytes = 0

    def mem_get_info(self, device: torch.device) -> tuple[int, int]:
        return 5 * 1024**3 - self.cached_bytes, 24 * 1024**3

    def device(self, device: torch.device) -> AbstractContextManager[None]:
        return nullcontext()


def test_the_headroom_counts_the_blocks_the_capture_releases() -> None:
    declared = runner((1, 192, 0, 192))
    declared.module = CachingDeviceModule()
    declared.capture = Mock(return_value="captured")

    entry = declared.prepare(("key",), Mock(), {}, torch.zeros(1), torch.zeros(1))

    assert entry == "captured"
    assert declared.graphs == {("key",): "captured"}


@pytest.mark.accelerator
@pytest.mark.skipif(not torch.xpu.is_available(), reason="records an XPU graph")
def test_an_xpu_step_graph_replays_each_request_as_padded_eager() -> None:
    device = torch.device("xpu", 0)
    torch.manual_seed(42)
    flow = AuKFlowMatching(
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
    for parameter in flow.parameters():
        torch.nn.init.uniform_(parameter, -0.2, 0.2)
    flow = flow.to(device).eval()
    shape = AuKGraphShape(batch=1, frames=32, ref=8, text=16)

    def sample_request(
        text_tokens: int, target_frames: int, reference_frames: int, seed: int
    ) -> AuKSampleItem:
        return AuKSampleItem(
            torch.randn(text_tokens, 16, device=device),
            torch.ones(text_tokens, dtype=torch.bool, device=device),
            target_frames,
            torch.randn(reference_frames, 8, device=device),
            seed=seed,
            ref_length=reference_frames,
        )

    first = sample_request(10, 20, 6, seed=3)
    second = sample_request(14, 28, 8, seed=5)
    requests = [first, second, first]
    sampling = dict(steps=4, cfg_strength=2.0)
    graphed = AuKStepCudaGraphRunner(
        backend=current_platform.get_device_graph_backend(device),
        device=device,
        capture_shapes=[shape],
        min_free_gb=0,
    )
    padded = AuKStepCudaGraphRunner(
        backend=Mock(), device=device, capture_shapes=[shape]
    )
    padded.ready.add(shape)
    with torch.inference_mode():
        warmup_flow(flow, device, torch.float32, sampling, graphed)
        expected = [
            flow.sample_batch([request], **sampling, step_graph=padded)[0]
            for request in requests
        ]
        replayed = [
            flow.sample_batch([request], **sampling, step_graph=graphed)[0]
            for request in requests
        ]

    assert graphed.ready == {shape}
    assert len(graphed.graphs) == 1
    assert [latent.shape[0] for latent in replayed] == [20, 28, 20]
    for latent, reference in zip(replayed, expected):
        torch.testing.assert_close(latent, reference, rtol=1e-5, atol=1e-6)
