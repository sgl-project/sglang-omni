"""Graph normalization preserves native rounding and eager gradients."""

import importlib.util
import sys
from pathlib import Path

import pytest
import torch

from sglang_omni.modules.ming_audio.flow.modules import RMSNorm
from sglang_omni.modules.ming_audio.flow.normalization import (
    NormalizationDispatch,
    normalization_affine,
    normalize_native_rms,
    normalize_native_strided_rms,
    square_float32,
)


def test_normalization_import_without_triton_keeps_eager_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CPU-only installations must not need Triton to import shared acoustic code."""
    module_path = (
        Path(__file__).resolve().parents[3]
        / "sglang_omni/modules/ming_audio/flow/normalization.py"
    )
    spec = importlib.util.spec_from_file_location(
        "ming_normalization_without_triton", module_path
    )
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    monkeypatch.setitem(sys.modules, "triton", None)
    spec.loader.exec_module(module)
    assert module.HAS_TRITON is False
    hidden_states = torch.tensor([[1.0, 2.0]])
    torch.testing.assert_close(
        module.square_float32(hidden_states), hidden_states.square()
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_eager_normalization_preserves_outputs_and_gradients(
    dtype: torch.dtype,
) -> None:
    torch.manual_seed(42)
    layer = RMSNorm(32, eps=1e-6).to(dtype=dtype)
    layer.native_rms_norm = False
    layer.normalization_dispatch = NormalizationDispatch(
        square_float32=square_float32,
        affine=normalization_affine,
        normalize_contiguous=normalize_native_rms,
    )
    hidden_states = torch.randn(3, 32, dtype=dtype, requires_grad=True)
    reference_input = hidden_states.detach().clone().requires_grad_()
    reference_weight = layer.weight.detach().clone().requires_grad_()
    variance = reference_input.float().pow(2).mean(-1, keepdim=True)
    expected = (reference_input * torch.rsqrt(variance + layer.eps)).to(dtype)
    expected = expected * reference_weight
    actual = layer(hidden_states)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    actual.sum().backward()
    expected.sum().backward()
    torch.testing.assert_close(hidden_states.grad, reference_input.grad, rtol=0, atol=0)
    torch.testing.assert_close(layer.weight.grad, reference_weight.grad, rtol=0, atol=0)


@pytest.mark.accelerator
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("rows", [1, 3, 16, 33, 64])
@pytest.mark.parametrize("strided", [False, True])
def test_graph_normalization_matches_native_reduction_and_owns_output(
    dtype: torch.dtype, rows: int, strided: bool
) -> None:
    if not torch.cuda.is_available() or torch.__version__.split("+")[0] != "2.14.1":
        pytest.skip("Native reduction kernels target CUDA PyTorch 2.14.1")
    else:
        pass
    if strided and rows % 4:
        pytest.skip("Vocoder graph dispatch requires complete four-row tiles")
    else:
        pass
    torch.manual_seed(43)
    if strided:
        width = 896
        hidden_states = torch.randn(1, width, rows, device="cuda", dtype=dtype)
        hidden_states = hidden_states.transpose(1, 2)
        normalize = normalize_native_strided_rms
    else:
        width = 1024
        hidden_states = torch.randn(rows, width, device="cuda", dtype=dtype)
        normalize = normalize_native_rms
    weight = torch.randn(width, device="cuda", dtype=dtype)
    expected = normalization_affine(
        hidden_states,
        hidden_states.float().pow(2).mean(-1, keepdim=True),
        weight,
        1e-6,
    )
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.inference_mode(), torch.cuda.stream(stream):
        normalize(hidden_states, weight, 1e-6)
        normalize(hidden_states, weight, 1e-6)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.inference_mode(), torch.cuda.graph(graph, stream=stream):
        actual = normalize(hidden_states, weight, 1e-6)
    graph.replay()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    retained = actual.clone()
    hidden_states.mul_(2)
    graph.replay()
    torch.testing.assert_close(retained, expected, rtol=0, atol=0)
