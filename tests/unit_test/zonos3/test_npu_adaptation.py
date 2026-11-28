"""Tests of the Zonos3 Ascend NPU adaptation: backend dispatch of the sampling
and residual kernels, and the residual fallback's arithmetic."""

from __future__ import annotations

import pytest
import torch


class FakePlatform:
    """A platform that reports only the requested CUDA and NPU capabilities."""

    def __init__(self, cuda: bool, npu: bool) -> None:
        self.cuda = cuda
        self.npu = npu

    def is_cuda(self) -> bool:
        return self.cuda

    def is_npu(self) -> bool:
        return self.npu


def make_stub(tag: str) -> tuple[list, object]:
    """A replacement kernel that records its keyword arguments and returns tag."""
    calls: list = []

    def stub(*args, **kwargs) -> str:
        calls.append(kwargs)
        return tag

    return calls, stub


def sample_code_arguments() -> dict:
    """Minimal arguments for one request over two slots; the stubs ignore tensors."""
    return {
        "logits": torch.zeros(1, 1, 4),
        "slots": torch.zeros(1, dtype=torch.long),
        "knobs": torch.zeros(3, 2),
        "top_k": torch.ones(2, dtype=torch.long),
        "seeds": torch.zeros(2, dtype=torch.long),
        "steps": torch.zeros(2, dtype=torch.long),
        "windows": torch.zeros(2, 1, 2, dtype=torch.long),
        "window_counts": torch.zeros(2, 1, 4, dtype=torch.int32),
        "max_top_k": 4,
        "pad_id": 0,
        "codebook_size": 4,
        "softcap": 0.0,
    }


@pytest.mark.parametrize(
    ("cuda", "npu", "expected"),
    [(True, False, "triton"), (False, True, "npu"), (False, False, "torch")],
)
def test_sample_code_rows_dispatches_by_platform(
    monkeypatch, cuda: bool, npu: bool, expected: str
) -> None:
    kernels = pytest.importorskip("sglang_omni.models.zonos3.sampling_kernels")
    triton_calls, triton_stub = make_stub("triton")
    npu_calls, npu_stub = make_stub("npu")
    torch_calls, torch_stub = make_stub("torch")
    monkeypatch.setattr(kernels, "current_platform", FakePlatform(cuda=cuda, npu=npu))
    monkeypatch.setattr(kernels, "sample_code_rows_triton", triton_stub)
    monkeypatch.setattr(kernels, "sample_code_rows_npu", npu_stub)
    monkeypatch.setattr(kernels, "sample_code_rows_torch", torch_stub)

    result = kernels.sample_code_rows(**sample_code_arguments())

    assert result == expected
    calls = {"triton": triton_calls, "npu": npu_calls, "torch": torch_calls}
    assert len(calls[expected]) == 1
    for name, recorded in calls.items():
        if name != expected:
            assert len(recorded) == 0
        else:
            assert recorded[0]["softcap"] == 0.0
            assert recorded[0]["codebook_size"] == 4


def residual_arguments() -> dict:
    """Minimal arguments for one residual store and mix call."""
    tokens, queries, hidden = 2, 2, 4
    return {
        "value": torch.zeros(tokens, hidden),
        "values": torch.zeros(tokens, queries, hidden),
        "queries": torch.zeros(queries, hidden),
        "value_logits": torch.zeros(tokens, queries, queries),
        "norm_weight": torch.ones(hidden),
        "value_index": 0,
        "store_epsilon": 1e-5,
        "mix_epsilon": 1e-5,
        "output_dtype": torch.float32,
    }


@pytest.mark.parametrize(
    ("cuda", "npu", "expected"),
    [(True, False, "triton"), (False, True, "triton"), (False, False, "torch")],
)
def test_store_and_mix_dispatches_by_platform(
    monkeypatch, cuda: bool, npu: bool, expected: str
) -> None:
    kernels = pytest.importorskip("sglang_omni.models.zonos3.backbone_kernels")
    triton_calls, triton_stub = make_stub("triton")
    torch_calls, torch_stub = make_stub("torch")
    monkeypatch.setattr(kernels, "current_platform", FakePlatform(cuda=cuda, npu=npu))
    monkeypatch.setattr(kernels, "_store_and_mix_values_triton", triton_stub)
    monkeypatch.setattr(kernels, "_store_and_mix_values_torch", torch_stub)

    result = kernels.store_and_mix_values(**residual_arguments())

    assert result == expected
    if expected == "triton":
        assert len(triton_calls) == 1
        assert len(torch_calls) == 0
    else:
        assert len(torch_calls) == 1
        assert len(triton_calls) == 0


def test_store_and_mix_torch_stores_row_and_mixes_it() -> None:
    kernels = pytest.importorskip("sglang_omni.models.zonos3.backbone_kernels")
    torch.manual_seed(0)
    tokens, queries, hidden = 3, 2, 4
    value = torch.randn(tokens, hidden)
    values = torch.zeros(tokens, queries, hidden)
    queries_tensor = torch.randn(queries, hidden)
    value_logits = torch.zeros(tokens, queries, queries)
    norm_weight = torch.randn(hidden)

    out = kernels._store_and_mix_values_torch(  # noqa: leading-underscore
        value,
        values,
        queries_tensor,
        value_logits,
        norm_weight,
        value_index=0,
        store_epsilon=1e-5,
        mix_epsilon=1e-5,
        output_dtype=torch.float32,
    )

    # The first value is stored verbatim, and mixing over one stored row leaves it.
    assert torch.equal(values[:, 0], value)
    inverse_rms = torch.rsqrt(value.square().mean(dim=-1, keepdim=True) + 1e-5)
    assert torch.allclose(out, value * inverse_rms * norm_weight, atol=1e-6)
