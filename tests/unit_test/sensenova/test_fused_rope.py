# SPDX-License-Identifier: Apache-2.0
"""SenseNova RoPE dispatch preserves its eager behavior and NPU layout."""

import sys
from types import SimpleNamespace

import pytest
import torch

from sglang_omni.models.sensenova_u1.neo_unify import modeling_qwen3


def _inputs(device="cpu"):
    q = torch.arange(24, dtype=torch.float16, device=device).reshape(1, 2, 3, 4)
    k = q + 1
    cos = torch.full((1, 3, 4), 0.5, dtype=torch.float16, device=device)
    sin = torch.full((1, 3, 4), 0.25, dtype=torch.float16, device=device)
    return q, k, cos, sin


def _eager(q, k, cos, sin):
    cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)
    return (
        q * cos + modeling_qwen3.rotate_half(q) * sin,
        k * cos + modeling_qwen3.rotate_half(k) * sin,
    )


def test_npu_rotary_mul_opt_in_uses_half_layout(monkeypatch):
    q, k, cos, sin = _inputs()
    calls = []

    def fake_rotary_mul(x, cos_arg, sin_arg, rotary_mode):
        assert rotary_mode == "half"
        assert cos_arg.shape == sin_arg.shape == (1, 1, 3, 4)
        calls.append(x)
        return x * cos_arg + modeling_qwen3.rotate_half(x) * sin_arg

    monkeypatch.setitem(
        sys.modules, "torch_npu", SimpleNamespace(npu_rotary_mul=fake_rotary_mul)
    )
    monkeypatch.setattr(modeling_qwen3, "_SENSENOVA_NPU_ROTARY_MUL_DISABLED", False)
    monkeypatch.setattr(
        modeling_qwen3, "_can_use_sensenova_npu_fused_rope", lambda *_: True
    )
    monkeypatch.setenv("SGLANG_SENSENOVA_NPU_FUSED_ROPE", "rotary_mul")

    actual = modeling_qwen3.apply_rotary_pos_emb(q, k, cos, sin)
    for result, expected in zip(actual, _eager(q, k, cos, sin)):
        torch.testing.assert_close(result, expected, atol=0, rtol=0)
    assert len(calls) == 2


def test_npu_rotary_mul_failure_falls_back(monkeypatch):
    q, k, cos, sin = _inputs()

    def fail(*args, **kwargs):
        raise RuntimeError("unsupported")

    monkeypatch.setitem(sys.modules, "torch_npu", SimpleNamespace(npu_rotary_mul=fail))
    monkeypatch.setattr(modeling_qwen3, "_SENSENOVA_NPU_ROTARY_MUL_DISABLED", False)
    monkeypatch.setattr(
        modeling_qwen3, "_can_use_sensenova_npu_fused_rope", lambda *_: True
    )
    monkeypatch.setenv("SGLANG_SENSENOVA_NPU_FUSED_ROPE", "1")

    actual = modeling_qwen3.apply_rotary_pos_emb(q, k, cos, sin)
    for result, expected in zip(actual, _eager(q, k, cos, sin)):
        torch.testing.assert_close(result, expected, atol=0, rtol=0)
    assert modeling_qwen3._SENSENOVA_NPU_ROTARY_MUL_DISABLED


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("axis", ["t", "h", "w"])
@torch.no_grad()
def test_cuda_bf16_rope_matches_eager(axis, monkeypatch):
    from sglang.kernels.ops.diffusion.rope import rope_rotate_half_bitexact

    fused = rope_rotate_half_bitexact.fused_rope_rotate_half_bitexact
    calls = []

    def count_fused_calls(*args, **kwargs):
        calls.append(None)
        return fused(*args, **kwargs)

    monkeypatch.setattr(
        rope_rotate_half_bitexact,
        "fused_rope_rotate_half_bitexact",
        count_fused_calls,
    )
    generator = torch.Generator(device="cuda").manual_seed(2026)
    q = torch.randn(
        1, 129, 32, 64, device="cuda", dtype=torch.bfloat16, generator=generator
    )
    k = torch.randn(
        1, 129, 8, 64, device="cuda", dtype=torch.bfloat16, generator=generator
    )
    if axis != "t":
        start = 0 if axis == "h" else 32
        q, k = q[..., start : start + 32], k[..., start : start + 32]
    angles = torch.randn(1, 129, q.shape[-1], device="cuda", generator=generator)
    cos, sin = angles.cos().bfloat16(), angles.sin().bfloat16()
    q, k = q.transpose(1, 2), k.transpose(1, 2)

    actual = modeling_qwen3.apply_rotary_pos_emb(q, k, cos, sin)
    for result, expected in zip(actual, _eager(q, k, cos, sin)):
        torch.testing.assert_close(result, expected, atol=0, rtol=0)
    assert len(calls) == 2
