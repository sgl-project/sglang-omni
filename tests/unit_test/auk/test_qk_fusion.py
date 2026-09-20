# SPDX-License-Identifier: Apache-2.0
"""Default Q/K fusion wiring, and the trig tables the fused kernel reads."""

from unittest.mock import Mock

import pytest
import torch

from sglang_omni.models.auk import stages
from sglang_omni.models.auk.dit import Attention, AuKDit, Rope
from sglang_omni.models.auk.flow_matching import AuKFlowMatching
from sglang_omni.models.auk.hf_config import AuKRuntimeConfig


def _rope(freqs: torch.Tensor) -> Rope:
    """What ``AuKDit.forward`` hands a block once the fused kernel is on."""
    return Rope(freqs, 1.0, freqs.cos(), freqs.sin())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_qk_fusion_matches_native_norm_and_rope():
    from sglang_omni.models.auk.fused_qk_norm_rope import fused_qk_norm_rope

    torch.manual_seed(0)
    attention = Attention(dim=128, heads=2, dim_head=64).cuda()
    qkv = torch.randn(2, 17, 384, device="cuda", dtype=torch.bfloat16)
    query, key, _ = qkv.chunk(3, dim=-1)
    query = attention._split_heads(query, 2, 64)
    key = attention._split_heads(key, 2, 64)
    rope = _rope(torch.randn(2, 17, 64, device="cuda"))

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        expected = attention._apply_rope(
            attention.q_norm(query), attention.k_norm(key), rope
        )
        actual = fused_qk_norm_rope(
            query, key, attention.q_norm, attention.k_norm, rope
        )

    assert all(torch.equal(left, right) for left, right in zip(actual, expected))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_qk_fusion_supports_native_bf16_backbone():
    from sglang_omni.models.auk.fused_qk_norm_rope import fused_qk_norm_rope

    torch.manual_seed(0)
    attention = Attention(dim=128, heads=2, dim_head=64).cuda().to(torch.bfloat16)
    qkv = torch.randn(2, 17, 384, device="cuda", dtype=torch.bfloat16)
    query, key, _ = qkv.chunk(3, dim=-1)
    query = attention._split_heads(query, 2, 64)
    key = attention._split_heads(key, 2, 64)
    rope = _rope(torch.randn(2, 17, 64, device="cuda"))

    with torch.inference_mode():
        actual = fused_qk_norm_rope(
            query, key, attention.q_norm, attention.k_norm, rope
        )

    assert all(output.dtype == torch.bfloat16 for output in actual)
    assert all(output.shape == query.shape for output in actual)
    assert all(torch.isfinite(output).all() for output in actual)


def test_request_lengths_do_not_specialize_the_kernel():
    pytest.importorskip("triton")
    from sglang_omni.models.auk.fused_qk_norm_rope import _norm_rope_kernel

    parameters = {parameter.name: parameter for parameter in _norm_rope_kernel.params}
    assert parameters["HEAD_DIM"].is_constexpr
    for name in ("SEQ", "QB", "KB", "CB"):
        parameter = parameters[name]
        assert not parameter.is_constexpr
        assert parameter.do_not_specialize
        assert parameter.do_not_specialize_on_alignment


def test_qk_fusion_is_shared_by_attention_blocks_and_fed_by_the_backbone(monkeypatch):
    pytest.importorskip("triton")
    from sglang_omni.models.auk.fused_qk_norm_rope import fused_qk_norm_rope

    dit = AuKDit(
        dim=64,
        heads=1,
        dim_head=64,
        text_hidden_dim=64,
        num_layers=1,
        num_single_layers=1,
    )
    flow = AuKFlowMatching(dit, num_llm_layers=2)
    monkeypatch.setattr(stages, "resolve_checkpoint", lambda path: path)
    monkeypatch.setattr(
        stages,
        "make_runtime_config",
        lambda path: AuKRuntimeConfig(model_path=path, name="AuK"),
    )
    monkeypatch.setattr(stages, "_load_flow", lambda *args: flow)
    monkeypatch.setattr(
        stages, "resolve_concrete_device", lambda device, index: torch.device(device)
    )
    monkeypatch.setattr(stages, "_scheduler", Mock())

    assert dit.qk_fusion is None
    assert dit._rope(8).cos is None
    stages.create_auk_engine_executor("stub", device="cuda", dtype="float32")
    assert dit.qk_fusion is fused_qk_norm_rope
    assert all(
        block.attn.qk_fusion is fused_qk_norm_rope
        for block in (*dit.transformer_blocks, *dit.single_transformer_blocks)
    )
    # The backbone, not the kernel, holds the tables a compiled block reads.
    rope = dit._rope(8)
    torch.testing.assert_close(rope.cos, rope.freqs.cos())
    torch.testing.assert_close(rope.sin, rope.freqs.sin())


def test_qk_fusion_can_be_disabled(monkeypatch):
    pytest.importorskip("triton")
    dit = AuKDit(
        dim=64,
        heads=1,
        dim_head=64,
        text_hidden_dim=64,
        num_layers=1,
        num_single_layers=1,
    )
    flow = AuKFlowMatching(dit, num_llm_layers=2)
    monkeypatch.setattr(stages, "resolve_checkpoint", lambda path: path)
    monkeypatch.setattr(
        stages,
        "make_runtime_config",
        lambda path: AuKRuntimeConfig(model_path=path, name="AuK"),
    )
    monkeypatch.setattr(stages, "_load_flow", lambda *args: flow)
    monkeypatch.setattr(
        stages, "resolve_concrete_device", lambda device, index: torch.device(device)
    )
    monkeypatch.setattr(stages, "_scheduler", Mock())

    stages.create_auk_engine_executor(
        "stub", device="cuda", enable_dit_fused_qk_norm_rope=False
    )

    assert dit.qk_fusion is None
    assert all(
        block.attn.qk_fusion is None
        for block in (*dit.transformer_blocks, *dit.single_transformer_blocks)
    )


@pytest.mark.parametrize("device,name", [("cpu", "AuK"), ("cuda", "AuK-Flash")])
def test_qk_fusion_falls_back_to_native(monkeypatch, device, name):
    monkeypatch.setattr(stages, "resolve_checkpoint", lambda path: path)
    monkeypatch.setattr(
        stages,
        "make_runtime_config",
        lambda path: AuKRuntimeConfig(model_path=path, name=name),
    )
    monkeypatch.setattr(stages, "_load_flow", Mock())
    monkeypatch.setattr(
        stages, "resolve_concrete_device", lambda device, index: torch.device(device)
    )
    monkeypatch.setattr(stages, "_scheduler", Mock())

    stages.create_auk_engine_executor("stub", device=device)
