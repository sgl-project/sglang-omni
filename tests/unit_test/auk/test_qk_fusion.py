# SPDX-License-Identifier: Apache-2.0
"""Opt-in wiring and request-local fused RoPE cache lifecycle."""

from unittest.mock import Mock

import pytest
import torch

from sglang_omni.models.auk import stages
from sglang_omni.models.auk.dit import Attention, AuKDit
from sglang_omni.models.auk.flow_matching import AuKFlowMatching
from sglang_omni.models.auk.hf_config import AuKRuntimeConfig


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_qk_fusion_matches_native_norm_and_rope():
    from sglang_omni.models.auk.fused_qk_norm_rope import QKFusion

    torch.manual_seed(0)
    attention = Attention(dim=128, heads=2, dim_head=64).cuda()
    qkv = torch.randn(2, 17, 384, device="cuda", dtype=torch.bfloat16)
    query, key, _ = qkv.chunk(3, dim=-1)
    query = attention._split_heads(query, 2, 64)
    key = attention._split_heads(key, 2, 64)
    freqs = torch.randn(2, 17, 64, device="cuda")
    rope = (freqs, 1.0)

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        expected = attention._apply_rope(
            attention.q_norm(query), attention.k_norm(key), rope
        )
        actual = QKFusion()(query, key, attention.q_norm, attention.k_norm, rope)

    assert all(torch.equal(left, right) for left, right in zip(actual, expected))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_qk_fusion_matches_native_bf16_backbone():
    from sglang_omni.models.auk.fused_qk_norm_rope import QKFusion

    torch.manual_seed(0)
    attention = Attention(dim=128, heads=2, dim_head=64).cuda().to(torch.bfloat16)
    qkv = torch.randn(2, 17, 384, device="cuda", dtype=torch.bfloat16)
    query, key, _ = qkv.chunk(3, dim=-1)
    query = attention._split_heads(query, 2, 64)
    key = attention._split_heads(key, 2, 64)
    rope = (torch.randn(2, 17, 64, device="cuda"), 1.0)

    with torch.inference_mode():
        expected = attention._apply_rope(
            attention.q_norm(query), attention.k_norm(key), rope
        )
        actual = QKFusion()(query, key, attention.q_norm, attention.k_norm, rope)

    assert all(torch.equal(left, right) for left, right in zip(actual, expected))


def test_request_lengths_do_not_specialize_the_kernel():
    pytest.importorskip("triton")
    from sglang_omni.models.auk.fused_qk_norm_rope import _norm_rope_kernel

    parameters = {parameter.name: parameter for parameter in _norm_rope_kernel.params}
    for name in ("SEQ", "QB", "KB", "CB"):
        parameter = parameters[name]
        assert not parameter.is_constexpr
        assert parameter.do_not_specialize
        assert parameter.do_not_specialize_on_alignment


def test_qk_fusion_is_shared_by_attention_blocks_and_cleared(monkeypatch):
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
    monkeypatch.setattr(stages, "resolve_device_spec", lambda device, index: device)
    monkeypatch.setattr(stages, "_scheduler", Mock())

    assert dit.qk_fusion is None
    stages.create_auk_engine_executor(
        "stub", device="cuda", enable_dit_fused_qk_norm_rope=True
    )
    assert dit.qk_fusion is not None
    assert all(
        block.attn.qk_fusion is dit.qk_fusion
        for block in (*dit.transformer_blocks, *dit.single_transformer_blocks)
    )
    dit.qk_fusion.tables["request"] = (torch.ones(1),) * 3
    dit.text_cond = dit.text_uncond = torch.ones(1)
    dit.clear_cache()
    assert not dit.qk_fusion.tables
    assert dit.text_cond is None and dit.text_uncond is None


def test_qk_fusion_rejects_unsupported_compute_mode(monkeypatch):
    monkeypatch.setattr(stages, "resolve_checkpoint", lambda path: path)
    monkeypatch.setattr(
        stages,
        "make_runtime_config",
        lambda path: AuKRuntimeConfig(model_path=path, name="AuK"),
    )
    monkeypatch.setattr(stages, "_load_flow", Mock())
    with pytest.raises(ValueError, match="CUDA with bfloat16"):
        stages.create_auk_engine_executor(
            "stub", device="cpu", enable_dit_fused_qk_norm_rope=True
        )


def test_qk_fusion_rejects_flash_checkpoint(monkeypatch):
    monkeypatch.setattr(stages, "resolve_checkpoint", lambda path: path)
    monkeypatch.setattr(
        stages,
        "make_runtime_config",
        lambda path: AuKRuntimeConfig(model_path=path, name="AuK-Flash"),
    )
    monkeypatch.setattr(stages, "_load_flow", Mock())
    monkeypatch.setattr(stages, "resolve_device_spec", lambda device, index: device)
    with pytest.raises(ValueError, match="does not support AuK-Flash"):
        stages.create_auk_engine_executor(
            "stub", device="cuda", enable_dit_fused_qk_norm_rope=True
        )
