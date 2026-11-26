# SPDX-License-Identifier: Apache-2.0
"""Default Q/K fusion wiring, and the trig tables the fused kernel reads."""

from unittest.mock import Mock

import pytest
import torch

from sglang_omni.models.auk import stages
from sglang_omni.models.auk.dit import Attention, AuKDit, Rope
from sglang_omni.models.auk.flow_matching import AuKFlowMatching
from sglang_omni.models.auk.hf_config import AuKRuntimeConfig
from sglang_omni.platforms import current_platform
from sglang_omni.platforms.xpu import XPUOmniPlatform


@pytest.fixture
def triton_device() -> torch.device:
    if current_platform.device_type not in ("cuda", "xpu"):
        pytest.skip("requires a CUDA or XPU device for the Triton kernel")
    else:
        pass
    return current_platform.get_device(0)


def make_rope(freqs: torch.Tensor) -> Rope:
    """What AuKDit.forward hands a block once the fused kernel is on."""
    return Rope(freqs, 1.0, freqs.cos(), freqs.sin())


def test_qk_fusion_matches_native_norm_and_rope(triton_device):
    from sglang_omni.models.auk.fused_qk_norm_rope import fused_qk_norm_rope

    torch.manual_seed(0)
    attention = Attention(dim=128, heads=2, dim_head=64).to(triton_device)
    qkv = torch.randn(2, 17, 384, device=triton_device, dtype=torch.bfloat16)
    query, key, _ = qkv.chunk(3, dim=-1)
    query = attention.split_heads(query, 2, 64)
    key = attention.split_heads(key, 2, 64)
    rope = make_rope(torch.randn(2, 17, 64, device=triton_device))

    with (
        torch.inference_mode(),
        torch.autocast(triton_device.type, dtype=torch.bfloat16),
    ):
        expected = attention.apply_rope(
            attention.q_norm(query), attention.k_norm(key), rope
        )
        actual = fused_qk_norm_rope(
            query, key, attention.q_norm, attention.k_norm, rope
        )

    if triton_device.type == "cuda":
        assert all(torch.equal(left, right) for left, right in zip(actual, expected))
    else:
        for left, right in zip(actual, expected):
            torch.testing.assert_close(left, right, rtol=1e-5, atol=1e-6)


def test_qk_fusion_supports_native_bf16_backbone(triton_device):
    from sglang_omni.models.auk.fused_qk_norm_rope import fused_qk_norm_rope

    torch.manual_seed(0)
    attention = (
        Attention(dim=128, heads=2, dim_head=64).to(triton_device).to(torch.bfloat16)
    )
    for norm in (attention.q_norm, attention.k_norm):
        torch.nn.init.uniform_(norm.weight, 0.5, 1.5)
    qkv = torch.randn(2, 17, 384, device=triton_device, dtype=torch.bfloat16)
    query, key, _ = qkv.chunk(3, dim=-1)
    query = attention.split_heads(query, 2, 64)
    key = attention.split_heads(key, 2, 64)
    rope = make_rope(torch.randn(2, 17, 64, device=triton_device))

    with torch.inference_mode():
        expected = attention.apply_rope(
            attention.q_norm(query), attention.k_norm(key), rope
        )
        actual = fused_qk_norm_rope(
            query, key, attention.q_norm, attention.k_norm, rope
        )

    assert all(output.dtype == torch.bfloat16 for output in actual)
    assert all(output.shape == query.shape for output in actual)
    assert all(torch.isfinite(output).all() for output in actual)
    for output, reference in zip(actual, expected, strict=True):
        assert reference.dtype == torch.bfloat16
        torch.testing.assert_close(
            output.float(),
            reference.float(),
            rtol=0,
            atol=2**-6 * reference.abs().max().item(),
        )


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
    monkeypatch.setattr(stages, "load_flow", lambda *args: flow)
    monkeypatch.setattr(
        stages, "resolve_concrete_device", lambda device, index: torch.device(device)
    )
    monkeypatch.setattr(stages, "scheduler", Mock())

    assert dit.qk_fusion is None
    assert dit.build_rope(8).cos is None
    stages.create_auk_engine_executor("stub", device="cuda", dtype="float32")
    assert dit.qk_fusion is fused_qk_norm_rope
    assert all(
        block.attn.qk_fusion is fused_qk_norm_rope
        for block in (*dit.transformer_blocks, *dit.single_transformer_blocks)
    )
    # The backbone, not the kernel, holds the tables a compiled block reads.
    rope = dit.build_rope(8)
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
    monkeypatch.setattr(stages, "load_flow", lambda *args: flow)
    monkeypatch.setattr(
        stages, "resolve_concrete_device", lambda device, index: torch.device(device)
    )
    monkeypatch.setattr(stages, "scheduler", Mock())

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
        lambda path: AuKRuntimeConfig(model_path=path, name=name),
    )
    monkeypatch.setattr(stages, "load_flow", lambda *args: flow)
    monkeypatch.setattr(
        stages, "resolve_concrete_device", lambda device, index: torch.device(device)
    )
    monkeypatch.setattr(stages, "scheduler", Mock())

    stages.create_auk_engine_executor("stub", device=device)

    assert dit.qk_fusion is None
    assert all(
        block.attn.qk_fusion is None
        for block in (*dit.transformer_blocks, *dit.single_transformer_blocks)
    )


@pytest.mark.parametrize(
    ("platform_opts_in", "name", "installed"),
    [(True, "AuK", True), (False, "AuK", False), (True, "AuK-Flash", False)],
    ids=["xpu_opts_in", "xpu_declines", "xpu_flash"],
)
def test_the_xpu_engine_installs_qk_fusion_only_where_the_platform_opts_in(
    monkeypatch, platform_opts_in, name, installed
):
    pytest.importorskip("triton")
    from sglang_omni.models.auk.fused_qk_norm_rope import fused_qk_norm_rope

    platform = XPUOmniPlatform()
    monkeypatch.setattr(
        platform, "enable_auk_fused_qk_norm_rope", lambda: platform_opts_in
    )
    monkeypatch.setattr(stages, "current_platform", platform)
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
        lambda path: AuKRuntimeConfig(model_path=path, name=name),
    )
    monkeypatch.setattr(stages, "load_flow", lambda *args: flow)
    monkeypatch.setattr(
        stages, "resolve_concrete_device", lambda device, index: torch.device(device)
    )
    monkeypatch.setattr(stages, "scheduler", Mock())

    stages.create_auk_engine_executor("stub", device="xpu", dtype="float32")

    expected = fused_qk_norm_rope if installed else None
    assert dit.qk_fusion is expected
    assert all(
        block.attn.qk_fusion is expected
        for block in (*dit.transformer_blocks, *dit.single_transformer_blocks)
    )
