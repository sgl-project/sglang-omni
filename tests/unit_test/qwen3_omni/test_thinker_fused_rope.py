# SPDX-License-Identifier: Apache-2.0
"""The thinker may only fuse QK-norm and RoPE where MRoPE degenerates.

Fusing a batch whose three position rows diverge would silently rotate image and
video tokens as text, so these pin the premise and the gate that enforces it.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from sglang_omni.models.qwen3_omni.components.thinker_fused_rope import (
    ThinkerFusedRopeGate,
    install_thinker_fused_rope,
)
from sglang_omni.platforms import current_platform

requires_xpu = pytest.mark.skipif(
    not (hasattr(torch, "xpu") and torch.xpu.is_available()),
    reason="requires XPU",
)
_MROPE_SECTION = [24, 20, 20]


def _yarnless_config() -> SimpleNamespace:
    return SimpleNamespace(rope_scaling=None, rope_parameters=None)


def _fake_attn(**overrides) -> SimpleNamespace:
    attrs = dict(
        apply_qk_norm_rope=lambda *a: None,
        head_dim=128,
        config=_yarnless_config(),
        rotary_emb=SimpleNamespace(
            cos_sin_cache=torch.zeros(8, 128), is_neox_style=True
        ),
    )
    attrs.update(overrides)
    return SimpleNamespace(**attrs)


def _pin_xpu(monkeypatch: pytest.MonkeyPatch) -> None:
    from sglang_omni.models.qwen3_omni.components import thinker_fused_rope

    monkeypatch.setattr(thinker_fused_rope.current_platform, "is_xpu", lambda: True)


def _extend_batch(*, mm_inputs) -> SimpleNamespace:
    return SimpleNamespace(
        mm_inputs=mm_inputs,
        forward_mode=SimpleNamespace(is_extend=lambda: True),
    )


def _decode_batch() -> SimpleNamespace:
    return SimpleNamespace(
        mm_inputs=None,
        forward_mode=SimpleNamespace(is_extend=lambda: False),
    )


def _positions(rows_equal: bool, tokens: int = 6) -> torch.Tensor:
    base = torch.arange(tokens, dtype=torch.int64)
    if rows_equal:
        return base.repeat(3, 1)
    return torch.stack([base, base + 1, base + 2])


def test_equal_rows_make_mrope_collapse_to_the_temporal_row() -> None:
    from sglang.srt.layers.rotary_embedding.mrope import apply_interleaved_rope

    cos = torch.randn(3, 6, 64)
    cos[1] = cos[0]
    cos[2] = cos[0]

    assert torch.equal(apply_interleaved_rope(cos, _MROPE_SECTION), cos[0])

    diverged = torch.randn(3, 6, 64)
    assert not torch.equal(
        apply_interleaved_rope(diverged, _MROPE_SECTION), diverged[0]
    )


def test_upstream_builds_identical_rows_for_a_text_only_batch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from sglang.srt.model_executor import forward_batch_info
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch

    monkeypatch.setattr(
        forward_batch_info,
        "get_exec",
        lambda: SimpleNamespace(
            deterministic=SimpleNamespace(rl_on_policy_target=None)
        ),
    )
    tokens = 6
    batch = SimpleNamespace(
        multimodal_inputs=[None],
        extend_lens=[tokens],
        prefix_lens=[0],
    )
    forward_batch = SimpleNamespace(
        seq_lens_cpu=torch.tensor([tokens]),
        mrope_positions=None,
    )

    ForwardBatch._compute_mrope_positions_extend(
        forward_batch, SimpleNamespace(device="cpu"), batch
    )

    built = forward_batch.mrope_positions
    assert built.shape == (3, tokens)
    assert torch.equal(built[0], built[1])
    assert torch.equal(built[0], built[2])


def test_gate_admits_a_text_only_batch_and_hands_over_one_row() -> None:
    gate = ThinkerFusedRopeGate()
    positions = _positions(rows_equal=True)

    gate.evaluate(positions, _extend_batch(mm_inputs=None))

    assert gate.enabled is True
    assert gate.positions.dtype == torch.int64
    assert torch.equal(gate.positions, positions[0])


@pytest.mark.parametrize(
    "mm_inputs",
    [[object()], [None, object()]],
    ids=["single", "mixed"],
)
def test_gate_refuses_any_batch_carrying_multimodal_input(mm_inputs: list) -> None:
    gate = ThinkerFusedRopeGate()

    gate.evaluate(_positions(rows_equal=False), _extend_batch(mm_inputs=mm_inputs))

    assert gate.enabled is False
    assert gate.positions is None


def test_installing_twice_leaves_one_wrapper_and_no_recursion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _pin_xpu(monkeypatch)
    calls: list[str] = []

    def original(qkv, positions, forward_batch):
        calls.append("original")
        return "unfused"

    attn = _fake_attn(apply_qk_norm_rope=original)
    model = SimpleNamespace(layers=[SimpleNamespace(self_attn=attn)])

    first = install_thinker_fused_rope(
        model,
        kernel_provider=lambda: (lambda *a: None),
        prefill_graph_enabled=False,
    )
    second = install_thinker_fused_rope(
        model,
        kernel_provider=lambda: (lambda *a: None),
        prefill_graph_enabled=False,
    )

    assert first is not None
    assert second is None
    assert attn.apply_qk_norm_rope(None, None, None) == "unfused"
    assert calls == ["original"]


def test_the_installed_wrapper_hands_the_kernel_views_of_the_packed_qkv(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _pin_xpu(monkeypatch)
    heads, kv_heads, dim, tokens = 4, 2, 128, 3
    seen: dict = {}

    def fake_kernel(q, k, q_weight, k_weight, cache, positions, is_neox, eps):
        seen.update(
            q=q,
            k=k,
            q_weight=q_weight,
            k_weight=k_weight,
            cache=cache,
            positions=positions,
            is_neox=is_neox,
            eps=eps,
        )
        q.fill_(1.0)
        k.fill_(2.0)

    q_weight = torch.ones(dim, dtype=torch.bfloat16)
    k_weight = torch.full((dim,), 0.5, dtype=torch.bfloat16)
    table = torch.arange(8 * dim, dtype=torch.bfloat16).reshape(8, dim)
    attn = _fake_attn(
        apply_qk_norm_rope=lambda *a: "unfused",
        head_dim=dim,
        num_heads=heads,
        num_kv_heads=kv_heads,
        q_size=heads * dim,
        kv_size=kv_heads * dim,
        q_norm=SimpleNamespace(weight=q_weight, variance_epsilon=1e-5),
        k_norm=SimpleNamespace(weight=k_weight),
        rotary_emb=SimpleNamespace(cos_sin_cache=table, is_neox_style=False),
    )
    gate = install_thinker_fused_rope(
        SimpleNamespace(layers=[SimpleNamespace(self_attn=attn)]),
        kernel_provider=lambda: fake_kernel,
        prefill_graph_enabled=False,
    )
    positions = _positions(rows_equal=True, tokens=tokens)
    gate.evaluate(positions, _extend_batch(mm_inputs=None))

    qkv = torch.randn(tokens, (heads + 2 * kv_heads) * dim, dtype=torch.bfloat16)
    v_before = qkv[:, (heads + kv_heads) * dim :].clone()

    q, k, v = attn.apply_qk_norm_rope(qkv, positions, None)

    assert seen["q"].shape == (tokens, heads, dim)
    assert seen["k"].shape == (tokens, kv_heads, dim)
    assert seen["q"].data_ptr() == qkv.data_ptr()
    assert seen["k"].data_ptr() == qkv[:, heads * dim :].data_ptr()
    assert (qkv[:, : heads * dim] == 1.0).all()
    assert (qkv[:, heads * dim : (heads + kv_heads) * dim] == 2.0).all()

    assert seen["q_weight"] is q_weight
    assert seen["k_weight"] is k_weight
    assert seen["cache"].dtype == torch.float32
    assert torch.equal(seen["cache"], table.float())
    assert torch.equal(seen["positions"], positions[0])
    assert seen["is_neox"] is False
    assert seen["eps"] == 1e-5

    assert q.shape == (tokens, heads * dim)
    assert k.shape == (tokens, kv_heads * dim)
    assert torch.equal(v, v_before)
    assert attn._used_fused_qk_norm_rope_last_call is True

    gate.evaluate(positions, _decode_batch())
    assert attn.apply_qk_norm_rope(qkv, positions, None) == "unfused"


def test_gate_refuses_decode_so_no_graph_can_freeze_the_decision() -> None:
    gate = ThinkerFusedRopeGate()

    gate.evaluate(_positions(rows_equal=True), _decode_batch())

    assert gate.enabled is False


def test_gate_refuses_when_positions_are_not_mrope() -> None:
    gate = ThinkerFusedRopeGate()

    gate.evaluate(torch.arange(4), _extend_batch(mm_inputs=None))

    assert gate.enabled is False


def test_install_is_refused_while_a_prefill_graph_could_freeze_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _pin_xpu(monkeypatch)
    layer = SimpleNamespace(self_attn=_fake_attn())

    assert (
        install_thinker_fused_rope(
            SimpleNamespace(layers=[layer]),
            kernel_provider=lambda: (lambda *a: None),
            prefill_graph_enabled=True,
        )
        is None
    )


@pytest.mark.accelerator
@requires_xpu
def test_install_succeeds_on_xpu() -> None:
    attn = _fake_attn()

    gate = install_thinker_fused_rope(
        SimpleNamespace(layers=[SimpleNamespace(self_attn=attn)]),
        kernel_provider=lambda: (lambda *a: None),
        prefill_graph_enabled=False,
    )

    assert gate is not None
    assert attn.apply_qk_norm_rope.__func__.__name__ == "_bound"


def test_install_is_refused_off_xpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from sglang_omni.models.qwen3_omni.components import thinker_fused_rope

    layer = SimpleNamespace(self_attn=_fake_attn())
    monkeypatch.setattr(thinker_fused_rope.current_platform, "is_xpu", lambda: False)

    assert (
        install_thinker_fused_rope(
            SimpleNamespace(layers=[layer]),
            kernel_provider=lambda: (lambda *a: None),
            prefill_graph_enabled=False,
        )
        is None
    )


def test_install_is_a_no_op_when_the_build_has_no_kernel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _pin_xpu(monkeypatch)

    assert (
        install_thinker_fused_rope(
            SimpleNamespace(layers=[]),
            kernel_provider=lambda: None,
            prefill_graph_enabled=False,
        )
        is None
    )


def test_the_kernel_is_not_acquired_off_xpu(monkeypatch: pytest.MonkeyPatch) -> None:
    from sglang_omni.models.qwen3_omni.components import thinker_fused_rope

    monkeypatch.setattr(thinker_fused_rope.current_platform, "is_xpu", lambda: False)

    def provider():
        raise AssertionError("the kernel was acquired off XPU")

    assert (
        install_thinker_fused_rope(
            SimpleNamespace(layers=[]),
            kernel_provider=provider,
            prefill_graph_enabled=False,
        )
        is None
    )


@pytest.mark.accelerator
@requires_xpu
def test_the_kernel_matches_an_unfused_reference() -> None:
    kernel = current_platform.get_fused_qk_norm_rope_with_cos_sin_cache()
    if kernel is None:
        pytest.skip("this sgl_kernel build has no fused QK-norm-RoPE kernel")

    device, dtype = torch.device("xpu"), torch.bfloat16
    heads, kv_heads, dim, seq, eps, theta = 8, 2, 128, 6, 1e-6, 1000000.0
    torch.manual_seed(0)
    q = torch.randn(seq, heads * dim, device=device, dtype=dtype)
    k = torch.randn(seq, kv_heads * dim, device=device, dtype=dtype)
    q_weight = torch.randn(dim, device=device, dtype=dtype)
    k_weight = torch.randn(dim, device=device, dtype=dtype)
    positions = torch.arange(seq, device=device, dtype=torch.int64)

    inv = 1.0 / (
        theta ** (torch.arange(0, dim, 2, device=device, dtype=torch.float32) / dim)
    )
    angle = positions.float()[:, None] * inv
    cos, sin = angle.cos()[:, None, :], angle.sin()[:, None, :]

    def reference(part: torch.Tensor, weight: torch.Tensor, count: int) -> torch.Tensor:
        flat = part.reshape(-1, dim).float()
        normed = (flat * torch.rsqrt(flat.pow(2).mean(-1, keepdim=True) + eps)) * (
            weight.float()
        )
        left, right = normed.reshape(seq, count, dim).split(dim // 2, dim=-1)
        rotated = torch.cat([left * cos - right * sin, right * cos + left * sin], -1)
        return rotated.to(dtype).reshape(seq, count * dim)

    expected_q = reference(q, q_weight, heads)
    expected_k = reference(k, k_weight, kv_heads)

    full = torch.arange(seq + 8, device=device, dtype=torch.float32)[:, None] * inv
    cache = torch.cat([full.cos(), full.sin()], -1).float().contiguous()
    fused_q = q.reshape(seq, heads, dim).contiguous()
    fused_k = k.reshape(seq, kv_heads, dim).contiguous()
    kernel(fused_q, fused_k, q_weight, k_weight, cache, positions, True, eps)

    torch.testing.assert_close(
        fused_q.reshape(seq, heads * dim), expected_q, atol=1e-2, rtol=1e-2
    )
    torch.testing.assert_close(
        fused_k.reshape(seq, kv_heads * dim), expected_k, atol=1e-2, rtol=1e-2
    )
