# SPDX-License-Identifier: Apache-2.0
"""The thinker may only fuse QK-norm and RoPE where MRoPE degenerates.

MRoPE carries three position rows and the kernel takes one, so fusing a batch
whose rows diverge silently applies text rotation to image and video tokens and
still produces fluent output. These cases pin the premise that makes the fused
path legal and the gate that decides when it applies.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from sglang.srt.utils import get_device

from sglang_omni.models.qwen3_omni.components.thinker_fused_rope import (
    ThinkerFusedRopeGate,
    install_thinker_fused_rope,
)
from sglang_omni.platforms import current_platform

_DEVICE = get_device()
requires_accelerator = pytest.mark.skipif(
    _DEVICE.partition(":")[0] not in ("cuda", "xpu"),
    reason="requires cuda or xpu",
)
_MROPE_SECTION = [24, 20, 20]


def _yarnless_config() -> SimpleNamespace:
    """The least install needs: compute_yarn_parameters reads rope_scaling."""
    return SimpleNamespace(rope_scaling=None, rope_parameters=None)


def _fake_attn(**overrides) -> SimpleNamespace:
    """An attention stub carrying what install reads."""
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
    """Make install's platform check deterministic on any runner."""
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
    """Why one position per token is enough: with equal rows MRoPE selects it.

    apply_interleaved_rope only ever copies from one of the three rows, so equal
    rows leave the temporal row untouched, which is what a 1-D RoPE would use.
    """
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
    """The invariant the gate relies on, pinned against SGLang's own builder.

    The gate collapses MRoPE to row 0 without re-comparing the rows per forward,
    because confirming it costs two device syncs. That is only sound while
    _compute_mrope_positions keeps building all three rows from one range for a
    text-only batch, so drive the real method and assert it. If upstream changes
    that construction this fails here, rather than rotating image or video tokens
    as text at runtime.
    """
    from sglang.srt.model_executor import forward_batch_info
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch

    monkeypatch.setattr(
        forward_batch_info,
        "get_server_args",
        lambda: SimpleNamespace(rl_on_policy_target=None),
    )
    tokens = 6
    batch = SimpleNamespace(
        multimodal_inputs=[None],
        extend_lens=[tokens],
        prefix_lens=[0],
    )
    forward_batch = SimpleNamespace(
        seq_lens_cpu=torch.tensor([tokens]),
        forward_mode=SimpleNamespace(
            is_decode=lambda: False,
            is_extend=lambda **kwargs: True,
        ),
        mrope_positions=None,
    )

    ForwardBatch._compute_mrope_positions(
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
    """Read off the raw list, not contains_mm_inputs().

    That helper answers per item type, so an item type this build does not
    recognize would read as "no image" and admit fusion for diverging rows.
    """
    gate = ThinkerFusedRopeGate()

    gate.evaluate(_positions(rows_equal=False), _extend_batch(mm_inputs=mm_inputs))

    assert gate.enabled is False
    assert gate.positions is None


def test_installing_twice_leaves_one_wrapper_and_no_recursion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A second install must not save the wrapper as its own fallback.

    The behaviour is platform independent, so pin the platform rather than
    letting a CUDA runner take the early return and never reach the guard.
    """
    _pin_xpu(monkeypatch)
    calls: list[str] = []

    def original(qkv, positions, forward_batch):
        calls.append("original")
        return "unfused"

    attn = _fake_attn(apply_qk_norm_rope=original)
    model = SimpleNamespace(layers=[SimpleNamespace(self_attn=attn)])

    first = install_thinker_fused_rope(
        model,
        None,
        kernel_provider=lambda: (lambda *a: None),
        prefill_graph_enabled=False,
    )
    second = install_thinker_fused_rope(
        model,
        None,
        kernel_provider=lambda: (lambda *a: None),
        prefill_graph_enabled=False,
    )

    assert first is not None
    assert second is None
    # The gate is off, so this is the fallback: it must reach the original once.
    assert attn.apply_qk_norm_rope(None, None, None) == "unfused"
    assert calls == ["original"]


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
    """A replayed prefill graph would keep whichever branch capture took."""
    _pin_xpu(monkeypatch)
    layer = SimpleNamespace(self_attn=_fake_attn())

    assert (
        install_thinker_fused_rope(
            SimpleNamespace(layers=[layer]),
            None,
            kernel_provider=lambda: (lambda *a: None),
            prefill_graph_enabled=True,
        )
        is None
    )


@pytest.mark.skipif(
    _DEVICE.partition(":")[0] != "xpu", reason="describes the xpu policy"
)
def test_install_succeeds_on_xpu() -> None:
    """The platform this exists for must actually get the patch."""
    attn = _fake_attn()

    gate = install_thinker_fused_rope(
        SimpleNamespace(layers=[SimpleNamespace(self_attn=attn)]),
        None,
        kernel_provider=lambda: (lambda *a: None),
        prefill_graph_enabled=False,
    )

    assert gate is not None
    assert attn.apply_qk_norm_rope.__func__.__name__ == "_bound"


def test_install_is_refused_off_xpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only XPU opts in; CUDA and ROCm keep their tuned unfused path."""
    from sglang_omni.models.qwen3_omni.components import thinker_fused_rope

    layer = SimpleNamespace(self_attn=_fake_attn())
    monkeypatch.setattr(thinker_fused_rope.current_platform, "is_xpu", lambda: False)

    assert (
        install_thinker_fused_rope(
            SimpleNamespace(layers=[layer]),
            None,
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
            None,
            kernel_provider=lambda: None,
            prefill_graph_enabled=False,
        )
        is None
    )


def test_the_kernel_is_not_acquired_off_xpu(monkeypatch: pytest.MonkeyPatch) -> None:
    """Same for the platform gate: CUDA must not pay for an XPU-only path."""
    from sglang_omni.models.qwen3_omni.components import thinker_fused_rope

    monkeypatch.setattr(thinker_fused_rope.current_platform, "is_xpu", lambda: False)

    def provider():
        raise AssertionError("the kernel was acquired off XPU")

    assert (
        install_thinker_fused_rope(
            SimpleNamespace(layers=[]),
            None,
            kernel_provider=provider,
            prefill_graph_enabled=False,
        )
        is None
    )


@pytest.mark.gpu
@requires_accelerator
def test_the_kernel_matches_an_unfused_reference() -> None:
    """The kernel's rotation equals RMS-norm plus neox RoPE, read off a table."""
    kernel = current_platform.get_fused_qk_norm_rope()
    if kernel is None:
        pytest.skip("this sgl_kernel build has no fused QK-norm-RoPE kernel")

    device, dtype = torch.device(_DEVICE), torch.bfloat16
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
