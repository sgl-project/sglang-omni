# SPDX-License-Identifier: Apache-2.0
"""Staleness guard tests for the fused MOSS-TTS audio LM heads."""

from __future__ import annotations

from types import MethodType, SimpleNamespace

import pytest
import torch

from sglang_omni.models.moss_tts.sglang_model import MossTTSDelaySGLangModel


def _make_stub(rows: int = 8, hidden: int = 4, n_audio: int = 3) -> SimpleNamespace:
    stacked = torch.randn(n_audio * rows, hidden)
    heads = [SimpleNamespace(weight=torch.randn(2, hidden))]  # text head
    for index in range(n_audio):
        heads.append(SimpleNamespace(weight=stacked[index * rows : (index + 1) * rows]))
    stub = SimpleNamespace(
        lm_heads=heads,
        _stacked_audio_head_weight=stacked,
        _audio_head_padded_vocab=rows,
        _audio_head_expected_ptrs=[
            stacked[index * rows : (index + 1) * rows].data_ptr()
            for index in range(n_audio)
        ],
        _fused_audio_heads_enabled=True,
    )
    stub._ensure_stacked_audio_heads = MethodType(
        lambda self: self._stacked_audio_head_weight is not None, stub
    )
    stub._fused_audio_heads_ready = MethodType(
        MossTTSDelaySGLangModel._fused_audio_heads_ready, stub
    )
    return stub


def test_fused_audio_heads_ready_when_aliased() -> None:
    stub = _make_stub()
    assert stub._fused_audio_heads_ready() is True


@pytest.mark.parametrize("replaced_index", [1, 2, 3])
def test_replacing_any_audio_head_disables_fused_path(replaced_index: int) -> None:
    stub = _make_stub()
    stub.lm_heads[replaced_index].weight = torch.randn_like(
        stub.lm_heads[replaced_index].weight
    )
    assert stub._fused_audio_heads_ready() is False
    assert stub._stacked_audio_head_weight is None
    assert stub._fused_audio_heads_enabled is False


def test_ready_never_stacks_lazily() -> None:
    # Stacking happens at load time; the request path may only observe it.
    stub = SimpleNamespace(
        lm_heads=[SimpleNamespace(weight=torch.randn(2, 4))],
        _stacked_audio_head_weight=None,
        _fused_audio_heads_enabled=None,
    )
    stub._fused_audio_heads_ready = MethodType(
        MossTTSDelaySGLangModel._fused_audio_heads_ready, stub
    )
    assert stub._fused_audio_heads_ready() is False


def _plain_mode_stub(n_audio: int = 2) -> SimpleNamespace:
    heads = [SimpleNamespace(weight=torch.randn(2, 4))]
    heads.extend(SimpleNamespace(weight=torch.randn(8, 4)) for _ in range(n_audio))
    processors = [
        SimpleNamespace(use_fp32_lm_head=False, rl_on_policy_target=None)
        for _ in range(n_audio + 1)
    ]
    stub = SimpleNamespace(lm_heads=heads, logits_processors=processors)
    stub._audio_heads_use_plain_lm_head = MethodType(
        MossTTSDelaySGLangModel._audio_heads_use_plain_lm_head, stub
    )
    return stub


def test_plain_mode_gate_accepts_default_configuration() -> None:
    assert _plain_mode_stub()._audio_heads_use_plain_lm_head() is True


def test_plain_mode_gate_rejects_fp32_lm_head() -> None:
    stub = _plain_mode_stub()
    stub.logits_processors[1].use_fp32_lm_head = True
    assert stub._audio_heads_use_plain_lm_head() is False


def test_plain_mode_gate_rejects_rl_on_policy_target() -> None:
    stub = _plain_mode_stub()
    stub.logits_processors[2].rl_on_policy_target = "actor"
    assert stub._audio_heads_use_plain_lm_head() is False


def test_plain_mode_gate_rejects_lora_wrapped_head() -> None:
    stub = _plain_mode_stub()
    stub.lm_heads[1].set_lora = lambda *a: None
    stub.lm_heads[1].apply_lora = lambda *a: None
    assert stub._audio_heads_use_plain_lm_head() is False


class ParallelLMHead:
    """Stand-in whose class name the fused-head gate checks."""

    def __init__(self, weight: torch.Tensor) -> None:
        self.weight = weight


def _share_stub(
    monkeypatch,
    rows: int = 8,
    hidden: int = 4,
    n_audio: int = 3,
    one_block: bool = True,
) -> tuple[SimpleNamespace, torch.Tensor]:
    monkeypatch.setattr(
        "sglang_omni.models.moss_tts.sglang_model."
        "get_tensor_model_parallel_world_size",
        lambda: 1,
    )
    shared = torch.randn(n_audio * rows, hidden)
    heads = [ParallelLMHead(torch.randn(2, hidden))]
    for index in range(n_audio):
        heads.append(
            ParallelLMHead(
                shared[index * rows : (index + 1) * rows]
                if one_block
                else torch.randn(rows, hidden)
            )
        )
    stub = SimpleNamespace(
        lm_heads=heads,
        config=SimpleNamespace(
            channels=n_audio + 1,
            vocab_size_list=[2, *([rows] * n_audio)],
            final_logit_softcapping=None,
        ),
        pp_group=SimpleNamespace(is_last_rank=True),
        logits_processors=[
            SimpleNamespace(use_fp32_lm_head=False, rl_on_policy_target=None)
            for _ in range(n_audio + 1)
        ],
        _stacked_audio_head_weight=None,
        _audio_head_padded_vocab=0,
        _audio_head_expected_ptrs=[],
        _fused_audio_heads_enabled=None,
    )
    for name in (
        "_audio_heads_use_plain_lm_head",
        "_fused_audio_heads_eligible",
        "_fused_audio_heads_ready",
        "_fused_audio_heads_requested",
        "_record_stacked_audio_heads",
        "on_weight_share_attached",
    ):
        setattr(stub, name, MethodType(getattr(MossTTSDelaySGLangModel, name), stub))
    stub._stacked_view_over_heads = MossTTSDelaySGLangModel._stacked_view_over_heads
    return stub, shared


def test_weight_share_attach_adopts_the_shared_stack(monkeypatch) -> None:
    stub, shared = _share_stub(monkeypatch)

    stub.on_weight_share_attached()

    assert stub._fused_audio_heads_ready() is True
    # A view over the leader's storage, not a second copy of it.
    assert stub._stacked_audio_head_weight.data_ptr() == shared.data_ptr()
    assert torch.equal(stub._stacked_audio_head_weight, shared)
    assert stub._audio_head_padded_vocab == 8


def test_weight_share_attach_fails_closed_when_heads_are_not_one_block(
    monkeypatch,
) -> None:
    stub, _ = _share_stub(monkeypatch, one_block=False)

    with pytest.raises(RuntimeError, match="contiguous"):
        stub.on_weight_share_attached()


def test_weight_share_attach_honors_the_disable_switch(monkeypatch) -> None:
    stub, _ = _share_stub(monkeypatch, one_block=False)
    monkeypatch.setenv("MOSS_DELAY_FUSED_AUDIO_HEADS", "0")

    stub.on_weight_share_attached()

    assert stub._fused_audio_heads_enabled is False
    assert stub._fused_audio_heads_ready() is False


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_fused_audio_logits_match_the_per_head_gemm(dtype: torch.dtype) -> None:
    """The two GEMM layouts are not bit identical, so hold the fused path to a
    tolerance against the per-head reference it replaces."""

    device = torch.device("cuda")
    rows, hidden, n_audio, audio_vocab = 1032, 512, 12, 1025
    gen = torch.Generator(device=device).manual_seed(20260826)
    stacked = torch.randn(
        n_audio * rows, hidden, device=device, dtype=dtype, generator=gen
    )
    hidden_states = torch.randn(16, hidden, device=device, dtype=dtype, generator=gen)
    stub = SimpleNamespace(
        config=SimpleNamespace(
            channels=n_audio + 1, vocab_size_list=[2, *([audio_vocab] * n_audio)]
        ),
        _stacked_audio_head_weight=stacked,
        _audio_head_padded_vocab=rows,
    )
    stub._compute_fused_audio_logits = MethodType(
        MossTTSDelaySGLangModel._compute_fused_audio_logits, stub
    )

    fused = stub._compute_fused_audio_logits(hidden_states)
    per_head = torch.stack(
        [
            torch.nn.functional.linear(
                hidden_states, stacked[index * rows : (index + 1) * rows]
            ).to(torch.float32)[:, :audio_vocab]
            for index in range(n_audio)
        ],
        dim=1,
    )

    assert fused.shape == (16, n_audio, audio_vocab)
    torch.testing.assert_close(fused, per_head, rtol=2e-2, atol=2e-2)
