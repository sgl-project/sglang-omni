# SPDX-License-Identifier: Apache-2.0
"""Exercise Ascend NPU adaptation paths in vendored SenseNova modules."""

from types import SimpleNamespace

import pytest

pytest.importorskip("sglang")

import torch
import torch.nn.functional as F

from sglang_omni.vendor.sensenova_u1 import modeling_neo_chat, modeling_qwen3


def test_right_aligns_bnsd_prefix_for_npu_fia() -> None:
    source = torch.tensor(
        [
            [[[1], [2], [99], [99], [99]]],
            [[[3], [4], [5], [6], [7]]],
        ]
    )
    destination = torch.zeros(2, 1, 8, 1, dtype=source.dtype)

    modeling_neo_chat._copy_right_aligned_prefix_bnsd(  # noqa: leading-underscore  # vendored helper under test
        destination, source, [2, 5]
    )

    assert destination[:, 0, :5, 0].tolist() == [
        [0, 0, 0, 1, 2],
        [3, 4, 5, 6, 7],
    ]
    assert destination[:, :, 5:].eq(0).all()


@pytest.mark.parametrize("available", [False, True])
def test_npu_fia_available_probe(
    monkeypatch: pytest.MonkeyPatch, available: bool
) -> None:
    namespace = SimpleNamespace()
    if available:
        namespace.npu_fused_infer_attention_score = object()
    else:
        pass
    monkeypatch.setattr(torch.ops, "npu", namespace, raising=False)
    assert modeling_qwen3.npu_fia_available() is available


@torch.no_grad()
def test_fused_dense_mlp_matches_original(monkeypatch: pytest.MonkeyPatch) -> None:
    config = SimpleNamespace(
        hidden_size=16,
        intermediate_size=24,
        hidden_act="silu",
    )
    with torch.random.fork_rng():
        torch.manual_seed(37)
        mlp = modeling_qwen3.Qwen3MLP(config).eval()
        hidden_states = torch.randn(2, 5, config.hidden_size)
        expected = mlp(hidden_states)

    monkeypatch.setattr(
        mlp,
        "_use_npu_fused_mlp",  # noqa: leading-underscore  # vendored method under test
        lambda hidden_states: True,
    )
    monkeypatch.setattr(
        torch.ops,
        "npu",
        SimpleNamespace(
            npu_swiglu=lambda gate_up, dim=-1: (
                F.silu(gate_up.chunk(2, dim=dim)[0]) * gate_up.chunk(2, dim=dim)[1]
            )
        ),
        raising=False,
    )
    actual = mlp(hidden_states)

    torch.testing.assert_close(actual, expected)
    assert set(mlp.state_dict()) == {
        "gate_proj.weight",
        "up_proj.weight",
        "down_proj.weight",
    }
    assert (
        mlp.gate_proj.weight.untyped_storage().data_ptr()
        == mlp.up_proj.weight.untyped_storage().data_ptr()
    )


def test_randn_fallback_preserves_cpu_rng(monkeypatch: pytest.MonkeyPatch) -> None:
    original_generator = torch.Generator

    def unsupported_device_generator(
        device: str | torch.device = "cpu",
    ) -> torch.Generator:
        if torch.device(device).type == "cpu":
            raise RuntimeError("Generator is unsupported on cpu")
        else:
            return original_generator(device)

    monkeypatch.setattr(torch, "Generator", unsupported_device_generator)
    rng_state = torch.get_rng_state().clone()

    first = modeling_neo_chat._randn_with_seed(  # noqa: leading-underscore  # vendored helper under test
        (2, 3), device="cpu", dtype=torch.float32, seed=17
    )
    second = modeling_neo_chat._randn_with_seed(  # noqa: leading-underscore  # vendored helper under test
        (2, 3), device="cpu", dtype=torch.float32, seed=17
    )

    assert torch.equal(first, second)
    assert torch.equal(torch.get_rng_state(), rng_state)
