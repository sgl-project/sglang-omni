# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch


def test_ming_sparse_moe_tp_collective_uses_forward_flags(monkeypatch) -> None:
    from sglang.srt.runtime_context import get_forward

    from sglang_omni.models.ming_tts import sglang_model

    helper_calls: list[tuple[bool, bool, bool]] = []

    def strict_0516_helper(*, is_tp_path: bool) -> bool:
        forward = get_forward()
        helper_calls.append(
            (
                is_tp_path,
                forward.fuse_mlp_allreduce,
                forward.mlp_reduce_scatter,
            )
        )
        return forward.fuse_mlp_allreduce or forward.mlp_reduce_scatter

    all_reduce_calls: list[torch.Tensor] = []

    def fake_all_reduce(hidden_states: torch.Tensor) -> torch.Tensor:
        all_reduce_calls.append(hidden_states.clone())
        return hidden_states + 100

    monkeypatch.setattr(
        sglang_model,
        "should_skip_post_experts_all_reduce",
        strict_0516_helper,
    )
    monkeypatch.setattr(
        sglang_model,
        "tensor_model_parallel_all_reduce",
        fake_all_reduce,
    )
    block = SimpleNamespace(
        tp_size=2,
        shared_experts=None,
        gate=lambda hidden_states: hidden_states,
        topk=lambda _hidden_states, _router_logits: object(),
        experts=lambda hidden_states, _topk_output: hidden_states + 1,
    )
    hidden_states = torch.tensor([[1.0, 2.0]])

    with get_forward().scoped(
        fuse_mlp_allreduce=False,
        mlp_reduce_scatter=False,
    ):
        ordinary = sglang_model.MingBailingMoeSparseMoeBlock.forward(
            block,
            hidden_states,
        )
    with get_forward().scoped(
        fuse_mlp_allreduce=True,
        mlp_reduce_scatter=False,
    ):
        fused = sglang_model.MingBailingMoeSparseMoeBlock.forward(
            block,
            hidden_states,
        )
    with get_forward().scoped(
        fuse_mlp_allreduce=False,
        mlp_reduce_scatter=True,
    ):
        reduce_scattered = sglang_model.MingBailingMoeSparseMoeBlock.forward(
            block,
            hidden_states,
        )

    assert torch.equal(ordinary, hidden_states + 101)
    assert torch.equal(fused, hidden_states + 1)
    assert torch.equal(reduce_scattered, hidden_states + 1)
    assert len(all_reduce_calls) == 1
    assert helper_calls == [
        (True, False, False),
        (True, True, False),
        (True, False, True),
    ]


@pytest.mark.parametrize(
    ("fuse_mlp_allreduce", "mlp_reduce_scatter", "postprocess_calls"),
    [(True, False, 0), (False, True, 1)],
)
def test_ming_decoder_scopes_mlp_collective_flags(
    fuse_mlp_allreduce: bool,
    mlp_reduce_scatter: bool,
    postprocess_calls: int,
) -> None:
    from sglang.srt.runtime_context import get_forward

    from sglang_omni.models.ming_tts.sglang_model import MingBailingMoeDecoderLayer

    class FakeCommunicator:
        def __init__(self) -> None:
            self.postprocess_calls = 0

        def prepare_attn_and_capture_last_layer_outputs(
            self,
            hidden_states,
            residual,
            forward_batch,
        ):
            del forward_batch
            return hidden_states, residual

        def prepare_mlp(self, *, hidden_states, residual, forward_batch):
            del forward_batch
            return hidden_states, residual

        def should_fuse_mlp_allreduce_with_next_layer(self, forward_batch):
            del forward_batch
            return fuse_mlp_allreduce

        def should_use_reduce_scatter(self, forward_batch):
            del forward_batch
            return mlp_reduce_scatter

        def postprocess_layer(self, hidden_states, residual, forward_batch):
            del forward_batch
            self.postprocess_calls += 1
            return hidden_states, residual

    seen_flags: list[tuple[bool, bool]] = []

    def mlp(hidden_states, forward_batch):
        del forward_batch
        forward = get_forward()
        seen_flags.append((forward.fuse_mlp_allreduce, forward.mlp_reduce_scatter))
        return hidden_states.clone()

    communicator = FakeCommunicator()
    layer = SimpleNamespace(
        layer_communicator=communicator,
        attention=lambda _positions, hidden_states, _forward_batch: hidden_states,
        mlp=mlp,
    )
    hidden_states = torch.ones((1, 2))

    with get_forward().scoped(
        fuse_mlp_allreduce=False,
        mlp_reduce_scatter=False,
    ):
        output, _ = MingBailingMoeDecoderLayer.forward(
            layer,
            positions=torch.tensor([0]),
            hidden_states=hidden_states,
            forward_batch=object(),
            residual=None,
        )
        assert get_forward().fuse_mlp_allreduce is False
        assert get_forward().mlp_reduce_scatter is False

    assert seen_flags == [(fuse_mlp_allreduce, mlp_reduce_scatter)]
    assert communicator.postprocess_calls == postprocess_calls
    assert getattr(output, "_sglang_needs_allreduce_fusion", False) is (
        fuse_mlp_allreduce
    )
