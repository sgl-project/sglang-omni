# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from sglang.srt.model_executor.forward_context import (
    get_forward_context,
    has_forward_context,
)

from sglang_omni.models.ming_tts.engine_builder import MingTtsEngineBuilder
from sglang_omni.models.ming_tts.model_runner import (
    MingTTSModelRunner,
    MingTTSTPStepUpdate,
)


def test_ming_tts_entry_tail_failure_is_published_before_reraise() -> None:
    runner = object.__new__(MingTTSModelRunner)
    runner._tp_rank = 0
    runner.model = SimpleNamespace(
        _decode_input_embedding=SimpleNamespace(weight=torch.empty(1, 4))
    )
    published = []

    def fail_tail(*_):
        raise RuntimeError("tail failed")

    runner._run_entry_tail_step = fail_tail
    runner._broadcast_tp_step_update = published.append
    result = SimpleNamespace(
        logits_output=SimpleNamespace(hidden_states=torch.ones(2, 1, 4))
    )

    with pytest.raises(RuntimeError, match="tail failed"):
        runner._collect_ming_tts_step(
            result,
            forward_batch=None,
            schedule_batch=SimpleNamespace(),
            requests=[SimpleNamespace(), SimpleNamespace()],
        )

    assert len(published) == 1
    assert published[0].tail_failed.tolist() == [1, 1]
    assert torch.count_nonzero(published[0].feedback_embeddings).item() == 0


def test_ming_tts_follower_rejects_tail_failure() -> None:
    runner = object.__new__(MingTTSModelRunner)
    update = MingTTSTPStepUpdate.empty_for_broadcast(
        batch_size=1,
        hidden_size=4,
        device=torch.device("cpu"),
        feedback_dtype=torch.float32,
    )
    update.tail_failed.fill_(1)

    with pytest.raises(RuntimeError, match="acoustic tail failed"):
        runner._apply_follower_step_update(update, [SimpleNamespace()])


def test_ming_tts_abort_callback_resets_runner_state() -> None:
    runner = object.__new__(MingTTSModelRunner)
    runner._request_states = {"req-ming-tts": object()}
    builder = object.__new__(MingTtsEngineBuilder)
    builder._model_runner = runner

    abort_callback = builder.make_abort_callback()
    abort_callback("req-ming-tts")
    abort_callback("req-ming-tts")

    assert runner._request_states == {}


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


def test_prefill_forward_publishes_sglang_forward_context() -> None:
    runner = MingTTSModelRunner.__new__(MingTTSModelRunner)
    attn_backend = SimpleNamespace(init_forward_metadata=lambda _batch: None)
    runner.tp_worker = SimpleNamespace(
        model_runner=SimpleNamespace(attn_backend=attn_backend)
    )

    seen = []

    def model(**kwargs):
        seen.append(get_forward_context().attn_backend)
        return "logits"

    model._decode_input_embedding = SimpleNamespace(
        weight=torch.zeros(1, dtype=torch.float32)
    )
    runner.model = model
    forward_batch = SimpleNamespace(
        positions=torch.tensor([0]),
        mrope_positions=None,
        input_ids=torch.tensor([1]),
    )

    assert not has_forward_context()
    result = runner._forward_with_input_embeds(forward_batch, torch.ones(1, 2))

    assert seen == [attn_backend]
    assert result.logits_output == "logits"
