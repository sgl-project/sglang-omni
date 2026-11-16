# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace

import pytest
from torch import nn

from sglang_omni.models.qwen3_omni.components import thinker_model
from tests.unit_test.fixtures.moe_fakes import FakeExperts, RecordingTopK


def test_moe_block_binds_its_topk_to_the_experts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The talker's MoE block built a TopK and never bound it to its experts, which the routing forward refuses."""
    monkeypatch.setattr(thinker_model, "TopK", RecordingTopK)
    monkeypatch.setattr(
        thinker_model, "get_moe_impl_class", lambda quant_config: FakeExperts
    )
    monkeypatch.setattr(
        thinker_model, "ReplicatedLinear", lambda *args, **kwargs: nn.Identity()
    )
    monkeypatch.setattr(
        thinker_model, "get_parallel", lambda: SimpleNamespace(tp_size=1)
    )
    monkeypatch.setattr(
        thinker_model,
        "get_exec",
        lambda: SimpleNamespace(moe=SimpleNamespace(ep_num_redundant_experts=0)),
    )
    config = SimpleNamespace(
        num_experts=8,
        num_experts_per_tok=2,
        norm_topk_prob=True,
        hidden_size=16,
        moe_intermediate_size=32,
    )

    block = thinker_model.Qwen3OmniMoeThinkerTextSparseMoeBlock(
        layer_id=0, config=config
    )

    assert block.topk.bound == (
        FakeExperts.runner_backend,
        FakeExperts.routing_method_type,
    )
