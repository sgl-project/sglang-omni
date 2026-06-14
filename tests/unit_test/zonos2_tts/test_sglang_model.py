from __future__ import annotations

from types import SimpleNamespace

import torch

from sglang_omni.models.zonos2_tts.sglang_model import Zonos2SGLangModel


class _RecordingMoEParam:
    def __init__(self) -> None:
        self.calls: list[tuple[torch.Tensor, str, str, int]] = []

    def weight_loader(
        self,
        param: object,
        loaded_weight: torch.Tensor,
        weight_name: str,
        *,
        shard_id: str,
        expert_id: int,
    ) -> None:
        assert param is self
        self.calls.append((loaded_weight.clone(), weight_name, shard_id, expert_id))


def _model_with_experts(num_experts: int = 2) -> Zonos2SGLangModel:
    model = object.__new__(Zonos2SGLangModel)
    model.config = SimpleNamespace(moe_n_experts=num_experts)
    return model


def test_packed_moe_loader_marks_sonic_weights_as_weights() -> None:
    model = _model_with_experts(num_experts=2)
    param = _RecordingMoEParam()
    loaded = torch.arange(2 * 3 * 4, dtype=torch.float32).view(2, 3, 4)

    assert model._load_packed_moe_experts(
        param,
        loaded,
        "layers.3.feed_forward.experts.w13",
        "w1",
    )

    assert [call[1:] for call in param.calls] == [
        ("layers.3.feed_forward.experts.w13.weight", "w1", 0),
        ("layers.3.feed_forward.experts.w13.weight", "w1", 1),
    ]
    assert torch.equal(param.calls[0][0], loaded[0])
    assert torch.equal(param.calls[1][0], loaded[1])


def test_packed_moe_loader_preserves_weight_names() -> None:
    model = _model_with_experts(num_experts=2)
    param = _RecordingMoEParam()
    loaded = torch.zeros(2, 3, 4)

    assert model._load_packed_moe_experts(
        param,
        loaded,
        "layers.3.feed_forward.experts.w2.weight",
        "w2",
    )

    assert [call[1] for call in param.calls] == [
        "layers.3.feed_forward.experts.w2.weight",
        "layers.3.feed_forward.experts.w2.weight",
    ]
