# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from types import SimpleNamespace

import pytest
from torch import nn

from sglang_omni.models.ming3_tts import sglang_model


class StubMoE(nn.Module):
    def __init__(
        self, routed_scaling_factor: float = 2.5, a2a_moe: bool = False
    ) -> None:
        super().__init__()
        self.routed_scaling_factor = routed_scaling_factor
        self.topk = SimpleNamespace(
            topk_config=SimpleNamespace(apply_routed_scaling_factor_on_output=False)
        )
        # sglang starts with the model-layer factor enabled; the fix turns it off
        # once the top-k weights carry the factor.
        self._apply_routed_scaling_factor_on_output = True  # noqa: leading-underscore
        self._enable_a2a_moe = a2a_moe  # noqa: leading-underscore


def apply_scaling(
    monkeypatch: pytest.MonkeyPatch,
    is_npu: bool,
    factor: float = 2.5,
    a2a_moe: bool = False,
) -> StubMoE:
    monkeypatch.setattr(sglang_model.current_platform, "is_npu", lambda: is_npu)
    monkeypatch.setattr(sglang_model, "BailingMoE", StubMoE)
    model = nn.Sequential(StubMoE(factor, a2a_moe))
    sglang_model.enable_ascend_moe_routing_scale(model)
    return model[0]


def test_npu_scales_routing_weights_once(monkeypatch: pytest.MonkeyPatch) -> None:
    moe = apply_scaling(monkeypatch, is_npu=True)

    assert moe.topk.topk_config.apply_routed_scaling_factor_on_output is True
    repeats = moe._apply_routed_scaling_factor_on_output  # noqa: leading-underscore
    assert repeats is False


def test_unit_scale_is_left_alone(monkeypatch: pytest.MonkeyPatch) -> None:
    moe = apply_scaling(monkeypatch, is_npu=True, factor=1.0)

    assert moe.topk.topk_config.apply_routed_scaling_factor_on_output is False


def test_non_npu_is_left_alone(monkeypatch: pytest.MonkeyPatch) -> None:
    moe = apply_scaling(monkeypatch, is_npu=False)

    assert moe.topk.topk_config.apply_routed_scaling_factor_on_output is False
    repeats = moe._apply_routed_scaling_factor_on_output  # noqa: leading-underscore
    assert repeats is True


def test_a2a_moe_is_left_alone(monkeypatch: pytest.MonkeyPatch) -> None:
    # A2A already scales in the model layer; folding it into top-k would double it.
    moe = apply_scaling(monkeypatch, is_npu=True, a2a_moe=True)

    assert moe.topk.topk_config.apply_routed_scaling_factor_on_output is False
    repeats = moe._apply_routed_scaling_factor_on_output  # noqa: leading-underscore
    assert repeats is True
