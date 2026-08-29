# SPDX-License-Identifier: Apache-2.0
"""Sharing one copy of a stage's weights between two PD halves on one GPU."""

from __future__ import annotations

import dataclasses
from types import SimpleNamespace

import pytest
import torch

from sglang_omni.model_runner.weight_sharing import (
    WeightLayoutMismatch,
    WeightParameterHandle,
    _check_parameters_match,
    adopt_parameter_handles,
)


def _model(*names: str) -> dict:
    return {
        name: torch.nn.Parameter(torch.zeros((4, 4), dtype=torch.float32))
        for name in names
    }


def _handle(tensor: torch.Tensor, *, rebuild=None) -> WeightParameterHandle:
    if rebuild is None:
        rebuild = lambda value: value
    return WeightParameterHandle.from_tensor(tensor, handle=(rebuild, (tensor,)))


def test_matching_names_pass() -> None:
    named = _model("layer.0.weight", "layer.1.weight")
    handles = {name: _handle(param.data) for name, param in named.items()}

    _check_parameters_match(named, handles)


def test_an_exported_parameter_this_model_lacks_is_refused() -> None:
    """The two halves built different models; adopting would read the wrong bytes."""
    named = _model("layer.0.weight")
    handles = {
        name: _handle(param.data)
        for name, param in _model("layer.0.weight", "layer.1.weight").items()
    }

    with pytest.raises(WeightLayoutMismatch, match="absent from this model"):
        _check_parameters_match(named, handles)


def test_a_parameter_that_was_not_exported_is_refused() -> None:
    """Skipping it silently would cost the memory without saying so."""
    named = _model("layer.0.weight", "layer.1.weight")
    handles = {"layer.0.weight": _handle(named["layer.0.weight"].data)}

    with pytest.raises(WeightLayoutMismatch, match="were not exported"):
        _check_parameters_match(named, handles)


def test_the_message_names_the_offending_parameters() -> None:
    """A count alone does not tell the reader where the models diverged."""
    named = _model("a", "b")
    handles = {
        name: _handle(param.data)
        for name, param in _model("a", "b", "c", "d", "e", "f").items()
    }

    with pytest.raises(WeightLayoutMismatch) as excinfo:
        _check_parameters_match(named, handles)

    assert "'c'" in str(excinfo.value)


def test_nothing_is_mutated_before_the_check_passes() -> None:
    """The check runs first so a mismatch leaves the model as it was."""
    named = _model("layer.0.weight")
    before = dict(named)

    with pytest.raises(WeightLayoutMismatch):
        _check_parameters_match(named, {"other.weight": _handle(torch.zeros((4, 4)))})

    assert named == before


@pytest.mark.parametrize(
    ("shared", "match"),
    [
        (torch.zeros((2, 8), dtype=torch.float32), "shape"),
        (torch.zeros((4, 4), dtype=torch.float16), "dtype"),
    ],
)
def test_same_name_with_incompatible_tensor_is_rejected(shared, match) -> None:
    model = torch.nn.Module()
    model.register_parameter("weight", torch.nn.Parameter(torch.zeros((4, 4))))

    with pytest.raises(WeightLayoutMismatch, match=match):
        adopt_parameter_handles(model, {"weight": _handle(shared)})


def test_same_name_with_incompatible_stride_is_rejected() -> None:
    model = torch.nn.Module()
    model.register_parameter("weight", torch.nn.Parameter(torch.zeros((4, 4))))
    shared = torch.zeros((4, 4)).t()

    with pytest.raises(WeightLayoutMismatch, match="stride"):
        adopt_parameter_handles(model, {"weight": _handle(shared)})


def test_same_name_with_incompatible_device_is_rejected() -> None:
    named = _model("weight")
    record = _handle(named["weight"].data)
    record = dataclasses.replace(record, device_type="cuda", device_index=0)

    with pytest.raises(WeightLayoutMismatch, match="device"):
        _check_parameters_match(named, {"weight": record})


def test_rebuild_failure_leaves_the_entire_local_model_unchanged() -> None:
    model = torch.nn.Module()
    model.register_parameter("first", torch.nn.Parameter(torch.ones(2)))
    model.register_parameter("second", torch.nn.Parameter(torch.ones(2) * 2))
    before = {
        name: (param.data.data_ptr(), param.data.clone())
        for name, param in model.named_parameters()
    }

    def fail(_value):
        raise RuntimeError("manifest entry failed")

    handles = {
        "first": _handle(torch.zeros(2)),
        "second": _handle(torch.zeros(2), rebuild=fail),
    }
    with pytest.raises(RuntimeError, match="manifest entry failed"):
        adopt_parameter_handles(model, handles)

    assert all(
        param.data.data_ptr() == before[name][0]
        and torch.equal(param.data, before[name][1])
        for name, param in model.named_parameters()
    )


def test_parameter_swap_failure_rolls_back_already_committed_parameters() -> None:
    class ParameterSlot:
        def __init__(self, tensor, *, fail_once=False):
            self._data = tensor
            self.fail_once = fail_once

        @property
        def data(self):
            return self._data

        @data.setter
        def data(self, value):
            if self.fail_once:
                self.fail_once = False
                raise RuntimeError("swap failed")
            self._data = value

    first = ParameterSlot(torch.ones(2))
    second = ParameterSlot(torch.ones(2) * 2, fail_once=True)
    model = SimpleNamespace(
        named_parameters=lambda: iter((("first", first), ("second", second)))
    )
    before = {"first": first.data, "second": second.data}

    with pytest.raises(RuntimeError, match="swap failed"):
        adopt_parameter_handles(
            model,
            {
                "first": _handle(torch.zeros(2)),
                "second": _handle(torch.zeros(2)),
            },
        )

    assert first.data is before["first"]
    assert second.data is before["second"]


def test_the_publishing_half_publishes(tmp_path) -> None:
    from sglang_omni.model_runner.weight_sharing import (
        WeightSharingPlan,
        apply_weight_sharing,
    )

    model = SimpleNamespace(named_parameters=lambda: iter(()))
    plan = WeightSharingPlan(
        stage_name="thinker_prefill",
        peer_stage="thinker_decode",
        rendezvous_dir=tmp_path,
        gpu_id=0,
        publishes=True,
    )

    assert apply_weight_sharing(model, plan) == 0
    assert (tmp_path / "pd-weights" / "thinker_prefill.pkl").exists()


def test_the_adopting_half_publishes_nothing(tmp_path) -> None:
    """Two publishers would leave each half holding its own copy."""
    from sglang_omni.model_runner.weight_sharing import (
        WeightSharingPlan,
        apply_weight_sharing,
    )

    plan = WeightSharingPlan(
        stage_name="thinker_decode",
        peer_stage="thinker_prefill",
        rendezvous_dir=tmp_path,
        gpu_id=0,
        publishes=False,
    )

    apply_weight_sharing(SimpleNamespace(named_parameters=lambda: iter(())), plan)

    assert not (tmp_path / "pd-weights" / "thinker_decode.pkl").exists()


def test_an_adopter_that_got_nothing_keeps_its_weights(tmp_path) -> None:
    """Giving up costs memory; failing the stage would cost the run."""
    from sglang_omni.model_runner.weight_sharing import (
        WeightSharingPlan,
        apply_weight_sharing,
    )

    plan = WeightSharingPlan(
        stage_name="thinker_decode",
        peer_stage="thinker_prefill",
        rendezvous_dir=tmp_path,
        gpu_id=0,
        publishes=False,
        adopted=None,
    )

    assert apply_weight_sharing(SimpleNamespace(), plan) == 0


def test_adopter_rejects_a_stale_publisher_generation_before_mutation(tmp_path) -> None:
    from sglang_omni.model_runner.weight_rendezvous import (
        publish_parameter_handles,
        read_parameter_handles,
    )
    from sglang_omni.model_runner.weight_sharing import (
        WeightSharingPlan,
        apply_weight_sharing,
    )

    kwargs = dict(rendezvous_dir=tmp_path, stage_name="thinker_prefill", gpu_id=0)
    publish_parameter_handles({}, **kwargs)
    stale = read_parameter_handles(**kwargs)
    publish_parameter_handles({}, **kwargs)
    plan = WeightSharingPlan(
        stage_name="thinker_decode",
        peer_stage="thinker_prefill",
        rendezvous_dir=tmp_path,
        gpu_id=0,
        publishes=False,
        adopted=stale,
    )

    with pytest.raises(RuntimeError, match="generation changed"):
        apply_weight_sharing(SimpleNamespace(named_parameters=lambda: iter(())), plan)
