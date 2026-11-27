# SPDX-License-Identifier: Apache-2.0
"""Unit tests for transactional ODE solver / diffusion scheduler overrides."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from sglang_omni.lora.solver_override import (
    ODESolverFieldMap,
    ODESolverOverride,
    build_attr_solver_controller,
    build_dict_solver_controller,
)


def test_ode_solver_override_validation_and_round_trip() -> None:
    override = ODESolverOverride.from_dict(
        {
            "steps": 4,
            "cfg_strength": 1.0,
            "ode_method": "euler",
            "clear_sway_sampling_coef": True,
            "t_grid": [0.0, 0.25, 0.5, 0.75, 1.0],
            "sigma_min": 0.001,
        }
    )
    assert override.num_steps == 4
    assert override.cfg_scale == 1.0
    assert override.solver_method == "euler"
    assert override.clear_sway_sampling_coef is True
    assert override.t_grid == (0.0, 0.25, 0.5, 0.75, 1.0)
    assert override.extra == {"sigma_min": 0.001}
    assert override.is_empty() is False

    round_tripped = ODESolverOverride.from_dict(override.to_dict())
    assert round_tripped == override

    # Explicit null clears sway_sampling_coef and t_grid.
    null_cleared = ODESolverOverride.from_dict(
        {"sway_sampling_coef": None, "t_grid": None}
    )
    assert null_cleared.clear_sway_sampling_coef is True
    assert null_cleared.clear_t_grid is True

    assert ODESolverOverride().is_empty() is True


def test_ode_solver_override_rejects_invalid_values() -> None:
    with pytest.raises(ValueError, match="num_steps"):
        ODESolverOverride(num_steps=0)

    with pytest.raises(ValueError, match="num_steps"):
        ODESolverOverride(num_steps=True)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="cfg_scale"):
        ODESolverOverride(cfg_scale=-0.5)

    with pytest.raises(ValueError, match="cfg_scale"):
        ODESolverOverride(cfg_scale=float("nan"))

    with pytest.raises(ValueError, match="solver_method"):
        ODESolverOverride(solver_method="   ")

    with pytest.raises(ValueError, match="cannot both be set"):
        ODESolverOverride(sway_sampling_coef=-1.0, clear_sway_sampling_coef=True)

    with pytest.raises(ValueError, match="t_grid"):
        ODESolverOverride(t_grid=(0.0,))

    with pytest.raises(ValueError, match="t_grid"):
        ODESolverOverride(t_grid=(0.0, 1.0), clear_t_grid=True)

    with pytest.raises(TypeError, match="mapping"):
        ODESolverOverride.from_dict("not-a-dict")  # type: ignore[arg-type]


def test_dict_solver_controller_transaction_lifecycle() -> None:
    sampling: dict[str, object] = {
        "steps": 32,
        "cfg_strength": 4.0,
        "sway_sampling_coef": -1.0,
        "t_grid": None,
        "unrelated": "keep",
    }
    field_map = ODESolverFieldMap(
        num_steps="steps",
        cfg_scale="cfg_strength",
        sway_sampling_coef="sway_sampling_coef",
        t_grid="t_grid",
    )
    controller = build_dict_solver_controller(sampling, field_map)

    flash_grid = (0.0, 0.35, 0.7, 0.9, 1.0)
    flash = ODESolverOverride(
        num_steps=4,
        cfg_scale=1.0,
        clear_sway_sampling_coef=True,
        t_grid=flash_grid,
    )

    # Staging does not mutate live sampling dict until commit.
    staged = controller.stage_override(flash)
    assert staged["steps"] == 4
    assert sampling["steps"] == 32
    assert sampling["cfg_strength"] == 4.0
    assert sampling["sway_sampling_coef"] == -1.0
    assert sampling["t_grid"] is None

    # Rollback discards staged values and leaves live dict pristine.
    controller.rollback_override()
    noop_committed = controller.commit_override()
    assert noop_committed["steps"] == 32
    assert sampling["steps"] == 32

    # Stage and commit applies distilled schedule cleanly.
    controller.stage_override(flash)
    committed = controller.commit_override()
    assert committed is not None
    assert sampling["steps"] == 4
    assert sampling["cfg_strength"] == 1.0
    assert sampling["sway_sampling_coef"] is None
    assert sampling["t_grid"] == flash_grid
    assert sampling["unrelated"] == "keep"
    assert controller.active_override == flash

    # Switching to a non-distilled LoRA (override=None) resets solver back to base.
    controller.stage_override(None)
    assert sampling["steps"] == 4  # Still active until commit.
    controller.commit_override()
    assert sampling["steps"] == 32
    assert sampling["cfg_strength"] == 4.0
    assert sampling["sway_sampling_coef"] == -1.0
    assert sampling["t_grid"] is None
    assert controller.active_override is None

    # Re-apply override, then restore_base returns pristine values.
    controller.stage_override(ODESolverOverride(num_steps=8, cfg_scale=2.0))
    controller.commit_override()
    assert sampling["steps"] == 8
    assert sampling["cfg_strength"] == 2.0

    restored = controller.restore_base()
    assert restored["steps"] == 32
    assert sampling["steps"] == 32
    assert sampling["cfg_strength"] == 4.0
    assert sampling["sway_sampling_coef"] == -1.0
    assert sampling["t_grid"] is None


def test_attr_solver_controller_strict_and_non_strict_fields() -> None:
    decoder = SimpleNamespace(dit_steps=50, dit_cfg_scale=7.0)
    field_map = ODESolverFieldMap(num_steps="dit_steps", cfg_scale="dit_cfg_scale")
    strict_ctrl = build_attr_solver_controller(decoder, field_map, strict_fields=True)

    # Rejects unmapped field in strict mode without mutating decoder.
    with pytest.raises(ValueError, match="is not supported by this stage"):
        strict_ctrl.stage_override(ODESolverOverride(num_steps=4, solver_method="heun"))
    assert decoder.dit_steps == 50

    strict_ctrl.stage_override(ODESolverOverride(num_steps=6, cfg_scale=1.5))
    strict_ctrl.commit_override()
    assert decoder.dit_steps == 6
    assert decoder.dit_cfg_scale == 1.5

    strict_ctrl.restore_base()
    assert decoder.dit_steps == 50
    assert decoder.dit_cfg_scale == 7.0

    # Non-strict mode ignores unmapped fields while applying supported ones.
    lenient_ctrl = build_attr_solver_controller(decoder, field_map, strict_fields=False)
    lenient_ctrl.stage_override(
        ODESolverOverride(num_steps=8, solver_method="heun", extra={"unused": 1})
    )
    lenient_ctrl.commit_override()
    assert decoder.dit_steps == 8
    assert decoder.dit_cfg_scale == 7.0
    lenient_ctrl.restore_base()
    assert decoder.dit_steps == 50
