# SPDX-License-Identifier: Apache-2.0
"""Transactional ODE solver / diffusion scheduler config overrides for step-distillation LoRAs."""

from __future__ import annotations

import copy
import math
from collections.abc import Mapping, MutableMapping, Sequence
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class ODESolverOverride:
    """Declarative ODE solver / diffusion schedule override carried by a step-distillation LoRA.

    Attributes:
        num_steps: Replacement NFE / denoising step count (for example 4 or 8 for
            few-step distilled checkpoints).
        cfg_scale: Replacement classifier-free guidance scale (often 1.0 or 0.0 when
            guidance is distilled into the LoRA weights).
        solver_method: Optional replacement ODE integrator name (for example "euler",
            "midpoint", "heun", or "unipc").
        sway_sampling_coef: Optional sway time-grid warping coefficient.
        clear_sway_sampling_coef: Explicitly clear the target's sway coefficient to None
            (for example when switching from base sway sampling to a fixed distilled grid).
        t_grid: Optional explicit monotonic sequence of time points (at least two).
        clear_t_grid: Explicitly clear any custom t_grid back to None.
        extra: Optional model-specific solver parameters applied verbatim when mapped.
    """

    num_steps: int | None = None
    cfg_scale: float | None = None
    solver_method: str | None = None
    sway_sampling_coef: float | None = None
    clear_sway_sampling_coef: bool = False
    t_grid: tuple[float, ...] | None = None
    clear_t_grid: bool = False
    extra: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.num_steps is not None:
            if isinstance(self.num_steps, bool) or not isinstance(self.num_steps, int):
                raise ValueError(
                    f"num_steps must be an int, got {type(self.num_steps).__name__}"
                )
            elif self.num_steps < 1:
                raise ValueError(f"num_steps must be >= 1, got {self.num_steps}")
            else:
                pass
        else:
            pass

        if self.cfg_scale is not None:
            if isinstance(self.cfg_scale, bool) or not isinstance(
                self.cfg_scale, (int, float)
            ):
                raise ValueError(
                    f"cfg_scale must be numeric, got {type(self.cfg_scale).__name__}"
                )
            else:
                pass
            cfg_value = float(self.cfg_scale)
            if not math.isfinite(cfg_value) or cfg_value < 0.0:
                raise ValueError(
                    f"cfg_scale must be finite and >= 0.0, got {self.cfg_scale!r}"
                )
            else:
                pass
            object.__setattr__(self, "cfg_scale", cfg_value)
        else:
            pass

        if self.solver_method is not None:
            if (
                not isinstance(self.solver_method, str)
                or not self.solver_method.strip()
            ):
                raise ValueError(
                    f"solver_method must be a non-empty str, got {self.solver_method!r}"
                )
            else:
                pass
            object.__setattr__(self, "solver_method", self.solver_method.strip())
        else:
            pass

        if self.sway_sampling_coef is not None and self.clear_sway_sampling_coef:
            raise ValueError(
                "sway_sampling_coef and clear_sway_sampling_coef cannot both be set"
            )
        else:
            pass

        if self.sway_sampling_coef is not None:
            if isinstance(self.sway_sampling_coef, bool) or not isinstance(
                self.sway_sampling_coef, (int, float)
            ):
                raise ValueError(
                    "sway_sampling_coef must be numeric, "
                    f"got {type(self.sway_sampling_coef).__name__}"
                )
            else:
                pass
            sway_value = float(self.sway_sampling_coef)
            if not math.isfinite(sway_value):
                raise ValueError(
                    f"sway_sampling_coef must be finite, got {self.sway_sampling_coef!r}"
                )
            else:
                pass
            object.__setattr__(self, "sway_sampling_coef", sway_value)
        else:
            pass

        if self.t_grid is not None and self.clear_t_grid:
            raise ValueError("t_grid and clear_t_grid cannot both be set")
        else:
            pass

        if self.t_grid is not None:
            if isinstance(self.t_grid, (str, bytes)) or not isinstance(
                self.t_grid, Sequence
            ):
                raise ValueError("t_grid must be a sequence of at least two floats")
            else:
                pass
            parsed_grid = tuple(float(point) for point in self.t_grid)
            if len(parsed_grid) < 2:
                raise ValueError(
                    f"t_grid must hold at least two time points, got {len(parsed_grid)}"
                )
            else:
                pass
            for point in parsed_grid:
                if not math.isfinite(point):
                    raise ValueError(f"t_grid contains non-finite value {point!r}")
                else:
                    pass
            object.__setattr__(self, "t_grid", parsed_grid)
        else:
            pass

        if not isinstance(self.extra, Mapping):
            raise ValueError("extra must be a mapping")
        else:
            pass
        object.__setattr__(self, "extra", dict(self.extra))

    def is_empty(self) -> bool:
        """Return True when no solver field is overridden."""
        return (
            self.num_steps is None
            and self.cfg_scale is None
            and self.solver_method is None
            and self.sway_sampling_coef is None
            and not self.clear_sway_sampling_coef
            and self.t_grid is None
            and not self.clear_t_grid
            and not self.extra
        )

    def to_dict(self) -> dict[str, object]:
        """Serialize into a JSON/msgspec-compatible dictionary."""
        result: dict[str, object] = {}
        if self.num_steps is not None:
            result["num_steps"] = self.num_steps
        else:
            pass
        if self.cfg_scale is not None:
            result["cfg_scale"] = self.cfg_scale
        else:
            pass
        if self.solver_method is not None:
            result["solver_method"] = self.solver_method
        else:
            pass
        if self.clear_sway_sampling_coef:
            result["sway_sampling_coef"] = None
        elif self.sway_sampling_coef is not None:
            result["sway_sampling_coef"] = self.sway_sampling_coef
        else:
            pass
        if self.clear_t_grid:
            result["t_grid"] = None
        elif self.t_grid is not None:
            result["t_grid"] = list(self.t_grid)
        else:
            pass
        if self.extra:
            result["extra"] = dict(self.extra)
        else:
            pass
        return result

    @classmethod
    def from_dict(
        cls, data: Mapping[str, object] | "ODESolverOverride"
    ) -> "ODESolverOverride":
        """Parse an ODESolverOverride from a dict with common diffusion/flow field aliases."""
        if isinstance(data, cls):
            return data
        else:
            pass
        if not isinstance(data, Mapping):
            raise TypeError(
                f"solver_override must be a mapping or ODESolverOverride, got {type(data).__name__}"
            )
        else:
            pass

        num_steps: int | None = None
        for key in ("num_steps", "steps", "nfe", "num_inference_steps", "dit_steps"):
            if key in data and data[key] is not None:
                num_steps = data[key]  # type: ignore[assignment]
                break
            else:
                pass

        cfg_scale: float | None = None
        for key in (
            "cfg_scale",
            "cfg_strength",
            "guidance_scale",
            "cfg_rate",
            "dit_cfg_scale",
        ):
            if key in data and data[key] is not None:
                cfg_scale = data[key]  # type: ignore[assignment]
                break
            else:
                pass

        solver_method: str | None = None
        for key in ("solver_method", "ode_method", "method", "solver"):
            if key in data and data[key] is not None:
                solver_method = data[key]  # type: ignore[assignment]
                break
            else:
                pass

        sway_sampling_coef: float | None = None
        clear_sway = bool(data.get("clear_sway_sampling_coef", False))
        if "sway_sampling_coef" in data:
            if data["sway_sampling_coef"] is None:
                clear_sway = True
            else:
                sway_sampling_coef = data["sway_sampling_coef"]  # type: ignore[assignment]
        else:
            pass

        t_grid: tuple[float, ...] | None = None
        clear_t_grid = bool(data.get("clear_t_grid", False))
        if "t_grid" in data:
            if data["t_grid"] is None:
                clear_t_grid = True
            else:
                raw_grid = data["t_grid"]
                if isinstance(raw_grid, (str, bytes)) or not isinstance(
                    raw_grid, Sequence
                ):
                    raise ValueError("t_grid must be a sequence of floats")
                else:
                    pass
                t_grid = tuple(float(x) for x in raw_grid)
        else:
            pass

        known_keys = {
            "num_steps",
            "steps",
            "nfe",
            "num_inference_steps",
            "dit_steps",
            "cfg_scale",
            "cfg_strength",
            "guidance_scale",
            "cfg_rate",
            "dit_cfg_scale",
            "solver_method",
            "ode_method",
            "method",
            "solver",
            "sway_sampling_coef",
            "clear_sway_sampling_coef",
            "t_grid",
            "clear_t_grid",
            "extra",
        }
        extra: dict[str, object] = {}
        raw_extra = data.get("extra")
        if raw_extra is not None:
            if not isinstance(raw_extra, Mapping):
                raise ValueError("solver_override['extra'] must be a mapping")
            else:
                pass
            extra.update(raw_extra)
        else:
            pass
        for key, value in data.items():
            if key not in known_keys:
                extra[str(key)] = value
            else:
                pass

        return cls(
            num_steps=num_steps,
            cfg_scale=cfg_scale,
            solver_method=solver_method,
            sway_sampling_coef=sway_sampling_coef,
            clear_sway_sampling_coef=clear_sway,
            t_grid=t_grid,
            clear_t_grid=clear_t_grid,
            extra=extra,
        )


@dataclass(frozen=True)
class ODESolverFieldMap:
    """Maps canonical ODESolverOverride fields onto model-specific keys or attributes."""

    num_steps: str | None = None
    cfg_scale: str | None = None
    solver_method: str | None = None
    sway_sampling_coef: str | None = None
    t_grid: str | None = None
    extra_fields: Mapping[str, str] = field(default_factory=dict)

    def managed_names(self) -> tuple[str, ...]:
        """Return ordered unique target key/attribute names managed by this map."""
        names: list[str] = []
        for target_name in (
            self.num_steps,
            self.cfg_scale,
            self.solver_method,
            self.sway_sampling_coef,
            self.t_grid,
            *self.extra_fields.values(),
        ):
            if target_name is not None and target_name not in names:
                names.append(target_name)
            else:
                pass
        return tuple(names)


@runtime_checkable
class SupportsODESolverOverride(Protocol):
    """Protocol implemented by diffusion/flow stages supporting transactional solver overrides."""

    def snapshot_solver_config(self) -> dict[str, object]:
        """Return a deep copy of the current solver configuration."""

    def apply_solver_values(self, values: Mapping[str, object]) -> None:
        """Apply a validated mapping of target solver key/attribute values."""


class DictSolverTarget:
    """Adapter exposing a mutable sampling dict as a SupportsODESolverOverride target."""

    def __init__(
        self,
        target: MutableMapping[str, object],
        managed_keys: Sequence[str],
    ) -> None:
        self.target = target
        self.managed_keys = tuple(managed_keys)

    def snapshot_solver_config(self) -> dict[str, object]:
        return {
            key: copy.deepcopy(self.target[key])
            for key in self.managed_keys
            if key in self.target
        }

    def apply_solver_values(self, values: Mapping[str, object]) -> None:
        for key, value in values.items():
            self.target[key] = copy.deepcopy(value)


class AttrSolverTarget:
    """Adapter exposing mutable attributes on a stage object as a solver target."""

    def __init__(self, target: object, managed_attrs: Sequence[str]) -> None:
        self.target = target
        self.managed_attrs = tuple(managed_attrs)

    def snapshot_solver_config(self) -> dict[str, object]:
        return {
            attr: copy.deepcopy(getattr(self.target, attr))
            for attr in self.managed_attrs
            if hasattr(self.target, attr)
        }

    def apply_solver_values(self, values: Mapping[str, object]) -> None:
        for attr, value in values.items():
            setattr(self.target, attr, copy.deepcopy(value))


class ODESolverOverrideController:
    """Transactional controller pairing ODE solver overrides with merged LoRA lifecycles.

    The pristine base solver snapshot is captured once on the first staged override
    and is never overwritten by subsequent adapter swaps. Switching from a
    step-distillation LoRA to a non-distilled LoRA (``override=None``) resets the
    solver configuration back to the pristine base snapshot on commit.
    """

    def __init__(
        self,
        target: SupportsODESolverOverride,
        field_map: ODESolverFieldMap,
        *,
        strict_fields: bool = True,
    ) -> None:
        self.target = target
        self.field_map = field_map
        self.strict_fields = bool(strict_fields)
        self.base_snapshot: dict[str, object] | None = None
        self.staged_override: ODESolverOverride | None = None
        self.staged_values: dict[str, object] | None = None
        self.has_staged_change: bool = False
        self.active_override: ODESolverOverride | None = None

    def ensure_base_snapshot(self) -> dict[str, object]:
        """Capture the pristine base solver config once before any mutation."""
        if self.base_snapshot is None:
            self.base_snapshot = self.target.snapshot_solver_config()
        else:
            pass
        return copy.deepcopy(self.base_snapshot)

    def resolve_target_values(
        self, override: ODESolverOverride | None
    ) -> dict[str, object]:
        """Validate ``override`` against ``field_map`` and compute target field values."""
        base = self.ensure_base_snapshot()
        resolved = copy.deepcopy(base)
        if override is None or override.is_empty():
            return resolved
        else:
            pass

        def assign_field(
            canonical_name: str,
            mapped_name: str | None,
            value: object,
        ) -> None:
            if mapped_name is None:
                if self.strict_fields:
                    raise ValueError(
                        f"ODE solver override field {canonical_name!r} is not supported "
                        f"by this stage (managed fields: {self.field_map.managed_names()})"
                    )
                else:
                    return
            else:
                resolved[mapped_name] = value

        if override.num_steps is not None:
            assign_field("num_steps", self.field_map.num_steps, override.num_steps)
        else:
            pass
        if override.cfg_scale is not None:
            assign_field("cfg_scale", self.field_map.cfg_scale, override.cfg_scale)
        else:
            pass
        if override.solver_method is not None:
            assign_field(
                "solver_method",
                self.field_map.solver_method,
                override.solver_method,
            )
        else:
            pass
        if override.clear_sway_sampling_coef:
            assign_field("sway_sampling_coef", self.field_map.sway_sampling_coef, None)
        elif override.sway_sampling_coef is not None:
            assign_field(
                "sway_sampling_coef",
                self.field_map.sway_sampling_coef,
                override.sway_sampling_coef,
            )
        else:
            pass
        if override.clear_t_grid:
            assign_field("t_grid", self.field_map.t_grid, None)
        elif override.t_grid is not None:
            assign_field("t_grid", self.field_map.t_grid, override.t_grid)
        else:
            pass

        for extra_key, extra_value in override.extra.items():
            mapped = self.field_map.extra_fields.get(extra_key)
            assign_field(extra_key, mapped, extra_value)

        return resolved

    def stage_override(
        self, override: ODESolverOverride | Mapping[str, object] | None
    ) -> dict[str, object]:
        """Validate and stage an ODE solver override without mutating the live target."""
        parsed = None if override is None else ODESolverOverride.from_dict(override)
        if parsed is not None and parsed.is_empty():
            parsed = None
        else:
            pass
        values = self.resolve_target_values(parsed)
        self.staged_override = parsed
        self.staged_values = values
        self.has_staged_change = True
        return copy.deepcopy(values)

    def commit_override(self) -> dict[str, object]:
        """Commit the previously staged ODE solver values onto the live target."""
        if not self.has_staged_change or self.staged_values is None:
            return self.target.snapshot_solver_config()
        else:
            pass
        self.target.apply_solver_values(self.staged_values)
        self.active_override = self.staged_override
        committed = copy.deepcopy(self.staged_values)
        self.staged_override = None
        self.staged_values = None
        self.has_staged_change = False
        return committed

    def rollback_override(self) -> None:
        """Discard any uncommitted staged ODE solver override."""
        self.staged_override = None
        self.staged_values = None
        self.has_staged_change = False

    def restore_base(self) -> dict[str, object]:
        """Restore the pristine base solver configuration and clear active overrides."""
        self.rollback_override()
        if self.base_snapshot is None:
            self.active_override = None
            return self.target.snapshot_solver_config()
        else:
            pass
        restored = copy.deepcopy(self.base_snapshot)
        self.target.apply_solver_values(restored)
        self.active_override = None
        return restored


def build_dict_solver_controller(
    target: MutableMapping[str, object],
    field_map: ODESolverFieldMap,
    *,
    strict_fields: bool = True,
) -> ODESolverOverrideController:
    """Create a transactional ODE solver controller backed by a mutable dict."""
    adapter = DictSolverTarget(target, field_map.managed_names())
    return ODESolverOverrideController(adapter, field_map, strict_fields=strict_fields)


def build_attr_solver_controller(
    target: object,
    field_map: ODESolverFieldMap,
    *,
    strict_fields: bool = True,
) -> ODESolverOverrideController:
    """Create a transactional ODE solver controller backed by object attributes."""
    adapter = AttrSolverTarget(target, field_map.managed_names())
    return ODESolverOverrideController(adapter, field_map, strict_fields=strict_fields)


__all__ = [
    "AttrSolverTarget",
    "DictSolverTarget",
    "ODESolverFieldMap",
    "ODESolverOverride",
    "ODESolverOverrideController",
    "SupportsODESolverOverride",
    "build_attr_solver_controller",
    "build_dict_solver_controller",
]
