# SPDX-License-Identifier: Apache-2.0
"""Validated SenseNova-U1 image generation options."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class SenseNovaU1Sampling:
    width: int = 2048
    height: int = 2048
    num_inference_steps: int = 50
    guidance_scale: float = 4.0
    seed: int = 42
    enable_cache_dit: bool | None = None
    cache_dit_params: dict[str, int | float] | None = None

    @classmethod
    def from_params(cls, params: dict[str, Any]) -> SenseNovaU1Sampling:
        enable_cache_dit, cache_dit_params = _cache_dit_values(params)
        return cls(
            **_validated_values(cls, params, ("guidance_scale",)),
            enable_cache_dit=enable_cache_dit,
            cache_dit_params=cache_dit_params,
        )


@dataclass(frozen=True)
class SenseNovaU1ImageEditSampling:
    """I2I defaults aligned with SGLang's SenseNova serving pipeline."""

    width: int = 2048
    height: int = 2048
    num_inference_steps: int = 50
    guidance_scale: float = 4.0
    img_cfg_scale: float = 1.0
    seed: int = 42
    input_max_pixels: int | None = None
    do_resize: bool = True
    size_explicit: bool = False
    enable_cache_dit: bool | None = None
    cache_dit_params: dict[str, int | float] | None = None

    @classmethod
    def from_params(cls, params: dict[str, Any]) -> SenseNovaU1ImageEditSampling:
        input_max_pixels = params.get("input_max_pixels")
        if input_max_pixels is not None and (
            type(input_max_pixels) is not int or input_max_pixels < 512 * 512
        ):
            raise ValueError("input_max_pixels must be at least 262144")
        do_resize = params.get("do_resize", True)
        if type(do_resize) is not bool:
            raise ValueError("do_resize must be a boolean")
        size_explicit = params.get("size_explicit", False)
        if type(size_explicit) is not bool:
            raise ValueError("size_explicit must be a boolean")
        values = _validated_values(cls, params, ("guidance_scale", "img_cfg_scale"))
        cache_options = dict(params)
        if (
            image_guidance_branch_count(
                values["guidance_scale"], values["img_cfg_scale"]
            )
            > 2
        ):
            cache_options["cache_dit_params"] = None
        enable_cache_dit, cache_dit_params = _cache_dit_values(cache_options)
        return cls(
            **values,
            input_max_pixels=input_max_pixels,
            do_resize=do_resize,
            size_explicit=size_explicit,
            enable_cache_dit=enable_cache_dit,
            cache_dit_params=cache_dit_params,
        )


def _validated_values(
    defaults: type, params: dict[str, Any], scale_names: tuple[str, ...]
) -> dict[str, Any]:
    if params.get("think_mode", False) is not False:
        raise ValueError("SenseNova-U1 think_mode is not supported yet")
    if params.get("n", 1) != 1:
        raise ValueError("SenseNova-U1 currently supports exactly one image")
    names = ("width", "height", "num_inference_steps", *scale_names, "seed")
    values = {name: params.get(name, getattr(defaults, name)) for name in names}
    for name in ("width", "height", "num_inference_steps", "seed"):
        value = values[name]
        if type(value) is not int or value < (0 if name == "seed" else 1):
            qualifier = "non-negative" if name == "seed" else "positive"
            raise ValueError(f"{name} must be a {qualifier} integer")
    if values["width"] % 32 or values["height"] % 32:
        raise ValueError("SenseNova-U1 width and height must be divisible by 32")
    for name in scale_names:
        value = values[name]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value < 0
        ):
            raise ValueError(f"{name} must be a finite non-negative number")
    return values


CACHE_DIT_PARAMETER_NAMES = frozenset(
    {
        "Fn_compute_blocks",
        "Bn_compute_blocks",
        "max_warmup_steps",
        "residual_diff_threshold",
        "max_continuous_cached_steps",
    }
)


def _cache_dit_values(
    params: dict[str, Any],
) -> tuple[bool | None, dict[str, int | float] | None]:
    enabled = params.get("enable_cache_dit")
    if enabled is not None and type(enabled) is not bool:
        raise ValueError("enable_cache_dit must be a boolean")
    if enabled is False:
        return False, None
    raw = params.get("cache_dit_params")
    if raw is None:
        return enabled, None
    # The serving default is only known by the generation controller. Defer
    # cache parameter validation until it resolves the effective enable flag.
    if enabled is None:
        return enabled, raw
    if not isinstance(raw, dict):
        raise ValueError("cache_dit_params must be a mapping")  # noqa: TRY004
    unknown = set(raw) - CACHE_DIT_PARAMETER_NAMES
    if unknown:
        raise ValueError(f"unsupported Cache-DiT parameters: {sorted(unknown)}")
    values: dict[str, int | float] = {}
    for name, value in raw.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"cache_dit_params.{name} must be numeric")  # noqa: TRY004
        if name == "residual_diff_threshold":
            if not 0 <= value <= 1:
                raise ValueError(f"cache_dit_params.{name} must be between 0 and 1")
        elif type(value) is not int or value < 0:
            raise ValueError(f"cache_dit_params.{name} must be a non-negative integer")
        values[name] = value
    return enabled, values


def resolve_cache_dit_params(
    params: dict[str, Any], default_enabled: bool
) -> tuple[bool, dict[str, int | float] | None]:
    resolved = dict(params)
    if resolved.get("enable_cache_dit") is None:
        resolved["enable_cache_dit"] = default_enabled
    enabled, cache_params = _cache_dit_values(resolved)
    return enabled, cache_params


def image_guidance_branch_count(cfg_scale: float, img_cfg_scale: float) -> int:
    if cfg_scale == 1 and img_cfg_scale == 1:
        return 1
    if img_cfg_scale == 1 or cfg_scale == img_cfg_scale:
        return 2
    return 3


def cache_dit_batch_key(params: Any) -> str:
    """Allow deferred JSON parameters in scheduler compatibility keys."""
    return json.dumps(params, sort_keys=True, separators=(",", ":"))
