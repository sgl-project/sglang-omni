# SPDX-License-Identifier: Apache-2.0
"""Validated SenseNova-U1 image generation options."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass

from pydantic import JsonValue


@dataclass(frozen=True)
class SenseNovaU1Sampling:
    width: int = 2048
    height: int = 2048
    num_inference_steps: int = 50
    guidance_scale: float = 4.0
    seed: int = 42
    enable_cache_dit: bool | None = None
    cache_dit_params: JsonValue = None
    n: int = 1

    @classmethod
    def from_params(cls, params: dict[str, JsonValue]) -> SenseNovaU1Sampling:
        n = params.get("n", 1)
        if type(n) is not int or not 1 <= n <= 10:
            raise ValueError("SenseNova-U1 n must be an integer between 1 and 10")
        else:
            pass
        enable_cache_dit, cache_dit_params = _cache_dit_values(params)
        return cls(
            **_validated_values(cls, {**params, "n": 1}, ("guidance_scale",)),
            n=n,
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
    cache_dit_params: JsonValue = None

    @classmethod
    def from_params(cls, params: dict[str, JsonValue]) -> SenseNovaU1ImageEditSampling:
        input_max_pixels = params.get("input_max_pixels")
        if input_max_pixels is not None and (
            type(input_max_pixels) is not int or input_max_pixels < 512 * 512
        ):
            raise ValueError("input_max_pixels must be at least 262144")
        else:
            pass
        do_resize = params.get("do_resize", True)
        if type(do_resize) is not bool:
            raise ValueError("do_resize must be a boolean")
        else:
            pass
        size_explicit = params.get("size_explicit", False)
        if type(size_explicit) is not bool:
            raise ValueError("size_explicit must be a boolean")
        else:
            pass
        values = _validated_values(cls, params, ("guidance_scale", "img_cfg_scale"))
        cache_options = dict(params)
        if (
            image_guidance_branch_count(
                values["guidance_scale"], values["img_cfg_scale"]
            )
            > 2
        ):
            cache_options["cache_dit_params"] = None
        else:
            pass
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
    defaults: type[SenseNovaU1Sampling | SenseNovaU1ImageEditSampling],
    params: dict[str, JsonValue],
    scale_names: tuple[str, ...],
) -> dict[str, JsonValue]:
    if params.get("think_mode", False) is not False:
        raise ValueError("SenseNova-U1 think_mode is not supported yet")
    else:
        pass
    if params.get("n", 1) != 1:
        raise ValueError("SenseNova-U1 currently supports exactly one image")
    else:
        pass
    default_values: dict[str, int | float] = {
        "width": defaults.width,
        "height": defaults.height,
        "num_inference_steps": defaults.num_inference_steps,
        "guidance_scale": defaults.guidance_scale,
        "seed": defaults.seed,
    }
    if "img_cfg_scale" in scale_names:
        default_values["img_cfg_scale"] = SenseNovaU1ImageEditSampling.img_cfg_scale
    else:
        pass
    names = ("width", "height", "num_inference_steps", *scale_names, "seed")
    values = {name: params.get(name, default_values[name]) for name in names}
    for name in ("width", "height", "num_inference_steps", "seed"):
        value = values[name]
        if type(value) is not int or value < (0 if name == "seed" else 1):
            qualifier = "non-negative" if name == "seed" else "positive"
            raise ValueError(f"{name} must be a {qualifier} integer")
        else:
            pass
    if values["width"] % 32 or values["height"] % 32:
        raise ValueError("SenseNova-U1 width and height must be divisible by 32")
    else:
        pass
    for name in scale_names:
        value = values[name]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value < 0
        ):
            raise ValueError(f"{name} must be a finite non-negative number")
        else:
            pass
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
    params: dict[str, JsonValue],
) -> tuple[bool | None, JsonValue]:
    enabled = params.get("enable_cache_dit")
    if enabled is not None and type(enabled) is not bool:
        raise ValueError("enable_cache_dit must be a boolean")
    else:
        pass
    if enabled is False:
        return False, None
    else:
        pass
    raw = params.get("cache_dit_params")
    if raw is None:
        return enabled, None
    else:
        pass
    if enabled is None:
        return enabled, raw
    else:
        pass
    if not isinstance(raw, dict):
        error_message = "cache_dit_params must be a mapping"
        raise ValueError(error_message)  # noqa: TRY004 -- request validation
    else:
        pass
    unknown = set(raw) - CACHE_DIT_PARAMETER_NAMES
    if unknown:
        raise ValueError(f"unsupported Cache-DiT parameters: {sorted(unknown)}")
    else:
        pass
    values: dict[str, int | float] = {}
    for name, value in raw.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            error_message = f"cache_dit_params.{name} must be numeric"
            raise ValueError(error_message)  # noqa: TRY004 -- request validation
        else:
            pass
        if name == "residual_diff_threshold":
            if not 0 <= value <= 1:
                raise ValueError(f"cache_dit_params.{name} must be between 0 and 1")
            else:
                pass
        elif name == "Fn_compute_blocks" and (type(value) is not int or value < 1):
            raise ValueError(f"cache_dit_params.{name} must be a positive integer")
        elif type(value) is not int or value < 0:
            raise ValueError(f"cache_dit_params.{name} must be a non-negative integer")
        else:
            pass
        values[name] = value
    return enabled, values


def resolve_cache_dit_params(
    params: dict[str, JsonValue], default_enabled: bool
) -> tuple[bool, dict[str, int | float] | None]:
    resolved = dict(params)
    if resolved.get("enable_cache_dit") is None:
        resolved["enable_cache_dit"] = default_enabled
    else:
        pass
    enabled, cache_params = _cache_dit_values(resolved)
    if cache_params is None:
        return bool(enabled), None
    elif isinstance(cache_params, dict):
        validated_params: dict[str, int | float] = {}
        for name, value in cache_params.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                error_message = f"cache_dit_params.{name} must be numeric"
                raise ValueError(error_message)  # noqa: TRY004 -- request validation
            else:
                validated_params[name] = value
        return bool(enabled), validated_params
    else:
        raise ValueError("cache_dit_params must be a mapping")


def resolve_cache_dit_defaults(
    default_enabled: bool,
    params: dict[str, int | float] | None,
) -> tuple[bool, dict[str, int | float] | None]:
    if params is None:
        return default_enabled, None
    else:
        pass
    _, validated_params = resolve_cache_dit_params(
        {"enable_cache_dit": True, "cache_dit_params": params}, True
    )
    return default_enabled, validated_params


def image_guidance_branch_count(cfg_scale: float, img_cfg_scale: float) -> int:
    if cfg_scale == 1 and img_cfg_scale == 1:
        return 1
    else:
        pass
    if img_cfg_scale == 1 or cfg_scale == img_cfg_scale:
        return 2
    else:
        return 3


def cache_dit_batch_key(params: JsonValue) -> str:
    """Allow deferred JSON parameters in scheduler compatibility keys."""
    return json.dumps(params, sort_keys=True, separators=(",", ":"))
