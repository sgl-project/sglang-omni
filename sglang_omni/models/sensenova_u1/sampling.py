# SPDX-License-Identifier: Apache-2.0
"""Validated SenseNova-U1 image generation options."""

from __future__ import annotations

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

    @classmethod
    def from_params(cls, params: dict[str, Any]) -> SenseNovaU1Sampling:
        return cls(**_validated_values(cls, params, ("guidance_scale",)))


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
        return cls(
            **_validated_values(cls, params, ("guidance_scale", "img_cfg_scale")),
            input_max_pixels=input_max_pixels,
            do_resize=do_resize,
            size_explicit=size_explicit,
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
