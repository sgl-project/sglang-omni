# SPDX-License-Identifier: Apache-2.0
"""Validated first-pass T2I options, matching the source pipeline defaults."""

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
        if params.get("think_mode", False) is not False:
            raise ValueError("SenseNova-U1 think_mode is not supported yet")
        if params.get("n", 1) != 1:
            raise ValueError("SenseNova-U1 currently supports exactly one image")
        values = {
            name: params.get(name, getattr(cls, name))
            for name in (
                "width",
                "height",
                "num_inference_steps",
                "guidance_scale",
                "seed",
            )
        }
        for name in ("width", "height", "num_inference_steps", "seed"):
            value = values[name]
            if type(value) is not int or value < (0 if name == "seed" else 1):
                raise ValueError(
                    f"{name} must be a {'non-negative' if name == 'seed' else 'positive'} integer"
                )
        if values["width"] % 32 or values["height"] % 32:
            raise ValueError("SenseNova-U1 width and height must be divisible by 32")
        scale = values["guidance_scale"]
        if (
            isinstance(scale, bool)
            or not isinstance(scale, (int, float))
            or not math.isfinite(scale)
            or scale < 0
        ):
            raise ValueError("guidance_scale must be a finite non-negative number")
        return cls(**values)
