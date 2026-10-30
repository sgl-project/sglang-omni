# SPDX-License-Identifier: Apache-2.0
"""Resolve SenseNova sampling parameters and request extras."""
from argparse import Namespace
from dataclasses import dataclass, field
from typing import TypedDict

from pydantic import BaseModel, Field, JsonValue
from sglang.multimodal_gen.configs.sample.sampling_params import (
    DataType,
    SamplingParams,
)

from sglang_omni.models.sensenova_u1.native.options import (
    DEFAULT_CFG_INTERVAL,
    DEFAULT_CFG_NORM,
    DEFAULT_ENABLE_TIMESTEP_SHIFT,
    DEFAULT_T_EPS,
    DEFAULT_THINK_MODE,
    DEFAULT_TIMESTEP_SHIFT,
    MIN_INPUT_MAX_PIXELS,
    RESOLUTION_ALIGNMENT,
    SENSENOVA_U1_CFG_NORM_CHOICES,
    SENSENOVA_U1_REQUEST_EXTRA_KEY,
    ClassifierFreeGuidanceNormalization,
)

PUBLIC_OVERRIDE_FIELDS = {
    "prompt",
    "prompt_path",
    "image_path",
    "img_cfg_scale",
    "input_max_pixels",
    "do_resize",
    "height",
    "width",
    "num_inference_steps",
    "guidance_scale",
    "num_outputs_per_prompt",
    "seed",
    "save_output",
    "output_path",
    "output_file_name",
    "output_quality",
    "output_compression",
    "quality",
    "profile",
    "profile_all_stages",
    "num_profiled_timesteps",
    "perf_dump_path",
}


class SenseNovaGenerationExtra(TypedDict):
    cfg_norm: ClassifierFreeGuidanceNormalization
    timestep_shift: float
    enable_timestep_shift: bool
    cfg_interval: tuple[float, float]
    t_eps: float
    think_mode: bool
    img_cfg_scale: float
    input_max_pixels: int | None
    do_resize: bool


class SenseNovaEditParameters(BaseModel):
    img_cfg_scale: float = Field(ge=0, allow_inf_nan=False)
    input_max_pixels: int | None = Field(ge=MIN_INPUT_MAX_PIXELS)
    do_resize: bool


@dataclass
class SenseNovaU1SamplingParams(SamplingParams):
    data_type: DataType = field(default=DataType.IMAGE, init=False)
    height: int = 2048
    width: int = 2048
    num_frames: int = 1
    fps: int = 1
    num_inference_steps: int = 50
    guidance_scale: float = 4.0
    cfg_norm: ClassifierFreeGuidanceNormalization = DEFAULT_CFG_NORM
    timestep_shift: float = DEFAULT_TIMESTEP_SHIFT
    enable_timestep_shift: bool = DEFAULT_ENABLE_TIMESTEP_SHIFT
    cfg_interval: tuple[float, float] = DEFAULT_CFG_INTERVAL
    t_eps: float = DEFAULT_T_EPS
    think_mode: bool = DEFAULT_THINK_MODE
    img_cfg_scale: float = 1.0
    input_max_pixels: int | None = None
    do_resize: bool = True
    negative_prompt: None = field(default=None, init=False)

    @classmethod
    def supported_override_fields(cls) -> set[str]:
        return set(PUBLIC_OVERRIDE_FIELDS)

    @classmethod
    def image_request_extra_fields(cls) -> frozenset[str]:
        return frozenset({"img_cfg_scale", "input_max_pixels", "do_resize"})

    @classmethod
    def get_cli_args(
        cls, args: Namespace
    ) -> dict[str, str | bool | int | float | list[str] | list[int] | None]:
        cli_args = super().get_cli_args(args)
        return {
            field_name: value
            for field_name, value in cli_args.items()
            if field_name in PUBLIC_OVERRIDE_FIELDS
        }

    def __post_init__(self) -> None:
        if isinstance(self.cfg_interval, list):
            self.cfg_interval = tuple(float(fraction) for fraction in self.cfg_interval)
        else:
            pass
        super().__post_init__()

    def _validate(
        self,
    ) -> None:  # noqa: leading-underscore -- SamplingParams validation hook.
        super()._validate()  # noqa: leading-underscore -- SamplingParams validation hook.
        edit_parameters = SenseNovaEditParameters.model_validate(
            {
                "img_cfg_scale": self.img_cfg_scale,
                "input_max_pixels": self.input_max_pixels,
                "do_resize": self.do_resize,
            }
        )
        self.img_cfg_scale = edit_parameters.img_cfg_scale
        self.input_max_pixels = edit_parameters.input_max_pixels
        self.do_resize = edit_parameters.do_resize
        if self.image_path:
            if not isinstance(self.prompt, str):
                raise ValueError(
                    "SenseNova editing requires one text prompt per request"
                )
            else:
                pass
            if (
                self.think_mode
                or self.cfg_norm != DEFAULT_CFG_NORM
                or self.cfg_interval != DEFAULT_CFG_INTERVAL
                or self.timestep_shift != DEFAULT_TIMESTEP_SHIFT
                or not self.enable_timestep_shift
                or self.t_eps != DEFAULT_T_EPS
            ):
                raise ValueError(
                    "SenseNova editing requires the default generation controls"
                )
            else:
                pass
        else:
            pass
        if (
            self.width % RESOLUTION_ALIGNMENT != 0
            or self.height % RESOLUTION_ALIGNMENT != 0
        ):
            raise ValueError(
                "SenseNova-U1 requires width and height to be divisible by "
                f"{RESOLUTION_ALIGNMENT}, got "
                f"{self.width}x{self.height}."
            )
        else:
            pass
        if self.num_frames != 1:
            raise ValueError(
                f"SenseNova-U1 is an image model and requires num_frames=1, got {self.num_frames}."
            )
        else:
            pass
        if self.cfg_norm not in SENSENOVA_U1_CFG_NORM_CHOICES:
            raise ValueError(
                f"cfg_norm must be one of {SENSENOVA_U1_CFG_NORM_CHOICES}, "
                f"got {self.cfg_norm!r}"
            )
        else:
            pass
        if len(self.cfg_interval) != 2:
            raise ValueError("cfg_interval must contain exactly two values")
        else:
            pass
        guidance_start_fraction, guidance_end_fraction = self.cfg_interval
        if (
            not 0.0
            <= float(guidance_start_fraction)
            <= float(guidance_end_fraction)
            <= 1.0
        ):
            raise ValueError(
                f"cfg_interval must satisfy 0 <= start <= end <= 1, got {self.cfg_interval!r}"
            )
        else:
            pass

    def build_request_extra(self) -> dict[str, JsonValue | SenseNovaGenerationExtra]:
        request_extra = super().build_request_extra()
        request_extra[SENSENOVA_U1_REQUEST_EXTRA_KEY] = {
            "cfg_norm": self.cfg_norm,
            "timestep_shift": self.timestep_shift,
            "enable_timestep_shift": self.enable_timestep_shift,
            "cfg_interval": tuple(self.cfg_interval),
            "t_eps": self.t_eps,
            "think_mode": self.think_mode,
            "img_cfg_scale": self.img_cfg_scale,
            "input_max_pixels": self.input_max_pixels,
            "do_resize": self.do_resize,
        }
        return request_extra
