# SPDX-License-Identifier: Apache-2.0
"""Validated SenseNova-U1 image generation options."""

from __future__ import annotations

import json
from typing import Annotated, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    StrictBool,
    StrictInt,
    field_validator,
)

MAX_GENERATION_SEED = 2**64 - 1


def validate_seed_range(seed: int, image_count: int) -> None:
    """Every generated seed must fit the random number generator's uint64 range."""
    if seed + image_count - 1 > MAX_GENERATION_SEED:
        raise ValueError(f"seed + n - 1 must not exceed {MAX_GENERATION_SEED}")
    else:
        pass


def resolve_generation_parameters(
    parameters: dict[str, JsonValue],
) -> dict[str, JsonValue]:
    """Apply generate-stage overrides before sampling and resource validation."""
    resolved_parameters = dict(parameters)
    stage_parameters = resolved_parameters.pop("stage_params", None)
    if stage_parameters is None:
        return resolved_parameters
    elif not isinstance(stage_parameters, dict):
        raise ValueError("stage_params must be a mapping")
    else:
        generate_parameters = stage_parameters.get("generate")
    if generate_parameters is None:
        return resolved_parameters
    elif not isinstance(generate_parameters, dict):
        raise ValueError("stage_params.generate must be a mapping")
    else:
        resolved_parameters.update(generate_parameters)
    return resolved_parameters


IMAGE_DIMENSION_MULTIPLE = 32
# note (Wes): Supported vision checkpoints cache 10,000 positions per axis at 16 pixels per patch.
MAX_IMAGE_DIMENSION_PIXELS = 160000
MIN_INPUT_MAX_PIXELS = 512 * 512

ImageDimension = Annotated[
    int,
    Field(
        gt=0,
        le=MAX_IMAGE_DIMENSION_PIXELS,
        multiple_of=IMAGE_DIMENSION_MULTIPLE,
        strict=True,
    ),
]
ImageCount = Annotated[int, Field(ge=1, le=10, strict=True)]
GenerationSeed = Annotated[int, Field(ge=0, strict=True)]
GenerationSteps = Annotated[int, Field(gt=0, strict=True)]
GuidanceScale = Annotated[float, Field(ge=0, allow_inf_nan=False, strict=True)]

GENERATION_TIMESTEP_SHIFT = 3.0
GENERATION_TIME_EPSILON = 0.02


class SenseNovaU1Sampling(BaseModel):
    model_config = ConfigDict(frozen=True)

    width: ImageDimension = 2048
    height: ImageDimension = 2048
    num_inference_steps: GenerationSteps = 50
    guidance_scale: GuidanceScale = 4.0
    seed: GenerationSeed = 42
    enable_cache_dit: StrictBool | None = None
    cache_dit_params: JsonValue = None
    n: ImageCount = 1

    def batch_key(self) -> tuple[int, int, int, float, bool | None, str]:
        """Group requests with identical denoising and cache settings."""
        return (
            self.width,
            self.height,
            self.num_inference_steps,
            self.guidance_scale,
            self.enable_cache_dit,
            cache_dit_batch_key(self.cache_dit_params),
        )

    @classmethod
    def from_params(cls, params: dict[str, JsonValue]) -> Self:
        parameters = resolve_generation_parameters(params)
        if parameters.get("think_mode", False) is not False:
            raise ValueError("SenseNova-U1 think_mode is not supported")
        else:
            sampling = cls.model_validate(parameters)
        validate_seed_range(sampling.seed, sampling.n)
        if (
            isinstance(sampling, SenseNovaU1ImageEditSampling)
            and image_guidance_branch_count(
                sampling.guidance_scale, sampling.img_cfg_scale
            )
            > 2
        ):
            parameters["cache_dit_params"] = None
        else:
            pass
        cache_parameters: JsonValue
        if sampling.enable_cache_dit is False:
            cache_parameters = None
        elif sampling.enable_cache_dit is None:
            cache_parameters = parameters.get("cache_dit_params")
        else:
            cache_parameters = validate_cache_dit_params(
                parameters.get("cache_dit_params")
            )
        return sampling.model_copy(update={"cache_dit_params": cache_parameters})


class SenseNovaU1ImageEditSampling(SenseNovaU1Sampling):
    """Image editing runs one output per request; the Images API sends one request per output."""

    n: StrictInt = 1
    img_cfg_scale: GuidanceScale = 1.0
    input_max_pixels: (
        Annotated[int, Field(ge=MIN_INPUT_MAX_PIXELS, strict=True)] | None
    ) = None
    do_resize: StrictBool = True
    size_explicit: StrictBool = False

    @field_validator("n")
    @classmethod
    def validate_single_output(cls, n: int) -> int:
        if n != 1:
            raise ValueError(
                "SenseNova-U1 image edits produce one image per request; "
                "use n on /v1/images/edits or send one request per seed"
            )
        else:
            return n


CACHE_DIT_PARAMETER_NAMES = frozenset(
    {
        "Fn_compute_blocks",
        "Bn_compute_blocks",
        "max_warmup_steps",
        "residual_diff_threshold",
        "max_continuous_cached_steps",
    }
)


def validate_cache_dit_params(
    cache_parameters: JsonValue,
) -> dict[str, int | float] | None:
    if cache_parameters is None:
        return None
    else:
        pass
    if not isinstance(cache_parameters, dict):
        error_message = "cache_dit_params must be a mapping"
        raise ValueError(error_message)  # noqa: TRY004 -- request validation
    else:
        pass
    unknown = set(cache_parameters) - CACHE_DIT_PARAMETER_NAMES
    if unknown:
        raise ValueError(f"unsupported Cache-DiT parameters: {sorted(unknown)}")
    else:
        pass
    values: dict[str, int | float] = {}
    for name, value in cache_parameters.items():
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
    return values


def resolve_cache_dit_params(
    params: dict[str, JsonValue], default_enabled: bool
) -> tuple[bool, dict[str, int | float] | None]:
    enabled = params.get("enable_cache_dit")
    if enabled is None:
        enabled = default_enabled
    else:
        pass
    if not isinstance(enabled, bool):
        raise ValueError("enable_cache_dit must be a boolean")
    elif enabled:
        return True, validate_cache_dit_params(params.get("cache_dit_params"))
    else:
        return False, None


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
