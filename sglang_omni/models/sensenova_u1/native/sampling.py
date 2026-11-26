# SPDX-License-Identifier: Apache-2.0
"""Resolve SenseNova sampling parameters and request extras."""
import math
from argparse import Namespace
from dataclasses import dataclass, field
from typing import NotRequired, TypedDict

from pydantic import BaseModel, Field, JsonValue
from sglang.multimodal_gen import envs
from sglang.multimodal_gen.configs.sample.sampling_params import (
    DataType,
    SamplingParams,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs

from sglang_omni.models.sensenova_u1.native.cache_dit import (
    check_compute_blocks,
    checkpoint_decoder_layer_count,
    guidance_branch_count,
    request_cache_dit_overrides,
    resolve_cache_dit_config,
    resolve_cache_dit_switch,
)
from sglang_omni.models.sensenova_u1.native.image_edit import (
    MIN_REFERENCE_PIXELS,
    TOTAL_REFERENCE_MAX_PIXELS,
)
from sglang_omni.models.sensenova_u1.native.options import (
    DEFAULT_CFG_INTERVAL,
    DEFAULT_CFG_NORM,
    DEFAULT_ENABLE_TIMESTEP_SHIFT,
    DEFAULT_T_EPS,
    DEFAULT_THINK_MODE,
    DEFAULT_TIMESTEP_SHIFT,
    RESOLUTION_ALIGNMENT,
    SENSENOVA_U1_CFG_NORM_CHOICES,
    SENSENOVA_U1_REQUEST_EXTRA_KEY,
    ClassifierFreeGuidanceNormalization,
)

MAX_REFERENCE_COUNT = TOTAL_REFERENCE_MAX_PIXELS // MIN_REFERENCE_PIXELS
PUBLIC_OVERRIDE_FIELDS = {
    "prompt",
    "prompt_path",
    "image_path",
    "img_cfg_scale",
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
    "enable_cache_dit",
    "cache_dit_params",
}


class SenseNovaGenerationExtra(TypedDict):
    cfg_norm: ClassifierFreeGuidanceNormalization
    timestep_shift: float
    enable_timestep_shift: bool
    cfg_interval: tuple[float, float]
    t_eps: float
    think_mode: bool
    img_cfg_scale: NotRequired[float]


class SenseNovaEditParameters(BaseModel):
    img_cfg_scale: float = Field(ge=0, allow_inf_nan=False)


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
    negative_prompt: None = field(default=None, init=False)

    @classmethod
    def supported_override_fields(cls) -> set[str]:
        return set(PUBLIC_OVERRIDE_FIELDS)

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

    def _adjust(
        self, server_args: ServerArgs
    ) -> None:  # noqa: leading-underscore -- SamplingParams adjustment hook.
        super()._adjust(server_args)  # noqa: leading-underscore -- SamplingParams hook.
        self.resolve_cache_dit(server_args.model_path, server_args.revision)

    def resolve_cache_dit(self, model_path: str, revision: str | None) -> None:
        """Settle this request's Cache-DiT switch and overrides, rejecting what it cannot run."""
        if self.enable_cache_dit is not None and not isinstance(
            self.enable_cache_dit, bool
        ):
            # note (Wes): the JSON generations route passes this field through unchecked, so a string "false" would enable the cache.
            raise ValueError(
                f"enable_cache_dit must be true, false or omitted, got {self.enable_cache_dit!r}"
            )
        else:
            pass
        switched_on = (
            envs.SGLANG_CACHE_DIT_ENABLED
            if self.enable_cache_dit is None
            else self.enable_cache_dit
        )
        if switched_on:
            # note (Wes): check the settings before the guidance fallback, so a request the server runs uncached still gets a 400 for bad settings.
            overrides = request_cache_dit_overrides(self.cache_dit_params)
            config = resolve_cache_dit_config(overrides, int(self.num_inference_steps))
            check_compute_blocks(
                config.Fn_compute_blocks,
                config.Bn_compute_blocks,
                checkpoint_decoder_layer_count(model_path, revision),
            )
        else:
            overrides = {}
        self.enable_cache_dit = resolve_cache_dit_switch(
            self.enable_cache_dit,
            branch_count=guidance_branch_count(
                is_edit=bool(self.image_path),
                cfg_scale=float(self.guidance_scale),
                img_cfg_scale=float(self.img_cfg_scale),
            ),
            cfg_interval=tuple(self.cfg_interval),
        )
        if self.enable_cache_dit:
            self.cache_dit_params = overrides or None
        else:
            # note (Wes): a disabled request ignores its settings, so it batches with other disabled requests.
            self.cache_dit_params = None

    def _validate(
        self,
    ) -> None:  # noqa: leading-underscore -- SamplingParams validation hook.
        super()._validate()  # noqa: leading-underscore -- SamplingParams validation hook.
        if self.image_path:
            edit_parameters = SenseNovaEditParameters.model_validate(
                {"img_cfg_scale": self.img_cfg_scale}
            )
            self.img_cfg_scale = edit_parameters.img_cfg_scale
            if not isinstance(self.prompt, str):
                raise ValueError(
                    "SenseNova editing requires one text prompt per request"
                )
            else:
                pass
            if "<IMG_CONTEXT>" in self.prompt:
                raise ValueError(
                    "SenseNova editing prompts must not contain <IMG_CONTEXT>; "
                    "use <image> to reference an input image"
                )
            else:
                pass
            reference_count = (
                len(self.image_path) if isinstance(self.image_path, list) else 1
            )
            if reference_count > MAX_REFERENCE_COUNT:
                raise ValueError(
                    "SenseNova editing accepts at most "
                    f"{MAX_REFERENCE_COUNT} reference images"
                )
            else:
                pass
            if self.prompt.count("<image>") > reference_count:
                raise ValueError(
                    "Prompt has more image placeholders than reference images"
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
        elif self.img_cfg_scale == type(self).img_cfg_scale:
            # note (Richard Wang): keep default edit-only guidance out of T2I request pickles.
            self.__dict__.pop("img_cfg_scale", None)
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
        if not 0.0 < self.timestep_shift < math.inf:
            raise ValueError(
                f"timestep_shift must be a finite number above 0, got {self.timestep_shift!r}"
            )
        else:
            pass
        if not 0.0 <= self.t_eps < 1.0:
            raise ValueError(f"t_eps must satisfy 0 <= t_eps < 1, got {self.t_eps!r}")
        else:
            pass

    def build_request_extra(self) -> dict[str, JsonValue | SenseNovaGenerationExtra]:
        request_extra = super().build_request_extra()
        generation_extra: SenseNovaGenerationExtra = {
            "cfg_norm": self.cfg_norm,
            "timestep_shift": self.timestep_shift,
            "enable_timestep_shift": self.enable_timestep_shift,
            "cfg_interval": tuple(self.cfg_interval),
            "t_eps": self.t_eps,
            "think_mode": self.think_mode,
        }
        if self.image_path:
            generation_extra["img_cfg_scale"] = self.img_cfg_scale
        else:
            pass
        request_extra[SENSENOVA_U1_REQUEST_EXTRA_KEY] = generation_extra
        return request_extra
