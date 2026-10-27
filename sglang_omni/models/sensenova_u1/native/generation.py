# SPDX-License-Identifier: Apache-2.0
"""Execute native SenseNova image generation."""
from __future__ import annotations

from dataclasses import dataclass

from sglang.multimodal_gen.runtime.disaggregation.roles import RoleType
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import OutputBatch, Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.base import PipelineStage
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from transformers import PreTrainedTokenizerBase

from sglang_omni.models.sensenova_u1.native.options import (
    DEFAULT_CFG_INTERVAL,
    DEFAULT_CFG_NORM,
    DEFAULT_ENABLE_TIMESTEP_SHIFT,
    DEFAULT_T_EPS,
    DEFAULT_THINK_MODE,
    DEFAULT_TIMESTEP_SHIFT,
    SENSENOVA_U1_REQUEST_EXTRA_KEY,
    ClassifierFreeGuidanceNormalization,
)
from sglang_omni.vendor.sensenova_u1.modeling_neo_chat import NEOChatModel


@dataclass(frozen=True)
class SenseNovaU1GenerationOptions:
    """Resolve model extras independently of sampling-parameter overrides."""

    cfg_norm: ClassifierFreeGuidanceNormalization = DEFAULT_CFG_NORM
    timestep_shift: float = DEFAULT_TIMESTEP_SHIFT
    enable_timestep_shift: bool = DEFAULT_ENABLE_TIMESTEP_SHIFT
    cfg_interval: tuple[float, float] = DEFAULT_CFG_INTERVAL
    t_eps: float = DEFAULT_T_EPS
    think_mode: bool = DEFAULT_THINK_MODE

    @classmethod
    def from_batch(cls, batch: Req) -> SenseNovaU1GenerationOptions:
        generation_extra = batch.extra.get(SENSENOVA_U1_REQUEST_EXTRA_KEY, {})
        return cls(
            cfg_norm=generation_extra.get("cfg_norm", DEFAULT_CFG_NORM),
            timestep_shift=float(
                generation_extra.get("timestep_shift", DEFAULT_TIMESTEP_SHIFT)
            ),
            enable_timestep_shift=bool(
                generation_extra.get(
                    "enable_timestep_shift", DEFAULT_ENABLE_TIMESTEP_SHIFT
                )
            ),
            cfg_interval=tuple(
                generation_extra.get("cfg_interval", DEFAULT_CFG_INTERVAL)
            ),
            t_eps=float(generation_extra.get("t_eps", DEFAULT_T_EPS)),
            think_mode=bool(generation_extra.get("think_mode", DEFAULT_THINK_MODE)),
        )


class SenseNovaU1GenerationStage(PipelineStage):
    def __init__(self, model: NEOChatModel, tokenizer: PreTrainedTokenizerBase) -> None:
        super().__init__()
        self.model: NEOChatModel = model
        self.tokenizer: PreTrainedTokenizerBase = tokenizer

    @property
    def role_affinity(self) -> RoleType:
        return RoleType.DENOISER

    def forward(self, batch: Req, server_args: ServerArgs) -> OutputBatch:
        generation_options = SenseNovaU1GenerationOptions.from_batch(batch)
        if int(batch.num_outputs_per_prompt) != 1:
            raise ValueError(
                "SenseNova-U1 expects output expansion before generation; "
                f"got num_outputs_per_prompt={batch.num_outputs_per_prompt}."
            )
        else:
            pass
        prompt_list = batch.prompt if isinstance(batch.prompt, list) else [batch.prompt]
        prompt_count = len(prompt_list)
        if prompt_count == 0:
            raise ValueError(
                "SenseNova-U1 dynamic batch must contain at least one prompt"
            )
        else:
            pass
        if prompt_count > 1 and generation_options.think_mode:
            raise ValueError(
                "SenseNova-U1 dynamic batching does not support think_mode"
            )
        else:
            pass

        dynamic_seeds = batch.extra.get("dynamic_batch_seeds")
        if dynamic_seeds is None:
            dynamic_seeds = batch.seed if isinstance(batch.seed, list) else [batch.seed]
        elif not isinstance(dynamic_seeds, list):
            dynamic_seeds = [dynamic_seeds]
        else:
            pass
        seeds = []
        for seed in dynamic_seeds:
            if isinstance(seed, list):
                if len(seed) != 1:
                    raise ValueError(
                        "SenseNova-U1 dynamic batching requires one seed per request"
                    )
                else:
                    pass
                seed = seed[0]
            else:
                pass
            seeds.append(int(seed))
        if len(seeds) != prompt_count:
            raise ValueError(
                "SenseNova-U1 dynamic batch requires one seed per prompt; "
                f"got {len(seeds)} seeds for {prompt_count} prompts"
            )
        else:
            pass
        seed = seeds[0] if prompt_count == 1 else seeds

        generation_result = self.model.t2i_generate(
            self.tokenizer,
            batch.prompt,
            image_size=(int(batch.width), int(batch.height)),
            cfg_scale=float(batch.guidance_scale),
            cfg_norm=generation_options.cfg_norm,
            timestep_shift=generation_options.timestep_shift,
            enable_timestep_shift=generation_options.enable_timestep_shift,
            cfg_interval=generation_options.cfg_interval,
            num_steps=int(batch.num_inference_steps),
            batch_size=prompt_count,
            t_eps=generation_options.t_eps,
            think_mode=generation_options.think_mode,
            seed=seed,
        )
        think_text = None
        if generation_options.think_mode:
            image_tensors, think_text = generation_result
        else:
            image_tensors = generation_result

        image_tensors = ((image_tensors.float() + 1.0) * 0.5).clamp(0, 1)
        output_images = [image.contiguous() for image in image_tensors]
        usage = {"think_text": think_text} if think_text is not None else None
        return OutputBatch(
            output=output_images,
            metrics=batch.metrics,
            usage=usage,
        )
