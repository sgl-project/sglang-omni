# SPDX-License-Identifier: Apache-2.0
"""Compose the SenseNova generation and editing pipeline."""
from __future__ import annotations

import logging
import os

import torch
from sglang.multimodal_gen import envs
from sglang.multimodal_gen.runtime.pipelines_core.composed_pipeline_base import (
    ComposedPipelineBase,
)
from sglang.multimodal_gen.runtime.platforms import current_platform
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from transformers import PreTrainedTokenizerBase

from sglang_omni.models.sensenova_u1.native.cache_dit import check_compute_blocks
from sglang_omni.models.sensenova_u1.native.generation import SenseNovaU1GenerationStage
from sglang_omni.models.sensenova_u1.native.image_edit import SenseNovaU1InputStage
from sglang_omni.models.sensenova_u1.native.loader import load_model_and_tokenizer

logger = logging.getLogger(__name__)


class SenseNovaU1Pipeline(ComposedPipelineBase):
    pipeline_name: str = "SenseNovaU1Pipeline"
    _required_config_modules: list[str] = []

    def load_modules(
        self,
        server_args: ServerArgs,
        loaded_modules: (
            dict[str, torch.nn.Module | PreTrainedTokenizerBase] | None
        ) = None,
    ) -> dict[str, torch.nn.Module | PreTrainedTokenizerBase]:
        allocator_env_keys = (
            {"PYTORCH_NPU_ALLOC_CONF", "PYTORCH_ALLOC_CONF"}
            if current_platform.is_npu()
            else {"PYTORCH_CUDA_ALLOC_CONF", "PYTORCH_ALLOC_CONF"}
        )
        if not allocator_env_keys & os.environ.keys():
            # note (Richard Wang): Edit prefix attention buffers change size per query and fragment fixed segments.
            try:
                torch._C._accelerator_setAllocatorSettings(  # noqa: leading-underscore  # upstream spelling, or the public name is already taken
                    "expandable_segments:True"
                )
            except (RuntimeError, NotImplementedError) as allocator_error:
                if current_platform.is_npu():
                    logger.warning(
                        f"Could not configure expandable_segments on NPU; "
                        f"set PYTORCH_NPU_ALLOC_CONF=expandable_segments:True before launch: {allocator_error}"
                    )
                else:
                    raise
        else:
            pass
        if loaded_modules is not None and {"model", "tokenizer"} <= set(loaded_modules):
            return loaded_modules
        else:
            pass

        modules = load_model_and_tokenizer(self.model_path, server_args)
        logger.info(f"Loaded SenseNova-U1 model from {self.model_path}")
        check_compute_blocks(
            envs.SGLANG_CACHE_DIT_FN,
            envs.SGLANG_CACHE_DIT_BN,
            modules["model"].config.llm_config.num_hidden_layers,
        )
        return modules

    def create_pipeline_stages(self, server_args: ServerArgs) -> None:
        self.add_stage(SenseNovaU1InputStage())
        self.add_stage(
            SenseNovaU1GenerationStage(
                model=self.get_module("model"),
                tokenizer=self.get_module("tokenizer"),
            ),
            "sensenova_u1_generation_stage",
        )
