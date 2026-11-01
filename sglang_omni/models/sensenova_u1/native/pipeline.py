# SPDX-License-Identifier: Apache-2.0
"""Compose the native SenseNova text-to-image pipeline."""
from __future__ import annotations

import logging

import torch
from sglang.multimodal_gen.runtime.pipelines_core.composed_pipeline_base import (
    ComposedPipelineBase,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from transformers import PreTrainedTokenizerBase

from sglang_omni.models.sensenova_u1.native.generation import SenseNovaU1GenerationStage
from sglang_omni.models.sensenova_u1.native.image_edit import (
    SenseNovaInputValidationStage,
)
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
        if loaded_modules is not None and {"model", "tokenizer"} <= set(loaded_modules):
            return loaded_modules
        else:
            pass

        modules = load_model_and_tokenizer(self.model_path, server_args)
        logger.info(f"Loaded SenseNova-U1 model from {self.model_path}")
        return modules

    def create_pipeline_stages(self, server_args: ServerArgs) -> None:
        self.add_stage(SenseNovaInputValidationStage())
        self.add_stage(
            SenseNovaU1GenerationStage(
                model=self.get_module("model"),
                tokenizer=self.get_module("tokenizer"),
            ),
            "sensenova_u1_generation_stage",
        )
