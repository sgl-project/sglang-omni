# SPDX-License-Identifier: Apache-2.0
"""SenseNova-U1 image generation configuration backed by SGLang multimodal_gen."""

from typing import ClassVar

from sglang_omni.native_generation.config import NativeGenerationPipelineConfig


class SenseNovaU1PipelineConfig(NativeGenerationPipelineConfig):
    architecture: ClassVar[str] = "NEOChatModel"

    def resolved_env_defaults(self) -> dict[str, str]:
        return {
            "SGLANG_EXTERNAL_MODEL_PACKAGE": "sglang_omni.models.sensenova_u1.native",
            **super().resolved_env_defaults(),
        }


EntryClass = SenseNovaU1PipelineConfig
