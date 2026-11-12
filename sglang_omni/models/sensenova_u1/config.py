# SPDX-License-Identifier: Apache-2.0
"""SenseNova-U1 image generation configuration backed by SGLang multimodal_gen."""

from typing import ClassVar

from sglang_omni.native_generation.config import NativeGenerationPipelineConfig


class SenseNovaU1PipelineConfig(NativeGenerationPipelineConfig):
    architecture: ClassVar[str] = "NEOChatModel"


EntryClass = SenseNovaU1PipelineConfig
