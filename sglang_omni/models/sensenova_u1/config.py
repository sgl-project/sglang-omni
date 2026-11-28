# SPDX-License-Identifier: Apache-2.0
"""SenseNova-U1 image generation configuration backed by SGLang multimodal_gen."""

import os
from typing import ClassVar

from sglang_omni.native_generation.config import NativeGenerationPipelineConfig
from sglang_omni.platforms import current_platform


class SenseNovaU1PipelineConfig(NativeGenerationPipelineConfig):
    architecture: ClassVar[str] = "NEOChatModel"

    def resolved_env_defaults(self) -> dict[str, str]:
        return {
            "SGLANG_EXTERNAL_MODEL_PACKAGE": "sglang_omni.models.sensenova_u1.native",
            **super().resolved_env_defaults(),
        }

    def resolved_stage_env_defaults(self, stage_name: str) -> dict[str, str]:
        env_defaults = super().resolved_stage_env_defaults(stage_name)
        if (
            current_platform.is_npu()
            and "PYTORCH_NPU_ALLOC_CONF" not in env_defaults
            and "PYTORCH_NPU_ALLOC_CONF" not in os.environ
            and "PYTORCH_ALLOC_CONF" not in env_defaults
            and "PYTORCH_ALLOC_CONF" not in os.environ
        ):
            # note (Wes): Edit prefix attention buffers change size per query and fragment fixed segments.
            env_defaults["PYTORCH_NPU_ALLOC_CONF"] = "expandable_segments:True"
        else:
            pass
        return env_defaults


EntryClass = SenseNovaU1PipelineConfig
