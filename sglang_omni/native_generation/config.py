# SPDX-License-Identifier: Apache-2.0
"""Configuration of a model that native SGLang multimodal_gen serves."""

from sglang_omni.config import FactoryArgs, PipelineConfig, StageConfig
from sglang_omni.native_generation import ROUTE_PREFIXES


class NativeGenerationPipelineConfig(PipelineConfig):
    model_path: str
    stages: list[StageConfig] = [
        StageConfig(
            name="generation",
            process="generation",
            factory_path="sglang_omni.native_generation.stage.create_generation_scheduler",
            allow_child_processes=True,
            factory=FactoryArgs(),
            gpu=0,
            terminal=True,
            # The stage serves native SGLang's media routes from its own process.
            http_routes=list(ROUTE_PREFIXES),
        )
    ]
