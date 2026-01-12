# SPDX-License-Identifier: Apache-2.0
"""SGLang-Omni: Multi-stage pipeline framework for omni models."""

from sglang_omni.engines.base import EchoEngine, Engine
from sglang_omni.pipeline.coordinator import Coordinator
from sglang_omni.pipeline.input_handler import (
    AggregatedInput,
    DirectInput,
    InputHandler,
)
from sglang_omni.pipeline.stage import Stage
from sglang_omni.pipeline.worker import Worker

# Re-export from submodules for convenience
from sglang_omni.proto import (
    AbortMessage,
    CompleteMessage,
    DataReadyMessage,
    RequestState,
    SHMMetadata,
    StageInfo,
)

__version__ = "0.1.0"

__all__ = [
    # Core classes
    "Coordinator",
    "Stage",
    "Worker",
    "Engine",
    "EchoEngine",
    # Input handlers
    "InputHandler",
    "DirectInput",
    "AggregatedInput",
    # Types
    "RequestState",
    "StageInfo",
    "DataReadyMessage",
    "AbortMessage",
    "CompleteMessage",
    "SHMMetadata",
]
