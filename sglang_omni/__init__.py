# SPDX-License-Identifier: Apache-2.0
"""SGLang-Omni: Multi-stage pipeline framework for omni models."""

from sglang_omni.engines.base import EchoEngine, Engine
from sglang_omni.pipeline.coordinator import Coordinator
from sglang_omni.pipeline.stage import AggregatedInput, DirectInput, InputHandler, Stage
from sglang_omni.pipeline.worker import Worker

# Re-export from submodules for convenience
# Re-export from submodules for convenience
from sglang_omni.proto import (
    AbortMessage,
    CompleteMessage,
    DataReadyMessage,
    OmniRequest,
    RequestState,
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
    "OmniRequest",
    "StageInfo",
    "DataReadyMessage",
    "AbortMessage",
    "CompleteMessage",
]
