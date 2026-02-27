# SPDX-License-Identifier: Apache-2.0
"""OmniEngine - unified engine for all model types."""

from .engine import OmniEngine
from .factory import create_ar_engine, create_encoder_engine
from .model_runner import ModelRunner
from .runtime.ar import ARRequestData
from .runtime.encoder import EncoderRequestData
from .scheduler import Scheduler
from .types import (
    ModelRunnerOutput,
    RequestOutput,
    SchedulerOutput,
    SchedulerRequest,
    SchedulerStatus,
)

__all__ = [
    # Types
    "SchedulerRequest",
    "SchedulerStatus",
    "SchedulerOutput",
    "RequestOutput",
    "ModelRunnerOutput",
    # Core components
    "Scheduler",
    "ModelRunner",
    "OmniEngine",
    # Encoder
    "EncoderRequestData",
    "create_encoder_engine",
    # AR (Simple)
    "ARRequestData",
    "create_ar_engine",
]

# DualAR (FishAudio) — canonical location: sglang_omni.models.fishaudio_s1
# Re-exported here for backward compatibility.
from sglang_omni.models.fishaudio_s1 import (  # noqa: E402
    DualARRequestData,
    create_dual_ar_engine,
)
