# SPDX-License-Identifier: Apache-2.0
"""OmniEngine - unified engine for all model types."""

from .engine import OmniEngine
from .factory import create_ar_engine, create_dual_ar_engine, create_encoder_engine
from .model_runner import ModelRunner
from .runtime.ar import ARRequestData
from .runtime.dual_ar import DualARRequestData
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
    # DualAR (FishAudio)
    "DualARRequestData",
    "create_dual_ar_engine",
]
