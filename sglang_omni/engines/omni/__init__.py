# SPDX-License-Identifier: Apache-2.0
"""OmniEngine - unified engine for all model types."""

from .engine import OmniEngine
from .factory import (
    create_ar_engine,
    create_encoder_engine,
    create_sglang_ar_engine,
    create_single_pass_engine,
)
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
    "create_single_pass_engine",
    # AR (Simple)
    "ARRequestData",
    "create_ar_engine",
    # AR (SGLang)
    "SGLangARRequestData",
    "create_sglang_ar_engine",
]


def __getattr__(name: str):
    if name == "SGLangARRequestData":
        from .runtime.sglang_ar import SGLangARRequestData

        return SGLangARRequestData
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
