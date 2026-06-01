"""Vendor wrapper for sglang.srt.model_executor.forward_batch_info.

Use static imports to preserve IDE navigation, and apply optional patches below.
"""

from __future__ import annotations

from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.environ import envs
from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
from sglang.srt.managers.schedule_policy import PrefillAdder
from sglang.srt.managers.scheduler import GenerationBatchResult
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.server_args import (
    ATTENTION_BACKEND_CHOICES,
    PortArgs,
    ServerArgs,
    get_global_server_args,
)

__all__ = [
    "Req",
    "envs",
    "ScheduleBatch",
    "PrefillAdder",
    "ForwardBatch",
    "ModelRunner",
    "ModelConfig",
    "get_global_server_args",
    "ServerArgs",
    "PortArgs",
    "ATTENTION_BACKEND_CHOICES",
    "GenerationBatchResult",
]
