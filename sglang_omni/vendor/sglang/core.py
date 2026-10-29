"""Vendor wrapper for sglang.srt.model_executor.forward_batch_info.

Use static imports to preserve IDE navigation, and apply optional patches below.
"""

from __future__ import annotations

from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.environ import envs
from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
from sglang.srt.managers.schedule_batch import (
    set_mamba_track_indices_from_reqs as sglang_set_mamba_track_indices,
)
from sglang.srt.managers.schedule_policy import PrefillAdder
from sglang.srt.managers.scheduler import GenerationBatchResult
from sglang.srt.mem_cache.memory_pool import HybridReqToTokenPool
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.server_args import ATTENTION_BACKEND_CHOICES, PortArgs, ServerArgs


def set_mamba_track_indices(
    batch: ScheduleBatch, track_positions: list[int] | None = None
) -> None:
    """Select the sole checkpoint buffer without uploading zero column indices."""
    request_pool = batch.req_to_token_pool
    assert isinstance(request_pool, HybridReqToTokenPool)
    if (
        request_pool.mamba_ping_pong_track_buffer_size != 1
        or track_positions is not None
    ):
        sglang_set_mamba_track_indices(batch, track_positions)
    else:
        batch.mamba_track_buffer_indices = [0] * len(batch.reqs)
        # note (wirybeaver): keep donation updates, but exclude appended beam rows.
        batch.mamba_track_indices = (
            request_pool.req_index_to_mamba_ping_pong_track_buffer_mapping[
                batch.req_pool_indices[: len(batch.reqs)], 0
            ]
        )


__all__ = [
    "Req",
    "envs",
    "ScheduleBatch",
    "PrefillAdder",
    "ForwardBatch",
    "ModelRunner",
    "ModelConfig",
    "ServerArgs",
    "PortArgs",
    "ATTENTION_BACKEND_CHOICES",
    "GenerationBatchResult",
    "set_mamba_track_indices",
]
