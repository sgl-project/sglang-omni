# SPDX-License-Identifier: Apache-2.0
"""Pickle-safe ModelWorkerBatch copies for TP follower broadcast.

First call pickle-verifies to fail fast if a new non-picklable field
is added upstream.
"""
from __future__ import annotations

import copy
import logging
import pickle
from enum import Enum
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import ModelWorkerBatch

logger = logging.getLogger(__name__)

_FIELDS_TO_STRIP = ("sampling_info", "reqs")

_pickle_verified = False


def make_follower_batch(model_worker_batch: "ModelWorkerBatch") -> "ModelWorkerBatch":
    """Shallow copy with non-picklable fields nulled and all tensors moved to CPU.

    - sampling_info: penalizer weakrefs + Event + custom processors
    - reqs:          request objects with callbacks/threads
    - tensors:       recursively copied to CPU so followers unpickle onto
                     host memory instead of rank 0's CUDA device
    """
    global _pickle_verified

    follower = copy.copy(model_worker_batch)
    for field in _FIELDS_TO_STRIP:
        setattr(follower, field, None)

    memo: dict[int, Any] = {id(model_worker_batch): follower}
    for attr, val in vars(follower).items():
        setattr(follower, attr, _to_cpu_copy(val, memo))

    if not _pickle_verified:
        _verify_pickle_safe(follower)
        _pickle_verified = True

    return follower


def _to_cpu_copy(obj: Any, memo: dict[int, Any]) -> Any:
    """Non-mutating: containers and objects holding tensors are rebuilt so the
    caller's original (rank 0's live batch) is never touched. Cycles are
    handled via *memo* keyed on id().
    """
    if isinstance(obj, torch.Tensor):
        return obj.cpu() if obj.device.type != "cpu" else obj
    if isinstance(obj, Enum) or isinstance(obj, type):
        return obj
    if obj is None or isinstance(obj, (int, float, bool, str, bytes)):
        return obj

    obj_id = id(obj)
    if obj_id in memo:
        return memo[obj_id]

    if isinstance(obj, list):
        new = [None] * len(obj)
        memo[obj_id] = new
        for i, v in enumerate(obj):
            new[i] = _to_cpu_copy(v, memo)
        return new
    if isinstance(obj, tuple):
        return tuple(_to_cpu_copy(v, memo) for v in obj)
    if isinstance(obj, dict):
        new = {}
        memo[obj_id] = new
        for k, v in obj.items():
            new[k] = _to_cpu_copy(v, memo)
        return new
    if hasattr(obj, "__dict__"):
        cloned = copy.copy(obj)
        memo[obj_id] = cloned
        for attr, val in vars(obj).items():
            setattr(cloned, attr, _to_cpu_copy(val, memo))
        return cloned
    return obj


def _verify_pickle_safe(batch: "ModelWorkerBatch") -> None:
    """Try to pickle *batch*; raise RuntimeError with field-level diagnosis on failure."""
    try:
        pickle.dumps(batch)
    except Exception as exc:
        bad_fields = []
        for attr, val in vars(batch).items():
            if val is None:
                continue
            try:
                pickle.dumps(val)
            except Exception:
                bad_fields.append(attr)
        raise RuntimeError(
            f"TP follower batch is not pickle-safe after stripping "
            f"{_FIELDS_TO_STRIP}. Unpicklable fields: {bad_fields}. "
            f"Add them to _FIELDS_TO_STRIP in "
            f"sglang_omni/engines/tp/serialization.py. "
            f"Original error: {exc}"
        ) from exc
    logger.debug("Follower batch pickle verification passed")


def attach_page_table_snapshot(batch: "ModelWorkerBatch", req_to_token_pool) -> None:
    """Copy relevant req_to_token rows onto *batch* so followers can sync their page table.

    Must be called on rank 0 AFTER the scheduler writes to req_to_token_pool
    and BEFORE make_follower_batch().
    """

    pool_tensor = req_to_token_pool.req_to_token  # (max_reqs, max_ctx_len) int32
    rows = []
    for i in range(len(batch.seq_lens)):
        idx = int(batch.req_pool_indices[i])
        seq_len = int(batch.seq_lens[i])
        rows.append(pool_tensor[idx, :seq_len].clone())
    batch.tp_page_table_rows = rows
