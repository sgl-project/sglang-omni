# SPDX-License-Identifier: Apache-2.0
"""Leader-side export of shared CUDA weights."""

from __future__ import annotations

import hashlib
import os
import time

import torch

from sglang_omni.distributed.weight_ipc.cuda_handles import (
    allocation_offset_bytes,
    dtype_to_name,
    export_storage,
)
from sglang_omni.distributed.weight_ipc.select import ArParametersPolicy, SharePolicy
from sglang_omni.distributed.weight_ipc.types import (
    SCHEMA_VERSION,
    IpcTensorMeta,
    WeightIpcBundle,
)


def compute_name_digest(tensors: list[IpcTensorMeta]) -> str:
    h = hashlib.sha256()
    for meta in sorted(tensors, key=lambda m: m.name):
        h.update(meta.name.encode("utf-8"))
        h.update(b"\0")
        h.update(repr(meta.shape).encode("utf-8"))
        h.update(b"\0")
        h.update(repr(meta.stride).encode("utf-8"))
        h.update(b"\0")
        h.update(meta.dtype.encode("utf-8"))
        h.update(b"\0")
    return h.hexdigest()


def export_shared_weights(
    model: torch.nn.Module,
    *,
    model_path: str,
    model_revision: str | None = None,
    policy: SharePolicy | None = None,
    cuda_driver: str | None = None,
) -> WeightIpcBundle:
    """Export selected CUDA tensors from ``model`` into a durable IPC bundle."""
    share_policy = policy or ArParametersPolicy()
    selected = share_policy.select(model)
    if not selected:
        raise ValueError("SharePolicy selected no CUDA tensors to export")

    metas: list[IpcTensorMeta] = []
    for name, tensor in selected:
        shared = export_storage(tensor)
        nbytes = tensor.numel() * tensor.element_size()
        metas.append(
            IpcTensorMeta(
                name=name,
                shape=tuple(int(x) for x in tensor.shape),
                stride=tuple(int(x) for x in tensor.stride()),
                dtype=dtype_to_name(tensor.dtype),
                nbytes=int(nbytes),
                device_index=int(tensor.device.index or 0),
                handle=shared.handle,
                storage_size_bytes=shared.storage_size_bytes,
                storage_offset_bytes=shared.storage_offset_bytes,
                allocation_offset_bytes=allocation_offset_bytes(tensor),
                tensor_offset=int(tensor.storage_offset()),
                requires_grad=bool(getattr(tensor, "requires_grad", False)),
                ref_counter_handle=shared.ref_counter_handle,
                ref_counter_offset=shared.ref_counter_offset,
                event_handle=shared.event_handle,
                event_sync_required=shared.event_sync_required,
            )
        )

    return WeightIpcBundle(
        schema_version=SCHEMA_VERSION,
        model_path=model_path,
        model_revision=model_revision,
        cuda_driver=cuda_driver,
        created_unix_s=time.time(),
        leader_pid=os.getpid(),
        tensors=metas,
        name_digest=compute_name_digest(metas),
    )
