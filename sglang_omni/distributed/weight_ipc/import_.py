# SPDX-License-Identifier: Apache-2.0
"""Follower-side import and parameter aliasing."""

from __future__ import annotations

import torch

from sglang_omni.distributed.weight_ipc.cuda_handles import (
    SharedCudaStorage,
    dtype_from_name,
    open_storage,
    rebuild_tensor,
)
from sglang_omni.distributed.weight_ipc.export import compute_name_digest
from sglang_omni.distributed.weight_ipc.select import ArParametersPolicy, SharePolicy
from sglang_omni.distributed.weight_ipc.types import IpcTensorMeta, WeightIpcBundle


class WeightIpcImportError(ValueError):
    """Raised when a follower refuses a weight IPC bundle."""


def validate_bundle(
    bundle: WeightIpcBundle,
    *,
    model_path: str,
    model_revision: str | None,
    expected_names: set[str] | None = None,
) -> None:
    if bundle.model_path != model_path:
        raise WeightIpcImportError(
            f"model_path mismatch: bundle={bundle.model_path!r} local={model_path!r}"
        )
    if bundle.model_revision != model_revision:
        raise WeightIpcImportError(
            "model_revision mismatch: "
            f"bundle={bundle.model_revision!r} local={model_revision!r}"
        )
    digest = compute_name_digest(bundle.tensors)
    if digest != bundle.name_digest:
        raise WeightIpcImportError("name_digest mismatch against bundle contents")
    names = [m.name for m in bundle.tensors]
    if len(names) != len(set(names)):
        raise WeightIpcImportError("bundle contains duplicate tensor names")
    if expected_names is not None and set(names) != expected_names:
        missing = sorted(expected_names - set(names))
        unexpected = sorted(set(names) - expected_names)
        raise WeightIpcImportError(
            f"shared name set mismatch: missing={missing} unexpected={unexpected}"
        )


def _validate_meta(local: torch.Tensor, meta: IpcTensorMeta) -> None:
    if tuple(local.shape) != meta.shape:
        raise WeightIpcImportError(
            f"{meta.name}: shape mismatch local={tuple(local.shape)} remote={meta.shape}"
        )
    if tuple(local.stride()) != meta.stride:
        raise WeightIpcImportError(
            f"{meta.name}: stride mismatch local={tuple(local.stride())} "
            f"remote={meta.stride}"
        )
    if dtype_from_name(meta.dtype) != local.dtype:
        raise WeightIpcImportError(
            f"{meta.name}: dtype mismatch local={local.dtype} remote={meta.dtype}"
        )


def open_tensor(meta: IpcTensorMeta) -> torch.Tensor:
    storage = open_storage(
        SharedCudaStorage(
            device_index=meta.device_index,
            handle=meta.handle,
            storage_size_bytes=meta.storage_size_bytes,
            storage_offset_bytes=meta.storage_offset_bytes,
            ref_counter_handle=meta.ref_counter_handle,
            ref_counter_offset=meta.ref_counter_offset,
            event_handle=meta.event_handle,
            event_sync_required=meta.event_sync_required,
        )
    )
    return rebuild_tensor(
        storage=storage,
        dtype=dtype_from_name(meta.dtype),
        shape=meta.shape,
        stride=meta.stride,
        tensor_offset=meta.tensor_offset,
        device_index=meta.device_index,
    )


def import_and_alias(
    model: torch.nn.Module,
    bundle: WeightIpcBundle,
    *,
    model_path: str,
    model_revision: str | None = None,
    policy: SharePolicy | None = None,
) -> dict[str, torch.Tensor]:
    """Open shared tensors and alias them onto ``model``."""
    share_policy = policy or ArParametersPolicy()
    local_items = dict(share_policy.select(model))
    validate_bundle(
        bundle,
        model_path=model_path,
        model_revision=model_revision,
        expected_names=set(local_items),
    )

    imported: dict[str, torch.Tensor] = {}
    for meta in bundle.tensors:
        local = local_items[meta.name]
        _validate_meta(local, meta)
        remote = open_tensor(meta)
        if isinstance(local, torch.nn.Parameter):
            local.data = remote
            local.requires_grad_(False)
        else:
            local.set_(remote)
        imported[meta.name] = remote

    # Note (guozhihao): keep strong refs so GC cannot reclaim imported storages.
    model._weight_ipc_imported_tensors = imported  # type: ignore[attr-defined]
    return imported
