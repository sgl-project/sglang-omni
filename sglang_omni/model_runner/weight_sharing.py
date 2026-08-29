# SPDX-License-Identifier: Apache-2.0
"""Share one copy of a stage's weights between two processes on one GPU.

A PD-disaggregated stage runs its two halves in separate processes. On one
device that means two copies of the same weights: measured at 57 GiB each for
the Qwen3-Omni thinker, which leaves a 140 GiB card room for about 21,500 KV
tokens per half against 677,613 on a colocated card.

The halves already map each other's GPU memory for the KV plane. Weights are
the easier case: static, read-only, allocated once, never reclaimed until
shutdown, so none of the reserve, commit and abort machinery applies. One half
exports handles to its parameter storage; the other points its own parameters
at that storage and releases what it loaded.

Peak memory is unchanged, because the adopting half still constructs and loads
before it swaps. The KV pools are sized after that, so they see the freed
space.
"""

from __future__ import annotations

import dataclasses
import logging
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)


class WeightLayoutMismatch(RuntimeError):
    """The two halves disagree about which parameters exist.

    Raised rather than skipped. A silently unshared parameter costs the memory
    without saying so, and a wrongly shared one is worse.
    """


@dataclasses.dataclass(frozen=True)
class WeightParameterHandle:
    """CUDA IPC handle plus every tensor property required for safe adoption."""

    handle: Any
    shape: tuple[int, ...]
    dtype: str
    device_type: str
    device_index: int | None
    stride: tuple[int, ...]
    layout: str
    storage_offset: int
    storage_nbytes: int

    @classmethod
    def from_tensor(cls, tensor: Any, *, handle: Any) -> "WeightParameterHandle":
        return cls(
            handle=handle,
            shape=tuple(tensor.shape),
            dtype=str(tensor.dtype),
            device_type=tensor.device.type,
            device_index=tensor.device.index,
            stride=tuple(tensor.stride()),
            layout=str(tensor.layout),
            storage_offset=int(tensor.storage_offset()),
            storage_nbytes=int(tensor.untyped_storage().nbytes()),
        )


def export_parameter_handles(model: Any) -> dict[str, WeightParameterHandle]:
    """Return one CUDA IPC handle per parameter, keyed by parameter name.

    Handles are small tokens, not copies, so exporting allocates nothing. The
    exporting process must outlive every process that adopts them.
    """
    from torch.multiprocessing.reductions import reduce_tensor

    handles: dict[str, WeightParameterHandle] = {}
    for name, param in model.named_parameters():
        if not param.is_cuda:
            continue
        handles[name] = WeightParameterHandle.from_tensor(
            param.data, handle=reduce_tensor(param.data)
        )
    logger.info("exported %d parameter handles for weight sharing", len(handles))
    return handles


def adopt_parameter_handles(
    model: Any,
    handles: dict[str, WeightParameterHandle],
    *,
    before_commit: Callable[[], None] | None = None,
) -> int:
    """Point this model's parameters at exported storage; return bytes released.

    Every exported name must exist here and every local name must have been
    exported. A mismatch means the two halves built different models, and
    continuing would either waste the memory silently or read the wrong bytes.

    Calls ``empty_cache`` at the end, which is required rather than tidy.
    Dropping the references returns the blocks to torch's caching allocator,
    where they stay invisible to every other process -- and the other process
    using them is the entire point. Measured on one H200 with 57.17 GiB across
    1757 tensors: after dropping the references the device still reported all
    of it held while ``memory_allocated`` reported zero, so a check on
    ``memory_allocated`` alone would have called that a success.
    """
    import torch

    named = dict(model.named_parameters())
    _check_parameters_match(named, handles)

    rebuilt: dict[str, Any] = {}
    released = 0
    for name, record in handles.items():
        param = named[name]
        rebuild, args = record.handle
        shared = rebuild(*args)
        _check_tensor_layout(name, shared, record, local=param.data)
        rebuilt[name] = shared
        released += param.data.numel() * param.data.element_size()

    if before_commit is not None:
        before_commit()

    originals = {name: named[name].data for name in rebuilt}
    committed: list[str] = []
    try:
        for name, shared in rebuilt.items():
            named[name].data = shared
            committed.append(name)
    except Exception:
        for name in reversed(committed):
            named[name].data = originals[name]
        raise
    del originals
    del rebuilt

    torch.cuda.empty_cache()
    logger.info(
        "adopted %d shared parameters, released %.2f GiB to the device",
        len(handles),
        released / 1024**3,
    )
    return released


def _check_parameters_match(
    named: dict[str, Any],
    handles: dict[str, WeightParameterHandle],
) -> None:
    """Fail before mutating anything if the two models disagree."""
    missing = sorted(set(handles) - set(named))
    if missing:
        raise WeightLayoutMismatch(
            f"{len(missing)} exported parameters are absent from this model, "
            f"starting with {missing[:3]}"
        )
    extra = sorted(set(named) - set(handles))
    if extra:
        raise WeightLayoutMismatch(
            f"{len(extra)} of this model's parameters were not exported, "
            f"starting with {extra[:3]}"
        )
    for name, record in handles.items():
        if not isinstance(record, WeightParameterHandle):
            raise WeightLayoutMismatch(
                f"parameter {name!r} has no typed weight-sharing manifest"
            )
        _check_tensor_layout(name, named[name].data, record)


def _check_tensor_layout(
    name: str,
    tensor: Any,
    record: WeightParameterHandle,
    *,
    local: Any | None = None,
) -> None:
    expected = {
        "shape": record.shape,
        "dtype": record.dtype,
        "device": (record.device_type, record.device_index),
        "stride": record.stride,
        "layout": record.layout,
        "storage_offset": record.storage_offset,
        "storage_nbytes": record.storage_nbytes,
    }
    actual = {
        "shape": tuple(tensor.shape),
        "dtype": str(tensor.dtype),
        "device": (tensor.device.type, tensor.device.index),
        "stride": tuple(tensor.stride()),
        "layout": str(tensor.layout),
        "storage_offset": int(tensor.storage_offset()),
        "storage_nbytes": int(tensor.untyped_storage().nbytes()),
    }
    for field, value in expected.items():
        if actual[field] != value:
            raise WeightLayoutMismatch(
                f"parameter {name!r} {field} mismatch: "
                f"local/shared={actual[field]!r}, published={value!r}"
            )
    if local is not None:
        local_device = (local.device.type, local.device.index)
        if actual["device"] != local_device:
            raise WeightLayoutMismatch(
                f"parameter {name!r} device mismatch: "
                f"local={local_device!r}, shared={actual['device']!r}"
            )


@dataclasses.dataclass(frozen=True)
class WeightSharingPlan:
    """What this half does about weights at startup, and with whom.

    Sharing applies only when the two halves are on one device, and that is
    settled by the published handles rather than by this plan: a CUDA IPC
    handle names memory on a particular GPU, the publisher records which, and
    :func:`apply_weight_sharing` declines handles from another one.
    """

    stage_name: str
    peer_stage: str
    rendezvous_dir: Path
    gpu_id: int
    publishes: bool = True
    adopted: Any | None = None


def apply_weight_sharing(model: Any, plan: WeightSharingPlan) -> int:
    """Publish this half's weights, or adopt the peer's. Returns bytes released.

    Which half publishes is decided from the declared shares, not from load
    order: the publisher keeps the copy it loaded, so its budget must hold the
    weights as well as its KV, and letting a race pick that half makes the same
    placement start one time and fail the next.

    The adopter's wait happens before ``gpu_startup_lock`` is taken, so by the
    time this runs the handles are already in hand.

    Call this after the weights are loaded and before the KV pool is sized.
    Peak memory is unchanged either way, because the adopting half still loads
    before it swaps, but the pool is sized after this returns and so sees the
    space the swap released.
    """
    from sglang_omni.model_runner.weight_rendezvous import (
        publish_parameter_handles,
        read_parameter_handles,
    )

    if plan.publishes:
        handles = export_parameter_handles(model)
        publish_parameter_handles(
            handles,
            rendezvous_dir=plan.rendezvous_dir,
            stage_name=plan.stage_name,
            gpu_id=plan.gpu_id,
            weight_bytes=_parameter_bytes(model),
        )
        return 0

    if plan.adopted is None:
        return 0

    def validate_publisher_generation() -> None:
        current = read_parameter_handles(
            rendezvous_dir=plan.rendezvous_dir,
            stage_name=plan.peer_stage,
            gpu_id=plan.gpu_id,
        )
        if current is None or current.generation != plan.adopted.generation:
            raise RuntimeError(
                f"weight publisher generation changed for {plan.peer_stage}; "
                "independent publisher restart is unsupported"
            )

    validate_publisher_generation()
    return adopt_parameter_handles(
        model,
        plan.adopted.parameters,
        before_commit=validate_publisher_generation,
    )


def _parameter_bytes(model: Any) -> int:
    """Total bytes of this model's CUDA parameters."""
    total = 0
    for _name, param in model.named_parameters():
        if getattr(param, "is_cuda", False):
            total += param.data.numel() * param.data.element_size()
    return total
