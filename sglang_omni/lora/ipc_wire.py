# SPDX-License-Identifier: Apache-2.0
"""Allowlisted JSON/msgspec-safe wire codec for tensor and CUDA IPC handles."""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from typing import Any

import pybase64
import torch
from torch.multiprocessing import reductions

logger = logging.getLogger(__name__)

WIRE_CODEC_VERSION = 1
ENCODING_CUDA_IPC = "cuda_ipc_v1"
ENCODING_INLINE_B64 = "inline_b64_v1"
ALLOWED_REBUILD_MODULE = "torch.multiprocessing.reductions"
ALLOWED_REBUILD_NAME = "rebuild_cuda_tensor"

ALLOWED_DTYPES: dict[str, torch.dtype] = {
    "torch.float16": torch.float16,
    "torch.bfloat16": torch.bfloat16,
    "torch.float32": torch.float32,
    "torch.float64": torch.float64,
    "torch.int8": torch.int8,
    "torch.uint8": torch.uint8,
    "torch.int16": torch.int16,
    "torch.int32": torch.int32,
    "torch.int64": torch.int64,
    "torch.bool": torch.bool,
}
DTYPE_TO_NAME: dict[torch.dtype, str] = {v: k for k, v in ALLOWED_DTYPES.items()}

ALLOWED_STORAGE_TYPES: dict[str, type] = {
    "torch.storage.UntypedStorage": torch.storage.UntypedStorage,
    "torch.UntypedStorage": torch.UntypedStorage,
}


def encode_bytes_b64(raw: bytes | None) -> str | None:
    if raw is None:
        return None
    else:
        pass
    if not isinstance(raw, (bytes, bytearray)):
        raise ValueError(f"expected bytes for IPC handle, got {type(raw).__name__}")
    else:
        pass
    return pybase64.b64encode(bytes(raw)).decode("ascii")


def decode_bytes_b64(
    encoded: object, field_name: str, *, allow_none: bool
) -> bytes | None:
    if encoded is None:
        if allow_none:
            return None
        else:
            raise ValueError(f"missing required handle field {field_name!r}")
    else:
        pass
    if not isinstance(encoded, str):
        raise ValueError(
            f"handle field {field_name!r} must be base64 str, "
            f"got {type(encoded).__name__}"
        )
    else:
        pass
    try:
        return pybase64.b64decode(encoded, validate=True)
    except Exception as exc:
        raise ValueError(f"invalid base64 in handle field {field_name!r}") from exc


def validate_int_sequence(value: object, field_name: str) -> tuple[int, ...]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(
            f"field {field_name!r} must be list/tuple of ints, "
            f"got {type(value).__name__}"
        )
    else:
        pass
    out: list[int] = []
    for elem in value:
        if isinstance(elem, bool) or not isinstance(elem, int) or elem < 0:
            raise ValueError(
                f"field {field_name!r} must contain non-negative ints, got {elem!r}"
            )
        else:
            pass
        out.append(elem)
    return tuple(out)


def validate_non_negative_int(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(
            f"field {field_name!r} must be a non-negative int, got {value!r}"
        )
    else:
        pass
    return value


def encode_cuda_ipc_handle(reduced: tuple[Any, ...]) -> dict[str, Any]:
    """Encode a torch.multiprocessing.reductions.reduce_tensor tuple without pickle."""
    if not isinstance(reduced, tuple) or len(reduced) != 2:
        raise ValueError("expected a 2-tuple (rebuild_fn, args) from reduce_tensor")
    else:
        pass
    rebuild_fn, args = reduced
    fn_module = getattr(rebuild_fn, "__module__", "")
    fn_name = getattr(rebuild_fn, "__name__", "")
    if fn_module != ALLOWED_REBUILD_MODULE or fn_name != ALLOWED_REBUILD_NAME:
        raise ValueError(
            f"disallowed tensor rebuild callable {fn_module}.{fn_name}; "
            f"only {ALLOWED_REBUILD_MODULE}.{ALLOWED_REBUILD_NAME} is permitted"
        )
    else:
        pass
    if not isinstance(args, tuple) or len(args) not in (14, 15):
        got = len(args) if isinstance(args, tuple) else type(args).__name__
        raise ValueError(f"rebuild_cuda_tensor expects 14 or 15 arguments, got {got}")
    else:
        pass

    (
        tensor_cls,
        tensor_size,
        tensor_stride,
        tensor_offset,
        storage_cls,
        dtype,
        storage_device,
        storage_handle,
        storage_size_bytes,
        storage_offset_bytes,
        requires_grad,
        ref_counter_handle,
        ref_counter_offset,
        event_handle,
    ) = args[:14]
    event_sync_required = bool(args[14]) if len(args) == 15 else False
    if tensor_cls is not torch.Tensor:
        raise ValueError(f"disallowed tensor class {tensor_cls!r}")
    else:
        pass
    storage_key = f"{storage_cls.__module__}.{storage_cls.__qualname__}"
    if storage_key not in ALLOWED_STORAGE_TYPES:
        raise ValueError(f"disallowed storage class {storage_key!r}")
    else:
        pass
    dtype_name = DTYPE_TO_NAME.get(dtype)
    if dtype_name is None:
        raise ValueError(f"unsupported tensor dtype {dtype!r}")
    else:
        pass

    return {
        "version": WIRE_CODEC_VERSION,
        "encoding": ENCODING_CUDA_IPC,
        "shape": list(validate_int_sequence(tuple(tensor_size), "shape")),
        "stride": list(validate_int_sequence(tuple(tensor_stride), "stride")),
        "tensor_offset": validate_non_negative_int(tensor_offset, "tensor_offset"),
        "storage_cls": storage_key,
        "dtype": dtype_name,
        "device": validate_non_negative_int(storage_device, "device"),
        "storage_handle_b64": encode_bytes_b64(storage_handle),
        "storage_size_bytes": validate_non_negative_int(
            storage_size_bytes, "storage_size_bytes"
        ),
        "storage_offset_bytes": validate_non_negative_int(
            storage_offset_bytes, "storage_offset_bytes"
        ),
        "requires_grad": bool(requires_grad),
        "ref_counter_handle_b64": encode_bytes_b64(ref_counter_handle),
        "ref_counter_offset": validate_non_negative_int(
            ref_counter_offset, "ref_counter_offset"
        ),
        "event_handle_b64": encode_bytes_b64(event_handle),
        "event_sync_required": bool(event_sync_required),
    }


def encode_cuda_ipc_handle_from_args(args: tuple[Any, ...]) -> dict[str, Any]:
    """Encode a 15-element rebuild_cuda_tensor argument tuple (with event_sync_required)."""
    if not isinstance(args, tuple) or len(args) != 15:
        raise ValueError(
            f"expected 15 rebuild_cuda_tensor args, got {len(args) if isinstance(args, tuple) else type(args).__name__}"
        )
    else:
        pass
    encoded = encode_cuda_ipc_handle((reductions.rebuild_cuda_tensor, args[:14]))
    encoded["event_sync_required"] = bool(args[14])
    return encoded


def encode_inline_tensor(tensor: torch.Tensor) -> dict[str, Any]:
    """Encode a CPU or small tensor into a JSON/msgspec-safe inline payload."""
    dtype_name = DTYPE_TO_NAME.get(tensor.dtype)
    if dtype_name is None:
        raise ValueError(f"unsupported tensor dtype {tensor.dtype!r}")
    else:
        pass
    cpu_contig = tensor.detach().to(device="cpu").contiguous()
    raw_bytes = cpu_contig.view(torch.uint8).numpy().tobytes()
    return {
        "version": WIRE_CODEC_VERSION,
        "encoding": ENCODING_INLINE_B64,
        "shape": list(cpu_contig.shape),
        "dtype": dtype_name,
        "data_b64": pybase64.b64encode(raw_bytes).decode("ascii"),
    }


def decode_wire_tensor(
    payload: Mapping[str, Any],
    *,
    rebuild_cuda_fn: Callable[..., torch.Tensor] | None = None,
) -> torch.Tensor:
    """Decode an allowlisted wire payload back into a torch.Tensor without pickle."""
    if not isinstance(payload, Mapping):
        raise ValueError(
            f"wire tensor payload must be a mapping, got {type(payload).__name__}"
        )
    else:
        pass
    version = payload.get("version")
    if version != WIRE_CODEC_VERSION:
        raise ValueError(
            f"unsupported wire codec version {version!r}, expected {WIRE_CODEC_VERSION}"
        )
    else:
        pass
    encoding = payload.get("encoding")
    dtype_name = payload.get("dtype")
    if not isinstance(dtype_name, str) or dtype_name not in ALLOWED_DTYPES:
        raise ValueError(f"disallowed or missing dtype {dtype_name!r}")
    else:
        pass
    dtype = ALLOWED_DTYPES[dtype_name]
    shape = validate_int_sequence(payload.get("shape"), "shape")

    if encoding == ENCODING_INLINE_B64:
        raw = decode_bytes_b64(payload.get("data_b64"), "data_b64", allow_none=False)
        assert raw is not None
        expected_numel = 1
        for dim in shape:
            expected_numel *= dim
        elem_size = torch.empty((), dtype=dtype).element_size()
        expected_bytes = expected_numel * elem_size
        if len(raw) != expected_bytes:
            raise ValueError(
                f"inline tensor byte length mismatch: expected {expected_bytes}, "
                f"got {len(raw)}"
            )
        else:
            pass
        if expected_bytes == 0:
            return torch.empty(shape, dtype=dtype)
        else:
            pass
        buf = torch.frombuffer(bytearray(raw), dtype=torch.uint8)
        return buf.view(dtype).reshape(shape).clone()
    elif encoding == ENCODING_CUDA_IPC:
        storage_cls_name = payload.get("storage_cls")
        if (
            not isinstance(storage_cls_name, str)
            or storage_cls_name not in ALLOWED_STORAGE_TYPES
        ):
            raise ValueError(f"disallowed storage_cls {storage_cls_name!r}")
        else:
            pass
        storage_cls = ALLOWED_STORAGE_TYPES[storage_cls_name]
        stride = validate_int_sequence(payload.get("stride"), "stride")
        if len(stride) != len(shape):
            raise ValueError(
                f"stride rank {len(stride)} does not match shape rank {len(shape)}"
            )
        else:
            pass
        tensor_offset = validate_non_negative_int(
            payload.get("tensor_offset"), "tensor_offset"
        )
        device = validate_non_negative_int(payload.get("device"), "device")
        storage_handle = decode_bytes_b64(
            payload.get("storage_handle_b64"), "storage_handle_b64", allow_none=False
        )
        storage_size_bytes = validate_non_negative_int(
            payload.get("storage_size_bytes"), "storage_size_bytes"
        )
        storage_offset_bytes = validate_non_negative_int(
            payload.get("storage_offset_bytes"), "storage_offset_bytes"
        )
        requires_grad = bool(payload.get("requires_grad", False))
        ref_counter_handle = decode_bytes_b64(
            payload.get("ref_counter_handle_b64"),
            "ref_counter_handle_b64",
            allow_none=False,
        )
        ref_counter_offset = validate_non_negative_int(
            payload.get("ref_counter_offset"), "ref_counter_offset"
        )
        event_handle = decode_bytes_b64(
            payload.get("event_handle_b64"), "event_handle_b64", allow_none=True
        )
        event_sync_required = bool(payload.get("event_sync_required", False))
        rebuilder = (
            rebuild_cuda_fn
            if rebuild_cuda_fn is not None
            else reductions.rebuild_cuda_tensor
        )
        return rebuilder(
            torch.Tensor,
            shape,
            stride,
            tensor_offset,
            storage_cls,
            dtype,
            device,
            storage_handle,
            storage_size_bytes,
            storage_offset_bytes,
            requires_grad,
            ref_counter_handle,
            ref_counter_offset,
            event_handle,
            event_sync_required,
        )
    else:
        raise ValueError(f"unsupported wire encoding {encoding!r}")


def release_unpublished_ipc_handle(
    payload: Mapping[str, Any],
    *,
    release_fn: Callable[[bytes, int], object] | None = None,
) -> bool:
    """Release ref-counter ownership for a CUDA IPC handle that was never published."""
    if payload.get("encoding") != ENCODING_CUDA_IPC:
        return False
    else:
        pass
    ref_handle = decode_bytes_b64(
        payload.get("ref_counter_handle_b64"),
        "ref_counter_handle_b64",
        allow_none=True,
    )
    if ref_handle is None:
        return False
    else:
        pass
    ref_offset = validate_non_negative_int(
        payload.get("ref_counter_offset", 0), "ref_counter_offset"
    )
    releaser = (
        release_fn
        if release_fn is not None
        else getattr(
            torch.storage.UntypedStorage,
            "_release_ipc_counter",  # noqa: leading-underscore
            None,
        )
    )
    if releaser is None:
        return False
    else:
        pass
    try:
        releaser(ref_handle, ref_offset)
        return True
    except Exception:
        logger.warning(
            "Failed to release unpublished CUDA IPC handle counter",
            exc_info=True,
        )
        return False
