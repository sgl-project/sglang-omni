# SPDX-License-Identifier: Apache-2.0
"""Exact bytes+dtype+shape round-trip for integer code tensors in pipeline state.

Packs as {key}_bytes/_shape/_dtype (narrowest of uint16/int32), decodes to int64.

Note(Chenchen Hong): use this when a code tensor must survive a control-plane
message, not just a relay hop. A field left in pipeline state reaches the
terminal CompleteMessage, which control_plane.send_complete msgpack-packs and
msgpack cannot pack a Tensor (it can pack bytes). Voxtral's audio_codes ends up
there, so it must be bytes. A keep-CPU-tensor field (serialize_value) only works
when it never crosses a control-plane message (relay side-channel hops only).
"""

from __future__ import annotations

from typing import Any


def encode_typed_tensor(value: Any, *, key: str) -> dict[str, Any]:
    """Pack an integer code tensor as {key}_bytes/_shape/_dtype (merge into payload.data)."""
    import numpy as np

    try:
        import torch
    except ImportError:
        torch = None

    if torch is not None and isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    array = np.asarray(value)
    if array.size == 0:
        array = array.astype(np.uint16, copy=False)
    elif int(array.min()) >= 0 and int(array.max()) <= np.iinfo(np.uint16).max:
        array = array.astype(np.uint16, copy=False)
    else:
        array = array.astype(np.int32, copy=False)
    contiguous = np.ascontiguousarray(array)
    return {
        f"{key}_bytes": contiguous.tobytes(),
        f"{key}_shape": list(contiguous.shape),
        f"{key}_dtype": str(contiguous.dtype),
    }


def decode_typed_tensor(
    data: dict[str, Any], *, key: str, legacy_key: str | None = None
) -> Any | None:
    """Inverse of encode_typed_tensor; legacy_key reads pre-encoding list/tensor payloads."""
    import numpy as np
    import torch

    if legacy_key is not None:
        legacy = data.get(legacy_key)
        if legacy is not None:
            if isinstance(legacy, list):
                return torch.tensor(legacy)
            return legacy

    raw = data.get(f"{key}_bytes")
    shape = data.get(f"{key}_shape")
    if raw is None or shape is None:
        return None
    dtype = np.dtype(data.get(f"{key}_dtype", "uint16"))
    array = np.frombuffer(raw, dtype=dtype).reshape(shape).astype(np.int64)
    return torch.from_numpy(array)
