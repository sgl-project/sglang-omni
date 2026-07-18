# SPDX-License-Identifier: Apache-2.0
"""Lazy tensor handoff for large multimodal tensors crossing pipeline stages.

A ``TensorRef`` stands in for a tensor that has been externalized to the
relay rather than inlined into a ``StagePayload``. Intermediate stages that
only forward the value (e.g. ``mm_aggregate``) never materialize it; only
the declared ``consumer_stage`` resolves it back into a real tensor.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

TENSOR_REF_MARKER = "__sglang_omni_tensor_ref__"
DEFAULT_TENSOR_REF_THRESHOLD_MB = 2.0
DEFAULT_TENSOR_REF_PATHS = (
    "video_embeds",
    "deepstack_visual_embeds_image",
    "deepstack_visual_embeds_video",
)


@dataclass(frozen=True)
class TensorRef:
    ref_id: str
    request_id: str
    producer_stage: str
    consumer_stage: str
    path: str
    shape: tuple[int, ...]
    dtype: str
    nbytes: int
    blob_key: str
    blob_metadata: dict[str, Any]
    backend: str = "relay_lazy"

    def to_dict(self) -> dict[str, Any]:
        return {
            TENSOR_REF_MARKER: True,
            "ref_id": self.ref_id,
            "request_id": self.request_id,
            "producer_stage": self.producer_stage,
            "consumer_stage": self.consumer_stage,
            "path": self.path,
            "shape": list(self.shape),
            "dtype": self.dtype,
            "nbytes": self.nbytes,
            "blob_key": self.blob_key,
            "blob_metadata": self.blob_metadata,
            "backend": self.backend,
        }

    @classmethod
    def from_dict(cls, obj: dict[str, Any]) -> "TensorRef":
        return cls(
            ref_id=str(obj["ref_id"]),
            request_id=str(obj["request_id"]),
            producer_stage=str(obj["producer_stage"]),
            consumer_stage=str(obj["consumer_stage"]),
            path=str(obj["path"]),
            shape=tuple(int(x) for x in obj["shape"]),
            dtype=str(obj["dtype"]),
            nbytes=int(obj["nbytes"]),
            blob_key=str(obj["blob_key"]),
            blob_metadata=dict(obj["blob_metadata"]),
            backend=str(obj.get("backend", "relay_lazy")),
        )


def is_tensor_ref_dict(obj: Any) -> bool:
    return isinstance(obj, dict) and obj.get(TENSOR_REF_MARKER) is True


def tensor_ref_numel(obj: dict[str, Any] | TensorRef) -> int:
    ref = TensorRef.from_dict(obj) if isinstance(obj, dict) else obj
    numel = 1
    for dim in ref.shape:
        numel *= int(dim)
    return numel


@dataclass(frozen=True)
class TensorRefPolicy:
    """Per-edge policy controlling which tensor leaves get externalized."""

    threshold_bytes: int
    from_stage: str
    to_stage: str
    consumer_stage: str
    path_allowlist: tuple[str, ...]

    def should_externalize(self, path: str, tensor: torch.Tensor) -> bool:
        """Whether to externalize this tensor leaf as a TensorRef.

        The allowlist matches the *leaf name* with any list index stripped
        (``foo[0]`` -> ``foo``), so an allowlisted ``list[Tensor]`` (e.g.
        deepstack embeds) is externalized element-by-element -- each list
        element becomes its own blob / relay segment.
        """
        leaf_name = path.rsplit(".", 1)[-1].split("[")[0]
        if leaf_name not in self.path_allowlist:
            return False
        nbytes = tensor.numel() * tensor.element_size()
        return nbytes >= self.threshold_bytes
