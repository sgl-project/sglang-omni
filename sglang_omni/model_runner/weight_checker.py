# SPDX-License-Identifier: Apache-2.0
"""Strict SHA256 weight checker for online RL verification."""

from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class TensorDigest:
    name: str
    shape: tuple[int, ...]
    dtype: str
    sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "shape": list(self.shape),
            "dtype": self.dtype,
            "sha256": self.sha256,
        }


class StrictWeightChecker:
    """Compute strict per-tensor and aggregate SHA256 digests."""

    def __init__(self, model_runner: Any):
        self._model_runner = model_runner
        self._snapshot: dict[str, TensorDigest] | None = None

    def run(self, action: str) -> dict[str, Any]:
        if action == "snapshot":
            return self.snapshot()
        if action == "reset_tensors":
            return self.reset_tensors()
        if action == "compare":
            return self.compare()
        if action == "checksum":
            return self.checksum()
        raise ValueError(
            "Unsupported weights_checker action "
            f"{action!r}; expected snapshot, reset_tensors, compare, or checksum"
        )

    def snapshot(self) -> dict[str, Any]:
        self._snapshot = self._digest_model()
        return self._summary(self._snapshot, action="snapshot")

    def reset_tensors(self) -> dict[str, Any]:
        self._snapshot = self._digest_model()
        return self._summary(self._snapshot, action="reset_tensors")

    def checksum(self) -> dict[str, Any]:
        return self._summary(self._digest_model(), action="checksum")

    def compare(self) -> dict[str, Any]:
        if self._snapshot is None:
            raise RuntimeError("weights_checker compare requires snapshot first")
        current = self._digest_model()
        missing = sorted(set(self._snapshot) - set(current))
        unexpected = sorted(set(current) - set(self._snapshot))
        changed = [
            name
            for name in sorted(set(self._snapshot) & set(current))
            if self._snapshot[name].sha256 != current[name].sha256
            or self._snapshot[name].shape != current[name].shape
            or self._snapshot[name].dtype != current[name].dtype
        ]
        summary = self._summary(current, action="compare")
        summary.update(
            {
                "matched": not missing and not unexpected and not changed,
                "missing": missing,
                "unexpected": unexpected,
                "changed": changed,
            }
        )
        return summary

    def _digest_model(self) -> dict[str, TensorDigest]:
        model = getattr(self._model_runner, "model", None)
        if model is None:
            raise RuntimeError("model_runner has no model for weights_checker")

        logger.warning(
            "weights_checker: starting full-model SHA256 digest; "
            "inference is blocked until this completes. "
            "Elapsed time will be reported in the response."
        )
        t0 = time.time()
        digests: dict[str, TensorDigest] = {}
        for name, tensor in self._iter_named_tensors(model):
            digests[name] = _digest_tensor(name, tensor)
        logger.warning(
            "weights_checker: digest complete; %d tensors in %.1fs",
            len(digests),
            time.time() - t0,
        )
        return digests

    @staticmethod
    def _iter_named_tensors(model: Any):
        seen: set[int] = set()
        named_parameters = getattr(model, "named_parameters", None)
        if callable(named_parameters):
            for name, tensor in named_parameters():
                obj_id = id(tensor)
                if obj_id in seen:
                    continue
                seen.add(obj_id)
                yield name, tensor

        named_buffers = getattr(model, "named_buffers", None)
        if callable(named_buffers):
            for name, tensor in named_buffers():
                obj_id = id(tensor)
                if obj_id in seen:
                    continue
                seen.add(obj_id)
                yield name, tensor

    @staticmethod
    def _summary(
        digests: dict[str, TensorDigest],
        *,
        action: str,
    ) -> dict[str, Any]:
        started = time.time()
        tensor_sha = {name: digest.sha256 for name, digest in digests.items()}
        overall = _aggregate_checksum(tensor_sha)
        return {
            "action": action,
            "tensor_count": len(digests),
            "checksums": tensor_sha,
            "tensor_metadata": {
                name: digest.to_dict() for name, digest in digests.items()
            },
            "per_gpu_checksum": overall,
            "elapsed_s": time.time() - started,
        }


def _digest_tensor(name: str, tensor: Any) -> TensorDigest:
    detached = tensor.detach() if hasattr(tensor, "detach") else tensor
    contiguous = detached.contiguous() if hasattr(detached, "contiguous") else detached
    cpu = contiguous.cpu() if hasattr(contiguous, "cpu") else contiguous
    shape = tuple(int(x) for x in getattr(cpu, "shape", ()))
    dtype = str(getattr(cpu, "dtype", type(cpu).__name__))
    h = hashlib.sha256()
    h.update(name.encode())
    h.update(dtype.encode())
    h.update(str(shape).encode())
    h.update(_tensor_bytes(cpu))
    return TensorDigest(name=name, shape=shape, dtype=dtype, sha256=h.hexdigest())


def _tensor_bytes(tensor: Any) -> bytes:
    numpy = getattr(tensor, "numpy", None)
    if callable(numpy):
        try:
            return numpy().tobytes()
        except (TypeError, RuntimeError):
            pass

    view = getattr(tensor, "view", None)
    if callable(view):
        try:
            import torch

            byte_view = tensor.view(torch.uint8)
            return byte_view.numpy().tobytes()
        except Exception:
            pass

    tobytes = getattr(tensor, "tobytes", None)
    if callable(tobytes):
        return tobytes()
    raise TypeError(
        f"Cannot extract raw bytes from tensor type {type(tensor).__name__}"
    )


def _aggregate_checksum(checksums: dict[str, str]) -> str:
    h = hashlib.sha256()
    for name in sorted(checksums):
        h.update(name.encode())
        h.update(checksums[name].encode())
    return h.hexdigest()
