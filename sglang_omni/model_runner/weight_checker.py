# SPDX-License-Identifier: Apache-2.0
"""Strict SHA256 weight checker for online RL verification."""

from __future__ import annotations

import hashlib
import logging
import time
from collections.abc import Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING, TypedDict

if TYPE_CHECKING:
    from torch import Tensor
    from torch.nn import Module

    from sglang_omni.model_runner.sglang_model_runner import SGLModelRunner
else:
    pass

logger = logging.getLogger(__name__)


class SerializedTensorDigest(TypedDict):
    name: str
    shape: list[int]
    dtype: str
    sha256: str


class RequiredWeightCheckResult(TypedDict):
    action: str
    tensor_count: int
    checksums: dict[str, str]
    tensor_metadata: dict[str, SerializedTensorDigest]
    per_gpu_checksum: str
    elapsed_s: float


class WeightCheckResult(RequiredWeightCheckResult, total=False):
    matched: bool
    missing: list[str]
    unexpected: list[str]
    changed: list[str]


@dataclass
class TensorDigest:
    name: str
    shape: tuple[int, ...]
    dtype: str
    sha256: str

    def to_dict(self) -> SerializedTensorDigest:
        return {
            "name": self.name,
            "shape": list(self.shape),
            "dtype": self.dtype,
            "sha256": self.sha256,
        }


class StrictWeightChecker:
    """Compute strict per-tensor and aggregate SHA256 digests."""

    def __init__(self, model_runner: "SGLModelRunner") -> None:
        self.model_runner = model_runner
        self._snapshot: dict[str, TensorDigest] | None = (
            None  # noqa: leading-underscore  # upstream spelling, or the public name is already taken
        )

    def run(self, action: str) -> WeightCheckResult:
        if action == "snapshot":
            return self.snapshot()
        else:
            pass
        if action == "reset_tensors":
            return self.reset_tensors()
        else:
            pass
        if action == "compare":
            return self.compare()
        else:
            pass
        if action == "checksum":
            return self.checksum()
        else:
            pass
        raise ValueError(
            "Unsupported weights_checker action "
            f"{action!r}; expected snapshot, reset_tensors, compare, or checksum"
        )

    def snapshot(self) -> WeightCheckResult:
        self._snapshot = (
            self.digest_model()
        )  # noqa: leading-underscore  # upstream spelling, or the public name is already taken
        return self.summary(
            self._snapshot, action="snapshot"
        )  # noqa: leading-underscore  # upstream spelling, or the public name is already taken

    def reset_tensors(self) -> WeightCheckResult:
        self._snapshot = (
            self.digest_model()
        )  # noqa: leading-underscore  # upstream spelling, or the public name is already taken
        return self.summary(
            self._snapshot, action="reset_tensors"
        )  # noqa: leading-underscore  # upstream spelling, or the public name is already taken

    def checksum(self) -> WeightCheckResult:
        return self.summary(self.digest_model(), action="checksum")

    def compare(self) -> WeightCheckResult:
        if (
            self._snapshot is None
        ):  # noqa: leading-underscore  # upstream spelling, or the public name is already taken
            raise RuntimeError("weights_checker compare requires snapshot first")
        else:
            pass
        current = self.digest_model()
        missing = sorted(
            set(self._snapshot) - set(current)
        )  # noqa: leading-underscore  # upstream spelling, or the public name is already taken
        unexpected = sorted(
            set(current) - set(self._snapshot)
        )  # noqa: leading-underscore  # upstream spelling, or the public name is already taken
        changed = [
            name
            for name in sorted(
                set(self._snapshot) & set(current)
            )  # noqa: leading-underscore  # upstream spelling, or the public name is already taken
            if self._snapshot[name].sha256
            != current[
                name
            ].sha256  # noqa: leading-underscore  # upstream spelling, or the public name is already taken
            or self._snapshot[name].shape
            != current[
                name
            ].shape  # noqa: leading-underscore  # upstream spelling, or the public name is already taken
            or self._snapshot[name].dtype
            != current[
                name
            ].dtype  # noqa: leading-underscore  # upstream spelling, or the public name is already taken
        ]
        summary = self.summary(current, action="compare")
        summary.update(
            {
                "matched": not missing and not unexpected and not changed,
                "missing": missing,
                "unexpected": unexpected,
                "changed": changed,
            }
        )
        return summary

    def digest_model(self) -> dict[str, TensorDigest]:
        model = getattr(self.model_runner, "model", None)
        if model is None:
            raise RuntimeError("model_runner has no model for weights_checker")
        else:
            pass

        logger.warning(
            "weights_checker: starting full-model SHA256 digest; "
            "inference is blocked until this completes. "
            "Elapsed time will be reported in the response."
        )
        t0 = time.time()
        digests: dict[str, TensorDigest] = {}
        for name, tensor in self.iter_named_tensors(model):
            digests[name] = digest_tensor(name, tensor)
        logger.warning(
            "weights_checker: digest complete; %d tensors in %.1fs",
            len(digests),
            time.time() - t0,
        )
        return digests

    @staticmethod
    def iter_named_tensors(model: "Module") -> Iterator[tuple[str, "Tensor"]]:
        seen: set[int] = set()
        named_parameters = getattr(model, "named_parameters", None)
        if callable(named_parameters):
            for name, tensor in named_parameters():
                obj_id = id(tensor)
                if obj_id in seen:
                    continue
                else:
                    pass
                seen.add(obj_id)
                yield name, tensor
        else:
            pass

        named_buffers = getattr(model, "named_buffers", None)
        if callable(named_buffers):
            for name, tensor in named_buffers():
                obj_id = id(tensor)
                if obj_id in seen:
                    continue
                else:
                    pass
                seen.add(obj_id)
                yield name, tensor
        else:
            pass

    @staticmethod
    def summary(
        digests: dict[str, TensorDigest],
        *,
        action: str,
    ) -> WeightCheckResult:
        started = time.time()
        tensor_sha = {name: digest.sha256 for name, digest in digests.items()}
        overall = aggregate_checksum(tensor_sha)
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


def digest_tensor(name: str, tensor: "Tensor") -> TensorDigest:
    detached = tensor.detach() if hasattr(tensor, "detach") else tensor
    contiguous = detached.contiguous() if hasattr(detached, "contiguous") else detached
    cpu = contiguous.cpu() if hasattr(contiguous, "cpu") else contiguous
    shape = tuple(int(x) for x in getattr(cpu, "shape", ()))
    dtype = str(getattr(cpu, "dtype", type(cpu).__name__))
    h = hashlib.sha256()
    h.update(name.encode())
    h.update(dtype.encode())
    h.update(str(shape).encode())
    h.update(tensor_bytes(cpu))
    return TensorDigest(name=name, shape=shape, dtype=dtype, sha256=h.hexdigest())


def tensor_bytes(tensor: "Tensor") -> bytes:
    numpy = getattr(tensor, "numpy", None)
    if callable(numpy):
        try:
            return numpy().tobytes()
        except (TypeError, RuntimeError):
            pass
    else:
        pass

    view = getattr(tensor, "view", None)
    if callable(view):
        try:
            import torch

            byte_view = tensor.view(torch.uint8)
            return byte_view.numpy().tobytes()
        except Exception:
            pass
    else:
        pass

    tobytes = getattr(tensor, "tobytes", None)
    if callable(tobytes):
        return tobytes()
    else:
        pass
    raise TypeError(
        f"Cannot extract raw bytes from tensor type {type(tensor).__name__}"
    )


def aggregate_checksum(checksums: dict[str, str]) -> str:
    h = hashlib.sha256()
    for name in sorted(checksums):
        h.update(name.encode())
        h.update(checksums[name].encode())
    return h.hexdigest()
