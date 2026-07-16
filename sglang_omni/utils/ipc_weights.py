# SPDX-License-Identifier: Apache-2.0
"""Same-GPU weight sharing over CUDA IPC for same-GPU data parallelism.

One replica per GPU is the weight LEADER: it loads checkpoint weights
normally, serializes CUDA-IPC handles for every model parameter and buffer
(via sglang's ``MultiprocessingSerializer`` + ``monkey_patch_torch_reductions``,
the ``update_weights_from_tensor`` lineage), and atomically publishes the
handle blob to a file. FOLLOWER replicas build the same module tree with
dummy weights (``load_format=dummy``, no checkpoint I/O), then alias every
parameter/buffer onto the leader's storage (``param.data = shared_tensor``,
assign — never copy). KV cache, CUDA graphs, and sampler state stay private
per replica.

Ordering contract (enforced by the caller, verified here):

- The leader exports at the end of ``ModelRunner.load_model``. Builder-level
  hooks (e.g. Higgs ``setup_model``'s ``truncate_rope_to_bf16``) run LATER and
  mutate the already-shared storage: such hooks must stay in-place,
  idempotent, and deterministic, so a follower's redundant application writes
  identical bytes to the shared buffer.
- Followers attach before KV-pool profiling, warmup forwards, and CUDA graph
  capture. :func:`verify_attachment` re-checks storage identity right before
  graph capture so a silently re-initialized tensor fails loudly.
- The leader must outlive followers (CUDA IPC mappings die with the exporting
  process); restart replicas together.

Role selection is environment-driven so process supervisors (e.g.
``examples/mps_dp/launch.sh``) can stay declarative::

    SGLANG_OMNI_WEIGHT_SHARE=leader:/path/to/dir     # replica 0
    SGLANG_OMNI_WEIGHT_SHARE=follower:/path/to/dir   # replicas 1..N-1

The path is a *directory*; each SGLang engine writes/reads
``<dir>/<ModelClass>.weights-ipc``, so multi-engine stages cannot clobber
each other. Unset (or empty) means weight sharing is off.
"""

from __future__ import annotations

import hashlib
import io
import logging
import os
import pickle
import time
from dataclasses import dataclass
from typing import Any, Callable

import torch

logger = logging.getLogger(__name__)

ENV_WEIGHT_SHARE = "SGLANG_OMNI_WEIGHT_SHARE"
ENV_WEIGHT_SHARE_TIMEOUT_S = "SGLANG_OMNI_WEIGHT_SHARE_TIMEOUT_S"
DEFAULT_ATTACH_TIMEOUT_S = 1800.0
_FORMAT_VERSION = 1
_FILE_SUFFIX = ".weights-ipc"

# Paths this process has already exported to; a second export to the same
# file would silently invalidate handles followers may already hold.
_EXPORTED_FILES: set[str] = set()


class WeightShareError(RuntimeError):
    """Protocol violation during weight-share export/attach."""


@dataclass(frozen=True)
class WeightShareConfig:
    role: str  # "leader" | "follower"
    dir_path: str
    attach_timeout_s: float


def get_weight_share_config(environ=None) -> WeightShareConfig | None:
    """Parse ``SGLANG_OMNI_WEIGHT_SHARE``; ``None`` when unset/empty (off)."""
    env = os.environ if environ is None else environ
    raw = (env.get(ENV_WEIGHT_SHARE) or "").strip()
    if not raw:
        return None
    role, sep, dir_path = raw.partition(":")
    role = role.strip().lower()
    dir_path = dir_path.strip()
    if not sep or role not in ("leader", "follower") or not dir_path:
        raise ValueError(
            f"{ENV_WEIGHT_SHARE} must be 'leader:<dir>' or 'follower:<dir>', "
            f"got {raw!r}"
        )
    timeout_raw = (env.get(ENV_WEIGHT_SHARE_TIMEOUT_S) or "").strip()
    if timeout_raw:
        try:
            timeout_s = float(timeout_raw)
        except ValueError as exc:
            raise ValueError(
                f"{ENV_WEIGHT_SHARE_TIMEOUT_S} must be a number of seconds, "
                f"got {timeout_raw!r}"
            ) from exc
        if timeout_s <= 0:
            raise ValueError(f"{ENV_WEIGHT_SHARE_TIMEOUT_S} must be > 0")
    else:
        timeout_s = DEFAULT_ATTACH_TIMEOUT_S
    return WeightShareConfig(
        role=role, dir_path=dir_path, attach_timeout_s=timeout_s
    )


def handle_file_for_model(dir_path: str, model: torch.nn.Module) -> str:
    """``<dir>/<ModelClass>.weights-ipc`` — one handle file per engine."""
    return os.path.join(dir_path, type(model).__name__ + _FILE_SUFFIX)


class _SglangIpcSerializer:
    """CUDA-IPC (de)serialization via sglang's RLHF weight-update machinery.

    ``monkey_patch_torch_reductions`` rewrites the device index in the reduced
    tensor args to the device UUID, which is what makes the handles robust to
    per-process CUDA_VISIBLE_DEVICES orderings (each MPS-DP replica sees the
    GPU under a UUID-pinned CUDA_VISIBLE_DEVICES).
    """

    @staticmethod
    def serialize(obj: Any) -> bytes:
        from sglang.srt.utils.common import MultiprocessingSerializer
        from sglang.srt.utils.patch_torch import monkey_patch_torch_reductions

        monkey_patch_torch_reductions()
        return MultiprocessingSerializer.serialize(obj)

    @staticmethod
    def deserialize(data: bytes) -> Any:
        from sglang.srt.utils.common import MultiprocessingSerializer
        from sglang.srt.utils.patch_torch import monkey_patch_torch_reductions

        monkey_patch_torch_reductions()
        return MultiprocessingSerializer.deserialize(data)


def _named_shared_tensors(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    """All named parameters + buffers, deduplicated, name-collision checked.

    ``remove_duplicate=True`` (the default) folds tied parameters (e.g. the
    Higgs ``modality_head.weight`` ↔ embedding tie) into a single canonical
    name on both leader and follower; assigning ``.data`` through that single
    Parameter object updates every module that holds it.
    """
    tensors: dict[str, torch.Tensor] = {}
    for source in (model.named_parameters(), model.named_buffers()):
        for name, tensor in source:
            if name in tensors:
                raise WeightShareError(
                    f"duplicate tensor name across parameters/buffers: {name!r}"
                )
            tensors[name] = tensor
    return tensors


def _manifest_hash(tensors: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(tensors):
        t = tensors[name]
        digest.update(
            f"{name}|{t.dtype}|{tuple(t.shape)}\n".encode("utf-8")
        )
    return digest.hexdigest()


def _tensor_to_value_bytes(tensor: torch.Tensor) -> bytes:
    buf = io.BytesIO()
    torch.save(tensor.detach().cpu(), buf)
    return buf.getvalue()


def _value_bytes_to_tensor(data: bytes) -> torch.Tensor:
    return torch.load(io.BytesIO(data), map_location="cpu", weights_only=True)


def _atomic_write(file_path: str, data: bytes) -> None:
    """Write ``data`` to ``file_path`` via tmp+fsync+rename (atomic publish).

    Followers poll for ``file_path``; rename atomicity guarantees they can
    never observe a partially written blob.
    """
    dir_path = os.path.dirname(os.path.abspath(file_path))
    os.makedirs(dir_path, exist_ok=True)
    tmp_path = f"{file_path}.tmp.{os.getpid()}"
    try:
        with open(tmp_path, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.rename(tmp_path, file_path)
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
    try:
        dir_fd = os.open(dir_path, os.O_RDONLY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
    except OSError:
        pass  # best effort; rename atomicity already holds on the same fs


def export_weights(
    model: torch.nn.Module,
    file_path: str,
    *,
    serializer: Any | None = None,
    alias_predicate: Callable[[torch.Tensor], bool] | None = None,
) -> dict[str, tuple[int, tuple[int, ...], torch.dtype]]:
    """Publish CUDA-IPC handles for every model parameter/buffer to a file.

    CUDA tensors are shared zero-copy through ``serializer`` (default:
    sglang's ForkingPickler-based ``MultiprocessingSerializer``). Non-CUDA
    tensors (rare; e.g. CPU-resident buffers) are embedded *by value* because
    ForkingPickler's CPU shared-memory reduction does not survive a
    file-based handoff between unrelated processes; followers copy those.

    Returns a record ``{name: (data_ptr, shape, dtype)}`` of the IPC-shared
    (leader-side) tensors for later :func:`verify_attachment` — the leader's
    own storages must stay put too, since followers alias them. In-place
    mutation (``copy_``) is fine; rebinding ``.data`` after export is not.
    """
    abs_path = os.path.abspath(file_path)
    if abs_path in _EXPORTED_FILES:
        raise WeightShareError(
            f"this process already exported weights to {abs_path}; a second "
            "export would invalidate handles held by attached followers"
        )
    serializer = _SglangIpcSerializer if serializer is None else serializer
    alias_predicate = (
        (lambda t: t.is_cuda) if alias_predicate is None else alias_predicate
    )

    tensors = _named_shared_tensors(model)
    if not tensors:
        raise WeightShareError("model has no parameters or buffers to export")
    ipc_tensors = {n: t for n, t in tensors.items() if alias_predicate(t)}
    value_tensors = {n: t for n, t in tensors.items() if n not in ipc_tensors}

    payload = {
        "format_version": _FORMAT_VERSION,
        "model_class": type(model).__name__,
        "manifest_hash": _manifest_hash(tensors),
        "torch_version": torch.__version__,
        "pid": os.getpid(),
        "ipc_blob": serializer.serialize(ipc_tensors),
        "ipc_names": sorted(ipc_tensors),
        "value_blobs": {
            n: _tensor_to_value_bytes(t) for n, t in value_tensors.items()
        },
    }
    _atomic_write(abs_path, pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL))
    _EXPORTED_FILES.add(abs_path)

    record = {
        name: (t.data_ptr(), tuple(t.shape), t.dtype)
        for name, t in ipc_tensors.items()
    }
    shared_bytes = sum(t.numel() * t.element_size() for t in ipc_tensors.values())
    logger.info(
        f"[weight-share] leader exported {len(ipc_tensors)} CUDA tensors "
        f"({shared_bytes / (1 << 30):.2f} GiB shared zero-copy) + "
        f"{len(value_tensors)} by-value tensors to {abs_path}"
    )
    return record


def wait_for_export(
    file_path: str,
    timeout_s: float,
    poll_interval_s: float = 0.5,
) -> None:
    """Block until the leader's handle file exists (atomic rename = complete)."""
    deadline = time.monotonic() + timeout_s
    while not os.path.exists(file_path):
        if time.monotonic() >= deadline:
            raise TimeoutError(
                f"[weight-share] timed out after {timeout_s:.0f}s waiting for "
                f"leader handle file {file_path}; is the leader replica up?"
            )
        time.sleep(poll_interval_s)


def wait_for_any_export(
    dir_path: str,
    timeout_s: float,
    poll_interval_s: float = 0.5,
) -> None:
    """Block until any leader handle file exists under ``dir_path``.

    Used by followers *before* allocating dummy weights, when the engine's
    model class (hence exact file name) is not known yet; attach still waits
    on and validates the engine-specific file afterwards.
    """
    deadline = time.monotonic() + timeout_s
    while True:
        try:
            entries = os.listdir(dir_path)
        except FileNotFoundError:
            entries = []
        if any(e.endswith(_FILE_SUFFIX) for e in entries):
            return
        if time.monotonic() >= deadline:
            raise TimeoutError(
                f"[weight-share] timed out after {timeout_s:.0f}s waiting for "
                f"a leader handle file under {dir_path}; is the leader "
                "replica up?"
            )
        time.sleep(poll_interval_s)


def _load_payload(file_path: str, model: torch.nn.Module) -> dict[str, Any]:
    with open(file_path, "rb") as fh:
        payload = pickle.load(fh)
    if not isinstance(payload, dict) or payload.get("format_version") != _FORMAT_VERSION:
        raise WeightShareError(
            f"unsupported weight-share handle file format in {file_path} "
            f"(got {payload.get('format_version') if isinstance(payload, dict) else type(payload)})"
        )
    if payload["model_class"] != type(model).__name__:
        raise WeightShareError(
            f"handle file {file_path} was exported for model class "
            f"{payload['model_class']!r}, follower model is {type(model).__name__!r}"
        )
    return payload


def attach_weights(
    model: torch.nn.Module,
    file_path: str,
    *,
    timeout_s: float = DEFAULT_ATTACH_TIMEOUT_S,
    poll_interval_s: float = 0.5,
    serializer: Any | None = None,
) -> dict[str, tuple[int, tuple[int, ...], torch.dtype]]:
    """Alias every model parameter/buffer onto the leader's exported storage.

    ``param.data = shared`` / ``module._buffers[leaf] = shared`` — assign,
    never copy — so the follower's dummy-initialized storages are dropped and
    freed. Hard-errors on any name/shape/dtype mismatch in either direction.

    Returns the attachment record for :func:`verify_attachment`.
    """
    serializer = _SglangIpcSerializer if serializer is None else serializer
    wait_for_export(file_path, timeout_s, poll_interval_s)
    payload = _load_payload(file_path, model)

    own_tensors = _named_shared_tensors(model)
    own_hash = _manifest_hash(own_tensors)
    if own_hash != payload["manifest_hash"]:
        _raise_manifest_mismatch(model, payload, own_tensors, file_path)

    shared: dict[str, torch.Tensor] = serializer.deserialize(payload["ipc_blob"])
    if sorted(shared) != payload["ipc_names"]:
        raise WeightShareError(
            f"handle blob in {file_path} deserialized to a different tensor "
            "set than its manifest declares"
        )
    values = {
        n: _value_bytes_to_tensor(b) for n, b in payload["value_blobs"].items()
    }

    params = dict(model.named_parameters())
    # A buffer tensor may be registered under several module paths (tied /
    # shared modules); named_buffers deduplicates to one canonical name, but
    # rebinding must cover every registration or non-canonical holders would
    # keep the follower's dummy storage.
    buffer_paths_by_id: dict[int, list[str]] = {}
    for dotted, buf in model.named_buffers(remove_duplicate=False):
        buffer_paths_by_id.setdefault(id(buf), []).append(dotted)
    aliased_bytes = 0
    for name, own in own_tensors.items():
        if name in shared:
            incoming = shared[name]
        elif name in values:
            incoming = values[name]
        else:  # unreachable while the manifest hash matches; belt & braces
            raise WeightShareError(f"leader export is missing tensor {name!r}")
        if tuple(incoming.shape) != tuple(own.shape) or incoming.dtype != own.dtype:
            raise WeightShareError(
                f"tensor {name!r} mismatch: leader "
                f"{tuple(incoming.shape)}/{incoming.dtype}, follower "
                f"{tuple(own.shape)}/{own.dtype}"
            )
        if incoming.is_cuda and incoming.device != own.device:
            # A UUID-remap failure would otherwise surface later as a
            # mixed-device kernel error that doesn't name weight sharing.
            raise WeightShareError(
                f"tensor {name!r} arrived on {incoming.device}, expected "
                f"{own.device}: CUDA IPC handle mapped to the wrong device"
            )
        if name in shared:
            # Alias (assign, not copy): the follower drops its own storage.
            if name in params:
                # Tied params are one Parameter object; .data assignment
                # propagates to every module holding it.
                params[name].data = incoming
            else:
                for dotted in buffer_paths_by_id.get(id(own), [name]):
                    _rebind_buffer(model, dotted, incoming)
            aliased_bytes += incoming.numel() * incoming.element_size()
        else:
            # By-value (non-CUDA) tensors keep private storage; copy contents.
            with torch.no_grad():
                own.copy_(incoming.to(own.device))

    record: dict[str, tuple[int, tuple[int, ...], torch.dtype]] = {}
    for name, tensor in _named_shared_tensors(model).items():
        if name in shared:
            expected = shared[name]
            if tensor.data_ptr() != expected.data_ptr():
                raise WeightShareError(
                    f"aliasing failed for {name!r}: module tensor does not "
                    "point at the shared storage after attach"
                )
            record[name] = (tensor.data_ptr(), tuple(tensor.shape), tensor.dtype)
    logger.info(
        f"[weight-share] follower attached {len(record)} shared tensors "
        f"({aliased_bytes / (1 << 30):.2f} GiB aliased, zero-copy) from "
        f"{file_path}"
    )
    return record


def _rebind_buffer(
    model: torch.nn.Module, dotted_name: str, tensor: torch.Tensor
) -> None:
    module_path, _, leaf = dotted_name.rpartition(".")
    module = model.get_submodule(module_path) if module_path else model
    if leaf not in module._buffers:
        raise WeightShareError(
            f"{dotted_name!r} is not a registered buffer on the follower model"
        )
    module._buffers[leaf] = tensor


def _raise_manifest_mismatch(
    model: torch.nn.Module,
    payload: dict[str, Any],
    own_tensors: dict[str, torch.Tensor],
    file_path: str,
) -> None:
    leader_names = set(payload["ipc_names"]) | set(payload["value_blobs"])
    own_names = set(own_tensors)
    missing = sorted(own_names - leader_names)[:5]
    extra = sorted(leader_names - own_names)[:5]
    raise WeightShareError(
        f"weight manifest mismatch between leader export {file_path} and this "
        f"follower ({type(model).__name__}): the replicas are not running "
        f"identical models/configs. follower-only names (first 5): {missing}; "
        f"leader-only names (first 5): {extra}. If names match, a shape/dtype "
        "differs — compare replica configs."
    )


def verify_attachment(
    model: torch.nn.Module,
    record: dict[str, tuple[int, tuple[int, ...], torch.dtype]],
) -> None:
    """Assert attached tensors were not re-initialized since attach.

    Call this immediately before warmup/CUDA-graph capture: any load-path
    step that re-created a parameter after attach (breaking the alias) turns
    into a hard error here instead of silently serving dummy weights.
    """
    current = _named_shared_tensors(model)
    drifted = []
    for name, (ptr, shape, dtype) in record.items():
        tensor = current.get(name)
        if tensor is None:
            drifted.append(f"{name} (tensor disappeared)")
        elif tensor.data_ptr() != ptr:
            drifted.append(f"{name} (storage rebound after attach)")
        elif tuple(tensor.shape) != shape or tensor.dtype != dtype:
            drifted.append(f"{name} (shape/dtype changed after attach)")
    if drifted:
        raise WeightShareError(
            "weight-share attachment was broken after attach; offending "
            f"tensors (first 10): {drifted[:10]}"
        )


def leader_export(
    model: torch.nn.Module,
    dir_path: str,
    *,
    serializer: Any | None = None,
) -> dict[str, tuple[int, tuple[int, ...], torch.dtype]]:
    """Leader role: export this engine's weights under ``dir_path``."""
    return export_weights(
        model, handle_file_for_model(dir_path, model), serializer=serializer
    )


def follower_attach(
    model: torch.nn.Module,
    dir_path: str,
    *,
    timeout_s: float = DEFAULT_ATTACH_TIMEOUT_S,
    serializer: Any | None = None,
) -> dict[str, tuple[int, tuple[int, ...], torch.dtype]]:
    """Follower role: attach this engine's weights from ``dir_path``."""
    return attach_weights(
        model,
        handle_file_for_model(dir_path, model),
        timeout_s=timeout_s,
        serializer=serializer,
    )
