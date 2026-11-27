# SPDX-License-Identifier: Apache-2.0
"""Coordinated trainer-to-worker merged LoRA weight transfer and session lifecycle."""

from __future__ import annotations

import logging
import os
import threading
import uuid
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch.multiprocessing import reductions

from sglang_omni.lora.ipc_wire import (
    decode_wire_tensor,
    encode_cuda_ipc_handle,
    encode_inline_tensor,
    release_unpublished_ipc_handle,
)
from sglang_omni.lora.merged_manager import (
    LoRAAdapterTensor,
    LoRARequestPayload,
    MergedLoRAManager,
    normalize_target_modules,
)
from sglang_omni.proto.admin import (
    ADMIN_ABORT_LORA_UPDATE,
    ADMIN_FINISH_LORA_UPDATE,
    ADMIN_RESTORE_BASE_WEIGHTS,
    ADMIN_START_LORA_UPDATE,
    ADMIN_UPDATE_LORA_WEIGHTS,
)

logger = logging.getLogger(__name__)

EXPANDABLE_SEGMENTS_KEY = "expandable_segments:True"
WARNED_ALLOCATOR_PIDS: set[int] = set()


def warn_if_allocator_not_expandable(
    *,
    is_cuda: bool | None = None,
    environ: Mapping[str, str] | None = None,
) -> bool:
    """Warn once per CUDA worker process when expandable_segments:True is unset."""
    cuda_active = torch.cuda.is_available() if is_cuda is None else bool(is_cuda)
    if not cuda_active:
        return False
    else:
        pass
    pid = os.getpid()
    if pid in WARNED_ALLOCATOR_PIDS:
        return False
    else:
        pass
    env = os.environ if environ is None else environ
    alloc_conf = str(env.get("PYTORCH_CUDA_ALLOC_CONF") or "")
    if EXPANDABLE_SEGMENTS_KEY.lower() in alloc_conf.lower():
        return False
    else:
        pass
    WARNED_ALLOCATOR_PIDS.add(pid)
    logger.warning(
        "Merged LoRA weight updates on CUDA work best with "
        "PYTORCH_CUDA_ALLOC_CONF=%s; current value is %r",
        EXPANDABLE_SEGMENTS_KEY,
        alloc_conf,
    )
    return True


def parse_lora_tensor_name(param_name: str) -> tuple[str, str] | None:
    """Split a PEFT-style LoRA parameter name into (base_module_name, 'a' | 'b')."""
    for suffix, side in (
        (".lora_A.weight", "a"),
        (".lora_B.weight", "b"),
        (".lora_A.default.weight", "a"),
        (".lora_B.default.weight", "b"),
        (".lora_a.weight", "a"),
        (".lora_b.weight", "b"),
    ):
        if param_name.endswith(suffix):
            return param_name[: -len(suffix)], side
        else:
            pass
    return None


class LoRAAdapterWeightSource:
    """Deterministic sorted LoRA tensor source pairing lora_A and lora_B factors."""

    def __init__(
        self,
        adapter_id: str,
        target_modules: object,
        named_tensors: Mapping[str, torch.Tensor] | Iterable[tuple[str, torch.Tensor]],
        *,
        alpha: float | None = None,
        scale: float = 1.0,
    ) -> None:
        if not isinstance(adapter_id, str) or not adapter_id.strip():
            raise ValueError("adapter_id must be a non-empty str")
        else:
            pass
        self.adapter_id = adapter_id.strip()
        self.target_modules = normalize_target_modules(target_modules)
        self.alpha = alpha
        self.scale = float(scale)
        items = (
            list(named_tensors.items())
            if isinstance(named_tensors, Mapping)
            else list(named_tensors)
        )
        self.named_tensors: tuple[tuple[str, torch.Tensor], ...] = tuple(
            sorted(items, key=lambda pair: pair[0])
        )

    def sorted_items(self) -> list[tuple[str, torch.Tensor]]:
        """Return (name, tensor) entries sorted deterministically by name."""
        return list(self.named_tensors)

    def build_adapter_tensors(self) -> tuple[LoRAAdapterTensor, ...]:
        """Pair sorted lora_A and lora_B entries into validated LoRAAdapterTensors."""
        grouped: dict[str, dict[str, torch.Tensor]] = {}
        for name, tensor in self.named_tensors:
            parsed = parse_lora_tensor_name(name)
            if parsed is None:
                raise ValueError(
                    f"unrecognized LoRA tensor name {name!r}; expected "
                    "*.lora_A.weight / *.lora_B.weight"
                )
            else:
                pass
            module_name, side = parsed
            pair = grouped.setdefault(module_name, {})
            if side in pair:
                raise ValueError(
                    f"duplicate LoRA {side} tensor for module {module_name!r}"
                )
            else:
                pass
            pair[side] = tensor

        results: list[LoRAAdapterTensor] = []
        for module_name in sorted(grouped.keys()):
            pair = grouped[module_name]
            if "a" not in pair or "b" not in pair:
                raise ValueError(
                    f"incomplete LoRA pair for module {module_name!r}: "
                    f"present={sorted(pair.keys())}"
                )
            else:
                pass
            lora_a = pair["a"]
            lora_b = pair["b"]
            rank = int(lora_a.shape[0])
            if rank <= 0:
                raise ValueError(f"invalid LoRA rank {rank} for {module_name!r}")
            else:
                pass
            scaling = (
                (float(self.alpha) / float(rank)) if self.alpha is not None else 1.0
            )
            results.append(
                LoRAAdapterTensor(
                    module_name=module_name,
                    lora_a=lora_a,
                    lora_b=lora_b,
                    scaling=scaling,
                )
            )
        return tuple(results)


class MergedLoRAWorkerSession:
    """Transactional worker-side session for staged merged LoRA weight updates."""

    def __init__(
        self,
        manager: MergedLoRAManager,
        *,
        on_cleanup: Callable[[], None] | None = None,
        is_cuda: bool | None = None,
        rebuild_cuda_fn: Callable[..., torch.Tensor] | None = None,
    ) -> None:
        self.manager = manager
        self.on_cleanup = on_cleanup
        self.is_cuda = is_cuda
        self.rebuild_cuda_fn = rebuild_cuda_fn
        self.active_update_id: str | None = None
        self.staged_adapter_id: str | None = None
        self.staged_park_location: str | None = None
        self.staged_match_count: int = 0
        self.session_finished: bool = False
        self.finish_attempted: bool = False

    def run_cleanup_safely(self, context: str) -> None:
        if self.on_cleanup is None:
            return
        else:
            pass
        try:
            self.on_cleanup()
        except Exception:
            logger.warning(
                "Merged LoRA session cleanup failed during %s",
                context,
                exc_info=True,
            )

    def start_update(self, update_id: str) -> dict[str, Any]:
        """Begin a transactional merged LoRA update session identified by update_id."""
        if not isinstance(update_id, str) or not update_id.strip():
            raise ValueError("update_id must be a non-empty str")
        else:
            pass
        warn_if_allocator_not_expandable(is_cuda=self.is_cuda)
        if self.active_update_id is not None:
            if self.active_update_id == update_id:
                return {
                    "update_id": update_id,
                    "resumed": True,
                }
            else:
                raise RuntimeError(
                    f"another LoRA update session is already active: "
                    f"active={self.active_update_id!r}, requested={update_id!r}"
                )
        else:
            pass
        self.active_update_id = update_id
        self.staged_adapter_id = None
        self.staged_park_location = None
        self.staged_match_count = 0
        self.session_finished = False
        self.finish_attempted = False
        return {"update_id": update_id, "resumed": False}

    def stage_update(
        self,
        update_id: str,
        payload: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Decode incoming LoRA wire tensors, bind them, and stage merged weights."""
        if self.active_update_id != update_id:
            raise RuntimeError(
                f"stage_update update_id mismatch: active={self.active_update_id!r}, "
                f"got={update_id!r}"
            )
        else:
            pass

        adapter_id = str(payload.get("adapter_id") or "")
        target_modules = payload.get("target_modules")
        alpha = payload.get("alpha")
        scale = float(payload.get("scale", 1.0))
        restore_omitted = bool(payload.get("restore_omitted", True))

        raw_tensors = payload.get("tensors")
        if not isinstance(raw_tensors, Mapping) or not raw_tensors:
            raise ValueError("payload['tensors'] must be a non-empty mapping")
        else:
            pass

        decoded: dict[str, torch.Tensor] = {}
        for name, encoded in raw_tensors.items():
            if isinstance(encoded, torch.Tensor):
                decoded[str(name)] = encoded
            elif isinstance(encoded, Mapping):
                decoded[str(name)] = decode_wire_tensor(
                    encoded, rebuild_cuda_fn=self.rebuild_cuda_fn
                )
            else:
                raise ValueError(
                    f"unsupported tensor value for {name!r}: {type(encoded).__name__}"
                )

        source = LoRAAdapterWeightSource(
            adapter_id=adapter_id,
            target_modules=target_modules,
            named_tensors=decoded,
            alpha=float(alpha) if alpha is not None else None,
            scale=scale,
        )
        request = LoRARequestPayload(
            adapter_id=source.adapter_id,
            target_modules=source.target_modules,
            tensors=source.build_adapter_tensors(),
            scale=source.scale,
        )

        # Drop any prior uncommitted staging before preparing the replacement batch.
        if self.staged_adapter_id is not None:
            self.manager.rollback_adapter()
            self.staged_adapter_id = None
        else:
            pass

        park_location, matched = self.manager.prepare_adapter(
            request, restore_omitted=restore_omitted
        )
        self.staged_adapter_id = request.adapter_id
        self.staged_park_location = park_location
        self.staged_match_count = matched
        return {
            "update_id": update_id,
            "adapter_id": request.adapter_id,
            "matched_modules": matched,
            "park_location": park_location,
        }

    def finish_update(self, update_id: str) -> dict[str, Any]:
        """Atomically commit staged weights in-place and close the session."""
        if self.active_update_id != update_id:
            raise RuntimeError(
                f"finish_update update_id mismatch: active={self.active_update_id!r}, "
                f"got={update_id!r}"
            )
        else:
            pass
        if self.staged_adapter_id is None:
            raise RuntimeError(
                f"cannot finish_update {update_id!r}: no staged LoRA weights"
            )
        else:
            pass
        self.finish_attempted = True
        adapter_id = self.staged_adapter_id
        updated = self.manager.commit_adapter(adapter_id)
        # Mark finished only after commit succeeds so interrupted finishes abort cleanly.
        self.session_finished = True
        self.active_update_id = None
        self.staged_adapter_id = None
        self.staged_park_location = None
        self.staged_match_count = 0
        self.run_cleanup_safely("finish_update")
        return {
            "update_id": update_id,
            "adapter_id": adapter_id,
            "updated_params": updated,
        }

    def abort_update(self, update_id: str | None = None) -> dict[str, Any]:
        """Abort the active session without committing any staged weights."""
        if (
            update_id is not None
            and self.active_update_id is not None
            and self.active_update_id != update_id
        ):
            raise RuntimeError(
                f"abort_update update_id mismatch: active={self.active_update_id!r}, "
                f"got={update_id!r}"
            )
        else:
            pass
        aborted_id = self.active_update_id
        try:
            self.manager.rollback_adapter()
        finally:
            self.active_update_id = None
            self.staged_adapter_id = None
            self.staged_park_location = None
            self.staged_match_count = 0
            self.session_finished = False
            self.finish_attempted = False
            self.run_cleanup_safely("abort_update")
        return {
            "aborted_update_id": aborted_id,
            "committed": False,
        }

    def restore_base(self) -> dict[str, Any]:
        """Restore pristine host base weights when no update session is active."""
        if self.active_update_id is not None:
            raise RuntimeError(
                f"cannot restore_base while session {self.active_update_id!r} is active"
            )
        else:
            pass
        restored = self.manager.restore_base()
        return {"restored_params": restored}


@dataclass(frozen=True)
class TrainerSyncResult:
    """Outcome summary returned by MergedLoRATrainerSync.sync_adapter."""

    update_id: str
    adapter_id: str
    updated_params: int
    tensor_count: int


class MergedLoRATrainerSync:
    """Trainer-side multi-rank coordinator for engine-side merged LoRA IPC sync."""

    def __init__(
        self,
        *,
        start_rpc: Callable[[str], Mapping[str, Any]],
        stage_rpc: Callable[[str, Mapping[str, Any]], Mapping[str, Any]],
        finish_rpc: Callable[[str], Mapping[str, Any]],
        abort_rpc: Callable[[str], Mapping[str, Any]],
        all_gather_fn: Callable[[object], list[object]] | None = None,
        synchronize_fn: Callable[[], None] | None = None,
        ipc_collect_fn: Callable[[], None] | None = None,
        export_tensor_fn: Callable[[torch.Tensor], dict[str, Any]] | None = None,
        release_handle_fn: Callable[[Mapping[str, Any]], bool] | None = None,
    ) -> None:
        self.start_rpc = start_rpc
        self.stage_rpc = stage_rpc
        self.finish_rpc = finish_rpc
        self.abort_rpc = abort_rpc
        self.all_gather_fn = all_gather_fn
        self.synchronize_fn = synchronize_fn
        self.ipc_collect_fn = ipc_collect_fn
        self.export_tensor_fn = export_tensor_fn
        self.release_handle_fn = release_handle_fn

    def coordinate_statuses(self, local_error: str | None) -> list[str | None]:
        if self.all_gather_fn is None:
            return [local_error]
        else:
            raw = self.all_gather_fn(local_error)
            return [None if item is None else str(item) for item in raw]

    def default_export_tensor(self, tensor: torch.Tensor) -> dict[str, Any]:
        if self.export_tensor_fn is not None:
            return self.export_tensor_fn(tensor)
        else:
            pass
        if tensor.device.type == "cuda":
            reduced = reductions.reduce_tensor(tensor)
            return encode_cuda_ipc_handle(reduced)
        else:
            return encode_inline_tensor(tensor)

    def release_exports(self, exported: Sequence[Mapping[str, Any]]) -> None:
        releaser = (
            self.release_handle_fn
            if self.release_handle_fn is not None
            else release_unpublished_ipc_handle
        )
        for handle in exported:
            releaser(handle)

    def abort_remote_safely(self, update_id: str) -> None:
        try:
            self.abort_rpc(update_id)
        except Exception:
            logger.warning(
                "Remote abort_rpc failed for update_id=%s",
                update_id,
                exc_info=True,
            )

    def sync_adapter(
        self,
        source: LoRAAdapterWeightSource,
        *,
        update_id: str | None = None,
        restore_omitted: bool = True,
    ) -> TrainerSyncResult:
        """Execute coordinated two-phase LoRA sync from trainer to worker(s)."""
        session_id = update_id if update_id is not None else uuid.uuid4().hex
        staged_refs: list[torch.Tensor] = []
        exported_wire: dict[str, dict[str, Any]] = {}
        local_prep_error: str | None = None
        prepared_items: list[tuple[str, torch.Tensor]] = []

        # Phase 1a: local preparation and device synchronization before any IPC export.
        try:
            # Validate pairs and sorted alignment deterministically before exporting.
            source.build_adapter_tensors()
            for name, tensor in source.sorted_items():
                contig = tensor.detach().contiguous()
                staged_refs.append(contig)
                prepared_items.append((name, contig))
            if self.synchronize_fn is not None:
                self.synchronize_fn()
            else:
                pass
        except Exception as exc:
            local_prep_error = f"{type(exc).__name__}: {exc}"

        # Phase 1b: coordinate preparation outcomes across all ranks before exporting handles.
        prep_errors = [
            err for err in self.coordinate_statuses(local_prep_error) if err is not None
        ]
        if prep_errors:
            staged_refs.clear()
            raise RuntimeError(
                f"LoRA sync preparation failed before IPC export: {'; '.join(prep_errors)}"
            )
        else:
            pass

        # Phase 2: export wire handles only after every rank passed Phase 1.
        export_error: str | None = None
        try:
            for name, contig in prepared_items:
                exported_wire[name] = self.default_export_tensor(contig)
        except Exception as exc:
            export_error = f"{type(exc).__name__}: {exc}"

        export_errors = [
            err for err in self.coordinate_statuses(export_error) if err is not None
        ]
        if export_errors:
            self.release_exports(list(exported_wire.values()))
            staged_refs.clear()
            raise RuntimeError(
                f"LoRA IPC export failed before publish: {'; '.join(export_errors)}"
            )
        else:
            pass

        # Phase 3: transactional remote start -> stage -> finish with abort recovery.
        stage_sent = False
        session_finished = False
        updated_params = 0
        rpc_error: str | None = None
        try:
            self.start_rpc(session_id)
            stage_payload: dict[str, Any] = {
                "adapter_id": source.adapter_id,
                "target_modules": (
                    list(source.target_modules)
                    if isinstance(source.target_modules, tuple)
                    else source.target_modules
                ),
                "alpha": source.alpha,
                "scale": source.scale,
                "restore_omitted": restore_omitted,
                "tensors": exported_wire,
            }
            stage_sent = True
            self.stage_rpc(session_id, stage_payload)
            finish_resp = self.finish_rpc(session_id)
            updated_params = int(finish_resp.get("updated_params", 0))
            session_finished = True
        except Exception as exc:
            rpc_error = f"{type(exc).__name__}: {exc}"
            if not session_finished:
                self.abort_remote_safely(session_id)
            else:
                pass
            if not stage_sent:
                self.release_exports(list(exported_wire.values()))
            else:
                pass

        # Hold staging tensor references alive past the post-ack collective barrier.
        rpc_errors = [
            err for err in self.coordinate_statuses(rpc_error) if err is not None
        ]
        if rpc_errors and session_finished:
            # Peer failed after our local finish; report failure cleanly.
            pass
        else:
            pass

        staged_refs.clear()
        if self.ipc_collect_fn is not None:
            self.ipc_collect_fn()
        elif torch.cuda.is_available():
            torch.cuda.ipc_collect()
        else:
            pass

        if rpc_errors:
            raise RuntimeError(f"LoRA sync failed: {'; '.join(rpc_errors)}")
        else:
            pass

        return TrainerSyncResult(
            update_id=session_id,
            adapter_id=source.adapter_id,
            updated_params=updated_params,
            tensor_count=len(prepared_items),
        )


def build_merged_lora_admin_handler(
    session: MergedLoRAWorkerSession,
    *,
    lock: threading.Lock | None = None,
) -> Callable[[str, dict[str, object]], dict[str, object]]:
    """Build a SimpleScheduler-compatible admin handler for a LoRA worker session."""
    guard = lock if lock is not None else threading.Lock()

    def handle_admin(action: str, payload: dict[str, object]) -> dict[str, object]:
        with guard:
            try:
                if action == ADMIN_START_LORA_UPDATE:
                    data = session.start_update(str(payload.get("update_id") or ""))
                    return {"success": True, "message": "ok", "data": data}
                elif action == ADMIN_UPDATE_LORA_WEIGHTS:
                    data = session.stage_update(
                        str(payload.get("update_id") or ""), payload
                    )
                    return {"success": True, "message": "ok", "data": data}
                elif action == ADMIN_FINISH_LORA_UPDATE:
                    data = session.finish_update(str(payload.get("update_id") or ""))
                    return {"success": True, "message": "ok", "data": data}
                elif action == ADMIN_ABORT_LORA_UPDATE:
                    raw_id = payload.get("update_id")
                    data = session.abort_update(None if raw_id is None else str(raw_id))
                    return {"success": True, "message": "ok", "data": data}
                elif action == ADMIN_RESTORE_BASE_WEIGHTS:
                    data = session.restore_base()
                    return {"success": True, "message": "ok", "data": data}
                else:
                    return {
                        "success": True,
                        "message": f"unsupported admin action: {action}",
                        "data": {"skipped": True, "unsupported": True},
                    }
            except Exception as exc:
                return {
                    "success": False,
                    "message": str(exc),
                    "error": str(exc),
                    "data": {},
                }

    return handle_admin
