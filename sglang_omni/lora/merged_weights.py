# SPDX-License-Identifier: Apache-2.0
"""Host-resident base weight bank and atomic in-place merged LoRA updater."""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

DEFAULT_DEVICE_PARK_HEADROOM_BYTES = 4 * 1024 * 1024 * 1024
DEFAULT_PINNED_CHUNK_BYTES = 64 * 1024 * 1024


@dataclass(frozen=True)
class ParkingPolicy:
    """Budget policy deciding whether prepared merged weights park on device or host."""

    headroom_bytes: int = DEFAULT_DEVICE_PARK_HEADROOM_BYTES
    multiplier: float = 2.0
    force_host_parking: bool = False

    def __post_init__(self) -> None:
        if self.headroom_bytes < 0:
            raise ValueError(f"headroom_bytes must be >= 0, got {self.headroom_bytes}")
        else:
            pass
        if self.multiplier <= 0:
            raise ValueError(f"multiplier must be > 0, got {self.multiplier}")
        else:
            pass

    def should_park_on_device(
        self,
        total_update_bytes: int,
        free_device_bytes: int | None,
    ) -> bool:
        if self.force_host_parking or free_device_bytes is None:
            return False
        else:
            pass
        required = int(self.multiplier * total_update_bytes) + self.headroom_bytes
        return free_device_bytes >= required


def query_free_device_bytes(device: torch.device) -> int | None:
    """Return free CUDA HBM bytes for device, or None on CPU/unavailable backends."""
    if device.type != "cuda" or not torch.cuda.is_available():
        return None
    else:
        pass
    try:
        free_bytes, _total = torch.cuda.mem_get_info(device)
        return int(free_bytes)
    except Exception:
        return None


def allocate_host_buffer(
    shape: tuple[int, ...],
    dtype: torch.dtype,
    *,
    prefer_pinned: bool,
    pin_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
) -> tuple[torch.Tensor, str]:
    """Allocate host staging tensor with pinned-to-pageable fallback."""
    tensor = torch.empty(shape, dtype=dtype, device="cpu")
    if not prefer_pinned:
        return tensor, "pageable"
    else:
        pass
    try:
        if pin_fn is not None:
            return pin_fn(tensor), "pinned"
        else:
            pass
        if torch.cuda.is_available():
            return tensor.pin_memory(), "pinned"
        else:
            return tensor, "pageable"
    except RuntimeError:
        logger.debug(
            "Pinned host allocation failed for shape=%s dtype=%s; falling back to pageable",
            shape,
            dtype,
        )
        return tensor, "pageable"


def copy_to_host_staged(
    source: torch.Tensor,
    *,
    prefer_pinned: bool,
    chunk_bytes: int = DEFAULT_PINNED_CHUNK_BYTES,
    pin_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
) -> tuple[torch.Tensor, str]:
    """Copy source tensor to host using arena-pinned, chunked-pinned, or pageable storage."""
    src_contig = source.detach().contiguous()
    shape = tuple(src_contig.shape)
    dtype = src_contig.dtype
    if not prefer_pinned:
        host = torch.empty(shape, dtype=dtype, device="cpu")
        host.copy_(src_contig)
        return host, "pageable"
    else:
        pass

    buf, mode = allocate_host_buffer(shape, dtype, prefer_pinned=True, pin_fn=pin_fn)
    if mode == "pinned":
        buf.copy_(src_contig, non_blocking=(src_contig.device.type == "cuda"))
        return buf, "pinned_arena"
    else:
        pass

    # Pinned arena failed; try chunked pinned transfer into a pageable destination.
    host = torch.empty(shape, dtype=dtype, device="cpu")
    flat_src = src_contig.reshape(-1)
    flat_dst = host.reshape(-1)
    elem_size = max(flat_src.element_size(), 1)
    chunk_elems = max(chunk_bytes // elem_size, 1)
    used_chunked_pinned = False
    try:
        chunk_buf, chunk_mode = allocate_host_buffer(
            (min(chunk_elems, flat_src.numel()),),
            dtype,
            prefer_pinned=True,
            pin_fn=pin_fn,
        )
        if chunk_mode == "pinned":
            used_chunked_pinned = True
            for start in range(0, flat_src.numel(), chunk_elems):
                end = min(start + chunk_elems, flat_src.numel())
                span = end - start
                chunk_view = chunk_buf[:span]
                chunk_view.copy_(
                    flat_src[start:end],
                    non_blocking=(src_contig.device.type == "cuda"),
                )
                if src_contig.device.type == "cuda" and torch.cuda.is_available():
                    torch.cuda.current_stream(src_contig.device).synchronize()
                else:
                    pass
                flat_dst[start:end].copy_(chunk_view)
        else:
            flat_dst.copy_(flat_src)
    except RuntimeError:
        flat_dst.copy_(flat_src)
        used_chunked_pinned = False

    if used_chunked_pinned:
        return host, "chunked_pinned"
    else:
        return host, "pageable"


class ScratchBufferPool:
    """Reusable per-device fp32 scratch tensors for low-fragmentation merges."""

    def __init__(self) -> None:
        self.buffers: dict[tuple[str, torch.device], torch.Tensor] = {}

    def get_fp32(
        self,
        key: str,
        shape: tuple[int, ...],
        device: torch.device,
    ) -> torch.Tensor:
        needed = 1
        for dim in shape:
            needed *= dim
        cache_key = (key, device)
        existing = self.buffers.get(cache_key)
        if existing is None or existing.numel() < needed:
            existing = torch.empty((needed,), dtype=torch.float32, device=device)
            self.buffers[cache_key] = existing
        else:
            pass
        return existing[:needed].view(shape)

    def clear(self) -> None:
        self.buffers.clear()


class MergedLinearWeights:
    """Host-resident pristine base weight bank with atomic in-place commit/rollback."""

    def __init__(
        self,
        parameters: Mapping[str, nn.Parameter],
        *,
        parking_policy: ParkingPolicy | None = None,
        free_memory_fn: Callable[[torch.device], int | None] | None = None,
        prefer_pinned: bool = True,
        pin_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ) -> None:
        self.parameters: dict[str, nn.Parameter] = dict(parameters)
        self.parking_policy = (
            parking_policy if parking_policy is not None else ParkingPolicy()
        )
        self.free_memory_fn = (
            free_memory_fn if free_memory_fn is not None else query_free_device_bytes
        )
        self.prefer_pinned = prefer_pinned
        self.pin_fn = pin_fn
        self.scratch = ScratchBufferPool()
        self.base_bank: dict[str, torch.Tensor] = {}
        self.base_storage_modes: dict[str, str] = {}
        self.active_params: set[str] = set()
        self.prepared_weights: dict[str, torch.Tensor] = {}
        self.prepared_park_location: str | None = None
        self.pending_active_params: set[str] | None = None

    def ensure_base_snapshot(self, param_name: str) -> torch.Tensor:
        """Capture pristine host copy once per parameter lifecycle."""
        cached = self.base_bank.get(param_name)
        if cached is not None:
            return cached
        else:
            pass
        param = self.parameters.get(param_name)
        if param is None:
            raise KeyError(f"unknown managed parameter {param_name!r}")
        else:
            pass
        host_copy, mode = copy_to_host_staged(
            param.data,
            prefer_pinned=self.prefer_pinned,
            pin_fn=self.pin_fn,
        )
        self.base_bank[param_name] = host_copy
        self.base_storage_modes[param_name] = mode
        return host_copy

    def prepare(
        self,
        deltas: Mapping[str, torch.Tensor],
        *,
        restore_omitted: bool = True,
    ) -> str:
        """Stage merged weights (base + delta) without mutating live parameters.

        Returns the parking location used for staged tensors ("cuda" or "cpu").
        """
        if self.prepared_weights:
            raise RuntimeError(
                "MergedLinearWeights already has staged weights; commit or rollback first"
            )
        else:
            pass

        unknown = sorted(set(deltas.keys()) - set(self.parameters.keys()))
        if unknown:
            raise KeyError(f"unknown target parameter(s): {', '.join(unknown)}")
        else:
            pass

        restore_names: set[str] = (
            self.active_params - set(deltas.keys()) if restore_omitted else set()
        )
        total_update_bytes = 0
        device: torch.device | None = None
        for name in set(deltas.keys()) | restore_names:
            param = self.parameters[name]
            total_update_bytes += param.numel() * param.element_size()
            if device is None:
                device = param.device
            else:
                pass

        if device is None:
            self.prepared_weights = {}
            self.prepared_park_location = "cpu"
            self.pending_active_params = set(self.active_params)
            return "cpu"
        else:
            pass

        free_bytes = self.free_memory_fn(device)
        park_on_device = (
            device.type == "cuda"
            and self.parking_policy.should_park_on_device(
                total_update_bytes, free_bytes
            )
        )
        park_location = "cuda" if park_on_device else "cpu"

        staged: dict[str, torch.Tensor] = {}
        try:
            with torch.no_grad():
                for name, delta in deltas.items():
                    param = self.parameters[name]
                    if tuple(delta.shape) != tuple(param.shape):
                        raise ValueError(
                            f"delta shape {tuple(delta.shape)} does not match "
                            f"parameter {name!r} shape {tuple(param.shape)}"
                        )
                    else:
                        pass
                    base_host = self.ensure_base_snapshot(name)
                    compute_device = param.device
                    base_fp32 = self.scratch.get_fp32(
                        "base", tuple(param.shape), compute_device
                    )
                    base_fp32.copy_(
                        base_host,
                        non_blocking=(
                            compute_device.type == "cuda" and base_host.is_pinned()
                        ),
                    )
                    delta_fp32 = self.scratch.get_fp32(
                        "delta", tuple(param.shape), compute_device
                    )
                    delta_fp32.copy_(
                        delta.detach(),
                        non_blocking=(
                            compute_device.type == "cuda" and delta.is_pinned()
                        ),
                    )
                    base_fp32.add_(delta_fp32)
                    merged_on_device = base_fp32.to(dtype=param.dtype)
                    if park_on_device:
                        staged[name] = merged_on_device.clone()
                    else:
                        host_staged, _mode = copy_to_host_staged(
                            merged_on_device,
                            prefer_pinned=self.prefer_pinned,
                            pin_fn=self.pin_fn,
                        )
                        staged[name] = host_staged

                for name in sorted(restore_names):
                    base_host = self.ensure_base_snapshot(name)
                    if park_on_device:
                        param = self.parameters[name]
                        on_dev = torch.empty_like(param.data)
                        on_dev.copy_(
                            base_host,
                            non_blocking=(
                                param.device.type == "cuda" and base_host.is_pinned()
                            ),
                        )
                        staged[name] = on_dev
                    else:
                        staged[name] = base_host
        except Exception:
            staged.clear()
            self.prepared_weights.clear()
            self.prepared_park_location = None
            self.pending_active_params = None
            raise

        self.prepared_weights = staged
        self.prepared_park_location = park_location
        if restore_omitted:
            self.pending_active_params = set(deltas.keys())
        else:
            self.pending_active_params = self.active_params | set(deltas.keys())
        return park_location

    def commit(
        self,
        *,
        copy_fn: Callable[[nn.Parameter, torch.Tensor], None] | None = None,
    ) -> int:
        """Atomically copy prepared weights into live parameters in-place."""
        if not self.prepared_weights and self.pending_active_params is None:
            return 0
        else:
            pass

        backups: dict[str, torch.Tensor] = {}
        committed_names: list[str] = []
        try:
            with torch.no_grad():
                for name, prepared in self.prepared_weights.items():
                    param = self.parameters[name]
                    backups[name] = param.data.detach().to(device="cpu").clone()
                    if copy_fn is not None:
                        copy_fn(param, prepared)
                    else:
                        param.data.copy_(
                            prepared,
                            non_blocking=(
                                param.device.type == "cuda"
                                and prepared.device.type == "cpu"
                                and prepared.is_pinned()
                            ),
                        )
                    committed_names.append(name)
        except Exception:
            with torch.no_grad():
                for name in reversed(committed_names):
                    param = self.parameters[name]
                    param.data.copy_(backups[name])
            self.prepared_weights.clear()
            self.prepared_park_location = None
            self.pending_active_params = None
            raise

        count = len(committed_names)
        if self.pending_active_params is not None:
            self.active_params = set(self.pending_active_params)
        else:
            pass
        self.prepared_weights.clear()
        self.prepared_park_location = None
        self.pending_active_params = None
        return count

    def rollback(self) -> None:
        """Discard any staged prepared weights without modifying live parameters."""
        self.prepared_weights.clear()
        self.prepared_park_location = None
        self.pending_active_params = None

    def restore_base(self, param_names: Iterable[str] | None = None) -> int:
        """Restore live parameters in-place from the host-resident pristine bank."""
        if self.prepared_weights:
            raise RuntimeError(
                "cannot restore_base while staged weights are pending; rollback first"
            )
        else:
            pass
        targets = (
            sorted(self.active_params)
            if param_names is None
            else sorted(set(param_names))
        )
        restored = 0
        with torch.no_grad():
            for name in targets:
                base_host = self.base_bank.get(name)
                if base_host is None:
                    continue
                else:
                    pass
                param = self.parameters[name]
                param.data.copy_(
                    base_host,
                    non_blocking=(
                        param.device.type == "cuda" and base_host.is_pinned()
                    ),
                )
                self.active_params.discard(name)
                restored += 1
        return restored

    def memory_stats(self) -> dict[str, int | str | None]:
        """Return diagnostic memory statistics for the host bank and staging state."""
        host_base_bytes = sum(
            tensor.numel() * tensor.element_size() for tensor in self.base_bank.values()
        )
        staged_bytes = sum(
            tensor.numel() * tensor.element_size()
            for tensor in self.prepared_weights.values()
        )
        return {
            "host_base_bytes": host_base_bytes,
            "host_base_tensors": len(self.base_bank),
            "active_params": len(self.active_params),
            "staged_bytes": staged_bytes,
            "staged_park_location": self.prepared_park_location,
        }
