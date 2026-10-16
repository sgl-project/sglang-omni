# SPDX-License-Identifier: Apache-2.0
"""Serial GPU residency for the MiniMax Music 3 AR and DIT/DAV stages.

``--stage-offload-components ar,dit`` colocates both stages in one process
and lets only one of them hold GPU weights at a time: AR generates a whole
request, hands off to DIT/DAV, and stays off the GPU until DIT/DAV is finished
with that request.

The handoff has one request owner. A duplicated, out-of-order, or late terminal
event cannot wake AR while DIT/DAV is decoding another request.

A handoff that never ends stops admission for good. That is reported rather
than force-corrected:
waking AR while DIT/DAV still holds the GPU is how this configuration runs out
of memory, so a stuck server is preferable to a crashed one, and the log names
the requests still outstanding.
"""

from __future__ import annotations

import logging
import threading
import time
import traceback
from collections.abc import Callable

import torch

logger = logging.getLogger(__name__)

STALL_REPORT_SECONDS = 900.0
BYTES_PER_GIB = 1024.0**3


def clear_exception_frames(error: BaseException) -> None:
    """Drop tensors in unwound tracebacks, including chained exceptions."""
    logger.error("".join(traceback.format_exception(error)))
    pending_errors = [error]
    seen_errors: set[int] = set()
    while pending_errors:
        current_error = pending_errors.pop()
        if id(current_error) in seen_errors:
            continue
        else:
            seen_errors.add(id(current_error))
        traceback.clear_frames(current_error.__traceback__)
        for chained_error in (current_error.__cause__, current_error.__context__):
            if chained_error is not None:
                pending_errors.append(chained_error)
            else:
                pass


def owned_tensors(
    modules: dict[str, torch.nn.Module],
) -> list[tuple[tuple[str, str], torch.Tensor]]:
    """Every tensor owned by a stage's modules, tied tensors listed once.

    ``named_parameters``/``named_buffers`` de-duplicate by default, so a tied
    weight appears under one name per module. De-duplicate once more across the
    group so modules that share a tensor still get one canonical host copy.
    Residency mutates the tensor object in place, so every name that shares it
    follows along and the tying survives the round trip.
    """
    owned: list[tuple[tuple[str, str], torch.Tensor]] = []
    seen: set[int] = set()
    for module_name, module in modules.items():
        for tensor_name, tensor in (
            *module.named_parameters(),
            *module.named_buffers(),
        ):
            if id(tensor) in seen:
                continue
            else:
                pass
            seen.add(id(tensor))
            owned.append(((module_name, tensor_name), tensor))
    return owned


def gpu_memory_note(device: torch.device) -> str:
    """Device occupancy, for attributing what a handoff actually frees.

    ``torch_*`` covers only this process's caching allocator, while ``free``
    is the whole device, so a gap between them is memory held outside torch's
    allocator. Reporting both is what distinguishes weights coming off the GPU
    from the pools that stay put -- notably the SGLang KV cache, which
    ``mem_fraction_static`` reserves for the life of the process and which no
    weight offload releases.
    """
    if device.type != "cuda":
        return "cpu"
    else:
        pass
    free, total = torch.cuda.mem_get_info(device)
    return (
        f"free={free / BYTES_PER_GIB:.2f}/{total / BYTES_PER_GIB:.2f}GiB "
        f"torch_allocated={torch.cuda.memory_allocated(device) / BYTES_PER_GIB:.2f}GiB "
        f"torch_reserved={torch.cuda.memory_reserved(device) / BYTES_PER_GIB:.2f}GiB"
    )


class StageResidency:
    """GPU residency for a group of stage modules whose weights never change.

    Inference never writes these weights, so the host copy is canonical: it is
    filled once and every wake refills the GPU from it. Sleeping then only
    drops the GPU replica -- no device-to-host copy and no fresh host
    allocation per request, which is what ``module.to()`` in both directions
    was paying for on every single handoff.

    Reusing one host copy also stops the sleep path handing the caching
    allocator a differently sized block each round trip. That is what lets a
    long-running server fragment its way into an OOM after tens of requests
    rather than failing on the first one.
    """

    def __init__(
        self,
        modules: dict[str, torch.nn.Module],
        device: torch.device,
        *,
        resident: bool = True,
        label: str = "stage",
    ):
        if not modules:
            raise ValueError("StageResidency requires at least one module")
        else:
            pass
        self.modules = dict(modules)
        self.device = torch.device(device)
        self.is_resident = bool(resident)
        self.label = label
        self.host_weights: dict[tuple[str, str], torch.Tensor] = {}
        if not self.is_resident:
            # Built on the host: retain its existing storage through a distinct
            # tensor handle. Keeping the Parameter itself here would alias the
            # object whose ``.data`` wake replaces, moving the supposed host
            # copy onto the GPU along with the module.
            self.host_weights = {
                name: tensor.detach() for name, tensor in owned_tensors(self.modules)
            }
        else:
            pass
        weight_bytes = sum(
            tensor.numel() * tensor.element_size()
            for _, tensor in owned_tensors(self.modules)
        )
        logger.info(
            f"MiniMax Music 3 residency {label}: {weight_bytes / BYTES_PER_GIB:.2f}GiB of "
            f"weights, starts {'resident' if self.is_resident else 'offloaded'} "
            f"({gpu_memory_note(self.device)})"
        )

    @property
    def resident(self) -> bool:
        return self.is_resident

    def sleep(self) -> None:
        """Drop the GPU replica; a cheap no-op once already asleep."""
        if not self.is_resident:
            return
        else:
            pass
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        else:
            pass
        owned = owned_tensors(self.modules)
        if not self.host_weights:
            self.host_weights = {
                name: tensor.detach().to("cpu", copy=True) for name, tensor in owned
            }
        else:
            pass
        for name, tensor in owned:
            tensor.data = self.host_weights[name]
        self.is_resident = False
        logger.info(
            f"MiniMax Music 3 residency {self.label} -> host "
            f"({gpu_memory_note(self.device)})"
        )

    def wake(self) -> None:
        """Refill the GPU replica from the host copy; no-op once resident."""
        if self.is_resident:
            return
        else:
            pass
        for name, tensor in owned_tensors(self.modules):
            tensor.data = self.host_weights[name].to(
                self.device, copy=True, non_blocking=True
            )
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        else:
            pass
        self.is_resident = True
        logger.info(
            f"MiniMax Music 3 residency {self.label} -> gpu "
            f"({gpu_memory_note(self.device)})"
        )


class SerialOffloadCoordinator:
    """Process-wide ownership and optional CUDA residency coordinator."""

    def __init__(self) -> None:
        self.condition = threading.Condition()
        self.is_enabled = False
        self.ar_residency: StageResidency | None = None
        self.owner_request_id: str | None = None
        self.phase = "disabled"
        self.failure: BaseException | None = None
        self.paused_at_seconds: float | None = None
        self.has_reported_stall = False
        self.has_retired_acoustic = False
        self.pending_acoustic_release: Callable[[], None] | None = None

    @property
    def enabled(self) -> bool:
        return self.is_enabled

    def enable(self) -> None:
        """Enable request ownership for a backend without CUDA residency."""
        with self.condition:
            if self.is_enabled:
                return
            else:
                pass
            self.is_enabled = True
            self.phase = "idle"

    def register_ar(self, model: torch.nn.Module, device: torch.device) -> None:
        residency = StageResidency({"ar": model}, device, label="ar")
        with self.condition:
            if self.is_enabled:
                raise RuntimeError("MiniMax Music 3 serial offload is already enabled")
            else:
                pass
            self.failure = None
            self.owner_request_id = None
            self.ar_residency = residency
            self.is_enabled = True
            self.phase = "idle"
        logger.info(
            f"MiniMax Music 3 serial offload enabled device={device}; AR "
            "starts GPU-resident, DIT/DAV starts offloaded"
        )

    def try_acquire_ar(self, request_id: str) -> bool:
        """Atomically claim the complete AR-to-acoustic request lifecycle."""
        if not self.is_enabled:
            return True
        else:
            pass
        with self.condition:
            self.raise_if_failed_locked()
            if self.owner_request_id == request_id and self.phase == "ar":
                return True
            else:
                pass
            if self.owner_request_id is None and self.phase == "idle":
                self.owner_request_id = request_id
                self.phase = "ar"
                return True
            else:
                pass
            self.report_stall_locked()
            return False

    def acquire_ar(
        self,
        request_id: str,
        *,
        should_abort: Callable[[], bool] | None = None,
    ) -> None:
        """Wait for and claim AR ownership without holding the lock during work."""
        if not self.is_enabled:
            return
        else:
            pass
        with self.condition:
            while not self.try_acquire_ar(request_id):
                if should_abort is not None and should_abort():
                    raise InterruptedError(
                        "MiniMax Music 3 serial offload admission aborted"
                    )
                else:
                    pass
                self.condition.wait(timeout=0.1)

    def begin_dit_handoff(self, request_id: str) -> None:
        """Hand the GPU to DIT/DAV for *request_id* and take AR off it."""
        if not self.is_enabled:
            return
        else:
            pass
        with self.condition:
            self.raise_if_failed_locked()
            if self.owner_request_id != request_id:
                raise RuntimeError(
                    "MiniMax Music 3 serial offload handoff does not own request "
                    f"{request_id!r} (owner={self.owner_request_id!r})"
                )
            else:
                pass
            if self.phase == "acoustic":
                return
            else:
                pass
            if self.phase != "ar":
                raise RuntimeError(
                    f"MiniMax Music 3 cannot hand off from phase {self.phase!r}"
                )
            else:
                pass
            self.phase = "ar_unloading"

        try:
            if self.ar_residency is not None:
                self.ar_residency.sleep()
            else:
                pass
        except BaseException as exc:
            self.record_failure(exc)
            raise

        with self.condition:
            self.raise_if_failed_locked()
            self.phase = "acoustic"
            self.paused_at_seconds = time.monotonic()
            self.has_reported_stall = False
            has_retired_acoustic = self.has_retired_acoustic
            release_acoustic = self.pending_acoustic_release
            self.condition.notify_all()
        logger.info(
            f"MiniMax Music 3 serial offload: AR -> CPU (DIT/DAV's turn, "
            f"request={request_id})"
        )
        if has_retired_acoustic:
            self.end_dit_handoff(request_id, release_acoustic=release_acoustic)
        else:
            pass

    def require_acoustic(self, request_id: str) -> None:
        """Reject compute that is not owned by the active acoustic request."""
        if not self.is_enabled:
            return
        else:
            pass
        with self.condition:
            self.raise_if_failed_locked()
            if self.owner_request_id != request_id or self.phase != "acoustic":
                raise RuntimeError(
                    "MiniMax Music 3 acoustic request does not own residency: "
                    f"request={request_id!r} owner={self.owner_request_id!r} "
                    f"phase={self.phase!r}"
                )
            else:
                pass

    def end_dit_handoff(
        self,
        request_id: str,
        *,
        release_acoustic: Callable[[], None] | None = None,
    ) -> None:
        """Retire *request_id* and return residency to AR.

        Unknown and duplicate terminal events are no-ops. Acoustic release and
        AR wake run without the ownership mutex held.
        """
        if not self.is_enabled:
            return
        else:
            pass
        with self.condition:
            self.raise_if_failed_locked()
            if self.owner_request_id != request_id:
                return
            elif self.phase in {"ar", "ar_unloading"}:
                if not self.has_retired_acoustic:
                    self.has_retired_acoustic = True
                    self.pending_acoustic_release = release_acoustic
                else:
                    pass
                return
            elif self.phase != "acoustic":
                return
            else:
                pass
            self.phase = "acoustic_unloading"

        try:
            if release_acoustic is not None:
                release_acoustic()
            else:
                pass
            if self.ar_residency is not None:
                self.ar_residency.wake()
            else:
                pass
        except BaseException as exc:
            self.record_failure(exc)
            raise

        with self.condition:
            self.owner_request_id = None
            self.phase = "idle"
            self.paused_at_seconds = None
            self.has_reported_stall = False
            self.has_retired_acoustic = False
            self.pending_acoustic_release = None
            self.condition.notify_all()
        logger.info(
            f"MiniMax Music 3 serial offload: AR -> GPU (AR's turn, "
            f"request={request_id})"
        )

    def cancel_ar(self, request_id: str) -> None:
        """Release an AR owner after its compute has stopped, before handoff."""
        if not self.is_enabled:
            return
        else:
            pass
        with self.condition:
            if self.owner_request_id != request_id or self.phase != "ar":
                return
            else:
                pass
            self.owner_request_id = None
            self.phase = "idle"
            self.has_retired_acoustic = False
            self.pending_acoustic_release = None
            self.condition.notify_all()

    def record_failure(self, exc: BaseException) -> None:
        with self.condition:
            self.failure = exc
            self.phase = "failed"
            self.condition.notify_all()

    def fail_closed(self, exc: BaseException) -> None:
        """Prevent later admission after an unsafe backend cleanup failure."""
        if self.is_enabled:
            self.record_failure(exc)
        else:
            pass

    def raise_if_failed_locked(self) -> None:
        if self.failure is not None:
            raise RuntimeError(
                "MiniMax Music 3 serial offload is unavailable after a residency "
                "transition failed"
            ) from self.failure
        else:
            pass

    def report_stall_locked(self) -> None:
        """Name the owner once AR has been parked too long."""
        if self.paused_at_seconds is None or self.has_reported_stall:
            return
        else:
            pass
        elapsed = time.monotonic() - self.paused_at_seconds
        if elapsed < STALL_REPORT_SECONDS:
            return
        else:
            pass
        self.has_reported_stall = True
        logger.error(
            f"MiniMax Music 3 serial offload: AR has been off the GPU for "
            f"{elapsed:.0f}s and is still waiting on "
            f"{self.owner_request_id!r}; AR admits nothing until DIT/DAV "
            "retires them, so this server needs a restart if the requests "
            "are gone"
        )


COORDINATOR = SerialOffloadCoordinator()


def get_coordinator() -> SerialOffloadCoordinator:
    return COORDINATOR


def reset_coordinator() -> None:
    """Reset process-local state after a failed build or during test teardown."""
    global COORDINATOR
    COORDINATOR = SerialOffloadCoordinator()


__all__ = [
    "STALL_REPORT_SECONDS",
    "SerialOffloadCoordinator",
    "StageResidency",
    "get_coordinator",
    "reset_coordinator",
]
