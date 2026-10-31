# SPDX-License-Identifier: Apache-2.0
"""Request ownership and component residency for AuK serial offload."""

from __future__ import annotations

import threading
from functools import lru_cache
from queue import Queue
from typing import Literal, Protocol

import torch

from sglang_omni.models.auk.vae import BigVGANFlowVAE
from sglang_omni.scheduling.message import IncomingMessage
from sglang_omni.scheduling.stage_residency import StageResidency
from sglang_omni.scheduling.weight_cache import StagingRing

Component = Literal["conditioning", "dit", "vae"]
StageName = Literal["conditioning", "auk_engine", "decode"]
Phase = Literal["idle", "conditioning", "auk_engine", "decode", "result"]


class CancellationCheck(Protocol):
    def __call__(self, request_id: str) -> bool: ...


class AuKOffloadCoordinator:
    """Keep one owner from conditioning admission through terminal delivery.

    The phase names the current or next stage; computation always has an owner.
    """

    def __init__(self, device: torch.device) -> None:
        self.device: torch.device = device
        self.condition: threading.Condition = threading.Condition()
        self.owner_request_id: str | None = None
        self.phase: Phase = "idle"
        self.is_computing: bool = False
        self.is_closing: bool = False
        self.failure: Exception | None = None
        self.abort_event: threading.Event = threading.Event()
        self.residencies: dict[Component, StageResidency] = {}
        self.resident_component: Component | None = None
        self.inboxes: list[Queue[IncomingMessage]] = []
        self.staging_ring: StagingRing | None = None

    def require_healthy(self) -> None:
        if self.failure is not None:
            raise RuntimeError(
                "AuK weight transition failed; restart the worker"
            ) from self.failure
        else:
            pass

    def fail(self, failure: Exception) -> None:
        with self.condition:
            self.failure = failure
            self.abort_event.set()
            self.condition.notify_all()
            for inbox in self.inboxes:
                inbox.put(IncomingMessage(request_id="", type="wakeup"))

    def begin(
        self, request_id: str, stage: StageName, is_aborted: CancellationCheck
    ) -> None:
        with self.condition:
            while stage == "conditioning" and self.owner_request_id is not None:
                self.require_healthy()
                if self.is_closing or is_aborted(request_id):
                    raise InterruptedError(f"AuK request {request_id} cancelled")
                else:
                    self.condition.wait()
            self.require_healthy()
            if self.is_closing or is_aborted(request_id):
                raise InterruptedError(f"AuK request {request_id} cancelled")
            elif stage == "conditioning":
                self.owner_request_id = request_id
                self.phase = stage
                self.abort_event.clear()
            elif self.owner_request_id != request_id or self.phase != stage:
                raise RuntimeError(
                    f"AuK request {request_id} does not own stage {stage}"
                )
            else:
                pass
            assert not self.is_computing
            self.is_computing = True

    def finish(self, request_id: str, stage: StageName, *, succeeded: bool) -> None:
        with self.condition:
            assert self.owner_request_id == request_id and self.is_computing
            self.is_computing = False
            if not succeeded or self.abort_event.is_set() or self.is_closing:
                self.release_owner_locked()
            elif stage == "conditioning":
                self.phase = "auk_engine"
            elif stage == "auk_engine":
                self.phase = "decode"
            else:
                self.phase = "result"
            self.condition.notify_all()

    def cancel(self, request_id: str) -> None:
        with self.condition:
            if self.owner_request_id == request_id:
                self.abort_event.set()
                if not self.is_computing:
                    self.release_owner_locked()
                else:
                    pass
            else:
                pass
            self.condition.notify_all()

    def claim_result(self, request_id: str) -> bool:
        with self.condition:
            return (
                self.owner_request_id == request_id
                and not self.abort_event.is_set()
                and not self.is_closing
                and self.failure is None
            )

    def release_result(
        self, request_id: str, *, terminal: bool, delivered: bool
    ) -> None:
        if not delivered:
            self.cancel(request_id)
        elif terminal:
            with self.condition:
                if self.owner_request_id == request_id:
                    assert self.phase == "result"
                    self.release_owner_locked()
                    self.condition.notify_all()
                else:
                    pass
        else:
            pass

    def release_owner_locked(self) -> None:
        """Release ownership only after the compute thread has synchronized."""
        assert not self.is_computing
        self.owner_request_id = None
        self.phase = "idle"

    def stop(self) -> None:
        with self.condition:
            self.is_closing = True
            self.abort_event.set()
            self.condition.notify_all()

    def switch(self, component: Component | None) -> None:
        """Only the owning compute thread or serialized startup may switch weights."""
        self.require_healthy()
        if self.resident_component == component:
            return
        else:
            pass
        try:
            if self.resident_component is not None:
                self.residencies[self.resident_component].sleep()
                self.resident_component = None
            else:
                pass
            if component is not None:
                self.residencies[component].wake()
                self.resident_component = component
            else:
                pass
        except Exception as failure:
            self.fail(failure)
            raise

    def register_conditioning(
        self,
        encoder: torch.nn.Module,
        checkpoint_contents: dict[str, str],
        *,
        source: Literal["mmap", "ram"],
        cache_dir: str | None,
    ) -> None:
        """Initialize the shared restoration resources with the first component."""
        if self.residencies:
            raise RuntimeError("AuK offload is already initialized in this process")
        else:
            pass
        self.staging_ring = StagingRing(self.device)
        self.residencies["conditioning"] = StageResidency(
            {"encoder": encoder},
            self.device,
            label="auk_conditioning",
            tags=("auk_conditioning",),
            source=source,
            cache_dir=cache_dir,
            checkpoint_contents=checkpoint_contents,
            staging_ring=self.staging_ring,
        )
        self.resident_component = "conditioning"

    def require_vae(self) -> BigVGANFlowVAE:
        """Return the VAE shared by reference encoding and decoding."""
        if "vae" not in self.residencies:
            raise RuntimeError("Construct AuK conditioning before the DiT and decoder")
        else:
            vae = self.residencies["vae"].modules["vae"]
            assert isinstance(vae, BigVGANFlowVAE)
            return vae

    def register(
        self,
        component: Literal["dit", "vae"],
        modules: dict[str, torch.nn.Module],
        checkpoint_contents: dict[str, str],
        *,
        folding_backend: str = "none",
    ) -> None:
        assert component not in self.residencies and self.resident_component is None
        conditioning_weights = self.residencies["conditioning"].weights
        self.residencies[component] = StageResidency(
            modules,
            self.device,
            label=f"auk_{component}",
            tags=(f"auk_{component}",),
            source=conditioning_weights.source,
            cache_dir=(
                str(conditioning_weights.cache_path.parent)
                if conditioning_weights.cache_path is not None
                else None
            ),
            checkpoint_contents=checkpoint_contents,
            folding_backend=folding_backend,
            staging_ring=self.staging_ring,
        )
        self.resident_component = component


@lru_cache(maxsize=None)
def get_coordinator(checkpoint: str, device: str) -> AuKOffloadCoordinator:
    """Factories in one process share ownership for the same checkpoint/device."""
    return AuKOffloadCoordinator(torch.device(device))
