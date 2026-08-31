# SPDX-License-Identifier: Apache-2.0
"""Transactional pipeline ownership of one MPS lease per eligible GPU."""

from __future__ import annotations

import asyncio
import getpass
import logging
import os
import secrets
import shutil
import sys
import tempfile
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from sglang_omni.mps.decision import (
    MPS_MODES,
    MpsDecisionError,
    MpsProcessFact,
    collect_mps_facts,
)
from sglang_omni.mps.devices import MpsPhysicalDevice
from sglang_omni.mps.manager import (
    MPS_CLIENT_TOKEN_ENV,
    MpsClientRef,
    MpsControlClient,
    MpsDirtyStateError,
    MpsError,
    MpsLease,
    MpsManager,
)
from sglang_omni.mps.state import MpsGpuPaths

logger = logging.getLogger(__name__)


class _MpsDeviceInfo(Protocol):
    def inspect(self, gpu_ids: Iterable[int]) -> dict[int, MpsPhysicalDevice]: ...


@dataclass(frozen=True)
class _PhysicalMpsPlan:
    logical_gpu_ids: tuple[int, ...]
    client_process_names: tuple[str, ...]


def _default_state_root() -> Path:
    # Keep this short: the control socket must fit Linux's AF_UNIX path budget.
    override = os.environ.get("SGLANG_OMNI_MPS_STATE_ROOT")
    if override:
        return Path(override)
    return Path(tempfile.gettempdir()) / f"sglang-omni-mps-{getpass.getuser()}"


def _resolve_physical_plans(
    *,
    mode: str,
    process_facts: tuple[MpsProcessFact, ...],
    device_info: _MpsDeviceInfo,
) -> dict[str, _PhysicalMpsPlan]:
    """Resolve physical identity before applying any MPS-specific gate."""

    potential_clients = [
        fact
        for fact in process_facts
        if fact.placement_gpu_ids and not fact.contains_tp
    ]
    if not potential_clients:
        if mode == "on":
            raise MpsDecisionError(
                "mps=on but no process is eligible for MPS (TP and CPU-only "
                "processes cannot attach)"
            )
        return {}

    placement_ordinals_by_process: dict[str, set[int]] = {
        fact.process_name: set(fact.placement_gpu_ids) for fact in potential_clients
    }

    gpu_ids = tuple(
        sorted(
            {
                gpu_id
                for process_gpu_ids in placement_ordinals_by_process.values()
                for gpu_id in process_gpu_ids
            }
        )
    )
    try:
        devices = device_info.inspect(gpu_ids)
    except Exception as exc:
        devices = {}
        resolution_errors = [f"CUDA device inspection failed: {exc}"]
    else:
        resolution_errors = []
    for gpu_id in gpu_ids:
        device = devices.get(gpu_id)
        if device is None:
            resolution_errors.append(
                f"CUDA ordinal {gpu_id}: device inspection returned no result"
            )
        elif device.gpu_uuid is None:
            resolution_errors.append(
                f"CUDA ordinal {gpu_id}: "
                f"{device.unsupported_reason or 'physical GPU UUID is unavailable'}"
            )

    uuid_for = {
        gpu_id: device.gpu_uuid
        for gpu_id, device in devices.items()
        if device.gpu_uuid is not None
    }
    for process_name, process_gpu_ids in placement_ordinals_by_process.items():
        resolved = {
            gpu_id: uuid_for[gpu_id]
            for gpu_id in sorted(process_gpu_ids)
            if gpu_id in uuid_for
        }
        physical_uuids = set(resolved.values())
        if len(physical_uuids) > 1:
            unresolved = sorted(process_gpu_ids - resolved.keys())
            unresolved_detail = (
                f"; unresolved CUDA ordinals: {unresolved}" if unresolved else ""
            )
            raise MpsError(
                f"process {process_name!r} resolves CUDA ordinals to multiple "
                f"physical GPUs: {resolved}{unresolved_detail}; native MPS "
                "requires one physical "
                "GPU per process. Use mps=off for this placement."
            )

    if resolution_errors:
        detail = "; ".join(resolution_errors)
        if mode == "on":
            raise MpsError(
                "mps=on could not resolve the physical GPU mapping: " + detail
            )
        logger.warning(
            "MPS auto: physical GPU mapping is incomplete (%s); running " "without MPS",
            detail,
        )
        return {}

    clients_by_uuid: dict[str, list[str]] = {}
    logical_ids_by_uuid: dict[str, set[int]] = {}
    blocked: dict[str, list[str]] = {}

    def block(gpu_uuids: set[str], reason: str) -> None:
        for gpu_uuid in gpu_uuids:
            blocked.setdefault(gpu_uuid, []).append(reason)

    for fact in potential_clients:
        placement_uuids = {uuid_for[gpu_id] for gpu_id in fact.placement_gpu_ids}
        (placement_uuid,) = placement_uuids
        names = clients_by_uuid.setdefault(placement_uuid, [])
        if fact.process_name not in names:
            names.append(fact.process_name)
        logical_ids_by_uuid.setdefault(placement_uuid, set()).update(
            fact.placement_gpu_ids
        )

        nonzero_explicit = set(fact.explicit_cuda_gpu_ids) - {0}
        if nonzero_explicit:
            block(
                {placement_uuid},
                f"process {fact.process_name!r} contains explicit CUDA "
                f"ordinal(s) {sorted(nonzero_explicit)} that would be invalid "
                "after single-device MPS normalization; use cuda:0 for the "
                "worker-local device or use mps=off",
            )

    unsupported_by_uuid: dict[str, list[str]] = {}
    for gpu_id, device in devices.items():
        if device.unsupported_reason is not None:
            assert device.gpu_uuid is not None
            unsupported_by_uuid.setdefault(device.gpu_uuid, []).append(
                f"CUDA ordinal {gpu_id}: {device.unsupported_reason}"
            )
    unsupported_candidates = {
        gpu_uuid: reasons
        for gpu_uuid, reasons in unsupported_by_uuid.items()
        if gpu_uuid in clients_by_uuid
    }
    if mode == "on" and unsupported_candidates:
        detail = "; ".join(
            f"{gpu_uuid}: {'; '.join(reasons)}"
            for gpu_uuid, reasons in sorted(unsupported_candidates.items())
        )
        raise MpsError(f"mps=on but a physical GPU does not support MPS: {detail}")
    for gpu_uuid, reasons in unsupported_candidates.items():
        block({gpu_uuid}, "; ".join(reasons))

    physical_plans: dict[str, _PhysicalMpsPlan] = {}
    for gpu_uuid, process_names in sorted(clients_by_uuid.items()):
        reasons = blocked.get(gpu_uuid, ())
        logical_gpu_ids = tuple(sorted(logical_ids_by_uuid[gpu_uuid]))
        if reasons:
            logger.warning(
                "MPS (%s): skipping physical GPU %s (logical GPUs %s): %s",
                mode,
                gpu_uuid,
                list(logical_gpu_ids),
                "; ".join(dict.fromkeys(reasons)),
            )
            continue
        if mode == "auto" and len(process_names) < 2:
            logger.info(
                "MPS auto: physical GPU %s (logical GPUs %s) has one client; "
                "running without MPS",
                gpu_uuid,
                list(logical_gpu_ids),
            )
            continue
        physical_plans[gpu_uuid] = _PhysicalMpsPlan(
            logical_gpu_ids=logical_gpu_ids,
            client_process_names=tuple(process_names),
        )

    if mode == "on" and not physical_plans:
        reasons = sorted(
            {reason for gpu_reasons in blocked.values() for reason in gpu_reasons}
        )
        detail = f": {'; '.join(reasons)}" if reasons else ""
        raise MpsDecisionError(
            "mps=on but no physical GPU is eligible for MPS" + detail
        )
    return physical_plans


_UNSUPPORTED_PROCESS_ENV = (
    "CUDA_VISIBLE_DEVICES",
    "CUDA_DEVICE_ORDER",
    "CUDA_MPS_PIPE_DIRECTORY",
    "SGLANG_OMNI_WEIGHT_SHARE",
)


def _reject_process_env_overrides(process_specs) -> None:
    conflicts: list[str] = []
    for process_spec in process_specs:
        for stage_spec in process_spec.stage_specs:
            env_defaults = getattr(stage_spec, "env_defaults", {})
            for name in _UNSUPPORTED_PROCESS_ENV:
                if name in env_defaults:
                    conflicts.append(
                        f"process {process_spec.process_name!r}, stage "
                        f"{stage_spec.stage_name!r} sets "
                        f"{name}={env_defaults[name]!r}"
                    )
    if conflicts:
        raise MpsError(
            "native MPS does not support per-worker CUDA visibility, device "
            "ordering, external MPS, or CUDA IPC weight-sharing overrides: "
            f"{'; '.join(conflicts)}. Configure CUDA visibility and device "
            "order in the parent environment, remove external MPS or weight "
            "sharing settings, or use mps=off."
        )


class MpsPipelineRuntime:
    def __init__(
        self,
        managers: dict[str, MpsManager],
        plans: dict[str, _PhysicalMpsPlan],
        mode: str = "auto",
    ):
        self.managers = managers
        self._plans = plans
        self._mode = mode
        self._leases: dict[str, MpsLease] = {}
        self._operation_lock = asyncio.Lock()
        self._client_uuid: dict[str, str] = {
            name: gpu_uuid
            for gpu_uuid, plan in plans.items()
            for name in plan.client_process_names
        }
        self._client_tokens = {
            process_name: secrets.token_hex(16) for process_name in self._client_uuid
        }

    @property
    def has_leases(self) -> bool:
        return bool(self._leases)

    @classmethod
    def create(
        cls,
        *,
        mode: str,
        process_specs,
        device_info: _MpsDeviceInfo,
        client: MpsControlClient,
        state_root: Path | None = None,
    ) -> MpsPipelineRuntime | None:
        if mode not in MPS_MODES:
            raise MpsDecisionError(f"invalid mps mode {mode!r}; expected {MPS_MODES}")
        if mode == "off":
            return None
        process_specs = list(process_specs)
        _reject_process_env_overrides(process_specs)
        process_facts = collect_mps_facts(process_specs)
        physical_plans = _resolve_physical_plans(
            mode=mode,
            process_facts=process_facts,
            device_info=device_info,
        )

        if not physical_plans:
            return None

        root = state_root if state_root is not None else _default_state_root()
        managers = {
            gpu_uuid: MpsManager(
                paths=MpsGpuPaths(
                    state_root=root,
                    gpu_uuid=gpu_uuid,
                ),
                gpu_uuid=gpu_uuid,
                client=client,
            )
            for gpu_uuid in physical_plans
        }
        return cls(managers, physical_plans, mode=mode)

    async def start(self) -> None:
        async with self._operation_lock:
            try:
                await self._run_blocking(self._start)
            except asyncio.CancelledError as cancellation:
                try:
                    await self._run_blocking(
                        self._close,
                        frozenset(),
                    )
                except asyncio.CancelledError:
                    raise
                except BaseException as rollback_error:
                    raise cancellation from rollback_error
                raise

    def _start(self) -> None:
        """Acquire every GPU transactionally, rolling back in reverse order."""

        if self._leases:
            raise MpsError("MPS pipeline runtime is already acquired")
        acquired: list[str] = []
        try:
            for gpu_uuid, manager in self.managers.items():
                lease = manager.acquire(self._tokens_on(gpu_uuid))
                self._leases[gpu_uuid] = lease
                acquired.append(gpu_uuid)
                logger.info(
                    "MPS daemon ready on physical GPU %s (logical GPUs %s, pipe "
                    "dir %s)",
                    gpu_uuid,
                    list(self._plans[gpu_uuid].logical_gpu_ids),
                    manager.paths.pipe_dir,
                )
        except BaseException as startup_error:
            rollback_errors: list[tuple[str, MpsError]] = []
            for gpu_uuid in reversed(acquired):
                error = self._release_one(
                    gpu_uuid,
                    suppress_errors=True,
                    clients_could_have_attached=False,
                )
                if error is not None:
                    rollback_errors.append((gpu_uuid, error))
            if rollback_errors:
                details = "; ".join(
                    f"physical GPU {gpu_uuid}: {error}"
                    for gpu_uuid, error in rollback_errors
                )
                error_type = (
                    MpsDirtyStateError
                    if any(
                        isinstance(error, MpsDirtyStateError)
                        for _, error in rollback_errors
                    )
                    else MpsError
                )
                rollback_error = error_type(details)
                prior_cause = startup_error.__cause__ or startup_error.__context__
                if prior_cause is not None:
                    rollback_error.__cause__ = prior_cause
                raise startup_error from rollback_error
            raise
        logger.info(
            "MPS summary: mode=%s %s",
            self._mode,
            {
                gpu_uuid: {
                    "logical_gpus": list(self._plans[gpu_uuid].logical_gpu_ids),
                    "daemon_pid": self._leases[gpu_uuid].daemon_pid,
                    "clients": sorted(self._names_on(gpu_uuid)),
                }
                for gpu_uuid in self._leases
            },
        )

    def _names_on(self, gpu_uuid: str) -> list[str]:
        return [
            name for name, physical in self._client_uuid.items() if physical == gpu_uuid
        ]

    def _tokens_on(self, gpu_uuid: str) -> dict[str, str]:
        return {name: self._client_tokens[name] for name in self._names_on(gpu_uuid)}

    def env_for_process(self, process_name: str) -> dict[str, str]:
        gpu_uuid = self._client_uuid.get(process_name)
        if gpu_uuid is None:
            return {}
        env = self.managers[gpu_uuid].env_for_stage()
        # UUID visibility makes the physical device local ordinal zero.
        env["SGLANG_ONE_VISIBLE_DEVICE_PER_PROCESS"] = "true"
        env[MPS_CLIENT_TOKEN_ENV] = self._client_tokens[process_name]
        return env

    async def verify(self) -> None:
        async with self._operation_lock:
            await self._run_blocking(self._verify)

    def _verify(self) -> None:
        for gpu_uuid, lease in self._leases.items():
            self.managers[gpu_uuid].verify(lease)

    async def retire_process_clients(self, process_name: str) -> set[MpsClientRef]:
        """Retire one process's MPS clients before the runner signals it."""

        async with self._operation_lock:
            return await self._run_blocking(
                self._retire_process_clients,
                process_name,
            )

    def _retire_process_clients(self, process_name: str) -> set[MpsClientRef]:
        gpu_uuid = self._client_uuid.get(process_name)
        lease = self._leases.get(gpu_uuid) if gpu_uuid is not None else None
        if lease is None:
            return set()
        return self.managers[gpu_uuid].retire_clients_for(lease, process_name)

    async def probe_failures(self) -> dict[str, str]:
        async with self._operation_lock:
            return await self._run_blocking(self._probe_failures)

    def _probe_failures(self) -> dict[str, str]:
        failures: dict[str, str] = {}
        for gpu_uuid, lease in self._leases.items():
            reason = self.managers[gpu_uuid].probe(lease)
            if reason is not None:
                failures[gpu_uuid] = reason
        return failures

    async def close(
        self,
        *,
        process_start_attempts: set[str] | None = None,
    ) -> None:
        """Close leases, preserving ambiguity per physical GPU after spawn."""

        attempts = (
            None
            if process_start_attempts is None
            else frozenset(process_start_attempts)
        )
        async with self._operation_lock:
            await self._run_blocking(
                self._close,
                attempts,
            )

    def _close(self, process_start_attempts: frozenset[str] | None) -> None:
        errors: list[tuple[str, MpsError]] = []
        for gpu_uuid in reversed(list(self._leases)):
            clients_could_have_attached = (
                process_start_attempts is None
                or not process_start_attempts.isdisjoint(self._names_on(gpu_uuid))
            )
            error = self._release_one(
                gpu_uuid,
                suppress_errors=False,
                clients_could_have_attached=clients_could_have_attached,
            )
            if error is not None:
                errors.append((gpu_uuid, error))
        if errors:
            details = "; ".join(
                f"physical GPU {gpu_uuid}: {error}" for gpu_uuid, error in errors
            )
            error_type = (
                MpsDirtyStateError
                if any(isinstance(error, MpsDirtyStateError) for _, error in errors)
                else MpsError
            )
            raise error_type(details)

    @staticmethod
    async def _run_blocking(call: Callable[..., Any], *args: Any) -> Any:
        """Finish an ownership-changing call before propagating cancellation."""

        task = asyncio.create_task(asyncio.to_thread(call, *args))
        cancelled: asyncio.CancelledError | None = None
        while not task.done():
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError as exc:
                cancelled = cancelled or exc
                if task.done():
                    break
            except BaseException:
                break

        try:
            result = task.result()
        except BaseException as operation_error:
            if cancelled is not None and not isinstance(
                operation_error, asyncio.CancelledError
            ):
                raise cancelled from operation_error
            raise
        if cancelled is not None:
            raise cancelled
        return result

    def _release_one(
        self,
        gpu_uuid: str,
        *,
        suppress_errors: bool,
        clients_could_have_attached: bool = True,
    ) -> MpsError | None:
        lease = self._leases[gpu_uuid]
        error: MpsError | None = None
        try:
            self.managers[gpu_uuid].release(
                lease,
                clients_could_have_attached=clients_could_have_attached,
            )
        except MpsError as exc:
            error = exc
            if suppress_errors:
                logger.error("MPS rollback incomplete on GPU %s: %s", gpu_uuid, exc)
        finally:
            # A released owner fd means the token no longer carries cleanup
            # authority, even when later daemon cleanup failed.
            if lease.owner_fd < 0:
                self._leases.pop(gpu_uuid, None)
        return error


def create_for_pipeline(
    mode: str,
    process_specs,
) -> MpsPipelineRuntime | None:
    """Build the orchestrator with production device inspection and control I/O."""

    if mode == "off":
        return None
    process_specs = list(process_specs)
    _reject_process_env_overrides(process_specs)
    if "CUDA_MPS_PIPE_DIRECTORY" in os.environ:
        raise MpsError(
            "native MPS cannot join CUDA_MPS_PIPE_DIRECTORY="
            f"{os.environ['CUDA_MPS_PIPE_DIRECTORY']!r} from the parent "
            "environment; remove it or use mps=off."
        )

    weight_share = os.environ.get("SGLANG_OMNI_WEIGHT_SHARE", "").strip()
    if weight_share:
        raise MpsError(
            "native MPS cannot combine with parent "
            f"SGLANG_OMNI_WEIGHT_SHARE={weight_share!r}; remove it or use "
            "mps=off (examples/mps_dp/launch.sh manages the weight-sharing "
            "deployment shape)"
        )

    from sglang_omni.platforms import current_platform

    if not current_platform.is_cuda():
        if mode == "on":
            raise MpsError("mps=on requires an NVIDIA CUDA platform")
        logger.warning("MPS auto: platform is not NVIDIA CUDA; running without MPS")
        return None

    if shutil.which("nvidia-cuda-mps-control") is None:
        if mode == "on":
            raise MpsError("mps=on but nvidia-cuda-mps-control is not on PATH")
        logger.warning(
            "MPS auto: nvidia-cuda-mps-control not found; running without MPS"
        )
        return None

    torch = sys.modules.get("torch")
    if torch is not None and torch.cuda.is_initialized():
        logger.warning(
            "CUDA was initialized in the parent before MPS setup; the parent's "
            "own context will run outside MPS"
        )

    from sglang_omni.mps.control import SubprocessMpsControlClient
    from sglang_omni.mps.devices import NvmlDeviceInfo

    return MpsPipelineRuntime.create(
        mode=mode,
        process_specs=process_specs,
        device_info=NvmlDeviceInfo(),
        client=SubprocessMpsControlClient(),
    )
