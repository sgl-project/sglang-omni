# SPDX-License-Identifier: Apache-2.0
"""Serve-local MPS lifecycle and placement eligibility."""

from __future__ import annotations

import asyncio
import getpass
import logging
import os
import secrets
import shutil
import sys
import tempfile
import time
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any, Protocol

from sglang_omni.mps.control import (
    MPS_CLIENT_TOKEN_ENV,
    MpsControlClient,
    MpsControlError,
    MpsDaemonNotStartedError,
    MpsDirtyStateError,
    MpsError,
)
from sglang_omni.mps.decision import (
    MPS_MODES,
    MpsDecisionError,
    MpsProcessFact,
    collect_mps_facts,
)
from sglang_omni.mps.devices import MpsPhysicalDevice
from sglang_omni.mps.state import ensure_private_state_root, validate_control_socket

logger = logging.getLogger(__name__)


class _MpsDeviceInfo(Protocol):
    def inspect(self, gpu_ids: Iterable[int]) -> dict[int, MpsPhysicalDevice]: ...


def _default_state_root() -> Path:
    # Keep this short: the control socket must fit Linux's AF_UNIX path budget.
    override = os.environ.get("SGLANG_OMNI_MPS_STATE_ROOT")
    if override:
        return Path(override)
    return Path(tempfile.gettempdir()) / f"sglang-omni-mps-{getpass.getuser()}"


def _resolve_worker_devices(
    *,
    mode: str,
    process_facts: tuple[MpsProcessFact, ...],
    device_info: _MpsDeviceInfo,
) -> dict[str, str]:
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

    gpu_ids = tuple(
        sorted(
            {gpu_id for fact in potential_clients for gpu_id in fact.placement_gpu_ids}
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
    for fact in potential_clients:
        resolved = {
            gpu_id: uuid_for[gpu_id]
            for gpu_id in fact.placement_gpu_ids
            if gpu_id in uuid_for
        }
        physical_uuids = set(resolved.values())
        if len(physical_uuids) > 1:
            unresolved = sorted(set(fact.placement_gpu_ids) - resolved.keys())
            unresolved_detail = (
                f"; unresolved CUDA ordinals: {unresolved}" if unresolved else ""
            )
            raise MpsError(
                f"process {fact.process_name!r} resolves CUDA ordinals to multiple "
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

    for fact in potential_clients:
        placement_uuids = {uuid_for[gpu_id] for gpu_id in fact.placement_gpu_ids}
        (placement_uuid,) = placement_uuids
        clients_by_uuid.setdefault(placement_uuid, []).append(fact.process_name)
        logical_ids_by_uuid.setdefault(placement_uuid, set()).update(
            fact.placement_gpu_ids
        )

        nonzero_explicit = set(fact.explicit_cuda_gpu_ids) - {0}
        if nonzero_explicit:
            blocked.setdefault(placement_uuid, []).append(
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
        blocked.setdefault(gpu_uuid, []).append("; ".join(reasons))

    worker_devices: dict[str, str] = {}
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
        worker_devices.update((name, gpu_uuid) for name in process_names)

    if mode == "on" and not worker_devices:
        reasons = sorted(
            {reason for gpu_reasons in blocked.values() for reason in gpu_reasons}
        )
        detail = f": {'; '.join(reasons)}" if reasons else ""
        raise MpsDecisionError(
            "mps=on but no physical GPU is eligible for MPS" + detail
        )
    return worker_devices


class MpsPipelineRuntime:
    def __init__(
        self,
        client: MpsControlClient,
        process_names: Iterable[str],
        state_root: Path,
    ):
        self.client = client
        self._state_root = state_root
        self._client_tokens = {name: secrets.token_hex(16) for name in process_names}
        self._operation_lock = asyncio.Lock()
        self.run_dir: Path | None = None
        self.daemon_pid: int | None = None
        self.server_pid: int | None = None
        self.poll_interval = 0.2
        self.start_timeout = 5.0
        self.verify_timeout = 30.0
        self.stop_timeout = 10.0

    @property
    def pipe_dir(self) -> Path:
        if self.run_dir is None:
            raise MpsError("MPS runtime has not started")
        return self.run_dir / "pipe"

    @property
    def log_dir(self) -> Path:
        if self.run_dir is None:
            raise MpsError("MPS runtime has not started")
        return self.run_dir / "log"

    @property
    def has_resources(self) -> bool:
        return self.run_dir is not None

    async def start(self, gpu_uuids: Iterable[str]) -> None:
        async with self._operation_lock:
            if self.has_resources:
                raise MpsError("MPS runtime is already started")
            try:
                await self._run_blocking(self._start, tuple(sorted(set(gpu_uuids))))
            except BaseException as startup_error:
                try:
                    await self._run_blocking(self._close)
                except BaseException as cleanup_error:
                    raise startup_error from cleanup_error
                raise

    def _start(self, gpu_uuids: tuple[str, ...]) -> None:
        if not gpu_uuids or not self._client_tokens:
            raise MpsError("MPS startup requires GPUs and managed processes")
        ensure_private_state_root(self._state_root)
        run_dir = Path(tempfile.mkdtemp(prefix="run-", dir=self._state_root))
        try:
            validate_control_socket(run_dir / "pipe" / "control")
            (run_dir / "pipe").mkdir(mode=0o700)
            (run_dir / "log").mkdir(mode=0o700)
        except BaseException:
            shutil.rmtree(run_dir)
            raise
        self.run_dir = run_dir
        startup_error: MpsControlError | None = None
        try:
            self.client.start_daemon(self.pipe_dir, self.log_dir, gpu_uuids)
        except MpsDaemonNotStartedError:
            shutil.rmtree(run_dir)
            self.run_dir = None
            raise
        except MpsControlError as exc:
            startup_error = exc
        deadline = time.monotonic() + self.start_timeout
        while True:
            try:
                self._check_daemon_identity()
                self.client.snapshot(self.pipe_dir)
                break
            except MpsControlError as exc:
                if time.monotonic() >= deadline:
                    if startup_error is not None:
                        raise startup_error from exc
                    raise MpsError(
                        f"MPS control daemon did not become ready: {exc}"
                    ) from exc
                time.sleep(self.poll_interval)
        if startup_error is not None:
            raise startup_error
        logger.info(
            "MPS daemon %s ready on GPUs %s (run dir %s)",
            self.daemon_pid,
            gpu_uuids,
            run_dir,
        )

    def _check_daemon_identity(self) -> None:
        pid = self.client.read_daemon_identity(self.pipe_dir)
        if self.daemon_pid is not None and pid != self.daemon_pid:
            raise MpsControlError(
                f"MPS daemon identity changed from {self.daemon_pid} to {pid}"
            )
        self.daemon_pid = pid

    def env_for_process(self, process_name: str) -> dict[str, str]:
        token = self._client_tokens.get(process_name)
        if token is None:
            return {}
        return {
            "CUDA_MPS_PIPE_DIRECTORY": str(self.pipe_dir),
            "CUDA_MPS_LOG_DIRECTORY": str(self.log_dir),
            MPS_CLIENT_TOKEN_ENV: token,
        }

    async def verify(self) -> None:
        async with self._operation_lock:
            await self._run_blocking(self._verify)

    def _verify(self) -> None:
        expected = {token: name for name, token in self._client_tokens.items()}
        deadline = time.monotonic() + self.verify_timeout
        while True:
            observed: set[str] = set()
            servers: set[int] = set()
            last_error = None
            try:
                for ref in self.client.snapshot(self.pipe_dir):
                    token = self.client.client_token(ref.client_pid)
                    if token in expected:
                        observed.add(token)
                        servers.add(ref.server_pid)
                if observed == expected.keys():
                    if len(servers) != 1:
                        raise MpsError(
                            f"managed MPS clients must share one server, got {sorted(servers)}"
                        )
                    (self.server_pid,) = servers
                    return
            except MpsControlError as exc:
                last_error = exc
            if time.monotonic() >= deadline:
                missing = sorted(
                    expected[token] for token in expected.keys() - observed
                )
                raise MpsError(
                    f"stage process(es) {missing} never attached to the MPS server (pipe dir {self.pipe_dir}); last control error: {last_error}"
                )
            time.sleep(self.poll_interval)

    async def retire_process_clients(self, process_name: str) -> None:
        """Best-effort CUDA context termination before worker shutdown."""

        async with self._operation_lock:
            try:
                await self._run_blocking(self._retire_process_clients, process_name)
            except MpsControlError as exc:
                logger.warning(
                    "Could not query MPS clients for %s; continuing worker shutdown: %s",
                    process_name,
                    exc,
                )

    def _retire_process_clients(self, process_name: str) -> None:
        token = self._client_tokens.get(process_name)
        if token is None or not self.has_resources:
            return
        self._check_daemon_identity()
        for ref in sorted(self.client.snapshot(self.pipe_dir)):
            try:
                if self.client.client_token(ref.client_pid) == token:
                    self.client.terminate_client(self.pipe_dir, ref)
            except MpsControlError as exc:
                logger.warning(
                    "Could not retire MPS client %s; continuing worker shutdown: %s",
                    ref,
                    exc,
                )

    async def probe(self) -> str | None:
        async with self._operation_lock:
            return await self._run_blocking(self._probe)

    def _probe(self) -> str | None:
        if self.server_pid is None:
            return "MPS server attachment is not verified"
        try:
            status = self.client.get_server_status(self.pipe_dir, self.server_pid)
        except MpsControlError as exc:
            return f"server {self.server_pid} status query failed: {exc}"
        if status != "ACTIVE":
            return f"server {self.server_pid} is not ACTIVE: {status!r}"
        return None

    async def close(self) -> None:
        async with self._operation_lock:
            await self._run_blocking(self._close)

    def _close(self) -> None:
        if self.run_dir is None:
            return
        try:
            self._check_daemon_identity()
            assert self.daemon_pid is not None
            try:
                self.client.quit_daemon(self.pipe_dir)
            except MpsControlError:
                if self.client.daemon_process_alive(self.daemon_pid):
                    raise
            deadline = time.monotonic() + self.stop_timeout
            while self.client.daemon_process_alive(self.daemon_pid):
                if time.monotonic() >= deadline:
                    raise MpsError(
                        f"MPS daemon {self.daemon_pid} did not exit after quit"
                    )
                time.sleep(self.poll_interval)
            shutil.rmtree(self.run_dir)
        except Exception as exc:
            raise MpsDirtyStateError(
                f"MPS cleanup incomplete: {exc}. Run directory preserved: {self.run_dir}"
            ) from exc
        self.run_dir = None
        self.daemon_pid = None
        self.server_pid = None

    @staticmethod
    async def _run_blocking(call: Callable[..., Any], *args: Any) -> Any:
        """Finish the native operation before propagating cancellation."""

        task = asyncio.create_task(asyncio.to_thread(call, *args))
        cancelled: asyncio.CancelledError | None = None
        while not task.done():
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError as exc:
                cancelled = cancelled or exc
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


def create_for_pipeline(
    mode: str,
    process_specs,
    *,
    device_info: _MpsDeviceInfo | None = None,
    client: MpsControlClient | None = None,
    state_root: Path | None = None,
) -> tuple[MpsPipelineRuntime | None, dict[str, str]]:
    """Resolve eligible workers and build their serve-local MPS runtime."""

    if mode not in MPS_MODES:
        raise MpsDecisionError(f"invalid mps mode {mode!r}; expected {MPS_MODES}")
    if mode == "off":
        return None, {}
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
            f"SGLANG_OMNI_WEIGHT_SHARE={weight_share!r}; remove it and request "
            "weight sharing with weight_share=on, which assigns replica roles "
            "itself, or use mps=off with the external supervisor"
        )

    from sglang_omni.platforms import current_platform

    if not current_platform.is_cuda():
        if mode == "on":
            raise MpsError("mps=on requires an NVIDIA CUDA platform")
        logger.warning("MPS auto: platform is not NVIDIA CUDA; running without MPS")
        return None, {}

    if shutil.which("nvidia-cuda-mps-control") is None:
        if mode == "on":
            raise MpsError("mps=on but nvidia-cuda-mps-control is not on PATH")
        logger.warning(
            "MPS auto: nvidia-cuda-mps-control not found; running without MPS"
        )
        return None, {}

    torch = sys.modules.get("torch")
    if torch is not None and torch.cuda.is_initialized():
        logger.warning(
            "CUDA was initialized in the parent before MPS setup; the parent's "
            "own context will run outside MPS"
        )

    from sglang_omni.mps.control import SubprocessMpsControlClient
    from sglang_omni.mps.devices import NvmlDeviceInfo

    worker_devices = _resolve_worker_devices(
        mode=mode,
        process_facts=collect_mps_facts(process_specs),
        device_info=NvmlDeviceInfo() if device_info is None else device_info,
    )
    if not worker_devices:
        return None, {}
    root = state_root if state_root is not None else _default_state_root()
    control = SubprocessMpsControlClient() if client is None else client
    return MpsPipelineRuntime(control, worker_devices, root), worker_devices
