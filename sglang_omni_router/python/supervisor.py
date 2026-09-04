# SPDX-License-Identifier: Apache-2.0
"""Multi-process router supervisor.

Process model: the supervisor binds the public data socket once and passes
the inherited fd to N data-plane (DP) subprocesses, which accept from the
shared listen queue. A separate control-plane (CP) process serves the
registry, the admin plane, and the internal channel on a
permission-controlled UDS or a token-guarded localhost port.

Lifecycle invariants:
- Slot reclaim ordering: a dead DP is first reaped (wait()), only then is its
  slot reclaimed (on_dp_exit hook) and the generation bumped; the replacement
  is spawned with the new generation. A still-running process is never
  reclaimed.
- A CP restart gets a fresh cp_epoch, so snapshot readers accept the restarted
  seq stream.
- Shutdown order: DPs first, then the CP, then the launcher (caller-owned),
  then unlink of workdir artifacts.
- Fail closed: a DP slot that keeps dying immediately after spawn stops the
  whole supervisor instead of flapping forever.
"""

from __future__ import annotations

import logging
import mmap
import os
import platform
import secrets
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass, replace
from typing import Callable, Protocol

from sglang_omni_router.python.admission_shm import (
    SeqlockUnstableError,
    SlotCodec,
    admission_file_size,
    create_admission_file,
)
from sglang_omni_router.python.app_factory import CONFIG_FILE_ENV
from sglang_omni_router.python.config import RouterConfig
from sglang_omni_router.python.internal_channel import (
    CONTROL_CHANNEL_CONNECTIONS,
    FORWARD_CHANNEL_CONNECTIONS,
    INTERNAL_TOKEN_ENV,
)

logger = logging.getLogger("sglang_omni_router.python.supervisor")

SOCKET_FD_ENV = "SGLANG_OMNI_ROUTER_SOCKET_FD"
DEATH_PIPE_FD_ENV = "SGLANG_OMNI_ROUTER_DEATH_PIPE_FD"
DP_INDEX_ENV = "SGLANG_OMNI_ROUTER_DP_INDEX"
DP_GENERATION_ENV = "SGLANG_OMNI_ROUTER_DP_GENERATION"
CP_EPOCH_ENV = "SGLANG_OMNI_ROUTER_CP_EPOCH"
INTERNAL_UDS_ENV = "SGLANG_OMNI_ROUTER_INTERNAL_UDS"
INTERNAL_TCP_URL_ENV = "SGLANG_OMNI_ROUTER_INTERNAL_TCP_URL"
SNAPSHOT_PATH_ENV = "SGLANG_OMNI_ROUTER_SNAPSHOT_PATH"
EXPECTED_DPS_ENV = "SGLANG_OMNI_ROUTER_EXPECTED_DPS"
ADMISSION_SHM_ENV = "SGLANG_OMNI_ROUTER_ADMISSION_SHM"
LOG_LEVEL_ENV = "SGLANG_OMNI_ROUTER_LOG_LEVEL"

# Note (Jiaxin Deng): the shared admission seqlock relies on x86-64 store
# ordering, so anywhere else the protocol is simply unvalidated; refuse rather
# than run it and hope.
SUPPORTED_MACHINES = ("x86_64", "amd64")

_LISTEN_BACKLOG = 2048
_SHUTDOWN_GRACE_SECS = 10.0
# Note (Jiaxin Deng): CP only. A DP's graceful deadline is the configurable
# shutdown drain (default --request-timeout-secs); the CP is stopped after the
# DPs have drained and runs short admin handlers.
CHILD_GRACEFUL_SHUTDOWN_SECS = 5
# Note (Jiaxin Deng): a dead supervisor cannot escalate, so the child arms its
# own hard exit. Deliberately short and NOT scaled by the shutdown drain: an
# orphan is holding the public socket of a router that no longer exists.
_ORPHAN_HARD_EXIT_SECS = 2 * CHILD_GRACEFUL_SHUTDOWN_SECS


class ChildProcess(Protocol):
    """The subset of subprocess.Popen the supervisor relies on (fakeable)."""

    pid: int

    def poll(self) -> int | None: ...

    def wait(self, timeout: float | None = None) -> int: ...

    def terminate(self) -> None: ...

    def kill(self) -> None: ...


@dataclass
class DataPlaneSlot:
    index: int
    generation: int
    process: ChildProcess
    spawned_at: float
    rapid_deaths: int = 0


@dataclass
class SupervisorContext:
    config_path: str
    socket_fd: int
    cp_epoch: str
    internal_token: str
    internal_uds: str | None
    internal_tcp_url: str | None
    workdir: str
    snapshot_path: str
    death_pipe_fd: int = -1
    expected_data_planes: int = 0
    admission_shm_path: str = ""

    def child_env(self) -> dict[str, str]:
        env = os.environ.copy()
        env[CONFIG_FILE_ENV] = self.config_path
        env[CP_EPOCH_ENV] = self.cp_epoch
        env[INTERNAL_TOKEN_ENV] = self.internal_token
        env[SNAPSHOT_PATH_ENV] = self.snapshot_path
        if self.admission_shm_path:
            env[ADMISSION_SHM_ENV] = self.admission_shm_path
        if self.expected_data_planes > 0:
            env[EXPECTED_DPS_ENV] = str(self.expected_data_planes)
        if self.death_pipe_fd >= 0:
            env[DEATH_PIPE_FD_ENV] = str(self.death_pipe_fd)
        # Note (Jiaxin Deng): exactly one transport; a stale variable inherited
        # from the parent environment must not misdirect the children.
        if self.internal_uds:
            env[INTERNAL_UDS_ENV] = self.internal_uds
            env.pop(INTERNAL_TCP_URL_ENV, None)
        else:
            env.pop(INTERNAL_UDS_ENV, None)
        if self.internal_tcp_url:
            env[INTERNAL_TCP_URL_ENV] = self.internal_tcp_url
        return env


class SupervisorFailure(RuntimeError):
    """A slot kept dying immediately; the supervisor fails closed."""


def watch_supervisor_liveness(
    *,
    sleep: Callable[[float], None] = time.sleep,
    hard_exit: Callable[[], None] = lambda: os._exit(1),
) -> threading.Thread | None:
    """Child-side: exit when the supervisor dies for ANY reason.

    The supervisor holds the write end of a pipe; every child watches the
    inherited read end from a daemon thread. A supervisor crash or SIGKILL
    closes the write end, the read returns EOF, and the child SIGTERMs
    itself instead of serving as an orphan on the still-open public socket.
    A request that never completes would survive that SIGTERM, so the same
    thread escalates to a hard exit after _ORPHAN_HARD_EXIT_SECS.
    """
    fd_value = os.environ.get(DEATH_PIPE_FD_ENV)
    if not fd_value:
        return None

    def _watch() -> None:
        try:
            while os.read(int(fd_value), 1):
                pass
        except OSError:
            pass
        os.kill(os.getpid(), signal.SIGTERM)
        sleep(_ORPHAN_HARD_EXIT_SECS)
        hard_exit()

    thread = threading.Thread(target=_watch, daemon=True)
    thread.start()
    return thread


def _default_spawn_dp(
    ctx: SupervisorContext, index: int, generation: int
) -> ChildProcess:
    env = ctx.child_env()
    env[SOCKET_FD_ENV] = str(ctx.socket_fd)
    env[DP_INDEX_ENV] = str(index)
    env[DP_GENERATION_ENV] = str(generation)
    return subprocess.Popen(
        [sys.executable, "-m", "sglang_omni_router.python.dp_runner"],
        env=env,
        pass_fds=(ctx.socket_fd, ctx.death_pipe_fd),
    )


def _default_spawn_cp(ctx: SupervisorContext) -> ChildProcess:
    return subprocess.Popen(
        [sys.executable, "-m", "sglang_omni_router.python.cp_runner"],
        env=ctx.child_env(),
        pass_fds=(ctx.death_pipe_fd,),
    )


def validate_multiprocess_settings(
    *,
    router_processes: int,
    effective_max_inflight: int | None,
    policy: str,
) -> None:
    """Deterministic rejects, callable before any GPU worker is launched.

    effective_max_inflight is None when the bound is still to be derived from
    the worker count; the supervisor re-validates with the resolved value.
    """
    if router_processes < 1:
        raise ValueError("router_processes must be >= 1")
    if effective_max_inflight is not None and effective_max_inflight < router_processes:
        raise ValueError(
            f"max in-flight bound {effective_max_inflight} is below "
            f"router_processes={router_processes}; raise --max-connections/"
            "--max-inflight or lower the process count (the soft bound's "
            "N-1 overshoot would dominate such a small budget)"
        )
    machine = platform.machine().lower()
    if router_processes > 1 and machine not in SUPPORTED_MACHINES:
        raise ValueError(
            f"multi-process admission is validated on x86-64 only and this "
            f"machine reports {machine or 'unknown'}; run with "
            "--router-processes 1"
        )
    if policy == "least_request" and router_processes > 1:
        raise ValueError(
            "least_request needs the cross-request counters of a single "
            "process and is not supported with multiple router processes; "
            "use --policy round_robin, --policy random, or run with "
            "--router-processes 1"
        )


class RouterSupervisor:
    def __init__(
        self,
        config: RouterConfig,
        *,
        router_processes: int,
        workdir: str | None = None,
        prefer_uds: bool | None = None,
        spawn_dp: Callable[[SupervisorContext, int, int], ChildProcess] | None = None,
        spawn_cp: Callable[[SupervisorContext], ChildProcess] | None = None,
        clock: Callable[[], float] = time.monotonic,
        rapid_window_secs: float = 5.0,
        max_rapid_restarts: int = 3,
    ) -> None:
        validate_multiprocess_settings(
            router_processes=router_processes,
            effective_max_inflight=config.effective_max_inflight,
            policy=config.policy,
        )
        self._config = config
        self._router_processes = router_processes
        self._workdir = workdir
        self._owns_workdir = workdir is None
        self._prefer_uds = (
            prefer_uds if prefer_uds is not None else hasattr(socket, "AF_UNIX")
        )
        self._spawn_dp = spawn_dp or _default_spawn_dp
        self._spawn_cp = spawn_cp or _default_spawn_cp
        self._clock = clock
        self._rapid_window_secs = rapid_window_secs
        self._max_rapid_restarts = max_rapid_restarts

        self._socket: socket.socket | None = None
        self._context: SupervisorContext | None = None
        self._cp_process: ChildProcess | None = None
        self._cp_spawned_at: float = 0.0
        self._cp_rapid_deaths: int = 0
        self._death_pipe_read: int | None = None
        self._death_pipe_write: int | None = None
        self._admission_file = None
        self._admission_mmap: mmap.mmap | None = None
        self._dp_slots: dict[int, DataPlaneSlot] = {}
        self._stop_requested = False
        self._interrupted = False
        # Note (Jiaxin Deng): hook point for the shared-admission slot reclaim;
        # called only after the dead process is reaped.
        self.on_dp_exit: Callable[[int, int, int], None] | None = None

    def request_stop(self, *, interrupted: bool = False) -> None:
        """Signal-safe: makes run_forever leave its loop and shut down."""
        self._stop_requested = True
        if interrupted:
            self._interrupted = True

    @property
    def stopped_by_interrupt(self) -> bool:
        return self._interrupted

    @property
    def context(self) -> SupervisorContext:
        if self._context is None:
            raise RuntimeError("supervisor is not started")
        return self._context

    @property
    def dp_slots(self) -> dict[int, DataPlaneSlot]:
        return self._dp_slots

    @property
    def cp_process(self) -> ChildProcess | None:
        return self._cp_process

    def start(self) -> None:
        if self._context is not None:
            raise RuntimeError("supervisor already started")
        try:
            self._start_inner()
        except BaseException:
            # Note (Jiaxin Deng): a half-started supervisor must not leak
            # spawned children, the bound socket, or workdir artifacts.
            self._cleanup_partial_start()
            raise

    def _cleanup_partial_start(self) -> None:
        # Note (Jiaxin Deng): same order as shutdown(): drop the listener,
        # signal every DP, then await; a blocked drain must not leave siblings
        # unsignaled or connections queued in an acceptorless backlog.
        if self._socket is not None:
            self._socket.close()
            self._socket = None
        for slot in self._dp_slots.values():
            self._signal_child(slot.process)
        for slot in self._dp_slots.values():
            self._await_child(slot.process)
        self._dp_slots.clear()
        if self._cp_process is not None:
            self._stop_child(self._cp_process)
            self._cp_process = None
        self._close_death_pipe()
        self._close_admission_shm()
        self._context = None
        if self._workdir:
            for name in (
                "router_config.json",
                "internal.sock",
                "workers.json",
                "admission.shm",
            ):
                try:
                    os.unlink(os.path.join(self._workdir, name))
                except OSError:
                    pass
            if self._owns_workdir:
                try:
                    os.rmdir(self._workdir)
                except OSError:
                    pass
                self._workdir = None

    def _start_inner(self) -> None:
        if self._workdir is None:
            self._workdir = tempfile.mkdtemp(prefix="sglang-omni-router-")
        os.chmod(self._workdir, 0o700)

        config_path = os.path.join(self._workdir, "router_config.json")
        with open(config_path, "w", encoding="utf-8") as f:
            f.write(self._config.model_dump_json())

        # Note (Jiaxin Deng): mirror uvicorn's bind_socket family choice so N=1
        # and N>=2 listen identically; AF_INET gaierrors on ::/::1 and
        # getaddrinfo[0] picks an OS-dependent family for names like localhost.
        family = socket.AF_INET6 if ":" in (self._config.host or "") else socket.AF_INET
        self._socket = socket.socket(family, socket.SOCK_STREAM)
        self._socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._socket.bind((self._config.host, self._config.port))
        self._socket.listen(_LISTEN_BACKLOG)
        self._socket.set_inheritable(True)

        internal_uds = None
        internal_tcp_url = None
        if self._prefer_uds:
            internal_uds = os.path.join(self._workdir, "internal.sock")
        else:
            probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            probe.bind(("127.0.0.1", 0))
            internal_tcp_url = f"http://127.0.0.1:{probe.getsockname()[1]}"
            probe.close()

        self._death_pipe_read, self._death_pipe_write = os.pipe()
        os.set_inheritable(self._death_pipe_read, True)

        admission_shm_path = os.path.join(self._workdir, "admission.shm")
        create_admission_file(admission_shm_path, self._router_processes)
        self._admission_file = open(admission_shm_path, "r+b")
        self._admission_mmap = mmap.mmap(
            self._admission_file.fileno(),
            admission_file_size(self._router_processes),
        )

        self._context = SupervisorContext(
            config_path=config_path,
            socket_fd=self._socket.fileno(),
            cp_epoch=uuid.uuid4().hex,
            internal_token=secrets.token_hex(16),
            internal_uds=internal_uds,
            internal_tcp_url=internal_tcp_url,
            workdir=self._workdir,
            snapshot_path=os.path.join(self._workdir, "workers.json"),
            death_pipe_fd=self._death_pipe_read,
            expected_data_planes=self._router_processes,
            admission_shm_path=admission_shm_path,
        )
        pool = self._config.upstream_pool_size
        internal_fds = CONTROL_CHANNEL_CONNECTIONS + FORWARD_CHANNEL_CONNECTIONS
        per_process_fds = 2 * pool + 64
        logger.info(
            f"multi-process fd budget (upstream-relay estimate): per-process "
            f"~{per_process_fds} (2 x pool {pool} + headroom, plus a bounded "
            f"internal channel of <= {internal_fds} fds, split "
            f"{CONTROL_CHANNEL_CONNECTIONS} control + "
            f"{FORWARD_CHANNEL_CONNECTIONS} forwarding); cluster total across "
            f"{self._router_processes} DPs "
            f"~{self._router_processes * per_process_fds}. Inbound idle "
            f"keep-alive sockets are not bounded by admission, matching the "
            f"single-process router."
        )
        self._cp_process = self._spawn_cp(self._context)
        self._cp_spawned_at = self._clock()
        for index in range(self._router_processes):
            self._spawn_dp_slot(index, generation=1)

    def _spawn_dp_slot(self, index: int, generation: int) -> None:
        process = self._spawn_dp(self.context, index, generation)
        previous = self._dp_slots.get(index)
        self._dp_slots[index] = DataPlaneSlot(
            index=index,
            generation=generation,
            process=process,
            spawned_at=self._clock(),
            rapid_deaths=previous.rapid_deaths if previous else 0,
        )

    def poll_once(self) -> None:
        """Reap dead children and restart them under the fixed ordering."""
        for slot in list(self._dp_slots.values()):
            if slot.process.poll() is None:
                continue
            returncode = slot.process.wait()
            logger.warning(
                f"data-plane {slot.index} (generation {slot.generation}, "
                f"pid {slot.process.pid}) exited with {returncode}"
            )
            # Note (Jiaxin Deng): reclaim only after the reap; fold the dead
            # DP's rejected/peak into the retired slot first so aggregate
            # totals never move backwards.
            if self._admission_mmap is not None:
                self._fold_and_reclaim_slot(slot.index)
            if self.on_dp_exit is not None:
                self.on_dp_exit(slot.index, slot.generation, returncode)
            lifetime = self._clock() - slot.spawned_at
            if lifetime < self._rapid_window_secs:
                slot.rapid_deaths += 1
            else:
                slot.rapid_deaths = 0
            if slot.rapid_deaths >= self._max_rapid_restarts:
                raise SupervisorFailure(
                    f"data-plane {slot.index} died {slot.rapid_deaths} times "
                    f"within {self._rapid_window_secs}s of spawn; failing "
                    "closed instead of flapping"
                )
            self._spawn_dp_slot(slot.index, generation=slot.generation + 1)

        if self._cp_process is not None and self._cp_process.poll() is not None:
            returncode = self._cp_process.wait()
            logger.warning(f"control-plane exited with {returncode}; restarting")
            if self._clock() - self._cp_spawned_at < self._rapid_window_secs:
                self._cp_rapid_deaths += 1
            else:
                self._cp_rapid_deaths = 0
            if self._cp_rapid_deaths >= self._max_rapid_restarts:
                raise SupervisorFailure(
                    f"control plane died {self._cp_rapid_deaths} times within "
                    f"{self._rapid_window_secs}s of spawn; failing closed "
                    "instead of flapping"
                )
            # Note (Jiaxin Deng): a hard-killed CP leaves its UDS socket file
            # behind, and the replacement cannot bind until it is gone.
            if self.context.internal_uds:
                try:
                    os.unlink(self.context.internal_uds)
                except OSError:
                    pass
            # Note (Jiaxin Deng): a restarted CP is a new epoch; snapshot seqs
            # restart from 1.
            self._context = replace(self.context, cp_epoch=uuid.uuid4().hex)
            self._cp_process = self._spawn_cp(self._context)
            self._cp_spawned_at = self._clock()

    def run_forever(self, poll_interval_secs: float = 0.5) -> None:
        # Note (Jiaxin Deng): without a SIGTERM handler the default action
        # kills the supervisor with no cleanup and orphans every child on the
        # still-bound public socket.
        previous_handlers: dict[int, object] = {}

        def _handler(signum, _frame):
            self.request_stop(interrupted=signum == signal.SIGINT)

        for signum in (signal.SIGTERM, signal.SIGINT):
            try:
                previous_handlers[signum] = signal.signal(signum, _handler)
            except (ValueError, OSError):
                # Note (Jiaxin Deng): not the main thread; caller owns signals.
                pass
        try:
            while not self._stop_requested:
                self.poll_once()
                time.sleep(poll_interval_secs)
        finally:
            # Note (Jiaxin Deng): shut down with our handler still installed,
            # so a second signal during the drain is absorbed instead of
            # killing us mid-cleanup and leaking workdir/shm/UDS.
            try:
                self.shutdown()
            finally:
                for signum, handler in previous_handlers.items():
                    # Note (Jiaxin Deng): a C-level previous handler reads back
                    # as None and cannot be reinstalled; leave ours.
                    if handler is not None:
                        signal.signal(signum, handler)

    def _signal_child(self, process: ChildProcess) -> None:
        if process.poll() is None:
            process.terminate()

    def _await_child(
        self, process: ChildProcess, *, timeout: float = _SHUTDOWN_GRACE_SECS
    ) -> None:
        if process.poll() is not None:
            return
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()

    def _stop_child(self, process: ChildProcess) -> None:
        self._signal_child(process)
        self._await_child(process)

    def _dp_stop_budget(self) -> float:
        # Note (Jiaxin Deng): the wait must outlast the DP's own drain
        # deadline, or the supervisor SIGKILLs mid-drain and truncates exactly
        # what the drain protects.
        return self._config.effective_shutdown_drain_secs + _SHUTDOWN_GRACE_SECS

    def _close_death_pipe(self) -> None:
        for fd_attr in ("_death_pipe_write", "_death_pipe_read"):
            fd = getattr(self, fd_attr)
            if fd is not None:
                try:
                    os.close(fd)
                except OSError:
                    pass
                setattr(self, fd_attr, None)

    def _fold_and_reclaim_slot(self, index: int) -> None:
        dying = SlotCodec(self._admission_mmap, index)
        retired = SlotCodec(self._admission_mmap, self._router_processes)
        try:
            view = dying.read(fail_fast=True)
            accumulated = retired.read(fail_fast=True)
        except SeqlockUnstableError:
            # Note (Jiaxin Deng): the owner died mid-write; its final counters
            # are unreadable and drop out of the fold.
            logger.warning(
                f"admission slot {index} unreadable at reclaim; its retired "
                "counters were skipped"
            )
            dying.reclaim()
            return
        # Note (Jiaxin Deng): hold the retired slot mid-write across both the
        # field update and the reclaim, so an aggregate reader retries until
        # the whole transfer completes (no double count).
        marker = retired.begin_write()
        retired.write_fields(
            inflight=0,
            peak_sum=max(accumulated.peak_sum, view.peak_sum),
            rejected_total=accumulated.rejected_total + view.rejected_total,
            generation=0,
            pid=0,
            heartbeat_ts=time.time(),
        )
        dying.reclaim()
        retired.end_write(marker)

    def _close_admission_shm(self) -> None:
        if self._admission_mmap is not None:
            try:
                self._admission_mmap.close()
            except (OSError, ValueError):
                pass
            self._admission_mmap = None
        if self._admission_file is not None:
            try:
                self._admission_file.close()
            except OSError:
                pass
            self._admission_file = None

    def shutdown(self) -> None:
        # Note (Jiaxin Deng): drop the parent's listener FIRST: once the DPs
        # exit the kernel socket dies with their inherited fds and connections
        # are refused, instead of queueing in a backlog nobody will serve.
        if self._socket is not None:
            self._socket.close()
            self._socket = None
        # Note (Jiaxin Deng): signal every DP before awaiting any, so none keeps
        # accepting from the shared socket while its siblings drain.
        for slot in self._dp_slots.values():
            self._signal_child(slot.process)
        for slot in self._dp_slots.values():
            self._await_child(slot.process, timeout=self._dp_stop_budget())
        self._dp_slots.clear()
        if self._cp_process is not None:
            self._stop_child(self._cp_process)
            self._cp_process = None
        self._close_death_pipe()
        self._close_admission_shm()
        context, self._context = self._context, None
        if context is not None:
            for path in (
                context.config_path,
                context.internal_uds,
                context.snapshot_path,
                context.admission_shm_path,
            ):
                if path:
                    try:
                        os.unlink(path)
                    except OSError:
                        pass
            if self._owns_workdir:
                try:
                    os.rmdir(context.workdir)
                except OSError:
                    pass
