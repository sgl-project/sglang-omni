# SPDX-License-Identifier: Apache-2.0
"""Ownership-based lifecycle for one shared per-GPU CUDA MPS daemon.

The manager has no lifecycle state of its own. A successful :meth:`acquire`
returns the only cleanup authority, an :class:`MpsLease`; every later operation
requires that token. Existing state is joined only when the native daemon
identity is provable and every published owner lease is held. Anything
ambiguous is preserved for an operator instead of being repaired in place.
"""

from __future__ import annotations

import fcntl
import logging
import os
import shlex
import shutil
import time
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import NoReturn, Protocol

from sglang_omni.mps.state import MpsGpuPaths, state_root_lock, validate_control_socket

logger = logging.getLogger(__name__)


class MpsError(RuntimeError):
    """Raised when the MPS lifecycle cannot proceed safely."""


class MpsDirtyStateError(MpsError):
    """Cleanup persisted dirty state and released its owner lock."""


class MpsControlError(MpsError):
    """Raised when a strict MPS control or process query fails."""


class MpsRetryableControlError(MpsControlError):
    """A control observation or precondition may succeed on a later attempt."""


class MpsDaemonNotStartedError(MpsControlError):
    """The control binary was not executed, so it cannot have created a daemon."""


_OWNER_ACTIVE = "active"
_OWNER_RETAINED = "retained"
_OWNER_STATUSES = {_OWNER_ACTIVE, _OWNER_RETAINED}
MPS_CLIENT_TOKEN_ENV = "SGLANG_OMNI_MPS_CLIENT_TOKEN"


@dataclass(frozen=True, order=True)
class MpsClientRef:
    """One CUDA client as identified by the MPS server that owns it."""

    server_pid: int
    client_pid: int


@dataclass
class MpsLease:
    """All authority and runtime-local evidence owned by one acquisition."""

    daemon_pid: int
    owner_fd: int
    client_tokens: dict[str, str]
    server_pid: int | None = None


class MpsControlClient(Protocol):
    """Native I/O; retryable failures raise MpsRetryableControlError."""

    def start_daemon(self, pipe_dir: Path, log_dir: Path, gpu_uuid: str) -> None: ...

    def read_daemon_identity(self, pipe_dir: Path) -> int: ...

    def snapshot(self, pipe_dir: Path) -> set[MpsClientRef]: ...

    def get_server_status(self, pipe_dir: Path, server_pid: int) -> str: ...

    def terminate_client(self, pipe_dir: Path, client: MpsClientRef) -> None: ...

    def quit_daemon(self, pipe_dir: Path) -> None: ...

    def daemon_process_alive(self, pid: int) -> bool: ...

    def client_token(self, pid: int) -> str | None: ...

    def owner_lease_held(self, lease_file: Path) -> bool: ...


@dataclass
class ExistingState:
    daemon_pid: int | None = None
    owners: dict[int, bool] = field(default_factory=dict)
    owner_statuses: dict[int, str] = field(default_factory=dict)
    clients: set[MpsClientRef] | None = None
    errors: list[str] = field(default_factory=list)


@dataclass
class MpsManager:
    """Own GPU transactions; the runtime serializes calls using each lease."""

    paths: MpsGpuPaths
    client: MpsControlClient
    poll_interval: float = 0.2
    start_timeout: float = 5.0
    verify_timeout: float = 30.0
    drain_timeout: float = 60.0
    stop_timeout: float = 10.0

    @property
    def gpu_uuid(self) -> str:
        return self.paths.gpu_uuid

    @property
    def owner_file(self) -> Path:
        return self.paths.owners_dir / str(os.getpid())

    @contextmanager
    def gpu_transaction(self) -> Iterator[None]:
        """Serialize shared state and native control commands for this GPU."""

        try:
            with state_root_lock(self.paths.state_root, f".lock-{self.gpu_uuid}"):
                yield
        except (OSError, ValueError) as exc:
            raise MpsControlError(
                f"MPS GPU transaction failed on {self.gpu_uuid}: {exc}"
            ) from exc

    def acquire(self, client_tokens: Mapping[str, str]) -> MpsLease:
        """Create or join the daemon and return the sole cleanup token."""

        tokens = dict(client_tokens)
        if not tokens or len(set(tokens.values())) != len(tokens):
            raise MpsError(
                "MPS acquisition requires one unique client token per process"
            )
        else:
            pass
        validate_control_socket(self.paths.control_socket)
        with self.gpu_transaction():
            if not self.paths.state_dir.exists():
                return self.create_locked(tokens)
            else:
                pass
            return self.join_locked(tokens)

    def create_locked(self, client_tokens: dict[str, str]) -> MpsLease:
        self.paths.pipe_dir.mkdir(parents=True)
        self.paths.log_dir.mkdir()
        self.paths.owners_dir.mkdir()
        owner_fd = self.publish_owner()
        lease: MpsLease | None = None
        try:
            self.client.start_daemon(
                self.paths.pipe_dir, self.paths.log_dir, self.gpu_uuid
            )
            lease = MpsLease(
                daemon_pid=self.client.read_daemon_identity(self.paths.pipe_dir),
                owner_fd=owner_fd,
                client_tokens=client_tokens,
            )
            self.wait_for_snapshot(
                self.start_timeout,
                "MPS control daemon did not answer on its control socket",
            )
            return lease
        except MpsDaemonNotStartedError as startup_error:
            try:
                self.discard_owner_fd(owner_fd)
                shutil.rmtree(self.paths.state_dir)
            except OSError as cleanup_error:
                raise startup_error from cleanup_error
            raise
        except Exception as startup_error:
            try:
                if lease is None:
                    try:
                        lease = MpsLease(
                            daemon_pid=self.client.read_daemon_identity(
                                self.paths.pipe_dir
                            ),
                            owner_fd=owner_fd,
                            client_tokens=client_tokens,
                        )
                    except MpsControlError as identity_error:
                        self.persist_unidentified_dirty(owner_fd, identity_error)
                else:
                    pass
                self.rollback_create(lease)
            except Exception as cleanup_error:
                raise startup_error from cleanup_error
            raise

    def join_locked(self, client_tokens: dict[str, str]) -> MpsLease:
        state = self.inspect_existing_state()
        if (
            not state.errors
            and state.daemon_pid is not None
            and state.clients is not None
            and state.owners
            and all(state.owners.values())
            and set(state.owner_statuses.values()) == {_OWNER_ACTIVE}
        ):
            owner_fd = self.publish_owner()
            logger.info(
                "Joining shared MPS daemon pid %d on %s (owners: %s)",
                state.daemon_pid,
                self.gpu_uuid,
                sorted(state.owners),
            )
            return MpsLease(
                daemon_pid=state.daemon_pid,
                owner_fd=owner_fd,
                client_tokens=client_tokens,
            )
        else:
            pass
        raise MpsError(self.dirty_state_report(state))

    def publish_owner(self) -> int:
        try:
            owner_fd = os.open(
                self.owner_file,
                os.O_CREAT | os.O_EXCL | os.O_RDWR,
                0o600,
            )
        except FileExistsError as exc:
            raise MpsError(
                f"owner lease {self.owner_file} already exists; refusing to "
                "replace ambiguous state"
            ) from exc
        try:
            fcntl.flock(owner_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.write_owner_status(owner_fd, _OWNER_ACTIVE)
        except OSError:
            os.close(owner_fd)
            self.owner_file.unlink(missing_ok=True)
            raise
        return owner_fd

    @staticmethod
    def write_owner_status(owner_fd: int, status: str) -> None:
        value = f"{status}\n".encode()
        if os.pwrite(owner_fd, value, 0) != len(value):
            raise OSError("short write while updating MPS owner status")
        else:
            pass
        os.ftruncate(owner_fd, len(value))
        os.fsync(owner_fd)

    @staticmethod
    def read_owner_status(owner_file: Path) -> str:
        try:
            status = owner_file.read_text().strip()
        except OSError as exc:
            raise MpsError(
                f"cannot read owner lease status {owner_file}: {exc}"
            ) from exc
        if status not in _OWNER_STATUSES:
            raise MpsError(f"owner lease {owner_file} has invalid status {status!r}")
        else:
            pass
        return status

    def owner_files(self) -> dict[int, Path]:
        if not self.paths.owners_dir.is_dir():
            raise MpsError(f"owner lease directory is missing: {self.paths.owners_dir}")
        else:
            pass
        owners: dict[int, Path] = {}
        for entry in self.paths.owners_dir.iterdir():
            if not entry.name.isdigit() or int(entry.name) <= 0 or not entry.is_file():
                raise MpsError(f"malformed owner lease entry: {entry}")
            else:
                pass
            owners[int(entry.name)] = entry
        return owners

    def inspect_existing_state(self) -> ExistingState:
        state = ExistingState()
        try:
            state.daemon_pid = self.client.read_daemon_identity(self.paths.pipe_dir)
        except MpsControlError as exc:
            state.errors.append(f"daemon identity: {exc}")
        try:
            state.clients = self.client.snapshot(self.paths.pipe_dir)
        except MpsControlError as exc:
            state.errors.append(f"control snapshot: {exc}")
        try:
            owner_files = self.owner_files()
        except MpsError as exc:
            state.errors.append(str(exc))
        else:
            for pid, owner_file in owner_files.items():
                try:
                    state.owners[pid] = self.client.owner_lease_held(owner_file)
                    state.owner_statuses[pid] = self.read_owner_status(owner_file)
                except MpsError as exc:
                    state.errors.append(f"owner lease {pid}: {exc}")
        return state

    def dirty_state_report(self, state: ExistingState) -> str:
        daemon = (
            f"pid {state.daemon_pid} with verified native identity"
            if state.daemon_pid is not None
            else "identity unverified"
        )
        owners = {
            pid: {
                "lock": "held" if held else "dead",
                "status": state.owner_statuses.get(pid, "unknown"),
            }
            for pid, held in sorted(state.owners.items())
        }
        clients = sorted(state.clients) if state.clients is not None else "unavailable"
        details = f"; query errors: {state.errors}" if state.errors else ""
        return (
            f"MPS state dir {self.paths.state_dir} holds dirty state from a previous "
            f"run: daemon {daemon}; owner leases {owners or 'none'}; clients "
            f"{clients}{details}. Refusing to start and preserving all evidence. "
            f"{self.cleanup_guidance(state.clients, owned_clients=set())}"
        )

    def cleanup_guidance(
        self,
        clients: set[MpsClientRef] | None,
        *,
        owned_clients: set[MpsClientRef] | None,
    ) -> str:
        control = (
            f"CUDA_MPS_PIPE_DIRECTORY={shlex.quote(str(self.paths.pipe_dir))} "
            "nvidia-cuda-mps-control"
        )

        def command(value: str) -> str:
            return f"printf '%s\\n' {shlex.quote(value)} | {control}"

        actionable_clients = (
            None if clients is None else clients & (owned_clients or set())
        )
        foreign_clients = (
            set() if clients is None else clients - (owned_clients or set())
        )

        if actionable_clients is None:
            client_steps = (
                "The control snapshot is unavailable, so no safe client command "
                "can be generated. Restore control access before signaling any "
                "possible CUDA client."
            )
        elif actionable_clients:
            client_steps = (
                "Run only these commands for clients proven to belong to this "
                "owner before any forced OS signal:\n  "
                + "\n  ".join(
                    command(f"terminate_client {client.server_pid} {client.client_pid}")
                    for client in sorted(actionable_clients)
                )
            )
        else:
            client_steps = "No current client is proven to belong to this owner."

        if foreign_clients:
            client_steps += (
                f" Other observed refs are not proven to belong to this lease: "
                f"{sorted(foreign_clients)}. Do not terminate them from this report."
            )
        else:
            pass

        current_clients = "unavailable" if clients is None else repr(sorted(clients))
        prefix = (
            f"Current MPS client refs: {current_clients}. After confirming no "
            "workload owned by this serve should remain, clean up in this order. "
            f"{client_steps}\n"
        )
        return prefix + (
            "Stop every remaining workload process, repeat the snapshot and client "
            "termination if needed, and only after every owner lease is unlocked "
            "and a fresh snapshot is empty run:\n  "
            f"{command('quit')}\n  "
            f"rm -rf {shlex.quote(str(self.paths.state_dir))}"
        )

    def rollback_create(self, lease: MpsLease) -> None:
        """Attempt cleanup once under the creation transaction."""

        try:
            remaining_owner_pids = self.check_release_locked(lease)
        except Exception as exc:
            self.persist_dirty_locked(lease, exc)
        self.finish_release_locked(lease, remaining_owner_pids)

    def env_for_stage(self) -> dict[str, str]:
        return {
            "CUDA_MPS_PIPE_DIRECTORY": str(self.paths.pipe_dir),
            "CUDA_MPS_LOG_DIRECTORY": str(self.paths.log_dir),
            "CUDA_VISIBLE_DEVICES": self.gpu_uuid,
        }

    def verify(self, lease: MpsLease) -> set[MpsClientRef]:
        """Gate startup on one current MPS client per managed process."""

        expected_by_token = {
            token: process_name for process_name, token in lease.client_tokens.items()
        }
        missing = set(lease.client_tokens)
        deadline = time.monotonic() + self.verify_timeout
        while True:
            with self.gpu_transaction():
                self.require_live_lease(lease)
                try:
                    self.require_same_daemon(lease)
                    snapshot = self.client.snapshot(self.paths.pipe_dir)
                    attached, observed_tokens, _ = self.classify_clients(
                        snapshot,
                        lease,
                    )
                except MpsRetryableControlError as exc:
                    read_error = exc
                else:
                    read_error = None
                    missing = {
                        expected_by_token[token]
                        for token in expected_by_token.keys() - observed_tokens
                    }
                    if not missing:
                        server_pids = {client.server_pid for client in attached}
                        if len(server_pids) != 1:
                            raise MpsError(
                                f"managed MPS clients must share one server, got "
                                f"{sorted(server_pids)} (pipe dir {self.paths.pipe_dir})"
                            )
                        else:
                            pass
                        (lease.server_pid,) = server_pids
                        return attached
                    else:
                        pass
            self.wait_to_retry(
                deadline,
                f"stage process(es) {sorted(missing)} never attached to the MPS "
                f"server (pipe dir {self.paths.pipe_dir})",
                read_error,
            )

    def retire_clients_for(
        self,
        lease: MpsLease,
        process_name: str,
    ) -> set[MpsClientRef]:
        """Destroy one managed process's CUDA contexts through the daemon.

        # Note (Jiaxin Deng): NVIDIA documents signalling a client that still has
        # work in flight as leaving the MPS server and its other clients in an
        # undefined state, so our own SIGTERM escalation must not reach a client
        # that a colocated serve is sharing a daemon with.
        """

        self.require_live_lease(lease)
        token = lease.client_tokens.get(process_name)
        if token is None:
            return set()
        else:
            pass
        deadline = time.monotonic() + self.stop_timeout
        while True:
            with self.gpu_transaction():
                self.require_live_lease(lease)
                try:
                    self.require_same_daemon(lease)
                    targets = {
                        client
                        for client in self.client.snapshot(self.paths.pipe_dir)
                        if self.client.client_token(client.client_pid) == token
                    }
                except MpsRetryableControlError as exc:
                    read_error = exc
                else:
                    for client in sorted(targets):
                        self.client.terminate_client(self.paths.pipe_dir, client)
                    return targets
            self.wait_to_retry(
                deadline,
                f"MPS client ownership could not be inspected for process {process_name!r}",
                read_error,
            )

    def probe(self, lease: MpsLease) -> str | None:
        """Check only the native status of the startup-verified MPS server."""

        self.require_live_lease(lease)
        if lease.server_pid is None:
            return "MPS server attachment is not verified"
        else:
            pass
        try:
            with self.gpu_transaction():
                status = self.client.get_server_status(
                    self.paths.pipe_dir, lease.server_pid
                )
        except MpsControlError as exc:
            return f"server {lease.server_pid} status query failed: {exc}"
        if status != "ACTIVE":
            return f"server {lease.server_pid} is not ACTIVE: {status!r}"
        else:
            pass
        return None

    def release(
        self,
        lease: MpsLease,
        *,
        clients_could_have_attached: bool = True,
    ) -> None:
        """Release one lease, quitting only as the last owner.

        ``clients_could_have_attached`` may be false only for an acquisition
        rollback that finishes before any managed process can receive this
        manager's environment.
        """

        self.require_live_lease(lease)
        deadline = time.monotonic() + self.drain_timeout
        try:
            while True:
                with self.gpu_transaction():
                    try:
                        remaining_owner_pids = self.check_release_locked(
                            lease,
                            clients_could_have_attached=clients_could_have_attached,
                        )
                    except MpsRetryableControlError as exc:
                        read_error = exc
                    except Exception as exc:
                        self.persist_dirty_locked(lease, exc)
                    else:
                        self.finish_release_locked(lease, remaining_owner_pids)
                        return
                try:
                    self.wait_to_retry(
                        deadline,
                        "MPS release could not confirm clean client ownership",
                        read_error,
                    )
                except MpsError as exc:
                    with self.gpu_transaction():
                        self.persist_dirty_locked(lease, exc)
        except Exception as exc:
            if lease.owner_fd >= 0:
                owner_pid = os.getpid()
                self.abandon_owner(lease)
                raise MpsDirtyStateError(
                    f"MPS cleanup could not persist a retained status under the "
                    f"GPU lock for {self.gpu_uuid}: {exc}. Owner PID {owner_pid} "
                    f"marker {self.owner_file} was left in place with an "
                    f"unconfirmed status and its lock is released; state "
                    f"directory {self.paths.state_dir} is preserved. "
                    f"{self.cleanup_guidance(None, owned_clients=None)}"
                ) from exc
            else:
                raise

    def check_release_locked(
        self,
        lease: MpsLease,
        *,
        clients_could_have_attached: bool = True,
    ) -> set[int]:
        """Inspect release preconditions without changing shared state."""

        self.require_live_lease(lease)
        self.require_same_daemon(lease)
        snapshot = self.client.snapshot(self.paths.pipe_dir)
        if clients_could_have_attached:
            owned_clients, _, unknown_clients = self.classify_clients(snapshot, lease)
            pending_clients = owned_clients | unknown_clients
        else:
            pending_clients = set()

        remaining_owner_pids = {
            pid for pid, path in self.owner_files().items() if path != self.owner_file
        }

        if not remaining_owner_pids:
            pending_clients = snapshot
        else:
            pass
        if pending_clients:
            raise MpsRetryableControlError(
                f"MPS clients {sorted(pending_clients)} remain at shutdown; "
                f"refusing to release this owner's lease or quit daemon "
                f"{lease.daemon_pid}. State preserved: {self.paths.state_dir}"
            )
        else:
            pass

        if (
            remaining_owner_pids
            and snapshot
            and clients_could_have_attached
            and lease.server_pid is None
        ):
            raise MpsError(
                "MPS client ownership is incomplete at shutdown; refusing "
                "to release this owner while a shared daemon still has "
                f"clients. State preserved: {self.paths.state_dir}"
            )
        else:
            pass
        return remaining_owner_pids

    def finish_release_locked(
        self, lease: MpsLease, remaining_owner_pids: set[int]
    ) -> None:
        """Release a checked lease, retaining state on failure under the GPU lock."""

        try:
            if remaining_owner_pids:
                self.drop_owner(lease)
                logger.info(
                    "Leaving shared MPS daemon on %s to owner markers %s",
                    self.gpu_uuid,
                    sorted(remaining_owner_pids),
                )
                return
            else:
                pass

            quit_error: MpsControlError | None = None
            try:
                self.client.quit_daemon(self.paths.pipe_dir)
            except MpsControlError as exc:
                quit_error = exc

            stop_deadline = time.monotonic() + self.stop_timeout
            while True:
                try:
                    alive = self.client.daemon_process_alive(lease.daemon_pid)
                except MpsRetryableControlError as exc:
                    read_error = exc
                else:
                    read_error = None
                    if not alive:
                        break
                    elif quit_error is not None:
                        raise quit_error
                    else:
                        pass
                self.wait_to_retry(
                    stop_deadline,
                    "MPS daemon did not exit after quit",
                    read_error,
                )
            self.drop_owner(lease)
            shutil.rmtree(self.paths.state_dir)
        except Exception as exc:
            self.persist_dirty_locked(lease, exc)

    def persist_dirty_locked(
        self,
        lease: MpsLease,
        error: Exception,
    ) -> NoReturn:
        """Retain failed cleanup evidence and raise under the GPU lock."""

        if lease.owner_fd < 0:
            raise error
        else:
            pass
        owner_pid = os.getpid()
        status_error: Exception | None = None
        try:
            self.require_live_lease(lease)
            self.write_owner_status(lease.owner_fd, _OWNER_RETAINED)
        except (MpsError, OSError) as exc:
            status_error = exc

        observed_daemon_pid: int | None = None
        clients: set[MpsClientRef] | None = None
        owned_clients: set[MpsClientRef] | None = None
        unknown_clients: set[MpsClientRef] | None = None
        query_error: MpsControlError | None = None
        try:
            observed_daemon_pid = self.client.read_daemon_identity(self.paths.pipe_dir)
            clients = self.client.snapshot(self.paths.pipe_dir)
            owned_clients, _, unknown_clients = self.classify_clients(clients, lease)
        except MpsControlError as exc:
            query_error = exc

        guidance = self.cleanup_guidance(
            clients,
            owned_clients=owned_clients,
        )
        self.abandon_owner(lease)
        status = (
            "retained"
            if status_error is None
            else f"unconfirmed because the retained-status write failed: {status_error}"
        )
        observed = (
            str(observed_daemon_pid)
            if observed_daemon_pid is not None
            else f"unavailable ({query_error})"
        )
        snapshot = "unavailable" if clients is None else repr(sorted(clients))
        ownership = (
            f"owned={sorted(owned_clients)}, unattributable={sorted(unknown_clients)}"
            if owned_clients is not None and unknown_clients is not None
            else f"unavailable ({query_error})"
        )
        raise MpsDirtyStateError(
            f"MPS cleanup persisted dirty state for GPU {self.gpu_uuid}: {error}. "
            f"Owner PID {owner_pid} marker {self.owner_file} is {status} and its "
            f"lock is released; state directory {self.paths.state_dir} is preserved. "
            f"Expected daemon PID {lease.daemon_pid}; observed daemon PID {observed}; "
            f"current client ownership {ownership}; current snapshot {snapshot}. "
            f"{guidance}"
        ) from error

    def persist_unidentified_dirty(
        self,
        owner_fd: int,
        error: Exception,
    ) -> NoReturn:
        owner_pid = os.getpid()
        status = "retained"
        try:
            self.write_owner_status(owner_fd, _OWNER_RETAINED)
        except OSError as status_error:
            status = (
                f"unconfirmed because the retained-status write failed: {status_error}"
            )
            logger.error(
                "Could not retain MPS owner marker %s: %s",
                self.owner_file,
                status_error,
            )
        finally:
            os.close(owner_fd)
        raise MpsDirtyStateError(
            f"MPS startup persisted dirty state for GPU {self.gpu_uuid}: {error}. "
            f"Owner PID {owner_pid} marker {self.owner_file} is {status} and its "
            f"lock is released; state directory {self.paths.state_dir} is preserved. "
            "Daemon identity and client snapshot are unavailable. "
            f"{self.cleanup_guidance(None, owned_clients=set())}"
        ) from error

    def classify_clients(
        self,
        clients: set[MpsClientRef],
        lease: MpsLease,
    ) -> tuple[set[MpsClientRef], set[str], set[MpsClientRef]]:
        owned: set[MpsClientRef] = set()
        observed_tokens: set[str] = set()
        unknown: set[MpsClientRef] = set()
        expected_tokens = set(lease.client_tokens.values())
        for client in clients:
            token = self.client.client_token(client.client_pid)
            if token is None:
                unknown.add(client)
            elif token in expected_tokens:
                owned.add(client)
                observed_tokens.add(token)
            else:
                pass
        return owned, observed_tokens, unknown

    def require_live_lease(self, lease: MpsLease) -> None:
        try:
            fd_stat = os.fstat(lease.owner_fd)
            owner_stat = self.owner_file.stat()
        except (OSError, ValueError):
            raise MpsError("operation requires this manager's live MPS lease") from None
        if (fd_stat.st_dev, fd_stat.st_ino) != (owner_stat.st_dev, owner_stat.st_ino):
            raise MpsError("operation requires this manager's live MPS lease")
        else:
            pass

    def require_same_daemon(self, lease: MpsLease) -> None:
        """Require the original daemon identity without adopting a replacement PID."""

        daemon_pid = self.client.read_daemon_identity(self.paths.pipe_dir)
        if daemon_pid != lease.daemon_pid:
            raise MpsError(
                f"MPS daemon identity changed from {lease.daemon_pid} to {daemon_pid}; "
                "owner lease and shared state preserved"
            )
        else:
            pass

    def drop_owner(self, lease: MpsLease) -> None:
        owner_fd = lease.owner_fd
        lease.owner_fd = -1
        self.discard_owner_fd(owner_fd)

    def abandon_owner(self, lease: MpsLease) -> None:
        owner_fd = lease.owner_fd
        lease.owner_fd = -1
        os.close(owner_fd)

    def discard_owner_fd(self, owner_fd: int) -> None:
        os.close(owner_fd)
        self.owner_file.unlink(missing_ok=True)

    def wait_for_snapshot(self, timeout: float, message: str) -> set[MpsClientRef]:
        deadline = time.monotonic() + timeout
        while True:
            try:
                return self.client.snapshot(self.paths.pipe_dir)
            except MpsRetryableControlError as exc:
                self.wait_to_retry(deadline, message, exc)

    def wait_to_retry(
        self,
        deadline: float,
        message: str,
        error: MpsRetryableControlError | None,
    ) -> None:
        """Wait for the next poll, or fail at the deadline with the original cause."""

        remaining = deadline - time.monotonic()
        if remaining > 0:
            time.sleep(min(self.poll_interval, remaining))
        else:
            pass
        if time.monotonic() >= deadline:
            detail = f"; last control error: {error}" if error is not None else ""
            raise MpsError(
                f"{message}{detail}. State dir preserved for inspection: "
                f"{self.paths.state_dir}"
            ) from error
        else:
            pass
