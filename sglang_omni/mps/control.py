# SPDX-License-Identifier: Apache-2.0
"""Strict subprocess and ``/proc`` operations for CUDA MPS lifecycle control."""

from __future__ import annotations

import fcntl
import os
import subprocess
import time
from contextlib import contextmanager
from pathlib import Path

from sglang_omni.mps.manager import (
    MPS_CLIENT_TOKEN_ENV,
    MpsClientRef,
    MpsControlError,
    MpsDaemonNotStartedError,
)

_CONTROL_BINARY = "nvidia-cuda-mps-control"
_QUERY_TIMEOUT_SECONDS = 10
_SNAPSHOT_RETRY_DELAYS_SECONDS = (0.05, 0.15)
_CONTROL_LOCK_NAME = ".control.lock"


def stat_says_alive(stat_text: str) -> bool:
    """Parse ``/proc/<pid>/stat``; the state field follows the last ``)``."""

    fields = stat_text.rsplit(")", 1)[1].split()
    return bool(fields) and fields[0] != "Z"


def parse_pid_list(output: str, command: str) -> list[int]:
    tokens = output.split()
    if any(not token.isdigit() or int(token) <= 0 for token in tokens):
        raise MpsControlError(
            f"unexpected output from {_CONTROL_BINARY} {command!r}: {output!r}"
        )
    else:
        pass
    return [int(token) for token in tokens]


class SubprocessMpsControlClient:
    def control_env(self, pipe_dir: Path) -> dict[str, str]:
        env = os.environ.copy()
        env["CUDA_MPS_PIPE_DIRECTORY"] = str(pipe_dir)
        return env

    @contextmanager
    def _control_transaction(self, pipe_dir: Path):
        lock_path = pipe_dir.parent / _CONTROL_LOCK_NAME
        try:
            lock_file = lock_path.open("a+")
        except OSError as exc:
            raise MpsControlError(
                f"cannot open MPS control lock {lock_path}: {exc}"
            ) from exc

        with lock_file:
            try:
                fcntl.flock(lock_file, fcntl.LOCK_EX)
            except OSError as exc:
                raise MpsControlError(
                    f"cannot lock MPS control transaction {lock_path}: {exc}"
                ) from exc

            yield

    def _query_unlocked(self, pipe_dir: Path, command: str) -> str:
        try:
            result = subprocess.run(
                [_CONTROL_BINARY],
                input=command + "\n",
                capture_output=True,
                text=True,
                timeout=_QUERY_TIMEOUT_SECONDS,
                env=self.control_env(pipe_dir),
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise MpsControlError(
                f"{_CONTROL_BINARY} {command!r} failed: {exc}"
            ) from exc
        if result.returncode != 0:
            raise MpsControlError(
                f"{_CONTROL_BINARY} {command!r} failed "
                f"(rc={result.returncode}): {result.stderr.strip()}"
            )
        else:
            pass
        return result.stdout

    def query(self, pipe_dir: Path, command: str) -> str:
        """Run one serialized control transaction without automatic retries."""

        with self._control_transaction(pipe_dir):
            return self._query_unlocked(pipe_dir, command)

    def start_daemon(self, pipe_dir: Path, log_dir: Path, gpu_uuid: str) -> None:
        env = self.control_env(pipe_dir)
        env["CUDA_MPS_LOG_DIRECTORY"] = str(log_dir)
        # UUID visibility, not ordinal: an ordinal-scoped daemon remaps the
        # client-side ordinals used by examples/mps_dp.
        env["CUDA_VISIBLE_DEVICES"] = gpu_uuid
        try:
            subprocess.run(
                [_CONTROL_BINARY, "-d"],
                check=True,
                capture_output=True,
                timeout=_QUERY_TIMEOUT_SECONDS,
                env=env,
            )
        except OSError as exc:
            raise MpsDaemonNotStartedError(
                f"failed to execute {_CONTROL_BINARY}: {exc}"
            ) from exc
        except subprocess.SubprocessError as exc:
            raise MpsControlError(f"failed to start {_CONTROL_BINARY}: {exc}") from exc

    def read_daemon_identity(self, pipe_dir: Path) -> int:
        """Read and prove the native control-daemon identity for ``pipe_dir``."""

        pid_file = pipe_dir / f"{_CONTROL_BINARY}.pid"
        try:
            raw_pid = pid_file.read_text().strip()
        except OSError as exc:
            raise MpsControlError(
                f"cannot read native PID file {pid_file}: {exc}"
            ) from exc
        if not raw_pid.isdigit() or int(raw_pid) <= 0:
            raise MpsControlError(
                f"native PID file {pid_file} is malformed: {raw_pid!r}"
            )
        else:
            pass
        pid = int(raw_pid)
        if not self.daemon_process_alive(pid):
            raise MpsControlError(f"native PID file {pid_file} names dead pid {pid}")
        else:
            pass

        proc = Path(f"/proc/{pid}")
        try:
            cmdline = proc.joinpath("cmdline").read_bytes().split(b"\0", 1)[0]
            environ = proc.joinpath("environ").read_bytes().split(b"\0")
        except OSError as exc:
            raise MpsControlError(f"cannot inspect daemon pid {pid}: {exc}") from exc
        if Path(os.fsdecode(cmdline)).name != _CONTROL_BINARY:
            raise MpsControlError(
                f"native PID file {pid_file} names {os.fsdecode(cmdline)!r}, not "
                f"{_CONTROL_BINARY}"
            )
        else:
            pass
        expected_pipe = f"CUDA_MPS_PIPE_DIRECTORY={pipe_dir}".encode()
        if expected_pipe not in environ:
            raise MpsControlError(
                f"daemon pid {pid} does not own exact pipe directory {pipe_dir}"
            )
        else:
            pass
        return pid

    def _snapshot_unlocked(self, pipe_dir: Path) -> set[MpsClientRef]:
        servers = parse_pid_list(
            self._query_unlocked(pipe_dir, "get_server_list"), "get_server_list"
        )
        clients: set[MpsClientRef] = set()
        for server_pid in servers:
            command = f"get_client_list {server_pid}"
            client_pids = parse_pid_list(
                self._query_unlocked(pipe_dir, command), command
            )
            for client_pid in client_pids:
                clients.add(MpsClientRef(server_pid, client_pid))
        return clients

    def snapshot(self, pipe_dir: Path) -> set[MpsClientRef]:
        """Return one identity-stable, serialized server/client snapshot."""

        expected_daemon_pid = self.read_daemon_identity(pipe_dir)
        last_error: MpsControlError | None = None
        attempts = len(_SNAPSHOT_RETRY_DELAYS_SECONDS) + 1

        for attempt in range(attempts):
            try:
                with self._control_transaction(pipe_dir):
                    clients = self._snapshot_unlocked(pipe_dir)
            except MpsControlError as exc:
                last_error = exc
            else:
                last_error = None

            current_daemon_pid = self.read_daemon_identity(pipe_dir)
            if current_daemon_pid != expected_daemon_pid:
                identity_error = MpsControlError(
                    "MPS daemon identity changed during control snapshot "
                    f"from {expected_daemon_pid} to {current_daemon_pid}"
                )
                if last_error is not None:
                    raise identity_error from last_error
                raise identity_error
            if last_error is None:
                return clients
            if attempt < len(_SNAPSHOT_RETRY_DELAYS_SECONDS):
                time.sleep(_SNAPSHOT_RETRY_DELAYS_SECONDS[attempt])

        assert last_error is not None
        raise MpsControlError(
            f"MPS control snapshot failed after {attempts} attempts: {last_error}"
        ) from last_error

    def terminate_client(self, pipe_dir: Path, client: MpsClientRef) -> None:
        command = f"terminate_client {client.server_pid} {client.client_pid}"
        output = self.query(pipe_dir, command).strip()
        if output != "0":
            raise MpsControlError(
                f"{_CONTROL_BINARY} {command!r} returned {output!r}, expected '0'"
            )
        else:
            pass

    def quit_daemon(self, pipe_dir: Path) -> None:
        self.query(pipe_dir, "quit")

    def daemon_process_alive(self, pid: int) -> bool:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError as exc:
            raise MpsControlError(f"cannot probe daemon pid {pid}: {exc}") from exc
        try:
            return stat_says_alive(Path(f"/proc/{pid}/stat").read_text())
        except FileNotFoundError:
            return False
        except (OSError, IndexError) as exc:
            raise MpsControlError(f"cannot inspect daemon pid {pid}: {exc}") from exc

    def client_token(self, pid: int) -> str | None:
        try:
            entries = Path(f"/proc/{pid}/environ").read_bytes().split(b"\0")
        except FileNotFoundError:
            return None
        except OSError as exc:
            raise MpsControlError(f"cannot inspect client pid {pid}: {exc}") from exc
        prefix = f"{MPS_CLIENT_TOKEN_ENV}=".encode()
        values = [entry[len(prefix) :] for entry in entries if entry.startswith(prefix)]
        if not values:
            return None
        else:
            pass
        if len(values) != 1 or not values[0]:
            raise MpsControlError(
                f"client pid {pid} has malformed {MPS_CLIENT_TOKEN_ENV}"
            )
        else:
            pass
        try:
            return values[0].decode("ascii")
        except UnicodeDecodeError as exc:
            raise MpsControlError(
                f"client pid {pid} has non-ASCII {MPS_CLIENT_TOKEN_ENV}"
            ) from exc

    def owner_lease_held(self, lease_file: Path) -> bool:
        try:
            with lease_file.open("r+") as probe:
                try:
                    fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    return True
                fcntl.flock(probe, fcntl.LOCK_UN)
                return False
        except OSError as exc:
            raise MpsControlError(
                f"cannot inspect owner lease {lease_file}: {exc}"
            ) from exc
