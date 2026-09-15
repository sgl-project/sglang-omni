# SPDX-License-Identifier: Apache-2.0
"""Strict subprocess and ``/proc`` operations for CUDA MPS lifecycle control."""

from __future__ import annotations

import fcntl
import os
import subprocess
from contextlib import contextmanager
from pathlib import Path

from sglang_omni.mps.manager import (
    MPS_CLIENT_TOKEN_ENV,
    MpsClientRef,
    MpsControlError,
    MpsDaemonNotStartedError,
    MpsProcessIdentity,
)

_CONTROL_BINARY = "nvidia-cuda-mps-control"
_SERVER_BINARY = "nvidia-cuda-mps-server"
_QUERY_TIMEOUT_SECONDS = 10
_CONTROL_LOCK_NAME = ".control.lock"


def _parse_proc_stat(stat_text: str) -> tuple[str, int]:
    try:
        fields = stat_text.rsplit(")", 1)[1].split()
        return fields[0], int(fields[19])
    except (IndexError, ValueError) as exc:
        raise ValueError("malformed /proc/<pid>/stat") from exc


def _parse_pid_list(output: str, command: str) -> list[int]:
    tokens = output.split()
    if any(not token.isdigit() or int(token) <= 0 for token in tokens):
        raise MpsControlError(
            f"unexpected output from {_CONTROL_BINARY} {command!r}: {output!r}"
        )
    return [int(token) for token in tokens]


class SubprocessMpsControlClient:
    def _control_env(self, pipe_dir: Path) -> dict[str, str]:
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
                env=self._control_env(pipe_dir),
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
        return result.stdout

    def _query(self, pipe_dir: Path, command: str) -> str:
        with self._control_transaction(pipe_dir):
            return self._query_unlocked(pipe_dir, command)

    def start_daemon(self, pipe_dir: Path, log_dir: Path, gpu_uuid: str) -> None:
        env = self._control_env(pipe_dir)
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

    @staticmethod
    def _read_proc_stat(pid: int) -> tuple[str, int]:
        try:
            return _parse_proc_stat(Path(f"/proc/{pid}/stat").read_text())
        except FileNotFoundError as exc:
            raise MpsControlError(f"process pid {pid} does not exist") from exc
        except (OSError, ValueError) as exc:
            raise MpsControlError(f"cannot inspect process pid {pid}: {exc}") from exc

    def _read_process_identity(
        self,
        pid: int,
        pipe_dir: Path,
        expected_binary: str,
    ) -> MpsProcessIdentity:
        proc = Path(f"/proc/{pid}")
        try:
            state, starttime = self._read_proc_stat(pid)
            if state == "Z":
                raise MpsControlError(f"process pid {pid} is a zombie")
            cmdline = proc.joinpath("cmdline").read_bytes().split(b"\0", 1)[0]
            environ = proc.joinpath("environ").read_bytes().split(b"\0")
            final_state, final_starttime = self._read_proc_stat(pid)
        except OSError as exc:
            raise MpsControlError(
                f"cannot inspect {expected_binary} pid {pid}: {exc}"
            ) from exc
        if final_state == "Z":
            raise MpsControlError(
                f"process pid {pid} became a zombie during inspection"
            )
        if final_starttime != starttime:
            raise MpsControlError(
                f"process pid {pid} changed identity during inspection"
            )
        executable = Path(os.fsdecode(cmdline)).name
        if executable != expected_binary:
            raise MpsControlError(
                f"process pid {pid} names {os.fsdecode(cmdline)!r}, not "
                f"{expected_binary}"
            )
        expected_pipe = f"CUDA_MPS_PIPE_DIRECTORY={pipe_dir}".encode()
        if expected_pipe not in environ:
            raise MpsControlError(
                f"process pid {pid} does not own exact pipe directory {pipe_dir}"
            )
        return MpsProcessIdentity(pid=pid, starttime=starttime)

    def read_daemon_process_identity(self, pipe_dir: Path) -> MpsProcessIdentity:
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

        return self._read_process_identity(
            int(raw_pid),
            pipe_dir,
            _CONTROL_BINARY,
        )

    def read_server_process_identity(
        self,
        pipe_dir: Path,
        pid: int,
    ) -> MpsProcessIdentity:
        return self._read_process_identity(pid, pipe_dir, _SERVER_BINARY)

    def _snapshot_unlocked(self, pipe_dir: Path) -> set[MpsClientRef]:
        servers = _parse_pid_list(
            self._query_unlocked(pipe_dir, "get_server_list"), "get_server_list"
        )
        clients: set[MpsClientRef] = set()
        for server_pid in servers:
            command = f"get_client_list {server_pid}"
            client_pids = _parse_pid_list(
                self._query_unlocked(pipe_dir, command), command
            )
            for client_pid in client_pids:
                clients.add(MpsClientRef(server_pid, client_pid))
        return clients

    def snapshot(self, pipe_dir: Path) -> set[MpsClientRef]:
        expected_identity = self.read_daemon_process_identity(pipe_dir)
        with self._control_transaction(pipe_dir):
            clients = self._snapshot_unlocked(pipe_dir)
        current_identity = self.read_daemon_process_identity(pipe_dir)
        if current_identity != expected_identity:
            raise MpsControlError(
                "MPS daemon identity changed during control snapshot "
                f"from pid {expected_identity.pid} "
                f"starttime {expected_identity.starttime} to pid "
                f"{current_identity.pid} starttime {current_identity.starttime}"
            )
        return clients

    def terminate_client(self, pipe_dir: Path, client: MpsClientRef) -> None:
        command = f"terminate_client {client.server_pid} {client.client_pid}"
        output = self._query(pipe_dir, command).strip()
        if output != "0":
            raise MpsControlError(
                f"{_CONTROL_BINARY} {command!r} returned {output!r}, expected '0'"
            )

    def quit_daemon(self, pipe_dir: Path) -> None:
        self._query(pipe_dir, "quit")

    def daemon_process_alive(self, pid: int) -> bool:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError as exc:
            raise MpsControlError(f"cannot probe daemon pid {pid}: {exc}") from exc
        try:
            state, _ = _parse_proc_stat(Path(f"/proc/{pid}/stat").read_text())
            return state != "Z"
        except FileNotFoundError:
            return False
        except (OSError, ValueError) as exc:
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
        if len(values) != 1 or not values[0]:
            raise MpsControlError(
                f"client pid {pid} has malformed {MPS_CLIENT_TOKEN_ENV}"
            )
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
