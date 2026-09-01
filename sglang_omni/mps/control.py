# SPDX-License-Identifier: Apache-2.0
"""Strict subprocess and ``/proc`` operations for CUDA MPS lifecycle control."""

from __future__ import annotations

import fcntl
import os
import subprocess
from pathlib import Path

from sglang_omni.mps.manager import (
    MPS_CLIENT_TOKEN_ENV,
    MpsClientRef,
    MpsControlError,
    MpsDaemonNotStartedError,
)

_CONTROL_BINARY = "nvidia-cuda-mps-control"
_QUERY_TIMEOUT_SECONDS = 10


def _stat_says_alive(stat_text: str) -> bool:
    """Parse ``/proc/<pid>/stat``; the state field follows the last ``)``."""

    fields = stat_text.rsplit(")", 1)[1].split()
    return bool(fields) and fields[0] != "Z"


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

    def _query(self, pipe_dir: Path, command: str) -> str:
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
        pid = int(raw_pid)
        if not self.daemon_process_alive(pid):
            raise MpsControlError(f"native PID file {pid_file} names dead pid {pid}")

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
        expected_pipe = f"CUDA_MPS_PIPE_DIRECTORY={pipe_dir}".encode()
        if expected_pipe not in environ:
            raise MpsControlError(
                f"daemon pid {pid} does not own exact pipe directory {pipe_dir}"
            )
        return pid

    def snapshot(self, pipe_dir: Path) -> set[MpsClientRef]:
        """Return one strict server/client snapshot from the selected daemon."""

        servers = _parse_pid_list(
            self._query(pipe_dir, "get_server_list"), "get_server_list"
        )
        clients: set[MpsClientRef] = set()
        for server_pid in servers:
            command = f"get_client_list {server_pid}"
            client_pids = _parse_pid_list(self._query(pipe_dir, command), command)
            for client_pid in client_pids:
                clients.add(MpsClientRef(server_pid, client_pid))
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
            return _stat_says_alive(Path(f"/proc/{pid}/stat").read_text())
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
