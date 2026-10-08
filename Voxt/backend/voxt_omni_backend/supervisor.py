# SPDX-License-Identifier: Apache-2.0
"""Owns one local Qwen3-ASR MLX server for Voxt: launch, verify, report, reap.

Events go to stdout as JSON lines. Voxt holds this process's stdin open; a
shutdown command, end-of-file (Voxt exited or crashed) or a termination signal
stops the server and every process it started, at any point including startup.
The server runs under the lifeline runner, so it also dies if this supervisor
is killed outright. Logs carry state and timing only.
"""

from __future__ import annotations

import argparse
import http.client
import json
import logging
import os
import queue
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass
from pathlib import Path
from types import FrameType
from typing import Literal

import psutil

LOOPBACK_HOST = "127.0.0.1"
ModelKind = Literal["qwen3_asr"]
MODEL_KINDS: tuple[ModelKind, ...] = ("qwen3_asr",)
# sglang-omni's standalone MLX server: one process, the model loaded in it.
QWEN_SERVER_MODULE = "sglang_omni_mlx.qwen3_asr.server"
# Voxt's Swift Qwen live session decodes on its first 100 ms feed, then once a
# second (StreamingConfig default).
QWEN_REALTIME_OPTIONS = (
    "--decode-interval-ms",
    "1000",
    "--first-decode-ms",
    "100",
)
HEALTH_POLL_INTERVAL_S = 0.1
HTTP_PROBE_TIMEOUT_S = 1.0
TERMINATE_GRACE_S = 5.0
# A server stopped before it was ready has no in-flight work to finish.
STARTUP_TERMINATE_GRACE_S = 1.0
CONTROL_POLL_INTERVAL_S = 0.1
TREE_REFRESH_INTERVAL_S = 1.0
STOP_SIGNALS = (signal.SIGTERM, signal.SIGINT, signal.SIGHUP, signal.SIGQUIT)

logger = logging.getLogger("voxt_omni_backend.supervisor")
loopback_opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
ControlMessage = Literal["shutdown", "closed", "signal"]
CONTROL_MESSAGES: tuple[ControlMessage, ...] = ("shutdown", "closed", "signal")


@dataclass(kw_only=True, frozen=True)
class ServerLaunch:
    command: list[str]
    environment: dict[str, str]
    model_name: str
    port: int


def emit(event: dict[str, object]) -> None:
    try:
        sys.stdout.write(json.dumps(event) + "\n")
        sys.stdout.flush()
    except BrokenPipeError:
        logger.info("event pipe closed")


def free_loopback_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind((LOOPBACK_HOST, 0))
        return int(probe.getsockname()[1])


def get_json(url: str) -> dict[str, object] | None:
    try:
        with loopback_opener.open(url, timeout=HTTP_PROBE_TIMEOUT_S) as response:
            decoded = json.loads(response.read())
    except (
        urllib.error.URLError,
        http.client.HTTPException,
        ConnectionError,
        TimeoutError,
        ValueError,
    ):
        return None
    if isinstance(decoded, dict):
        return decoded
    else:
        return None


class OwnedProcessTree:
    """The server and every descendant seen so far, keyed by pid and start time.

    A descendant reparented after the server dies is still recognised, and a
    recycled pid with a different start time is never signalled. The root is
    only walked while the supervisor's own child handle says it is running.
    """

    def __init__(self, process: subprocess.Popen[bytes]) -> None:
        self.process = process
        self.start_times_by_pid: dict[int, float] = {}
        self.refreshed_at = 0.0
        self.root_start_time = psutil.Process(process.pid).create_time()
        self.refresh()

    def refresh(self) -> None:
        if self.process.poll() is not None:
            return
        else:
            pass
        try:
            root = psutil.Process(self.process.pid)
            members = [root, *root.children(recursive=True)]
        except psutil.NoSuchProcess:
            members = []
        for member in members:
            try:
                self.start_times_by_pid[member.pid] = member.create_time()
            except psutil.NoSuchProcess:
                pass
        self.refreshed_at = time.monotonic()

    def refresh_periodically(self) -> None:
        if time.monotonic() - self.refreshed_at >= TREE_REFRESH_INTERVAL_S:
            self.refresh()
        else:
            pass

    def session_members(self) -> list[psutil.Process]:
        """Processes still in the server's session, even after its leader exited.

        The server leads a new session, so its id is the server pid; it cannot
        be reused while any member lives, and members started before the
        server are excluded.
        """
        members: list[psutil.Process] = []
        for candidate in psutil.process_iter():
            try:
                if (
                    os.getsid(candidate.pid) == self.process.pid
                    and candidate.create_time() >= self.root_start_time
                ):
                    members.append(candidate)
                else:
                    pass
            except (ProcessLookupError, PermissionError, psutil.NoSuchProcess):
                pass
        return members

    def live_members(self) -> list[psutil.Process]:
        live: list[psutil.Process] = []
        for pid, start_time in self.start_times_by_pid.items():
            try:
                member = psutil.Process(pid)
                if member.create_time() == start_time:
                    live.append(member)
                else:
                    pass
            except psutil.NoSuchProcess:
                pass
        return live

    def reap(self, grace_s: float = TERMINATE_GRACE_S) -> None:
        """Stop the server and its descendants; no other process is signalled.

        Safe to call repeatedly: members that already exited are skipped.
        """
        self.refresh()
        members = list(
            {
                member.pid: member
                for member in self.live_members() + self.session_members()
            }.values()
        )
        for member in members:
            try:
                member.terminate()
            except psutil.NoSuchProcess:
                pass
        _gone, alive = psutil.wait_procs(members, timeout=grace_s)
        for survivor in alive:
            try:
                survivor.kill()
            except psutil.NoSuchProcess:
                pass
        psutil.wait_procs(alive, timeout=TERMINATE_GRACE_S)
        self.process.poll()


class ControlChannel:
    """Shutdown requests from Voxt's pipe and from termination signals."""

    def __init__(self) -> None:
        # SimpleQueue: put() is reentrant, so the signal handler cannot deadlock
        # on a mutex the main thread holds inside get().
        self.messages: queue.SimpleQueue[ControlMessage] = queue.SimpleQueue()

    def start(self) -> None:
        threading.Thread(target=self.read_pipe, daemon=True).start()
        for stop_signal in STOP_SIGNALS:
            signal.signal(stop_signal, self.on_signal)

    def read_pipe(self) -> None:
        for line in sys.stdin:
            try:
                command = json.loads(line).get("command")
            except (ValueError, AttributeError):
                logger.warning("ignored malformed control line")
                continue
            if command == "shutdown":
                self.messages.put("shutdown")
            else:
                logger.warning("ignored unknown control command")
        self.messages.put("closed")

    def on_signal(self, signal_number: int, frame: FrameType | None) -> None:
        del frame
        self.messages.put("signal")

    def poll(self, timeout_s: float) -> ControlMessage | None:
        try:
            return self.messages.get(timeout=timeout_s)
        except queue.Empty:
            return None


def ready_failure(
    owned_tree: OwnedProcessTree,
    control: ControlChannel,
    launch: ServerLaunch,
    deadline: float,
) -> str | None:
    """None once our own instance answers; a control message or a failure otherwise."""
    base_url = f"http://{LOOPBACK_HOST}:{launch.port}"
    while time.monotonic() < deadline:
        owned_tree.refresh_periodically()
        exit_code = owned_tree.process.poll()
        if exit_code is not None:
            return f"server exited with code {exit_code} before it was ready"
        else:
            pass
        health = get_json(f"{base_url}/health")
        models = get_json(f"{base_url}/v1/models") if health is not None else None
        if health is not None and health.get("status") == "healthy" and models:
            served = models.get("data")
            served_names = (
                [entry.get("id") for entry in served if isinstance(entry, dict)]
                if isinstance(served, list)
                else []
            )
            if launch.model_name in served_names:
                return None
            else:
                return "a different server instance answered on the launch port"
        else:
            pass
        message = control.poll(HEALTH_POLL_INTERVAL_S)
        if message is not None:
            return message
        else:
            pass
    return "startup timeout"


def server_launch(arguments: argparse.Namespace) -> ServerLaunch:
    model_kind: ModelKind = arguments.model_kind
    model_name = f"voxt-{model_kind}-{uuid.uuid4().hex[:12]}"
    port = free_loopback_port()
    if arguments.server_command:
        base_command = list(json.loads(arguments.server_command))
    else:
        base_command = [
            sys.executable,
            "-m",
            QWEN_SERVER_MODULE,
            *QWEN_REALTIME_OPTIONS,
        ]
    environment = dict(os.environ)
    environment.update(
        {
            # The lifeline watches this pid, not whatever its parent is by the
            # time it runs (launchd, if the supervisor was already killed).
            "VOXT_OMNI_SUPERVISOR_PID": str(os.getpid()),
            "NO_PROXY": f"{LOOPBACK_HOST},localhost",
            "no_proxy": f"{LOOPBACK_HOST},localhost",
        }
    )
    return ServerLaunch(
        command=[
            sys.executable,
            "-m",
            "voxt_omni_backend.lifeline",
            "--",
            *base_command,
            "--model-path",
            arguments.model_directory,
            "--model-name",
            model_name,
            "--host",
            LOOPBACK_HOST,
            "--port",
            str(port),
        ],
        environment=environment,
        model_name=model_name,
        port=port,
    )


def run(arguments: argparse.Namespace, control: ControlChannel) -> int:
    launch = server_launch(arguments)
    queued = control.poll(0)
    if queued is not None:
        # Voxt gave up on this launch before the server was even started.
        if queued == "shutdown":
            emit({"event": "stopped"})
        else:
            pass
        return 0
    else:
        pass
    log_directory = Path(arguments.derived_root) / "omni-logs"
    log_directory.mkdir(parents=True, exist_ok=True)
    started_s = time.monotonic()
    # A stop signal between spawning and owning the server must not leak it.
    previous_mask = signal.pthread_sigmask(signal.SIG_BLOCK, STOP_SIGNALS)
    try:
        with open(log_directory / f"{launch.model_name}.log", "wb") as server_log:
            process = subprocess.Popen(
                launch.command,
                env=launch.environment,
                stdin=subprocess.DEVNULL,
                stdout=server_log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        owned_tree = OwnedProcessTree(process)
    finally:
        signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask)
    logger.info(f"server started pid={process.pid} port={launch.port}")
    try:
        outcome = ready_failure(
            owned_tree,
            control,
            launch,
            started_s + float(arguments.startup_timeout_s),
        )
        if outcome in CONTROL_MESSAGES:
            owned_tree.reap(STARTUP_TERMINATE_GRACE_S)
            message = outcome
        elif outcome is not None:
            emit({"event": "failed", "reason": outcome})
            return 1
        else:
            emit(
                {
                    "event": "ready",
                    "host": LOOPBACK_HOST,
                    "port": launch.port,
                    "model_name": launch.model_name,
                    "server_pid": process.pid,
                    "startup_s": round(time.monotonic() - started_s, 3),
                }
            )
            message = None
        while message is None:
            owned_tree.refresh_periodically()
            exit_code = process.poll()
            if exit_code is not None:
                emit({"event": "exited", "exit_code": exit_code})
                return 1
            else:
                pass
            message = control.poll(CONTROL_POLL_INTERVAL_S)
        owned_tree.reap()
        if message == "shutdown":
            emit({"event": "stopped"})
        else:
            logger.info(f"stopped after control message {message}")
        return 0
    finally:
        # A second signal must not interrupt cleanup before escalation.
        signal.pthread_sigmask(signal.SIG_BLOCK, STOP_SIGNALS)
        owned_tree.reap()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-kind", choices=MODEL_KINDS, required=True)
    parser.add_argument("--model-directory", required=True)
    parser.add_argument("--derived-root", required=True)
    parser.add_argument("--startup-timeout-s", type=float, default=180.0)
    parser.add_argument(
        "--server-command",
        default=None,
        help="JSON argv replacing the server module, for tests",
    )
    arguments = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, stream=sys.stderr, format="%(asctime)s %(message)s"
    )
    control = ControlChannel()
    control.start()
    try:
        return run(arguments, control)
    except (OSError, ValueError) as error:
        emit({"event": "failed", "reason": f"{type(error).__name__}: {error}"})
        return 1


if __name__ == "__main__":
    sys.exit(main())
