# SPDX-License-Identifier: Apache-2.0
"""Session-scoped access to the D2 GPU worker in its own Python environment."""

import socket
import subprocess
import threading
from multiprocessing.connection import Connection
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from sglang_omni.admission import ContextExhaustedError


class D2Frame(BaseModel):
    """Playback and Thinker output for one native frame."""

    model_config = ConfigDict(val_json_bytes="base64")

    pcm16: bytes
    text_delta: str
    sequence: int
    is_interruption: bool


class WorkerReply(BaseModel):
    frames: list[D2Frame] = Field(default_factory=list)
    error: str | None = None


class D2Runtime:
    """Keep a single resident model and serialize access to its captured graphs."""

    def __init__(
        self,
        model_path: str,
        device: str,
        runtime_python: str,
        kv_budget: int,
        seed: int,
    ) -> None:
        self.operation_lock = threading.Lock()
        stage_socket, worker_socket = socket.socketpair()
        with stage_socket, worker_socket:
            self.process = subprocess.Popen(
                [
                    runtime_python,
                    "-I",
                    str(Path(__file__).with_name("worker.py")),
                    str(worker_socket.fileno()),
                    model_path,
                    device,
                    str(kv_budget),
                    str(seed),
                ],
                pass_fds=(worker_socket.fileno(),),
            )
            self.connection = Connection(stage_socket.detach())
        try:
            self.receive_reply()
        except BaseException:
            self.shutdown()
            raise

    def receive_reply(self) -> list[D2Frame]:
        try:
            reply = WorkerReply.model_validate_json(self.connection.recv_bytes())
        except (EOFError, ConnectionResetError) as error:
            raise RuntimeError(
                "The D2 worker exited. Check the stage logs and runtime_python environment."
            ) from error
        if reply.error is None:
            return reply.frames
        elif reply.error.startswith(f"{ContextExhaustedError.CODE}:"):
            raise ContextExhaustedError(reply.error)
        else:
            raise RuntimeError(reply.error)

    def execute(self, command: bytes) -> list[D2Frame]:
        with self.operation_lock:
            self.connection.send_bytes(command)
            return self.receive_reply()

    def open_session(self) -> None:
        self.execute(b"open")

    def step(self, pcm16: bytes) -> list[D2Frame]:
        return self.execute(b"pcm:" + pcm16)

    def close_session(self) -> None:
        self.execute(b"close")

    def shutdown(self) -> None:
        with self.operation_lock:
            self.connection.close()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
