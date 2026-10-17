# SPDX-License-Identifier: Apache-2.0
"""Decode and resample Qwen3-ASR uploads in worker processes."""

from __future__ import annotations

import logging
import os
import pickle
import queue
import struct
import subprocess
import sys
import threading
from typing import BinaryIO, Protocol

import torch

from sglang_omni.preprocessing.transcription import PreparedAudio, prepare_audio
from sglang_omni.proto import StagePayload

logger = logging.getLogger(__name__)

MESSAGE_LENGTH_HEADER = struct.Struct("<Q")
# The worker keeps the original stdout descriptor for protocol frames and
# points descriptor 1 at stderr before any import can print to it; it asks
# for SIGKILL when its parent exits, after exec, where prctl is safe.
WORKER_BOOTSTRAP = (
    "import ctypes, os, signal, sys\n"
    "ctypes.CDLL(None).prctl(1, signal.SIGKILL, 0, 0, 0)\n"
    "if os.getppid() != int(sys.argv[1]):\n"
    "    sys.exit(0)\n"
    "protocol_out = os.fdopen(os.dup(1), 'wb')\n"
    "os.dup2(2, 1)\n"
    "sys.stdout = sys.stderr\n"
    "from sglang_omni.models.qwen3_asr.audio_workers import worker_main\n"
    "worker_main(sys.stdin.buffer, protocol_out)\n"
)


class AudioPreparer(Protocol):
    def __call__(
        self, payload: StagePayload, *, source_name: str, target_sample_rate: int
    ) -> PreparedAudio: ...


class WorkerBrokenError(Exception):
    """The worker process or its pipe can no longer serve uploads."""


def write_message(stream: BinaryIO, message: object) -> None:
    encoded_message = pickle.dumps(message, protocol=pickle.HIGHEST_PROTOCOL)
    stream.write(MESSAGE_LENGTH_HEADER.pack(len(encoded_message)))
    stream.write(encoded_message)
    stream.flush()


def read_message(stream: BinaryIO) -> object:
    header = stream.read(MESSAGE_LENGTH_HEADER.size)
    if len(header) < MESSAGE_LENGTH_HEADER.size:
        raise EOFError("audio worker pipe closed")
    else:
        pass
    (message_length,) = MESSAGE_LENGTH_HEADER.unpack(header)
    encoded_message = stream.read(message_length)
    if len(encoded_message) < message_length:
        raise EOFError("audio worker pipe closed")
    else:
        pass
    return pickle.loads(encoded_message)


def worker_main(protocol_in: BinaryIO, protocol_out: BinaryIO) -> None:
    torch.set_num_threads(1)
    write_message(protocol_out, ("ready", None))
    while True:
        try:
            payload, source_name, target_sample_rate = read_message(protocol_in)
        except EOFError:
            return
        try:
            prepared = prepare_audio(
                payload, source_name=source_name, target_sample_rate=target_sample_rate
            )
        except Exception:
            # The caller re-runs the call in-process to raise the exact error.
            write_message(protocol_out, ("error", None))
        else:
            write_message(protocol_out, ("ok", prepared))


class AudioWorker:
    def __init__(self) -> None:
        self.process = subprocess.Popen(
            [sys.executable, "-c", WORKER_BOOTSTRAP, str(os.getpid())],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            env={**os.environ, "OMP_NUM_THREADS": "1"},
        )

    def wait_ready(self) -> None:
        try:
            status, _ = read_message(self.process.stdout)
        except (EOFError, OSError, pickle.UnpicklingError) as exc:
            raise WorkerBrokenError("audio worker exited during startup") from exc
        if status != "ready":
            raise WorkerBrokenError("audio worker failed to start")
        else:
            pass

    def run(
        self, payload: StagePayload, source_name: str, target_sample_rate: int
    ) -> PreparedAudio | None:
        """Return the prepared audio, or None when prepare_audio itself failed."""
        try:
            write_message(
                self.process.stdin, (payload, source_name, target_sample_rate)
            )
            status, prepared = read_message(self.process.stdout)
        except Exception as exc:
            raise WorkerBrokenError("audio worker stopped responding") from exc
        if status == "ok":
            return prepared
        else:
            return None

    def close(self) -> None:
        try:
            self.process.stdin.close()
        except OSError:
            pass
        try:
            self.process.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait()
        self.process.stdout.close()


class AudioWorkerPool:
    """Hands each build thread an idle worker for one upload."""

    def __init__(self, worker_count: int) -> None:
        self.lock = threading.Lock()
        self.workers: list[AudioWorker] = []
        self.idle_workers: queue.SimpleQueue[AudioWorker] = queue.SimpleQueue()
        self.closed = False
        self.serving_worker_count = 0
        try:
            for _ in range(worker_count):
                self.workers.append(AudioWorker())
            for worker in self.workers:
                worker.wait_ready()
        except Exception:
            self.close()
            raise
        for worker in self.workers:
            self.idle_workers.put(worker)
        self.serving_worker_count = len(self.workers)

    def checkout(self) -> AudioWorker | None:
        """Wait for an idle worker; None once no worker is left to wait for."""
        while True:
            with self.lock:
                if self.serving_worker_count <= 0 or self.closed:
                    return None
                else:
                    pass
            try:
                return self.idle_workers.get(timeout=0.05)
            except queue.Empty:
                continue

    def prepare_audio(
        self, payload: StagePayload, *, source_name: str, target_sample_rate: int
    ) -> PreparedAudio:
        worker = self.checkout()
        if worker is None:
            prepared = None
        else:
            try:
                prepared = worker.run(payload, source_name, target_sample_rate)
            except WorkerBrokenError:
                self.retire(worker)
                prepared = None
            else:
                self.idle_workers.put(worker)
        if prepared is None:
            # Failed or unserved uploads re-run here, so errors match the
            # in-process path exactly.
            return prepare_audio(
                payload, source_name=source_name, target_sample_rate=target_sample_rate
            )
        else:
            return prepared

    def retire(self, worker: AudioWorker) -> None:
        logger.warning(
            f"Audio worker pid={worker.process.pid} stopped; preparing uploads in-process"
        )
        with self.lock:
            self.serving_worker_count -= 1
        worker.close()

    def close(self) -> None:
        with self.lock:
            self.closed = True
            self.serving_worker_count = 0
        for worker in self.workers:
            worker.close()
