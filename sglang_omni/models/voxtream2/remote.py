# SPDX-License-Identifier: Apache-2.0
"""One persistent model process with its own Python dependency environment."""

from __future__ import annotations

import atexit
import base64
import math
import os
import queue
import subprocess
import threading
import time
import uuid
from pathlib import Path
from typing import Generator

import numpy as np
from numpy.typing import NDArray

from sglang_omni.models.voxtream2.wire import PROTOCOL_VERSION, read_frame, write_frame
from sglang_omni.proto import StagePayload


class RemoteVoXtreamBackend:
    """Serial request adapter; no model packages are imported in this process.

    Only the scheduler thread consumes responses. The reader holds at most one
    queued frame; OS pipes provide backpressure to the worker. Cancellation is
    acknowledged after generator cleanup, before another request is admitted.
    """

    def __init__(
        self,
        *,
        python_executable: str,
        model_config: dict[str, str | bool],
        source_paths: tuple[str, ...] | list[str],
        startup_timeout: float,
        response_timeout: float,
        cancel_timeout: float,
        worker_command: list[str] | None = None,
    ) -> None:
        if any(
            not math.isfinite(t) or t <= 0
            for t in (startup_timeout, response_timeout, cancel_timeout)
        ):
            raise ValueError("Worker timeouts must be positive")
        else:
            pass
        python = Path(python_executable)
        if not python.is_absolute() or not python.is_file():
            raise ValueError("model_python must name an existing absolute Python path")
        else:
            pass
        self.response_timeout_seconds = response_timeout
        self.cancel_timeout_seconds = cancel_timeout
        self.closed = threading.Event()
        self.responses = queue.Queue(maxsize=1)
        self.write_lock = threading.Lock()
        self.state_lock = threading.Lock()
        self.close_lock = threading.Lock()
        self.active_request = None
        self.healthy = True
        env = dict(os.environ)
        for key in ("PYTHONHOME", "PYTHONPATH", "CUDA_HOME", "CUDA_PATH"):
            env.pop(key, None)
        env["PYTHONNOUSERSITE"] = "1"
        env["PYTHONPATH"] = os.pathsep.join(
            [str(Path(__file__).resolve().parents[3]), *map(str, source_paths)]
        )
        # Keep system audio libraries, but not another venv's CUDA libraries.
        env["LD_LIBRARY_PATH"] = os.pathsep.join(
            p
            for p in env.get("LD_LIBRARY_PATH", "").split(os.pathsep)
            if p and "site-packages" not in p
        )
        command = worker_command or [
            str(python),
            "-u",
            "-m",
            "sglang_omni.models.voxtream2.worker_main",
        ]
        self.process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            # Inherit stderr so diagnostics cannot fill an unread private pipe.
            stderr=None,
            env=env,
        )
        self.reader = threading.Thread(
            target=self.read_loop, name="voxtream-worker-output", daemon=True
        )
        self.reader.start()
        try:
            self.send_frame({"type": "init", "config": model_config})
            ready = self.receive_frame(startup_timeout)
            if ready.get("type") != "ready" or ready.get("version") != PROTOCOL_VERSION:
                raise RuntimeError(f"VoXtream2 worker did not initialize: {ready}")
            else:
                pass
            self.sample_rate = int(ready["sample_rate"])
            self.max_samples = int(ready["max_samples"])
            if self.sample_rate <= 0 or self.max_samples <= 0:
                raise RuntimeError("VoXtream2 worker returned invalid audio limits")
            else:
                pass
            self.runtime = ready.get("runtime", {})
        except BaseException:
            self.close()
            raise
        atexit.register(self.close)

    def queue_response(self, value: dict[str, object] | Exception) -> None:
        while not self.closed.is_set():
            try:
                self.responses.put(value, timeout=0.05)
                return
            except queue.Full:
                pass

    def read_loop(self) -> None:
        try:
            while not self.closed.is_set():
                self.queue_response(read_frame(self.process.stdout))
        except Exception as exc:
            self.queue_response(exc)

    def send_frame(self, value: dict[str, object]) -> None:
        if self.closed.is_set() or not self.healthy:
            raise RuntimeError("VoXtream2 model worker is unavailable")
        else:
            pass
        with self.write_lock:
            write_frame(self.process.stdin, value)

    def receive_frame(self, timeout: float) -> dict[str, object]:
        try:
            value = self.responses.get(timeout=max(timeout, 0.001))
        except queue.Empty as exc:
            raise TimeoutError("VoXtream2 model worker response timed out") from exc
        if isinstance(value, Exception):
            raise RuntimeError("VoXtream2 model worker transport failed") from value
        else:
            pass
        return value

    def cancel(self, request_id: str) -> None:
        with self.state_lock:
            active = self.active_request
            if active is None or active[0] != request_id:
                return
            else:
                pass
            token = active[1]
        try:
            self.send_frame({"type": "cancel", "id": token})
        except (OSError, RuntimeError):
            # The consuming thread reports transport failure for this request.
            pass

    def cancel_active(self) -> None:
        with self.state_lock:
            active = self.active_request
        if active is not None:
            self.cancel(active[0])
        else:
            pass

    def iter_audio(
        self, payload: StagePayload
    ) -> Generator[NDArray[np.float32], None, None]:
        if not self.healthy or self.closed.is_set():
            raise RuntimeError(
                "VoXtream2 model worker is unavailable; restart the stage"
            )
        else:
            pass
        token = uuid.uuid4().hex
        terminal = False
        submitted = False
        owned = False
        try:
            with self.state_lock:
                if self.active_request is not None:
                    raise RuntimeError(
                        "VoXtream2 model worker only supports serial requests"
                    )
                else:
                    pass
                self.active_request = (payload.request_id, token)
                owned = True
                # Publish generate before cancel can see this active token.
                self.send_frame(
                    {
                        "type": "generate",
                        "id": token,
                        "request": payload.request.to_dict(),
                    }
                )
                submitted = True
            while True:
                item = self.receive_frame(self.response_timeout_seconds)
                if item.get("id") != token:
                    raise RuntimeError(
                        "VoXtream2 worker returned a mismatched request id"
                    )
                else:
                    pass
                if item.get("type") == "done":
                    terminal = True
                    if item.get("error"):
                        raise RuntimeError(item["error"])
                    else:
                        pass
                    return
                else:
                    pass
                if item.get("type") != "audio":
                    raise RuntimeError("VoXtream2 worker returned an unexpected frame")
                else:
                    pass
                raw = base64.b64decode(item["data"], validate=True)
                if not raw or len(raw) % 4:
                    raise RuntimeError(
                        "VoXtream2 worker returned invalid float32 audio"
                    )
                else:
                    pass
                yield np.frombuffer(raw, dtype="<f4")
        except ValueError:
            if submitted and not terminal:
                self.close()
            else:
                pass
            raise
        except Exception:
            if owned and not terminal:
                self.close()
            else:
                pass
            raise
        finally:
            if submitted and not terminal and not self.closed.is_set():
                try:
                    self.cancel(payload.request_id)
                    deadline = time.monotonic() + self.cancel_timeout_seconds
                    while not terminal:
                        item = self.receive_frame(deadline - time.monotonic())
                        if item.get("id") != token:
                            raise RuntimeError(
                                "VoXtream2 cancellation response id mismatch"
                            )
                        else:
                            pass
                        if item.get("type") == "done":
                            terminal = True
                            if item.get("error"):
                                raise RuntimeError(item["error"])
                            else:
                                pass
                        elif item.get("type") != "audio":
                            raise RuntimeError(
                                "Unexpected VoXtream2 cancellation response"
                            )
                        else:
                            pass
                        if not terminal and time.monotonic() >= deadline:
                            raise TimeoutError(
                                "VoXtream2 cancellation cleanup timed out"
                            )
                        else:
                            pass
                except Exception:
                    self.healthy = False
                    self.close()
                    with self.state_lock:
                        self.active_request = None
                    raise
            else:
                pass
            if owned:
                with self.state_lock:
                    self.active_request = None
            else:
                pass

    def close(self) -> None:
        """Stop only the owned child and release its pipes and reader thread."""
        with self.close_lock:
            if self.closed.is_set():
                return
            else:
                pass
            self.closed.set()
            self.healthy = False
            if self.process.poll() is None:
                self.process.terminate()
                try:
                    self.process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait(timeout=3)
            else:
                pass
            self.reader.join(timeout=2)
            self.process.stdin.close()
            self.process.stdout.close()
            atexit.unregister(self.close)
