# SPDX-License-Identifier: Apache-2.0
"""VoXtream2 worker protocol and serial generation loop."""

from __future__ import annotations

import base64
import queue
import sys
import threading
import traceback
from typing import BinaryIO

import numpy as np

from sglang_omni.models.voxtream2.scheduler import AudioBackend
from sglang_omni.models.voxtream2.wire import read_frame, write_frame
from sglang_omni.proto import OmniRequest, StagePayload


def serve(backend: AudioBackend, source: BinaryIO, target: BinaryIO) -> None:
    """Receive cancellation independently of GPU generation and output writes."""
    pending = queue.Queue(maxsize=1)
    stopped = threading.Event()
    lock = threading.Lock()
    active = {}

    def receive() -> None:
        try:
            while not stopped.is_set():
                item = read_frame(source)
                kind, token = item.get("type"), item.get("id")
                with lock:
                    if kind == "cancel":
                        if token in active:
                            active[token].set()
                        else:
                            pass
                    elif kind == "generate" and isinstance(token, str) and not active:
                        event = threading.Event()
                        active[token] = event
                        pending.put_nowait((item, event))
                    else:
                        raise RuntimeError(
                            "Invalid VoXtream2 worker command or overlap"
                        )
        except EOFError:
            pass
        except Exception:
            traceback.print_exc(file=sys.stderr)
        finally:
            stopped.set()
            with lock:
                for event in active.values():
                    event.set()

    receiver = threading.Thread(
        target=receive, name="voxtream-worker-input", daemon=True
    )
    receiver.start()
    while not stopped.is_set():
        try:
            item, cancelled = pending.get(timeout=0.05)
        except queue.Empty:
            continue
        token = item["id"]
        error = None
        iterator = None
        try:
            payload = StagePayload(token, OmniRequest.from_dict(item["request"]), None)
            iterator = backend.iter_audio(payload)
            while not cancelled.is_set():
                try:
                    audio = next(iterator)
                except StopIteration:
                    break
                if cancelled.is_set():
                    break
                else:
                    pass
                array = np.asarray(audio, dtype="<f4")
                if array.ndim != 1 or not array.size or not np.isfinite(array).all():
                    raise RuntimeError("VoXtream2 model returned invalid audio")
                else:
                    pass
                write_frame(
                    target,
                    {
                        "type": "audio",
                        "id": token,
                        "data": base64.b64encode(array.tobytes()).decode("ascii"),
                    },
                )
        except Exception as exc:
            error = str(exc)
            traceback.print_exc(file=sys.stderr)
        finally:
            if iterator is not None:
                try:
                    iterator.close()
                except Exception as exc:
                    error = str(exc)
                    traceback.print_exc(file=sys.stderr)
            else:
                pass
            # Clear before acknowledging so the next request cannot race with
            # stale active state. Cleanup has already finished at this point.
            with lock:
                active.pop(token, None)
        write_frame(target, {"type": "done", "id": token, "error": error})
