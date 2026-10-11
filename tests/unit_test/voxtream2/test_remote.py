# SPDX-License-Identifier: Apache-2.0
"""Real subprocess tests without a model checkpoint or CUDA device."""

import io
import json
import sys
import time
import wave

import numpy as np
import pytest

from sglang_omni.models.voxtream2 import remote
from sglang_omni.models.voxtream2.remote import RemoteVoXtreamBackend
from sglang_omni.models.voxtream2.wire import MAX_FRAME_BYTES, read_frame, write_frame
from sglang_omni.proto import OmniRequest, StagePayload
from tests.test_model.test_voxtream2 import generate_and_record

FAKE_WORKER = r"""
import os, sys, time
from pathlib import Path
import numpy as np
from sglang_omni.models.voxtream2.wire import read_frame, write_frame
from sglang_omni.models.voxtream2.worker import serve
source, target = sys.stdin.buffer, sys.stdout.buffer
config = read_frame(source)["config"]
if config.get("startup_hang"):
    time.sleep(60)
if config.get("bad_startup"):
    write_frame(target, {"type":"ready", "version":999})
    time.sleep(60)
write_frame(target, {"type":"ready", "version":1, "sample_rate":24000, "max_samples":10000000})
class Backend:
    def iter_audio(self, payload):
        text=payload.request.inputs
        if text == "crash":
            os._exit(7)
        if text == "oversize":
            target.write(b"x"*2000000)
            target.flush()
            time.sleep(60)
        if text == "hang":
            time.sleep(60)
        if text == "mismatch":
            write_frame(target, {"type":"audio", "id":"wrong", "data":"AAAAAA=="})
            time.sleep(60)
        try:
            if text == "invalid":
                raise ValueError("VoXtream2 invalid request: test rejection")
            buffer=np.zeros(1920,dtype=np.float32)
            count=1000 if text in ("flood", "cleanup-fail") else 3
            for index in range(count):
                buffer.fill(index)
                yield buffer
        finally:
            with open(config["log"],"a") as log:
                log.write(text+"\n")
            if text == "cancel-hang":
                time.sleep(60)
            if text == "cleanup-fail":
                raise RuntimeError("state reset failed")
serve(Backend(),source,target)
"""


def payload(text="normal", rid="request"):
    return StagePayload(rid, OmniRequest(text), None)


@pytest.fixture
def worker(tmp_path, monkeypatch):
    script = tmp_path / "fake_worker.py"
    script.write_text(FAKE_WORKER)
    log = tmp_path / "closed.log"
    children, backends = [], []
    original = remote.subprocess.Popen

    def capture(*args, **kwargs):
        child = original(*args, **kwargs)
        children.append(child)
        return child

    monkeypatch.setattr(remote.subprocess, "Popen", capture)

    def start(**kwargs):
        config = {"log": str(log), **kwargs.pop("config", {})}
        backend = RemoteVoXtreamBackend(
            python_executable=sys.executable,
            model_config=config,
            worker_command=[sys.executable, "-u", str(script)],
            startup_timeout=kwargs.pop("startup_timeout", 15),
            response_timeout=kwargs.pop("response_timeout", 10),
            cancel_timeout=kwargs.pop("cancel_timeout", 5),
            source_paths=kwargs.pop("source_paths", ()),
            **kwargs,
        )
        backends.append(backend)
        return backend, log

    yield start
    for backend in backends:
        backend.close()
        assert not backend.reader.is_alive()
    assert all(child.poll() is not None for child in children)


def test_worker_audio_order_reuse_and_cleanup(worker):
    backend, log = worker()
    for rid in ("first", "second"):
        chunks = list(backend.iter_audio(payload(rid=rid)))
        assert len(chunks) == 3
        for index, chunk in enumerate(chunks):
            np.testing.assert_array_equal(chunk, np.full(1920, index))
    assert log.read_text().splitlines() == ["normal", "normal"]
    assert backend.active_request is None


def test_acceptance_measurements_and_audio_artifacts(worker, tmp_path):
    backend, log = worker()
    audio = generate_and_record(backend, tmp_path, "measured", "normal", 101)
    measurement = json.loads((tmp_path / "measured.json").read_text())
    assert measurement["samples"] == 5760
    assert measurement["chunks"] == 3
    assert measurement["audio_seconds"] == 0.24
    assert 0 < measurement["first_audio_seconds"] <= measurement["elapsed_seconds"]
    assert measurement["real_time_factor"] == measurement["elapsed_seconds"] / 0.24
    np.testing.assert_array_equal(np.load(tmp_path / "measured.npy"), audio)
    with wave.open(str(tmp_path / "measured.wav"), "rb") as recording:
        assert recording.getnframes() == 5760
        assert recording.getframerate() == 24000
    assert log.read_text().splitlines() == ["normal"]


def test_close_drains_backpressure_and_acknowledges_cleanup(worker):
    backend, log = worker()
    iterator = backend.iter_audio(payload("flood"))
    next(iterator)
    time.sleep(0.1)
    assert backend.responses.qsize() <= 1
    iterator.close()
    assert log.read_text().splitlines() == ["flood"]
    assert len(list(backend.iter_audio(payload(rid="next")))) == 3


def test_explicit_cancel_suppresses_remaining_generation(worker):
    backend, log = worker()
    iterator = backend.iter_audio(payload("flood"))
    next(iterator)
    backend.cancel("unrelated")
    backend.cancel("request")
    remaining = list(iterator)
    assert len(remaining) < 999
    assert log.read_text().splitlines() == ["flood"]
    assert len(list(backend.iter_audio(payload(rid="next")))) == 3


def test_request_error_does_not_poison_worker(worker):
    backend, _ = worker()
    with pytest.raises(RuntimeError, match="invalid request"):
        list(backend.iter_audio(payload("invalid")))
    assert len(list(backend.iter_audio(payload()))) == 3


@pytest.mark.parametrize("text", ["crash", "oversize", "hang", "mismatch"])
def test_worker_failure_is_bounded_and_prevents_reuse(worker, text):
    backend, _ = worker(response_timeout=0.3)
    with pytest.raises((RuntimeError, TimeoutError)):
        list(backend.iter_audio(payload(text)))
    assert backend.process.poll() is not None
    with pytest.raises(RuntimeError, match="unavailable"):
        list(backend.iter_audio(payload()))


def test_cleanup_failure_closes_worker(worker):
    backend, _ = worker()
    iterator = backend.iter_audio(payload("cleanup-fail"))
    next(iterator)
    with pytest.raises(RuntimeError, match="reset failed"):
        iterator.close()
    assert backend.process.poll() is not None
    assert backend.active_request is None


def test_cancel_timeout_reaps_stuck_cleanup(worker):
    backend, _ = worker(cancel_timeout=0.2)
    iterator = backend.iter_audio(payload("cancel-hang"))
    next(iterator)
    started = time.monotonic()
    with pytest.raises(TimeoutError):
        iterator.close()
    assert time.monotonic() - started < 4
    assert backend.process.poll() is not None


@pytest.mark.parametrize("config", [{"startup_hang": True}, {"bad_startup": True}])
def test_startup_failure_reaps_child(worker, config):
    with pytest.raises((RuntimeError, TimeoutError)):
        worker(config=config, startup_timeout=0.3)


def test_oversized_request_is_rejected_without_poisoning_worker(worker):
    backend, _ = worker()
    with pytest.raises(ValueError, match="exceeds 1 MiB"):
        list(backend.iter_audio(payload("x" * MAX_FRAME_BYTES)))
    assert len(list(backend.iter_audio(payload()))) == 3


@pytest.mark.parametrize(
    "data",
    [b"[]\n", b"{", b"x" * (MAX_FRAME_BYTES + 1)],
    ids=["non-object", "truncated-json", "oversized-frame"],
)
def test_invalid_wire_frames(data):
    with pytest.raises((RuntimeError, ValueError)):
        read_frame(io.BytesIO(data))


def test_wire_roundtrip_and_nonfinite_rejection():
    stream = io.BytesIO()
    write_frame(stream, {"text": "hello", "seed": 17})
    stream.seek(0)
    assert read_frame(stream) == {"text": "hello", "seed": 17}
    with pytest.raises(ValueError, match="invalid request"):
        write_frame(io.BytesIO(), {"rate": float("nan")})


def test_accidental_overlap_does_not_corrupt_active_request(worker):
    backend, _ = worker()
    iterator = backend.iter_audio(payload())
    next(iterator)
    with pytest.raises(RuntimeError, match="serial requests"):
        next(backend.iter_audio(payload(rid="overlap")))
    assert len(list(iterator)) == 2
    assert len(list(backend.iter_audio(payload(rid="next")))) == 3
