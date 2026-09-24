# SPDX-License-Identifier: Apache-2.0
"""Guard decoding, device placement, and admission before optional inference."""

import io
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from types import MethodType, SimpleNamespace

import numpy as np
import pytest
import soundfile as sf
import torch

from sglang_omni.client import Client, GenerateRequest
from sglang_omni.models.nemotron_diarization import stages
from sglang_omni.models.nemotron_diarization.backend import NemotronDiarizer
from sglang_omni.models.nemotron_diarization.model import NemotronDiarizationModel
from sglang_omni.proto import StagePayload
from sglang_omni.scheduling.messages import IncomingMessage


@pytest.fixture
def cuda_platform(monkeypatch):
    import sglang_omni.platforms as platforms

    monkeypatch.setattr(
        platforms, "current_platform", SimpleNamespace(device_type="cuda")
    )


@pytest.fixture
def executor(cuda_platform, monkeypatch):
    calls = []

    class RecordingDiarizer:
        def __init__(self, model_path, *, device, profile):
            # A hard-coded cuda:0 would put every stage on the wrong GPU.
            assert device == torch.device("cuda:3")
            self.device = torch.device("cpu")
            self.model = SimpleNamespace(
                profile=(8, 8, 340, 40, 3),
                preprocessor=self.preprocess,
                forward_chunk=lambda features, state, right: torch.zeros(
                    1, features.shape[2] - right, 8
                ),
            )

            self.model.iter_chunks = MethodType(
                NemotronDiarizationModel.iter_chunks, self.model
            )

        def preprocess(self, signal):
            calls.append(signal[0].numpy())
            length = signal.shape[1] // 160
            return torch.zeros(1, 128, length), length

    monkeypatch.setattr(stages, "NemotronDiarizer", RecordingDiarizer)
    monkeypatch.setattr(
        torch.cuda,
        "current_stream",
        lambda device: SimpleNamespace(synchronize=lambda: None),
    )
    scheduler = stages.create_diarization_executor("unused", device="cuda", gpu_id=3)
    return scheduler, calls


def run_payload(scheduler, request):
    thread = threading.Thread(target=scheduler.start)
    thread.start()
    try:
        scheduler.inbox.put(IncomingMessage(request.request_id, "new_request", request))
        result = scheduler.outbox.get(timeout=5)
        if result.type == "error":
            raise result.data
        return result.data
    finally:
        scheduler.stop()
        thread.join(timeout=5)
        assert not thread.is_alive()


def payload(audio, *, task="diarization"):
    request = Client.build_omni_request(
        GenerateRequest(
            prompt={"audio_bytes": audio}, stream=False, metadata={"task": task}
        )
    )
    return StagePayload(request_id="recording", request=request, data={})


def wav(samples, rate):
    stream = io.BytesIO()
    sf.write(stream, samples, rate, format="WAV", subtype="FLOAT")
    return stream.getvalue()


@pytest.mark.parametrize("samples", [1, 8000])
def test_stereo_resampling_preserves_duration_and_request_identity(executor, samples):
    scheduler, calls = executor
    audio = wav(np.column_stack([np.ones(samples), -np.ones(samples)]), 8000)
    result = run_payload(scheduler, payload(audio))
    assert calls[0].shape == (samples * 2,)
    np.testing.assert_array_equal(calls[0], np.zeros(samples * 2))
    assert result.request_id == "recording"
    assert result.data == {"diarization": {"duration": samples / 8000, "segments": []}}


@pytest.mark.parametrize(
    "audio",
    [b"garbage", b"", wav(np.array([]), 16000), wav(np.full(1600, np.nan), 16000)],
)
def test_invalid_audio_never_reaches_inference(executor, audio):
    scheduler, calls = executor
    with pytest.raises(ValueError, match="could not decode the uploaded audio"):
        run_payload(scheduler, payload(audio))
    assert not calls


def test_transcription_request_never_runs_diarization(executor):
    scheduler, calls = executor
    with pytest.raises(ValueError, match="/v1/audio/diarizations"):
        run_payload(scheduler, payload(b"unused", task="asr"))
    assert not calls


@pytest.mark.parametrize(
    "params",
    [
        {"stream": True},
        {"temperature": 0.5},
        {"max_new_tokens": 20},
        {"num_speakers": 9},
    ],
)
def test_low_level_client_cannot_silently_apply_generation_controls(executor, params):
    scheduler, calls = executor
    request = payload(wav(np.zeros(16000), 16000))
    request.request.params.update(params)
    with pytest.raises(ValueError, match="Unsupported diarization controls"):
        run_payload(scheduler, request)
    assert not calls


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"device": "cpu"}, "CUDA device"),
        ({"device": "cuda:0", "profile": "live"}, "Unknown diarization profile"),
    ],
)
def test_unsupported_deployment_fails_before_checkpoint_download(kwargs, message):
    with pytest.raises(ValueError, match=message):
        NemotronDiarizer("missing/repository", **kwargs)


@pytest.mark.parametrize("concurrency", [0, -1, 1.5, True, "2"])
def test_invalid_concurrency_fails_before_model_loading(concurrency, monkeypatch):
    def unexpected_load(*args, **kwargs):
        pytest.fail("invalid concurrency reached model loading")

    monkeypatch.setattr(stages, "NemotronDiarizer", unexpected_load)
    with pytest.raises(ValueError, match="max_concurrency must be a positive integer"):
        stages.create_diarization_executor(
            "unused", device="cpu", max_concurrency=concurrency
        )


def test_failed_inference_waits_for_gpu_before_propagating_error(
    cuda_platform, monkeypatch
):
    launched = threading.Event()
    waiting = threading.Event()
    finished = threading.Event()
    initialized = object()

    class Ready:
        def record(self, stream):
            assert stream is initialized

    class Stream:
        def wait_event(self, event):
            assert isinstance(event, Ready)

        def synchronize(self):
            assert launched.is_set()
            waiting.set()
            assert finished.wait(timeout=5)

    class FailingDiarizer:
        def __init__(self, *args, **kwargs):
            self.device = torch.device("cpu")
            self.model = SimpleNamespace(preprocessor=self.preprocess)

            self.model.iter_chunks = MethodType(
                NemotronDiarizationModel.iter_chunks, self.model
            )

        def preprocess(self, signal):
            launched.set()
            raise RuntimeError("inference failed after launching GPU work")

    monkeypatch.setattr(stages, "NemotronDiarizer", FailingDiarizer)
    monkeypatch.setattr(torch.cuda, "Event", Ready)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda device: initialized)
    monkeypatch.setattr(torch.cuda, "Stream", lambda device: Stream())
    monkeypatch.setattr(torch.cuda, "stream", lambda stream: nullcontext())
    scheduler = stages.create_diarization_executor(
        "unused", device="cuda", gpu_id=3, max_concurrency=2
    )
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(
            run_payload, scheduler, payload(wav(np.zeros(1600), 16000))
        )
        try:
            assert waiting.wait(timeout=5)
            assert not future.done()
        finally:
            finished.set()
        with pytest.raises(RuntimeError, match="inference failed"):
            future.result(timeout=5)


def test_rejected_live_close_releases_session_even_when_queue_is_full():
    from sglang_omni.admission import QueueFullError
    from sglang_omni.scheduling.step_scheduler import StepResult, StepScheduler

    entered, release = threading.Event(), threading.Event()

    class BlockedTask:
        def step(self):
            entered.set()
            assert release.wait(5)
            return StepResult(done=True)

        def close(self, *, aborted):
            pass

    worker = stages.DiarizationWorker(SimpleNamespace(model=None, device="cpu"), 1, 1)
    worker.live_sessions.sessions["connected"] = SimpleNamespace(
        lock=threading.Lock(), last_used=time.monotonic()
    )
    request = Client.build_omni_request(
        GenerateRequest(
            prompt={"session_id": "connected", "operation": "close"},
            stream=False,
            metadata={"task": "diarization_stream"},
        )
    )
    close = StagePayload(request_id="close", request=request, data={})
    scheduler = StepScheduler(
        lambda value: worker.build(value) if value is close else BlockedTask(),
        max_running_requests=1,
        max_queued_requests=1,
    )
    thread = threading.Thread(target=scheduler.start)
    thread.start()
    try:
        scheduler.inbox.put(IncomingMessage("active", "new_request", None))
        assert entered.wait(5)
        scheduler.inbox.put(IncomingMessage("waiting", "new_request", None))
        scheduler.inbox.put(IncomingMessage("close", "new_request", close))
        rejection = scheduler.outbox.get(timeout=5)
        assert rejection.request_id == "close"
        assert isinstance(rejection.data, QueueFullError)
        assert not worker.live_sessions.sessions
        assert not worker.live_sessions.pending
    finally:
        scheduler.stop()
        release.set()
        thread.join(5)
        assert not thread.is_alive()
