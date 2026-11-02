# SPDX-License-Identifier: Apache-2.0
"""Guard decoding, device placement, admission, and GPU settlement."""

import io
import threading
from collections.abc import Iterator
from contextlib import contextmanager, nullcontext
from types import SimpleNamespace
from typing import NoReturn

import numpy as np
import pytest
import soundfile as sf
import torch

import sglang_omni.platforms as platforms
from sglang_omni.admission import QueueFullError
from sglang_omni.client.client import Client
from sglang_omni.client.types import GenerateRequest
from sglang_omni.models.nemotron_diarization import stages
from sglang_omni.models.nemotron_diarization.backend import load_diarization_model
from sglang_omni.models.nemotron_diarization.model import NemotronDiarizationModel
from sglang_omni.models.nemotron_diarization.speaker_cache import SpeakerCache
from sglang_omni.proto import StagePayload
from sglang_omni.scheduling.message import IncomingMessage
from sglang_omni.scheduling.step_scheduler import StepResult, StepScheduler, TaskBuilder


@pytest.fixture(autouse=True)
def cuda_platform(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        platforms,
        "current_platform",
        SimpleNamespace(device_type="cuda", supports_torchaudio_resample=lambda: True),
    )
    monkeypatch.setattr(
        torch.cuda,
        "current_stream",
        lambda device: SimpleNamespace(synchronize=lambda: None),
    )


class RecordingModel:
    profile = (8, 8, 340, 40, 3)
    iter_chunks = NemotronDiarizationModel.iter_chunks

    def __init__(self) -> None:
        self.calls: list[np.ndarray] = []

    def preprocessor(self, signal: torch.Tensor) -> tuple[torch.Tensor, int]:
        self.calls.append(signal[0].numpy())
        frame_count = signal.shape[1] // 160
        return torch.zeros(1, 128, frame_count), frame_count

    def forward_chunk(
        self, features: torch.Tensor, state: SpeakerCache, right_frames: int
    ) -> torch.Tensor:
        return torch.zeros(1, features.shape[2] - right_frames, 8)


@contextmanager
def running(
    worker: stages.DiarizationWorker,
    build_task: TaskBuilder | None = None,
    *,
    max_running_requests: int = 16,
    max_queued_requests: int = 64,
) -> Iterator[StepScheduler]:
    scheduler = StepScheduler(
        build_task or (lambda request: stages.DiarizationTask(worker, request)),
        max_workers=1,
        max_running_requests=max_running_requests,
        max_queued_requests=max_queued_requests,
        shutdown_callback=worker.live_sessions.close,
    )
    thread = threading.Thread(target=scheduler.start)
    thread.start()
    try:
        yield scheduler
    finally:
        scheduler.stop()
        thread.join(timeout=5)
        assert not thread.is_alive()


def run_payload(
    worker: stages.DiarizationWorker, request: StagePayload
) -> StagePayload:
    with running(worker) as scheduler:
        scheduler.inbox.put(IncomingMessage(request.request_id, "new_request", request))
        result = scheduler.outbox.get(timeout=5)
    if result.type == "error":
        raise result.data
    else:
        return result.data


def payload(
    prompt: dict[str, bytes | str],
    *,
    task: str = "diarization",
    request_id: str = "recording",
    params: dict[str, int] | None = None,
) -> StagePayload:
    request = Client.build_omni_request(
        GenerateRequest(prompt=prompt, stream=False, metadata={"task": task})
    )
    request.params.update(params or {})
    return StagePayload(request_id=request_id, request=request, data={})


def wav(samples: np.ndarray, sample_rate: int) -> bytes:
    stream = io.BytesIO()
    sf.write(stream, samples, sample_rate, format="WAV", subtype="FLOAT")
    return stream.getvalue()


def test_stereo_resampling_preserves_duration_and_request_identity() -> None:
    model = RecordingModel()
    worker = stages.DiarizationWorker(model, torch.device("cpu"), 1, 1)
    audio = wav(np.column_stack([np.ones(8000), -np.ones(8000)]), 8000)
    result = run_payload(worker, payload({"audio_bytes": audio}))
    np.testing.assert_array_equal(model.calls[0], np.zeros(16000))
    assert result.request_id == "recording"
    assert result.data == {"diarization": {"duration": 1.0, "segments": []}}


@pytest.mark.parametrize(
    "prompt, task, params, message",
    [
        ({"audio_bytes": b"garbage"}, "diarization", {}, "could not decode"),
        ({"audio_bytes": wav(np.array([]), 16000)}, "diarization", {}, "decode"),
        (
            {"audio_bytes": wav(np.full(160, np.nan), 16000)},
            "diarization",
            {},
            "decode",
        ),
        ({"audio_bytes": b"unused"}, "asr", {}, "/v1/audio/diarizations"),
        ({"audio_bytes": b"unused"}, "diarization", {"num_speakers": 9}, "controls"),
    ],
)
def test_invalid_requests_never_reach_inference(
    prompt: dict[str, bytes], task: str, params: dict[str, int], message: str
) -> None:
    model = RecordingModel()
    worker = stages.DiarizationWorker(model, torch.device("cpu"), 1, 1)
    with pytest.raises(ValueError, match=message):
        run_payload(worker, payload(prompt, task=task, params=params))
    assert not model.calls


def test_unsupported_deployments_fail_before_loading(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    devices = []

    def load(
        model_path: str, *, device: torch.device, profile: str
    ) -> NemotronDiarizationModel:
        devices.append(device)
        raise LookupError("loaded")

    for device, profile, message in [
        (torch.device("cpu"), "offline", "CUDA device"),
        (torch.device("cuda:0"), "live", "Unknown diarization profile"),
    ]:
        with pytest.raises(ValueError, match=message):
            load_diarization_model("missing/repository", device=device, profile=profile)
    monkeypatch.setattr(stages, "load_diarization_model", load)
    for concurrency in [0, True]:
        with pytest.raises(ValueError, match="max_concurrency must be a positive"):
            stages.create_diarization_executor(
                "unused",
                profile="offline",
                max_concurrency=concurrency,
                max_live_sessions=8,
                max_running_requests=16,
                max_queued_requests=64,
            )
    assert not devices
    with pytest.raises(LookupError):
        stages.create_diarization_executor(
            "unused",
            device="cuda",
            gpu_id=3,
            profile="offline",
            max_concurrency=1,
            max_live_sessions=8,
            max_running_requests=16,
            max_queued_requests=64,
        )
    # note (Ryan Stewart): A hard-coded cuda:0 would put every stage on the wrong GPU.
    assert devices == [torch.device("cuda:3")]


def test_failed_inference_waits_for_gpu_before_propagating_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    launched, waiting, finished = (
        threading.Event(),
        threading.Event(),
        threading.Event(),
    )

    class Stream:
        def wait_event(self, event: torch.cuda.Event) -> None:
            pass

        def synchronize(self) -> None:
            assert launched.is_set()
            waiting.set()
            assert finished.wait(timeout=5)

    class FailingModel(RecordingModel):
        def preprocessor(self, signal: torch.Tensor) -> NoReturn:
            launched.set()
            raise RuntimeError("inference failed after launching GPU work")

    monkeypatch.setattr(
        torch.cuda, "Event", lambda: SimpleNamespace(record=lambda stream: None)
    )
    monkeypatch.setattr(torch.cuda, "Stream", lambda device: Stream())
    monkeypatch.setattr(torch.cuda, "stream", lambda stream: nullcontext())
    worker = stages.DiarizationWorker(FailingModel(), torch.device("cpu"), 2, 1)
    errors: list[RuntimeError] = []

    def run() -> None:
        try:
            run_payload(worker, payload({"audio_bytes": wav(np.zeros(1600), 16000)}))
        except RuntimeError as error:
            errors.append(error)

    thread = threading.Thread(target=run)
    thread.start()
    try:
        assert waiting.wait(timeout=5)
        assert not errors
    finally:
        finished.set()
        thread.join(timeout=5)
    assert "inference failed" in str(errors[0])


@pytest.mark.parametrize(
    "operation, fields, pending, sessions",
    [
        ("append", {"pcm": b"\0\0" * 160}, {}, ["connected"]),
        ("reset", {"next_session_id": "replacement"}, {}, ["replacement"]),
        # note (Richard Wang): A pending count marks an aborted operation whose step still runs.
        ("close", {}, {"connected": 1}, []),
    ],
)
def test_live_operations_under_a_full_queue(
    operation: str,
    fields: dict[str, bytes | str],
    pending: dict[str, int],
    sessions: list[str],
) -> None:
    entered, release = threading.Event(), threading.Event()

    class BlockedTask:
        bounded_by_stage = False

        def step(self) -> StepResult:
            entered.set()
            assert release.wait(5)
            return StepResult(done=True)

        def close(self, *, aborted: bool) -> None:
            pass

    worker = stages.DiarizationWorker(RecordingModel(), torch.device("cpu"), 1, 1)
    worker.live_sessions.compute({"session_id": "connected", "operation": "open"})
    worker.live_sessions.pending.update(pending)
    live = payload(
        {"session_id": "connected", "operation": operation, **fields},
        task="diarization_stream",
        request_id="live",
    )

    def build_task(request: StagePayload) -> stages.DiarizationTask | BlockedTask:
        if request is live:
            return stages.DiarizationTask(worker, request)
        else:
            return BlockedTask()

    with running(
        worker, build_task, max_running_requests=1, max_queued_requests=1
    ) as scheduler:
        scheduler.inbox.put(IncomingMessage("active", "new_request", None))
        assert entered.wait(5)
        scheduler.inbox.put(IncomingMessage("waiting", "new_request", None))
        scheduler.inbox.put(IncomingMessage("live", "new_request", live))
        if pending:
            rejection = scheduler.outbox.get(timeout=5)
            assert rejection.request_id == "live"
            assert isinstance(rejection.data, QueueFullError)
            release.set()
        else:
            release.set()
            messages = [scheduler.outbox.get(timeout=5) for _ in range(3)]
            assert {message.request_id: message.type for message in messages}[
                "live"
            ] == "result"
        assert list(worker.live_sessions.sessions) == sessions
        assert worker.live_sessions.pending == pending
