# SPDX-License-Identifier: Apache-2.0
"""Cooperative Sortformer tasks scheduled by StepScheduler."""

from contextlib import contextmanager
from threading import local

import msgspec
import numpy as np
import torch

from sglang_omni.client.types import SamplingParams
from sglang_omni.models.nemotron_diarization.backend import (
    SAMPLE_RATE,
    NemotronDiarizer,
    probabilities_to_segments,
)
from sglang_omni.models.nemotron_diarization.streaming import LiveSessions
from sglang_omni.preprocessing.transcription import resolve_audio_source
from sglang_omni.proto import StagePayload
from sglang_omni.scheduling.step_scheduler import StepResult, StepScheduler
from sglang_omni.utils.audio import AudioDecodeError, load_audio
from sglang_omni.utils.device import resolve_concrete_device


class DiarizationTask:
    """Advance one recording by at most one uploaded-audio model window."""

    def __init__(self, worker, payload: StagePayload):
        task = payload.request.metadata.get("task")
        if task not in {"diarization", "diarization_stream"}:
            raise ValueError(
                "This model accepts requests through /v1/audio/diarizations"
            )
        unsupported = [
            name
            for name, value in payload.request.params.items()
            if name not in worker.default_params or value != worker.default_params[name]
        ]
        if unsupported:
            raise ValueError(f"Unsupported diarization controls: {sorted(unsupported)}")
        self.worker = worker
        self.payload = payload
        self.live = task == "diarization_stream"
        self.chunks = None
        self.outputs = []
        self.duration = 0.0
        if self.live:
            worker.live_sessions.reserve(payload.request.inputs["session_id"])

    def result(self, result) -> StepResult:
        return StepResult(
            done=True,
            output=StagePayload(
                request_id=self.payload.request_id,
                request=self.payload.request,
                data={"diarization": msgspec.to_builtins(result)},
            ),
        )

    @torch.inference_mode()
    def step(self) -> StepResult:
        with self.worker.compute_context():
            if self.live:
                return self.result(
                    self.worker.live_sessions.compute(self.payload.request.inputs)
                )
            return self.upload_step()

    def upload_step(self) -> StepResult:
        diarizer = self.worker.diarizer
        if self.chunks is None:
            try:
                waveform = load_audio(
                    resolve_audio_source(self.payload),
                    source_name="diarization",
                    target_sample_rate=SAMPLE_RATE,
                )
            except AudioDecodeError as exc:
                raise ValueError("could not decode the uploaded audio") from exc
            if (
                waveform.ndim != 1
                or waveform.size == 0
                or not np.isfinite(waveform).all()
            ):
                raise ValueError(
                    "could not decode the uploaded audio: empty or non-finite waveform"
                )
            self.duration = len(waveform) / SAMPLE_RATE
            signal = torch.as_tensor(
                waveform, dtype=torch.float32, device=diarizer.device
            )[None, :]
            self.chunks = diarizer.model.iter_chunks(signal)
        predictions, done = next(self.chunks)
        if predictions.shape[1]:
            self.outputs.append(predictions[0].cpu().numpy())
        if not done:
            return StepResult()
        predictions = (
            np.concatenate(self.outputs, axis=0)
            if self.outputs
            else np.empty((0, 8), dtype=np.float32)
        )
        return self.result(
            probabilities_to_segments(predictions, duration=self.duration)
        )

    def close(self, *, aborted: bool) -> None:
        if self.live:
            self.worker.live_sessions.release(
                self.payload.request.inputs["session_id"], aborted=aborted
            )
        if self.chunks is not None:
            self.chunks.close()
            self.chunks = None
        self.outputs.clear()


class DiarizationWorker:
    def __init__(self, diarizer, max_concurrency: int, max_live_sessions: int):
        self.diarizer = diarizer
        self.live_sessions = LiveSessions(diarizer, max_live_sessions)
        self.worker_state = local() if max_concurrency > 1 else None
        self.ready = None
        if self.worker_state is not None:
            self.ready = torch.cuda.Event()
            self.ready.record(torch.cuda.current_stream(diarizer.device))
        self.default_params = SamplingParams().to_dict()
        self.default_params.pop("max_new_tokens")
        self.default_params["stream"] = False

    def build(self, payload: StagePayload) -> DiarizationTask:
        return DiarizationTask(self, payload)

    @contextmanager
    def compute_context(self):
        device = self.diarizer.device
        if self.worker_state is None:
            try:
                yield
            finally:
                torch.cuda.current_stream(device).synchronize()
            return
        stream = getattr(self.worker_state, "stream", None)
        if stream is None:
            stream = torch.cuda.Stream(device=device)
            stream.wait_event(self.ready)
            self.worker_state.stream = stream
        try:
            with torch.cuda.stream(stream):
                yield
        finally:
            stream.synchronize()


def create_diarization_executor(
    model_path: str,
    *,
    device: str | None = None,
    gpu_id: int | None = None,
    profile: str = "offline",
    max_concurrency: int = 1,
    max_live_sessions: int = 8,
    max_running_requests: int = 16,
    max_queued_requests: int = 64,
) -> StepScheduler:
    for name, value in (
        ("max_concurrency", max_concurrency),
        ("max_live_sessions", max_live_sessions),
        ("max_running_requests", max_running_requests),
        ("max_queued_requests", max_queued_requests),
    ):
        if type(value) is not int or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if max_concurrency > max_running_requests:
        raise ValueError("max_concurrency cannot exceed max_running_requests")
    concrete_device = resolve_concrete_device(device, gpu_id)
    diarizer = NemotronDiarizer(model_path, device=concrete_device, profile=profile)
    worker = DiarizationWorker(diarizer, max_concurrency, max_live_sessions)
    return StepScheduler(
        worker.build,
        max_workers=max_concurrency,
        max_running_requests=max_running_requests,
        max_queued_requests=max_queued_requests,
        shutdown_callback=worker.live_sessions.close,
    )
