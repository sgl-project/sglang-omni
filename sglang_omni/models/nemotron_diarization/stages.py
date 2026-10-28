# SPDX-License-Identifier: Apache-2.0
"""Cooperative Sortformer tasks scheduled by StepScheduler."""

from __future__ import annotations

from collections.abc import Generator, Iterator
from contextlib import contextmanager, nullcontext
from threading import local

import msgspec
import numpy as np
import torch

from sglang_omni.client.types import DiarizationResult, SamplingParams
from sglang_omni.models.nemotron_diarization.backend import (
    SAMPLE_RATE,
    DiarizationProfile,
    load_diarization_model,
    probabilities_to_segments,
)
from sglang_omni.models.nemotron_diarization.model import NemotronDiarizationModel
from sglang_omni.models.nemotron_diarization.streaming import LiveSessions
from sglang_omni.preprocessing.transcription import resolve_audio_source
from sglang_omni.proto import StagePayload
from sglang_omni.scheduling.step_scheduler import StepResult, StepScheduler
from sglang_omni.utils.audio import AudioDecodeError, load_audio
from sglang_omni.utils.device import resolve_concrete_device


class DiarizationTask:
    """Advance one recording by one model window, or run one live operation."""

    def __init__(self, worker: DiarizationWorker, payload: StagePayload) -> None:
        task = payload.request.metadata.get("task")
        unsupported = [
            name
            for name, value in payload.request.params.items()
            if name not in worker.default_params or value != worker.default_params[name]
        ]
        if task not in ("diarization", "diarization_stream"):
            raise ValueError(
                "This model accepts requests through /v1/audio/diarizations"
            )
        elif unsupported:
            raise ValueError(f"Unsupported diarization controls: {sorted(unsupported)}")
        else:
            pass
        self.worker = worker
        self.payload = payload
        self.live = task == "diarization_stream"
        self.windows: Generator[tuple[torch.Tensor, bool], None, None] | None = None
        self.probabilities: list[np.ndarray] = []
        self.duration_s = 0.0
        # note (Richard Wang): max_live_sessions already bounds the operations of open sessions.
        self.bounded_by_stage = self.live and worker.live_sessions.reserve(
            payload.request.inputs["session_id"]
        )

    @torch.inference_mode()
    def step(self) -> StepResult:
        with self.worker.compute_context():
            result = (
                self.worker.live_sessions.compute(self.payload.request.inputs)
                if self.live
                else self.upload_step()
            )
            # note (Richard Wang): Build the result before the exit synchronize yields the GIL.
            if result is None:
                return StepResult()
            else:
                return StepResult(
                    done=True,
                    output=StagePayload(
                        request_id=self.payload.request_id,
                        request=self.payload.request,
                        data={"diarization": msgspec.to_builtins(result)},
                    ),
                )

    def upload_step(self) -> DiarizationResult | None:
        if self.windows is None:
            try:
                waveform = load_audio(
                    resolve_audio_source(self.payload),
                    source_name="diarization",
                    target_sample_rate=SAMPLE_RATE,
                )
            except AudioDecodeError as decode_error:
                raise ValueError(
                    "could not decode the uploaded audio"
                ) from decode_error
            if (
                waveform.ndim != 1
                or waveform.size == 0
                or not np.isfinite(waveform).all()
            ):
                raise ValueError(
                    "could not decode the uploaded audio: empty or non-finite waveform"
                )
            else:
                pass
            self.duration_s = len(waveform) / SAMPLE_RATE
            self.windows = self.worker.model.iter_chunks(
                torch.as_tensor(
                    waveform, dtype=torch.float32, device=self.worker.device
                )[None, :]
            )
        else:
            pass
        predictions, done = next(self.windows)
        self.probabilities.append(predictions[0].cpu().numpy())
        if done:
            return probabilities_to_segments(
                np.concatenate(self.probabilities), duration_s=self.duration_s
            )
        else:
            return None

    def close(self, *, aborted: bool) -> None:
        if self.live:
            self.worker.live_sessions.release(
                self.payload.request.inputs["session_id"], aborted=aborted
            )
        elif self.windows is not None:
            self.windows.close()
        else:
            pass


class DiarizationWorker:
    """Model state shared by every task, and one CUDA stream per worker thread."""

    def __init__(
        self,
        model: NemotronDiarizationModel,
        device: torch.device,
        max_concurrency: int,
        max_live_sessions: int,
    ) -> None:
        self.model = model
        self.device = device
        self.live_sessions = LiveSessions(model, device, max_live_sessions)
        if max_concurrency > 1:
            # note (Richard Wang): One stream per thread, matching cuBLAS per-stream workspaces.
            self.thread_streams = local()
            self.loaded = torch.cuda.Event()
            self.loaded.record(torch.cuda.current_stream(device))
        else:
            self.thread_streams = None
        self.default_params = SamplingParams().to_dict()
        self.default_params["stream"] = False

    @contextmanager
    def compute_context(self) -> Iterator[None]:
        """Run one step and settle its device work before the task can move."""
        if self.thread_streams is None:
            stream = torch.cuda.current_stream(self.device)
            context = nullcontext()
        else:
            stream = getattr(self.thread_streams, "stream", None)
            if stream is None:
                stream = self.thread_streams.stream = torch.cuda.Stream(self.device)
                stream.wait_event(self.loaded)
            else:
                pass
            context = torch.cuda.stream(stream)
        try:
            with context:
                yield
        finally:
            stream.synchronize()


def create_diarization_executor(
    model_path: str,
    *,
    device: str | None = None,
    gpu_id: int | None = None,
    profile: DiarizationProfile,
    max_concurrency: int,
    max_live_sessions: int,
    max_running_requests: int,
    max_queued_requests: int,
) -> StepScheduler:
    for name, value in (
        ("max_concurrency", max_concurrency),
        ("max_live_sessions", max_live_sessions),
        ("max_running_requests", max_running_requests),
        ("max_queued_requests", max_queued_requests),
    ):
        if type(value) is not int or value < 1:
            raise ValueError(f"{name} must be a positive integer")
        else:
            pass
    if max_concurrency > max_running_requests:
        raise ValueError("max_concurrency cannot exceed max_running_requests")
    else:
        pass
    resolved_device = resolve_concrete_device(device, gpu_id)
    model = load_diarization_model(model_path, device=resolved_device, profile=profile)
    worker = DiarizationWorker(
        model, resolved_device, max_concurrency, max_live_sessions
    )
    return StepScheduler(
        lambda payload: DiarizationTask(worker, payload),
        max_workers=max_concurrency,
        max_running_requests=max_running_requests,
        max_queued_requests=max_queued_requests,
        shutdown_callback=worker.live_sessions.close,
    )
