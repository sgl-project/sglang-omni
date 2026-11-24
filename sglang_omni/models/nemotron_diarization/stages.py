# SPDX-License-Identifier: Apache-2.0
"""Cooperative Sortformer tasks scheduled by StepScheduler."""

from __future__ import annotations

from collections.abc import Generator, Iterator
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from threading import local

import msgspec
import numpy as np
import torch
from numpy.typing import NDArray

import sglang_omni.platforms as platforms
from sglang_omni.client.types import DiarizationResult, SamplingParams
from sglang_omni.models.nemotron_diarization.backend import (
    SAMPLE_RATE,
    DiarizationProfile,
    load_diarization_model,
    probabilities_to_segments,
)
from sglang_omni.models.nemotron_diarization.model import (
    MEL_FRAMES_PER_ENCODER_FRAME,
    NemotronDiarizationModel,
)
from sglang_omni.models.nemotron_diarization.speaker_cache import SpeakerCache
from sglang_omni.models.nemotron_diarization.streaming import LiveSessions
from sglang_omni.preprocessing.transcription import resolve_audio_source
from sglang_omni.proto import StagePayload
from sglang_omni.scheduling.batched_step_scheduler import (
    BatchedStepScheduler,
    BatchRequestStep,
)
from sglang_omni.scheduling.step_scheduler import StepResult, StepScheduler, StepTask
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
        self.probabilities: list[NDArray[np.float32]] = []
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
                return self.build_result(result)

    def build_result(self, diarization: DiarizationResult) -> StepResult:
        return StepResult(
            done=True,
            output=StagePayload(
                request_id=self.payload.request_id,
                request=self.payload.request,
                data={"diarization": msgspec.to_builtins(diarization)},
            ),
        )

    def load_waveform(self) -> NDArray[np.float32]:
        try:
            waveform = load_audio(
                resolve_audio_source(self.payload),
                source_name="diarization",
                target_sample_rate=SAMPLE_RATE,
            )
        except AudioDecodeError as decode_error:
            raise ValueError("could not decode the uploaded audio") from decode_error
        if waveform.ndim != 1 or waveform.size == 0 or not np.isfinite(waveform).all():
            raise ValueError(
                "could not decode the uploaded audio: empty or non-finite waveform"
            )
        else:
            pass
        self.duration_s = len(waveform) / SAMPLE_RATE
        return waveform

    def upload_step(self) -> DiarizationResult | None:
        if self.windows is None:
            waveform = self.load_waveform()
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
    """Model state shared by every task, and one device stream per worker thread."""

    def __init__(
        self,
        model: NemotronDiarizationModel,
        device: torch.device,
        max_concurrency: int,
        max_live_sessions: int,
    ) -> None:
        self.model = model
        self.device = device
        self.device_module = torch.get_device_module(
            platforms.current_platform.device_type
        )
        self.live_sessions = LiveSessions(model, device, max_live_sessions)
        if max_concurrency > 1:
            # note (Richard Wang): One stream per thread, matching cuBLAS per-stream workspaces.
            self.thread_streams = local()
            self.loaded = self.device_module.Event()
            self.loaded.record(self.device_module.current_stream(device))
        else:
            self.thread_streams = None
        self.default_params = SamplingParams().to_dict()
        self.default_params["stream"] = False

    @contextmanager
    def compute_context(self) -> Iterator[None]:
        """Run one step and settle its device work before the task can move."""
        if self.thread_streams is None:
            stream = self.device_module.current_stream(self.device)
            context = nullcontext()
        else:
            stream = getattr(self.thread_streams, "stream", None)
            if stream is None:
                stream = self.thread_streams.stream = self.device_module.Stream(
                    self.device
                )
                stream.wait_event(self.loaded)
            else:
                pass
            context = self.device_module.stream(stream)
        try:
            with context:
                yield
        finally:
            stream.synchronize()


class BatchedDiarizationTask(DiarizationTask):
    def __init__(self, worker: DiarizationWorker, payload: StagePayload) -> None:
        super().__init__(worker, payload)
        self.mel_features: torch.Tensor | None = None
        self.mel_frame_count: int = 0
        self.mel_frame_offset: int = 0
        self.speaker_cache: SpeakerCache | None = None

    def prepare(self) -> None:
        if self.mel_features is not None:
            return
        else:
            pass
        waveform = self.load_waveform()
        waveform_tensor = torch.as_tensor(
            waveform, dtype=torch.float32, device=self.worker.device
        )[None]
        # note (eranik): Singleton preprocessing preserves preemphasis at each recording's boundary.
        mel_features, self.mel_frame_count = self.worker.model.preprocessor(
            waveform_tensor
        )
        self.mel_features = mel_features[:, :, : self.mel_frame_count]
        cache_encoder_frames, fifo_encoder_frames, _, _, update_encoder_frames = (
            self.worker.model.profile
        )
        self.speaker_cache = SpeakerCache(
            self.mel_features,
            cache_size=cache_encoder_frames,
            fifo_size=fifo_encoder_frames,
            update_period=update_encoder_frames,
        )

    def next_mel_chunk(self) -> tuple[torch.Tensor, int, int]:
        assert self.mel_features is not None
        _, _, chunk_encoder_frames, right_context_encoder_frames, _ = (
            self.worker.model.profile
        )
        end_mel_frame = min(
            self.mel_frame_offset + chunk_encoder_frames * MEL_FRAMES_PER_ENCODER_FRAME,
            self.mel_frame_count,
        )
        right_context_mel_frames = min(
            right_context_encoder_frames * MEL_FRAMES_PER_ENCODER_FRAME,
            self.mel_frame_count - end_mel_frame,
        )
        return (
            self.mel_features[
                :, :, self.mel_frame_offset : end_mel_frame + right_context_mel_frames
            ],
            right_context_mel_frames,
            end_mel_frame,
        )

    def finish_result(self) -> StepResult:
        probabilities = (
            np.concatenate(self.probabilities)
            if self.probabilities
            else np.empty((0, 8), dtype=np.float32)
        )
        return self.build_result(
            probabilities_to_segments(probabilities, duration_s=self.duration_s)
        )

    def close(self, *, aborted: bool) -> None:
        self.mel_features = None
        self.speaker_cache = None
        self.probabilities.clear()
        super().close(aborted=aborted)


@dataclass(kw_only=True)
class DiarizationChunk:
    request_id: str
    task: BatchedDiarizationTask
    mel_features: torch.Tensor
    speaker_cache: SpeakerCache
    right_context_mel_frames: int
    end_mel_frame: int


class DiarizationBatchRunner:
    """Return step results or exceptions for the scheduler's normal error routing."""

    def __init__(self, worker: DiarizationWorker) -> None:
        self.worker: DiarizationWorker = worker

    def supports_batching(self, task: StepTask) -> bool:
        assert isinstance(task, BatchedDiarizationTask)
        return not task.live

    @torch.inference_mode()
    def __call__(
        self, request_steps: list[BatchRequestStep]
    ) -> dict[str, StepResult | Exception]:
        assert request_steps
        first_request = request_steps[0]
        assert isinstance(first_request.task, BatchedDiarizationTask)
        if first_request.task.live:
            try:
                return {first_request.request_id: first_request.task.step()}
            except Exception as error:
                return {first_request.request_id: error}
        else:
            pass
        outcomes: dict[str, StepResult | Exception] = {}
        active_chunks: list[DiarizationChunk] = []
        with self.worker.compute_context():
            for request_step in request_steps:
                task = request_step.task
                assert isinstance(task, BatchedDiarizationTask)
                try:
                    task.prepare()
                    if task.mel_frame_count == 0:
                        outcomes[request_step.request_id] = task.finish_result()
                    else:
                        mel_features, right_context_mel_frames, end_mel_frame = (
                            task.next_mel_chunk()
                        )
                        assert task.speaker_cache is not None
                        active_chunks.append(
                            DiarizationChunk(
                                request_id=request_step.request_id,
                                task=task,
                                mel_features=mel_features,
                                speaker_cache=task.speaker_cache,
                                right_context_mel_frames=right_context_mel_frames,
                                end_mel_frame=end_mel_frame,
                            )
                        )
                except Exception as error:
                    outcomes[request_step.request_id] = error
            if active_chunks:
                try:
                    predictions = self.worker.model.forward_chunks(
                        [chunk.mel_features for chunk in active_chunks],
                        [chunk.speaker_cache for chunk in active_chunks],
                        [chunk.right_context_mel_frames for chunk in active_chunks],
                    )
                except Exception as error:
                    outcomes.update(
                        {chunk.request_id: error for chunk in active_chunks}
                    )
                else:
                    for chunk, prediction in zip(active_chunks, predictions):
                        try:
                            task = chunk.task
                            task.probabilities.append(
                                prediction[
                                    0, : chunk.end_mel_frame - task.mel_frame_offset
                                ]
                                .cpu()
                                .numpy()
                            )
                            task.mel_frame_offset = chunk.end_mel_frame
                            outcomes[chunk.request_id] = (
                                task.finish_result()
                                if chunk.end_mel_frame == task.mel_frame_count
                                else StepResult()
                            )
                        except Exception as error:
                            outcomes[chunk.request_id] = error
            else:
                pass
        return outcomes


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
    # note (eranik): Existing direct factory callers retain singleton execution by default.
    max_batch_size: int = 1,
) -> StepScheduler:
    for name, value in (
        ("max_concurrency", max_concurrency),
        ("max_live_sessions", max_live_sessions),
        ("max_running_requests", max_running_requests),
        ("max_queued_requests", max_queued_requests),
        ("max_batch_size", max_batch_size),
    ):
        if type(value) is not int or value < 1:
            raise ValueError(f"{name} must be a positive integer")
        else:
            pass
    if max_concurrency > max_running_requests:
        raise ValueError("max_concurrency cannot exceed max_running_requests")
    else:
        pass
    if max_batch_size > 1 and max_concurrency != 1:
        raise ValueError("Tensor batching uses one GPU worker; set max_concurrency=1")
    else:
        pass
    resolved_device = resolve_concrete_device(device, gpu_id)
    model = load_diarization_model(model_path, device=resolved_device, profile=profile)
    worker = DiarizationWorker(
        model, resolved_device, max_concurrency, max_live_sessions
    )

    def build_task(payload: StagePayload) -> DiarizationTask:
        if max_batch_size > 1:
            return BatchedDiarizationTask(worker, payload)
        else:
            return DiarizationTask(worker, payload)

    if max_batch_size > 1:
        return BatchedStepScheduler(
            build_task,
            run_batch=DiarizationBatchRunner(worker),
            max_batch_size=max_batch_size,
            max_running_requests=max_running_requests,
            max_queued_requests=max_queued_requests,
            shutdown_callback=worker.live_sessions.close,
        )
    else:
        return StepScheduler(
            build_task,
            max_workers=max_concurrency,
            max_running_requests=max_running_requests,
            max_queued_requests=max_queued_requests,
            shutdown_callback=worker.live_sessions.close,
        )
