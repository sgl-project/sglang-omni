# SPDX-License-Identifier: Apache-2.0
"""Omni stage boundary for standalone speaker diarization."""

from threading import local

import msgspec
import numpy as np
import torch

from sglang_omni.client.types import SamplingParams
from sglang_omni.models.nemotron_diarization.backend import (
    SAMPLE_RATE,
    NemotronDiarizer,
)
from sglang_omni.models.nemotron_diarization.streaming import LiveSessions
from sglang_omni.preprocessing.transcription import resolve_audio_source
from sglang_omni.proto import StagePayload
from sglang_omni.scheduling.simple_scheduler import SimpleScheduler
from sglang_omni.scheduling.threaded_simple_scheduler import ThreadedSimpleScheduler
from sglang_omni.utils.audio import AudioDecodeError, load_audio
from sglang_omni.utils.device import resolve_concrete_device


def create_diarization_executor(
    model_path: str,
    *,
    device: str | None = None,
    gpu_id: int | None = None,
    profile: str = "offline",
    max_concurrency: int = 1,
    max_live_sessions: int = 8,
) -> SimpleScheduler | ThreadedSimpleScheduler:
    if type(max_concurrency) is not int or max_concurrency < 1:
        raise ValueError("max_concurrency must be a positive integer")
    if type(max_live_sessions) is not int or max_live_sessions < 1:
        raise ValueError("max_live_sessions must be a positive integer")
    concrete_device = resolve_concrete_device(device, gpu_id)
    diarizer = NemotronDiarizer(model_path, device=concrete_device, profile=profile)
    live_sessions = LiveSessions(diarizer, max_live_sessions)
    worker_state = None
    if max_concurrency > 1:
        worker_state = local()
        ready = torch.cuda.Event()
        ready.record(torch.cuda.current_stream(concrete_device))
    default_params = SamplingParams().to_dict()
    default_params.pop("max_new_tokens")
    default_params["stream"] = False

    def infer(payload: StagePayload):
        task = payload.request.metadata.get("task")
        if task not in {"diarization", "diarization_stream"}:
            raise ValueError(
                "This model accepts requests through /v1/audio/diarizations"
            )
        unsupported = [
            name
            for name, value in payload.request.params.items()
            if name not in default_params or value != default_params[name]
        ]
        if unsupported:
            raise ValueError(f"Unsupported diarization controls: {sorted(unsupported)}")
        if task == "diarization_stream":
            return live_sessions.compute(payload.request.inputs)
        try:
            waveform = load_audio(
                resolve_audio_source(payload),
                source_name="diarization",
                target_sample_rate=SAMPLE_RATE,
            )
        except AudioDecodeError as exc:
            raise ValueError("could not decode the uploaded audio") from exc
        if waveform.ndim != 1 or waveform.size == 0 or not np.isfinite(waveform).all():
            raise ValueError(
                "could not decode the uploaded audio: empty or non-finite waveform"
            )
        return diarizer.diarize(waveform)

    def compute(payload: StagePayload) -> StagePayload:
        if worker_state is None:
            result = infer(payload)
        else:
            stream = getattr(worker_state, "stream", None)
            if stream is None:
                stream = torch.cuda.Stream(device=concrete_device)
                stream.wait_event(ready)
                worker_state.stream = stream
            try:
                with torch.cuda.stream(stream):
                    result = infer(payload)
            finally:
                # Keep this worker's slot occupied until its GPU work ends,
                # including when inference raises or the client disconnects.
                stream.synchronize()
        return StagePayload(
            request_id=payload.request_id,
            request=payload.request,
            data={"diarization": msgspec.to_builtins(result)},
        )

    if max_concurrency == 1:
        return SimpleScheduler(compute)
    # Reuse each thread's stream: cuBLAS keeps workspaces per thread/stream pair.
    return ThreadedSimpleScheduler(compute, max_concurrency=max_concurrency)
