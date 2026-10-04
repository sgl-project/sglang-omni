# SPDX-License-Identifier: Apache-2.0
"""YuE2 stage factories."""

from __future__ import annotations

import logging

import torch
from sglang_omni.platforms import current_platform
from sglang_omni.proto import StagePayload
from sglang_omni.scheduling.pipeline_state import load_state, store_state
from sglang_omni.scheduling.simple_scheduler import SimpleScheduler
from sglang_omni.utils.audio_payload import audio_waveform_payload

from .constants import OUTPUT_SAMPLE_RATE
from .payload_types import Yue2State
from .request_builders import build_yue2_state
from .synth import Yue2Synthesizer

logger = logging.getLogger(__name__)


def create_preprocessing_executor(
    model_path: str,
) -> SimpleScheduler[StagePayload, StagePayload]:
    del model_path

    def _preprocess(payload: StagePayload) -> StagePayload:
        state = build_yue2_state(payload)
        logger.info(
            f"YuE2 preprocessing request={payload.request_id} cot={state.cot} "
            f"seed={state.seed} semantic_max_tokens={state.semantic_max_tokens}"
        )
        return store_state(payload, state)

    return SimpleScheduler(_preprocess, max_concurrency=1)


def create_synth_executor(
    model_path: str,
    *,
    gpu_id: int | None = None,
    device: str | None = None,
) -> SimpleScheduler[StagePayload, StagePayload]:
    if not current_platform.is_cuda():
        raise RuntimeError("YuE2 requires a CUDA backend")
    else:
        pass
    torch.backends.cudnn.enabled = False
    torch.backends.cuda.enable_cudnn_sdp(False)

    from sglang_omni.utils.device import resolve_concrete_device

    device = str(resolve_concrete_device(device, gpu_id))
    synthesizer = Yue2Synthesizer(model_path, device=device)

    def _synth(payload: StagePayload) -> StagePayload:
        state = load_state(payload, Yue2State)
        waveform, seconds = synthesizer.synthesize(state)
        data = dict(
            audio_waveform_payload(
                waveform,
                sample_rate=OUTPUT_SAMPLE_RATE,
                modality="audio",
                source_hint="YuE2",
                keep_channels=True,
            )
        )
        data["finish_reason"] = state.finish_reason or "stop"
        logger.info(
            f"YuE2 synth done request={payload.request_id} "
            f"samples={waveform.shape[-1]} duration={waveform.shape[-1] / OUTPUT_SAMPLE_RATE:.2f}s "
            f"elapsed={seconds:.2f}s"
        )
        return StagePayload(
            request_id=payload.request_id, request=payload.request, data=data
        )

    return SimpleScheduler(_synth, max_concurrency=1)


__all__ = ["create_preprocessing_executor", "create_synth_executor"]
