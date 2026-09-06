# SPDX-License-Identifier: Apache-2.0
"""Conservative final-only Qwen3-TTS vocoder for Torch MPS."""

from __future__ import annotations

from typing import Any

import torch

from sglang_omni.models.qwen3_tts.payload_types import Qwen3TTSState
from sglang_omni.proto import StagePayload
from sglang_omni.scheduling.pipeline_state import build_usage
from sglang_omni.scheduling.simple_scheduler import SimpleScheduler
from sglang_omni.utils.audio_payload import audio_waveform_payload


def create_torch_mps_vocoder_scheduler(tokenizer: Any) -> SimpleScheduler:
    """Decode one complete codec sequence without CUDA streaming machinery."""

    def vocode(payload: StagePayload) -> StagePayload:
        params = payload.request.params
        if isinstance(params, dict) and params.get("stream", False):
            raise ValueError(
                "Qwen3-TTS Torch MPS currently supports non-streaming only"
            )
        state = Qwen3TTSState.from_dict(payload.data)
        if state.audio_codes is None:
            raise RuntimeError("Qwen3-TTS vocoder requires audio_codes from tts_engine")
        codes = torch.as_tensor(state.audio_codes, dtype=torch.long)
        waveforms, sample_rate = tokenizer.decode([{"audio_codes": codes}])
        if len(waveforms) != 1 or waveforms[0] is None:
            raise RuntimeError("Qwen3-TTS speech tokenizer did not return audio")
        waveform = waveforms[0]
        if state.ref_code_len:
            cut = int(state.ref_code_len / max(len(codes), 1) * int(waveform.shape[0]))
            waveform = waveform[cut:]
        data = audio_waveform_payload(
            waveform,
            sample_rate=int(sample_rate),
            modality="audio",
            source_hint="Qwen3-TTS Torch MPS",
        )
        usage = build_usage(state)
        if usage is not None:
            data["usage"] = usage
        payload.data = data
        return payload

    return SimpleScheduler(vocode)


__all__ = ["create_torch_mps_vocoder_scheduler"]
