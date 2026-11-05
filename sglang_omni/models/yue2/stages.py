# SPDX-License-Identifier: Apache-2.0
"""YuE2 synthesis with cooperative request cancellation."""

import torch

from sglang_omni.models.yue2.request import build_request
from sglang_omni.models.yue2.runtime import Yue2Runtime
from sglang_omni.proto.request import StagePayload
from sglang_omni.scheduling.simple_scheduler import SimpleScheduler
from sglang_omni.utils.audio_payload import audio_waveform_payload
from sglang_omni.utils.device import resolve_concrete_device


class Yue2Scheduler(SimpleScheduler[StagePayload, StagePayload]):
    def __init__(self, runtime: Yue2Runtime) -> None:
        self.runtime = runtime
        super().__init__(self.generate)

    def generate(self, payload: StagePayload) -> StagePayload:
        try:
            request = build_request(payload)
        except ValueError as error:
            raise ValueError(f"YuE2 request: {error}") from error
        song = self.runtime.generate(
            request, lambda: self.is_aborted(payload.request_id)
        )
        result = audio_waveform_payload(
            song.audio,
            sample_rate=song.sample_rate,
            modality="audio",
            source_hint="YuE2",
            keep_channels=True,
        )
        result["finish_reason"] = song.finish_reason
        return StagePayload(
            request_id=payload.request_id, request=payload.request, data=result
        )


def create_synthesis_executor(
    model_path: str,
    *,
    vae_path: str,
    vae_core_frames: int,
    vae_halo_frames: int,
    gpu_id: int | None = None,
    device: str | None = None,
) -> Yue2Scheduler:
    resolved_device = torch.device(resolve_concrete_device(device, gpu_id))
    if resolved_device.type != "cuda":
        raise ValueError("YuE2 serving requires a CUDA device")
    else:
        pass
    runtime = Yue2Runtime.from_pretrained(
        model_path,
        vae_path,
        resolved_device,
        vae_core_frames=vae_core_frames,
        vae_halo_frames=vae_halo_frames,
    )
    return Yue2Scheduler(runtime)
