# SPDX-License-Identifier: Apache-2.0
"""YuE2 stage factories: SGLang AR engine and acoustic rendering."""

from collections.abc import Mapping

import torch

from sglang_omni.models.yue2.payload_types import Yue2State
from sglang_omni.models.yue2.runtime import Yue2Runtime
from sglang_omni.proto.request import StagePayload
from sglang_omni.scheduling.pipeline_state import load_state
from sglang_omni.scheduling.simple_scheduler import SimpleScheduler
from sglang_omni.utils.audio_payload import audio_waveform_payload
from sglang_omni.utils.device import resolve_concrete_device


def cuda_device(device: str | None, gpu_id: int | None) -> torch.device:
    resolved = torch.device(resolve_concrete_device(device, gpu_id))
    if resolved.type != "cuda":
        raise ValueError("YuE2 serving requires a CUDA device")
    else:
        return resolved


def create_ar_executor(
    model_path: str,
    *,
    gpu_id: int | None = None,
    device: str | None = None,
    max_concurrency: int = 16,
    server_args_overrides: Mapping[str, object] | None = None,
):
    cuda_device(device, gpu_id)
    # note (ruiling): imported here so only the AR process loads SGLang.
    from sglang_omni.models.yue2.engine_builder import Yue2EngineBuilder

    builder = Yue2EngineBuilder(max_running_requests=max_concurrency)
    return builder.build(
        model_path,
        device=device,
        gpu_id=gpu_id,
        dtype="bfloat16",
        server_args_overrides=server_args_overrides,
    )


class Yue2AcousticScheduler(SimpleScheduler[StagePayload, StagePayload]):
    def __init__(self, runtime: Yue2Runtime) -> None:
        self.runtime = runtime
        super().__init__(self.render)

    def render(self, payload: StagePayload) -> StagePayload:
        state = load_state(payload, Yue2State)
        audio = self.runtime.render(
            state.semantic_prefix,
            state.semantic_tokens,
            state.seed,
            state.ode_steps,
            lambda: self.is_aborted(payload.request_id),
        )
        result = audio_waveform_payload(
            audio,
            sample_rate=self.runtime.vae.config.sample_rate,
            modality="audio",
            source_hint="YuE2",
            keep_channels=True,
        )
        result["finish_reason"] = state.finish_reason
        return StagePayload(
            request_id=payload.request_id, request=payload.request, data=result
        )


def create_acoustic_executor(
    model_path: str,
    *,
    vae_path: str,
    vae_core_frames: int,
    vae_halo_frames: int,
    nar_cuda_graph: bool,
    fuse_projections: bool,
    fused_kernels: bool,
    gpu_id: int | None = None,
    device: str | None = None,
) -> Yue2AcousticScheduler:
    runtime = Yue2Runtime.from_pretrained(
        model_path,
        vae_path,
        cuda_device(device, gpu_id),
        vae_core_frames=vae_core_frames,
        vae_halo_frames=vae_halo_frames,
        nar_cuda_graph=nar_cuda_graph,
        fuse_projections=fuse_projections,
        fused_kernels=fused_kernels,
    )
    return Yue2AcousticScheduler(runtime)
