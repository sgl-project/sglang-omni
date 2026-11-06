# SPDX-License-Identifier: Apache-2.0
"""YuE2 synthesis with cooperative request cancellation."""

from functools import partial

import torch

from sglang_omni.models.yue2.request import Yue2Request, build_request
from sglang_omni.models.yue2.runtime import SongResult, Yue2Runtime
from sglang_omni.proto.request import StagePayload
from sglang_omni.scheduling.simple_scheduler import SimpleScheduler
from sglang_omni.utils.audio_payload import audio_waveform_payload
from sglang_omni.utils.device import resolve_concrete_device


class Yue2Scheduler(SimpleScheduler[StagePayload, StagePayload]):
    def __init__(
        self,
        runtime: Yue2Runtime,
        *,
        max_batch_size: int,
        max_batch_wait_ms: int,
        batch_graph_cache_size: int,
    ) -> None:
        self.runtime = runtime
        self.batch_graph_cache_size = batch_graph_cache_size
        super().__init__(
            self.generate,
            batch_compute_fn=self.generate_batch,
            max_batch_size=max_batch_size,
            max_batch_wait_ms=max_batch_wait_ms,
        )

    def generate(self, payload: StagePayload) -> StagePayload:
        try:
            request = build_request(payload)
        except ValueError as error:
            raise ValueError(f"YuE2 request: {error}") from error
        song = self.runtime.generate(
            request, lambda: self.is_aborted(payload.request_id)
        )
        return self.output(payload, song)

    def output(self, payload: StagePayload, song: SongResult) -> StagePayload:
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

    def generate_batch(
        self, payloads: list[StagePayload]
    ) -> list[StagePayload | Exception]:
        results: dict[int, StagePayload | Exception] = {}
        requests: list[Yue2Request] = []
        indices: list[int] = []
        for index, payload in enumerate(payloads):
            try:
                request = build_request(payload)
            except ValueError as error:
                results[index] = ValueError(f"YuE2 request: {error}")
            else:
                indices.append(index)
                requests.append(request)
        songs = self.runtime.generate_batch(
            requests,
            [partial(self.is_aborted, payloads[index].request_id) for index in indices],
            self.batch_graph_cache_size,
        )
        for index, song in zip(indices, songs):
            if isinstance(song, Exception):
                results[index] = song
            else:
                results[index] = self.output(payloads[index], song)
        return [results[index] for index in range(len(payloads))]


def create_synthesis_executor(
    model_path: str,
    *,
    vae_path: str,
    vae_core_frames: int,
    vae_halo_frames: int,
    ar_cuda_graph: bool,
    nar_cuda_graph: bool,
    sampling_cuda_graph: bool,
    fuse_projections: bool,
    max_batch_size: int,
    max_batch_wait_ms: int,
    batch_graph_cache_size: int,
    gpu_id: int | None = None,
    device: str | None = None,
) -> Yue2Scheduler:
    resolved_device = torch.device(resolve_concrete_device(device, gpu_id))
    if resolved_device.type != "cuda":
        raise ValueError("YuE2 serving requires a CUDA device")
    elif max_batch_size > 1 and not (ar_cuda_graph and sampling_cuda_graph):
        raise ValueError("YuE2 batching requires AR and sampling CUDA graphs")
    else:
        pass
    runtime = Yue2Runtime.from_pretrained(
        model_path,
        vae_path,
        resolved_device,
        vae_core_frames=vae_core_frames,
        vae_halo_frames=vae_halo_frames,
        ar_cuda_graph=ar_cuda_graph,
        nar_cuda_graph=nar_cuda_graph,
        sampling_cuda_graph=sampling_cuda_graph,
        fuse_projections=fuse_projections,
    )
    return Yue2Scheduler(
        runtime,
        max_batch_size=max_batch_size,
        max_batch_wait_ms=max_batch_wait_ms,
        batch_graph_cache_size=batch_graph_cache_size,
    )
