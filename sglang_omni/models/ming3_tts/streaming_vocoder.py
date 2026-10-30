# SPDX-License-Identifier: Apache-2.0
"""Ming 3 payload adapter for shared batched AudioVAE streaming."""

from dataclasses import replace

import torch

from sglang_omni.modules.ming_audio.audio_decode import MingAudioDecoder
from sglang_omni.modules.ming_audio.streaming_vocoder import (
    AudioVAEStreamingVocoder,
    FullDecode,
    StreamState,
)
from sglang_omni.modules.ming_audio.vae.modeling_audio_vae import AudioVAE
from sglang_omni.pipeline.stage.stream_queue import StreamItem
from sglang_omni.proto.request import StagePayload


class Ming3StreamingVocoder(AudioVAEStreamingVocoder):
    def __init__(
        self,
        compute: FullDecode,
        *,
        vae: AudioVAE,
        sample_rate: int,
        stream_chunk_steps: int,
        stream_capacity: int = 32,
        streaming_cuda_graph: bool = False,
    ) -> None:
        if stream_chunk_steps < 1:
            raise ValueError("stream_chunk_steps must be positive")
        else:
            pass
        self.vae = vae
        decoder = MingAudioDecoder(
            vae,
            stream_capacity=stream_capacity,
            max_stream_step_latents=stream_chunk_steps,
            streaming_cuda_graph_required=streaming_cuda_graph,
        )
        super().__init__(
            decoder,
            patch_size=1,
            latent_dim=vae.decoder.latent_dim,
            initial_chunk_patches=stream_chunk_steps,
            steady_chunk_patches=stream_chunk_steps,
            compute=compute,
            source_hint="Ming-3.0-TTS",
        )

    def ingest_stream_item(
        self, request_id: str, item: StreamItem
    ) -> StreamState | None:
        # note (Zhuofeng Li): Ming 3 sends terminal state through stream_done after sampling the final latent.
        item = replace(item, metadata={**(item.metadata or {}), "is_last": False})
        return super().ingest_stream_item(request_id, item)

    def has_executable_work(self, state: StreamState) -> bool:
        if state.terminal_received:
            return super().has_executable_work(state)
        else:
            # note (Zhuofeng Li): Retain one latent for the terminal interpolation and ISTFT transition.
            return len(state.pending_patches) > self.next_chunk_patches(state)

    def decode_delta(
        self, request_id: str, state: StreamState, *, is_final: bool
    ) -> torch.Tensor | None:
        if not is_final:
            return None
        else:
            state.terminal_received = True
        if self.slot_bindings.try_bind(request_id) is None:
            return None
        else:
            pass
        waveforms = []
        while not state.terminal_committed:
            if not state.pending_patches:
                raise RuntimeError(f"Ming 3 stream {request_id} ended without latents")
            else:
                pass
            participants = [(request_id, state)]
            decoded = self.run_step(participants, self.build_step_plan(participants))
            waveform = decoded.get(request_id)
            if waveform is not None:
                waveforms.append(waveform)
            else:
                pass
        return torch.cat(waveforms) if waveforms else None

    def final_result_data(
        self, request_id: str, payload: StagePayload, state: StreamState
    ) -> dict[str, int | float | str | dict[str, int] | None]:
        if state.emitted_samples == 0:
            raise RuntimeError(f"Ming 3 stream {request_id} completed without audio")
        else:
            return {
                "modality": "audio",
                "sample_rate": self.sample_rate,
                "duration_s": state.emitted_samples / self.sample_rate,
                "finish_reason": payload.data["finish_reason"],
                "usage": payload.data["usage"],
            }
