# SPDX-License-Identifier: Apache-2.0
"""Ming 3 payload adapter for shared batched AudioVAE streaming."""

import logging
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
from sglang_omni.proto.streams import SESSION_CONTROL
from sglang_omni.scheduling.message import OutgoingMessage

logger = logging.getLogger(__name__)


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
        # note (guozhihao): Each Ming3 message carries one latent; budget a chunk's messages per slot without waiting for new arrivals.
        self.stream_chunk_batch_max = stream_capacity * stream_chunk_steps

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

    def on_stream_done_before_payload(self, request_id: str) -> list[OutgoingMessage]:
        """Make the tail runnable before the terminal payload arrives."""
        stream_state = self.get_or_create_stream_state(request_id)
        if stream_state is not None:
            stream_state.terminal_received = True
        else:
            pass
        return []

    def on_stream_done(self, request_id: str) -> list[OutgoingMessage] | None:
        """Defer the tail so ready streams can share its decoder step."""
        stream_state = self.get_or_create_stream_state(request_id)
        session_operation = self.session_appends.get(request_id)
        if stream_state is None:
            return []
        elif (
            session_operation is not None
            and session_operation.chunk.format == SESSION_CONTROL
        ):
            return super().on_stream_done(request_id)
        elif stream_state.terminal_committed or not stream_state.pending_patches:
            return super().on_stream_done(request_id)
        elif (
            self.slot_bindings.slot_for(request_id) is None
            and not self.slot_bindings.free_slots
        ):
            return super().on_stream_done(request_id)
        else:
            stream_state.terminal_received = True
            return None

    def has_ready_work(self) -> bool:
        with self.state_lock:
            if self.pending_release_ids:
                return True
            else:
                return any(
                    not self.is_aborted(request_id)
                    and self.has_executable_work(stream_state)
                    and (
                        self.slot_bindings.slot_for(request_id) is not None
                        or bool(self.slot_bindings.free_slots)
                    )
                    for request_id, stream_state in self.stream_state_items()
                )

    def pump_one_step(self) -> list[str] | None:
        """Publish terminal results after the batch's final audio."""
        failed_request_ids = super().pump_one_step()
        if failed_request_ids is None or failed_request_ids:
            return failed_request_ids
        else:
            pass
        for request_id, stream_state in self.stream_state_items():
            if (
                not stream_state.terminal_committed
                or request_id not in self.stream_payloads
            ):
                continue
            else:
                pass
            try:
                terminal_messages = self.finish_stream(request_id)
            except Exception as exc:
                logger.exception(f"Ming 3 stream finalization failed for {request_id}")
                self.emit_error(request_id, exc)
                self.abort_state(request_id)
                failed_request_ids.append(request_id)
            else:
                self.complete_stream_request(request_id, terminal_messages)
        return failed_request_ids

    def decode_delta(
        self, request_id: str, state: StreamState, *, is_final: bool
    ) -> torch.Tensor | None:
        if not is_final or state.terminal_committed:
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
