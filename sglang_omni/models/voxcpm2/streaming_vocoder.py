# SPDX-License-Identifier: Apache-2.0
"""VoxCPM2 streaming vocoder: decodes latent patches as the engine emits them."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import torch

from sglang_omni.models.voxcpm2 import constants as C
from sglang_omni.models.voxcpm2.components.audio_vae import AudioVAE
from sglang_omni.models.voxcpm2.payload_types import VoxCPM2State
from sglang_omni.proto import StagePayload
from sglang_omni.scheduling.pipeline_state import build_usage, load_state, store_state
from sglang_omni.scheduling.streaming_vocoder import StreamingVocoderBase
from sglang_omni.utils.audio_payload import audio_waveform_payload

logger = logging.getLogger(__name__)


@dataclass
class VoxCPM2StreamState:
    patches: list[torch.Tensor] = field(default_factory=list)
    emitted_patches: int = 0
    next_decode_patches: int = 0
    context_len: int | None = None


class VoxCPM2StreamingVocoder(StreamingVocoderBase[VoxCPM2StreamState, None]):
    """Re-decodes a trailing window of patches per chunk instead of keeping state.

    The decoder is causal, so a chunk decoded on its own starts from silence and
    seams audibly. Prepending already-emitted patches and dropping their samples
    from the output gives the decoder that context back; the repeated patches are
    the price of not threading per-request convolution state through it.
    """

    def __init__(
        self,
        audio_vae: AudioVAE,
        *,
        device: str,
        patch_size: int,
        stream_stride: int = 8,
        stream_followup_stride: int = 4,
        overlap_patches: int = C.DEFAULT_STREAMING_PREFIX_LEN - 1,
        max_batch_size: int = 1,
    ) -> None:
        if min(stream_stride, stream_followup_stride) <= 0:
            raise ValueError("VoxCPM2 stream strides must be positive")
        if overlap_patches < 0:
            raise ValueError("VoxCPM2 overlap_patches must be >= 0")
        self.vae = audio_vae
        self.device = device
        self.patch_size = int(patch_size)
        self.stride = int(stream_stride)
        self.followup_stride = int(stream_followup_stride)
        self.overlap = int(overlap_patches)
        self.samples_per_patch = int(audio_vae.decode_chunk_size) * int(patch_size)
        super().__init__(
            self.decode_payload,
            sample_rate=int(audio_vae.out_sample_rate),
            stream_source_hint="VoxCPM2",
            stream_input_modality="audio_latents",
            max_batch_size=max_batch_size,
        )

    def warmup_now(self) -> None:
        """Warm up the audio decoder before accepting requests.

        note (Xinhao Tan): TorchScript optimizes the decoder's Snake activation
        after its initial calls, which can change floating-point results.
        Identical audio latents can then decode to different waveforms before
        and after warmup. Decode dummy latents at startup so the first request
        also uses the warmed-up computation.
        """
        latents = torch.zeros(
            (1, self.vae.latent_dim, self.patch_size * self.stride),
            device=self.device,
            dtype=torch.float32,
        )
        for _ in range(C.WARMUP_ITERATIONS):
            self.vae.decode(latents)

    def create_stream_state(self, request_id: str) -> VoxCPM2StreamState:
        return VoxCPM2StreamState()

    def latch_stream_contract(
        self,
        request_id: str,
        state: VoxCPM2StreamState,
        source: StagePayload | Mapping[str, Any],
        *,
        origin: str,
    ) -> None:
        metadata = source.data if origin == "payload" else source
        if "context_len" not in metadata:
            return
        count = metadata["context_len"]
        if type(count) is not int or count < 0:
            raise ValueError("VoxCPM2 context_len must be a nonnegative integer")
        if state.context_len is None:
            if state.patches or state.emitted_patches:
                raise ValueError(
                    "VoxCPM2 context_len must arrive before latent patches"
                )
            state.context_len = count
            state.emitted_patches = count
            state.next_decode_patches = count + self.stride
        else:
            if count != state.context_len:
                raise ValueError(f"VoxCPM2 context_len changed for {request_id!r}")

    def validate_chunk(
        self, request_id: str, state: VoxCPM2StreamState, codes: torch.Tensor
    ) -> torch.Tensor:
        if codes.ndim != 2 or codes.shape[0] != self.patch_size:
            raise ValueError(
                "VoxCPM2 latent chunks must have shape "
                f"[{self.patch_size}, feat_dim], got {list(codes.shape)}"
            )
        return codes

    def ingest(
        self, request_id: str, state: VoxCPM2StreamState, codes: torch.Tensor
    ) -> None:
        state.patches.append(codes)

    def should_decode(self, state: VoxCPM2StreamState, *, is_final: bool) -> bool:
        return len(state.patches) >= (state.next_decode_patches or self.stride)

    def decode_delta(
        self, request_id: str, state: VoxCPM2StreamState, *, is_final: bool
    ) -> torch.Tensor | None:
        total = len(state.patches)
        if total <= state.emitted_patches:
            return None
        audio = self.decode_window(state)
        if is_final:
            return audio
        else:
            state.next_decode_patches = total + self.followup_stride
        return audio

    def final_result_data(
        self, request_id: str, payload: StagePayload, state: VoxCPM2StreamState
    ) -> dict[str, Any]:
        # note (Xinhao Tan): a streaming request already received its audio as
        # chunks, so the terminal result carries only what the client needs to
        # close the stream. Putting the waveform here too would play the whole
        # utterance a second time.
        result: dict[str, Any] = {
            "modality": "audio",
            "sample_rate": self.sample_rate,
        }
        usage = build_usage(load_state(payload, VoxCPM2State))
        if usage is not None:
            result["usage"] = usage
        return result

    def fallback_full_decode(
        self, request_id: str, payload: StagePayload, state: VoxCPM2StreamState
    ) -> torch.Tensor | None:
        tts_state = load_state(payload, VoxCPM2State)
        if tts_state.generated_latents is None:
            return None
        return self.decode_latents(tts_state.generated_latents)[
            ..., tts_state.context_len * self.samples_per_patch :
        ].squeeze(1)

    def decode_latents(self, latents: Any) -> torch.Tensor:
        tensor = torch.as_tensor(latents)
        tensor = tensor.unsqueeze(0) if tensor.ndim == 2 else tensor
        return self.vae.decode(tensor.to(device=self.device, dtype=torch.float32))

    def decode_window(self, state: VoxCPM2StreamState) -> torch.Tensor | None:
        total = len(state.patches)
        window_start = (
            0
            if state.emitted_patches == (state.context_len or 0)
            else max(0, state.emitted_patches - self.overlap)
        )
        stacked = torch.stack(state.patches[window_start:], dim=0)
        latents = stacked.permute(2, 0, 1).reshape(1, stacked.shape[2], -1)
        latents = latents.to(device=self.device, dtype=torch.float32)
        audio = self.vae.decode(latents).squeeze(0).squeeze(0)
        drop = (state.emitted_patches - window_start) * self.samples_per_patch
        state.emitted_patches = total
        if audio.shape[-1] <= drop:
            return None
        return audio[drop:].detach().cpu()

    def decode_payload(self, payload: StagePayload) -> StagePayload:
        """Non-streaming path: decode everything the engine accumulated."""
        state = load_state(payload, VoxCPM2State)
        if state.generated_latents is None:
            raise ValueError("VoxCPM2 vocoder received a payload without latents")
        audio = self.decode_latents(state.generated_latents)
        audio = audio[..., state.context_len * self.samples_per_patch :]
        state.generated_latents = None
        state.sample_rate = state.out_sample_rate
        payload = store_state(payload, state)
        # note (Xinhao Tan): the payload crosses a msgpack control-plane hop, so
        # the waveform goes through the shared serializer rather than riding as
        # a tensor. A tensor here fails inside send_complete, far from this line.
        payload.data.update(
            audio_waveform_payload(
                audio.squeeze(1),
                sample_rate=state.sample_rate,
                modality="audio",
                source_hint="VoxCPM2",
            )
        )
        return payload


__all__ = ["VoxCPM2StreamingVocoder"]
