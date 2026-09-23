# SPDX-License-Identifier: Apache-2.0
"""CPU stage-boundary tensors around native MLX AudioVAE execution."""

from __future__ import annotations

from collections.abc import Sequence

import mlx.core as mx
import numpy as np
import torch

from sglang_omni.models.ming_tts.reference_encode import (
    MingSpeakerEmbeddingExtractor,
    MingTTSReferenceEncoder,
    MingTTSReferenceEncodeHook,
)
from sglang_omni.scheduling.reference_encoder import ReferenceEncodeService

from .audio_vae import AudioDecoderState, AudioVAE


class MingTTSMlxReferenceEncoder(MingTTSReferenceEncoder):
    def __init__(
        self,
        audio_vae: AudioVAE,
        speaker_encoder: MingSpeakerEmbeddingExtractor,
        *,
        patch_size: int,
        cache_model_identity: str | None,
        cache_max_items: int,
        cache_max_bytes: int,
    ) -> None:
        self._audio_vae = audio_vae
        self.sample_rate = int(audio_vae.config["sample_rate"])
        self.patch_size = patch_size
        self.speaker_encoder = speaker_encoder
        self.device = torch.device("cpu")
        self.dtype = torch.float32
        self._service = None
        if cache_model_identity is not None:
            self._service = ReferenceEncodeService(
                MingTTSReferenceEncodeHook(self, model_identity=cache_model_identity + ":mlx"),
                max_items=cache_max_items, max_bytes=cache_max_bytes,
                log_prefix="Ming MLX ref cache",
            )

    def encode_reference(self, ref_audio: str) -> dict[str, object]:
        waveform, speaker_waveform = self.load_reference_waveform(ref_audio)
        waveform = self.pad_waveform(waveform)
        with mx.stream(mx.new_thread_local_stream(mx.gpu)):
            latent = self._audio_vae.encode_latent(mx.array(waveform.float().numpy()))
            prompt_latent = torch.from_numpy(np.array(latent.astype(mx.float32)))
        return {
            "spk_emb": self.speaker_encoder(speaker_waveform).cpu().float(),
            "prompt_latent": prompt_latent,
            "prompt_latent_token_count": prompt_latent.shape[1] // self.patch_size,
        }


class MingMlxAudioDecoder:
    def __init__(self, audio_vae: AudioVAE, *, stream_capacity: int = 1) -> None:
        self._audio_vae = audio_vae
        self.sample_rate = int(audio_vae.config["sample_rate"])
        self.stream_capacity = stream_capacity
        self.streaming_ready = False
        self._states: dict[int, AudioDecoderState] = {}

    def prepare_streaming(self) -> None:
        self.streaming_ready = True

    def decode_full(self, latents: torch.Tensor) -> torch.Tensor:
        if latents.numel() == 0:
            return torch.empty(0, dtype=torch.float32)
        with mx.stream(mx.new_thread_local_stream(mx.gpu)):
            latent = mx.array(latents.detach().cpu().float().numpy()).reshape(1, -1, latents.shape[-1])
            audio, _ = self._audio_vae.decoder(latent)
            return torch.from_numpy(np.array(audio[0].astype(mx.float32)))

    def run_streaming(
        self, *, slot_ids: tuple[int, ...],
        patch_groups: tuple[tuple[torch.Tensor, ...], ...],
        terminal_flags: tuple[bool, ...],
    ) -> tuple[torch.Tensor, ...]:
        waveforms = []
        try:
            with mx.stream(mx.new_thread_local_stream(mx.gpu)):
                for slot, patches, terminal in zip(slot_ids, patch_groups, terminal_flags, strict=True):
                    latent = torch.cat(patches, dim=0).detach().cpu().float().numpy()
                    audio, state = self._audio_vae.decoder(
                        mx.array(latent)[None], state=self._states.get(slot),
                        streaming=True, last_chunk=terminal,
                    )
                    waveform = torch.from_numpy(np.array(audio[0].astype(mx.float32)))
                    if terminal:
                        self._states.pop(slot, None)
                    else:
                        self._states[slot] = state
                        arrays = [v for c in state.cache for v in (c.keys, c.values) if v is not None]
                        if state.upsample is not None:
                            arrays.append(state.upsample.pending)
                            if state.upsample.left is not None:
                                arrays.append(state.upsample.left)
                        if state.overlap is not None:
                            arrays.extend(state.overlap)
                        mx.eval(arrays)
                    waveforms.append(waveform)
        except Exception:
            self.reset_stream_rows(slot_ids)
            raise
        return tuple(waveforms)

    def reset_stream_rows(self, slot_ids: Sequence[int]) -> None:
        for slot in slot_ids:
            self._states.pop(slot, None)

    def reset_all_stream_rows(self) -> None:
        self._states.clear()

    def close(self) -> None:
        self.reset_all_stream_rows()
        self.streaming_ready = False
