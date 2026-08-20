# SPDX-License-Identifier: Apache-2.0
"""Audio decode helpers for Ming-Omni-TTS."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any

import torch
from transformers.cache_utils import Cache

from sglang_omni.models.ming_omni.talker.audio_vae.modeling_audio_vae import AudioVAE
from sglang_omni.models.ming_tts.audio_config import AudioVAEconfig
from sglang_omni.models.ming_tts.payload_types import (
    load_ming_tts_state,
    store_ming_tts_state,
)
from sglang_omni.proto import StagePayload
from sglang_omni.scheduling.pipeline_state import build_usage
from sglang_omni.utils.audio_payload import audio_waveform_payload


@dataclass
class MingAudioDecoderState:
    dynamic_cache: Cache | None = None
    upsample_state: dict[str, Any] | None = None
    audio_buffer: torch.Tensor | None = None
    window_buffer: torch.Tensor | None = None


class MingAudioDecoder(torch.nn.Module):
    """Official-path AudioVAE decoder wrapper."""

    def __init__(self, audio_vae: AudioVAE, *, sample_rate: int) -> None:
        super().__init__()
        self.audio_vae = audio_vae
        self.sample_rate = int(sample_rate)

    @classmethod
    def from_config(
        cls,
        audio_config: AudioVAEconfig,
        *,
        device: str | torch.device = "cuda:0",
        dtype: str | torch.dtype = "bfloat16",
    ) -> "MingAudioDecoder":
        if getattr(audio_config, "semantic_module_kwargs", None) is not None:
            raise ValueError(
                "Ming-Omni-TTS serving currently uses the talker AudioVAE "
                "encode/decode path and does not support semantic_module_kwargs"
            )

        if isinstance(dtype, torch.dtype):
            torch_dtype = dtype
        elif dtype == "auto":
            torch_dtype = torch.bfloat16
        elif isinstance(dtype, str):
            value = dtype.removeprefix("torch.")
            torch_dtype = getattr(torch, value, None)
            if not isinstance(torch_dtype, torch.dtype):
                raise ValueError(f"Unsupported Ming-Omni-TTS AudioVAE dtype: {dtype!r}")
        else:
            raise TypeError(f"Unsupported Ming-Omni-TTS AudioVAE dtype: {dtype!r}")

        model = AudioVAE(audio_config).eval()
        model.to(device=torch.device(device), dtype=torch_dtype)
        return cls(model, sample_rate=int(audio_config.sample_rate))

    @property
    def device(self) -> torch.device:
        return next(self.audio_vae.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return next(self.audio_vae.parameters()).dtype

    @torch.inference_mode()
    def decode_streaming_step(
        self,
        latent_sequence: torch.Tensor,
        *,
        state: MingAudioDecoderState,
        is_last: bool,
    ) -> torch.Tensor:
        device = self.device
        dtype = self.dtype
        latent_sequence = latent_sequence.to(
            device=device,
            dtype=dtype,
        ).unsqueeze(0)
        context = (
            torch.autocast(device_type="cuda", dtype=dtype)
            if device.type == "cuda" and dtype in (torch.float16, torch.bfloat16)
            else nullcontext()
        )
        with context:
            waveform, stream_state, dynamic_cache = self.audio_vae.decode(
                latent_sequence,
                past_key_values=state.dynamic_cache,
                use_cache=True,
                stream_state=(
                    state.upsample_state,
                    state.audio_buffer,
                    state.window_buffer,
                ),
                last_chunk=is_last,
            )

        state.dynamic_cache = dynamic_cache
        state.upsample_state, state.audio_buffer, state.window_buffer = stream_state
        return waveform[0, 0].detach()

    @torch.inference_mode()
    def decode_nonstreaming(
        self,
        latents: torch.Tensor,
    ) -> torch.Tensor:
        if int(latents.shape[0]) == 0:
            return latents.new_empty((0,), dtype=torch.float32)

        device = self.device
        dtype = self.dtype
        context = (
            torch.autocast(device_type="cuda", dtype=dtype)
            if device.type == "cuda" and dtype in (torch.float16, torch.bfloat16)
            else nullcontext()
        )
        with context:
            latents = latents.to(device=device, dtype=dtype)
            sequence = latents.reshape(1, -1, latents.shape[-1])
            waveform, _, _ = self.audio_vae.decode(
                sequence,
                past_key_values=None,
                use_cache=False,
                stream_state=(None, None, None),
                last_chunk=True,
            )

        return waveform[0, 0].detach()


def decode_ming_tts_audio_payload(
    payload: StagePayload,
    decoder: MingAudioDecoder,
    *,
    keep_latents: bool = False,
) -> StagePayload:
    """Decode generated acoustic latents into the terminal waveform payload."""

    state = load_ming_tts_state(payload)
    waveform = decoder.decode_nonstreaming(state.generated_latents)
    state.sample_rate = int(decoder.sample_rate)
    state.duration_s = float(waveform.numel() / int(decoder.sample_rate))
    if not keep_latents:
        state.generated_latents = None

    payload = store_ming_tts_state(payload, state)
    payload.data.update(
        audio_waveform_payload(
            waveform,
            sample_rate=int(decoder.sample_rate),
            modality="audio",
            source_hint="Ming-Omni-TTS",
        )
    )
    usage = build_usage(state)
    if usage is not None:
        payload.data["usage"] = usage
    return payload


__all__ = [
    "MingAudioDecoder",
    "MingAudioDecoderState",
    "decode_ming_tts_audio_payload",
]
