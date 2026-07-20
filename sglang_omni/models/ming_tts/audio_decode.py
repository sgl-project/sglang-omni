# SPDX-License-Identifier: Apache-2.0
"""Terminal audio decode helpers for Ming-Omni-TTS."""

from __future__ import annotations

import time
from contextlib import nullcontext
from typing import Any

import torch

from sglang_omni.models.ming_omni.talker.audio_vae.modeling_audio_vae import AudioVAE
from sglang_omni.models.ming_tts.audio_config import AudioVAEconfig
from sglang_omni.models.ming_tts.payload_types import (
    MingTTSState,
    decode_generated_latents,
    load_ming_tts_state,
    store_ming_tts_state,
)
from sglang_omni.proto import StagePayload
from sglang_omni.scheduling.pipeline_state import build_usage
from sglang_omni.scheduling.vocoder_base import BatchVocoderBase
from sglang_omni.utils.audio_payload import audio_waveform_payload


class MingAudioDecoder(torch.nn.Module):
    """Chunked official-path AudioVAE decoder wrapper."""

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
        if audio_config.semantic_module_kwargs is not None:
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
        try:
            return next(self.audio_vae.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    @property
    def dtype(self) -> torch.dtype:
        try:
            return next(self.audio_vae.parameters()).dtype
        except StopIteration:
            return torch.float32

    @torch.inference_mode()
    def decode_chunks(
        self,
        latents: torch.Tensor,
        last_chunks: list[bool],
        *,
        decode_mode: str = "chunked",
    ) -> torch.Tensor:
        if decode_mode != "chunked":
            raise NotImplementedError(
                "Ming-Omni-TTS currently supports only chunked AudioVAE decode"
            )

        if not isinstance(latents, torch.Tensor):
            latents = torch.as_tensor(latents)
        latents = latents.to(device=self.device, dtype=self.dtype)

        last_chunks = [bool(item) for item in last_chunks]
        if int(latents.shape[0]) == 0:
            return torch.empty((0,), device=self.device, dtype=torch.float32)

        stream_state = (None, None, None)
        past_key_values = None
        waveform_chunks = []
        autocast_dtype = self.dtype
        if autocast_dtype not in (torch.float16, torch.bfloat16):
            autocast_dtype = torch.bfloat16
        context = (
            torch.autocast(device_type="cuda", dtype=autocast_dtype)
            if self.device.type == "cuda"
            else nullcontext()
        )
        with context:
            for step, last_chunk in enumerate(last_chunks):
                chunk = latents[step : step + 1]
                wav, stream_state, past_key_values = self.audio_vae.decode(
                    chunk,
                    past_key_values=past_key_values,
                    use_cache=True,
                    stream_state=stream_state,
                    last_chunk=last_chunk,
                )
                wav = self._normalize_waveform_chunk(wav)
                waveform_chunks.append(wav)

        return torch.cat(waveform_chunks, dim=0)

    @staticmethod
    def _normalize_waveform_chunk(wav: Any) -> torch.Tensor:
        if not isinstance(wav, torch.Tensor):
            wav = torch.as_tensor(wav)
        while wav.ndim > 1:
            wav = wav[0]
        return wav.detach()


class MingTTSBatchVocoder(BatchVocoderBase):
    """Terminal AudioVAE decode stage on the shared batch-vocoder base.

    AudioVAE chunk decode carries streaming KV state across the chunks of a
    single request, so decode_batch stays a per-item loop; the base provides
    the shared scheduler/batching shell.
    """

    def __init__(
        self,
        decoder: MingAudioDecoder,
        *,
        decode_mode: str = "chunked",
        keep_latents: bool = False,
    ) -> None:
        self._decoder = decoder
        self._decode_mode = str(decode_mode)
        self._keep_latents = bool(keep_latents)

    def prepare_item(self, payload: StagePayload) -> tuple[MingTTSState, torch.Tensor]:
        state = load_ming_tts_state(payload)
        latents = decode_generated_latents(
            state,
            device=self._decoder.device,
            dtype=self._decoder.dtype,
        )
        return state, latents

    def _decode_item(
        self, state: MingTTSState, latents: torch.Tensor
    ) -> tuple[torch.Tensor, int]:
        started = time.perf_counter()
        waveform = self._decoder.decode_chunks(
            latents,
            state.generated_last_chunk,
            decode_mode=self._decode_mode,
        )
        state.audio_decode_time_s = time.perf_counter() - started
        waveform = MingAudioDecoder._normalize_waveform_chunk(waveform)
        return waveform, int(self._decoder.sample_rate)

    async def decode_batch(
        self, items: list[tuple[MingTTSState, torch.Tensor]]
    ) -> list[tuple[torch.Tensor, int]]:
        return [self._decode_item(state, latents) for state, latents in items]

    def store_result(
        self,
        payload: StagePayload,
        state: MingTTSState,
        wav: torch.Tensor,
        sample_rate: int,
    ) -> StagePayload:
        state.sample_rate = int(sample_rate)
        state.duration_s = float(wav.numel() / int(sample_rate))
        if not self._keep_latents:
            state.generated_latents_bytes = None
            state.generated_latents_shape = None
            state.generated_latents_dtype = None

        payload = store_ming_tts_state(payload, state)
        payload.data.update(
            audio_waveform_payload(
                wav,
                sample_rate=int(sample_rate),
                modality="audio",
                source_hint="Ming-Omni-TTS",
            )
        )
        usage = build_usage(state)
        if usage is not None:
            payload.data["usage"] = usage
        return payload


def decode_ming_tts_audio_payload(
    payload: StagePayload,
    decoder: MingAudioDecoder,
    *,
    decode_mode: str = "chunked",
    keep_latents: bool = False,
) -> StagePayload:
    """Decode generated acoustic latents into the terminal waveform payload."""

    vocoder = MingTTSBatchVocoder(
        decoder,
        decode_mode=decode_mode,
        keep_latents=keep_latents,
    )
    state, latents = vocoder.prepare_item(payload)
    wav, sample_rate = vocoder._decode_item(state, latents)
    return vocoder.store_result(payload, state, wav, sample_rate)


__all__ = [
    "MingAudioDecoder",
    "MingTTSBatchVocoder",
    "decode_ming_tts_audio_payload",
]
