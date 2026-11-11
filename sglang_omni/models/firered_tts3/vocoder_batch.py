# SPDX-License-Identifier: Apache-2.0
"""Full-payload RedAE vocoder and shared audio result metadata."""

from __future__ import annotations

import contextlib
import os
import threading
from _thread import LockType
from typing import Literal, TypedDict

import torch

from sglang_omni.models.firered_tts3.payload_types import (
    FireRedTTS3State,
    load_firered_tts3_state,
)
from sglang_omni.models.firered_tts3.profiling import ProfilingProbe
from sglang_omni.models.firered_tts3.redae_decoder import RedAEAudioDecoder
from sglang_omni.proto.request import StagePayload
from sglang_omni.scheduling.pipeline_state import build_usage
from sglang_omni.scheduling.vocoder_base import BatchVocoderBase
from sglang_omni.utils.audio_payload import audio_waveform_payload

SAMPLES_PER_LATENT_FRAME = 960

DEBUG_DUMP_DIR_ENV = "FIREREDTTS3_DEBUG_DUMP_DIR"


class VocoderResultMetadata(TypedDict, total=False):
    modality: Literal["audio"]
    sample_rate: int
    finish_reason: str | None
    text: str
    usage: dict[str, int | float]


def vocoder_result_metadata(
    state: FireRedTTS3State, sample_rate: int
) -> VocoderResultMetadata:
    metadata: VocoderResultMetadata = {
        "modality": "audio",
        "sample_rate": sample_rate,
        "finish_reason": state.finish_reason,
    }
    if state.gen_text:
        metadata["text"] = state.gen_text
    else:
        pass
    usage = build_usage(state)
    if usage is not None:
        metadata["usage"] = usage
    else:
        pass
    return metadata


class FireRedTTS3BatchVocoder(BatchVocoderBase[FireRedTTS3State, torch.Tensor]):
    """Full-payload decoding through the shared vocoder scheduler."""

    def __init__(
        self,
        decoder: RedAEAudioDecoder,
        *,
        redae_scale: float,
        sample_rate: int,
        decode_stream: torch.cuda.Stream | None = None,
    ) -> None:
        self.decoder: RedAEAudioDecoder = decoder
        self.redae_scale: float = float(redae_scale)
        self.sample_rate: int = int(sample_rate)
        self.decode_stream: torch.cuda.Stream | None = decode_stream
        self.lock: LockType = threading.Lock()
        self.debug_dump_count: int = 0

    def decode_stream_context(
        self,
    ) -> contextlib.AbstractContextManager[None]:
        if self.decode_stream is None:
            return contextlib.nullcontext()
        else:
            pass
        return torch.cuda.stream(self.decode_stream)

    def maybe_dump_latents(self, latents: torch.Tensor) -> None:
        dump_dir = os.environ.get(DEBUG_DUMP_DIR_ENV)
        if dump_dir:
            os.makedirs(dump_dir, exist_ok=True)
            path = os.path.join(dump_dir, f"latents_{self.debug_dump_count}.pt")
            torch.save(latents.detach().cpu(), path)
            self.debug_dump_count += 1
        else:
            pass

    def prepare_item(
        self, payload: StagePayload
    ) -> tuple[FireRedTTS3State, torch.Tensor]:
        state = load_firered_tts3_state(payload)
        if state.generated_latents is None or state.generated_latents.numel() == 0:
            raise RuntimeError("FireRedTTS3 vocoder received no generated latents")
        else:
            pass
        latent_parts: list[torch.Tensor] = []
        if state.prompt_latents is not None and state.prompt_latents.numel():
            latent_parts.append(state.prompt_latents)
        else:
            pass
        latent_parts.append(state.generated_latents)
        return state, torch.cat(latent_parts, dim=1)

    def decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode a full latent sequence and read back on its decode stream."""
        if latents.ndim != 3 or latents.shape[0] != 1:
            raise ValueError(
                "FireRedTTS3 latents must have shape [1, frames, redae_dim], "
                f"got {tuple(latents.shape)}"
            )
        else:
            pass
        if latents.shape[2] != self.decoder.in_proj.in_features:
            raise ValueError(
                f"FireRedTTS3 latent width must be {self.decoder.in_proj.in_features}, "
                f"got {latents.shape[2]}"
            )
        else:
            pass
        self.maybe_dump_latents(latents)
        with self.lock, torch.no_grad(), self.decode_stream_context():
            probe = ProfilingProbe.create()
            with probe.segment("redae_decode"):
                waveform, _ = self.decoder.forward_stream(
                    (latents / self.redae_scale).to(
                        device=next(self.decoder.parameters()).device,
                        dtype=torch.float32,
                    ),
                    None,
                    last_chunk=True,
                )
            probe.log("batch-vocoder")
            return waveform.cpu()

    async def decode_batch(
        self, items: list[tuple[FireRedTTS3State, torch.Tensor]]
    ) -> list[tuple[torch.Tensor, int]]:
        outputs: list[tuple[torch.Tensor, int]] = []
        for state, latents in items:
            outputs.append((self.decode_item(state, latents), self.sample_rate))
        return outputs

    def decode_item(
        self, state: FireRedTTS3State, latents: torch.Tensor
    ) -> torch.Tensor:
        waveform = self.decode_latents(latents)
        prompt_samples = int(state.left_context_size) * SAMPLES_PER_LATENT_FRAME
        if prompt_samples > 0:
            return waveform[:, prompt_samples:]
        else:
            return waveform

    def store_result(
        self,
        payload: StagePayload,
        state: FireRedTTS3State,
        wav: torch.Tensor,
        sample_rate: int,
    ) -> StagePayload:
        payload.data = audio_waveform_payload(
            wav,
            sample_rate=sample_rate,
            modality="audio",
            source_hint="fireredtts3",
        )
        payload.data.update(vocoder_result_metadata(state, sample_rate))
        return payload

    async def decode_payload(self, payload: StagePayload) -> StagePayload:
        state, latents = self.prepare_item(payload)
        [(wav, sample_rate)] = await self.decode_batch([(state, latents)])
        return self.store_result(payload, state, wav, sample_rate)
