# SPDX-License-Identifier: Apache-2.0
"""MLX streaming vocoder stage for Qwen3-TTS.

Plugs the MLX speech-tokenizer decoder into Omni's
:class:`StreamingVocoderBase`, so the stage lifecycle, chunk ingestion and
payload emission are the shared ones and only the codec call is model specific.

Two differences from the CUDA vocoder. It has no CUDA graphs or pinned staging,
because neither exists on Metal. And it decodes *statefully*: the MLX decoder
keeps convolution tails, overlap-add remainders and transformer KV per request,
so each step decodes only the newly arrived frames instead of re-running a
left-context window. Those buffers live on the shared decoder module, so every
request owns a session that is swapped in around its own steps.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Mapping

import mlx.core as mx
import numpy as np
import torch

from sglang_omni.models.qwen3_tts.payload_types import Qwen3TTSState
from sglang_omni.proto import StagePayload
from sglang_omni.scheduling.pipeline_state import build_usage
from sglang_omni.scheduling.streaming_vocoder import StreamingVocoderBase
from sglang_omni.utils.audio_payload import audio_waveform_payload

logger = logging.getLogger(__name__)

DEFAULT_MLX_STREAM_STRIDE = 8
CODEBOOK_SIZE = 2048


@dataclass
class Qwen3TTSMlxStreamState:
    """Per-request vocoder state.

    ``session`` holds this request's decoder streaming buffers; ``ref_frames``
    is how many leading frames are reference audio whose waveform must not be
    emitted.
    """

    session: dict = field(default_factory=dict)
    pending: list[np.ndarray] = field(default_factory=list)
    num_quantizers: int | None = None
    ref_frames: int = 0
    pending_ref_frames: int = 0
    ref_frames_consumed: int = 0
    emitted_frames: int = 0
    total_frames: int = 0


class Qwen3TTSMlxStreamingVocoder(StreamingVocoderBase[Qwen3TTSMlxStreamState, None]):
    """Decodes Qwen3-TTS codec frames to 24 kHz PCM with MLX."""

    def __init__(
        self,
        speech_tokenizer: Any,
        *,
        sample_rate: int = 24000,
        stream_stride: int = DEFAULT_MLX_STREAM_STRIDE,
        initial_chunk_frames: int | None = None,
        max_batch_size: int = 1,
        max_batch_wait_ms: int = 0,
        abort_callback: Any = None,
    ) -> None:
        self._tokenizer = speech_tokenizer
        self._decoder = speech_tokenizer.decoder
        self._stream_stride = max(1, int(stream_stride))
        self._initial_chunk_frames = (
            self._stream_stride
            if initial_chunk_frames is None
            else max(1, int(initial_chunk_frames))
        )
        super().__init__(
            self._vocode_payload,
            sample_rate=sample_rate,
            stream_source_hint="qwen3-tts-mlx",
            batch_compute_fn=None,
            max_batch_size=max_batch_size,
            max_batch_wait_ms=max_batch_wait_ms,
            abort_callback=abort_callback,
        )

    # -- lifecycle ------------------------------------------------------

    def create_stream_state(self, request_id: str) -> Qwen3TTSMlxStreamState:
        del request_id
        return Qwen3TTSMlxStreamState(session=self._decoder.new_streaming_session())

    def latch_stream_contract(
        self,
        request_id: str,
        state: Qwen3TTSMlxStreamState,
        source: StagePayload | Mapping[str, Any],
        *,
        origin: str,
    ) -> None:
        """Latch how many leading frames are reference audio.

        Voice cloning streams the reference codes ahead of the generated ones so
        the decoder starts from the reference's acoustic state; their waveform is
        dropped. The count is immutable once latched.
        """
        metadata = source.data if origin == "payload" else source
        if not isinstance(metadata, Mapping) or "ref_code_len" not in metadata:
            return
        ref_frames = int(metadata["ref_code_len"] or 0)
        if ref_frames < 0:
            raise ValueError(
                f"Qwen3-TTS ref_code_len for {request_id!r} must not be negative"
            )
        if state.total_frames or state.ref_frames:
            if ref_frames != state.ref_frames:
                raise ValueError(
                    f"Qwen3-TTS ref_code_len for {request_id!r} changed after "
                    "frames were already ingested"
                )
            return
        state.pending_ref_frames = ref_frames

    def release_stream_resources(
        self, request_id: str, state: Qwen3TTSMlxStreamState
    ) -> None:
        del request_id
        # Dropping the session frees this request's conv tails and KV.
        state.session = {}
        state.pending = []

    def on_serving_stop(self) -> None:
        # Reclaim MLX buffers once, at teardown, never inside the decode loop.
        mx.clear_cache()

    # -- ingestion ------------------------------------------------------

    def validate_chunk(
        self,
        request_id: str,
        state: Qwen3TTSMlxStreamState,
        codes: torch.Tensor,
    ) -> torch.Tensor:
        if codes.ndim != 2:
            raise ValueError(
                f"Qwen3-TTS stream chunk for {request_id!r} must be "
                f"[frames, quantizers], got shape {tuple(codes.shape)}"
            )
        if state.num_quantizers is None:
            state.num_quantizers = int(codes.shape[1])
        elif int(codes.shape[1]) != state.num_quantizers:
            raise ValueError(
                f"Qwen3-TTS stream chunk has {int(codes.shape[1])} quantizers, "
                f"expected {state.num_quantizers}"
            )
        if bool((codes < 0).any()) or bool((codes >= CODEBOOK_SIZE).any()):
            raise ValueError(
                f"Qwen3-TTS stream chunk for {request_id!r} contains codec ids "
                f"outside [0, {CODEBOOK_SIZE})"
            )
        return codes

    def ingest(
        self,
        request_id: str,
        state: Qwen3TTSMlxStreamState,
        codes: torch.Tensor,
    ) -> None:
        del request_id
        if state.pending_ref_frames:
            state.ref_frames = state.pending_ref_frames
            state.pending_ref_frames = 0
        frames = np.asarray(codes.detach().cpu().numpy(), dtype=np.int32)
        state.pending.append(frames)
        state.total_frames += int(frames.shape[0])

    def should_decode(self, state: Qwen3TTSMlxStreamState, *, is_final: bool) -> bool:
        if is_final:
            return True
        buffered = sum(int(chunk.shape[0]) for chunk in state.pending)
        # Reference frames only prime the decoder, so they never count toward
        # the emit threshold.
        priming = max(0, state.ref_frames - state.ref_frames_consumed)
        threshold = (
            self._initial_chunk_frames
            if state.emitted_frames == 0
            else self._stream_stride
        )
        return buffered - priming >= threshold

    # -- decode ---------------------------------------------------------

    def decode_delta(
        self,
        request_id: str,
        state: Qwen3TTSMlxStreamState,
        *,
        is_final: bool,
    ) -> torch.Tensor | None:
        del request_id, is_final
        if not state.pending:
            return None

        frames = np.concatenate(state.pending, axis=0)
        state.pending = []

        # [frames, groups] -> [1, groups, frames]
        codes = mx.array(frames)[None].transpose(0, 2, 1)
        with self._decoder.streaming_session(state.session):
            waveform = self._decoder.streaming_step(codes)
        audio = waveform.squeeze(1)[0]
        mx.eval(audio)

        upsample = self._decoder.total_upsample
        emitted = int(frames.shape[0])

        # Drop the waveform belonging to reference frames still being consumed.
        priming = min(max(0, state.ref_frames - state.ref_frames_consumed), emitted)
        if priming:
            state.ref_frames_consumed += priming
            audio = audio[priming * upsample :]
        state.emitted_frames += emitted - priming

        if audio.shape[0] == 0:
            return None
        return torch.from_numpy(np.array(audio, dtype=np.float32))

    # -- terminal payloads ----------------------------------------------

    def final_result_data(
        self,
        request_id: str,
        payload: StagePayload,
        state: Qwen3TTSMlxStreamState,
    ) -> dict[str, Any]:
        del request_id, state
        final_state = Qwen3TTSState.from_dict(payload.data)
        data: dict[str, Any] = {
            "modality": "audio",
            "sample_rate": self._sample_rate,
        }
        usage = build_usage(final_state)
        if usage is not None:
            data["usage"] = usage
        return data

    def fallback_full_decode(
        self,
        request_id: str,
        payload: StagePayload,
        state: Qwen3TTSMlxStreamState,
    ) -> torch.Tensor | None:
        """Decode the whole utterance when streaming emitted nothing."""
        del request_id
        codes = self._payload_codes(payload)
        if codes is None:
            return None
        return self._decode_whole(codes, ref_frames=state.ref_frames)

    # -- non-streaming requests -----------------------------------------

    async def _vocode_payload(self, payload: StagePayload) -> StagePayload:
        """Whole-utterance path for a request that never streamed."""
        state = Qwen3TTSState.from_dict(payload.data)
        codes = self._payload_codes(payload)
        if codes is None:
            raise RuntimeError(
                "Qwen3-TTS MLX vocoder requires audio_codes from tts_engine"
            )
        waveform = self._decode_whole(codes, ref_frames=int(state.ref_code_len or 0))

        data = audio_waveform_payload(
            waveform,
            sample_rate=self._sample_rate,
            modality="audio",
            source_hint=self._stream_source_hint,
        )
        usage = build_usage(state)
        if usage is not None:
            data["usage"] = usage
        payload.data = data
        return payload

    @staticmethod
    def _payload_codes(payload: StagePayload) -> np.ndarray | None:
        state = Qwen3TTSState.from_dict(payload.data)
        codes = state.audio_codes
        if codes is None:
            return None
        if isinstance(codes, list):
            if not codes:
                return None
            stacked = [
                (
                    entry.detach().cpu().numpy()
                    if hasattr(entry, "detach")
                    else np.asarray(entry)
                )
                for entry in codes
            ]
            return np.stack(stacked, axis=0).astype(np.int32)
        if hasattr(codes, "detach"):
            return codes.detach().cpu().numpy().astype(np.int32)
        return np.asarray(codes, dtype=np.int32)

    def _decode_whole(self, frames: np.ndarray, *, ref_frames: int) -> torch.Tensor:
        """Decode ``[frames, groups]`` in one pass and trim the reference span."""
        codes = mx.array(frames)[None]
        waveform, _ = self._tokenizer.decode(codes)
        audio = waveform[0]
        mx.eval(audio)
        if ref_frames:
            cut = ref_frames * self._decoder.total_upsample
            if 0 < cut < audio.shape[0]:
                audio = audio[cut:]
        return torch.from_numpy(np.array(audio, dtype=np.float32))
