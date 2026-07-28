# SPDX-License-Identifier: Apache-2.0
"""Streaming entry point for the ZONOS2 DAC vocoder.

``decode_to_pcm`` decodes the full delayed code sequence in one shot (the
non-streaming terminal path). ``Zonos2StreamingVocoderScheduler`` adds true
streaming: it consumes the per-frame delayed code rows emitted by the AR engine
and decodes them incrementally with raised-cosine overlap-add (OLA), withholding
the delay/flush tail until ``stream_done``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

import torch

from sglang_omni.models.zonos2.components.audio_codec import (
    DAC_HOP_LENGTH,
    Zonos2DACVocoder,
)
from sglang_omni.models.zonos2.payload_types import (
    N_CODEBOOKS,
    ZONOS2_SAMPLE_RATE,
    Zonos2State,
)
from sglang_omni.proto import StagePayload
from sglang_omni.scheduling.pipeline_state import build_usage
from sglang_omni.scheduling.streaming_vocoder import (
    StreamingVocoderBase,
    resolve_initial_codec_chunk_frames,
)
from sglang_omni.utils.audio_payload import audio_waveform_payload

# Process-wide cache: load the DAC checkpoint at most once per device.
_vocoder_cache: tuple[str, Zonos2DACVocoder] | None = None

# New streaming-chunk constants (not changes to existing values). At 86.1328125
# fps (44100/512) one frame = DAC_HOP_LENGTH=512 PCM samples.
_STREAM_STEADY_CHUNK_FRAMES = (
    40  # ~0.46 s steady chunk (fewer vocoder forwards/chunk -> lower streaming RTF)
)
_STREAM_INITIAL_CHUNK_FRAMES = 5  # ~0.06 s first chunk for low TTFB
_STREAM_OLA_OVERLAP_FRAMES = 2  # note (Yue Yin): cross-fade width; TODO calibrate
# shear_up needs the trailing N_CODEBOOKS-1 future rows to de-shear a frame.
_STREAM_WITHHOLD_TAIL = N_CODEBOOKS - 1
# note (Yue Yin): the AR engine appends n+1 post-EOS countdown rows that the
# one-shot path trims via eos_frame. eos_frame is unknown mid-stream, so steady
# pulls hold this region back; the eos_frame-capped flush at stream_done emits
# the true tail, keeping streamed length == one-shot length.
_STREAM_EOS_GUARD_FRAMES = N_CODEBOOKS + 1


def _get_vocoder(device: str) -> Zonos2DACVocoder:
    global _vocoder_cache
    if _vocoder_cache is None or _vocoder_cache[0] != device:
        _vocoder_cache = (device, Zonos2DACVocoder(device=device))
    return _vocoder_cache[1]


def decode_to_pcm(
    audio_codes: torch.Tensor,
    eos_frame: int | None = None,
    device: str = "cuda",
) -> torch.Tensor:
    """Decode delayed ``[T, 9]`` AR codes to 1-D float32 PCM @ 44.1 kHz.

    Args:
        audio_codes: delayed per-frame codes ``[T, 9]`` (or batched ``[B, T, 9]``)
            straight from AR decode, before the delay is sheared out.
        eos_frame: number of aligned frames to keep before EOS, if known.
        device: torch device the DAC model runs on.

    Returns:
        1-D ``float32`` PCM tensor at 44.1 kHz on CPU.
    """
    return _get_vocoder(device).decode(audio_codes, eos_frame=eos_frame)


def decode_batch(
    audio_codes_list: list[torch.Tensor],
    eos_frames: list[int | None],
    device: str = "cuda",
) -> list[torch.Tensor]:
    """Batched analogue of ``decode_to_pcm``: one DAC forward for many items.

    Reuses the process-wide DAC cache, so no second checkpoint load.
    """
    return _get_vocoder(device).decode_batch(audio_codes_list, eos_frames)


# ---- streaming (incremental raised-cosine OLA) ----


class _Zonos2OLADecoder:
    """Incremental de-shear + DAC decode of delayed rows with OLA cross-fade.

    Reuses ``Zonos2DACVocoder.decode`` per window: given delayed rows
    ``[lo, hi + WITHHOLD)`` it returns the aligned PCM for frames ``[lo, hi)``.
    Consecutive windows overlap by ``overlap`` frames; the overlap PCM is held
    back and raised-cosine cross-faded into the next window to mask the DAC
    ConvTranspose edge transients at chunk boundaries.
    """

    def __init__(self, device: str, overlap_frames: int, hop: int) -> None:
        self.device = device
        self.overlap = int(overlap_frames)
        self.hop = int(hop)
        self.rows: list[torch.Tensor] = []
        self.emitted = 0  # next aligned frame to emit
        self.decoded_to = 0  # right edge of decoded frames (== emitted + overlap)
        self.tail: torch.Tensor | None = None  # held PCM for the overlap region
        # note (Yue Yin): the cross-fade window is fixed for the decoder's life
        # (overlap*hop), so build the raised-cosine ramps once instead of per chunk.
        hold = self.overlap * self.hop
        self._ramp_up, self._ramp_down = self._ramps(hold) if hold > 0 else (None, None)

    def add(self, rows: list[torch.Tensor]) -> None:
        self.rows.extend(rows)

    @staticmethod
    def _ramps(n: int) -> tuple[torch.Tensor, torch.Tensor]:
        i = torch.arange(n, dtype=torch.float32)
        up = 0.5 * (1.0 - torch.cos(math.pi * i / max(n - 1, 1)))
        return up, 1.0 - up

    def pull(
        self,
        vocoder: Zonos2DACVocoder,
        *,
        chunk_frames: int,
        flush: bool,
        eos_frame: int | None = None,
    ) -> list[torch.Tensor]:
        chunks: list[torch.Tensor] = []
        ovl = self.overlap
        hop = self.hop
        hold = ovl * hop
        trail = (
            _STREAM_WITHHOLD_TAIL
            if flush
            else _STREAM_WITHHOLD_TAIL + _STREAM_EOS_GUARD_FRAMES
        )
        max_frame = len(self.rows) - trail
        if eos_frame is not None:
            max_frame = min(max_frame, int(eos_frame))
        if max_frame <= 0:
            return chunks
        while True:
            hi = max_frame if flush else self.decoded_to + chunk_frames
            if not flush and hi > max_frame:
                break
            lo = self.emitted
            if hi <= lo:
                break
            block = torch.stack(self.rows[lo : hi + _STREAM_WITHHOLD_TAIL], dim=0)
            pcm = vocoder.decode(block)  # aligned PCM for frames [lo, hi)
            if pcm.numel() == 0:
                break
            if self.tail is not None and hold > 0 and pcm.numel() >= hold:
                up, down = self._ramp_up, self._ramp_down
                pcm = pcm.clone()
                pcm[:hold] = self.tail * down + pcm[:hold] * up
            if flush:
                self.tail = None
                self.emitted = hi
                self.decoded_to = hi
                if pcm.numel() > 0:
                    chunks.append(pcm.contiguous())
                break
            if pcm.numel() > hold:
                chunks.append(pcm[: pcm.numel() - hold].contiguous())
            # note (Yue Yin): the decoder owns pcm's lifetime; the tail is only
            # read (cross-faded) next chunk, never mutated, so a view is safe.
            self.tail = pcm[pcm.numel() - hold :] if hold > 0 else None
            self.emitted = hi - ovl
            self.decoded_to = hi
        return chunks


@dataclass
class _Zonos2StreamState:
    decoder: _Zonos2OLADecoder | None = None
    n_codebooks: int = N_CODEBOOKS
    initial_chunk_frames: int = 0
    latched: bool = False


class Zonos2StreamingVocoderScheduler(StreamingVocoderBase[_Zonos2StreamState, None]):
    """Decode ZONOS2 delayed code rows incrementally with raised-cosine OLA.

    The base owns the streaming lifecycle; this scheduler owns the OLA cursor,
    the withhold/EOS-guard tail, and the stream-done top-up. Non-streaming
    requests use the one-shot ``compute_fn`` / ``batch_compute_fn`` (the terminal
    DAC decode) exactly like the base's non-streaming path.
    """

    def __init__(
        self,
        *,
        device: str = "cuda",
        compute_fn: Any = None,
        batch_compute_fn: Any = None,
        steady_chunk_frames: int = _STREAM_STEADY_CHUNK_FRAMES,
        initial_chunk_frames: int = _STREAM_INITIAL_CHUNK_FRAMES,
        overlap_frames: int = _STREAM_OLA_OVERLAP_FRAMES,
        max_batch_size: int = 1,
        max_batch_wait_ms: int = 0,
        request_cost_fn: Any = None,
        max_batch_cost: int | None = None,
    ) -> None:
        if steady_chunk_frames <= 0:
            raise ValueError(
                f"steady_chunk_frames must be positive, got {steady_chunk_frames}"
            )
        self._device = device
        self._steady_chunk_frames = int(steady_chunk_frames)
        self._default_initial_chunk_frames = max(
            0, min(int(initial_chunk_frames), int(steady_chunk_frames))
        )
        self._overlap_frames = int(overlap_frames)
        super().__init__(
            compute_fn,
            sample_rate=ZONOS2_SAMPLE_RATE,
            stream_source_hint="ZONOS2 streaming",
            batch_compute_fn=batch_compute_fn,
            max_batch_size=max_batch_size,
            max_batch_wait_ms=max_batch_wait_ms,
            request_cost_fn=request_cost_fn,
            max_batch_cost=max_batch_cost,
        )

    # ---- streaming hooks ----

    def create_stream_state(self, request_id: str) -> _Zonos2StreamState:
        del request_id
        return _Zonos2StreamState()

    def latch_stream_contract(
        self,
        request_id: str,
        state: _Zonos2StreamState,
        source: StagePayload | Mapping[str, Any],
        *,
        origin: str,
    ) -> None:
        del request_id
        if state.latched:
            return
        if origin == "payload":
            params = source.request.params
        else:
            params = source
        if not isinstance(params, dict):
            return
        n_vq = params.get("n_codebooks")
        if n_vq is not None:
            state.n_codebooks = int(n_vq)
        state.initial_chunk_frames = (
            resolve_initial_codec_chunk_frames(
                params, steady_chunk_frames=self._steady_chunk_frames
            )
            or self._default_initial_chunk_frames
        )
        state.latched = True

    def validate_chunk(
        self, request_id: str, state: _Zonos2StreamState, codes: torch.Tensor
    ) -> torch.Tensor:
        del request_id
        # Accept either a single [9] row or a coalesced [k, 9] batch (the AR engine
        # may group several frames per message); keep the leading codebook columns.
        rows_t = codes.to(dtype=torch.long)
        if rows_t.ndim == 1:
            rows_t = rows_t.reshape(1, -1)
        return rows_t[:, : state.n_codebooks]

    def ingest(
        self, request_id: str, state: _Zonos2StreamState, codes: torch.Tensor
    ) -> None:
        del request_id
        if state.decoder is None:
            state.decoder = _Zonos2OLADecoder(
                self._device, self._overlap_frames, DAC_HOP_LENGTH
            )
        state.decoder.add([codes[i] for i in range(codes.shape[0])])

    def decode_delta(
        self, request_id: str, state: _Zonos2StreamState, *, is_final: bool
    ) -> torch.Tensor | None:
        if is_final:
            return self._flush(request_id, state)
        if state.decoder is None:
            return None
        chunk_frames = (
            state.initial_chunk_frames
            if (
                not self._stream_has_emitted(request_id)
                and state.initial_chunk_frames > 0
            )
            else self._steady_chunk_frames
        )
        # note (Yue Yin): a request-supplied chunk size <= overlap would drive the
        # OLA cursor negative; keep at least one non-overlap frame per window.
        chunk_frames = max(chunk_frames, self._overlap_frames + 1)
        pcms = [
            pcm
            for pcm in state.decoder.pull(
                _get_vocoder(self._device), chunk_frames=chunk_frames, flush=False
            )
            if pcm.numel() > 0
        ]
        if not pcms:
            return None
        return torch.cat(pcms) if len(pcms) > 1 else pcms[0]

    def _flush(self, request_id: str, state: _Zonos2StreamState) -> torch.Tensor | None:
        zstate = Zonos2State.from_dict(self._stream_payloads[request_id].data)
        # note (Yue Yin): coalescing (stream_emit_chunk_frames>1) or retraction can
        # leave the streamed OLA decoder SHORT of the full aligned length (a held or
        # dropped tail), which truncates the audio (observed: emit=32 drops a stochastic
        # ~1% of tails -> WER spike). The complete code sequence is in zstate.audio_codes,
        # so top the decoder up to full length before the flush: a no-op when every row
        # was streamed (audio-neutral), and it recovers the missing tail otherwise. The
        # flush's eos_frame cap then trims to the aligned length == the non-stream path.
        if zstate.audio_codes is not None:
            full = torch.as_tensor(zstate.audio_codes, dtype=torch.long)
            if full.shape[0] > 0:
                if state.decoder is None:
                    state.decoder = _Zonos2OLADecoder(
                        self._device, self._overlap_frames, DAC_HOP_LENGTH
                    )
                have = len(state.decoder.rows)
                if full.shape[0] > have:
                    state.decoder.add([full[i] for i in range(have, full.shape[0])])
        if state.decoder is None or not state.decoder.rows:
            return None
        pcms = [
            pcm
            for pcm in state.decoder.pull(
                _get_vocoder(self._device),
                chunk_frames=self._steady_chunk_frames,
                flush=True,
                eos_frame=zstate.eos_frame,
            )
            if pcm.numel() > 0
        ]
        if not pcms:
            return None
        return torch.cat(pcms) if len(pcms) > 1 else pcms[0]

    def fallback_full_decode(
        self, request_id: str, payload: StagePayload, state: _Zonos2StreamState
    ) -> torch.Tensor | None:
        # Nothing streamed (slot-starved / sub-window utterance): fall back to the
        # one-shot decode so streaming output matches the non-stream path.
        del request_id, state
        zstate = Zonos2State.from_dict(payload.data)
        if zstate.audio_codes is None:
            return None
        codes = torch.as_tensor(zstate.audio_codes, dtype=torch.long)
        if codes.numel() == 0:
            return None
        pcm = decode_to_pcm(codes, zstate.eos_frame, device=self._device)
        return pcm if pcm.numel() > 0 else None

    def stream_payload(self, request_id: str, waveform: torch.Tensor) -> dict[str, Any]:
        del request_id
        return audio_waveform_payload(
            waveform.detach().to("cpu", torch.float32),
            sample_rate=self._sample_rate,
            modality="audio",
            source_hint="ZONOS2 streaming",
        )

    def final_result_data(
        self, request_id: str, payload: StagePayload, state: _Zonos2StreamState
    ) -> dict[str, Any]:
        del request_id, state
        zstate = Zonos2State.from_dict(payload.data)
        final_data: dict[str, Any] = {
            "modality": "audio",
            "sample_rate": int(zstate.sample_rate),
        }
        usage = build_usage(zstate)
        if usage is not None:
            final_data["usage"] = usage
        return final_data


__all__ = ["decode_to_pcm", "decode_batch", "Zonos2StreamingVocoderScheduler"]
