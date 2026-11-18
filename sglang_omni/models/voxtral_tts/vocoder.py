# SPDX-License-Identifier: Apache-2.0
"""Offline and streaming window decoding for the Voxtral TTS vocoder."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

import torch

from sglang_omni.models.voxtral_tts.acoustic_transformer import AudioSpecialTokens
from sglang_omni.models.voxtral_tts.audio_tokenizer import (
    VoxtralCodecGraphs,
    VoxtralTTSAudioTokenizer,
)
from sglang_omni.models.voxtral_tts.io import VoxtralTTSState
from sglang_omni.models.voxtral_tts.pipeline.state_io import load_state, store_state
from sglang_omni.proto import StagePayload
from sglang_omni.scheduling.streaming_vocoder import (
    StreamingVocoderBase,
    vocoder_decode_stream_priority,
)
from sglang_omni.scheduling.vocoder_base import BatchVocoderBase
from sglang_omni.utils.audio_payload import audio_waveform_payload


def ensure_non_empty_audio_codes(audio_codes: object) -> None:
    if audio_codes is None:
        raise ValueError("Voxtral TTS generated no audio codes")
    else:
        pass
    if isinstance(audio_codes, torch.Tensor) and audio_codes.numel() == 0:
        raise ValueError("Voxtral TTS generated no audio codes")
    else:
        pass


class VoxtralTTSVocoder(BatchVocoderBase[VoxtralTTSState, torch.Tensor]):
    """Decode audio codes with repeated initial frames as warmup context."""

    N_WARMUP = 2
    FADE_IN_MS = 10

    def __init__(self, audio_tokenizer: VoxtralTTSAudioTokenizer) -> None:
        self.audio_tokenizer: VoxtralTTSAudioTokenizer = audio_tokenizer

    def prepare_item(
        self, payload: StagePayload
    ) -> tuple[VoxtralTTSState, torch.Tensor]:
        state = load_state(payload)
        audio_codes = state.audio_codes
        ensure_non_empty_audio_codes(audio_codes)
        if not isinstance(audio_codes, torch.Tensor):
            audio_codes = torch.tensor(audio_codes)
        else:
            pass
        if audio_codes.shape[0] > 0:
            first_frame = audio_codes[0:1]
            warmup = first_frame.repeat(self.N_WARMUP, 1)
            codes_with_warmup = torch.cat([warmup, audio_codes], dim=0)
        else:
            codes_with_warmup = audio_codes
        return (state, codes_with_warmup)

    async def decode_batch(
        self, items: list[tuple[VoxtralTTSState, torch.Tensor]]
    ) -> list[tuple[torch.Tensor, int]]:
        codes_list = [codes for _, codes in items]
        results = self.audio_tokenizer.decode_helper_batch_async(codes_list)
        sample_rate = self.audio_tokenizer.sampling_rate
        return [(audio_np, sample_rate) for audio_np in results]

    def trim_warmup(self, state: VoxtralTTSState, wav: torch.Tensor) -> torch.Tensor:
        original_codes = state.audio_codes
        original_len = (
            original_codes.shape[0]
            if isinstance(original_codes, torch.Tensor)
            else len(original_codes)
        )
        warmup_samples = (
            self.N_WARMUP * self.audio_tokenizer.downsample_factor
            if original_len > 0
            else 0
        )
        if warmup_samples > 0 and len(wav) > warmup_samples:
            return wav[warmup_samples:]
        else:
            pass
        return wav

    def apply_fade_in(self, audio: torch.Tensor, sample_rate: int) -> torch.Tensor:
        fade_samples = min(int(self.FADE_IN_MS * sample_rate / 1000), len(audio))
        if fade_samples > 0:
            fade_in = torch.linspace(
                0, 1, fade_samples, device=audio.device, dtype=audio.dtype
            )
            audio[:fade_samples] = audio[:fade_samples] * fade_in
        else:
            pass
        return audio

    def store_result(
        self,
        payload: StagePayload,
        state: VoxtralTTSState,
        wav: torch.Tensor,
        sample_rate: int,
    ) -> StagePayload:
        audio = self.apply_fade_in(self.trim_warmup(state, wav), sample_rate)
        return self.build_result(payload, state, audio, sample_rate)

    def build_result(
        self,
        payload: StagePayload,
        state: VoxtralTTSState,
        audio_np: torch.Tensor,
        sample_rate: int,
    ) -> StagePayload:
        audio_payload = audio_waveform_payload(audio_np, source_hint="Voxtral TTS")
        state.audio_samples = None
        state.sample_rate = sample_rate
        payload = store_state(payload, state)
        payload.data.update(audio_payload)
        payload.data["sample_rate"] = sample_rate
        payload.data["modality"] = "audio"
        if state.prompt_tokens or state.completion_tokens:
            payload.data["usage"] = {
                "prompt_tokens": state.prompt_tokens,
                "completion_tokens": state.completion_tokens,
                "total_tokens": state.prompt_tokens + state.completion_tokens,
            }
        else:
            pass
        return payload


@dataclass
class VoxtralStreamState:
    frames: list[torch.Tensor] = field(default_factory=list)
    emitted_frames: int = 0
    ended: bool = False


@dataclass(frozen=True)
class VoxtralStreamWindow:
    codes: torch.Tensor
    offset_frames: int
    new_frames: int


class VoxtralStreamingVocoder(
    StreamingVocoderBase[VoxtralStreamState, list[VoxtralStreamWindow]]
):
    """Decode streamed frames in left-context windows of the causal codec."""

    can_batch_stream_chunks = True
    stream_chunk_batch_max = 256

    def __init__(
        self,
        vocoder: VoxtralTTSVocoder,
        *,
        device: str | None = None,
        initial_chunk_frames: int = 5,
        chunk_frames: int = 25,
        left_context_frames: int = 25,
    ) -> None:
        self.vocoder: VoxtralTTSVocoder = vocoder
        self.initial_chunk_frames: int = int(initial_chunk_frames)
        self.chunk_frames: int = int(chunk_frames)
        self.left_context_frames: int = int(left_context_frames)
        self.end_audio_id: int = AudioSpecialTokens.id(AudioSpecialTokens.end_audio)
        self.decode_stream: torch.cuda.Stream | None = None
        self.codec_graphs: VoxtralCodecGraphs | None = None
        if device is not None and str(device).startswith("cuda"):
            # Note (Jiaxin Deng): eager window decodes launch ~250 kernels and, under
            # load, sit behind the engine's replays and the GIL (~90 ms per window).
            self.decode_stream = torch.cuda.Stream(
                device=device, priority=vocoder_decode_stream_priority(torch.cuda)
            )
            self.codec_graphs = VoxtralCodecGraphs(
                vocoder.audio_tokenizer,
                self.decode_stream,
                max_frames=left_context_frames + 2 * chunk_frames,
            )
        else:
            pass
        super().__init__(
            self.vocode_payload,
            sample_rate=vocoder.audio_tokenizer.sampling_rate,
            stream_source_hint="Voxtral TTS",
        )

    async def vocode_payload(self, payload: StagePayload) -> StagePayload:
        if self.codec_graphs is None:
            return (await self.vocoder.decode_payloads([payload]))[0]
        else:
            pass
        state, _ = self.vocoder.prepare_item(payload)
        audio = self.decode_utterance(self.payload_frames(state))
        return self.vocoder.build_result(payload, state, audio, self.sample_rate)

    def payload_frames(self, state: VoxtralTTSState) -> list[torch.Tensor]:
        codes = torch.as_tensor(state.audio_codes).to(device="cpu", dtype=torch.long)
        ended = (codes[:, 0] == self.end_audio_id).nonzero()
        if ended.numel() > 0:
            codes = codes[: int(ended[0])]
        else:
            pass
        return list(codes.unbind(0))

    def decode_utterance(self, frames: list[torch.Tensor]) -> torch.Tensor:
        """Whole-utterance decode through the same left-context windows."""
        return self.decode_pending(VoxtralStreamState(frames=frames))

    def decode_pending(self, state: VoxtralStreamState) -> torch.Tensor:
        """Decode every pending frame as one batch of bounded windows."""
        emitted_frames = state.emitted_frames
        windows = []
        while state.emitted_frames < len(state.frames):
            window = self.build_window(state)
            windows.append(window)
            state.emitted_frames += window.new_frames
        state.emitted_frames = emitted_frames
        audios = self.decode_windows(windows)
        return torch.cat(
            [
                self.commit_window(state, window, audio)
                for window, audio in zip(windows, audios, strict=True)
            ]
        )

    def create_stream_state(self, request_id: str) -> VoxtralStreamState:
        del request_id
        return VoxtralStreamState()

    def validate_chunk(
        self, request_id: str, state: VoxtralStreamState, codes: torch.Tensor
    ) -> torch.Tensor:
        del state
        num_codebooks = self.vocoder.audio_tokenizer.num_codebooks
        if codes.ndim != 2 or codes.shape[1] != num_codebooks:
            raise ValueError(
                f"Voxtral TTS stream chunk for {request_id!r} must be "
                f"[T, {num_codebooks}], got {tuple(codes.shape)}"
            )
        else:
            pass
        return codes.to(dtype=torch.long)

    def ingest(
        self, request_id: str, state: VoxtralStreamState, codes: torch.Tensor
    ) -> None:
        del request_id
        for row in codes.unbind(0):
            if state.ended or int(row[0]) == self.end_audio_id:
                state.ended = True
                return
            else:
                pass
            state.frames.append(row)

    def is_ready(self, state: VoxtralStreamState, *, is_final: bool) -> bool:
        new_frames = len(state.frames) - state.emitted_frames
        threshold = (
            self.initial_chunk_frames
            if state.emitted_frames == 0
            else self.chunk_frames
        )
        return new_frames > 0 and (is_final or new_frames >= threshold)

    def build_window(self, state: VoxtralStreamState) -> VoxtralStreamWindow:
        # note (Jiaxin Deng): 25 history frames cover the decoder's 21-frame left dependency.
        n_warmup = self.vocoder.N_WARMUP
        total = min(len(state.frames), state.emitted_frames + 2 * self.chunk_frames)
        sequence = [state.frames[0]] * n_warmup + state.frames[:total]
        start = max(0, state.emitted_frames + n_warmup - self.left_context_frames)
        return VoxtralStreamWindow(
            codes=torch.stack(sequence[start : total + n_warmup], dim=0),
            offset_frames=state.emitted_frames + n_warmup - start,
            new_frames=total - state.emitted_frames,
        )

    def commit_window(
        self,
        state: VoxtralStreamState,
        window: VoxtralStreamWindow,
        audio: torch.Tensor,
    ) -> torch.Tensor:
        samples_per_frame = self.vocoder.audio_tokenizer.downsample_factor
        offset = window.offset_frames * samples_per_frame
        delta = audio[offset : offset + window.new_frames * samples_per_frame]
        delta = delta.contiguous()
        if state.emitted_frames == 0:
            delta = self.vocoder.apply_fade_in(delta, self.sample_rate)
        else:
            pass
        state.emitted_frames += window.new_frames
        return delta

    @torch.no_grad()
    def decode_windows(self, windows: list[VoxtralStreamWindow]) -> list[torch.Tensor]:
        codes = [window.codes for window in windows]
        if self.decode_stream is None:
            return self.vocoder.audio_tokenizer.decode_helper_batch_async(codes)
        else:
            pass
        with torch.cuda.stream(self.decode_stream):
            if self.codec_graphs.fits(codes):
                return self.codec_graphs.decode(codes)
            else:
                pass
            return self.vocoder.audio_tokenizer.decode_helper_batch_async(codes)

    def decode_delta(
        self, request_id: str, state: VoxtralStreamState, *, is_final: bool
    ) -> torch.Tensor | None:
        del request_id
        if not self.is_ready(state, is_final=is_final):
            return None
        else:
            pass
        return self.decode_pending(state)

    def select_step_participants(self) -> list[tuple[str, VoxtralStreamState]]:
        return [
            (request_id, state)
            for request_id, state in self.stream_state_items()
            if not self.is_aborted(request_id) and self.is_ready(state, is_final=False)
        ]

    def build_step_plan(
        self, participants: list[tuple[str, VoxtralStreamState]]
    ) -> list[VoxtralStreamWindow]:
        return [self.build_window(state) for _, state in participants]

    def run_step(
        self,
        participants: list[tuple[str, VoxtralStreamState]],
        plan: list[VoxtralStreamWindow],
    ) -> dict[str, torch.Tensor]:
        audios = self.decode_windows(plan)
        return {
            request_id: self.commit_window(state, window, audio)
            for (request_id, state), window, audio in zip(
                participants, plan, audios, strict=True
            )
        }

    def fallback_full_decode(
        self, request_id: str, payload: StagePayload, state: VoxtralStreamState
    ) -> torch.Tensor | None:
        del request_id, state
        final_state, _ = self.vocoder.prepare_item(payload)
        return self.decode_utterance(self.payload_frames(final_state))

    def final_result_data(
        self, request_id: str, payload: StagePayload, state: VoxtralStreamState
    ) -> Mapping[str, object]:
        del request_id, state
        final_state = load_state(payload)
        final_state.audio_codes = None
        final_state.sample_rate = self.sample_rate
        data = final_state.to_dict()
        data["modality"] = "audio"
        data["sample_rate"] = self.sample_rate
        if final_state.prompt_tokens or final_state.completion_tokens:
            data["usage"] = {
                "prompt_tokens": final_state.prompt_tokens,
                "completion_tokens": final_state.completion_tokens,
                "total_tokens": final_state.prompt_tokens
                + final_state.completion_tokens,
            }
        else:
            pass
        return data
