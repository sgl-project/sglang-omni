# SPDX-License-Identifier: Apache-2.0
"""Render D2 codec frames to 80 ms of playback and the 16 kHz self feedback the model hears next."""

import math

import torch
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
    Qwen3OmniMoeCode2Wav,
)

from sglang_omni.models.qwen3_omni_d2.graphs import (
    CapturedDecoder,
    reset_cache,
    static_cache,
)

# The causal convolutions finish a frame from at most 25 earlier transformer frames.
LEFT_CONTEXT_FRAMES = 25
OUTPUT_SAMPLE_RATE_HZ = 24_000
FEEDBACK_SAMPLE_RATE_HZ = 16_000
OUTPUT_FRAME_SAMPLES = 1_920
FEEDBACK_FRAME_SAMPLES = 1_280
# Code2Wav's first frame is this many 24 kHz samples short of the 80 ms clock.
FIRST_FRAME_DELAY_SAMPLES = 555
DEZIPPER_TRANSITION_SAMPLES = 12
DEZIPPER_MINIMUM_JUMP = 0.03
PCM16_FULL_SCALE = 32767.0


class CapturedConvolution:
    """The Code2Wav upsampling and decoder convolutions for one fixed window of transformer frames."""

    def __init__(self, code2wav: Qwen3OmniMoeCode2Wav, window_frames: int) -> None:
        self.code2wav = code2wav
        weight = code2wav.code_embedding.weight
        self.transformed = torch.zeros(
            (1, window_frames, code2wav.config.hidden_size),
            device=weight.device,
            dtype=weight.dtype,
        )
        self.graph = torch.cuda.CUDAGraph()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            self.forward()
        torch.cuda.synchronize(weight.device)
        with (
            torch.autocast("cuda", dtype=torch.bfloat16),
            torch.cuda.graph(self.graph, capture_error_mode="thread_local"),
        ):
            self.waveform = self.forward()

    def forward(self) -> torch.Tensor:
        hidden_states = self.transformed.permute(0, 2, 1)
        for blocks in self.code2wav.upsample:
            for block in blocks:
                hidden_states = block(hidden_states)
        for block in self.code2wav.decoder:
            hidden_states = block(hidden_states)
        return hidden_states.clamp(min=-1, max=1)

    def replay(self, transformed: torch.Tensor) -> torch.Tensor:
        self.transformed.copy_(transformed)
        self.graph.replay()
        return self.waveform


class Code2WavRenderer:
    """Decode one 16-codebook frame at a time from a KV-cached transformer and a bounded window."""

    def __init__(self, code2wav: Qwen3OmniMoeCode2Wav, capacity_frames: int) -> None:
        self.code2wav = code2wav
        self.transformer = CapturedDecoder(
            code2wav.pre_transformer,
            static_cache(code2wav.pre_transformer, capacity_frames),
            (1, 1),
            capture_cursor=0,
        )
        self.convolutions = {
            frames: CapturedConvolution(code2wav, frames)
            for frames in range(1, LEFT_CONTEXT_FRAMES + 2)
        }
        self.reset()

    def reset(self) -> None:
        reset_cache(self.transformer.cache)
        self.frames = 0
        self.transformed: list[torch.Tensor] = []

    def append(self, codes: torch.Tensor) -> torch.Tensor:
        """Return the new float waveform samples at 24 kHz for one [16] frame of codec IDs."""
        device = self.code2wav.code_embedding.weight.device
        codes = codes.view(1, -1, 1).to(device)
        hidden_states = self.code2wav.code_embedding(
            codes + self.code2wav.code_offset
        ).mean(1)
        position_ids = torch.full((1, 1), self.frames, device=device)
        self.transformed.append(
            self.transformer.replay(hidden_states, position_ids, self.frames).clone()
        )
        self.transformed = self.transformed[-(LEFT_CONTEXT_FRAMES + 1) :]
        self.frames += 1
        window = torch.cat(self.transformed, dim=1)
        waveform = (
            self.convolutions[window.shape[1]].replay(window).float().cpu().view(-1)
        )
        if self.frames == 1:
            return waveform
        else:
            return waveform[-OUTPUT_FRAME_SAMPLES:]


class PlaybackStream:
    """Put Code2Wav samples on the 80 ms clock, silence inactive frames and derive the self feedback."""

    def __init__(self) -> None:
        numerators = torch.arange(FEEDBACK_FRAME_SAMPLES) * OUTPUT_SAMPLE_RATE_HZ
        self.feedback_left = numerators // FEEDBACK_SAMPLE_RATE_HZ
        self.feedback_weight = (numerators % FEEDBACK_SAMPLE_RATE_HZ).float() / (
            FEEDBACK_SAMPLE_RATE_HZ
        )
        phase = torch.arange(DEZIPPER_TRANSITION_SAMPLES) * (
            math.pi / (DEZIPPER_TRANSITION_SAMPLES - 1)
        )
        self.dezipper_weights = (phase.cos() + 1.0) * 0.5
        self.previous_source_sample: torch.Tensor | None = None
        self.previous_audible_sample: float | None = None

    def append(
        self, samples: torch.Tensor, is_active: bool
    ) -> tuple[bytes, torch.Tensor]:
        """Return the frame's 24 kHz PCM16 bytes and its 16 kHz float feedback."""
        if self.previous_source_sample is None:
            playback = torch.cat((torch.zeros(FIRST_FRAME_DELAY_SAMPLES), samples))
            previous = playback[:1]
        else:
            playback = samples
            previous = self.previous_source_sample
        assert playback.numel() == OUTPUT_FRAME_SAMPLES
        # note (cuzimi): Each 16 kHz sample interpolates the playback one 24 kHz sample earlier.
        source = torch.cat((previous, playback))
        feedback = (
            source[self.feedback_left] * (1.0 - self.feedback_weight)
            + source[self.feedback_left + 1] * self.feedback_weight
        )
        self.previous_source_sample = playback[-1:].clone()
        activity = float(is_active)
        audible = playback * activity
        if self.previous_audible_sample is not None:
            correction = self.previous_audible_sample - float(audible[0])
            if abs(correction) > DEZIPPER_MINIMUM_JUMP:
                audible[:DEZIPPER_TRANSITION_SAMPLES].add_(
                    self.dezipper_weights * correction
                ).clamp_(-1.0, 1.0)
            else:
                pass
        else:
            pass
        self.previous_audible_sample = float(audible[-1])
        pcm16 = (audible.clamp(-1.0, 1.0) * PCM16_FULL_SCALE).round().to(torch.int16)
        return pcm16.numpy().tobytes(), feedback * activity
