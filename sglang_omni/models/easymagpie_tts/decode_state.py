# SPDX-License-Identifier: Apache-2.0
"""Per-request decode state the EasyMagpie talker keeps on the GPU.

Each decode step conditions on the previous step's phoneme row and acoustic
frame. With async decode the scheduler launches step N+1 before the host has
read step N, so that feedback, together with each request's text and sampling
controls, lives in device tables indexed by SGLang's request-pool slot. The
host seeds a slot once, after the request's prefill; every decode step then
reads its inputs from the tables and writes its predictions back, all inside
the captured decode graph.

Slot 0 is SGLang's padding slot: CUDA-graph padding rows point at it and no
request is ever assigned it, so its row only carries padding traffic.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch

from sglang_omni.models.easymagpie_tts.hf_config import EasyMagpieTTSConfig
from sglang_omni.models.easymagpie_tts.payload_types import EasyMagpieTTSState
from sglang_omni.models.easymagpie_tts.request_builders import (
    CONTINUE_TOKEN_ID,
    STOP_TOKEN_ID,
)

# Step output columns after the stacked acoustic codes.
EMIT_COLUMN = -2
STOP_COLUMN = -1


@dataclass
class DecodeStepInputs:
    """One decode step's per-row inputs, gathered from the slot tables."""

    slots: torch.Tensor
    steps: torch.Tensor
    text_tokens: torch.Tensor
    text_valid: torch.Tensor
    phoneme_tokens: torch.Tensor
    phoneme_valid: torch.Tensor
    audio_codes: torch.Tensor
    audio_valid: torch.Tensor
    phoneme_ended: torch.Tensor
    temperatures: torch.Tensor
    top_ks: torch.Tensor
    seeds: torch.Tensor
    positions: torch.Tensor


@dataclass
class EasyMagpieDecodeState:
    config: EasyMagpieTTSConfig
    offsets: torch.Tensor
    text: torch.Tensor
    text_lens: torch.Tensor
    phoneme_delays: torch.Tensor
    speech_delays: torch.Tensor
    phoneme_ended: torch.Tensor
    last_phonemes: torch.Tensor
    last_audio: torch.Tensor
    has_last_audio: torch.Tensor
    temperatures: torch.Tensor
    top_ks: torch.Tensor
    seeds: torch.Tensor
    # Rows are batch rows: the stacked codes, whether the frame is audio
    # output, and the continue/stop token SGLang sees.
    step_output: torch.Tensor
    # Top-k width of the sampling kernels; rows may ask for any k up to it.
    max_top_k: int

    @classmethod
    def allocate(
        cls,
        config: EasyMagpieTTSConfig,
        *,
        num_slots: int,
        text_capacity: int,
        max_batch: int,
        max_top_k: int,
        device: torch.device,
    ) -> EasyMagpieDecodeState:
        if min(num_slots, text_capacity, max_batch, max_top_k) < 1:
            raise ValueError(
                "decode state needs positive slot, text, batch and top-k sizes"
            )
        else:
            pass
        rows = int(num_slots) + 1

        def zeros(*shape: int, dtype: torch.dtype = torch.long) -> torch.Tensor:
            return torch.zeros(shape, device=device, dtype=dtype)

        return cls(
            config=config,
            offsets=zeros(rows),
            text=zeros(rows, text_capacity),
            text_lens=zeros(rows),
            phoneme_delays=zeros(rows),
            speech_delays=zeros(rows),
            phoneme_ended=zeros(rows, dtype=torch.bool),
            last_phonemes=zeros(rows, config.phoneme_stacking_factor),
            last_audio=zeros(rows, config.num_stacked_codebooks),
            has_last_audio=zeros(rows, dtype=torch.bool),
            temperatures=torch.ones(rows, device=device, dtype=torch.float32),
            top_ks=torch.ones(rows, device=device, dtype=torch.long),
            seeds=zeros(rows),
            step_output=zeros(max_batch, config.num_stacked_codebooks + 2),
            max_top_k=min(int(max_top_k), config.codebook_vocab_size),
        )

    @property
    def num_slots(self) -> int:
        return int(self.offsets.shape[0]) - 1

    @property
    def max_batch(self) -> int:
        return int(self.step_output.shape[0])

    def seed(
        self,
        *,
        slots: Sequence[int],
        states: Sequence[EasyMagpieTTSState],
        seeds: Sequence[int],
        prefill_phonemes: torch.Tensor,
    ) -> None:
        """Start each request's decode from its prefill.

        ``prefill_phonemes`` are the phonemes predicted from each prompt's last
        row; decoding resumes at the first text step prefill did not fold in.
        """
        if not slots:
            return
        else:
            pass
        device = self.offsets.device
        index = torch.tensor(list(slots), dtype=torch.long).to(device)
        scalars = torch.tensor(
            [
                [
                    state.text_prefill_num,
                    len(state.text_token_ids),
                    state.phoneme_delay,
                    state.speech_delay,
                    state.top_k,
                    seed,
                ]
                for state, seed in zip(states, seeds)
            ],
            dtype=torch.long,
        ).to(device)
        for column, table in enumerate(
            (
                self.offsets,
                self.text_lens,
                self.phoneme_delays,
                self.speech_delays,
                self.top_ks,
                self.seeds,
            )
        ):
            table.index_copy_(0, index, scalars[:, column])
        temperatures = torch.tensor([state.temperature for state in states])
        self.temperatures.index_copy_(0, index, temperatures.to(device))
        longest = max(len(state.text_token_ids) for state in states)
        text = torch.zeros((len(states), longest), dtype=torch.long)
        for row, state in enumerate(states):
            text[row, : len(state.text_token_ids)] = torch.tensor(
                state.text_token_ids, dtype=torch.long
            )
        self.text[index, :longest] = text.to(device)
        self.last_phonemes.index_copy_(0, index, prefill_phonemes.to(torch.long))
        self.phoneme_ended.index_fill_(0, index, False)
        self.has_last_audio.index_fill_(0, index, False)

    def read(self, req_pool_indices: torch.Tensor) -> DecodeStepInputs:
        """Gather a step's inputs for the batch rows in ``req_pool_indices``.

        The text token sits at the request offset. The phoneme channel opens
        with BOS at the phoneme delay, feeds back each prediction, and closes
        one step after it feeds a phoneme EOS. The audio channel opens with BOS
        at the speech delay, then feeds back the previous frame.
        """
        config = self.config
        slots = req_pool_indices.to(torch.long)
        steps = self.offsets[slots]

        text_valid = steps < self.text_lens[slots]
        column = steps.clamp(max=int(self.text.shape[1]) - 1)
        text_tokens = torch.where(
            text_valid, self.text[slots, column], torch.zeros_like(steps)
        )

        phoneme_delay = self.phoneme_delays[slots]
        phoneme_valid = ~self.phoneme_ended[slots] & (steps >= phoneme_delay)
        last_phonemes = self.last_phonemes[slots]
        phoneme_tokens = torch.where(
            (steps == phoneme_delay).unsqueeze(1),
            torch.full_like(last_phonemes, config.phoneme_bos_id),
            last_phonemes,
        )
        predicted_eos = (last_phonemes == config.phoneme_eos_id).any(dim=1)

        speech_delay = self.speech_delays[slots]
        audio_valid = steps >= speech_delay
        last_audio = self.last_audio[slots]
        opens_audio = (steps == speech_delay) | ~self.has_last_audio[slots]
        audio_codes = torch.where(
            opens_audio.unsqueeze(1),
            torch.full_like(last_audio, config.audio_bos_id),
            last_audio,
        )

        return DecodeStepInputs(
            slots=slots,
            steps=steps,
            text_tokens=text_tokens,
            text_valid=text_valid,
            phoneme_tokens=phoneme_tokens,
            phoneme_valid=phoneme_valid,
            audio_codes=audio_codes,
            audio_valid=audio_valid,
            phoneme_ended=self.phoneme_ended[slots] | (phoneme_valid & predicted_eos),
            temperatures=self.temperatures[slots],
            top_ks=self.top_ks[slots],
            seeds=self.seeds[slots],
            # Prefill consumes sampling position zero; positions stay
            # request-local regardless of the batch a request lands in.
            positions=(steps + 1) * config.num_stacked_codebooks,
        )

    def commit(
        self,
        inputs: DecodeStepInputs,
        *,
        codes: torch.Tensor,
        phonemes: torch.Tensor,
        eos: torch.Tensor,
    ) -> None:
        """Feed a step's predictions to the next step and publish its output."""
        slots = inputs.slots
        self.offsets.index_copy_(0, slots, inputs.steps + 1)
        self.last_phonemes.index_copy_(0, slots, phonemes)
        self.phoneme_ended.index_copy_(0, slots, inputs.phoneme_ended)
        self.last_audio.index_copy_(0, slots, codes)
        self.has_last_audio.index_fill_(0, slots, True)
        batch = int(codes.shape[0])
        output = self.step_output[:batch]
        output[:, : codes.shape[1]].copy_(codes)
        output[:, EMIT_COLUMN].copy_(inputs.audio_valid & ~eos)
        output[:, STOP_COLUMN].copy_(
            torch.where(
                eos,
                torch.full_like(inputs.steps, STOP_TOKEN_ID),
                torch.full_like(inputs.steps, CONTINUE_TOKEN_ID),
            )
        )


__all__ = [
    "EMIT_COLUMN",
    "STOP_COLUMN",
    "DecodeStepInputs",
    "EasyMagpieDecodeState",
]
