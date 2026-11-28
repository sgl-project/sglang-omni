# SPDX-License-Identifier: Apache-2.0
"""One Qwen3-Omni D2 conversation: per 80 ms frame, causal audio, then one Thinker and one Talker step."""

from dataclasses import replace

import torch
from pydantic import BaseModel, ConfigDict

from sglang_omni.models.qwen3_omni_d2.components.audio_encoder import (
    Qwen3OmniD2AudioEncoder,
)
from sglang_omni.models.qwen3_omni_d2.components.code2wav import (
    Qwen3OmniD2Code2Wav,
    Qwen3OmniD2Playback,
)
from sglang_omni.models.qwen3_omni_d2.components.mandarin import MandarinOpening
from sglang_omni.models.qwen3_omni_d2.components.model import Qwen3OmniD2Model
from sglang_omni.models.qwen3_omni_d2.components.talker import Qwen3OmniD2Talker
from sglang_omni.models.qwen3_omni_d2.components.thinker import (
    Qwen3OmniD2Thinker,
    TurnState,
    advance_turn,
)

PCM16_SCALE = 32768.0


class Qwen3OmniD2Frame(BaseModel):
    """Playback and Thinker output for one native frame."""

    model_config = ConfigDict(val_json_bytes="base64")

    pcm16: bytes
    text_delta: str
    sequence: int
    is_interruption: bool


class Qwen3OmniD2Session:
    """The captured components and the recurrent state of the conversation they serve."""

    def __init__(
        self,
        model: Qwen3OmniD2Model,
        capacity_frames: int,
        prompt_token_ids: list[int],
        seed: int,
        opening: MandarinOpening | None,
    ) -> None:
        self.model = model
        self.seed = seed
        self.audio_encoders = [
            Qwen3OmniD2AudioEncoder(
                model.audio_tower,
                model.mel_filters,
                model.fft_window_samples,
                model.hop_samples,
            )
            for _ in range(2)
        ]
        self.thinker = Qwen3OmniD2Thinker(
            model, capacity_frames, prompt_token_ids, opening
        )
        self.talker = Qwen3OmniD2Talker(model, capacity_frames)
        self.code2wav = Qwen3OmniD2Code2Wav(model.code2wav, capacity_frames)
        self.reset()

    def reset(self) -> None:
        """Start a conversation: same seed, system prompt and speaker, empty history."""
        torch.manual_seed(self.seed)
        for audio_encoder in self.audio_encoders:
            audio_encoder.reset()
        self.thinker.reset()
        self.talker.reset()
        self.code2wav.reset()
        self.frames = 0
        self.turn = TurnState(
            is_responding=False, is_text_done=True, is_talker_responding=False
        )
        self.feedback: torch.Tensor | None = None
        self.playback = Qwen3OmniD2Playback()

    def step(self, pcm16: bytes) -> Qwen3OmniD2Frame:
        """Advance one 80 ms frame of 16 kHz PCM16 and return its playback and text."""
        model = self.model
        environment_encoder, self_encoder = self.audio_encoders
        waveform = torch.frombuffer(bytearray(pcm16), dtype=torch.int16).float()
        environment = environment_encoder.push(waveform / PCM16_SCALE)
        if self.feedback is None:
            heard_self = model.self_silence_row.view(1, -1)
        else:
            # note (cuzimi): The model hears its own playback one frame late, as in training.
            heard_self = self_encoder.push(self.feedback)
        text_hidden, accepted_hidden = self.thinker.decode(
            torch.cat(
                (
                    environment + model.environment_stream_row,
                    heard_self + model.self_stream_row,
                )
            )[None],
            self.frames,
        )
        token_id = self.thinker.next_token(text_hidden, self.turn)
        transition = advance_turn(self.turn, token_id, model)
        talker_frame = self.talker.step(
            token_id, accepted_hidden, self.frames, transition
        )
        if talker_frame.has_closed:
            self.turn = replace(transition.after, is_talker_responding=False)
        else:
            self.turn = transition.after
        pcm16_frame, self.feedback = self.playback.append(
            self.code2wav.append(talker_frame.codes), talker_frame.is_active
        )
        frame = Qwen3OmniD2Frame(
            pcm16=pcm16_frame,
            text_delta=(
                ""
                if token_id in model.special_token_ids
                else model.tokenizer.decode([token_id], skip_special_tokens=True)
            ),
            sequence=self.frames,
            is_interruption=token_id == model.interrupt_token_id,
        )
        self.frames += 1
        return frame
