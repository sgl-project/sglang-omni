# SPDX-License-Identifier: Apache-2.0
"""One D2 conversation: per interval, causal audio, k Thinker text steps and k Talker frames."""

from dataclasses import dataclass

import torch
from pydantic import BaseModel, ConfigDict

from sglang_omni.models.qwen3_omni_d2.audio_encoder import CausalAudioStream
from sglang_omni.models.qwen3_omni_d2.graphs import (
    CapturedDecoder,
    causal_cache_mask,
    reset_cache,
    static_cache,
)
from sglang_omni.models.qwen3_omni_d2.mandarin import MandarinOpening
from sglang_omni.models.qwen3_omni_d2.release import D2Model
from sglang_omni.models.qwen3_omni_d2.waveform import Code2WavRenderer, PlaybackStream

PCM16_SCALE = 32768.0
CODEBOOKS = 16
CODEBOOK_SIZE = 2048
# The Talker codec frame that the reference runtime treats as silence.
CODEC_SILENCE_FRAME = (
    1049, 1700, 1626, 546, 306, 1443, 1871, 2008, 1866, 374, 662, 1383, 1123, 1430, 644, 610,
)  # fmt: skip
CODEC_TEMPERATURE = 0.7
CODEC_TOP_K = 50
CODEC_REPETITION_PENALTY = 1.1
RESIDUAL_TOP_P = 0.8


class D2Frame(BaseModel):
    """Playback and Thinker output for one native frame."""

    model_config = ConfigDict(val_json_bytes="base64")

    pcm16: bytes
    text_delta: str
    sequence: int
    is_interruption: bool


@dataclass(frozen=True, kw_only=True)
class TurnState:
    """Where the Thinker and Talker are in a reply after one text token."""

    is_responding: bool
    is_text_done: bool
    is_talker_responding: bool


def advance_turn(
    state: TurnState, token_id: int, model: D2Model
) -> tuple[TurnState, bool, bool]:
    """Return the next state, whether the Talker reads the text, and whether it emits audio."""
    is_response = token_id == model.response_token_id
    is_interrupt = token_id == model.interrupt_token_id
    assert not (is_response and state.is_responding)
    is_responding = (state.is_responding or is_response) and not is_interrupt
    if is_response:
        is_text_done = False
    elif state.is_responding and token_id == model.pad_token_id:
        is_text_done = True
    else:
        is_text_done = state.is_text_done
    next_state = TurnState(
        is_responding=is_responding,
        is_text_done=is_text_done or not is_responding,
        is_talker_responding=(state.is_talker_responding or is_response)
        and not is_interrupt,
    )
    return (
        next_state,
        state.is_talker_responding or is_response,
        state.is_talker_responding and not is_interrupt,
    )


def allowed_tokens(
    state: TurnState, previous_token_id: int, model: D2Model
) -> torch.Tensor:
    """Words only inside an unfinished reply; RESPONSE opens, INTERRUPT always available."""
    allowed = torch.full(
        (model.lm_head.out_features,),
        state.is_responding and not state.is_text_done,
        dtype=torch.bool,
        device=model.device,
    )
    allowed[model.pad_token_id] = not (
        state.is_responding and previous_token_id == model.response_token_id
    )
    allowed[model.response_token_id] = not state.is_responding
    allowed[model.interrupt_token_id] = True
    return allowed


def sample(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    """Temperature, top-k and nucleus sampling in the logits' own dtype."""
    scaled = logits / CODEC_TEMPERATURE
    cutoff = torch.topk(scaled, CODEC_TOP_K, dim=-1).values[..., -1:]
    scaled = scaled.masked_fill(scaled < cutoff, float("-inf"))
    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(scaled, descending=True, dim=-1)
        sorted_probabilities = torch.softmax(sorted_logits, dim=-1)
        removed = sorted_probabilities.cumsum(dim=-1) - sorted_probabilities > top_p
        scaled = torch.full_like(scaled, float("-inf")).scatter(
            -1, sorted_indices, sorted_logits.masked_fill(removed, float("-inf"))
        )
    else:
        pass
    return torch.multinomial(torch.softmax(scaled, dim=-1), num_samples=1)[..., 0]


class D2Session:
    """Captured decoders and the recurrent state of the conversation they serve."""

    def __init__(
        self,
        model: D2Model,
        capacity_frames: int,
        prompt_token_ids: list[int],
        seed: int,
        opening: MandarinOpening | None,
    ) -> None:
        self.model = model
        self.interval_frames = model.interval_frames
        self.seed = seed
        self.opening = opening
        self.prompt_token_ids = torch.tensor([prompt_token_ids], device=model.device)
        self.streams = [
            CausalAudioStream(
                model.audio_tower,
                model.interval_frames,
                model.mel_filters,
                model.fft_window_samples,
                model.hop_samples,
            )
            for _ in range(2)
        ]
        talker = model.talker
        with torch.autocast("cuda", dtype=torch.bfloat16):
            self.speaker_prefix = talker.text_projection(
                model.thinker.embed_tokens(
                    torch.tensor([[model.tts_pad_token_id]], device=model.device)
                )
            ) + talker.model.codec_embedding(
                torch.tensor([[model.speaker_id]], device=model.device)
            )
        accepted_layer = model.thinker.layers[model.accept_hidden_layer - 1]
        audio_tokens = 2 * self.interval_frames
        thinker_cache = static_cache(model.thinker, 3 * capacity_frames)
        if self.interval_frames == 1:
            # note (cuzimi): At 80 ms the previous token is known, so one causal forward covers env, self and text.
            self.fused_thinker = CapturedDecoder(
                model.thinker, thinker_cache, (4, 1, 3), 0, accepted_layer
            )
        else:
            self.audio_thinker = CapturedDecoder(
                model.thinker, thinker_cache, (4, 1, audio_tokens), 0
            )
            self.text_thinker = CapturedDecoder(
                model.thinker, thinker_cache, (4, 1, 1), audio_tokens, accepted_layer
            )
        self.thinker_cache = thinker_cache
        self.talker = CapturedDecoder(
            talker.model, static_cache(talker.model, capacity_frames + 1), (1, 1), 1
        )
        predictor_cache = static_cache(talker.code_predictor.model, CODEBOOKS)
        self.predictor_start = CapturedDecoder(
            talker.code_predictor.model, predictor_cache, (1, 2), 0
        )
        self.predictor_step = CapturedDecoder(
            talker.code_predictor.model, predictor_cache, (1, 1), 2
        )
        self.renderer = Code2WavRenderer(model.code2wav, capacity_frames)
        self.silence_codes = torch.tensor(CODEC_SILENCE_FRAME)
        self.reset()

    def reset(self) -> None:
        """Start a conversation: same seed, system prompt and speaker, empty history."""
        model = self.model
        torch.manual_seed(self.seed)
        for stream in self.streams:
            stream.reset()
        reset_cache(self.thinker_cache)
        reset_cache(self.talker.cache)
        self.renderer.reset()
        prompt_tokens = self.prompt_token_ids.shape[1]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            model.thinker(
                inputs_embeds=model.thinker.embed_tokens(self.prompt_token_ids),
                attention_mask=causal_cache_mask(
                    0,
                    prompt_tokens,
                    self.thinker_cache.layers[0].max_cache_len,
                    model.device,
                ),
                position_ids=torch.arange(prompt_tokens, device=model.device).expand(
                    4, 1, -1
                ),
                past_key_values=self.thinker_cache,
                use_cache=True,
            )
            model.talker.model(
                inputs_embeds=self.speaker_prefix,
                attention_mask=causal_cache_mask(
                    0, 1, self.talker.cache.layers[0].max_cache_len, model.device
                ),
                position_ids=torch.zeros((1, 1), dtype=torch.long, device=model.device),
                past_key_values=self.talker.cache,
                use_cache=True,
            )
        self.frames = 0
        self.previous_token_id = model.pad_token_id
        self.turn = TurnState(
            is_responding=False, is_text_done=True, is_talker_responding=False
        )
        self.generated_token_ids: list[int] = []
        self.codec0_history: list[int] = []
        self.talker_codes: torch.Tensor | None = None
        self.feedback: torch.Tensor | None = None
        self.playback = PlaybackStream()

    def step(self, pcm16: bytes) -> list[D2Frame]:
        """Advance one interaction interval of 16 kHz PCM16 and return its native frames."""
        model = self.model
        interval_frames = self.interval_frames
        environment_stream, self_stream = self.streams
        waveform = torch.frombuffer(bytearray(pcm16), dtype=torch.int16).float()
        environment = environment_stream.push(waveform / PCM16_SCALE)
        if self.feedback is None:
            heard_self = model.self_silence_row.expand(interval_frames, -1)
        else:
            # note (cuzimi): The model hears its own playback one interval late, as in training.
            heard_self = self_stream.push(self.feedback)
        audio_inputs = torch.cat(
            (
                environment + model.environment_stream_row,
                heard_self + model.self_stream_row,
            )
        )[None]
        prompt_tokens = self.prompt_token_ids.shape[1]
        cursor = prompt_tokens + 3 * self.frames
        slots = torch.arange(2 * interval_frames, device=model.device)
        frame_positions = prompt_tokens + self.frames + slots % interval_frames
        audio_positions = torch.stack(
            (cursor + slots, frame_positions, frame_positions, frame_positions)
        )[:, None]
        if interval_frames == 1:
            pass
        else:
            self.audio_thinker.replay(audio_inputs, audio_positions, cursor)
        outputs: list[tuple[int, torch.Tensor, bool]] = []
        for index in range(interval_frames):
            text_cursor = cursor + 2 * interval_frames + index
            text_embedding = model.thinker.embed_tokens(
                torch.tensor([[self.previous_token_id]], device=model.device)
            )
            text_positions = torch.full(
                (4, 1, 1), text_cursor, dtype=torch.long, device=model.device
            )
            if interval_frames == 1:
                last_hidden = self.fused_thinker.replay(
                    torch.cat((audio_inputs, text_embedding), dim=1),
                    torch.cat((audio_positions, text_positions), dim=2),
                    cursor,
                )
                accepted_hidden = self.fused_thinker.hooked_states
            else:
                last_hidden = self.text_thinker.replay(
                    text_embedding, text_positions, text_cursor
                )
                accepted_hidden = self.text_thinker.hooked_states
            token_id = self.next_token(last_hidden[:, -1])
            codes, is_active = self.talker_step(
                token_id, accepted_hidden[:, -1:], self.frames + index
            )
            outputs.append((token_id, codes, is_active))
        frames: list[D2Frame] = []
        feedback: list[torch.Tensor] = []
        for index, (token_id, codes, is_active) in enumerate(outputs):
            pcm16_frame, frame_feedback = self.playback.append(
                self.renderer.append(codes), is_active
            )
            feedback.append(frame_feedback)
            frames.append(
                D2Frame(
                    pcm16=pcm16_frame,
                    text_delta=(
                        ""
                        if token_id in model.special_token_ids
                        else model.tokenizer.decode(
                            [token_id], skip_special_tokens=True
                        )
                    ),
                    sequence=self.frames + index,
                    is_interruption=token_id == model.interrupt_token_id,
                )
            )
        self.feedback = torch.cat(feedback)
        self.frames += interval_frames
        return frames

    def next_token(self, hidden_states: torch.Tensor) -> int:
        """Greedy text token under the turn rules, after the optional opening-word policy."""
        logits = self.model.text_logits(hidden_states).masked_fill(
            ~allowed_tokens(self.turn, self.previous_token_id, self.model),
            float("-inf"),
        )
        if self.opening is not None:
            logits = self.opening.mask(logits, self.generated_token_ids)
        else:
            pass
        token_id = int(logits.argmax(dim=-1).item())
        self.previous_token_id = token_id
        return token_id

    def talker_step(
        self, token_id: int, accepted_hidden: torch.Tensor, sequence: int
    ) -> tuple[torch.Tensor, bool]:
        """Run one Talker frame; return its physical codes and whether they are audible."""
        model = self.model
        talker = model.talker
        before = self.turn
        self.turn, reads_text, emits_audio = advance_turn(before, token_id, model)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            if reads_text:
                context = talker.text_projection(
                    model.thinker.embed_tokens(
                        torch.tensor([[token_id]], device=model.device)
                    )
                )
            else:
                context = talker.hidden_projection(accepted_hidden)
        if not reads_text:
            talker_input = context
        elif token_id == model.response_token_id or self.talker_codes is None:
            talker_input = self.embed_codes(self.opening_codes(token_id)) + context
        else:
            talker_input = self.embed_codes(self.talker_codes) + context
        hidden_states = self.talker.replay(
            talker_input,
            torch.full((1, 1), sequence + 1, device=model.device),
            sequence + 1,
        )[:, -1]
        if token_id == model.response_token_id:
            codes = self.opening_codes(token_id)
        elif token_id == model.interrupt_token_id and before.is_talker_responding:
            codes = self.silence_codes.clone()
            codes[0] = model.codec_eos_id
        elif emits_audio:
            codes = self.generate_codes(hidden_states)
            if int(codes[0]) == model.codec_eos_id:
                emits_audio = False
                self.turn = TurnState(
                    is_responding=self.turn.is_responding,
                    is_text_done=self.turn.is_text_done,
                    is_talker_responding=False,
                )
            else:
                pass
        else:
            codes = self.silence_codes
        self.generated_token_ids.append(token_id)
        if token_id == model.response_token_id or not self.turn.is_talker_responding:
            self.codec0_history.clear()
        elif emits_audio:
            self.codec0_history.append(int(codes[0]))
        else:
            pass
        self.talker_codes = codes if self.turn.is_talker_responding else None
        return (codes if emits_audio else self.silence_codes), emits_audio

    def opening_codes(self, token_id: int) -> torch.Tensor:
        """Silence, with codec BOS when a reply opens on this frame."""
        codes = self.silence_codes.clone()
        if token_id == self.model.response_token_id:
            codes[0] = self.model.codec_bos_id
        else:
            pass
        return codes

    def embed_codes(self, codes: torch.Tensor) -> torch.Tensor:
        """Sum the 16 codebook embeddings of one frame for the next Talker input."""
        talker = self.model.talker
        codes = codes.to(self.model.device)
        embedded = talker.model.codec_embedding(codes[None, None, 0])
        for group, embedding in enumerate(talker.code_predictor.model.codec_embedding):
            embedded = embedded + embedding(codes[None, None, group + 1])
        return embedded

    def generate_codes(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Sample codebook 0 from the Talker head, then 15 residual books from the CodePredictor."""
        model = self.model
        predictor = model.talker.code_predictor
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model.talker.codec_head(hidden_states)[
                ..., : model.codec_vocabulary_size
            ]
        is_special = (
            torch.arange(logits.shape[-1], device=model.device) >= CODEBOOK_SIZE
        )
        is_special[model.codec_eos_id] = False
        logits = logits.masked_fill(is_special, float("-inf"))
        if self.codec0_history:
            history = torch.tensor(
                sorted(set(self.codec0_history)), device=model.device
            )
            scores = logits.index_select(-1, history)
            logits = logits.index_copy(
                -1,
                history,
                torch.where(
                    scores < 0,
                    scores * CODEC_REPETITION_PENALTY,
                    scores / CODEC_REPETITION_PENALTY,
                ),
            )
        else:
            pass
        code = sample(logits, top_p=1.0)
        if int(code.item()) == model.codec_eos_id:
            codes = self.silence_codes.clone()
            codes[0] = model.codec_eos_id
            return codes
        else:
            reset_cache(self.predictor_start.cache)
        codes = [code]
        predictor_hidden = self.predictor_start.replay(
            torch.stack(
                (hidden_states, model.talker.model.codec_embedding(code)), dim=1
            ),
            torch.arange(2, device=model.device)[None],
            0,
        )[:, -1]
        for group, head in enumerate(predictor.lm_head, start=1):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = head(predictor_hidden)[..., :CODEBOOK_SIZE]
            code = sample(logits, top_p=RESIDUAL_TOP_P)
            codes.append(code)
            if group < CODEBOOKS - 1:
                predictor_hidden = self.predictor_step.replay(
                    predictor.model.codec_embedding[group - 1](code)[:, None],
                    torch.full((1, 1), group + 1, device=model.device),
                    group + 1,
                )[:, -1]
            else:
                pass
        return torch.cat(codes).cpu()
