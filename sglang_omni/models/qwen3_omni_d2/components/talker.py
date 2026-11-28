# SPDX-License-Identifier: Apache-2.0
"""Qwen3-Omni D2 Talker: one 16-codebook frame per text token, from the reply text or the Thinker."""

from dataclasses import dataclass

import torch

from sglang_omni.models.qwen3_omni_d2.components.decoder_graph import (
    DecoderGraph,
    causal_cache_mask,
    reset_cache,
    static_cache,
)
from sglang_omni.models.qwen3_omni_d2.components.model import Qwen3OmniD2Model
from sglang_omni.models.qwen3_omni_d2.components.thinker import TurnTransition

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


@dataclass(frozen=True, kw_only=True)
class TalkerFrame:
    """The codes to render for one frame and whether the Talker closed its reply on it."""

    codes: torch.Tensor
    is_active: bool
    has_closed: bool


class Qwen3OmniD2Talker:
    """One Talker token per native frame over a fixed cache, then the CodePredictor's residual books."""

    def __init__(self, model: Qwen3OmniD2Model, capacity_frames: int) -> None:
        self.model = model
        talker = model.talker
        with torch.autocast("cuda", dtype=torch.bfloat16):
            self.speaker_prefix = talker.text_projection(
                model.thinker.embed_tokens(
                    torch.tensor([[model.tts_pad_token_id]], device=model.device)
                )
            ) + talker.model.codec_embedding(
                torch.tensor([[model.speaker_id]], device=model.device)
            )
        self.graph = DecoderGraph(
            talker.model, static_cache(talker.model, capacity_frames + 1), (1, 1), 1
        )
        predictor_cache = static_cache(talker.code_predictor.model, CODEBOOKS)
        self.predictor_start = DecoderGraph(
            talker.code_predictor.model, predictor_cache, (1, 2), 0
        )
        self.predictor_step = DecoderGraph(
            talker.code_predictor.model, predictor_cache, (1, 1), 2
        )
        self.silence_codes = torch.tensor(CODEC_SILENCE_FRAME)

    def reset(self) -> None:
        """Empty the cache and prefill the speaker position."""
        model = self.model
        reset_cache(self.graph.cache)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            model.talker.model(
                inputs_embeds=self.speaker_prefix,
                attention_mask=causal_cache_mask(
                    0, 1, self.graph.cache.layers[0].max_cache_len, model.device
                ),
                position_ids=torch.zeros((1, 1), dtype=torch.long, device=model.device),
                past_key_values=self.graph.cache,
                use_cache=True,
            )
        self.codes: torch.Tensor | None = None
        self.codec0_history: list[int] = []

    def step(
        self,
        token_id: int,
        accepted_hidden: torch.Tensor,
        sequence: int,
        transition: TurnTransition,
    ) -> TalkerFrame:
        """Run the Talker at native frame sequence for the Thinker's token and accepted layer."""
        model = self.model
        talker = model.talker
        with torch.autocast("cuda", dtype=torch.bfloat16):
            if transition.reads_text:
                context = talker.text_projection(
                    model.thinker.embed_tokens(
                        torch.tensor([[token_id]], device=model.device)
                    )
                )
            else:
                context = talker.hidden_projection(accepted_hidden)
        if not transition.reads_text:
            talker_input = context
        elif token_id == model.response_token_id or self.codes is None:
            talker_input = self.embed_codes(self.opening_codes(token_id)) + context
        else:
            talker_input = self.embed_codes(self.codes) + context
        hidden_states = self.graph.replay(
            talker_input,
            torch.full((1, 1), sequence + 1, device=model.device),
            sequence + 1,
        )[:, -1]
        emits_audio = transition.emits_audio
        if token_id == model.response_token_id:
            codes = self.opening_codes(token_id)
        elif (
            token_id == model.interrupt_token_id
            and transition.before.is_talker_responding
        ):
            codes = self.silence_codes.clone()
            codes[0] = model.codec_eos_id
        elif emits_audio:
            codes = self.generate_codes(hidden_states)
            emits_audio = int(codes[0]) != model.codec_eos_id
        else:
            codes = self.silence_codes
        has_closed = transition.emits_audio and not emits_audio
        is_talker_responding = transition.after.is_talker_responding and not has_closed
        if token_id == model.response_token_id or not is_talker_responding:
            self.codec0_history.clear()
        elif emits_audio:
            self.codec0_history.append(int(codes[0]))
        else:
            pass
        self.codes = codes if is_talker_responding else None
        return TalkerFrame(
            codes=codes if emits_audio else self.silence_codes,
            is_active=emits_audio,
            has_closed=has_closed,
        )

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
