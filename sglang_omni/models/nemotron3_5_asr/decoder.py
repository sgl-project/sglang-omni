# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: N801  # Keep the Nemotron 3.5 API spelling.
"""Request-owned RNN-T state and batched greedy streaming decode."""

from collections.abc import Sequence
from dataclasses import dataclass, field

import torch
from transformers.cache_utils import DynamicCache

from sglang_omni.models.nemotron3_5_asr.hf_compat import (
    Nemotron3_5AsrForRNNT,
    Nemotron3_5AsrRNNTDecoderCache,
    NemotronAsrStreamingEncoderCausalConvPaddingCache,
)


@dataclass(kw_only=True, slots=True)
class Nemotron3_5ASRDecodeState:
    tokens: list[int]
    durations: list[int]
    attention_cache: DynamicCache
    decoder_cache: Nemotron3_5AsrRNNTDecoderCache
    padding_cache: NemotronAsrStreamingEncoderCausalConvPaddingCache = field(
        default_factory=NemotronAsrStreamingEncoderCausalConvPaddingCache
    )
    symbols_at_frame: int = 0
    encoder_frames: int = 0
    decoder_steps: int = 0


def decode_streaming_batch(
    model: Nemotron3_5AsrForRNNT,
    states: Sequence[Nemotron3_5ASRDecodeState],
    encoded_frames: torch.Tensor,
    token_limits: Sequence[int | None],
) -> None:
    decoder = model.decoder
    blank_token_id = model.config.blank_token_id
    frame_count = encoded_frames.shape[1]
    frame_indices = [0] * len(states)
    for state in states:
        state.encoder_frames += frame_count

    active_indices = list(range(len(states)))
    while active_indices:
        prediction_states = [
            states[index]
            for index in active_indices
            if not states[index].decoder_cache.is_initialized
            or states[index].tokens[-1] != blank_token_id
        ]
        # note (Li Gang): Only the initial blank needs a new LSTM prediction.
        if prediction_states:
            input_ids = torch.tensor(
                [[state.tokens[-1]] for state in prediction_states],
                dtype=torch.long,
                device=encoded_frames.device,
            )
            embeddings = decoder.embedding(input_ids)
            caches = [state.decoder_cache for state in prediction_states]
            for row, cache in enumerate(caches):
                if not cache.is_initialized:
                    cache.lazy_initialization(embeddings[row : row + 1])
                else:
                    pass
            hidden = torch.cat([cache.hidden_state for cache in caches], dim=1)
            cell = torch.cat([cache.cell_state for cache in caches], dim=1)
            output, (hidden, cell) = decoder.lstm(embeddings, (hidden, cell))
            output = decoder.decoder_projector(output)
            for row, cache in enumerate(caches):
                cache.update(
                    output[row : row + 1],
                    hidden[:, row : row + 1],
                    cell[:, row : row + 1],
                )
        else:
            pass

        predictions = torch.cat(
            [states[index].decoder_cache.cache for index in active_indices], dim=0
        )
        current_frames = encoded_frames[
            active_indices, [frame_indices[index] for index in active_indices]
        ][:, None, :]
        logits = model.joint(
            decoder_hidden_states=predictions,
            encoder_hidden_states=current_frames,
        )
        token_ids = logits[:, -1, :].argmax(dim=-1).tolist()

        next_active: list[int] = []
        for index, token_id in zip(active_indices, token_ids, strict=True):
            state = states[index]
            state.tokens.append(token_id)
            state.decoder_steps += 1
            state.symbols_at_frame += 1
            # note (Li Gang): Match ParakeetRNNTGenerationMixin frame advances.
            advance = (
                token_id == blank_token_id
                or state.symbols_at_frame >= model.max_symbols_per_step
            )
            state.durations.append(int(advance))
            if advance:
                state.symbols_at_frame = 0
                frame_indices[index] += 1
            else:
                pass
            limit = token_limits[index]
            if frame_indices[index] < frame_count and (
                limit is None or state.decoder_steps < limit
            ):
                next_active.append(index)
            else:
                pass
        active_indices = next_active
