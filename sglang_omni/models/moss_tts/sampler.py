# SPDX-License-Identifier: Apache-2.0
"""Fixed-shape sampling and Delay-FSM transition for MOSS-TTS."""

from __future__ import annotations

from typing import Any, NamedTuple

import torch
from sglang.srt.layers.sampler import multinomial_with_seed
from torch import nn

from sglang_omni.models.moss_tts.payload_types import (
    AUDIO_REPETITION_PENALTY,
    AUDIO_SAMPLING,
    TEXT_SAMPLING,
    ChannelSampling,
)
from sglang_omni.models.moss_tts.sampling_kernels import (
    multinomial_with_seed_and_token_ids,
    seeded_gumbel_argmax,
)

_NEG_INF = float("-inf")
_INT64_MAX = torch.iinfo(torch.int64).max


class DelayGraphBatch(NamedTuple):
    """Request state consumed by one fixed-shape sampling graph replay."""

    delay_state: torch.Tensor
    seeds: torch.Tensor
    generation_steps: torch.Tensor


class DelaySamplingOutput(NamedTuple):
    """One fixed-width token row and the state consumed by the next step."""

    rows: torch.Tensor
    next_delay_state: torch.Tensor


def matches_graph_profile(data: Any) -> bool:
    """Return whether a request matches the profile baked into the graph."""

    text = ChannelSampling(
        temperature=float(getattr(data, "text_temperature", TEXT_SAMPLING.temperature)),
        top_p=float(getattr(data, "text_top_p", TEXT_SAMPLING.top_p)),
        top_k=int(getattr(data, "text_top_k", TEXT_SAMPLING.top_k)),
    )
    audio = ChannelSampling(
        temperature=float(
            getattr(data, "audio_temperature", AUDIO_SAMPLING.temperature)
        ),
        top_p=float(getattr(data, "audio_top_p", AUDIO_SAMPLING.top_p)),
        top_k=int(getattr(data, "audio_top_k", AUDIO_SAMPLING.top_k)),
    )
    penalty = float(
        getattr(
            data,
            "audio_repetition_penalty",
            AUDIO_REPETITION_PENALTY,
        )
    )
    return (text, audio, penalty) == (
        TEXT_SAMPLING,
        AUDIO_SAMPLING,
        AUDIO_REPETITION_PENALTY,
    )


def _sample_default_audio_tokens(
    logits: torch.Tensor,
    *,
    seeds: torch.Tensor,
    positions: torch.Tensor,
    output: torch.Tensor | None,
) -> torch.Tensor:
    """Checkpoint-default audio sampling with full-vocabulary tie semantics."""

    scores = logits / AUDIO_SAMPLING.temperature
    vocab = int(scores.shape[-1])
    if 0 < AUDIO_SAMPLING.top_k < vocab:
        topk_scores = torch.topk(scores, k=AUDIO_SAMPLING.top_k, dim=-1).values
        scores = scores.masked_fill(scores < topk_scores[:, -1:], _NEG_INF)

    sorted_scores, sorted_indices = torch.sort(scores, descending=True, dim=-1)
    probs_sorted = torch.softmax(sorted_scores, dim=-1)
    cumulative = torch.cumsum(probs_sorted, dim=-1)
    remove = cumulative > AUDIO_SAMPLING.top_p
    remove[..., 1:] = remove[..., :-1].clone()
    remove[..., 0] = False
    remove_scattered = torch.zeros_like(scores, dtype=torch.bool).scatter_(
        -1, sorted_indices, remove
    )
    scores = scores.masked_fill(remove_scattered, _NEG_INF)

    if scores.device.type == "cuda" and output is not None:
        return seeded_gumbel_argmax(scores, seeds, positions, output)
    return multinomial_with_seed(scores, seeds, positions).view(-1).long()


class MossTTSDelayAudioGraphSampler(nn.Module):
    """Pure-tensor fixed-shape sampler for all-audio Delay batches.

    The graph path is specialized to the checkpoint-default stochastic profile.
    Arbitrary request parameters continue to use the general eager sampler in
    :mod:`model_runner`.
    """

    def __init__(self, config: Any) -> None:
        super().__init__()
        self.n_vq = int(config.n_vq)
        self.num_channels = self.n_vq + 1
        self.pad_token_id = int(config.pad_token_id)
        self.gen_slot_token_id = int(config.audio_assistant_gen_slot_token_id)
        self.delay_slot_token_id = int(config.audio_assistant_delay_slot_token_id)
        self.audio_start_token_id = int(config.audio_start_token_id)
        self.audio_end_token_id = int(config.audio_end_token_id)
        self.im_end_token_id = int(config.im_end_token_id)
        self.audio_pad_code = int(config.audio_pad_code)

        audio_vocab = int(config.vocab_size_list[1])
        self.register_buffer(
            "text_control_token_ids",
            torch.tensor(
                [
                    self.gen_slot_token_id,
                    self.delay_slot_token_id,
                ],
                dtype=torch.long,
            ),
            persistent=False,
        )
        self.register_buffer(
            "audio_invalid_mask",
            torch.arange(audio_vocab) >= self.audio_pad_code,
            persistent=False,
        )
        self.register_buffer(
            "channel_indices",
            torch.arange(self.n_vq, dtype=torch.long).view(1, -1),
            persistent=False,
        )

    def forward(
        self,
        control_logits: torch.Tensor,
        audio_logits: torch.Tensor,
        batch: DelayGraphBatch,
        *,
        audio_sample_output: torch.Tensor | None = None,
    ) -> DelaySamplingOutput:
        """Static checkpoint-default stochastic topology."""
        batch_size = int(control_logits.shape[0])
        num_controls = len(self.text_control_token_ids)
        if tuple(control_logits.shape) != (batch_size, num_controls):
            raise ValueError(
                "MOSS-TTS Delay graph control logits must have shape "
                f"[B, {num_controls}], "
                f"got {tuple(control_logits.shape)}"
            )
        if tuple(audio_logits.shape[:2]) != (batch_size, self.n_vq):
            raise ValueError(
                "MOSS-TTS Delay graph audio logits must have shape [B, n_vq, V], "
                f"got {tuple(audio_logits.shape)}"
            )

        seeds = batch.seeds
        generation_steps = batch.generation_steps
        audio_lengths, delayed, is_audio = batch.delay_state.unbind(dim=1)
        is_audio = is_audio.bool()

        next_text = torch.full(
            (batch_size,),
            self.pad_token_id,
            dtype=torch.long,
            device=control_logits.device,
        )
        next_text = torch.where(
            delayed < self.n_vq,
            torch.full_like(next_text, self.delay_slot_token_id),
            next_text,
        )
        is_audio_eos = delayed == self.n_vq
        next_text = torch.where(
            is_audio_eos,
            torch.full_like(next_text, self.audio_end_token_id),
            next_text,
        )
        is_audio = is_audio & ~is_audio_eos
        sampling_text_mask = is_audio & (delayed > self.n_vq)

        masked_control_logits = control_logits.float() / TEXT_SAMPLING.temperature
        delay_logits = masked_control_logits[:, -1]
        masked_control_logits[:, -1] = torch.where(
            generation_steps == 0,
            torch.full_like(delay_logits, _NEG_INF),
            delay_logits,
        )
        sampled_control = multinomial_with_seed_and_token_ids(
            masked_control_logits,
            seeds,
            generation_steps * self.num_channels,
            self.text_control_token_ids,
        )
        sampled_text = self.text_control_token_ids[sampled_control]
        next_text = torch.where(sampling_text_mask, sampled_text, next_text)
        is_audio = (is_audio | (next_text == self.audio_start_token_id)) & (
            next_text != self.im_end_token_id
        )

        pre_audio = audio_lengths.unsqueeze(1) > self.channel_indices
        post_audio = self.channel_indices > (delayed.unsqueeze(1) - 1)
        post_audio = post_audio | (delayed == _INT64_MAX).unsqueeze(1)
        sampling_audio_mask = pre_audio & post_audio

        masked_audio_logits = audio_logits.float().masked_fill(
            self.audio_invalid_mask.view(1, 1, -1), _NEG_INF
        )
        flat_audio_logits = masked_audio_logits.reshape(batch_size * self.n_vq, -1)
        audio_positions = generation_steps.unsqueeze(1) * self.num_channels
        audio_positions = (
            (audio_positions + self.channel_indices + 1).reshape(-1).contiguous()
        )
        audio_seeds = seeds.unsqueeze(1).expand(batch_size, self.n_vq).reshape(-1)
        sampled_audio = _sample_default_audio_tokens(
            flat_audio_logits,
            seeds=audio_seeds,
            positions=audio_positions,
            output=audio_sample_output,
        ).view(batch_size, self.n_vq)

        next_audio = torch.where(
            sampling_audio_mask,
            sampled_audio,
            torch.full_like(sampled_audio, self.audio_pad_code),
        )

        increment = (
            (next_text == self.audio_start_token_id)
            | (next_text == self.gen_slot_token_id)
            | (next_text == self.delay_slot_token_id)
        )
        next_audio_lengths = audio_lengths + increment.long()
        next_audio_lengths = torch.where(
            next_text == self.audio_end_token_id,
            torch.zeros_like(next_audio_lengths),
            next_audio_lengths,
        )
        next_delayed = torch.where(
            (delayed == _INT64_MAX) & (next_text == self.delay_slot_token_id),
            torch.zeros_like(delayed),
            delayed,
        )
        next_delayed = torch.where(
            next_delayed != _INT64_MAX,
            next_delayed + 1,
            next_delayed,
        )
        next_delayed = torch.where(
            next_delayed > self.n_vq,
            torch.full_like(next_delayed, _INT64_MAX),
            next_delayed,
        )
        next_delay_state = torch.stack(
            (next_audio_lengths, next_delayed, is_audio.long()), dim=1
        )
        rows = torch.cat((next_text.unsqueeze(1), next_audio), dim=1)
        return DelaySamplingOutput(rows=rows, next_delay_state=next_delay_state)
