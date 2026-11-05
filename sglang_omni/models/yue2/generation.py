# SPDX-License-Identifier: Apache-2.0
"""Request-local autoregressive sampling for YuE2."""

from dataclasses import dataclass
from typing import Literal

import torch

from sglang_omni.models.yue2.cuda_graph import CapturedGraph, DecodeGraph
from sglang_omni.models.yue2.layers import TokenCache
from sglang_omni.models.yue2.model import Yue2Model
from sglang_omni.models.yue2.request import (
    ABC_END,
    CODEC_OFFSET,
    CODEC_SIZE,
    MUSIC_END,
    TEXT_END,
    CancellationCheck,
    SamplingConfig,
    check_cancelled,
)


@dataclass(kw_only=True)
class TokenResult:
    tokens: list[int]
    truncated: bool


def sampling_scores(
    logits: torch.Tensor,
    sampling: SamplingConfig,
    history: list[int],
    phase: Literal["abc", "semantic"],
    legacy_off: bool,
) -> torch.Tensor:
    frequencies = torch.zeros_like(
        logits, dtype=logits.dtype if legacy_off else torch.float32
    )
    if sampling.repetition_penalty != 1.0 and history:
        recent = torch.tensor(
            [history[-sampling.penalty_window :]], device=logits.device
        )
        frequencies.scatter_add_(
            -1, recent, torch.ones_like(recent, dtype=frequencies.dtype)
        )
    else:
        pass
    return filter_scores(
        logits,
        sampling,
        frequencies,
        torch.tensor(len(history) >= sampling.min_tokens, device=logits.device),
        phase,
        legacy_off,
    )


def filter_scores(
    logits: torch.Tensor,
    sampling: SamplingConfig,
    frequencies: torch.Tensor,
    allow_end: torch.Tensor,
    phase: Literal["abc", "semantic"],
    legacy_off: bool,
) -> torch.Tensor:
    scores = logits.clone() if legacy_off else logits.float().clone()
    end_token = ABC_END if phase == "abc" else MUSIC_END
    allowed = torch.full_like(scores, -torch.inf)
    if phase == "abc":
        allowed[..., :TEXT_END] = 0
    else:
        allowed[..., CODEC_OFFSET : CODEC_OFFSET + CODEC_SIZE] = 0
    allowed[..., end_token] = 0
    scores = scores + allowed
    scores[..., end_token] = torch.where(allow_end, scores[..., end_token], -torch.inf)
    if sampling.repetition_penalty != 1.0:
        penalty = sampling.repetition_penalty**frequencies
        scores = torch.where(scores < 0, scores * penalty, scores / penalty)
    else:
        pass
    if sampling.temperature == 0:
        return scores
    else:
        scores = scores / sampling.temperature
    threshold = scores.topk(min(sampling.top_k, scores.shape[-1])).values[..., -1, None]
    scores = scores.masked_fill(scores < threshold, -torch.inf)
    if sampling.top_p < 1:
        values, indices = scores.sort(descending=True)
        probabilities = values.softmax(-1)
        removed = probabilities.cumsum(-1) - probabilities > sampling.top_p
        removed[..., : 3 if legacy_off else 1] = False
        values = values.masked_fill(removed, -torch.inf)
        scores = scores.scatter(-1, indices, values)
    else:
        pass
    return scores


class SamplingGraph:
    def __init__(
        self,
        logits: torch.Tensor,
        sampling: SamplingConfig,
        phase: Literal["abc", "semantic"],
        legacy_off: bool,
    ) -> None:
        self.logits = torch.empty_like(logits)
        self.frequencies = torch.zeros_like(
            logits, dtype=logits.dtype if legacy_off else torch.float32
        )
        self.allow_end = torch.zeros((), dtype=torch.bool, device=logits.device)
        self.sampling = sampling
        self.phase = phase
        self.legacy_off = legacy_off
        self.recent = torch.zeros(
            (1, sampling.penalty_window), dtype=torch.long, device=logits.device
        )
        self.active = torch.zeros_like(self.recent, dtype=self.frequencies.dtype)
        self.graph = CapturedGraph(logits.device)
        self.logits.copy_(logits)
        stream = self.graph.stream
        stream.wait_stream(torch.cuda.current_stream(logits.device))
        with torch.cuda.stream(stream):
            for _ in range(3):
                self.forward()
        torch.cuda.current_stream(logits.device).wait_stream(stream)
        with torch.cuda.graph(self.graph.graph, stream=stream):
            self.scores = self.forward()

    def forward(self) -> torch.Tensor:
        self.frequencies.zero_()
        self.frequencies.scatter_add_(-1, self.recent, self.active)
        return filter_scores(
            self.logits,
            self.sampling,
            self.frequencies,
            self.allow_end,
            self.phase,
            self.legacy_off,
        )

    def step(
        self, logits: torch.Tensor, step: int, previous: torch.Tensor | None
    ) -> torch.Tensor:
        if previous is not None:
            slot = (step - 1) % self.sampling.penalty_window
            self.recent[:, slot : slot + 1].copy_(previous)
            self.active[:, slot : slot + 1].fill_(1)
        else:
            pass
        self.logits.copy_(logits)
        self.allow_end.fill_(step >= self.sampling.min_tokens)
        self.graph.replay()
        return self.scores


@torch.inference_mode()
def generate_tokens(
    model: Yue2Model,
    prefix: list[int],
    sampling: SamplingConfig,
    seed: int,
    phase: Literal["abc", "semantic"],
    cancelled: CancellationCheck,
    *,
    negative: list[int] | None = None,
    cfg_scale: float = 1.0,
    legacy_off: bool = False,
    decode_graphs: list[DecodeGraph] | None = None,
    sampling_cuda_graph: bool = False,
) -> TokenResult:
    if cfg_scale == 1:
        prefixes = [prefix]
    else:
        assert negative is not None
        prefixes = [prefix, negative]
    if any(
        len(tokens) + sampling.max_tokens > model.config.max_position_embeddings
        for tokens in prefixes
    ):
        raise ValueError(
            "YuE2 request: prefix and token budget exceed the model context"
        )
    else:
        pass
    check_cancelled(cancelled)
    device = model.model.embed_tokens.weight.device
    generator = torch.Generator(device=device).manual_seed(seed)
    if decode_graphs is None:
        positive_cache = TokenCache(capacity=len(prefix) + sampling.max_tokens)
        conditional = model(torch.tensor([prefix], device=device), positive_cache)
    else:
        while len(decode_graphs) < len(prefixes):
            decode_graphs.append(DecodeGraph(model))
        conditional = decode_graphs[0].prefill(prefix)
        positive_cache = decode_graphs[0].cache
    negative_cache = None
    unconditional = None
    if cfg_scale != 1:
        assert negative is not None
        if decode_graphs is None:
            negative_cache = TokenCache(capacity=len(negative) + sampling.max_tokens)
            unconditional = model(
                torch.tensor([negative], device=device), negative_cache
            )
        else:
            unconditional = decode_graphs[1].prefill(negative)
            negative_cache = decode_graphs[1].cache
    else:
        pass
    history: list[int] = []
    sampler = (
        SamplingGraph(conditional, sampling, phase, legacy_off)
        if sampling_cuda_graph
        else None
    )
    next_token = None
    end_token = ABC_END if phase == "abc" else MUSIC_END
    for step in range(sampling.max_tokens):
        check_cancelled(cancelled)
        if unconditional is None:
            logits = conditional
        else:
            logits = unconditional + cfg_scale * (conditional - unconditional)
        scores = (
            sampling_scores(logits, sampling, history, phase, legacy_off)
            if sampler is None
            else sampler.step(logits, step, next_token)
        )
        if sampling.temperature == 0:
            if not torch.isfinite(scores.max()):
                raise FloatingPointError("YuE2 produced no finite selectable logits")
            else:
                next_token = scores.argmax(-1, keepdim=True)
        else:
            next_token = torch.multinomial(scores.softmax(-1), 1, generator=generator)
        token = int(next_token.item())
        if token == end_token:
            return TokenResult(
                tokens=history,
                truncated=False,
            )
        else:
            history.append(token)
        if step + 1 < sampling.max_tokens:
            conditional = (
                model(next_token, positive_cache)
                if decode_graphs is None
                else decode_graphs[0].step(next_token)
            )
            if negative_cache is not None:
                unconditional = (
                    model(next_token, negative_cache)
                    if decode_graphs is None
                    else decode_graphs[1].step(next_token)
                )
            else:
                pass
        else:
            pass
    return TokenResult(tokens=history, truncated=True)
