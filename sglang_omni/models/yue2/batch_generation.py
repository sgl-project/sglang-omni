# SPDX-License-Identifier: Apache-2.0
"""Batched YuE2 token generation with independent request state."""

from collections import OrderedDict
from dataclasses import dataclass
from typing import Literal

import torch

from sglang_omni.models.yue2.cuda_graph import DecodeGraph
from sglang_omni.models.yue2.generation import SamplingGraph, TokenResult
from sglang_omni.models.yue2.model import Yue2Model
from sglang_omni.models.yue2.request import (
    ABC_END,
    MUSIC_END,
    CancellationCheck,
    SamplingConfig,
)

CAPACITY_BUCKET_TOKENS = 256


@dataclass(kw_only=True)
class TokenRequest:
    prefix: list[int]
    sampling: SamplingConfig
    seed: int
    cancelled: CancellationCheck
    legacy_off: bool


@torch.inference_mode()
def generate_batch_tokens(
    model: Yue2Model,
    requests: list[TokenRequest],
    phase: Literal["abc", "semantic"],
    graphs: OrderedDict[tuple[int, int], DecodeGraph],
    graph_cache_size: int,
) -> list[TokenResult | ValueError | InterruptedError]:
    results: dict[int, TokenResult | ValueError | InterruptedError] = {}
    pending: list[tuple[int, TokenRequest]] = []
    for index, request in enumerate(requests):
        if request.cancelled():
            results[index] = InterruptedError("YuE2 request cancelled")
        elif (
            len(request.prefix) + request.sampling.max_tokens
            > model.config.max_position_embeddings
        ):
            results[index] = ValueError(
                "YuE2 request: prefix and token budget exceed the model context"
            )
        else:
            pending.append((index, request))
    if not pending:
        return [results[index] for index in range(len(requests))]
    else:
        pass
    required_tokens = max(
        len(request.prefix) + request.sampling.max_tokens for _, request in pending
    )
    capacity = min(
        model.config.max_position_embeddings,
        -(-required_tokens // CAPACITY_BUCKET_TOKENS) * CAPACITY_BUCKET_TOKENS,
    )
    cache_key = (len(pending), capacity)
    if cache_key in graphs:
        runner = graphs[cache_key]
        graphs.move_to_end(cache_key)
    else:
        while len(graphs) >= graph_cache_size:
            graphs.popitem(last=False)
        runner = DecodeGraph(model, len(pending), capacity)
        graphs[cache_key] = runner
    logits = runner.prefill([request.prefix for _, request in pending])
    samplers = [
        SamplingGraph(
            logits[row : row + 1], request.sampling, phase, request.legacy_off
        )
        for row, (_, request) in enumerate(pending)
    ]
    generators = [
        torch.Generator(device=runner.device).manual_seed(request.seed)
        for _, request in pending
    ]
    histories: list[list[int]] = [[] for _ in pending]
    finished = [False for _ in pending]
    active = [True for _ in pending]
    inactive_token = runner.tokens.new_zeros((1, 1))
    previous: list[torch.Tensor | None] = [None for _ in pending]
    end_token = ABC_END if phase == "abc" else MUSIC_END
    for step in range(max(request.sampling.max_tokens for _, request in pending)):
        next_tokens: list[torch.Tensor] = []
        for row, (index, request) in enumerate(pending):
            if not finished[row] and request.cancelled():
                results[index] = InterruptedError("YuE2 request cancelled")
                finished[row] = True
            else:
                pass
            if finished[row]:
                token = inactive_token
            else:
                scores = samplers[row].step(logits[row : row + 1], step, previous[row])
                if request.sampling.temperature == 0:
                    if not torch.isfinite(scores.max()):
                        raise FloatingPointError(
                            "YuE2 produced no finite selectable logits"
                        )
                    else:
                        token = scores.argmax(-1, keepdim=True)
                else:
                    token = torch.multinomial(
                        scores.softmax(-1), 1, generator=generators[row]
                    )
            next_tokens.append(token)
        tokens = torch.cat(next_tokens)
        for row, token in enumerate(tokens.flatten().tolist()):
            index, request = pending[row]
            if finished[row]:
                continue
            elif token == end_token:
                results[index] = TokenResult(tokens=histories[row], truncated=False)
                finished[row] = True
            else:
                histories[row].append(token)
                if step + 1 == request.sampling.max_tokens:
                    results[index] = TokenResult(tokens=histories[row], truncated=True)
                    finished[row] = True
                else:
                    pass
        if all(finished):
            break
        else:
            previous = next_tokens
            next_active = [not done for done in finished]
            logits = runner.step(tokens, next_active if next_active != active else None)
            active = next_active
    return [results[index] for index in range(len(requests))]
