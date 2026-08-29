# SPDX-License-Identifier: Apache-2.0
"""Omni sample path must apply repetition_penalty once (SGLang), not p²."""

from __future__ import annotations

from types import SimpleNamespace

import torch
from sglang.srt.sampling.penaltylib.repetition_penalty import apply_scaling_penalties

from sglang_omni.models.fun_cosyvoice3.model_runner import FunCosyVoice3ModelRunner


def _sglang_like_sample(logits_output, _forward_batch, requests):
    """Stand-in for SGLang sample(): apply_scaling_penalties then argmax."""
    logits = logits_output.next_token_logits
    acc = torch.ones(logits.shape, dtype=torch.float32, device=logits.device)
    vocab = logits.shape[1]
    for row, sched_req in enumerate(requests):
        penalty = float(sched_req.data.req.sampling_params.repetition_penalty)
        if penalty == 1.0:
            continue
        for tok in sched_req.data.req.output_ids or []:
            tok = int(tok)
            if 0 <= tok < vocab:
                acc[row, tok] = penalty
    apply_scaling_penalties(logits, acc)
    return logits.argmax(dim=-1)


def test_cosyvoice3_sample_applies_repetition_penalty_once() -> None:
    penalty = 1.1
    vocab = 32
    output_ids = [3, 7, 7]
    logits = torch.ones(1, vocab, dtype=torch.float32)
    logits[0, 3] = 2.2
    logits[0, 7] = -1.5
    logits[0, 9] = 0.8
    orig = logits.clone()

    captured = {}

    def sample(logits_output, forward_batch):
        captured["before"] = logits_output.next_token_logits.detach().clone()
        out = _sglang_like_sample(logits_output, forward_batch, requests)
        captured["after"] = logits_output.next_token_logits.detach().clone()
        return out

    runner = object.__new__(FunCosyVoice3ModelRunner)
    runner.tp_worker = SimpleNamespace(model_runner=SimpleNamespace(sample=sample))
    req = SimpleNamespace(
        sampling_params=SimpleNamespace(repetition_penalty=penalty, sampling_seed=None),
        output_ids=output_ids,
    )
    data = SimpleNamespace(req=req, suppress_tokens=None, return_logprob=False)
    requests = [SimpleNamespace(data=data)]
    forward_batch = SimpleNamespace(
        sampling_info=SimpleNamespace(device="cpu", sampling_seed=None)
    )
    logits_output = SimpleNamespace(next_token_logits=logits)

    runner._sample_next_token_ids(
        logits_output, forward_batch, SimpleNamespace(), requests
    )

    before = captured["before"]
    after = captured["after"]
    # Omni must not have applied penalty before SGLang sample.
    assert torch.equal(before, orig)
    once_pos = orig[0, 3] / penalty
    twice_pos = orig[0, 3] / (penalty * penalty)
    once_neg = orig[0, 7] * penalty
    twice_neg = orig[0, 7] * (penalty * penalty)
    assert torch.isclose(after[0, 3], once_pos, rtol=0, atol=1e-6)
    assert not torch.isclose(after[0, 3], twice_pos, rtol=0, atol=1e-5)
    assert torch.isclose(after[0, 7], once_neg, rtol=0, atol=1e-6)
    assert not torch.isclose(after[0, 7], twice_neg, rtol=0, atol=1e-5)
    assert torch.isclose(after[0, 9], orig[0, 9], rtol=0, atol=1e-6)
