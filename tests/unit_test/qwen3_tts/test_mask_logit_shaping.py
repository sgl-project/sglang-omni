# SPDX-License-Identifier: Apache-2.0
"""Mask-based logit shaping must match the scalar reference bit for bit."""

from __future__ import annotations

import types

import torch

from sglang_omni.models.qwen3_tts.model_runner import Qwen3TTSModelRunner


def _runner() -> Qwen3TTSModelRunner:
    runner = object.__new__(Qwen3TTSModelRunner)
    runner._mask_last_sampled = None
    runner._mask_prep_rids = None
    runner._mask_rep_active = False
    runner._mask_sup_active = False
    return runner


def _request(rid: str, epoch: int, penalty: float, output_ids, suppress):
    sp = types.SimpleNamespace(repetition_penalty=penalty)
    req = types.SimpleNamespace(sampling_params=sp, output_ids=output_ids)
    data = types.SimpleNamespace(
        req=req, suppress_tokens=suppress, _qwen3_tts_prep_epoch=epoch
    )
    return types.SimpleNamespace(request_id=rid, data=data)


def _reference(logits, requests):
    out = logits.clone()
    vocab = out.shape[1]
    for row, sched_req in enumerate(requests):
        req = sched_req.data.req
        penalty = float(req.sampling_params.repetition_penalty)
        ids = req.output_ids or []
        unique = {int(t) for t in ids if 0 <= int(t) < vocab}
        if penalty != 1.0 and unique:
            idx = torch.tensor(sorted(unique), dtype=torch.long)
            scores = out[row, idx].to(torch.float32)
            scores = torch.where(scores > 0, scores / penalty, scores * penalty)
            out[row, idx] = scores.to(out.dtype)
        for tok in sched_req.data.suppress_tokens or []:
            tok = int(tok)
            if 0 <= tok < vocab:
                out[row, tok] = float("-inf")
    return out


def _apply(runner, logits, requests):
    logits_output = types.SimpleNamespace(next_token_logits=logits)
    runner._apply_repetition_penalty(logits_output, requests)
    runner._apply_codec_suppress_tokens(logits_output, requests)
    return logits_output.next_token_logits


def test_mask_shaping_steady_steps_match_reference():
    torch.manual_seed(3)
    vocab = 128
    runner = _runner()
    suppress = [5, 90, 200]
    reqs = [
        _request("a", 1, 1.3, [3, 7], suppress),
        _request("b", 2, 1.0, [4], suppress),
    ]

    for step in range(6):
        logits = torch.randn(2, vocab, dtype=torch.float32)
        got = _apply(runner, logits.clone(), reqs)
        expected = _reference(logits, reqs)
        assert torch.equal(got, expected), step
        # next step: each row samples one new token
        new_a, new_b = 10 + step, 40 + step
        reqs[0].data.req.output_ids = reqs[0].data.req.output_ids + [new_a]
        reqs[1].data.req.output_ids = reqs[1].data.req.output_ids + [new_b]
        runner._mask_last_sampled = torch.tensor([new_a, new_b])


def test_mask_shaping_rebuilds_on_rid_reuse():
    torch.manual_seed(4)
    vocab = 64
    runner = _runner()
    reqs = [_request("a", 1, 1.5, [3, 9, 11], [2])]
    logits = torch.randn(1, vocab)
    assert torch.equal(_apply(runner, logits.clone(), reqs), _reference(logits, reqs))

    # same rid, new lifetime (new epoch), different history and suppress list
    reqs2 = [_request("a", 7, 1.5, [30], [40])]
    logits2 = torch.randn(1, vocab)
    got = _apply(runner, logits2.clone(), reqs2)
    assert torch.equal(got, _reference(logits2, reqs2))
    # old tokens must not leak through the mask
    assert got[0, 3] == logits2[0, 3]


def test_mask_shaping_batch_shrink_rebuilds():
    torch.manual_seed(5)
    vocab = 64
    runner = _runner()
    reqs = [
        _request("a", 1, 1.2, [1, 2], [50]),
        _request("b", 2, 1.2, [3, 4], [50]),
    ]
    logits = torch.randn(2, vocab)
    assert torch.equal(_apply(runner, logits.clone(), reqs), _reference(logits, reqs))

    survivors = [reqs[1]]
    logits2 = torch.randn(1, vocab)
    got = _apply(runner, logits2.clone(), survivors)
    assert torch.equal(got, _reference(logits2, survivors))
    # row 0 now belongs to request b; request a's tokens must not be penalized
    assert got[0, 1] == logits2[0, 1]


def test_mask_shaping_clears_bits_after_history_shrink():
    """A retract shrinks output_ids; stale mask bits must not survive it."""
    torch.manual_seed(9)
    vocab = 64
    runner = _runner()
    reqs = [_request("a", 1, 1.5, [3, 9], [])]

    logits = torch.randn(1, vocab)
    _apply(runner, logits.clone(), reqs)

    # retract/restart: history shrinks, and the next step must not penalize 9
    reqs[0].data.req.output_ids = [3]
    runner._mask_last_sampled = torch.tensor([3])
    logits2 = torch.randn(1, vocab)
    got = _apply(runner, logits2.clone(), reqs)
    assert torch.equal(got, _reference(logits2, reqs))
    assert got[0, 9] == logits2[0, 9]


def test_mask_shaping_empty_history_after_retract():
    torch.manual_seed(11)
    vocab = 32
    runner = _runner()
    reqs = [_request("a", 1, 1.4, [5], [])]
    logits = torch.randn(1, vocab)
    _apply(runner, logits.clone(), reqs)

    reqs[0].data.req.output_ids = []
    runner._mask_last_sampled = torch.tensor([5])
    logits2 = torch.randn(1, vocab)
    got = _apply(runner, logits2.clone(), reqs)
    assert torch.equal(got, logits2), "no history means no penalty anywhere"
