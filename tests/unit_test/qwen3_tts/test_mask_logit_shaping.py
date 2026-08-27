# SPDX-License-Identifier: Apache-2.0
"""Qwen3-TTS logit shaping keeps one owner for each model policy."""

from __future__ import annotations

import types

import torch

from sglang_omni.models.qwen3_tts.model_runner import Qwen3TTSModelRunner


def _runner(*, vocab_size: int, codec_eos_token_id: int) -> Qwen3TTSModelRunner:
    runner = object.__new__(Qwen3TTSModelRunner)
    runner.model = types.SimpleNamespace(
        config=types.SimpleNamespace(
            vocab_size=vocab_size,
            codec_eos_token_id=codec_eos_token_id,
        )
    )
    return runner


def _sampling_request(repetition_penalty: float) -> types.SimpleNamespace:
    sampling_params = types.SimpleNamespace(
        repetition_penalty=repetition_penalty,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        min_new_tokens=0,
    )
    return types.SimpleNamespace(
        sampling_params=sampling_params,
        custom_logit_processor=None,
    )


def test_qwen3_tts_leaves_repetition_penalty_to_sglang() -> None:
    runner = _runner(vocab_size=128, codec_eos_token_id=127)
    logits = torch.randn(2, 128)
    original = logits.clone()
    logits_output = types.SimpleNamespace(next_token_logits=logits)

    runner._apply_repetition_penalty(logits_output, [object(), object()])

    assert torch.equal(logits_output.next_token_logits, original)


def test_qwen3_tts_public_penalty_disables_async_lookahead() -> None:
    runner = _runner(vocab_size=128, codec_eos_token_id=127)

    assert runner.lookahead_eligible(
        types.SimpleNamespace(reqs=[_sampling_request(1.0)])
    )
    assert not runner.lookahead_eligible(
        types.SimpleNamespace(reqs=[_sampling_request(1.05)])
    )


def test_qwen3_tts_suppresses_configured_codec_tail_with_basic_slices() -> None:
    configured_vocab = 3072
    codec_eos = 2150
    materialized_vocab = 6144
    runner = _runner(
        vocab_size=configured_vocab,
        codec_eos_token_id=codec_eos,
    )
    logits = torch.randn(3, materialized_vocab)
    original = logits.clone()
    logits_output = types.SimpleNamespace(next_token_logits=logits)

    runner._apply_codec_suppress_tokens(logits_output, [object(), object()])

    suppress_start = configured_vocab - 1024
    assert torch.equal(logits[:2, :suppress_start], original[:2, :suppress_start])
    assert torch.isneginf(logits[:2, suppress_start:codec_eos]).all()
    assert torch.equal(logits[:2, codec_eos], original[:2, codec_eos])
    assert torch.isneginf(logits[:2, codec_eos + 1 : configured_vocab]).all()
    assert torch.equal(logits[:2, configured_vocab:], original[:2, configured_vocab:])
    assert torch.equal(logits[2], original[2])


def test_qwen3_tts_suppression_skips_empty_request_batch() -> None:
    runner = _runner(vocab_size=3072, codec_eos_token_id=2150)
    logits = torch.randn(1, 6144)
    original = logits.clone()

    runner._apply_codec_suppress_tokens(
        types.SimpleNamespace(next_token_logits=logits), []
    )

    assert torch.equal(logits, original)
