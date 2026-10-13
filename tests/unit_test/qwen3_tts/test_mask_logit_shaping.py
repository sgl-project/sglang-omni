# SPDX-License-Identifier: Apache-2.0
"""Qwen3-TTS logit shaping keeps one owner for each model policy."""

from __future__ import annotations

import types

import torch

from sglang_omni.models.qwen3_tts.model_runner import Qwen3TTSModelRunner


def make_runner(
    *,
    vocab_size: int,
    codec_eos_token_id: int,
    leading_silence_mask_frames: int = 0,
    silence_codec_ids: tuple[int, ...] = (),
) -> Qwen3TTSModelRunner:
    runner = object.__new__(Qwen3TTSModelRunner)
    runner.model = types.SimpleNamespace(
        config=types.SimpleNamespace(
            vocab_size=vocab_size,
            codec_eos_token_id=codec_eos_token_id,
        )
    )
    runner.leading_silence_mask_frames = leading_silence_mask_frames
    runner.silence_codec_ids = torch.tensor(silence_codec_ids, dtype=torch.long)
    return runner


def make_scheduled_request(
    *, mask_leading_silence: bool, generated_frames: int
) -> types.SimpleNamespace:
    return types.SimpleNamespace(
        data=types.SimpleNamespace(
            mask_leading_silence=mask_leading_silence,
            output_codes=[torch.zeros(16)] * generated_frames,
        )
    )


def test_qwen3_tts_suppresses_configured_codec_tail_with_basic_slices() -> None:
    configured_vocab = 3072
    codec_eos = 2150
    materialized_vocab = 6144
    runner = make_runner(
        vocab_size=configured_vocab,
        codec_eos_token_id=codec_eos,
    )
    logits = torch.randn(3, materialized_vocab)
    original = logits.clone()
    logits_output = types.SimpleNamespace(next_token_logits=logits)

    runner.apply_codec_suppress_tokens(logits_output, [object(), object()])

    suppress_start = configured_vocab - 1024
    assert torch.equal(logits[:2, :suppress_start], original[:2, :suppress_start])
    assert torch.isneginf(logits[:2, suppress_start:codec_eos]).all()
    assert torch.equal(logits[:2, codec_eos], original[:2, codec_eos])
    assert torch.isneginf(logits[:2, codec_eos + 1 : configured_vocab]).all()
    assert torch.equal(logits[:2, configured_vocab:], original[:2, configured_vocab:])
    assert torch.equal(logits[2], original[2])


def test_qwen3_tts_suppression_skips_empty_request_batch() -> None:
    runner = make_runner(vocab_size=3072, codec_eos_token_id=2150)
    logits = torch.randn(1, 6144)
    original = logits.clone()

    runner.apply_codec_suppress_tokens(
        types.SimpleNamespace(next_token_logits=logits), []
    )

    assert torch.equal(logits, original)


def test_qwen3_tts_masks_silence_ids_only_in_opening_frames_of_flagged_requests() -> (
    None
):
    configured_vocab = 3072
    silence_ids = (5, 7, 11)
    runner = make_runner(
        vocab_size=configured_vocab,
        codec_eos_token_id=2150,
        leading_silence_mask_frames=2,
        silence_codec_ids=silence_ids,
    )
    requests = [
        make_scheduled_request(mask_leading_silence=True, generated_frames=0),
        make_scheduled_request(mask_leading_silence=True, generated_frames=1),
        make_scheduled_request(mask_leading_silence=True, generated_frames=2),
        make_scheduled_request(mask_leading_silence=False, generated_frames=0),
    ]
    logits = torch.randn(len(requests), 6144)
    original = logits.clone()

    runner.apply_codec_suppress_tokens(
        types.SimpleNamespace(next_token_logits=logits), requests
    )

    silence = list(silence_ids)
    speech = [
        token for token in range(configured_vocab - 1024) if token not in silence_ids
    ]
    assert torch.isneginf(logits[:2, silence]).all()
    assert torch.equal(logits[:2, speech], original[:2, speech])
    assert torch.equal(logits[2:, silence], original[2:, silence])
