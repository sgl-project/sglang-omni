# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import torch

# These tests exercise request/session bookkeeping and pure tensor sampling.
# Importing SGLang on macOS otherwise initializes unrelated Triton decorators.
_original_torch_compile = torch.compile


def _identity_compile(model=None, *args, **kwargs):
    del args, kwargs
    if model is None:
        return lambda fn: fn
    return model


torch.compile = _identity_compile
try:
    from sglang_omni.models.minicpmo_4_5.model_runner import (
        MiniCPMO45ModelRunner,
        _apply_embedding_spans,
    )
    from sglang_omni.models.minicpmo_4_5.request_builders import (
        MiniCPMOUnitBuild,
        build_unit_request_data,
        prepare_session_prefix,
    )
    from sglang_omni.models.minicpmo_4_5.state import (
        EmbeddingSpan,
        MiniCPMOSessionState,
        MiniCPMOSpecialTokens,
        MiniCPMOUnitRequestData,
    )
finally:
    torch.compile = _original_torch_compile


class _Tokenizer:
    unk_token_id = 0
    bad_token_ids: tuple[int, ...] = ()

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        del add_special_tokens
        if text.startswith("<|im_start|>system"):
            return [101, 102]
        return [103]

    def decode(self, token_ids: list[int], **kwargs: Any) -> str:
        del kwargs
        return "".join(chr(65 + token_id % 26) for token_id in token_ids)


def _tokens() -> MiniCPMOSpecialTokens:
    return MiniCPMOSpecialTokens(
        unit_start=1,
        unit_end=2,
        image_start=3,
        image_end=4,
        slice_start=5,
        slice_end=6,
        listen=7,
        speak=8,
        tts_bos=9,
        tts_eos=10,
        tts_pad=11,
        chunk_eos=12,
        chunk_tts_eos=13,
        turn_eos=14,
        media_placeholder=0,
    )


def _state() -> MiniCPMOSessionState:
    return MiniCPMOSessionState(
        request_id="outer",
        session_id="session",
        generation=3,
        response_epoch=0,
        next_input_seq=2,
        system_prompt="system",
        inflight_input_seq=1,
        inflight_response_epoch=0,
    )


def _prepared_unit() -> SimpleNamespace:
    embedding = torch.arange(8, dtype=torch.float32).reshape(2, 4)
    return SimpleNamespace(
        input_ids=(1, 0, 0),
        embedding_spans=(EmbeddingSpan(1, 3, embedding, "audio"),),
        mode="AUDIO",
    )


def test_first_unit_setup_is_transactional_and_rollback_restores_prefix() -> None:
    state = _state()
    tokens = _tokens()
    tokenizer = _Tokenizer()
    reference = torch.ones(1, 4)
    prepare_session_prefix(
        state,
        tokenizer=tokenizer,
        special_tokens=tokens,
        reference_embedding=reference,
    )
    build = MiniCPMOUnitBuild(
        internal_request_id="session:g3:u1",
        state=state,
        prepared_unit=_prepared_unit(),
        forced_listen=False,
        close_speaking_turn=False,
        sampling={},
    )

    data = build_unit_request_data(
        build,
        tokenizer=tokenizer,
        vocab_size=256,
        special_tokens=tokens,
    )
    req = SimpleNamespace(origin_input_ids=list(data.local_input_ids))
    rollback = data.session_req_setup(req)

    assert callable(rollback)
    assert state.prefix_pending is False
    assert len(state.embedding_spans) == 2
    assert [(span.start, span.end) for span in state.embedding_spans] == [
        (2, 3),
        (5, 7),
    ]

    rollback()
    assert state.prefix_pending is True
    assert state.embedding_spans == []
    assert data.absolute_embedding_spans == []


def test_later_unit_appends_deferred_unit_end_without_replaying_prefix() -> None:
    state = _state()
    state.prefix_pending = False
    tokens = _tokens()
    data = build_unit_request_data(
        MiniCPMOUnitBuild(
            internal_request_id="session:g3:u1",
            state=state,
            prepared_unit=_prepared_unit(),
            forced_listen=False,
            close_speaking_turn=False,
            sampling={},
        ),
        tokenizer=_Tokenizer(),
        vocab_size=256,
        special_tokens=tokens,
    )

    assert data.local_input_ids == [tokens.unit_end, 1, 0, 0]


def test_unit_keeps_zero_epoch_if_interrupt_advances_session_during_build() -> None:
    state = _state()
    state.response_epoch = 1
    state.inflight_response_epoch = 0

    data = build_unit_request_data(
        MiniCPMOUnitBuild(
            internal_request_id="session:g3:u1",
            state=state,
            prepared_unit=_prepared_unit(),
            forced_listen=False,
            close_speaking_turn=False,
            sampling={},
        ),
        tokenizer=_Tokenizer(),
        vocab_size=256,
        special_tokens=_tokens(),
    )

    assert data.response_epoch == 0


def _runner() -> MiniCPMO45ModelRunner:
    runner = object.__new__(MiniCPMO45ModelRunner)
    runner.tokenizer = _Tokenizer()
    runner.special_tokens = _tokens()
    return runner


def test_sampler_forces_listen_and_prevents_listen_mid_speaking_turn() -> None:
    runner = _runner()
    state = _state()
    data = MiniCPMOUnitRequestData(session_state=state, forced_listen=True)
    data.generation_steps = 0
    logits = torch.zeros(1, 32)
    sampled = runner._sample_next_token_ids(
        SimpleNamespace(next_token_logits=logits),
        None,
        None,
        [SimpleNamespace(data=data)],
    )
    assert sampled.item() == runner.special_tokens.listen

    data.forced_listen = False
    data.extra_model_outputs["duplex_sampling"] = {"decode_mode": "greedy"}
    state.current_turn_ended = False
    logits[0, runner.special_tokens.listen] = 10
    sampled = runner._sample_next_token_ids(
        SimpleNamespace(next_token_logits=logits),
        None,
        None,
        [SimpleNamespace(data=data)],
    )
    assert sampled.item() == runner.special_tokens.tts_bos


def test_sampler_permanently_masks_tts_pad_like_landed_demo() -> None:
    runner = _runner()
    data = MiniCPMOUnitRequestData(session_state=_state())
    logits = torch.zeros(32)
    logits[runner.special_tokens.tts_pad] = 20
    logits[15] = 10

    sampled = runner._duplex_sample(
        logits,
        data,
        {"decode_mode": "greedy"},
        runner.special_tokens,
    )

    assert sampled == 15


def test_sampler_masks_generic_eos_without_masking_chunk_terminators() -> None:
    runner = _runner()
    tokens = runner.special_tokens
    runner.tokenizer.eos_token_id = 15
    runner.tokenizer.additional_stop_token_ids = [
        16,
        tokens.chunk_eos,
        tokens.chunk_tts_eos,
    ]
    data = MiniCPMOUnitRequestData(
        req=SimpleNamespace(eos_token_ids={17, tokens.listen}),
        session_state=_state(),
    )
    logits = torch.zeros(32)
    logits[15] = 30
    logits[16] = 29
    logits[17] = 28
    logits[tokens.chunk_tts_eos] = 20

    sampled = runner._duplex_sample(
        logits,
        data,
        {"decode_mode": "greedy"},
        tokens,
    )

    assert sampled == tokens.chunk_tts_eos


def test_hidden_states_align_to_previous_content_token_and_turn_end() -> None:
    runner = _runner()
    state = _state()
    data = MiniCPMOUnitRequestData(session_state=state)
    request = SimpleNamespace(request_id="unit", data=data)
    scheduler_output = SimpleNamespace(requests=[request])

    def step(generation_steps: int, sampled: int, hidden_value: float) -> None:
        data.generation_steps = generation_steps
        output = SimpleNamespace(
            data=sampled,
            extra={"hidden_states": torch.full((1, 4), hidden_value)},
        )
        runner.post_process_outputs(None, scheduler_output, {"unit": output})

    step(0, runner.special_tokens.speak, 0.0)
    step(1, 15, 1.0)
    step(2, runner.special_tokens.turn_eos, 2.0)
    step(3, runner.special_tokens.chunk_tts_eos, 3.0)

    assert data.generated_unit_ids == [15, runner.special_tokens.turn_eos]
    assert [pair[0] for pair in data.tts_pairs] == [15, runner.special_tokens.turn_eos]
    assert [pair[2] for pair in data.tts_pairs] == [False, True]
    torch.testing.assert_close(data.tts_pairs[0][1], torch.full((4,), 2.0))
    torch.testing.assert_close(data.tts_pairs[1][1], torch.full((4,), 3.0))


def test_embedding_span_replay_applies_only_current_extend_overlap() -> None:
    embeds = torch.zeros(4, 3)
    span = EmbeddingSpan(
        start=5,
        end=8,
        embedding=torch.tensor([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [3.0, 3.0, 3.0]]),
        modality="audio",
    )

    _apply_embedding_spans(
        embeds,
        absolute_start=6,
        absolute_end=10,
        spans=[span],
    )

    torch.testing.assert_close(embeds[0], torch.full((3,), 2.0))
    torch.testing.assert_close(embeds[1], torch.full((3,), 3.0))
    torch.testing.assert_close(embeds[2:], torch.zeros(2, 3))
